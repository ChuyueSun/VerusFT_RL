#!/usr/bin/env python3
"""
Fix truncated specifications in the IronKV dataset.

The issue: target_text is truncated when extracting from full_function.
Example: "ensures (," instead of the complete ensures clause.

This script re-extracts specifications from the full_function field.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple


def extract_specs_from_function(full_function: str) -> str:
    """
    Extract complete specifications (requires/ensures/invariant/decreases) from a full function.
    
    Args:
        full_function: The complete function code
        
    Returns:
        Extracted specifications as a string
    """
    lines = full_function.split('\n')
    spec_lines = []
    in_spec = False
    brace_depth = 0
    paren_depth = 0
    
    for line in lines:
        stripped = line.strip()
        
        # Check if we're starting a spec clause
        if re.match(r'^\s*(requires|ensures|invariant|decreases)\b', line):
            in_spec = True
        
        if in_spec:
            spec_lines.append(line)
            
            # Track brace and parenthesis depth to handle multi-line expressions
            for char in line:
                if char == '{':
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1
                elif char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
            
            # Check if we've reached the end of this spec clause
            # Specs end with a comma at depth 0, or when we hit the function body '{'
            if brace_depth == 0 and paren_depth == 0:
                if stripped.endswith(','):
                    in_spec = False
                elif stripped == '{' or ('{' in stripped and not stripped.startswith('//')):
                    # We've hit the function body, stop
                    # Remove the '{' line if it was included
                    if '{' in spec_lines[-1]:
                        spec_lines = spec_lines[:-1]
                    break
    
    # Join and clean up the extracted specs
    result = '\n'.join(spec_lines)
    
    # Remove leading/trailing whitespace but preserve internal structure
    result = result.strip()
    
    # Collapse multiple blank lines
    result = re.sub(r'\n\s*\n', '\n', result)
    
    return result


def fix_example(example: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Fix a single example if its target_text is truncated.
    
    Returns:
        (was_fixed, fixed_example)
    """
    target = example.get('target_text', '')
    
    # Check if target looks truncated
    # Patterns that indicate truncation:
    # - ends with "(,"
    # - ends with "("
    # - ends with incomplete clause like "requires (," or "ensures (,"
    is_truncated = (
        target.endswith('(,') or 
        target.endswith('(') or
        re.search(r'(requires|ensures|invariant|decreases)\s+\($', target) or
        re.search(r'(requires|ensures|invariant|decreases)\s+\(,$', target)
    )
    
    if not is_truncated:
        return False, example
    
    # Try to extract complete specs from full_function
    full_function = example.get('full_function', '')
    if not full_function:
        print(f"  WARNING: No full_function field for {example.get('id', 'unknown')}")
        return False, example
    
    # Extract complete specs
    complete_specs = extract_specs_from_function(full_function)
    
    if not complete_specs or complete_specs == target:
        print(f"  WARNING: Could not extract better specs for {example.get('id', 'unknown')}")
        return False, example
    
    # Create fixed example
    fixed = example.copy()
    fixed['target_text'] = complete_specs
    
    return True, fixed


def fix_file(input_path: Path, output_path: Path = None, dry_run: bool = True) -> Dict[str, Any]:
    """
    Fix truncated specifications in a JSONL file.
    
    Returns:
        Statistics about fixes applied
    """
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_fixed.jsonl"
    
    stats = {
        'total': 0,
        'fixed': 0,
        'fixed_ids': []
    }
    
    fixed_examples = []
    
    with open(input_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            stats['total'] += 1
            
            try:
                example = json.loads(line.strip())
                was_fixed, fixed = fix_example(example)
                
                if was_fixed:
                    stats['fixed'] += 1
                    stats['fixed_ids'].append((line_num, example.get('id', 'unknown')))
                    fixed_examples.append(fixed)
                    
                    if dry_run:
                        print(f"\n  Line {line_num}: {example.get('id', 'unknown')}")
                        print(f"    BEFORE: {example['target_text'][:100]}...")
                        print(f"    AFTER:  {fixed['target_text'][:100]}...")
                else:
                    fixed_examples.append(example)
                    
            except json.JSONDecodeError as e:
                print(f"  ERROR Line {line_num}: JSON decode error: {e}")
                fixed_examples.append({})  # Placeholder
    
    # Write fixed file if not dry run
    if not dry_run and stats['fixed'] > 0:
        with open(output_path, 'w') as f:
            for example in fixed_examples:
                f.write(json.dumps(example) + '\n')
        print(f"    ✓ Wrote fixed file: {output_path}")
    
    return stats


def regenerate_openai_format(raw_file: Path, openai_file: Path):
    """
    Regenerate OpenAI format from fixed raw JSONL file.
    """
    print(f"\n  Regenerating OpenAI format: {openai_file.name}")
    
    with open(raw_file, 'r') as f_in, open(openai_file, 'w') as f_out:
        for line in f_in:
            example = json.loads(line.strip())
            
            # Convert to OpenAI format
            openai_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert in Verus, a verification-oriented extension of Rust. Given a function, generate appropriate specifications including requires (preconditions), ensures (postconditions), invariants, and decreases clauses."
                    },
                    {
                        "role": "user",
                        "content": example['input_text']
                    },
                    {
                        "role": "assistant",
                        "content": example['target_text']
                    }
                ]
            }
            
            f_out.write(json.dumps(openai_example) + '\n')
    
    print(f"    ✓ Regenerated {openai_file}")


def main(auto_fix: bool = False):
    """Fix truncated specs in all dataset files."""
    base_dir = Path('/home/chuyue/VerusSFT/sft_data/verified-ironkv')
    raw_dir = base_dir / 'raw'
    openai_dir = base_dir / 'openai_format'
    
    print("=" * 80)
    print("FIXING TRUNCATED SPECIFICATIONS IN IRONKV DATASET")
    print("=" * 80)
    
    # Process all task files
    tasks = ['a', 'b', 'c']
    splits = ['all', 'train', 'val', 'test']
    
    print("\n" + "=" * 80)
    print("STEP 1: Scanning for truncated specifications (DRY RUN)")
    print("=" * 80)
    
    all_stats = []
    for task in tasks:
        for split in splits:
            raw_file = raw_dir / f'task_{task}_{split}.jsonl'
            if not raw_file.exists():
                continue
            
            print(f"\n📁 {raw_file.name}")
            stats = fix_file(raw_file, dry_run=True)
            print(f"   Total: {stats['total']}, Fixed: {stats['fixed']}")
            all_stats.append((raw_file, stats))
    
    total_fixed = sum(s['fixed'] for _, s in all_stats)
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: Found {total_fixed} truncated specifications")
    print("=" * 80)
    
    if total_fixed > 0:
        if not auto_fix:
            response = input("\nProceed with fixing? (yes/no): ")
            should_fix = response.lower() == 'yes'
        else:
            should_fix = True
            print("\nAuto-fix enabled, proceeding with fixes...")
        
        if should_fix:
            print("\n" + "=" * 80)
            print("STEP 2: Fixing raw JSONL files")
            print("=" * 80)
            
            for raw_file, _ in all_stats:
                stats = fix_file(raw_file, dry_run=False)
                if stats['fixed'] > 0:
                    print(f"✓ Fixed {stats['fixed']} examples in {raw_file.name}")
            
            print("\n" + "=" * 80)
            print("STEP 3: Regenerating OpenAI format files")
            print("=" * 80)
            
            for task in tasks:
                for split in splits:
                    raw_file = raw_dir / f'task_{task}_{split}_fixed.jsonl'
                    openai_file = openai_dir / f'task_{task}_{split}_openai_fixed.jsonl'
                    
                    if raw_file.exists():
                        regenerate_openai_format(raw_file, openai_file)
            
            print("\n" + "=" * 80)
            print("✓ ALL DONE!")
            print("=" * 80)
            print("\nFixed files created with '_fixed' suffix.")
            print("Please verify the fixes, then replace the original files.")


if __name__ == '__main__':
    import sys
    auto_fix = '--auto-fix' in sys.argv or '--yes' in sys.argv
    main(auto_fix=auto_fix)
