#!/usr/bin/env python3
"""
Fix Task A: Include function body in the input.

Current: Input has only signature
Desired: Input has signature + body (without specs)
"""

import json
import re
from pathlib import Path


def remove_specs_from_function(full_function):
    """
    Remove specification clauses from a function, keeping only signature and body.
    """
    lines = full_function.split('\n')

    # Collect signature lines and body lines separately
    sig_lines = []
    body_lines = []
    state = 'signature'  # signature -> specs -> body
    body_brace_count = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        if state == 'signature':
            # Check if we hit spec keywords
            if any(stripped.startswith(kw) for kw in ['requires', 'ensures', 'invariant', 'decreases']):
                state = 'specs'
                continue
            # Check if we hit body directly (no specs) - must be ONLY a brace
            elif stripped == '{':
                state = 'body'
                body_lines.append(line)
                body_brace_count = 1
            else:
                # Still in signature
                sig_lines.append(line)

        elif state == 'specs':
            # Check if we've entered the body - must be ONLY a brace on the line
            if stripped == '{':
                # This is the opening brace of the function body
                state = 'body'
                body_lines.append(line)
                body_brace_count = 1
            # else: skip spec lines

        elif state == 'body':
            body_lines.append(line)
            body_brace_count += line.count('{') - line.count('}')

            # Stop when we've closed all braces
            if body_brace_count == 0:
                break

    return '\n'.join(sig_lines + body_lines)


def extract_function_signature(full_function):
    """Extract just the function signature."""
    lines = full_function.split('\n')
    sig_lines = []

    for line in lines:
        stripped = line.strip()
        # Signature ends before specs or body
        if any(stripped.startswith(kw) for kw in ['requires', 'ensures', 'invariant', 'decreases', '{']):
            break
        if stripped:  # Skip empty lines
            sig_lines.append(line)

    return '\n'.join(sig_lines)


def create_new_input_text(entry):
    """Create new input text with signature + body (no specs)."""

    full_function = entry.get('full_function', '')
    if not full_function:
        return None

    # Remove specs but keep signature and body
    function_without_specs = remove_specs_from_function(full_function)

    # Create new prompt
    prompt = f"""Given the following Verus function implementation, infer the appropriate specifications (requires/ensures clauses).

Function:
```rust
{function_without_specs.strip()}
```

Write the specifications:"""

    return prompt


def fix_task_a_file(input_file: Path, output_file: Path):
    """Fix a single Task A file."""

    entries = []
    with open(input_file) as f:
        for line in f:
            entries.append(json.loads(line))

    fixed_entries = []
    skipped = 0

    for entry in entries:
        # Create new input text with body
        new_input = create_new_input_text(entry)

        if new_input:
            entry['input_text'] = new_input
            fixed_entries.append(entry)
        else:
            skipped += 1

    # Write fixed entries
    with open(output_file, 'w') as f:
        for entry in fixed_entries:
            f.write(json.dumps(entry) + '\n')

    return len(fixed_entries), skipped


def fix_openai_format(raw_file: Path, output_file: Path):
    """Fix OpenAI format based on fixed raw file."""

    entries = []
    with open(raw_file) as f:
        for line in f:
            entries.append(json.loads(line))

    openai_entries = []

    for entry in entries:
        openai_entry = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in Verus, a verification-oriented extension of Rust. Given a function implementation, infer the appropriate specifications including requires (preconditions), ensures (postconditions), invariants, and decreases clauses."
                },
                {
                    "role": "user",
                    "content": entry['input_text']
                },
                {
                    "role": "assistant",
                    "content": entry['target_text']
                }
            ]
        }
        openai_entries.append(openai_entry)

    with open(output_file, 'w') as f:
        for entry in openai_entries:
            f.write(json.dumps(entry) + '\n')

    return len(openai_entries)


def fix_sharegpt_format(raw_file: Path, output_file: Path):
    """Fix ShareGPT format based on fixed raw file."""

    entries = []
    with open(raw_file) as f:
        for line in f:
            entries.append(json.loads(line))

    sharegpt_entries = []

    for entry in entries:
        sharegpt_entry = {
            "conversations": [
                {
                    "from": "system",
                    "value": "You are an expert in Verus, a verification-oriented extension of Rust. Given a function implementation, infer the appropriate specifications including requires (preconditions), ensures (postconditions), invariants, and decreases clauses."
                },
                {
                    "from": "human",
                    "value": entry['input_text']
                },
                {
                    "from": "gpt",
                    "value": entry['target_text']
                }
            ]
        }
        sharegpt_entries.append(sharegpt_entry)

    with open(output_file, 'w') as f:
        for entry in sharegpt_entries:
            f.write(json.dumps(entry) + '\n')

    return len(sharegpt_entries)


def main():
    base_dir = Path(__file__).parent

    print("=" * 80)
    print("FIXING TASK A DATASET")
    print("Including function body in input (signature + body → specs)")
    print("=" * 80)

    # Fix raw format
    print("\n📝 Fixing raw format...")
    for split in ['all', 'train', 'val', 'test']:
        input_file = base_dir / 'raw' / f'task_a_{split}.jsonl'
        output_file = base_dir / 'raw' / f'task_a_{split}_fixed.jsonl'

        fixed, skipped = fix_task_a_file(input_file, output_file)
        print(f"  {split}: {fixed} entries fixed, {skipped} skipped")

    # Fix OpenAI format
    print("\n🤖 Fixing OpenAI format...")
    for split in ['all', 'train', 'val', 'test']:
        raw_file = base_dir / 'raw' / f'task_a_{split}_fixed.jsonl'
        output_file = base_dir / 'openai_format' / f'task_a_{split}_openai_fixed.jsonl'

        count = fix_openai_format(raw_file, output_file)
        print(f"  {split}: {count} entries")

    # Fix ShareGPT format
    print("\n💬 Fixing ShareGPT format...")
    for split in ['all', 'train', 'val', 'test']:
        raw_file = base_dir / 'raw' / f'task_a_{split}_fixed.jsonl'
        output_file = base_dir / 'sharegpt_format' / f'task_a_{split}_sharegpt_fixed.jsonl'

        count = fix_sharegpt_format(raw_file, output_file)
        print(f"  {split}: {count} entries")

    print("\n" + "=" * 80)
    print("FIXED FILES CREATED")
    print("=" * 80)
    print("\nNew files with '_fixed' suffix:")
    print("  - raw/task_a_*_fixed.jsonl")
    print("  - openai_format/task_a_*_openai_fixed.jsonl")
    print("  - sharegpt_format/task_a_*_sharegpt_fixed.jsonl")
    print("\nThese files now have function body in the input!")

    # Show example
    print("\n" + "=" * 80)
    print("EXAMPLE OF FIXED DATA")
    print("=" * 80)

    with open(base_dir / 'raw' / 'task_a_all_fixed.jsonl') as f:
        entry = json.loads(f.readline())
        print("\nInput preview:")
        print(entry['input_text'][:400] + "...")
        print("\nTarget:")
        print(entry['target_text'][:200])


if __name__ == "__main__":
    main()
