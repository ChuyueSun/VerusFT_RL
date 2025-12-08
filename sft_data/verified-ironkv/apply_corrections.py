#!/usr/bin/env python3
"""
Apply learned corrections to the dataset based on user feedback.
"""

import json
import re
from pathlib import Path


def should_remove_proof_functions(entry):
    """Check if entry is a proof function (should be removed)."""
    return entry.get('metadata', {}).get('function_mode') == 'proof'


def has_invariant_in_target(entry):
    """Check if target has invariant (should be removed - invariants are not function specs)."""
    target = entry.get('target_text', '')
    return 'invariant' in target.lower()


def has_empty_or_external_body(entry):
    """Check if function has no body or is external (can't infer specs from nothing)."""
    input_text = entry.get('input_text', '')

    # Check for external_body annotation
    if 'external_body' in input_text:
        return True

    # Check if there's no opening brace (no body)
    if '{' not in input_text:
        return True

    return False


def remove_spec_comments_from_input(entry):
    """Remove requires/ensures comments from input_text."""
    input_text = entry.get('input_text', '')

    # Pattern: Remove lines that are comments about specs
    # e.g., "// requires ...", "// ensures ...", etc.
    lines = input_text.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        # Remove comment lines that mention spec keywords
        if stripped.startswith('//') and any(keyword in stripped for keyword in ['requires', 'ensures', 'invariant', 'decreases']):
            continue
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def apply_corrections(input_file: Path, output_file: Path):
    """Apply all learned corrections to a file."""

    entries = []
    with open(input_file) as f:
        for line in f:
            entries.append(json.loads(line))

    print(f"Loaded {len(entries)} entries from {input_file.name}")

    # Statistics
    removed_proofs = 0
    removed_invariants = 0
    removed_empty_body = 0
    cleaned_comments = 0

    cleaned_entries = []

    for entry in entries:
        # Pattern 1: Remove proof functions
        if should_remove_proof_functions(entry):
            removed_proofs += 1
            continue

        # Pattern 3: Remove entries with invariants in target
        if has_invariant_in_target(entry):
            removed_invariants += 1
            continue

        # Pattern 4: Remove entries with empty/external body
        if has_empty_or_external_body(entry):
            removed_empty_body += 1
            continue

        # Pattern 2: Remove spec comments from input
        original_input = entry.get('input_text', '')
        cleaned_input = remove_spec_comments_from_input(entry)

        if cleaned_input != original_input:
            entry['input_text'] = cleaned_input
            cleaned_comments += 1

        cleaned_entries.append(entry)

    # Write cleaned entries
    with open(output_file, 'w') as f:
        for entry in cleaned_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"\nResults:")
    print(f"  Original entries: {len(entries)}")
    print(f"  Removed proof functions: {removed_proofs}")
    print(f"  Removed with invariants: {removed_invariants}")
    print(f"  Removed empty/external body: {removed_empty_body}")
    print(f"  Cleaned spec comments: {cleaned_comments}")
    print(f"  Final entries: {len(cleaned_entries)}")
    print(f"\nSaved to: {output_file}")

    return len(entries), len(cleaned_entries)


def main():
    base_dir = Path('.')

    print("=" * 80)
    print("APPLYING LEARNED CORRECTIONS")
    print("=" * 80)
    print("\nPatterns learned:")
    print("  1. Remove all proof functions (not executable code)")
    print("  2. Remove spec comments from input (no hints)")
    print("  3. Remove entries with invariants in target (invariants are not function specs)")
    print("  4. Remove entries with empty/external bodies (can't infer from nothing)")
    print()

    # Apply to all Task A files
    for split in ['all', 'train', 'val', 'test']:
        print(f"\n{'─' * 80}")
        print(f"Processing task_a_{split}_fixed.jsonl")
        print(f"{'─' * 80}")

        input_file = base_dir / 'raw' / f'task_a_{split}_fixed.jsonl'
        output_file = base_dir / 'raw' / f'task_a_{split}_fixed_v3.jsonl'

        if not input_file.exists():
            print(f"  Skipping (file not found)")
            continue

        apply_corrections(input_file, output_file)

    print("\n" + "=" * 80)
    print("CORRECTIONS COMPLETE")
    print("=" * 80)
    print("\nNew files created with '_v3' suffix:")
    print("  - raw/task_a_*_fixed_v3.jsonl")
    print("\nThese files have:")
    print("  ✓ Proof functions removed")
    print("  ✓ Entries with invariants removed")
    print("  ✓ Empty/external body entries removed")
    print("  ✓ Spec comments cleaned from inputs")
    print("\nReview the _v3 files, and if they look good, use them for training!")


if __name__ == "__main__":
    main()
