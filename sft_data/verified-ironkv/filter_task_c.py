#!/usr/bin/env python3
"""
Filter out Task C entries where broken_code == fixed_code.
Creates new filtered files with '_filtered' suffix.
"""

import json
from pathlib import Path


def filter_task_c_file(input_file: Path, output_file: Path, format_type: str):
    """Filter a single Task C file."""
    filtered = []
    removed = []

    with open(input_file) as f:
        for line_num, line in enumerate(f, 1):
            entry = json.loads(line)

            # Check based on format type
            should_remove = False

            if format_type == "raw":
                # Check if broken_code == fixed_code
                if entry.get("broken_code") == entry.get("fixed_code"):
                    should_remove = True
            elif format_type == "openai":
                # For OpenAI format, check if the entry would have had identical code
                # We need to check the input/output for similarity
                # For now, we'll use the ID to match with raw format
                pass  # Will use line number matching
            elif format_type == "sharegpt":
                # Same as OpenAI
                pass

            if should_remove:
                removed.append(line_num)
            else:
                filtered.append(entry)

    # Write filtered entries
    with open(output_file, 'w') as f:
        for entry in filtered:
            f.write(json.dumps(entry) + '\n')

    return len(filtered), len(removed)


def filter_task_c_by_ids(base_dir: Path):
    """Filter Task C files by identifying problematic IDs from raw format first."""

    print("=" * 80)
    print("FILTERING TASK C DATA")
    print("=" * 80)

    # Step 1: Identify problematic entry IDs from raw format
    print("\nStep 1: Identifying problematic entries from raw format...")

    problematic_ids = set()
    problematic_line_numbers = {}  # track line numbers for each split

    for split in ["all", "train", "val", "test"]:
        raw_file = base_dir / "raw" / f"task_c_{split}.jsonl"
        problematic_line_numbers[split] = []

        with open(raw_file) as f:
            for line_num, line in enumerate(f, 1):
                entry = json.loads(line)
                if entry.get("broken_code") == entry.get("fixed_code"):
                    problematic_ids.add(entry.get("id"))
                    problematic_line_numbers[split].append(line_num)

    print(f"  Found {len(problematic_ids)} problematic entries")

    # Step 2: Filter all formats
    print("\nStep 2: Filtering all formats...")

    for format_name, suffix in [("raw", ""), ("openai_format", "_openai"), ("sharegpt_format", "_sharegpt")]:
        format_dir = base_dir / format_name

        for split in ["all", "train", "val", "test"]:
            input_file = format_dir / f"task_c_{split}{suffix}.jsonl"
            output_file = format_dir / f"task_c_{split}{suffix}_filtered.jsonl"

            # Get line numbers to skip for this split
            skip_lines = set(problematic_line_numbers[split])

            filtered = []
            with open(input_file) as f:
                for line_num, line in enumerate(f, 1):
                    if line_num not in skip_lines:
                        filtered.append(line)

            # Write filtered file
            with open(output_file, 'w') as f:
                f.writelines(filtered)

            removed = len(skip_lines)
            print(f"  {format_name}/task_c_{split}{suffix}.jsonl: kept {len(filtered)}, removed {removed}")

    # Step 3: Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for split in ["all", "train", "val", "test"]:
        original = sum(1 for _ in open(base_dir / "raw" / f"task_c_{split}.jsonl"))
        filtered = sum(1 for _ in open(base_dir / "raw" / f"task_c_{split}_filtered.jsonl"))
        removed = original - filtered

        print(f"\ntask_c_{split}:")
        print(f"  Original: {original} entries")
        print(f"  Filtered: {filtered} entries")
        print(f"  Removed:  {removed} entries ({removed/original*100:.1f}%)")

    print("\n" + "=" * 80)
    print("FILTERED FILES CREATED")
    print("=" * 80)
    print("\nNew files created with '_filtered' suffix:")
    print("  - raw/task_c_*_filtered.jsonl")
    print("  - openai_format/task_c_*_openai_filtered.jsonl")
    print("  - sharegpt_format/task_c_*_sharegpt_filtered.jsonl")
    print("\nUse these filtered files for training to avoid problematic examples.")


def main():
    base_dir = Path(__file__).parent
    filter_task_c_by_ids(base_dir)


if __name__ == "__main__":
    main()
