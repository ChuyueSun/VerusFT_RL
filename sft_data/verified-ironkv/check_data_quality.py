#!/usr/bin/env python3
"""
Comprehensive data quality checker for Verus SFT training data.
Validates JSONL files across raw, openai_format, and sharegpt_format directories.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import hashlib


class DataQualityChecker:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.issues = defaultdict(list)
        self.stats = defaultdict(dict)

    def check_json_validity(self, file_path: Path) -> List[Dict]:
        """Check if JSONL file is valid and load all entries."""
        entries = []
        line_num = 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        self.issues[file_path].append(f"Line {line_num}: Empty line")
                        continue

                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        self.issues[file_path].append(f"Line {line_num}: Invalid JSON - {e}")

        except Exception as e:
            self.issues[file_path].append(f"Failed to read file: {e}")

        return entries

    def check_openai_format(self, entry: Dict, file_path: Path, line_num: int):
        """Validate OpenAI format structure."""
        if "messages" not in entry:
            self.issues[file_path].append(f"Line {line_num}: Missing 'messages' field")
            return

        messages = entry["messages"]
        if not isinstance(messages, list):
            self.issues[file_path].append(f"Line {line_num}: 'messages' is not a list")
            return

        if len(messages) < 2:
            self.issues[file_path].append(f"Line {line_num}: Need at least system + user message")
            return

        # Check message structure
        for i, msg in enumerate(messages):
            if "role" not in msg:
                self.issues[file_path].append(f"Line {line_num}, Message {i}: Missing 'role'")
            if "content" not in msg:
                self.issues[file_path].append(f"Line {line_num}, Message {i}: Missing 'content'")
            elif not msg["content"] or not msg["content"].strip():
                self.issues[file_path].append(f"Line {line_num}, Message {i}: Empty content")

        # Check role sequence
        roles = [msg.get("role") for msg in messages]
        if roles and roles[0] != "system":
            self.issues[file_path].append(f"Line {line_num}: First message should be 'system'")

        if len(roles) >= 2 and roles[1] != "user":
            self.issues[file_path].append(f"Line {line_num}: Second message should be 'user'")

        if len(roles) >= 3 and roles[-1] != "assistant":
            self.issues[file_path].append(f"Line {line_num}: Last message should be 'assistant'")

    def check_sharegpt_format(self, entry: Dict, file_path: Path, line_num: int):
        """Validate ShareGPT format structure."""
        if "conversations" not in entry:
            self.issues[file_path].append(f"Line {line_num}: Missing 'conversations' field")
            return

        conversations = entry["conversations"]
        if not isinstance(conversations, list):
            self.issues[file_path].append(f"Line {line_num}: 'conversations' is not a list")
            return

        for i, conv in enumerate(conversations):
            if "from" not in conv:
                self.issues[file_path].append(f"Line {line_num}, Conv {i}: Missing 'from'")
            if "value" not in conv:
                self.issues[file_path].append(f"Line {line_num}, Conv {i}: Missing 'value'")
            elif not conv["value"] or not conv["value"].strip():
                self.issues[file_path].append(f"Line {line_num}, Conv {i}: Empty value")

    def check_raw_format(self, entry: Dict, file_path: Path, line_num: int):
        """Validate raw format structure."""
        required_fields = ["id", "task", "input_text", "target_text"]

        for field in required_fields:
            if field not in entry:
                self.issues[file_path].append(f"Line {line_num}: Missing '{field}' field")
            elif not entry.get(field) or not str(entry[field]).strip():
                self.issues[file_path].append(f"Line {line_num}: Empty '{field}' field")

        # Check task type
        if "task" in entry and entry["task"] not in ["spec_from_code", "code_from_spec", "error_repair"]:
            self.issues[file_path].append(f"Line {line_num}: Unknown task type '{entry['task']}'")

    def check_content_quality(self, entry: Dict, file_path: Path, line_num: int, format_type: str):
        """Check content quality issues."""
        if format_type == "openai":
            messages = entry.get("messages", [])
            for i, msg in enumerate(messages):
                content = msg.get("content", "")
                self._check_text_quality(content, file_path, line_num, f"Message {i}")

        elif format_type == "sharegpt":
            conversations = entry.get("conversations", [])
            for i, conv in enumerate(conversations):
                value = conv.get("value", "")
                self._check_text_quality(value, file_path, line_num, f"Conv {i}")

        elif format_type == "raw":
            for field in ["input_text", "target_text"]:
                text = entry.get(field, "")
                self._check_text_quality(text, file_path, line_num, field)

    def _check_text_quality(self, text: str, file_path: Path, line_num: int, field: str):
        """Check individual text quality."""
        if not text:
            return

        # Check for truncation indicators
        truncation_markers = ["...", "/* truncated */", "// truncated", "[truncated]"]
        for marker in truncation_markers:
            if marker.lower() in text.lower():
                self.issues[file_path].append(
                    f"Line {line_num}, {field}: Contains truncation marker '{marker}'"
                )

        # Check for extremely long lines (might indicate formatting issues)
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if len(line) > 500:
                self.issues[file_path].append(
                    f"Line {line_num}, {field}, Text line {i+1}: Very long line ({len(line)} chars)"
                )

        # Check for placeholder text
        placeholders = ["TODO", "FIXME", "XXX", "placeholder"]
        for placeholder in placeholders:
            if placeholder in text:
                self.issues[file_path].append(
                    f"Line {line_num}, {field}: Contains placeholder '{placeholder}'"
                )

    def find_duplicates(self, entries: List[Dict], file_path: Path, format_type: str):
        """Find duplicate entries based on content hash."""
        content_hashes = defaultdict(list)

        for i, entry in enumerate(entries, 1):
            # Extract content for hashing based on format
            if format_type == "openai":
                content = json.dumps(entry.get("messages", []), sort_keys=True)
            elif format_type == "sharegpt":
                content = json.dumps(entry.get("conversations", []), sort_keys=True)
            else:  # raw
                content = f"{entry.get('input_text', '')}|{entry.get('target_text', '')}"

            content_hash = hashlib.md5(content.encode()).hexdigest()
            content_hashes[content_hash].append(i)

        # Report duplicates
        duplicates = {h: lines for h, lines in content_hashes.items() if len(lines) > 1}
        if duplicates:
            for hash_val, line_nums in duplicates.items():
                self.issues[file_path].append(
                    f"Duplicate content found at lines: {', '.join(map(str, line_nums))}"
                )

        return len(duplicates)

    def gather_statistics(self, entries: List[Dict], file_path: Path, format_type: str):
        """Gather statistics about the data."""
        stats = {
            "total_entries": len(entries),
            "duplicates": 0,
        }

        if format_type == "raw":
            task_counts = Counter(entry.get("task") for entry in entries)
            stats["task_distribution"] = dict(task_counts)

            # Count entries with metadata
            with_metadata = sum(1 for e in entries if "metadata" in e)
            stats["with_metadata"] = with_metadata

        elif format_type == "openai":
            msg_counts = [len(entry.get("messages", [])) for entry in entries]
            if msg_counts:
                stats["avg_messages"] = sum(msg_counts) / len(msg_counts)
                stats["min_messages"] = min(msg_counts)
                stats["max_messages"] = max(msg_counts)

        elif format_type == "sharegpt":
            conv_counts = [len(entry.get("conversations", [])) for entry in entries]
            if conv_counts:
                stats["avg_conversations"] = sum(conv_counts) / len(conv_counts)
                stats["min_conversations"] = min(conv_counts)
                stats["max_conversations"] = max(conv_counts)

        self.stats[file_path] = stats

    def check_train_val_test_split(self, task: str, format_dir: Path):
        """Verify train/val/test splits are consistent."""
        suffix = "_openai.jsonl" if "openai" in str(format_dir) else \
                 "_sharegpt.jsonl" if "sharegpt" in str(format_dir) else ".jsonl"

        all_file = format_dir / f"{task}_all{suffix}"
        train_file = format_dir / f"{task}_train{suffix}"
        val_file = format_dir / f"{task}_val{suffix}"
        test_file = format_dir / f"{task}_test{suffix}"

        if not all([f.exists() for f in [all_file, train_file, val_file, test_file]]):
            return

        # Count entries
        all_count = sum(1 for _ in open(all_file))
        train_count = sum(1 for _ in open(train_file))
        val_count = sum(1 for _ in open(val_file))
        test_count = sum(1 for _ in open(test_file))

        expected_split = train_count + val_count + test_count

        if all_count != expected_split:
            self.issues[format_dir].append(
                f"{task}: Split mismatch - all={all_count}, train+val+test={expected_split}"
            )

        # Report split ratios
        if all_count > 0:
            print(f"\n{task} split ({format_dir.name}):")
            print(f"  Train: {train_count} ({train_count/all_count*100:.1f}%)")
            print(f"  Val:   {val_count} ({val_count/all_count*100:.1f}%)")
            print(f"  Test:  {test_count} ({test_count/all_count*100:.1f}%)")
            print(f"  Total: {all_count}")

    def check_directory(self, dir_path: Path, format_type: str):
        """Check all JSONL files in a directory."""
        jsonl_files = sorted(dir_path.glob("*.jsonl"))

        print(f"\nChecking {dir_path.name}/ ({format_type} format)...")
        print(f"Found {len(jsonl_files)} JSONL files")

        for file_path in jsonl_files:
            print(f"  Checking {file_path.name}...", end=" ")

            # Load and validate JSON
            entries = self.check_json_validity(file_path)
            if not entries:
                print("FAILED (no valid entries)")
                continue

            # Check format-specific structure
            for i, entry in enumerate(entries, 1):
                if format_type == "openai":
                    self.check_openai_format(entry, file_path, i)
                elif format_type == "sharegpt":
                    self.check_sharegpt_format(entry, file_path, i)
                elif format_type == "raw":
                    self.check_raw_format(entry, file_path, i)

                # Check content quality
                self.check_content_quality(entry, file_path, i, format_type)

            # Find duplicates
            num_dupes = self.find_duplicates(entries, file_path, format_type)

            # Gather statistics
            self.gather_statistics(entries, file_path, format_type)

            # Report
            issue_count = len(self.issues.get(file_path, []))
            if issue_count == 0:
                print(f"OK ({len(entries)} entries)")
            else:
                print(f"ISSUES ({issue_count} issues, {len(entries)} entries)")

    def run_all_checks(self):
        """Run all quality checks."""
        print("=" * 80)
        print("DATA QUALITY CHECK")
        print("=" * 80)

        # Check each format directory
        self.check_directory(self.base_dir / "raw", "raw")
        self.check_directory(self.base_dir / "openai_format", "openai")
        self.check_directory(self.base_dir / "sharegpt_format", "sharegpt")

        # Check train/val/test splits
        print("\n" + "=" * 80)
        print("SPLIT VERIFICATION")
        print("=" * 80)
        for task in ["task_a", "task_b", "task_c"]:
            self.check_train_val_test_split(task, self.base_dir / "raw")
            self.check_train_val_test_split(task, self.base_dir / "openai_format")
            self.check_train_val_test_split(task, self.base_dir / "sharegpt_format")

        # Print summary and return total issues
        return self.print_summary()

    def print_summary(self):
        """Print detailed summary of all issues and statistics."""
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        total_issues = sum(len(issues) for issues in self.issues.values())

        if total_issues == 0:
            print("\n✓ All checks passed! No issues found.")
        else:
            print(f"\n✗ Found {total_issues} issues across {len(self.issues)} files\n")

            for file_path, issues in sorted(self.issues.items()):
                print(f"\n{file_path}:")
                for issue in issues[:20]:  # Show first 20 issues per file
                    print(f"  - {issue}")
                if len(issues) > 20:
                    print(f"  ... and {len(issues) - 20} more issues")

        # Print statistics
        print("\n" + "=" * 80)
        print("STATISTICS")
        print("=" * 80)

        for file_path, stats in sorted(self.stats.items()):
            print(f"\n{file_path.name}:")
            for key, value in sorted(stats.items()):
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                elif isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in sorted(value.items()):
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")

        return total_issues


def main():
    base_dir = Path(__file__).parent
    checker = DataQualityChecker(base_dir)

    try:
        total_issues = checker.run_all_checks()
        sys.exit(1 if total_issues > 0 else 0)
    except Exception as e:
        print(f"\n✗ Error during quality check: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
