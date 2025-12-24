#!/usr/bin/env python3
"""
Interactive JSONL dataset viewer.
Browse through each example with keyboard navigation.
"""

import json
import sys
from pathlib import Path


class DatasetViewer:
    def __init__(self, jsonl_file):
        self.jsonl_file = Path(jsonl_file)
        self.entries = []
        self.current_index = 0

        # Load all entries
        with open(self.jsonl_file) as f:
            for line in f:
                self.entries.append(json.loads(line))

        print(f"Loaded {len(self.entries)} entries from {self.jsonl_file.name}")

    def clear_screen(self):
        """Clear the terminal screen."""
        print("\033[2J\033[H", end="")

    def format_code(self, code, max_lines=None):
        """Format code with line numbers."""
        lines = code.split('\n')
        if max_lines and len(lines) > max_lines:
            lines = lines[:max_lines] + ['... (truncated)']

        formatted = []
        for i, line in enumerate(lines, 1):
            formatted.append(f"{i:4d} | {line}")
        return '\n'.join(formatted)

    def display_task_a(self, entry):
        """Display Task A (Code → Spec) entry."""
        metadata = entry.get('metadata', {})

        print("=" * 80)
        print(f"TASK A: CODE → SPECIFICATIONS")
        print("=" * 80)
        print(f"ID: {entry['id']}")
        print(f"Function: {metadata.get('function_name', 'unknown')}")
        print(f"Source: {metadata.get('source_file', 'unknown')}:{metadata.get('line_start', '?')}")
        print(f"Mode: {metadata.get('function_mode', 'unknown')} | Complexity: {metadata.get('complexity_score', 0)}")
        print(f"Has loop invariant: {metadata.get('has_loop_invariant', False)}")
        print(f"Has proof block: {metadata.get('has_proof_block', False)}")
        print()

        print("─" * 80)
        print("INPUT PROMPT:")
        print("─" * 80)
        print(entry.get('input_text', ''))
        print()

        print("─" * 80)
        print("EXPECTED OUTPUT (Specifications):")
        print("─" * 80)
        print(entry.get('target_text', ''))
        print()

        if 'full_function' in entry and entry['full_function']:
            print("─" * 80)
            print("FULL FUNCTION (Reference):")
            print("─" * 80)
            print(self.format_code(entry['full_function'], max_lines=50))
            print()

    def display_task_b(self, entry):
        """Display Task B (Spec → Code) entry."""
        metadata = entry.get('metadata', {})

        print("=" * 80)
        print(f"TASK B: SPECIFICATIONS → CODE")
        print("=" * 80)
        print(f"ID: {entry['id']}")
        print(f"Function: {metadata.get('function_name', 'unknown')}")
        print(f"Source: {metadata.get('source_file', 'unknown')}")
        print(f"Mode: {metadata.get('function_mode', 'unknown')} | Complexity: {metadata.get('complexity_score', 0)}")
        print()

        print("─" * 80)
        print("INPUT PROMPT (Specifications):")
        print("─" * 80)
        print(entry.get('input_text', ''))
        print()

        print("─" * 80)
        print("EXPECTED OUTPUT (Implementation):")
        print("─" * 80)
        print(entry.get('target_text', ''))
        print()

        if 'full_function' in entry and entry['full_function']:
            print("─" * 80)
            print("FULL FUNCTION (Reference):")
            print("─" * 80)
            print(self.format_code(entry['full_function'], max_lines=50))
            print()

    def display_task_c(self, entry):
        """Display Task C (Error Repair) entry."""
        metadata = entry.get('metadata', {})
        error = entry.get('error', {})

        is_identical = entry.get('broken_code') == entry.get('fixed_code')

        print("=" * 80)
        print(f"TASK C: ERROR-GUIDED REPAIR")
        if is_identical:
            print("⚠️  WARNING: Broken and Fixed code are IDENTICAL!")
        print("=" * 80)
        print(f"ID: {entry['id']}")
        print(f"Function: {metadata.get('function_name', 'unknown')}")
        print(f"Source: {metadata.get('source_file', 'unknown')}")
        print(f"Error Type: {error.get('error_type', 'unknown')}")
        print(f"Mutation Type: {metadata.get('mutation_type', 'unknown')}")
        print()

        print("─" * 80)
        print("ERROR MESSAGE:")
        print("─" * 80)
        print(error.get('message', 'N/A'))
        print()

        print("─" * 80)
        print("INPUT PROMPT:")
        print("─" * 80)
        input_text = entry.get('input_text', '')
        if len(input_text) > 1000:
            print(input_text[:1000] + "\n... (truncated, use 'f' to see full)")
        else:
            print(input_text)
        print()

        print("─" * 80)
        print("BROKEN CODE:")
        print("─" * 80)
        print(self.format_code(entry.get('broken_code', ''), max_lines=30))
        print()

        print("─" * 80)
        print("FIXED CODE (Target):")
        print("─" * 80)
        print(self.format_code(entry.get('fixed_code', ''), max_lines=30))
        print()

    def display_openai(self, entry):
        """Display OpenAI format entry."""
        messages = entry.get('messages', [])

        print("=" * 80)
        print(f"OPENAI FORMAT - {len(messages)} messages")
        print("=" * 80)
        print()

        for i, msg in enumerate(messages, 1):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')

            # Color codes for different roles
            colors = {
                'system': '\033[94m',  # Blue
                'user': '\033[95m',     # Magenta
                'assistant': '\033[92m' # Green
            }
            reset = '\033[0m'

            color = colors.get(role, '')

            print(f"{color}{'─' * 80}")
            print(f"MESSAGE {i}: {role.upper()}")
            print(f"{'─' * 80}{reset}")

            if len(content) > 1500:
                print(content[:1500] + "\n... (truncated, use 'f' to see full)")
            else:
                print(content)
            print()

    def display_sharegpt(self, entry):
        """Display ShareGPT format entry."""
        conversations = entry.get('conversations', [])

        print("=" * 80)
        print(f"SHAREGPT FORMAT - {len(conversations)} conversations")
        print("=" * 80)
        print()

        for i, conv in enumerate(conversations, 1):
            from_who = conv.get('from', 'unknown')
            value = conv.get('value', '')

            # Color codes
            colors = {
                'system': '\033[94m',
                'human': '\033[95m',
                'gpt': '\033[92m'
            }
            reset = '\033[0m'

            color = colors.get(from_who, '')

            print(f"{color}{'─' * 80}")
            print(f"CONVERSATION {i}: {from_who.upper()}")
            print(f"{'─' * 80}{reset}")

            if len(value) > 1500:
                print(value[:1500] + "\n... (truncated, use 'f' to see full)")
            else:
                print(value)
            print()

    def display_current(self):
        """Display current entry."""
        self.clear_screen()

        entry = self.entries[self.current_index]

        # Detect format and task type
        if 'messages' in entry:
            self.display_openai(entry)
        elif 'conversations' in entry:
            self.display_sharegpt(entry)
        else:
            # Raw format - detect task type
            task = entry.get('task', '')
            if task == 'code_to_spec':
                self.display_task_a(entry)
            elif task == 'spec_to_code':
                self.display_task_b(entry)
            elif task == 'error_repair':
                self.display_task_c(entry)
            else:
                # Fallback: print JSON
                print(json.dumps(entry, indent=2))

        # Navigation info
        print("=" * 80)
        print(f"Entry {self.current_index + 1} of {len(self.entries)}")
        print("=" * 80)
        print("Commands: [n]ext | [p]revious | [j]ump | [s]earch | [f]ull | [q]uit")
        print("─" * 80)

    def jump_to(self):
        """Jump to a specific entry number."""
        try:
            num = input(f"Enter entry number (1-{len(self.entries)}): ")
            index = int(num) - 1
            if 0 <= index < len(self.entries):
                self.current_index = index
                return True
            else:
                print(f"Invalid entry number. Must be between 1 and {len(self.entries)}")
                input("Press Enter to continue...")
                return False
        except ValueError:
            print("Invalid number")
            input("Press Enter to continue...")
            return False

    def search(self):
        """Search for entries by ID or content."""
        query = input("Enter search term (searches ID): ").lower()
        matches = []

        for i, entry in enumerate(self.entries):
            entry_id = entry.get('id', '')
            if query in entry_id.lower():
                matches.append((i, entry_id))

        if not matches:
            print(f"No matches found for '{query}'")
            input("Press Enter to continue...")
            return False

        print(f"\nFound {len(matches)} matches:")
        for i, (index, entry_id) in enumerate(matches[:20], 1):
            print(f"{i}. Entry {index + 1}: {entry_id}")

        if len(matches) > 20:
            print(f"... and {len(matches) - 20} more")

        try:
            choice = input("\nSelect match number (or Enter to cancel): ")
            if choice:
                match_idx = int(choice) - 1
                if 0 <= match_idx < len(matches):
                    self.current_index = matches[match_idx][0]
                    return True
        except ValueError:
            pass

        return False

    def show_full(self):
        """Show full entry as JSON."""
        self.clear_screen()
        entry = self.entries[self.current_index]
        print(json.dumps(entry, indent=2))
        print("\n" + "=" * 80)
        input("Press Enter to continue...")

    def run(self):
        """Run the interactive viewer."""
        while True:
            self.display_current()

            try:
                command = input("Command: ").lower().strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if command in ['n', 'next', '']:
                if self.current_index < len(self.entries) - 1:
                    self.current_index += 1
                else:
                    print("Already at last entry")
                    input("Press Enter to continue...")

            elif command in ['p', 'prev', 'previous']:
                if self.current_index > 0:
                    self.current_index -= 1
                else:
                    print("Already at first entry")
                    input("Press Enter to continue...")

            elif command in ['j', 'jump']:
                self.jump_to()

            elif command in ['s', 'search']:
                self.search()

            elif command in ['f', 'full']:
                self.show_full()

            elif command in ['q', 'quit', 'exit']:
                print("Exiting...")
                break

            else:
                print(f"Unknown command: {command}")
                input("Press Enter to continue...")


def main():
    if len(sys.argv) < 2:
        print("Interactive JSONL Dataset Viewer")
        print("=" * 80)
        print("\nUsage: python view_dataset.py <jsonl_file>")
        print("\nAvailable datasets:")

        base_dir = Path(__file__).parent

        print("\n📝 Raw format:")
        for f in sorted((base_dir / "raw").glob("*.jsonl")):
            count = sum(1 for _ in open(f))
            print(f"  {f.relative_to(base_dir)} ({count} entries)")

        print("\n🤖 OpenAI format:")
        for f in sorted((base_dir / "openai_format").glob("*.jsonl")):
            count = sum(1 for _ in open(f))
            print(f"  {f.relative_to(base_dir)} ({count} entries)")

        print("\n💬 ShareGPT format:")
        for f in sorted((base_dir / "sharegpt_format").glob("*.jsonl")):
            count = sum(1 for _ in open(f))
            print(f"  {f.relative_to(base_dir)} ({count} entries)")

        print("\nExample:")
        print("  python view_dataset.py raw/task_a_all.jsonl")
        print("  python view_dataset.py openai_format/task_a_train_openai.jsonl")
        print("  python view_dataset.py raw/task_c_all_filtered.jsonl")

        sys.exit(1)

    jsonl_file = sys.argv[1]

    if not Path(jsonl_file).exists():
        print(f"Error: File not found: {jsonl_file}")
        sys.exit(1)

    viewer = DatasetViewer(jsonl_file)
    viewer.run()


if __name__ == "__main__":
    main()
