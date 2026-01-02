#!/usr/bin/env python3
"""
Interactive JSONL dataset editor.
Browse and edit entries with your preferred text editor.
"""

import json
import sys
import os
import tempfile
import subprocess
from pathlib import Path


class DatasetEditor:
    def __init__(self, jsonl_file):
        self.jsonl_file = Path(jsonl_file)
        self.backup_file = self.jsonl_file.with_suffix('.jsonl.backup')
        self.comments_file = self.jsonl_file.with_suffix('.comments.txt')
        self.entries = []
        self.current_index = 0
        self.modified = False
        self.edit_comments = []  # Store comments for each edit

        # Load all entries
        with open(self.jsonl_file) as f:
            for line in f:
                self.entries.append(json.loads(line))

        print(f"Loaded {len(self.entries)} entries from {self.jsonl_file.name}")

        # Create backup
        if not self.backup_file.exists():
            self.create_backup()

        # Load existing comments if any
        if self.comments_file.exists():
            with open(self.comments_file) as f:
                content = f.read()
                if content.strip():
                    self.edit_comments = content.strip().split('\n---\n')
            print(f"Loaded {len(self.edit_comments)} existing comments")

    def create_backup(self):
        """Create a backup of the original file."""
        with open(self.backup_file, 'w') as f:
            for entry in self.entries:
                f.write(json.dumps(entry) + '\n')
        print(f"Created backup: {self.backup_file.name}")

    def save_changes(self):
        """Save all entries back to the file."""
        with open(self.jsonl_file, 'w') as f:
            for entry in self.entries:
                f.write(json.dumps(entry) + '\n')
        print(f"✓ Saved changes to {self.jsonl_file.name}")

        # Save comments
        if self.edit_comments:
            with open(self.comments_file, 'w') as f:
                f.write('\n---\n'.join(self.edit_comments))
            print(f"✓ Saved {len(self.edit_comments)} comments to {self.comments_file.name}")

        self.modified = False

    def clear_screen(self):
        """Clear the terminal screen."""
        print("\033[2J\033[H", end="")

    def display_entry(self, entry):
        """Display an entry in a readable format."""
        # Detect format
        if 'messages' in entry:
            self.display_openai(entry)
        elif 'conversations' in entry:
            self.display_sharegpt(entry)
        else:
            self.display_raw(entry)

    def display_raw(self, entry):
        """Display raw format entry."""
        print("=" * 80)
        print(f"ID: {entry.get('id', 'unknown')}")
        print(f"Task: {entry.get('task', 'unknown')}")
        print("=" * 80)
        print()

        print("─" * 80)
        print("INPUT:")
        print("─" * 80)
        print(entry.get('input_text', '')[:800])
        if len(entry.get('input_text', '')) > 800:
            print("... (truncated)")
        print()

        print("─" * 80)
        print("TARGET:")
        print("─" * 80)
        print(entry.get('target_text', '')[:400])
        if len(entry.get('target_text', '')) > 400:
            print("... (truncated)")
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

            colors = {
                'system': '\033[94m',
                'user': '\033[95m',
                'assistant': '\033[92m'
            }
            reset = '\033[0m'
            color = colors.get(role, '')

            print(f"{color}─ MESSAGE {i}: {role.upper()} ─{reset}")
            print(content[:600])
            if len(content) > 600:
                print("... (truncated)")
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

            colors = {
                'system': '\033[94m',
                'human': '\033[95m',
                'gpt': '\033[92m'
            }
            reset = '\033[0m'
            color = colors.get(from_who, '')

            print(f"{color}─ CONVERSATION {i}: {from_who.upper()} ─{reset}")
            print(value[:600])
            if len(value) > 600:
                print("... (truncated)")
            print()

    def display_current(self):
        """Display current entry."""
        self.clear_screen()

        entry = self.entries[self.current_index]
        self.display_entry(entry)

        # Navigation info
        print("=" * 80)
        print(f"Entry {self.current_index + 1} of {len(self.entries)}")
        if self.modified:
            print("⚠️  UNSAVED CHANGES")
        if self.edit_comments:
            print(f"📝 {len(self.edit_comments)} comments recorded")
        print("=" * 80)
        print("Commands: [n]ext | [p]revious | [e]dit | [d]elete | [s]ave | [c]omments | [j]ump | [q]uit")
        print("─" * 80)

    def edit_current(self):
        """Edit the current entry with a text editor."""
        entry = self.entries[self.current_index]

        # Determine editor
        editor = os.environ.get('EDITOR', 'nano')

        # Create temporary file with formatted JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
            json.dump(entry, f, indent=2)

        try:
            # Open editor
            subprocess.run([editor, temp_file])

            # Read back the edited content
            with open(temp_file) as f:
                edited_entry = json.load(f)

            # Check if entry was actually modified
            if edited_entry != entry:
                # Update entry
                self.entries[self.current_index] = edited_entry
                self.modified = True

                # Ask for comment about the edit
                print("\n✓ Entry updated")
                print("\nAdd a comment about what you changed (optional, press Enter to skip):")
                comment = input("> ").strip()

                if comment:
                    comment_entry = f"Entry {self.current_index + 1} ({entry.get('id', 'unknown')}):\n{comment}"
                    self.edit_comments.append(comment_entry)
                    print("✓ Comment saved")

                input("\nPress Enter to continue...")
            else:
                print("\n○ No changes made")
                input("Press Enter to continue...")

        except json.JSONDecodeError as e:
            print(f"\n✗ Invalid JSON: {e}")
            print("Entry not updated.")
            input("Press Enter to continue...")

        finally:
            # Clean up temp file
            os.unlink(temp_file)

    def delete_current(self):
        """Delete the current entry."""
        entry = self.entries[self.current_index]
        entry_id = entry.get('id', entry.get('messages', [{}])[0].get('content', '')[:50])

        confirm = input(f"Delete entry '{entry_id}'? [y/N]: ")
        if confirm.lower() == 'y':
            # Ask for reason
            print("\nWhy are you deleting this entry? (optional):")
            reason = input("> ").strip()

            if reason:
                comment_entry = f"Entry {self.current_index + 1} ({entry_id}):\nDELETED - {reason}"
                self.edit_comments.append(comment_entry)

            del self.entries[self.current_index]
            self.modified = True

            # Adjust index
            if self.current_index >= len(self.entries):
                self.current_index = len(self.entries) - 1

            print("✓ Entry deleted")
        else:
            print("Cancelled")

        input("Press Enter to continue...")

    def view_comments(self):
        """View all recorded comments."""
        self.clear_screen()
        print("=" * 80)
        print("RECORDED COMMENTS")
        print("=" * 80)

        if not self.edit_comments:
            print("\nNo comments recorded yet.")
        else:
            for i, comment in enumerate(self.edit_comments, 1):
                print(f"\n{i}. {comment}")
                print()

        print("=" * 80)
        print(f"Total: {len(self.edit_comments)} comments")
        print("=" * 80)
        input("\nPress Enter to continue...")

    def jump_to(self):
        """Jump to a specific entry number."""
        try:
            num = input(f"Enter entry number (1-{len(self.entries)}): ")
            index = int(num) - 1
            if 0 <= index < len(self.entries):
                self.current_index = index
            else:
                print(f"Invalid entry number. Must be between 1 and {len(self.entries)}")
                input("Press Enter to continue...")
        except ValueError:
            print("Invalid number")
            input("Press Enter to continue...")

    def run(self):
        """Run the interactive editor."""
        while True:
            if len(self.entries) == 0:
                print("No entries left in dataset!")
                break

            self.display_current()

            try:
                command = input("Command: ").lower().strip()
            except (EOFError, KeyboardInterrupt):
                print("\n")
                if self.modified:
                    save = input("Save changes before exiting? [Y/n]: ")
                    if save.lower() != 'n':
                        self.save_changes()
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

            elif command in ['e', 'edit']:
                self.edit_current()

            elif command in ['d', 'delete']:
                self.delete_current()

            elif command in ['s', 'save']:
                self.save_changes()
                input("Press Enter to continue...")

            elif command in ['c', 'comments']:
                self.view_comments()

            elif command in ['j', 'jump']:
                self.jump_to()

            elif command in ['q', 'quit', 'exit']:
                if self.modified:
                    save = input("Save changes before exiting? [Y/n]: ")
                    if save.lower() != 'n':
                        self.save_changes()
                break

            else:
                print(f"Unknown command: {command}")
                input("Press Enter to continue...")


def main():
    if len(sys.argv) < 2:
        print("Interactive JSONL Dataset Editor")
        print("=" * 80)
        print("\nUsage: python edit_dataset.py <jsonl_file>")
        print("\nThis tool allows you to:")
        print("  - Browse through dataset entries")
        print("  - Edit entries in your text editor ($EDITOR or nano)")
        print("  - Delete problematic entries")
        print("  - Save changes back to the file")
        print("\nA backup (.jsonl.backup) is automatically created")
        print("\nExample:")
        print("  python edit_dataset.py raw/task_a_train_fixed.jsonl")
        print("  export EDITOR=vim  # Set your preferred editor")
        print("  python edit_dataset.py raw/task_a_train_fixed.jsonl")
        sys.exit(1)

    jsonl_file = sys.argv[1]

    if not Path(jsonl_file).exists():
        print(f"Error: File not found: {jsonl_file}")
        sys.exit(1)

    editor = DatasetEditor(jsonl_file)
    editor.run()


if __name__ == "__main__":
    main()
