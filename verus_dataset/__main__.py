"""
Allow running the module with: python -m verus_dataset

Usage:
    python -m verus_dataset --db workspace/verus_dataset.sqlite3 <subcommand> [options]
"""

from .cli import main

if __name__ == "__main__":
    exit(main())
