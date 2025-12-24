"""
verus_dataset - Minimizer-Driven Extraction pipeline for Verus dataset building.

This module provides a complete pipeline for:
1. Scanning repos for Verus-compatible Rust files
2. Wrapping files into standalone crate roots
3. Running verification with the Verus verifier
4. Minimizing with C-Reduce while preserving PASS/FAIL properties
5. Segmenting code into exec/spec/proof zones
6. Generating SFT-style training tasks

Usage:
    python -m verus_dataset.cli --db workspace/verus_dataset.sqlite3 <subcommand> ...
"""

__version__ = "0.1.0"

from .db import Database, init_db
from .scan import scan_repo, score_verus_file
from .wrap import wrap_as_crate_root
from .verify import run_verify, VerifyResult
from .reduce import run_creduce, write_interestingness_script
from .segment import segment_code, Segment
from .tasks import generate_tasks
from .export import export_tasks_jsonl, export_samples_jsonl

__all__ = [
    "Database",
    "init_db",
    "scan_repo",
    "score_verus_file",
    "wrap_as_crate_root",
    "run_verify",
    "VerifyResult",
    "run_creduce",
    "write_interestingness_script",
    "segment_code",
    "Segment",
    "generate_tasks",
    "export_tasks_jsonl",
    "export_samples_jsonl",
]
