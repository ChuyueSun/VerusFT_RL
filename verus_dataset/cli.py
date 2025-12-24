"""
verus_dataset/cli.py - Command-line interface for the dataset pipeline.

Usage:
    python -m verus_dataset.cli --db workspace/verus_dataset.sqlite3 <subcommand> [options]

Subcommands:
    init-db       Initialize the database
    add-repo      Add a repository to track
    scan          Scan repositories for Verus files
    mine          Run the minimization pipeline
    gen-tasks     Generate training tasks
    export-tasks  Export tasks to JSONL
    export-samples Export samples to JSONL
    stats         Show pipeline statistics
"""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .db import Database, init_db
from .scan import scan_repo_to_db, scan_repo
from .wrap import wrap_as_crate_root, prepare_for_reduction
from .verify import run_verify, check_verus_available, VerifyResult
from .reduce import run_creduce, check_creduce_available, save_reduction_logs
from .segment import segment_code, segments_to_json
from .tasks import generate_tasks
from .export import (
    export_tasks_jsonl,
    export_samples_jsonl,
    export_stats,
    export_samples_csv,
    export_tasks_csv,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="verus_dataset",
        description="Minimizer-Driven Extraction pipeline for Verus dataset building",
    )

    parser.add_argument(
        "--db",
        type=str,
        default="workspace/verus_dataset.sqlite3",
        help="Path to SQLite database",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ─── init-db ─────────────────────────────────────────────────────────────
    init_parser = subparsers.add_parser(
        "init-db",
        help="Initialize the database",
    )

    # ─── add-repo ────────────────────────────────────────────────────────────
    repo_parser = subparsers.add_parser(
        "add-repo",
        help="Add a repository to track",
    )
    repo_parser.add_argument("--name", required=True, help="Repository name")
    repo_parser.add_argument("--url", help="Repository URL (for cloning)")
    repo_parser.add_argument("--local-path", help="Local path to existing repo")
    repo_parser.add_argument("--rev", help="Git revision to checkout")
    repo_parser.add_argument("--clone-to", help="Directory to clone into")

    # ─── scan ────────────────────────────────────────────────────────────────
    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan repositories for Verus files",
    )
    scan_parser.add_argument("--repo", help="Scan specific repo by name")
    scan_parser.add_argument(
        "--min-score",
        type=int,
        default=5,
        help="Minimum Verus score (default: 5)",
    )
    scan_parser.add_argument(
        "--dependency-free-only",
        action="store_true",
        help="Only include standalone files",
    )
    scan_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum files to scan per repo",
    )

    # ─── mine ────────────────────────────────────────────────────────────────
    mine_parser = subparsers.add_parser(
        "mine",
        help="Run the minimization pipeline",
    )
    mine_parser.add_argument("--repo", help="Process specific repo by name")
    mine_parser.add_argument(
        "--mode",
        choices=["pass", "fail"],
        default="pass",
        help="Reduction mode (default: pass)",
    )
    mine_parser.add_argument(
        "--min-score",
        type=int,
        default=10,
        help="Minimum Verus score (default: 10)",
    )
    mine_parser.add_argument(
        "--dependency-free-only",
        action="store_true",
        help="Only process standalone files",
    )
    mine_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum files to process",
    )
    mine_parser.add_argument(
        "--verus-cmd",
        default="verus",
        help="Path to verus binary",
    )
    mine_parser.add_argument(
        "--verus-args",
        default="",
        help="Additional verus arguments (space-separated)",
    )
    mine_parser.add_argument(
        "--rustc-args",
        default="--crate-type=lib",
        help="Rustc arguments (default: --crate-type=lib)",
    )
    mine_parser.add_argument(
        "--verus-timeout-sec",
        type=int,
        default=60,
        help="Verus timeout per run (default: 60)",
    )
    mine_parser.add_argument(
        "--creduce-cmd",
        default="creduce",
        help="Path to creduce binary",
    )
    mine_parser.add_argument(
        "--creduce-budget-sec",
        type=int,
        default=1800,
        help="Total creduce timeout per file (default: 1800)",
    )
    mine_parser.add_argument(
        "--skip-reduce",
        action="store_true",
        help="Skip reduction, just verify and store originals",
    )
    mine_parser.add_argument(
        "--work-dir",
        default="workspace/work",
        help="Directory for reduction work files",
    )

    # ─── gen-tasks ───────────────────────────────────────────────────────────
    tasks_parser = subparsers.add_parser(
        "gen-tasks",
        help="Generate training tasks from samples",
    )
    tasks_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum samples to process",
    )
    tasks_parser.add_argument(
        "--max-per-sample",
        type=int,
        default=3,
        help="Maximum tasks per sample (default: 3)",
    )
    tasks_parser.add_argument(
        "--task-types",
        nargs="+",
        choices=["spec_gen", "code_synth", "spec_and_code", "repair"],
        help="Task types to generate",
    )
    tasks_parser.add_argument(
        "--include-fail",
        action="store_true",
        help="Also generate tasks from failing samples",
    )

    # ─── export-tasks ────────────────────────────────────────────────────────
    export_tasks_parser = subparsers.add_parser(
        "export-tasks",
        help="Export tasks to JSONL",
    )
    export_tasks_parser.add_argument(
        "--out",
        required=True,
        help="Output file path",
    )
    export_tasks_parser.add_argument(
        "--format",
        choices=["sft", "full"],
        default="sft",
        help="Export format (default: sft)",
    )
    export_tasks_parser.add_argument(
        "--task-type",
        help="Filter by task type",
    )
    export_tasks_parser.add_argument(
        "--split",
        help="Filter by split",
    )
    export_tasks_parser.add_argument(
        "--include-fail",
        action="store_true",
        help="Include tasks from failing samples",
    )
    export_tasks_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum tasks to export",
    )

    # ─── export-samples ──────────────────────────────────────────────────────
    export_samples_parser = subparsers.add_parser(
        "export-samples",
        help="Export samples to JSONL",
    )
    export_samples_parser.add_argument(
        "--out",
        required=True,
        help="Output file path",
    )
    export_samples_parser.add_argument(
        "--result",
        choices=["pass", "fail"],
        help="Filter by verification result",
    )
    export_samples_parser.add_argument(
        "--minimized-only",
        action="store_true",
        help="Only export minimized samples",
    )
    export_samples_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum samples to export",
    )
    export_samples_parser.add_argument(
        "--csv",
        action="store_true",
        help="Export as CSV instead of JSONL",
    )

    # ─── stats ───────────────────────────────────────────────────────────────
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show pipeline statistics",
    )
    stats_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    stats_parser.add_argument(
        "--out",
        help="Save stats to file",
    )

    # ─── list-repos ──────────────────────────────────────────────────────────
    list_repos_parser = subparsers.add_parser(
        "list-repos",
        help="List registered repositories",
    )

    # ─── check-tools ─────────────────────────────────────────────────────────
    check_parser = subparsers.add_parser(
        "check-tools",
        help="Check if required tools are available",
    )
    check_parser.add_argument(
        "--verus-cmd",
        default="verus",
        help="Path to verus binary",
    )
    check_parser.add_argument(
        "--creduce-cmd",
        default="creduce",
        help="Path to creduce binary",
    )

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Command implementations
# ─────────────────────────────────────────────────────────────────────────────


def cmd_init_db(args: argparse.Namespace) -> int:
    """Initialize the database."""
    db = init_db(args.db)
    logger.info(f"Initialized database at {args.db}")
    db.close()
    return 0


def cmd_add_repo(args: argparse.Namespace) -> int:
    """Add a repository."""
    db = init_db(args.db)

    # Determine local path
    if args.local_path:
        local_path = Path(args.local_path).absolute()
        if not local_path.exists():
            logger.error(f"Local path does not exist: {local_path}")
            return 1
    elif args.url:
        # Clone the repo
        clone_dir = Path(args.clone_to or "workspace/repos")
        clone_dir.mkdir(parents=True, exist_ok=True)
        local_path = clone_dir / args.name

        if local_path.exists():
            logger.info(f"Repo already cloned at {local_path}")
        else:
            logger.info(f"Cloning {args.url} to {local_path}...")
            result = subprocess.run(
                ["git", "clone", "--filter=blob:none", args.url, str(local_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error(f"Clone failed: {result.stderr}")
                return 1
    else:
        logger.error("Must specify --url or --local-path")
        return 1

    # Checkout specific revision if requested
    head_rev = None
    if args.rev:
        logger.info(f"Checking out revision {args.rev}...")
        result = subprocess.run(
            ["git", "-C", str(local_path), "checkout", args.rev],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(f"Checkout failed: {result.stderr}")

    # Get current HEAD
    result = subprocess.run(
        ["git", "-C", str(local_path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        head_rev = result.stdout.strip()

    # Add to database
    repo_id = db.add_repo(
        name=args.name,
        local_path=str(local_path),
        url=args.url,
        pinned_rev=args.rev,
        head_rev=head_rev,
    )

    logger.info(f"Added repo '{args.name}' (id={repo_id})")
    db.close()
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Scan repositories for Verus files."""
    db = init_db(args.db)

    if args.repo:
        repos = [db.get_repo(args.repo)]
        if repos[0] is None:
            logger.error(f"Repo not found: {args.repo}")
            return 1
    else:
        repos = db.list_repos()

    total_files = 0
    for repo in repos:
        logger.info(f"Scanning repo: {repo['name']}")
        count = scan_repo_to_db(
            db,
            repo_name=repo["name"],
            min_score=args.min_score,
            dependency_free_only=args.dependency_free_only,
            limit=args.limit,
        )
        total_files += count

    logger.info(f"Scanned {total_files} total files")
    db.close()
    return 0


def cmd_mine(args: argparse.Namespace) -> int:
    """Run the minimization pipeline."""
    db = init_db(args.db)

    # Check tools
    if not check_verus_available(args.verus_cmd):
        logger.error(f"Verus not available: {args.verus_cmd}")
        return 1

    if not args.skip_reduce and not check_creduce_available(args.creduce_cmd):
        logger.error(f"C-Reduce not available: {args.creduce_cmd}")
        return 1

    # Parse arguments
    verus_args = args.verus_args.split() if args.verus_args else []
    rustc_args = args.rustc_args.split() if args.rustc_args else ["--crate-type=lib"]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Get files to process
    if args.repo:
        repo = db.get_repo(args.repo)
        if repo is None:
            logger.error(f"Repo not found: {args.repo}")
            return 1
        repo_id = repo["repo_id"]
    else:
        repo_id = None

    files = db.list_source_files(
        repo_id=repo_id,
        min_score=args.min_score,
        dependency_free_only=args.dependency_free_only,
        limit=args.limit,
    )

    logger.info(f"Processing {len(files)} files (mode={args.mode})")

    processed = 0
    succeeded = 0

    for file_row in files:
        file_id = file_row["file_id"]
        abs_path = Path(file_row["abs_path"])
        rel_path = file_row["rel_path"]

        logger.info(f"Processing {rel_path}...")

        try:
            # Read and wrap the file
            content = abs_path.read_text(encoding="utf-8")
            wrapped = wrap_as_crate_root(content)

            # Create job work directory
            job_work_dir = work_dir / f"job_{file_id}_{args.mode}"
            job_work_dir.mkdir(parents=True, exist_ok=True)

            # Prepare file for reduction
            input_path = job_work_dir / "lib.rs"
            input_path.write_text(wrapped, encoding="utf-8")

            # Pre-verify
            pre_result = run_verify(
                input_path,
                verus_cmd=args.verus_cmd,
                verus_args=verus_args,
                rustc_args=rustc_args,
                timeout_sec=args.verus_timeout_sec,
            )

            # Log verification run
            db.add_verify_run(
                file_id=file_id,
                stage="pre_verify",
                command=pre_result.command,
                cwd=pre_result.cwd,
                exit_code=pre_result.exit_code,
                timed_out=pre_result.timed_out,
                wall_time_sec=pre_result.wall_time_sec,
                stdout_trunc=pre_result.stdout[:5000],
                stderr_trunc=pre_result.stderr[:5000],
                fingerprint=pre_result.fingerprint,
            )

            # Check if we should proceed
            if args.mode == "pass" and not pre_result.success:
                logger.warning(f"Skipping {rel_path}: does not verify")
                continue
            if args.mode == "fail" and pre_result.success:
                logger.warning(f"Skipping {rel_path}: verifies (expected fail)")
                continue

            # Run reduction or skip
            if args.skip_reduce:
                output_code = wrapped
                reduction_success = True
            else:
                from .reduce import run_creduce

                reduction_result = run_creduce(
                    input_code=wrapped,
                    work_dir=job_work_dir,
                    mode=args.mode,
                    expected_fingerprint=pre_result.fingerprint if args.mode == "fail" else "",
                    verus_cmd=args.verus_cmd,
                    verus_args=verus_args,
                    rustc_args=rustc_args,
                    verus_timeout=args.verus_timeout_sec,
                    creduce_cmd=args.creduce_cmd,
                    creduce_timeout=args.creduce_budget_sec,
                )

                save_reduction_logs(reduction_result, job_work_dir / "logs")

                output_code = reduction_result.output_code
                reduction_success = reduction_result.success

                if not reduction_success:
                    logger.warning(
                        f"Reduction failed for {rel_path}: {reduction_result.error_message}"
                    )

            # Post-verify
            output_path = job_work_dir / "output.rs"
            output_path.write_text(output_code, encoding="utf-8")

            post_result = run_verify(
                output_path,
                verus_cmd=args.verus_cmd,
                verus_args=verus_args,
                rustc_args=rustc_args,
                timeout_sec=args.verus_timeout_sec,
            )

            # Segment the code
            segments = segment_code(output_code)
            segments_json = segments_to_json(segments)

            # Create sample
            repo_row = db.get_repo_by_id(file_row["repo_id"])
            sample_uid = hashlib.sha256(
                f"{repo_row['name']}:{rel_path}:{args.mode}:{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()[:16]

            code_sha256 = hashlib.sha256(output_code.encode()).hexdigest()

            meta = {
                "origin_path": rel_path,
                "reduction_mode": args.mode,
                "minimized": not args.skip_reduce,
                "pre_fingerprint": pre_result.fingerprint,
                "post_fingerprint": post_result.fingerprint,
            }

            verification_result = "pass" if post_result.success else "fail"

            db.add_sample(
                sample_uid=sample_uid,
                repo_id=file_row["repo_id"],
                file_id=file_id,
                minimized=not args.skip_reduce,
                verification_result=verification_result,
                fingerprint=post_result.fingerprint,
                code_sha256=code_sha256,
                code_text=output_code,
                segments_json=segments_json,
                meta_json=json.dumps(meta),
            )

            succeeded += 1
            logger.info(
                f"Created sample {sample_uid} "
                f"({len(wrapped)} -> {len(output_code)} bytes, {verification_result})"
            )

        except Exception as e:
            logger.error(f"Error processing {rel_path}: {e}")
            continue

        processed += 1

    logger.info(f"Processed {processed} files, {succeeded} samples created")
    db.close()
    return 0


def cmd_gen_tasks(args: argparse.Namespace) -> int:
    """Generate training tasks."""
    db = init_db(args.db)

    count = generate_tasks(
        db,
        limit=args.limit,
        task_types=args.task_types,
        max_per_sample=args.max_per_sample,
        only_pass=not args.include_fail,
    )

    logger.info(f"Generated {count} tasks")
    db.close()
    return 0


def cmd_export_tasks(args: argparse.Namespace) -> int:
    """Export tasks to JSONL."""
    db = init_db(args.db)

    count = export_tasks_jsonl(
        db,
        output_path=args.out,
        format=args.format,
        task_type=args.task_type,
        split=args.split,
        only_pass=not args.include_fail,
        limit=args.limit,
    )

    logger.info(f"Exported {count} tasks to {args.out}")
    db.close()
    return 0


def cmd_export_samples(args: argparse.Namespace) -> int:
    """Export samples."""
    db = init_db(args.db)

    if args.csv:
        count = export_samples_csv(
            db,
            output_path=args.out,
            limit=args.limit,
        )
    else:
        count = export_samples_jsonl(
            db,
            output_path=args.out,
            verification_result=args.result,
            minimized_only=args.minimized_only,
            limit=args.limit,
        )

    logger.info(f"Exported {count} samples to {args.out}")
    db.close()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show pipeline statistics."""
    db = init_db(args.db)

    stats = export_stats(db, output_path=args.out if args.out else None)

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print("\n=== Pipeline Statistics ===\n")
        print(f"Repositories:     {stats['repos']}")
        print(f"Source files:     {stats['source_files_total']}")
        print(f"  (dep-free):     {stats['source_files_dependency_free']}")
        print()
        print("Reduction jobs:")
        print(f"  pending:        {stats['jobs_pending']}")
        print(f"  running:        {stats['jobs_running']}")
        print(f"  done:           {stats['jobs_done']}")
        print(f"  timeout:        {stats['jobs_timeout']}")
        print(f"  error:          {stats['jobs_error']}")
        print()
        print("Samples:")
        print(f"  total:          {stats['samples_total']}")
        print(f"  pass:           {stats['samples_pass']}")
        print(f"  fail:           {stats['samples_fail']}")
        print()
        print("Tasks:")
        print(f"  total:          {stats['tasks_total']}")
        print(f"  spec_gen:       {stats['tasks_spec_gen']}")
        print(f"  code_synth:     {stats['tasks_code_synth']}")
        print(f"  spec_and_code:  {stats['tasks_spec_and_code']}")
        print(f"  repair:         {stats['tasks_repair']}")

    db.close()
    return 0


def cmd_list_repos(args: argparse.Namespace) -> int:
    """List registered repositories."""
    db = init_db(args.db)
    repos = db.list_repos()

    if not repos:
        print("No repositories registered.")
        return 0

    print("\nRegistered repositories:\n")
    for repo in repos:
        print(f"  {repo['name']}")
        print(f"    Path: {repo['local_path']}")
        if repo["url"]:
            print(f"    URL:  {repo['url']}")
        if repo["head_rev"]:
            print(f"    HEAD: {repo['head_rev'][:12]}")
        print()

    db.close()
    return 0


def cmd_check_tools(args: argparse.Namespace) -> int:
    """Check if required tools are available."""
    all_ok = True

    print("\nChecking required tools:\n")

    # Check Verus
    verus_ok = check_verus_available(args.verus_cmd)
    status = "OK" if verus_ok else "NOT FOUND"
    print(f"  verus ({args.verus_cmd}): {status}")
    if not verus_ok:
        all_ok = False

    # Check C-Reduce
    creduce_ok = check_creduce_available(args.creduce_cmd)
    status = "OK" if creduce_ok else "NOT FOUND"
    print(f"  creduce ({args.creduce_cmd}): {status}")
    if not creduce_ok:
        all_ok = False

    print()

    if all_ok:
        print("All tools available!")
        return 0
    else:
        print("Some tools are missing. Please install them before running the pipeline.")
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command is None:
        parser.print_help()
        return 1

    # Dispatch to command handler
    handlers = {
        "init-db": cmd_init_db,
        "add-repo": cmd_add_repo,
        "scan": cmd_scan,
        "mine": cmd_mine,
        "gen-tasks": cmd_gen_tasks,
        "export-tasks": cmd_export_tasks,
        "export-samples": cmd_export_samples,
        "stats": cmd_stats,
        "list-repos": cmd_list_repos,
        "check-tools": cmd_check_tools,
    }

    handler = handlers.get(args.command)
    if handler is None:
        logger.error(f"Unknown command: {args.command}")
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
