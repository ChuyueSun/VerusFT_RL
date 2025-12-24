"""
verus_dataset/mine_inplace.py - Mine samples by verifying files in-place.

This verifies files in their original repo context, which works better
for files that depend on repo-specific imports.
"""

import hashlib
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path

from .db import Database
from .segment import segment_code, segments_to_json

logger = logging.getLogger(__name__)


def is_stub_only(content: str) -> bool:
    """
    Check if a file only contains stub implementations.

    Returns True if ALL exec functions in the file are stubs
    (assume(false); unreached() without real implementation code).
    """
    # Must have a stub pattern (either assume(false) or unreached())
    if "assume(false)" not in content and "unreached()" not in content:
        return False

    # Find all exec functions and check if ANY has a real implementation
    lines = content.split('\n')

    # Check if file uses vc-code markers
    has_vc_markers = '// <vc-code>' in content

    in_exec_fn = False
    in_spec_fn = False
    in_fn_body = False  # True only after the actual body { starts
    in_spec_clause = False  # Track requires/ensures blocks
    brace_depth = 0
    saw_assume_false = False  # Track if we've seen assume(false) in current fn body

    for line in lines:
        stripped = line.strip()

        # Track spec/proof function bodies (skip these entirely)
        if 'spec fn ' in line or 'proof fn ' in line:
            in_spec_fn = True
            brace_depth = 0
            continue

        # End of spec fn
        if in_spec_fn:
            brace_depth += line.count('{') - line.count('}')
            if brace_depth <= 0 and '}' in line:
                in_spec_fn = False
                brace_depth = 0
            continue

        # Track exec function start
        if 'fn ' in line and 'spec fn' not in line and 'proof fn' not in line:
            in_exec_fn = True
            in_fn_body = False
            in_spec_clause = False
            brace_depth = 0
            saw_assume_false = False
            continue

        if in_exec_fn and not in_fn_body:
            # Check if we're entering a spec clause
            if any(kw in stripped for kw in ['requires', 'ensures', 'recommends', 'decreases']):
                in_spec_clause = True

            # The body starts at '// <vc-code>' marker
            if '// <vc-code>' in line:
                in_fn_body = True
                in_spec_clause = False
                brace_depth = 0
                continue

            # A standalone '{' indicates body start, but ONLY if file doesn't use vc markers
            # (vc-marked files may have complex expressions with braces in spec clauses)
            if stripped == '{' and not has_vc_markers:
                in_fn_body = True
                in_spec_clause = False
                brace_depth = 1
                continue

            # Skip spec clause content
            if in_spec_clause:
                continue

        if in_exec_fn and in_fn_body:
            open_b = line.count('{')
            close_b = line.count('}')

            if brace_depth == 0 and open_b > 0:
                brace_depth = 1

            if brace_depth > 0:
                brace_depth += open_b - close_b

                # Skip empty, comments
                if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
                    continue
                # Skip stub patterns
                if 'assume(false)' in stripped or 'unreached()' in stripped:
                    saw_assume_false = True
                    continue
                # Skip braces and markers
                if stripped in ['{', '}', '/* impl-start */', '/* impl-end */', 'proof {']:
                    continue

                # After assume(false), everything is unreachable placeholder code
                # Skip all remaining content in this function body
                if saw_assume_false:
                    continue

                # Real implementation patterns: any non-trivial code
                if stripped.startswith('let ') or stripped.startswith('let mut '):
                    return False
                if stripped.startswith('while ') or stripped.startswith('for '):
                    return False
                if stripped.startswith('if '):
                    return False
                if stripped.startswith('match '):
                    return False
                if '(' in stripped:
                    # Function call
                    return False
                # Any other non-empty, non-brace content is likely real code
                if stripped and stripped not in [';']:
                    return False

            if brace_depth <= 0:
                in_exec_fn = False
                in_fn_body = False

    return True  # Only stubs found


def verify_inplace(
    file_path: Path,
    verus_cmd: str,
    timeout_sec: int = 60,
) -> tuple[bool, str, float]:
    """
    Verify a file in its original location.

    Returns (success, output, wall_time_sec)
    """
    import os
    start = time.monotonic()
    try:
        env = os.environ.copy()
        env["NO_COLOR"] = "1"
        env["TERM"] = "dumb"
        result = subprocess.run(
            [verus_cmd, str(file_path)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=file_path.parent,
            env=env,
        )
        wall_time = time.monotonic() - start
        output = result.stdout + "\n" + result.stderr
        success = result.returncode == 0 and "0 errors" in output
        return success, output, wall_time
    except subprocess.TimeoutExpired:
        return False, "[TIMEOUT]", time.monotonic() - start
    except Exception as e:
        return False, f"[ERROR: {e}]", time.monotonic() - start


def mine_inplace(
    db: Database,
    verus_cmd: str,
    min_score: int = 10,
    limit: int | None = None,
    timeout_sec: int = 60,
    repo_name: str | None = None,
) -> int:
    """
    Mine samples by verifying files in-place.

    Only files that successfully verify are added as samples.
    """
    # Get repo filter
    repo_id = None
    if repo_name:
        repo = db.get_repo(repo_name)
        if repo:
            repo_id = repo["repo_id"]

    files = db.list_source_files(
        repo_id=repo_id,
        min_score=min_score,
        dependency_free_only=False,  # Don't filter - verify in place
        limit=limit,
    )

    logger.info(f"Processing {len(files)} files for in-place verification...")

    verified = 0
    failed = 0

    for file_row in files:
        file_path = Path(file_row["abs_path"])
        rel_path = file_row["rel_path"]

        if not file_path.exists():
            continue

        # Skip files marked as expect-failures
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            if "expect-fail" in content[:200].lower():
                logger.debug(f"Skipping {rel_path}: marked as expect-failures")
                continue
        except Exception:
            continue

        # Skip stub-only files (specs without real implementations)
        if is_stub_only(content):
            logger.debug(f"Skipping {rel_path}: stub-only (assume(false); unreached())")
            continue

        logger.info(f"Verifying {rel_path}...")

        success, output, wall_time = verify_inplace(file_path, verus_cmd, timeout_sec)

        if success:
            # Create sample from the verified file
            repo = db.get_repo_by_id(file_row["repo_id"])
            sample_uid = hashlib.sha256(
                f"{repo['name']}:{rel_path}:verified".encode()
            ).hexdigest()[:16]

            # Check if already exists
            if db.get_sample_by_uid(sample_uid):
                logger.debug(f"Sample already exists: {sample_uid}")
                verified += 1
                continue

            # Segment the code (with error handling)
            try:
                segments = segment_code(content)
                segments_json_str = segments_to_json(segments)
            except Exception as e:
                logger.debug(f"Segmentation failed for {rel_path}: {e}")
                segments_json_str = "[]"

            code_sha256 = hashlib.sha256(content.encode()).hexdigest()

            meta = {
                "origin_path": rel_path,
                "verified_inplace": True,
                "wall_time_sec": wall_time,
            }

            db.add_sample(
                sample_uid=sample_uid,
                repo_id=file_row["repo_id"],
                file_id=file_row["file_id"],
                minimized=False,
                verification_result="pass",
                fingerprint="PASS",
                code_sha256=code_sha256,
                code_text=content,
                segments_json=segments_json_str,
                meta_json=json.dumps(meta),
            )

            verified += 1
            logger.info(f"  PASS ({wall_time:.1f}s)")
        else:
            failed += 1
            logger.debug(f"  FAIL: {output[:200]}")

    logger.info(f"Verified {verified} files, {failed} failed")
    return verified


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python -m verus_dataset.mine_inplace <db_path> <verus_cmd> [limit]")
        sys.exit(1)

    db_path = sys.argv[1]
    verus_cmd = sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None

    from .db import init_db
    db = init_db(db_path)
    count = mine_inplace(db, verus_cmd, limit=limit)
    print(f"Created {count} verified samples")
    db.close()
