"""
verus_dataset/verify.py - Verification harness for running Verus.

Provides a unified interface for running Verus verification with:
- Configurable timeout
- Structured result parsing
- Stable fingerprinting for FAIL preservation
"""

import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_VERUS_CMD = "verus"
DEFAULT_TIMEOUT_SEC = 60
DEFAULT_MAX_OUTPUT_LEN = 50000


@dataclass
class VerifyResult:
    """Result of a Verus verification run."""

    # Basic result
    success: bool
    exit_code: int
    timed_out: bool
    wall_time_sec: float

    # Output
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    # Fingerprint for FAIL preservation
    fingerprint: str = ""

    # Parsed info (if available)
    verified_count: Optional[int] = None
    error_count: Optional[int] = None
    error_messages: list[str] = field(default_factory=list)
    error_type: Optional[str] = None

    # Command info for logging
    command: str = ""
    cwd: str = ""


def normalize_error_output(stderr: str) -> str:
    """
    Normalize error output for stable fingerprinting.

    Removes:
    - ANSI color codes
    - Absolute paths (replaced with <PATH>)
    - Line/column numbers (replaced with <LOC>)
    - Memory addresses
    - Timestamps
    """
    result = stderr

    # Remove ANSI escape codes
    result = re.sub(r"\x1b\[[0-9;]*m", "", result)
    result = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", result)

    # Replace absolute paths with placeholder
    # Matches /path/to/file.rs or C:\path\to\file.rs
    result = re.sub(r"[A-Za-z]?:?[/\\][\w\-./\\]+\.rs", "<PATH>", result)

    # Replace line:col patterns
    result = re.sub(r":\d+:\d+", ":<LOC>", result)
    result = re.sub(r"line \d+", "line <LOC>", result)
    result = re.sub(r"column \d+", "column <LOC>", result)

    # Replace memory addresses (0x...)
    result = re.sub(r"0x[0-9a-fA-F]+", "<ADDR>", result)

    # Replace timing info
    result = re.sub(r"\d+\.\d+s", "<TIME>", result)
    result = re.sub(r"\d+ms", "<TIME>", result)

    # Normalize whitespace
    result = re.sub(r"\s+", " ", result).strip()

    return result


def compute_fingerprint(stderr: str) -> str:
    """
    Compute a stable fingerprint for error output.

    Used for FAIL-preserving reduction to ensure the same error persists.
    """
    if not stderr.strip():
        return "PASS"

    normalized = normalize_error_output(stderr)
    hash_bytes = hashlib.sha256(normalized.encode("utf-8")).digest()
    return hash_bytes[:8].hex()


def parse_verification_output(stdout: str, stderr: str) -> dict:
    """
    Parse Verus output for structured information.

    Attempts to extract:
    - Number of verified functions
    - Number of errors
    - Error types
    """
    result = {
        "verified_count": None,
        "error_count": None,
        "error_messages": [],
        "error_type": None,
    }

    combined = stdout + "\n" + stderr

    # Look for verification summary
    # Pattern: "verification results:: N verified, M errors"
    match = re.search(r"(\d+)\s+verified", combined)
    if match:
        result["verified_count"] = int(match.group(1))

    match = re.search(r"(\d+)\s+errors?", combined)
    if match:
        result["error_count"] = int(match.group(1))

    # Extract error messages
    error_patterns = [
        r"error\[.*?\]:\s*(.+?)(?:\n|$)",
        r"error:\s*(.+?)(?:\n|$)",
        r"FAIL:\s*(.+?)(?:\n|$)",
    ]

    for pattern in error_patterns:
        for match in re.finditer(pattern, combined, re.IGNORECASE):
            msg = match.group(1).strip()
            if msg and len(msg) > 5:
                result["error_messages"].append(msg[:200])

    # Categorize error type
    if result["error_messages"]:
        first_error = result["error_messages"][0].lower()
        if "precondition" in first_error or "requires" in first_error:
            result["error_type"] = "precondition"
        elif "postcondition" in first_error or "ensures" in first_error:
            result["error_type"] = "postcondition"
        elif "invariant" in first_error:
            result["error_type"] = "invariant"
        elif "assertion" in first_error:
            result["error_type"] = "assertion"
        elif "type" in first_error:
            result["error_type"] = "type_error"
        elif "syntax" in first_error or "parse" in first_error:
            result["error_type"] = "syntax"
        elif "termination" in first_error or "decreases" in first_error:
            result["error_type"] = "termination"
        else:
            result["error_type"] = "other"

    return result


def truncate_output(output: str, max_len: int = DEFAULT_MAX_OUTPUT_LEN) -> tuple[str, bool]:
    """Truncate output if too long, return (output, was_truncated)."""
    if len(output) <= max_len:
        return output, False
    return output[:max_len] + "\n... [truncated]", True


def build_verus_command(
    file_path: Path,
    verus_cmd: str = DEFAULT_VERUS_CMD,
    verus_args: list[str] | None = None,
    rustc_args: list[str] | None = None,
) -> list[str]:
    """
    Build the Verus command line.

    Args:
        file_path: Path to the .rs file to verify
        verus_cmd: Path to verus binary or "verus"
        verus_args: Additional arguments for verus (e.g., ["--verify"])
        rustc_args: Additional rustc arguments (e.g., ["--crate-type=lib"])

    Returns:
        Command as a list of strings
    """
    cmd = [verus_cmd]

    if verus_args:
        cmd.extend(verus_args)

    cmd.append(str(file_path))

    if rustc_args:
        cmd.extend(rustc_args)

    return cmd


def run_verify(
    file_path: Path | str,
    verus_cmd: str = DEFAULT_VERUS_CMD,
    verus_args: list[str] | None = None,
    rustc_args: list[str] | None = None,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    cwd: Path | str | None = None,
    env: dict | None = None,
) -> VerifyResult:
    """
    Run Verus verification on a file.

    Args:
        file_path: Path to the .rs file to verify
        verus_cmd: Path to verus binary
        verus_args: Additional verus arguments
        rustc_args: Additional rustc arguments (e.g., ["--crate-type=lib"])
        timeout_sec: Timeout in seconds
        cwd: Working directory for the command
        env: Environment variables (merged with current env)

    Returns:
        VerifyResult with all verification info
    """
    file_path = Path(file_path)
    if cwd is None:
        cwd = file_path.parent
    cwd = Path(cwd)

    cmd = build_verus_command(
        file_path=file_path,
        verus_cmd=verus_cmd,
        verus_args=verus_args,
        rustc_args=rustc_args,
    )

    # Prepare environment
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    # Ensure deterministic output
    run_env["NO_COLOR"] = "1"
    run_env["TERM"] = "dumb"

    command_str = shlex.join(cmd)
    logger.debug(f"Running: {command_str}")

    start_time = time.monotonic()
    timed_out = False

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(cwd),
            env=run_env,
        )
        exit_code = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        timed_out = True
        exit_code = -1
        stdout = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        stderr += "\n[TIMEOUT]"
    except Exception as e:
        exit_code = -1
        stdout = ""
        stderr = f"Failed to run verus: {e}"

    wall_time = time.monotonic() - start_time

    # Truncate outputs
    stdout, stdout_trunc = truncate_output(stdout)
    stderr, stderr_trunc = truncate_output(stderr)

    # Determine success
    success = exit_code == 0 and not timed_out

    # Compute fingerprint
    fingerprint = "PASS" if success else compute_fingerprint(stderr)

    # Parse output
    parsed = parse_verification_output(stdout, stderr)

    return VerifyResult(
        success=success,
        exit_code=exit_code,
        timed_out=timed_out,
        wall_time_sec=wall_time,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_trunc,
        stderr_truncated=stderr_trunc,
        fingerprint=fingerprint,
        verified_count=parsed["verified_count"],
        error_count=parsed["error_count"],
        error_messages=parsed["error_messages"],
        error_type=parsed["error_type"],
        command=command_str,
        cwd=str(cwd),
    )


def check_verus_available(verus_cmd: str = DEFAULT_VERUS_CMD) -> bool:
    """Check if Verus is available and working."""
    try:
        result = subprocess.run(
            [verus_cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def run_verify_twice(
    file_path: Path | str,
    **kwargs,
) -> tuple[VerifyResult, VerifyResult, bool]:
    """
    Run verification twice to check for flakiness.

    Returns:
        (result1, result2, is_stable) where is_stable indicates both runs
        had the same success status and fingerprint.
    """
    r1 = run_verify(file_path, **kwargs)
    r2 = run_verify(file_path, **kwargs)

    is_stable = (
        r1.success == r2.success
        and r1.fingerprint == r2.fingerprint
        and not r1.timed_out
        and not r2.timed_out
    )

    return r1, r2, is_stable


def verify_code_string(
    code: str,
    work_dir: Path,
    filename: str = "lib.rs",
    **kwargs,
) -> VerifyResult:
    """
    Verify a code string by writing to a temp file.

    Args:
        code: The Verus code to verify
        work_dir: Directory to write temp file in
        filename: Name for the temp file
        **kwargs: Additional arguments for run_verify

    Returns:
        VerifyResult
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    file_path = work_dir / filename
    file_path.write_text(code, encoding="utf-8")

    return run_verify(file_path, **kwargs)
