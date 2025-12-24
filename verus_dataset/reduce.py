"""
verus_dataset/reduce.py - C-Reduce driver for minimizing Verus code.

Implements PASS-preserving and FAIL-preserving reduction using C-Reduce
with custom interestingness tests.
"""

import json
import logging
import os
import shutil
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .verify import VerifyResult, run_verify

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CREDUCE_CMD = "creduce"
DEFAULT_CREDUCE_TIMEOUT_SEC = 1800  # 30 minutes per file
DEFAULT_VERUS_TIMEOUT_SEC = 60


@dataclass
class ReductionResult:
    """Result of a C-Reduce reduction run."""

    success: bool
    timed_out: bool
    wall_time_sec: float

    # File info
    input_code: str
    output_code: str
    bytes_before: int
    bytes_after: int
    lines_before: int
    lines_after: int

    # Verification results
    pre_verify: Optional[VerifyResult]
    post_verify: Optional[VerifyResult]

    # Fingerprints
    expected_fingerprint: str
    actual_fingerprint: str

    # Logs
    creduce_stdout: str
    creduce_stderr: str
    work_dir: str

    # Error info
    error_message: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Interestingness test script generation
# ─────────────────────────────────────────────────────────────────────────────

INTERESTINGNESS_SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
"""
Interestingness test for C-Reduce.

This script is called by C-Reduce to determine if the current reduced
file is "interesting" (i.e., still exhibits the property we want to preserve).

Exit 0 = interesting (keep reducing)
Exit 1 = not interesting (revert this reduction)
"""

import hashlib
import os
import re
import subprocess
import sys

# Configuration from environment
EXPECTED_MODE = os.environ.get("EXPECTED_MODE", "pass")  # "pass" or "fail"
EXPECTED_FINGERPRINT = os.environ.get("EXPECTED_FINGERPRINT", "")
VERUS_CMD = os.environ.get("VERUS_CMD", "verus")
VERUS_ARGS = os.environ.get("VERUS_ARGS", "").split() if os.environ.get("VERUS_ARGS") else []
RUSTC_ARGS = os.environ.get("RUSTC_ARGS", "--crate-type=lib").split() if os.environ.get("RUSTC_ARGS") else ["--crate-type=lib"]
VERUS_TIMEOUT = int(os.environ.get("VERUS_TIMEOUT", "60"))

# The file being reduced (C-Reduce passes this)
INPUT_FILE = "lib.rs"


def normalize_error_output(stderr: str) -> str:
    """Normalize error output for fingerprinting."""
    result = stderr
    result = re.sub(r"\\x1b\\[[0-9;]*m", "", result)
    result = re.sub(r"\\x1b\\[[0-9;]*[a-zA-Z]", "", result)
    result = re.sub(r"[A-Za-z]?:?[/\\\\][\\w\\-./\\\\]+\\.rs", "<PATH>", result)
    result = re.sub(r":\\d+:\\d+", ":<LOC>", result)
    result = re.sub(r"line \\d+", "line <LOC>", result)
    result = re.sub(r"0x[0-9a-fA-F]+", "<ADDR>", result)
    result = re.sub(r"\\d+\\.\\d+s", "<TIME>", result)
    result = re.sub(r"\\s+", " ", result).strip()
    return result


def compute_fingerprint(stderr: str) -> str:
    """Compute fingerprint from error output."""
    if not stderr.strip():
        return "PASS"
    normalized = normalize_error_output(stderr)
    return hashlib.sha256(normalized.encode()).digest()[:8].hex()


def run_verus(file_path: str) -> tuple[int, str, str]:
    """Run Verus on the file."""
    cmd = [VERUS_CMD] + VERUS_ARGS + [file_path] + RUSTC_ARGS

    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env["TERM"] = "dumb"

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=VERUS_TIMEOUT,
            env=env,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "[TIMEOUT]"
    except Exception as e:
        return -1, "", f"[ERROR: {{e}}]"


def main():
    # Check if the file exists
    if not os.path.exists(INPUT_FILE):
        sys.exit(1)

    # Run verification
    exit_code, stdout, stderr = run_verus(INPUT_FILE)
    success = exit_code == 0

    if EXPECTED_MODE == "pass":
        # PASS-preserving: file must still verify
        if success:
            sys.exit(0)  # Interesting!
        else:
            sys.exit(1)  # Not interesting
    else:
        # FAIL-preserving: file must fail with same fingerprint
        if success:
            sys.exit(1)  # Became passing, not interesting

        fingerprint = compute_fingerprint(stderr)
        if fingerprint == EXPECTED_FINGERPRINT:
            sys.exit(0)  # Same error, interesting!
        else:
            sys.exit(1)  # Different error, not interesting


if __name__ == "__main__":
    main()
'''


def write_interestingness_script(
    output_dir: Path,
    mode: str = "pass",
    expected_fingerprint: str = "",
    verus_cmd: str = "verus",
    verus_args: list[str] | None = None,
    rustc_args: list[str] | None = None,
    verus_timeout: int = DEFAULT_VERUS_TIMEOUT_SEC,
) -> Path:
    """
    Write the interestingness test script.

    Args:
        output_dir: Directory to write script in
        mode: "pass" or "fail"
        expected_fingerprint: For fail mode, the fingerprint to preserve
        verus_cmd: Path to verus binary
        verus_args: Additional verus arguments
        rustc_args: Additional rustc arguments
        verus_timeout: Timeout for each verification run

    Returns:
        Path to the script
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = output_dir / "interestingness.py"
    script_path.write_text(INTERESTINGNESS_SCRIPT_TEMPLATE, encoding="utf-8")

    # Make executable
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

    # Write environment file for debugging
    env_path = output_dir / "interestingness.env"
    env_content = f"""\
EXPECTED_MODE={mode}
EXPECTED_FINGERPRINT={expected_fingerprint}
VERUS_CMD={verus_cmd}
VERUS_ARGS={' '.join(verus_args or [])}
RUSTC_ARGS={' '.join(rustc_args or ['--crate-type=lib'])}
VERUS_TIMEOUT={verus_timeout}
"""
    env_path.write_text(env_content, encoding="utf-8")

    return script_path


# ─────────────────────────────────────────────────────────────────────────────
# C-Reduce driver
# ─────────────────────────────────────────────────────────────────────────────


def run_creduce(
    input_code: str,
    work_dir: Path,
    mode: str = "pass",
    expected_fingerprint: str = "",
    verus_cmd: str = "verus",
    verus_args: list[str] | None = None,
    rustc_args: list[str] | None = None,
    verus_timeout: int = DEFAULT_VERUS_TIMEOUT_SEC,
    creduce_cmd: str = DEFAULT_CREDUCE_CMD,
    creduce_timeout: int = DEFAULT_CREDUCE_TIMEOUT_SEC,
    creduce_args: list[str] | None = None,
) -> ReductionResult:
    """
    Run C-Reduce to minimize Verus code.

    Args:
        input_code: The Verus code to reduce
        work_dir: Directory for reduction work
        mode: "pass" (preserve verification) or "fail" (preserve error)
        expected_fingerprint: For fail mode, the error fingerprint to preserve
        verus_cmd: Path to verus binary
        verus_args: Additional verus arguments
        rustc_args: Additional rustc arguments
        verus_timeout: Timeout for each verification run
        creduce_cmd: Path to creduce binary
        creduce_timeout: Total timeout for reduction
        creduce_args: Additional creduce arguments

    Returns:
        ReductionResult with all info
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Default rustc args
    if rustc_args is None:
        rustc_args = ["--crate-type=lib"]

    # Write input file
    input_file = work_dir / "lib.rs"
    input_file.write_text(input_code, encoding="utf-8")

    # Pre-verify to ensure starting state is valid
    logger.info("Running pre-verification...")
    pre_result = run_verify(
        input_file,
        verus_cmd=verus_cmd,
        verus_args=verus_args,
        rustc_args=rustc_args,
        timeout_sec=verus_timeout,
    )

    # Check if pre-verify matches expected mode
    if mode == "pass" and not pre_result.success:
        return ReductionResult(
            success=False,
            timed_out=False,
            wall_time_sec=pre_result.wall_time_sec,
            input_code=input_code,
            output_code=input_code,
            bytes_before=len(input_code),
            bytes_after=len(input_code),
            lines_before=len(input_code.splitlines()),
            lines_after=len(input_code.splitlines()),
            pre_verify=pre_result,
            post_verify=None,
            expected_fingerprint="PASS",
            actual_fingerprint=pre_result.fingerprint,
            creduce_stdout="",
            creduce_stderr="",
            work_dir=str(work_dir),
            error_message="Pre-verification failed: file does not verify",
        )

    if mode == "fail" and pre_result.success:
        return ReductionResult(
            success=False,
            timed_out=False,
            wall_time_sec=pre_result.wall_time_sec,
            input_code=input_code,
            output_code=input_code,
            bytes_before=len(input_code),
            bytes_after=len(input_code),
            lines_before=len(input_code.splitlines()),
            lines_after=len(input_code.splitlines()),
            pre_verify=pre_result,
            post_verify=None,
            expected_fingerprint=expected_fingerprint,
            actual_fingerprint=pre_result.fingerprint,
            creduce_stdout="",
            creduce_stderr="",
            work_dir=str(work_dir),
            error_message="Pre-verification succeeded: file verifies (expected fail)",
        )

    # For fail mode, use the actual fingerprint if not provided
    if mode == "fail" and not expected_fingerprint:
        expected_fingerprint = pre_result.fingerprint

    # Write interestingness script
    logger.info(f"Writing interestingness script (mode={mode})...")
    script_path = write_interestingness_script(
        output_dir=work_dir,
        mode=mode,
        expected_fingerprint=expected_fingerprint,
        verus_cmd=verus_cmd,
        verus_args=verus_args,
        rustc_args=rustc_args,
        verus_timeout=verus_timeout,
    )

    # Build creduce command
    cmd = [creduce_cmd]
    if creduce_args:
        cmd.extend(creduce_args)
    else:
        # Default args for non-C code
        cmd.extend(["--not-c"])

    cmd.extend([str(script_path), "lib.rs"])

    # Set up environment
    env = os.environ.copy()
    env["EXPECTED_MODE"] = mode
    env["EXPECTED_FINGERPRINT"] = expected_fingerprint
    env["VERUS_CMD"] = verus_cmd
    env["VERUS_ARGS"] = " ".join(verus_args or [])
    env["RUSTC_ARGS"] = " ".join(rustc_args)
    env["VERUS_TIMEOUT"] = str(verus_timeout)
    env["NO_COLOR"] = "1"

    # Run creduce
    logger.info(f"Running creduce with {creduce_timeout}s timeout...")
    start_time = time.monotonic()
    timed_out = False

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=creduce_timeout,
            cwd=str(work_dir),
            env=env,
        )
        creduce_stdout = proc.stdout
        creduce_stderr = proc.stderr
        creduce_exit = proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        creduce_stdout = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
        creduce_stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        creduce_stderr += "\n[CREDUCE TIMEOUT]"
        creduce_exit = -1
    except Exception as e:
        return ReductionResult(
            success=False,
            timed_out=False,
            wall_time_sec=time.monotonic() - start_time,
            input_code=input_code,
            output_code=input_code,
            bytes_before=len(input_code),
            bytes_after=len(input_code),
            lines_before=len(input_code.splitlines()),
            lines_after=len(input_code.splitlines()),
            pre_verify=pre_result,
            post_verify=None,
            expected_fingerprint=expected_fingerprint if mode == "fail" else "PASS",
            actual_fingerprint="",
            creduce_stdout="",
            creduce_stderr="",
            work_dir=str(work_dir),
            error_message=f"Failed to run creduce: {e}",
        )

    wall_time = time.monotonic() - start_time

    # Read reduced output
    output_code = input_file.read_text(encoding="utf-8")

    # Post-verify
    logger.info("Running post-verification...")
    post_result = run_verify(
        input_file,
        verus_cmd=verus_cmd,
        verus_args=verus_args,
        rustc_args=rustc_args,
        timeout_sec=verus_timeout,
    )

    # Determine success
    success = False
    error_message = ""

    if mode == "pass":
        if post_result.success:
            success = True
        else:
            error_message = "Post-verification failed"
    else:
        if not post_result.success and post_result.fingerprint == expected_fingerprint:
            success = True
        elif post_result.success:
            error_message = "Post-verification succeeded (expected fail)"
        else:
            error_message = f"Fingerprint mismatch: expected {expected_fingerprint}, got {post_result.fingerprint}"

    return ReductionResult(
        success=success,
        timed_out=timed_out,
        wall_time_sec=wall_time,
        input_code=input_code,
        output_code=output_code,
        bytes_before=len(input_code),
        bytes_after=len(output_code),
        lines_before=len(input_code.splitlines()),
        lines_after=len(output_code.splitlines()),
        pre_verify=pre_result,
        post_verify=post_result,
        expected_fingerprint=expected_fingerprint if mode == "fail" else "PASS",
        actual_fingerprint=post_result.fingerprint,
        creduce_stdout=creduce_stdout,
        creduce_stderr=creduce_stderr,
        work_dir=str(work_dir),
        error_message=error_message,
    )


def check_creduce_available(creduce_cmd: str = DEFAULT_CREDUCE_CMD) -> bool:
    """Check if C-Reduce is available."""
    try:
        result = subprocess.run(
            [creduce_cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Batch reduction helpers
# ─────────────────────────────────────────────────────────────────────────────


def create_reduction_work_dir(
    base_dir: Path,
    file_id: int,
    mode: str,
) -> Path:
    """Create a work directory for a reduction job."""
    work_dir = Path(base_dir) / f"job_{file_id}_{mode}"
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def save_reduction_logs(
    result: ReductionResult,
    output_dir: Path,
) -> Path:
    """Save reduction logs to a directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save reduced code
    (output_dir / "reduced.rs").write_text(result.output_code, encoding="utf-8")

    # Save original code
    (output_dir / "original.rs").write_text(result.input_code, encoding="utf-8")

    # Save creduce output
    (output_dir / "creduce_stdout.txt").write_text(result.creduce_stdout, encoding="utf-8")
    (output_dir / "creduce_stderr.txt").write_text(result.creduce_stderr, encoding="utf-8")

    # Save summary
    summary = {
        "success": result.success,
        "timed_out": result.timed_out,
        "wall_time_sec": result.wall_time_sec,
        "bytes_before": result.bytes_before,
        "bytes_after": result.bytes_after,
        "lines_before": result.lines_before,
        "lines_after": result.lines_after,
        "expected_fingerprint": result.expected_fingerprint,
        "actual_fingerprint": result.actual_fingerprint,
        "error_message": result.error_message,
        "reduction_ratio": result.bytes_after / result.bytes_before if result.bytes_before > 0 else 1.0,
    }

    if result.pre_verify:
        summary["pre_verify"] = {
            "success": result.pre_verify.success,
            "exit_code": result.pre_verify.exit_code,
            "fingerprint": result.pre_verify.fingerprint,
        }

    if result.post_verify:
        summary["post_verify"] = {
            "success": result.post_verify.success,
            "exit_code": result.post_verify.exit_code,
            "fingerprint": result.post_verify.fingerprint,
        }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return output_dir
