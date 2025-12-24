"""
verus_dataset/scan.py - Scan repositories for Verus-compatible Rust files.

Implements heuristics to identify files that:
1. Contain Verus verification constructs
2. Are likely standalone (dependency-free)
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Iterator, NamedTuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Verus detection patterns and scoring
# ─────────────────────────────────────────────────────────────────────────────

# Strong signals that a file contains Verus code
VERUS_STRONG_PATTERNS = [
    (r"\bverus!\s*\{", 8),  # verus! { ... } macro
    (r"\buse\s+vstd::prelude::\*", 5),  # vstd prelude import
    (r"\buse\s+vstd::", 3),  # any vstd import
    (r"\bspec\s+fn\b", 4),  # spec function
    (r"\bproof\s+fn\b", 4),  # proof function
    (r"\bbroadcast\s+proof\b", 3),  # broadcast proof
    (r"\bbroadcast\s+group\b", 3),  # broadcast group
]

# Moderate signals
VERUS_MODERATE_PATTERNS = [
    (r"\brequires\b", 2),  # precondition
    (r"\bensures\b", 2),  # postcondition
    (r"\binvariant\b", 1),  # loop invariant
    (r"\brecommends\b", 1),  # recommends clause
    (r"\bdecreases\b", 1),  # termination measure
    (r"\bassert\s*\(", 1),  # Verus assert (also Rust, but still useful)
    (r"\bproof\s*\{", 2),  # proof block inside exec
    (r"\bopen\s+spec\b", 2),  # open spec function
    (r"\bclosed\s+spec\b", 2),  # closed spec function
    (r"\btracked\b", 2),  # tracked argument
    (r"\bGhost\b", 1),  # Ghost type
    (r"\bTracked\b", 1),  # Tracked type
]

# Weak signals (common but not unique to Verus)
VERUS_WEAK_PATTERNS = [
    (r"#\[verifier::", 2),  # verifier attribute
    (r"#\[trigger\]", 1),  # trigger attribute
    (r"\bforall\b", 1),  # quantifier (also used in other contexts)
    (r"\bexists\b", 1),  # quantifier
    (r"===>", 1),  # implication operator
    (r"<===", 1),  # reverse implication
    (r"<==>", 1),  # equivalence
]

# Compile all patterns
VERUS_PATTERNS = [
    (re.compile(pat, re.MULTILINE), score)
    for pat, score in VERUS_STRONG_PATTERNS + VERUS_MODERATE_PATTERNS + VERUS_WEAK_PATTERNS
]

# Minimum score to consider a file as Verus code
DEFAULT_MIN_SCORE = 5

# ─────────────────────────────────────────────────────────────────────────────
# Dependency detection patterns
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that indicate external dependencies (not standalone)
DEPENDENCY_PATTERNS = [
    re.compile(r"\bcrate::", re.MULTILINE),  # crate-relative imports
    re.compile(r"\bsuper::", re.MULTILINE),  # parent module imports
    re.compile(r"mod\s+\w+\s*;", re.MULTILINE),  # external module declaration
]

# Allowed imports that don't break standalone-ness
ALLOWED_IMPORTS = [
    re.compile(r"use\s+vstd::", re.MULTILINE),  # vstd is always available
    re.compile(r"use\s+builtin::", re.MULTILINE),  # builtin is always available
    re.compile(r"use\s+std::", re.MULTILINE),  # std is always available
    re.compile(r"use\s+core::", re.MULTILINE),  # core is always available
]

# Imports that indicate external dependencies
EXTERNAL_IMPORT_PATTERN = re.compile(
    r"use\s+(\w+)::",
    re.MULTILINE,
)

# Crates that are always allowed
ALLOWED_CRATES = {"vstd", "builtin", "std", "core", "alloc"}

# ─────────────────────────────────────────────────────────────────────────────
# Directories and files to skip
# ─────────────────────────────────────────────────────────────────────────────

SKIP_DIRS = {
    "target",
    ".git",
    "node_modules",
    ".cargo",
    "vendor",
    "__pycache__",
    ".pytest_cache",
    "build",
    "dist",
}

SKIP_FILE_PATTERNS = [
    re.compile(r"\.generated\.rs$"),
    re.compile(r"_generated\.rs$"),
    re.compile(r"\.pb\.rs$"),  # protobuf generated
]


# ─────────────────────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────────────────────


class FileInfo(NamedTuple):
    """Information about a scanned file."""

    rel_path: str
    abs_path: str
    sha256: str
    nbytes: int
    nlines: int
    verus_score: int
    dependency_free: bool


def score_verus_file(content: str) -> int:
    """
    Score a file's content for Verus-ness.

    Returns a score >= 0, where higher scores indicate more Verus constructs.
    A score of 0 means no Verus patterns detected.
    """
    total_score = 0
    for pattern, score in VERUS_PATTERNS:
        matches = pattern.findall(content)
        # Cap each pattern's contribution to avoid over-counting
        contribution = min(len(matches), 3) * score
        total_score += contribution
    return total_score


def is_dependency_free(content: str) -> bool:
    """
    Check if a file is likely standalone (no external dependencies).

    Returns True if the file doesn't appear to depend on:
    - crate:: imports
    - super:: imports
    - External mod declarations
    - Non-standard external crates
    """
    # Check for obvious dependency patterns
    for pattern in DEPENDENCY_PATTERNS:
        if pattern.search(content):
            return False

    # Check for external crate imports
    for match in EXTERNAL_IMPORT_PATTERN.finditer(content):
        crate_name = match.group(1)
        if crate_name not in ALLOWED_CRATES:
            # Check if it might be a local module (self::)
            if crate_name != "self":
                return False

    return True


def compute_sha256(content: str) -> str:
    """Compute SHA256 hash of file content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def should_skip_dir(path: Path) -> bool:
    """Check if a directory should be skipped."""
    return path.name in SKIP_DIRS or path.name.startswith(".")


def should_skip_file(path: Path) -> bool:
    """Check if a file should be skipped."""
    name = path.name
    for pattern in SKIP_FILE_PATTERNS:
        if pattern.search(name):
            return True
    return False


def find_rust_files(repo_path: Path) -> Iterator[Path]:
    """
    Find all .rs files in a repository, skipping excluded directories.

    Yields absolute paths to .rs files.
    """
    if not repo_path.is_dir():
        logger.warning(f"Not a directory: {repo_path}")
        return

    for path in repo_path.rglob("*.rs"):
        # Check if any parent directory should be skipped
        skip = False
        for parent in path.relative_to(repo_path).parents:
            if should_skip_dir(repo_path / parent):
                skip = True
                break

        if skip:
            continue

        if should_skip_file(path):
            continue

        yield path


def scan_file(file_path: Path, repo_path: Path) -> FileInfo | None:
    """
    Scan a single file and compute its metrics.

    Returns FileInfo or None if the file cannot be read.
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return None

    rel_path = str(file_path.relative_to(repo_path))
    abs_path = str(file_path.absolute())
    sha256 = compute_sha256(content)
    nbytes = len(content.encode("utf-8"))
    nlines = len(content.splitlines())
    verus_score = score_verus_file(content)
    dependency_free = is_dependency_free(content)

    return FileInfo(
        rel_path=rel_path,
        abs_path=abs_path,
        sha256=sha256,
        nbytes=nbytes,
        nlines=nlines,
        verus_score=verus_score,
        dependency_free=dependency_free,
    )


def scan_repo(
    repo_path: Path,
    min_score: int = DEFAULT_MIN_SCORE,
    dependency_free_only: bool = False,
    limit: int | None = None,
) -> list[FileInfo]:
    """
    Scan a repository for Verus-compatible files.

    Args:
        repo_path: Path to the repository
        min_score: Minimum Verus score to include (default: 5)
        dependency_free_only: Only include standalone files
        limit: Maximum number of files to return

    Returns:
        List of FileInfo objects, sorted by verus_score descending
    """
    results: list[FileInfo] = []

    for file_path in find_rust_files(repo_path):
        info = scan_file(file_path, repo_path)
        if info is None:
            continue

        # Apply filters
        if info.verus_score < min_score:
            continue

        if dependency_free_only and not info.dependency_free:
            continue

        results.append(info)

    # Sort by score descending
    results.sort(key=lambda x: x.verus_score, reverse=True)

    # Apply limit
    if limit is not None:
        results = results[:limit]

    logger.info(
        f"Scanned {repo_path}: found {len(results)} files "
        f"(min_score={min_score}, dependency_free_only={dependency_free_only})"
    )

    return results


def scan_repo_to_db(
    db: "Database",
    repo_name: str,
    min_score: int = DEFAULT_MIN_SCORE,
    dependency_free_only: bool = False,
    limit: int | None = None,
) -> int:
    """
    Scan a repository and store results in the database.

    Args:
        db: Database instance
        repo_name: Name of the repo (must already be registered)
        min_score: Minimum Verus score
        dependency_free_only: Only include standalone files
        limit: Maximum files to store

    Returns:
        Number of files added
    """
    from .db import Database

    repo = db.get_repo(repo_name)
    if repo is None:
        raise ValueError(f"Repo not found: {repo_name}")

    repo_path = Path(repo["local_path"])
    if not repo_path.is_dir():
        raise ValueError(f"Repo path is not a directory: {repo_path}")

    files = scan_repo(
        repo_path,
        min_score=min_score,
        dependency_free_only=dependency_free_only,
        limit=limit,
    )

    count = 0
    for info in files:
        db.add_source_file(
            repo_id=repo["repo_id"],
            rel_path=info.rel_path,
            abs_path=info.abs_path,
            sha256=info.sha256,
            nbytes=info.nbytes,
            nlines=info.nlines,
            verus_score=info.verus_score,
            dependency_free=info.dependency_free,
        )
        count += 1

    logger.info(f"Added {count} source files from {repo_name} to database")
    return count
