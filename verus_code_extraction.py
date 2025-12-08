"""
Minimal, scriptable scaffold for extracting dependency-free Verus/Rust snippets
from larger repositories. The goal is to keep extraction logic deterministic and
traceable; adapt the functions to your environment and toolchain paths.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Heuristic tokens that signal Verus/verification-oriented code
VERUS_TOKENS = {
    "verus!",
    "#[verus::",
    "requires",
    "ensures",
    "decreases",
    "invariant",
    "ghost",
    "proof",
    "spec",
    "exec",
    "reveal",
    "opens_invariants",
}


@dataclass
class ExtractionResult:
    source_path: Path
    status: str
    message: str
    code: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    verify_time_ms: Optional[int] = None

    def to_json(self) -> str:
        payload = {
            "source_path": str(self.source_path),
            "status": self.status,
            "message": self.message,
            "code": self.code,
            "dependencies": self.dependencies,
            "verify_time_ms": self.verify_time_ms,
        }
        return json.dumps(payload, ensure_ascii=False)


def find_rust_files(repo_root: Path) -> List[Path]:
    ignore_dirs = {"target", "tests", "examples", "benches", "docs", "vendor", ".git"}
    rust_files = []
    for path in repo_root.rglob("*.rs"):
        if any(part in ignore_dirs for part in path.parts):
            continue
        rust_files.append(path)
    return rust_files


def contains_verus_tokens(text: str) -> bool:
    return any(token in text for token in VERUS_TOKENS)


def score_verus_tokens(text: str) -> int:
    return sum(text.count(token) for token in VERUS_TOKENS)


def load_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_local_dependencies(text: str) -> List[str]:
    pattern = re.compile(r"^use\s+([\w:]+)", re.MULTILINE)
    deps = []
    for match in pattern.finditer(text):
        deps.append(match.group(1))
    return deps


def build_temp_crate(snippet: str) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="verus_extract_"))
    src_dir = temp_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    lib_rs = src_dir / "lib.rs"
    if "verus!" not in snippet:
        snippet = f"verus! {{\n{snippet}\n}}\n"
    lib_rs.write_text(snippet, encoding="utf-8")

    cargo_toml = temp_dir / "Cargo.toml"
    cargo_toml.write_text(
        "\n".join(
            [
                "[package]",
                "name = \"verus_extract\"",
                "version = \"0.1.0\"",
                "edition = \"2021\"",
                "",
                "[dependencies]",
                "verus = \"*\"",
            ]
        ),
        encoding="utf-8",
    )
    return temp_dir


def verify_snippet(temp_crate: Path, timeout: int = 30) -> subprocess.CompletedProcess:
    cmd = ["verus", "--verify", "--crate-type=lib", str(temp_crate / "src" / "lib.rs")]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def attempt_extract(path: Path) -> ExtractionResult:
    text = load_file(path)
    if not contains_verus_tokens(text):
        return ExtractionResult(path, "skipped", "no_verus_tokens", code=text)

    score = score_verus_tokens(text)
    deps = extract_local_dependencies(text)

    temp_crate = build_temp_crate(text)
    try:
        start_time = time.perf_counter()
        proc = verify_snippet(temp_crate)
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        
        if proc.returncode == 0:
            return ExtractionResult(
                path,
                "verified",
                f"verified with score={score}",
                code=text,
                dependencies=deps,
                verify_time_ms=elapsed_ms,
            )
        return ExtractionResult(
            path,
            "failed",
            proc.stderr.strip() or proc.stdout.strip(),
            code=text,
            dependencies=deps,
            verify_time_ms=elapsed_ms,
        )
    except subprocess.TimeoutExpired:
        return ExtractionResult(path, "timeout", "verification_timeout", code=text, dependencies=deps)
    finally:
        shutil.rmtree(temp_crate, ignore_errors=True)


def main(repo: Path, out_dir: Path, limit: Optional[int] = None) -> None:
    rust_files = find_rust_files(repo)
    rust_files.sort()
    results: List[ExtractionResult] = []

    for idx, path in enumerate(rust_files):
        if limit is not None and idx >= limit:
            break
        result = attempt_extract(path)
        results.append(result)
        print(result.to_json())

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(item.to_json() + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract dependency-free Verus snippets from a repository")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Path to the repository root")
    parser.add_argument("--out", type=Path, default=Path("./extracted_snippets"), help="Output directory for manifest")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of files to process")
    args = parser.parse_args()

    main(args.repo, args.out, args.limit)
