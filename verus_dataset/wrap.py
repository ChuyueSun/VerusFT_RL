"""
verus_dataset/wrap.py - Wrap Verus code into standalone crate-root format.

Provides utilities to prepare code for verification and reduction.
"""

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Standard imports that should be present for standalone Verus code
# ─────────────────────────────────────────────────────────────────────────────

VSTD_PRELUDE = "use vstd::prelude::*;"

# Standard Verus boilerplate
VERUS_BOILERPLATE = """\
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]

use vstd::prelude::*;

"""

# Verus macro wrapper
VERUS_MACRO_START = "verus! {\n"
VERUS_MACRO_END = "\n} // verus!\n"


# ─────────────────────────────────────────────────────────────────────────────
# Wrapping utilities
# ─────────────────────────────────────────────────────────────────────────────


def detect_verus_macro(content: str) -> bool:
    """Check if content is already wrapped in verus! { ... }."""
    # Look for verus! { at the top level
    return bool(re.search(r"^\s*verus!\s*\{", content, re.MULTILINE))


def has_vstd_import(content: str) -> bool:
    """Check if content imports vstd."""
    return bool(re.search(r"use\s+vstd::", content, re.MULTILINE))


def extract_from_verus_macro(content: str) -> str:
    """
    Extract content from inside verus! { ... } macro.

    If not wrapped, returns original content.
    """
    # Simple regex extraction - works for well-formed code
    match = re.search(r"verus!\s*\{(.*)\}\s*(?://.*)?$", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content


def wrap_in_verus_macro(content: str) -> str:
    """Wrap content in verus! { ... } if not already wrapped."""
    if detect_verus_macro(content):
        return content
    return VERUS_MACRO_START + content + VERUS_MACRO_END


def ensure_vstd_import(content: str) -> str:
    """Ensure content has vstd import."""
    if has_vstd_import(content):
        return content
    return VSTD_PRELUDE + "\n" + content


def wrap_as_crate_root(
    content: str,
    add_boilerplate: bool = True,
    ensure_verus_macro: bool = True,
) -> str:
    """
    Wrap code as a standalone crate root (lib.rs style).

    This prepares code for verification with:
    - Standard allows for unused code
    - vstd import
    - verus! macro wrapper (optional)

    Args:
        content: The Verus code to wrap
        add_boilerplate: Add standard boilerplate (allows, vstd import)
        ensure_verus_macro: Wrap in verus! { } if not already

    Returns:
        Wrapped code ready for verification
    """
    result = content.strip()

    # If content is already in a verus! macro, we need to handle it carefully
    if detect_verus_macro(result):
        # Already has verus macro - just ensure vstd import is present
        if add_boilerplate and not has_vstd_import(result):
            # Add vstd import before the verus! macro
            lines = result.split("\n")
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("verus!"):
                    insert_idx = i
                    break
            lines.insert(insert_idx, VSTD_PRELUDE)
            result = "\n".join(lines)

        if add_boilerplate:
            # Add allow attributes at the top
            allows = "#![allow(unused_imports)]\n#![allow(dead_code)]\n#![allow(unused_variables)]\n\n"
            if not result.startswith("#![allow"):
                result = allows + result
    else:
        # No verus macro - need full wrapping
        if add_boilerplate:
            result = VERUS_BOILERPLATE + result
        elif not has_vstd_import(result):
            result = ensure_vstd_import(result)

        if ensure_verus_macro:
            result = wrap_in_verus_macro(result)

    return result


def write_crate_files(
    output_dir: Path,
    code: str,
    verus_root: Optional[Path] = None,
) -> Path:
    """
    Write a minimal Verus crate structure to output_dir.

    Creates:
    - Cargo.toml with vstd/builtin dependencies
    - src/lib.rs with the wrapped code

    Args:
        output_dir: Directory to create crate in
        code: The wrapped Verus code
        verus_root: Path to Verus source (for builtin/vstd deps)

    Returns:
        Path to the created crate directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    src_dir = output_dir / "src"
    src_dir.mkdir(exist_ok=True)

    # Write lib.rs
    lib_rs = src_dir / "lib.rs"
    lib_rs.write_text(code, encoding="utf-8")

    # Write Cargo.toml
    if verus_root:
        builtin_path = verus_root / "source" / "builtin"
        vstd_path = verus_root / "source" / "vstd"
        cargo_toml = f"""\
[package]
name = "verus_reduced"
version = "0.1.0"
edition = "2021"

[dependencies]
builtin = {{ path = "{builtin_path}" }}
vstd = {{ path = "{vstd_path}" }}
"""
    else:
        # Assume verus is installed globally and can find its deps
        cargo_toml = """\
[package]
name = "verus_reduced"
version = "0.1.0"
edition = "2021"

# Note: verus should be invoked with --crate-type=lib
# vstd will be provided by the verus toolchain
"""

    cargo_toml_path = output_dir / "Cargo.toml"
    cargo_toml_path.write_text(cargo_toml, encoding="utf-8")

    logger.debug(f"Created crate at {output_dir}")
    return output_dir


def prepare_for_reduction(
    content: str,
    output_path: Path,
) -> str:
    """
    Prepare code for C-Reduce reduction.

    The file should be self-contained so C-Reduce can operate on it.
    We use a simpler format than full crate wrapping.

    Args:
        content: Original Verus code
        output_path: Where to write the prepared file

    Returns:
        The wrapped content that was written
    """
    wrapped = wrap_as_crate_root(
        content,
        add_boilerplate=True,
        ensure_verus_macro=True,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(wrapped, encoding="utf-8")

    return wrapped


def unwrap_for_display(content: str) -> str:
    """
    Remove boilerplate from wrapped code for cleaner display.

    Useful for generating training examples where we want minimal cruft.
    """
    lines = content.strip().split("\n")
    result_lines = []

    # Skip allow attributes at the start
    skip_allows = True
    in_verus_macro = False
    verus_depth = 0

    for line in lines:
        stripped = line.strip()

        # Skip initial allow attributes
        if skip_allows:
            if stripped.startswith("#![allow") or stripped == "":
                continue
            skip_allows = False

        # Track verus! macro for potential extraction
        if stripped.startswith("verus!"):
            in_verus_macro = True
            verus_depth = 1
            continue

        if in_verus_macro:
            verus_depth += line.count("{") - line.count("}")
            if verus_depth <= 0 and stripped.startswith("}"):
                in_verus_macro = False
                continue

        result_lines.append(line)

    # If we're still in verus macro, we extracted the content
    # Otherwise return original without allows
    return "\n".join(result_lines).strip()
