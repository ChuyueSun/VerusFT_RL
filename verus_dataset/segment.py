"""
verus_dataset/segment.py - Segment Verus code into exec/spec/proof zones.

Implements a lightweight parser that identifies:
- spec fn: specification functions (ghost, not compiled)
- proof fn: proof functions (ghost, not compiled)
- fn (exec): executable functions
- proof { } blocks inside exec functions
- requires/ensures clauses
"""

import json
import re
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Iterator


class SegmentKind(str, Enum):
    """Types of code segments."""
    EXEC = "exec"
    SPEC = "spec"
    PROOF = "proof"
    REQUIRES = "requires"
    ENSURES = "ensures"
    INVARIANT = "invariant"
    DECREASES = "decreases"
    UNKNOWN = "unknown"


@dataclass
class Segment:
    """A segment of code with its type annotation."""
    kind: SegmentKind
    name: str  # function/block name if applicable
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed, inclusive
    content: str  # the actual code

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "kind": self.kind.value,
            "name": self.name,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


@dataclass
class FunctionInfo:
    """Information about a function in Verus code."""
    name: str
    kind: SegmentKind  # exec, spec, or proof
    start_line: int
    end_line: int
    signature_end_line: int  # where the signature ends (before body)
    has_requires: bool
    has_ensures: bool
    requires_lines: list[tuple[int, int]]  # (start, end) pairs
    ensures_lines: list[tuple[int, int]]
    proof_blocks: list[tuple[int, int]]  # proof { } blocks inside


# ─────────────────────────────────────────────────────────────────────────────
# Patterns for Verus constructs
# ─────────────────────────────────────────────────────────────────────────────

# Function declaration patterns
SPEC_FN_PATTERN = re.compile(
    r"^\s*(?:pub\s+)?(?:open\s+|closed\s+)?spec\s+fn\s+(\w+)",
    re.MULTILINE,
)

PROOF_FN_PATTERN = re.compile(
    r"^\s*(?:pub\s+)?(?:broadcast\s+)?proof\s+fn\s+(\w+)",
    re.MULTILINE,
)

EXEC_FN_PATTERN = re.compile(
    r"^\s*(?:pub\s+)?(?:const\s+)?fn\s+(\w+)",
    re.MULTILINE,
)

# Clause patterns
REQUIRES_PATTERN = re.compile(r"^\s*requires\b", re.MULTILINE)
ENSURES_PATTERN = re.compile(r"^\s*ensures\b", re.MULTILINE)
INVARIANT_PATTERN = re.compile(r"^\s*invariant\b", re.MULTILINE)
DECREASES_PATTERN = re.compile(r"^\s*decreases\b", re.MULTILINE)

# Proof block pattern
PROOF_BLOCK_PATTERN = re.compile(r"^\s*proof\s*\{", re.MULTILINE)


# ─────────────────────────────────────────────────────────────────────────────
# Brace matching
# ─────────────────────────────────────────────────────────────────────────────


def find_matching_brace(lines: list[str], start_line: int, start_col: int = 0) -> int:
    """
    Find the line number where a brace block ends.

    Args:
        lines: List of code lines (0-indexed)
        start_line: Line where the opening brace is (0-indexed)
        start_col: Column to start searching from

    Returns:
        Line number (0-indexed) where the closing brace is, or -1 if not found
    """
    depth = 0
    in_string = False
    string_char = None
    in_comment = False
    in_block_comment = False

    for line_idx in range(start_line, len(lines)):
        line = lines[line_idx]
        col_start = start_col if line_idx == start_line else 0

        i = col_start
        while i < len(line):
            # Handle block comments
            if in_block_comment:
                if i + 1 < len(line) and line[i:i+2] == "*/":
                    in_block_comment = False
                    i += 2
                    continue
                i += 1
                continue

            # Check for comment start
            if not in_string and i + 1 < len(line):
                if line[i:i+2] == "//":
                    break  # Rest of line is comment
                if line[i:i+2] == "/*":
                    in_block_comment = True
                    i += 2
                    continue

            # Handle strings
            if i >= len(line):
                break
            ch = line[i]
            if not in_string and ch in ('"', "'"):
                in_string = True
                string_char = line[i]
                i += 1
                continue

            if in_string:
                if i >= len(line):
                    break
                if line[i] == "\\" and i + 1 < len(line):
                    i += 2  # Skip escaped char
                    continue
                if line[i] == string_char:
                    in_string = False
                i += 1
                continue

            # Count braces
            if i >= len(line):
                break
            if line[i] == "{":
                depth += 1
            elif line[i] == "}":
                depth -= 1
                if depth == 0:
                    return line_idx

            i += 1

    return -1


def find_function_end(lines: list[str], start_line: int) -> int:
    """
    Find where a function ends.

    Args:
        lines: Code lines (0-indexed)
        start_line: Line where function starts (0-indexed)

    Returns:
        End line (0-indexed) or -1 if not found
    """
    # Find the opening brace
    for i in range(start_line, min(start_line + 50, len(lines))):
        line = lines[i]
        brace_pos = line.find("{")
        if brace_pos != -1:
            return find_matching_brace(lines, i, brace_pos)
    return -1


# ─────────────────────────────────────────────────────────────────────────────
# Main segmentation
# ─────────────────────────────────────────────────────────────────────────────


def find_functions(content: str) -> list[FunctionInfo]:
    """
    Find all functions in the code and their boundaries.

    Returns list of FunctionInfo objects.
    """
    lines = content.split("\n")
    functions: list[FunctionInfo] = []

    # Track which lines are already claimed
    claimed_lines = set()

    # Find spec functions
    for match in SPEC_FN_PATTERN.finditer(content):
        name = match.group(1)
        start_pos = match.start()
        start_line = content[:start_pos].count("\n")

        if start_line in claimed_lines:
            continue

        end_line = find_function_end(lines, start_line)
        if end_line == -1:
            end_line = min(start_line + 20, len(lines) - 1)

        for i in range(start_line, end_line + 1):
            claimed_lines.add(i)

        functions.append(FunctionInfo(
            name=name,
            kind=SegmentKind.SPEC,
            start_line=start_line + 1,  # 1-indexed
            end_line=end_line + 1,
            signature_end_line=start_line + 1,
            has_requires=False,
            has_ensures=False,
            requires_lines=[],
            ensures_lines=[],
            proof_blocks=[],
        ))

    # Find proof functions
    for match in PROOF_FN_PATTERN.finditer(content):
        name = match.group(1)
        start_pos = match.start()
        start_line = content[:start_pos].count("\n")

        if start_line in claimed_lines:
            continue

        end_line = find_function_end(lines, start_line)
        if end_line == -1:
            end_line = min(start_line + 20, len(lines) - 1)

        for i in range(start_line, end_line + 1):
            claimed_lines.add(i)

        functions.append(FunctionInfo(
            name=name,
            kind=SegmentKind.PROOF,
            start_line=start_line + 1,
            end_line=end_line + 1,
            signature_end_line=start_line + 1,
            has_requires=False,
            has_ensures=False,
            requires_lines=[],
            ensures_lines=[],
            proof_blocks=[],
        ))

    # Find exec functions (regular fn that aren't spec/proof)
    for match in EXEC_FN_PATTERN.finditer(content):
        name = match.group(1)
        start_pos = match.start()
        start_line = content[:start_pos].count("\n")

        if start_line in claimed_lines:
            continue

        # Check if this is actually a spec or proof fn
        line_start = content.rfind("\n", 0, start_pos) + 1
        prefix = content[line_start:start_pos]
        if "spec" in prefix or "proof" in prefix:
            continue

        end_line = find_function_end(lines, start_line)
        if end_line == -1:
            end_line = min(start_line + 20, len(lines) - 1)

        for i in range(start_line, end_line + 1):
            claimed_lines.add(i)

        # Check for requires/ensures
        func_text = "\n".join(lines[start_line:end_line + 1])
        has_requires = bool(REQUIRES_PATTERN.search(func_text))
        has_ensures = bool(ENSURES_PATTERN.search(func_text))

        # Find proof blocks
        proof_blocks = []
        for pb_match in PROOF_BLOCK_PATTERN.finditer(func_text):
            pb_start = func_text[:pb_match.start()].count("\n")
            pb_start_line = start_line + pb_start
            pb_end_line = find_matching_brace(
                lines, pb_start_line,
                lines[pb_start_line].find("{")
            )
            if pb_end_line != -1:
                proof_blocks.append((pb_start_line + 1, pb_end_line + 1))

        functions.append(FunctionInfo(
            name=name,
            kind=SegmentKind.EXEC,
            start_line=start_line + 1,
            end_line=end_line + 1,
            signature_end_line=start_line + 1,
            has_requires=has_requires,
            has_ensures=has_ensures,
            requires_lines=[],
            ensures_lines=[],
            proof_blocks=proof_blocks,
        ))

    # Sort by start line
    functions.sort(key=lambda f: f.start_line)
    return functions


def segment_code(content: str) -> list[Segment]:
    """
    Segment Verus code into exec/spec/proof zones.

    Returns a list of Segment objects covering the code.
    """
    lines = content.split("\n")
    functions = find_functions(content)
    segments: list[Segment] = []

    for func in functions:
        func_content = "\n".join(lines[func.start_line - 1:func.end_line])

        segments.append(Segment(
            kind=func.kind,
            name=func.name,
            start_line=func.start_line,
            end_line=func.end_line,
            content=func_content,
        ))

        # Add proof block sub-segments for exec functions
        if func.kind == SegmentKind.EXEC:
            for pb_start, pb_end in func.proof_blocks:
                pb_content = "\n".join(lines[pb_start - 1:pb_end])
                segments.append(Segment(
                    kind=SegmentKind.PROOF,
                    name=f"{func.name}::proof_block",
                    start_line=pb_start,
                    end_line=pb_end,
                    content=pb_content,
                ))

    # Sort by start line
    segments.sort(key=lambda s: (s.start_line, -len(s.content)))
    return segments


def segments_to_json(segments: list[Segment]) -> str:
    """Convert segments to JSON string for storage."""
    return json.dumps([s.to_dict() for s in segments])


def segments_from_json(json_str: str) -> list[Segment]:
    """Parse segments from JSON string."""
    data = json.loads(json_str)
    segments = []
    for item in data:
        segments.append(Segment(
            kind=SegmentKind(item["kind"]),
            name=item["name"],
            start_line=item["start_line"],
            end_line=item["end_line"],
            content="",  # Content not stored in JSON
        ))
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Utilities for task generation
# ─────────────────────────────────────────────────────────────────────────────


def get_exec_functions(content: str) -> list[FunctionInfo]:
    """Get only exec functions from the code."""
    functions = find_functions(content)
    return [f for f in functions if f.kind == SegmentKind.EXEC]


def get_exec_functions_with_specs(content: str) -> list[FunctionInfo]:
    """Get exec functions that have requires/ensures clauses."""
    functions = get_exec_functions(content)
    return [f for f in functions if f.has_requires or f.has_ensures]


def extract_function_body(content: str, func: FunctionInfo) -> str:
    """Extract just the body of a function (without signature)."""
    lines = content.split("\n")

    # Find the opening brace
    for i in range(func.start_line - 1, func.end_line):
        line = lines[i]
        brace_pos = line.find("{")
        if brace_pos != -1:
            # Body starts after the brace
            body_start = i
            body_lines = lines[body_start:func.end_line]
            # Remove first line's prefix up to and including {
            if body_lines:
                body_lines[0] = body_lines[0][brace_pos + 1:]
            # Remove last line's } and after
            if body_lines:
                last_brace = body_lines[-1].rfind("}")
                if last_brace != -1:
                    body_lines[-1] = body_lines[-1][:last_brace]
            return "\n".join(body_lines).strip()

    return ""


def extract_function_signature(content: str, func: FunctionInfo) -> str:
    """Extract just the signature of a function (up to the body)."""
    lines = content.split("\n")

    sig_lines = []
    for i in range(func.start_line - 1, func.end_line):
        line = lines[i]
        brace_pos = line.find("{")
        if brace_pos != -1:
            # Include up to but not including the brace
            sig_lines.append(line[:brace_pos].rstrip())
            break
        sig_lines.append(line)

    return "\n".join(sig_lines)


def remove_specs_from_function(content: str, func: FunctionInfo) -> str:
    """
    Remove requires/ensures clauses from a function.

    Returns the function with specs removed, keeping signature and body.
    """
    lines = content.split("\n")
    func_lines = lines[func.start_line - 1:func.end_line]

    # Find the signature end and body start
    # Strategy: Find where specs begin (requires/ensures/etc.) and where the body brace is

    signature_lines = []
    body_start_idx = -1
    brace_col = -1
    in_specs = False

    for i, line in enumerate(func_lines):
        stripped = line.strip()

        # Check if this line contains the opening brace of the function body
        # We need to find '{' that's NOT inside a comment or string
        brace_pos = -1
        in_string = False
        string_char = None
        for j, ch in enumerate(line):
            if not in_string and ch in ('"', "'"):
                in_string = True
                string_char = ch
            elif in_string and ch == string_char:
                # Check for escape
                if j > 0 and line[j-1] != '\\':
                    in_string = False
            elif not in_string and ch == '{':
                brace_pos = j
                break
            elif not in_string and j < len(line) - 1 and line[j:j+2] == '//':
                break  # Rest is comment

        # If this is a spec clause line
        if (stripped.startswith("requires") or
            stripped.startswith("ensures") or
            stripped.startswith("recommends") or
            stripped.startswith("decreases")):
            in_specs = True
            # Check if brace is on this line (rare but possible)
            if brace_pos != -1:
                body_start_idx = i
                brace_col = brace_pos
                break
            continue

        # If we're in specs (skipping spec content)
        if in_specs:
            # Check if we've reached the opening brace
            if brace_pos != -1:
                body_start_idx = i
                brace_col = brace_pos
                break
            # Otherwise, skip this spec content line
            continue

        # Before specs - this is signature
        if brace_pos != -1:
            # Opening brace before any specs - no specs to remove
            body_start_idx = i
            brace_col = brace_pos
            break

        signature_lines.append(line)

    # If no body found, return original
    if body_start_idx == -1:
        return "\n".join(func_lines)

    # If no specs were found (in_specs is False), return original function unchanged
    if not in_specs:
        return "\n".join(func_lines)

    # Build result: signature + brace + body
    result_lines = signature_lines.copy()

    # Add the line with the opening brace
    brace_line = func_lines[body_start_idx]
    # Keep only the brace and what's after it
    result_lines.append(brace_line[brace_col:])

    # Add remaining body lines
    result_lines.extend(func_lines[body_start_idx + 1:])

    return "\n".join(result_lines)


def stub_function_body(content: str, func: FunctionInfo) -> str:
    """
    Replace a function's body with a placeholder.

    Returns the full content with the function body stubbed.
    """
    lines = content.split("\n")

    # Find the opening brace
    body_start_line = -1
    body_start_col = -1
    for i in range(func.start_line - 1, func.end_line):
        line = lines[i]
        brace_pos = line.find("{")
        if brace_pos != -1:
            body_start_line = i
            body_start_col = brace_pos
            break

    if body_start_line == -1:
        return content

    # Find closing brace
    body_end_line = find_matching_brace(lines, body_start_line, body_start_col)
    if body_end_line == -1:
        return content

    # Build new content
    result_lines = lines[:body_start_line]
    result_lines.append(lines[body_start_line][:body_start_col + 1])
    result_lines.append("    unimplemented!()")
    result_lines.append("}")
    result_lines.extend(lines[body_end_line + 1:])

    return "\n".join(result_lines)
