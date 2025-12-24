"""
verus_dataset/tasks.py - Generate training tasks from Verus samples.

Supports multiple task types:
- spec_gen: Add requires/ensures to an exec function
- code_synth: Generate code from a spec
- spec_and_code: Generate both spec and code
- repair: Fix a failing verification
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Optional

from .db import Database
from .segment import (
    FunctionInfo,
    SegmentKind,
    find_functions,
    get_exec_functions,
    get_exec_functions_with_specs,
    extract_function_body,
    extract_function_signature,
    remove_specs_from_function,
    stub_function_body,
)

logger = logging.getLogger(__name__)


def is_function_stub(code: str, func: FunctionInfo) -> bool:
    """Check if a function's body is a stub (assume(false) with placeholder return)."""
    lines = code.split("\n")
    func_code = "\n".join(lines[func.start_line - 1:func.end_line])

    # Must have assume(false) to be a stub
    if "assume(false)" not in func_code:
        return False

    # Extract and analyze the body
    body = extract_function_body(code, func)
    body_stripped = body.strip()

    # Get non-empty, non-comment lines
    body_lines = [l.strip() for l in body_stripped.split('\n')
                  if l.strip() and not l.strip().startswith('//')]

    # Filter out stub patterns and simple placeholders
    real_lines = []
    for line in body_lines:
        # Skip stub patterns
        if 'assume(false)' in line or 'unreached()' in line:
            continue
        # Skip simple placeholder returns (after assume(false), these are just type satisfaction)
        if line in ['0', '-1', 'false', 'true', 'None', '()', '{}', 'i']:
            continue
        if line.endswith('::new()') or line.endswith('::empty()') or line.endswith('::Empty'):
            continue
        if line.startswith('Vec::') or line.startswith('loop') or line.startswith('vec!'):
            continue
        if line.startswith('Tree::') or line.startswith('List::'):
            continue
        # Skip tuple returns with simple values
        if line.startswith('(') and line.endswith(')') and 'let' not in line:
            continue
        # Skip single variable returns or simple expressions
        if line.replace('_', '').replace(' ', '').isalnum() and len(line) < 20:
            continue
        # Skip placeholder comments
        if 'placeholder' in line.lower() or 'dummy' in line.lower():
            continue
        real_lines.append(line)

    # If there's no real implementation content, it's a stub
    return len(real_lines) == 0


@dataclass
class GeneratedTask:
    """A generated training task."""
    task_uid: str
    sample_id: int
    task_type: str  # spec_gen, code_synth, spec_and_code, repair
    prompt_text: str
    target_text: str
    full_text: str  # Combined for SFT
    meta: dict


# ─────────────────────────────────────────────────────────────────────────────
# Task templates
# ─────────────────────────────────────────────────────────────────────────────

SPEC_GEN_PROMPT = """\
Add Verus verification specifications (requires/ensures clauses) to the following function. The specifications should capture the function's behavior and any necessary preconditions.

```rust
{input_code}
```
"""

SPEC_GEN_TARGET = """\
```verus
{output_code}
```
"""

CODE_SYNTH_PROMPT = """\
Implement the body of the following Verus function to satisfy its specification. The implementation must pass Verus verification.

```verus
{input_code}
```
"""

CODE_SYNTH_TARGET = """\
```verus
{output_code}
```
"""

SPEC_AND_CODE_PROMPT = """\
Complete the following Verus function by adding verification specifications (requires/ensures) and implementing the body. The result must pass Verus verification.

```rust
{input_code}
```
"""

SPEC_AND_CODE_TARGET = """\
```verus
{output_code}
```
"""

REPAIR_PROMPT = """\
Fix the following Verus code so that it passes verification. The current error is:
{error_message}

```verus
{input_code}
```
"""

REPAIR_TARGET = """\
```verus
{output_code}
```
"""


# ─────────────────────────────────────────────────────────────────────────────
# Task generation functions
# ─────────────────────────────────────────────────────────────────────────────


def compute_task_uid(sample_id: int, task_type: str, func_name: str) -> str:
    """Generate a unique ID for a task."""
    content = f"{sample_id}:{task_type}:{func_name}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def generate_spec_gen_task(
    code: str,
    func: FunctionInfo,
    sample_id: int,
) -> Optional[GeneratedTask]:
    """
    Generate a spec_gen task: add requires/ensures to an exec function.

    The input is the function without specs, output is the full function.
    """
    if func.kind != SegmentKind.EXEC:
        return None

    if not func.has_requires and not func.has_ensures:
        # No specs to learn from
        return None

    # Skip functions that are stubs (no real implementation to learn from)
    if is_function_stub(code, func):
        return None

    lines = code.split("\n")
    original_func = "\n".join(lines[func.start_line - 1:func.end_line])

    # Remove specs from the function
    stripped_func = remove_specs_from_function(code, func)
    if stripped_func == original_func:
        # Couldn't remove specs
        return None

    # Extract just the modified function portion
    stripped_lines = stripped_func.split("\n")
    if len(stripped_lines) > func.end_line:
        input_code = "\n".join(stripped_lines[func.start_line - 1:func.end_line])
    else:
        input_code = stripped_func

    prompt = SPEC_GEN_PROMPT.format(input_code=input_code.strip())
    target = SPEC_GEN_TARGET.format(output_code=original_func.strip())
    full_text = prompt + "\n" + target

    return GeneratedTask(
        task_uid=compute_task_uid(sample_id, "spec_gen", func.name),
        sample_id=sample_id,
        task_type="spec_gen",
        prompt_text=prompt,
        target_text=target,
        full_text=full_text,
        meta={
            "function_name": func.name,
            "original_lines": func.end_line - func.start_line + 1,
        },
    )


def generate_code_synth_task(
    code: str,
    func: FunctionInfo,
    sample_id: int,
) -> Optional[GeneratedTask]:
    """
    Generate a code_synth task: implement function body from spec.

    The input is the function with specs but stubbed body,
    output is the full function.
    """
    if func.kind != SegmentKind.EXEC:
        return None

    if not func.has_requires and not func.has_ensures:
        # No spec to guide synthesis
        return None

    # Skip functions that are already stubs
    if is_function_stub(code, func):
        return None

    lines = code.split("\n")
    original_func = "\n".join(lines[func.start_line - 1:func.end_line])

    # Stub the body
    stubbed_code = stub_function_body(code, func)
    stubbed_lines = stubbed_code.split("\n")

    # Find the stubbed function in the result
    # This is approximate - we extract similar line range
    if len(stubbed_lines) >= func.start_line:
        # The stubbed function will be shorter
        stub_end = func.start_line - 1
        for i in range(func.start_line - 1, min(func.start_line + 20, len(stubbed_lines))):
            if stubbed_lines[i].strip() == "}":
                stub_end = i + 1
                break

        input_code = "\n".join(stubbed_lines[func.start_line - 1:stub_end])
    else:
        # Fallback: just show the signature with unimplemented
        sig = extract_function_signature(code, func)
        input_code = sig + " {\n    unimplemented!()\n}"

    prompt = CODE_SYNTH_PROMPT.format(input_code=input_code.strip())
    target = CODE_SYNTH_TARGET.format(output_code=original_func.strip())
    full_text = prompt + "\n" + target

    return GeneratedTask(
        task_uid=compute_task_uid(sample_id, "code_synth", func.name),
        sample_id=sample_id,
        task_type="code_synth",
        prompt_text=prompt,
        target_text=target,
        full_text=full_text,
        meta={
            "function_name": func.name,
            "original_lines": func.end_line - func.start_line + 1,
        },
    )


def generate_spec_and_code_task(
    code: str,
    func: FunctionInfo,
    sample_id: int,
) -> Optional[GeneratedTask]:
    """
    Generate a spec_and_code task: add specs AND implement body.

    The input is the function signature only (no specs, no body),
    output is the full function.
    """
    if func.kind != SegmentKind.EXEC:
        return None

    if not func.has_requires and not func.has_ensures:
        # No spec to learn from
        return None

    # Skip functions that are already stubs
    if is_function_stub(code, func):
        return None

    lines = code.split("\n")
    original_func = "\n".join(lines[func.start_line - 1:func.end_line])

    # First, remove specs from the function entirely
    # This gives us signature + body without specs
    cleaned_func = remove_specs_from_function(code, func)

    # Extract the signature from the cleaned function (up to the opening brace)
    clean_sig_lines = []
    for line in cleaned_func.split("\n"):
        brace_pos = line.find("{")
        if brace_pos != -1:
            # Include content before the brace if there is any
            before_brace = line[:brace_pos].rstrip()
            if before_brace:
                clean_sig_lines.append(before_brace)
            break
        clean_sig_lines.append(line)

    clean_sig = "\n".join(clean_sig_lines).strip()

    # Create stubbed version
    input_code = clean_sig + " {\n    unimplemented!()\n}"

    prompt = SPEC_AND_CODE_PROMPT.format(input_code=input_code)
    target = SPEC_AND_CODE_TARGET.format(output_code=original_func.strip())
    full_text = prompt + "\n" + target

    return GeneratedTask(
        task_uid=compute_task_uid(sample_id, "spec_and_code", func.name),
        sample_id=sample_id,
        task_type="spec_and_code",
        prompt_text=prompt,
        target_text=target,
        full_text=full_text,
        meta={
            "function_name": func.name,
            "original_lines": func.end_line - func.start_line + 1,
        },
    )


def generate_repair_task(
    failing_code: str,
    fixed_code: str,
    error_message: str,
    sample_id: int,
    func_name: str = "unknown",
) -> GeneratedTask:
    """
    Generate a repair task from a failing/fixed code pair.

    Used when we have both failing and fixed versions.
    """
    prompt = REPAIR_PROMPT.format(
        error_message=error_message.strip(),
        input_code=failing_code.strip(),
    )
    target = REPAIR_TARGET.format(output_code=fixed_code.strip())
    full_text = prompt + "\n" + target

    return GeneratedTask(
        task_uid=compute_task_uid(sample_id, "repair", func_name),
        sample_id=sample_id,
        task_type="repair",
        prompt_text=prompt,
        target_text=target,
        full_text=full_text,
        meta={
            "error_type": "verification",
            "error_message_preview": error_message[:200],
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch task generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_tasks_from_code(
    code: str,
    sample_id: int,
    task_types: list[str] | None = None,
    max_per_sample: int = 5,
) -> list[GeneratedTask]:
    """
    Generate training tasks from a piece of Verus code.

    Args:
        code: The Verus source code
        sample_id: ID of the sample this code came from
        task_types: List of task types to generate (default: all)
        max_per_sample: Maximum tasks to generate per sample

    Returns:
        List of GeneratedTask objects
    """
    if task_types is None:
        task_types = ["spec_gen", "code_synth", "spec_and_code"]

    functions = get_exec_functions_with_specs(code)
    if not functions:
        # Try all exec functions
        functions = get_exec_functions(code)

    tasks: list[GeneratedTask] = []

    for func in functions:
        if len(tasks) >= max_per_sample:
            break

        for task_type in task_types:
            if len(tasks) >= max_per_sample:
                break

            task = None
            if task_type == "spec_gen":
                task = generate_spec_gen_task(code, func, sample_id)
            elif task_type == "code_synth":
                task = generate_code_synth_task(code, func, sample_id)
            elif task_type == "spec_and_code":
                task = generate_spec_and_code_task(code, func, sample_id)

            if task is not None:
                tasks.append(task)

    return tasks


def generate_tasks(
    db: Database,
    limit: int | None = None,
    task_types: list[str] | None = None,
    max_per_sample: int = 3,
    only_pass: bool = True,
) -> int:
    """
    Generate tasks from all samples in the database.

    Args:
        db: Database instance
        limit: Maximum number of samples to process
        task_types: Task types to generate
        max_per_sample: Max tasks per sample
        only_pass: Only use samples that pass verification

    Returns:
        Number of tasks generated
    """
    verification_result = "pass" if only_pass else None
    samples = db.list_samples(
        verification_result=verification_result,
        limit=limit,
    )

    total_tasks = 0

    for sample in samples:
        code = sample["code_text"]

        try:
            tasks = generate_tasks_from_code(
                code=code,
                sample_id=sample["sample_id"],
                task_types=task_types,
                max_per_sample=max_per_sample,
            )
        except Exception as e:
            logger.warning(f"Error generating tasks for sample {sample['sample_id']}: {e}")
            continue

        for task in tasks:
            # Check if task already exists
            existing = db.get_task_by_uid(task.task_uid)
            if existing:
                continue

            db.add_task(
                task_uid=task.task_uid,
                sample_id=task.sample_id,
                task_type=task.task_type,
                prompt_text=task.prompt_text,
                target_text=task.target_text,
                full_text=task.full_text,
                meta_json=json.dumps(task.meta),
            )
            total_tasks += 1

        logger.info(
            f"Generated {len(tasks)} tasks from sample {sample['sample_id']}"
        )

    logger.info(f"Generated {total_tasks} total tasks")
    return total_tasks


# ─────────────────────────────────────────────────────────────────────────────
# Task iteration for export
# ─────────────────────────────────────────────────────────────────────────────


def iter_tasks_for_export(
    db: Database,
    task_type: str | None = None,
    split: str | None = None,
    only_pass: bool = True,
    limit: int | None = None,
) -> Iterator[dict]:
    """
    Iterate over tasks formatted for export.

    Yields dicts ready for JSONL serialization.
    """
    tasks = db.list_tasks(
        task_type=task_type,
        split=split,
        limit=limit,
    )

    for task in tasks:
        sample = db.get_sample(task["sample_id"])
        if sample is None:
            continue

        if only_pass and sample["verification_result"] != "pass":
            continue

        meta = json.loads(task["meta_json"])
        meta["task_type"] = task["task_type"]
        meta["sample_uid"] = sample["sample_uid"]

        yield {
            "text": task["full_text"],
            "prompt": task["prompt_text"],
            "target": task["target_text"],
            "task_type": task["task_type"],
            "meta": meta,
        }
