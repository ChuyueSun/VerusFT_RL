"""
verus_dataset/export.py - Export utilities for samples and tasks.

Supports multiple export formats:
- JSONL for training (SFT format with single "text" field)
- JSONL with metadata for analysis
- CSV for spreadsheet analysis
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from .db import Database
from .tasks import iter_tasks_for_export

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sample export
# ─────────────────────────────────────────────────────────────────────────────


def iter_samples_for_export(
    db: Database,
    verification_result: str | None = None,
    minimized_only: bool = False,
    limit: int | None = None,
) -> Iterator[dict]:
    """
    Iterate over samples formatted for export.

    Yields dicts ready for JSONL serialization.
    """
    samples = db.list_samples(
        verification_result=verification_result,
        minimized_only=minimized_only,
        limit=limit,
    )

    for sample in samples:
        repo = db.get_repo_by_id(sample["repo_id"])
        repo_info = {
            "name": repo["name"] if repo else "unknown",
            "url": repo["url"] if repo else None,
        }

        segments = None
        if sample["segments_json"]:
            try:
                segments = json.loads(sample["segments_json"])
            except json.JSONDecodeError:
                pass

        meta = {}
        if sample["meta_json"]:
            try:
                meta = json.loads(sample["meta_json"])
            except json.JSONDecodeError:
                pass

        yield {
            "id": sample["sample_uid"],
            "text": f"```verus\n{sample['code_text']}\n```",
            "verus_code": sample["code_text"],
            "metadata": {
                "origin_repo": repo_info["name"],
                "origin_repo_url": repo_info["url"],
                "minimized": bool(sample["minimized"]),
                "verification_result": sample["verification_result"],
                "fingerprint": sample["fingerprint"],
                "nbytes": sample["nbytes"],
                "nlines": sample["nlines"],
                "segments": segments,
                **meta,
            },
        }


def export_samples_jsonl(
    db: Database,
    output_path: Path | str,
    verification_result: str | None = None,
    minimized_only: bool = False,
    limit: int | None = None,
    include_code_field: bool = True,
) -> int:
    """
    Export samples to JSONL file.

    Args:
        db: Database instance
        output_path: Path to output file
        verification_result: Filter by result ("pass" or "fail")
        minimized_only: Only export minimized samples
        limit: Maximum samples to export
        include_code_field: Include separate verus_code field

    Returns:
        Number of samples exported
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in iter_samples_for_export(
            db,
            verification_result=verification_result,
            minimized_only=minimized_only,
            limit=limit,
        ):
            if not include_code_field:
                sample.pop("verus_code", None)

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"Exported {count} samples to {output_path}")
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Task export
# ─────────────────────────────────────────────────────────────────────────────


def export_tasks_jsonl(
    db: Database,
    output_path: Path | str,
    format: str = "sft",
    task_type: str | None = None,
    split: str | None = None,
    only_pass: bool = True,
    limit: int | None = None,
) -> int:
    """
    Export tasks to JSONL file.

    Args:
        db: Database instance
        output_path: Path to output file
        format: "sft" (text field only) or "full" (all fields)
        task_type: Filter by task type
        split: Filter by split
        only_pass: Only export tasks from passing samples
        limit: Maximum tasks to export

    Returns:
        Number of tasks exported
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for task in iter_tasks_for_export(
            db,
            task_type=task_type,
            split=split,
            only_pass=only_pass,
            limit=limit,
        ):
            if format == "sft":
                # SFT format: only text field
                record = {"text": task["text"]}
            else:
                # Full format: all fields
                record = task

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"Exported {count} tasks to {output_path} (format={format})")
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Statistics export
# ─────────────────────────────────────────────────────────────────────────────


def export_stats(
    db: Database,
    output_path: Path | str | None = None,
) -> dict:
    """
    Export pipeline statistics.

    Args:
        db: Database instance
        output_path: Optional path to save stats JSON

    Returns:
        Stats dictionary
    """
    stats = db.get_stats()
    stats["exported_at"] = datetime.utcnow().isoformat()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(stats, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Exported stats to {output_path}")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CSV export for analysis
# ─────────────────────────────────────────────────────────────────────────────


def export_samples_csv(
    db: Database,
    output_path: Path | str,
    limit: int | None = None,
) -> int:
    """
    Export sample metadata to CSV for analysis.

    Args:
        db: Database instance
        output_path: Path to output file
        limit: Maximum samples to export

    Returns:
        Number of samples exported
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = db.list_samples(limit=limit)

    fieldnames = [
        "sample_uid",
        "repo_name",
        "minimized",
        "verification_result",
        "fingerprint",
        "nbytes",
        "nlines",
        "created_at",
    ]

    count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sample in samples:
            repo = db.get_repo_by_id(sample["repo_id"])
            writer.writerow({
                "sample_uid": sample["sample_uid"],
                "repo_name": repo["name"] if repo else "unknown",
                "minimized": bool(sample["minimized"]),
                "verification_result": sample["verification_result"],
                "fingerprint": sample["fingerprint"],
                "nbytes": sample["nbytes"],
                "nlines": sample["nlines"],
                "created_at": sample["created_at"],
            })
            count += 1

    logger.info(f"Exported {count} samples to CSV at {output_path}")
    return count


def export_tasks_csv(
    db: Database,
    output_path: Path | str,
    limit: int | None = None,
) -> int:
    """
    Export task metadata to CSV for analysis.

    Args:
        db: Database instance
        output_path: Path to output file
        limit: Maximum tasks to export

    Returns:
        Number of tasks exported
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = db.list_tasks(limit=limit)

    fieldnames = [
        "task_uid",
        "sample_id",
        "task_type",
        "split",
        "prompt_length",
        "target_length",
        "full_length",
        "created_at",
    ]

    count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for task in tasks:
            writer.writerow({
                "task_uid": task["task_uid"],
                "sample_id": task["sample_id"],
                "task_type": task["task_type"],
                "split": task["split"],
                "prompt_length": len(task["prompt_text"]),
                "target_length": len(task["target_text"]) if task["target_text"] else 0,
                "full_length": len(task["full_text"]),
                "created_at": task["created_at"],
            })
            count += 1

    logger.info(f"Exported {count} tasks to CSV at {output_path}")
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Hugging Face format helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_jsonl_as_hf_dataset(
    jsonl_path: Path | str,
    text_field: str = "text",
):
    """
    Load a JSONL file as a Hugging Face Dataset.

    Requires the `datasets` library.
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("datasets library required: pip install datasets")

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return Dataset.from_list(records)


def build_dataset_from_db(
    db: Database,
    task_type: str | None = None,
    only_pass: bool = True,
    limit: int | None = None,
):
    """
    Build a Hugging Face Dataset directly from the database.

    This matches the interface expected by sft_example.py.

    Returns a Dataset with "text" field.
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("datasets library required: pip install datasets")

    records = []
    for task in iter_tasks_for_export(
        db,
        task_type=task_type,
        only_pass=only_pass,
        limit=limit,
    ):
        records.append({"text": task["text"]})

    return Dataset.from_list(records)
