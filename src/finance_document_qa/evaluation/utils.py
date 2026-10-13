from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def context_text(context: Any) -> str:
    if isinstance(context, str):
        return context.strip()
    if isinstance(context, dict):
        return str(context.get("text") or "").strip()
    return str(context or "").strip()


def get_retrieved_contexts(
    row: dict[str, Any], max_contexts: int, max_context_chars: int
) -> list[str]:
    contexts = row.get("retrieved_contexts") or []
    if not isinstance(contexts, list):
        return []

    output: list[str] = []
    for context in contexts[:max_contexts]:
        text = context_text(context)
        if not text:
            continue
        if max_context_chars > 0:
            text = text[:max_context_chars]
        output.append(text)
    return output


def clean_score(value: Any) -> float | None:
    if value is None:
        return None
    result = getattr(value, "value", value)
    if isinstance(result, float) and math.isnan(result):
        return None
    try:
        number = float(result)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def selected_rows(
    rows: list[dict[str, Any]], question_ids: set[str], limit: int
) -> list[tuple[int, dict[str, Any]]]:
    selected = [
        (idx, row)
        for idx, row in enumerate(rows)
        if not question_ids or str(row.get("question_id")) in question_ids
    ]
    if limit > 0:
        selected = selected[:limit]
    return selected


def mean_metric(evaluations: list[dict[str, Any]], key: str) -> float | None:
    values = [item.get(key) for item in evaluations if item.get(key) is not None]
    return sum(values) / len(values) if values else None
