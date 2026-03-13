#!/usr/bin/env python3
"""Merge query predictions with gold QA rows for manual inspection."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```json\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^```\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_pred_answer(raw_answer: Any) -> str:
    if raw_answer is None:
        return ""
    if isinstance(raw_answer, dict):
        for key in ("answer", "final_answer"):
            value = raw_answer.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return json.dumps(raw_answer, ensure_ascii=False)

    text = str(raw_answer).strip()
    if not text:
        return ""

    candidate = _strip_json_fence(text)
    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            for key in ("answer", "final_answer"):
                value = payload.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
    except Exception:
        pass

    return text


def _gold_index(qa_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in qa_rows:
        question = str(row.get("question", "")).strip()
        if not question or question in out:
            continue
        out[question] = row
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _format_duration_minsec(seconds: float) -> str:
    total = max(0.0, float(seconds))
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = total % 60
    if hours > 0:
        return f"{hours}h {minutes}min {secs:.1f}s"
    return f"{minutes}min {secs:.1f}s"


def _normalize_latency(row: dict[str, Any]) -> str | None:
    latency = row.get("latency")
    if isinstance(latency, str) and latency.strip():
        return latency.strip()
    latency_seconds = row.get("latency_seconds")
    if isinstance(latency_seconds, (int, float)):
        return _format_duration_minsec(float(latency_seconds))
    latency_ms = row.get("latency_ms")
    if isinstance(latency_ms, (int, float)):
        return _format_duration_minsec(float(latency_ms) / 1000.0)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge prediction rows with gold QA rows for side-by-side inspection."
    )
    parser.add_argument(
        "--pred-path",
        default="output/financebench/pred_compare.json",
        help="Prediction JSON file from query_router.",
    )
    parser.add_argument(
        "--qa-path",
        default="output/financebench/qa.jsonl",
        help="Gold QA JSONL file.",
    )
    parser.add_argument(
        "--out-jsonl",
        default="output/financebench/pred_gold_merged.jsonl",
        help="Merged long-format JSONL output path.",
    )
    parser.add_argument(
        "--out-json",
        default="output/financebench/pred_gold_merged.json",
        help="Merged JSON array output path.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[],
        choices=["rag", "graphrag"],
        help="Optional filter by mode(s). If empty, keep all rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pred_path = Path(args.pred_path)
    qa_path = Path(args.qa_path)
    out_jsonl = Path(args.out_jsonl)
    out_json = Path(args.out_json)

    pred_rows = _read_json(pred_path)
    if not isinstance(pred_rows, list):
        raise ValueError(f"Prediction file must be a JSON array: {pred_path}")

    qa_rows = _read_jsonl(qa_path)
    gold_by_question = _gold_index(qa_rows)

    merged: list[dict[str, Any]] = []
    missing_gold = 0
    for row in pred_rows:
        mode = row.get("mode")
        if args.modes and mode not in args.modes:
            continue

        question = str(row.get("question", "")).strip()
        gold = gold_by_question.get(question)
        if gold is None:
            missing_gold += 1
            gold = {}

        merged_row = {
            "_id": gold.get("_id"),
            "question": question,
            "mode": mode,
            "retrieval": row.get("retrieval"),
            "pred_answer": _extract_pred_answer(row.get("answer")),
            "pred_answer_raw": row.get("answer"),
            "gold_answer": gold.get("gold_answer"),
            "gold_evidence": gold.get("gold_evidence", gold.get("gold_reference")),
            "gold_type": gold.get("gold_type"),
            "category": gold.get("category"),
            "reasoning": gold.get("reasoning"),
            "latency": _normalize_latency(row),
            "retriever": row.get("retriever"),
            "retrieved_chunk_ids": row.get("retrieved_chunk_ids"),
            "context_preview": row.get("context_preview"),
            "error": row.get("error"),
        }
        merged.append(merged_row)

    _write_jsonl(out_jsonl, merged)
    _write_json(out_json, merged)

    print("Done.")
    print(f"Pred rows:     {len(pred_rows)}")
    print(f"Merged rows:   {len(merged)}")
    print(f"Missing gold:  {missing_gold}")
    print(f"Out JSONL:     {out_jsonl.resolve()}")
    print(f"Out JSON:      {out_json.resolve()}")


if __name__ == "__main__":
    main()
