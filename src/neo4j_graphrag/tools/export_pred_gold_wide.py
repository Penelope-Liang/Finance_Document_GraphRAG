#!/usr/bin/env python3
"""Export side-by-side RAG/GraphRAG predictions with gold answers."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        rows.append(json.loads(raw))
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


def _gold_index(qa_rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    idx: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in qa_rows:
        question = str(row.get("question", row.get("text", ""))).strip()
        if not question or question in idx:
            continue
        idx[question] = row
        order.append(question)
    return idx, order


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


def _to_wide_rows(pred_rows: list[dict[str, Any]], qa_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gold_by_question, gold_order = _gold_index(qa_rows)
    pred_by_question: dict[str, dict[str, Any]] = {}

    for row in pred_rows:
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        mode = str(row.get("mode", "")).strip().lower()
        if mode not in {"rag", "graphrag"}:
            continue

        if question not in pred_by_question:
            gold = gold_by_question.get(question, {})
            pred_by_question[question] = {
                "_id": gold.get("_id", gold.get("id")),
                "question": question,
                "gold_answer": gold.get("gold_answer", gold.get("answer")),
                "gold_evidence": gold.get(
                    "gold_evidence",
                    gold.get("gold_reference", gold.get("references", gold.get("evidence"))),
                ),
                "gold_type": gold.get("gold_type", gold.get("question_type", gold.get("type"))),
                "rag_pred_answer": "",
                "graphrag_pred_answer": "",
                "rag_latency": None,
                "graphrag_latency": None,
                "rag_error": None,
                "graphrag_error": None,
            }

        w = pred_by_question[question]
        w[f"{mode}_pred_answer"] = _extract_pred_answer(row.get("answer"))
        w[f"{mode}_latency"] = _normalize_latency(row)
        w[f"{mode}_error"] = row.get("error")

    ordered_questions = [q for q in gold_order if q in pred_by_question]
    seen = set(ordered_questions)
    for q in pred_by_question:
        if q not in seen:
            ordered_questions.append(q)
    return [pred_by_question[q] for q in ordered_questions]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "_id",
        "question",
        "gold_type",
        "gold_answer",
        "rag_pred_answer",
        "graphrag_pred_answer",
        "rag_latency",
        "graphrag_latency",
        "rag_error",
        "graphrag_error",
        "gold_evidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a wide-format file with question, rag_pred_answer, "
            "graphrag_pred_answer, and gold_answer."
        )
    )
    parser.add_argument(
        "--pred-path",
        default="output/run_20260311_210657/pred_compare.json",
        help="Prediction JSON file produced by query.router --mode compare.",
    )
    parser.add_argument(
        "--qa-path",
        default="output/financebench/qa.jsonl",
        help="Gold QA JSONL file.",
    )
    parser.add_argument(
        "--out-jsonl",
        default="output/financebench/pred_gold_wide.jsonl",
        help="Wide-format output JSONL path.",
    )
    parser.add_argument(
        "--out-csv",
        default="output/financebench/pred_gold_wide.csv",
        help="Wide-format output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred_path)
    qa_path = Path(args.qa_path)
    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)

    pred_rows = _read_json(pred_path)
    if not isinstance(pred_rows, list):
        raise ValueError(f"Prediction file must be a JSON array: {pred_path}")
    qa_rows = _read_jsonl(qa_path)

    wide_rows = _to_wide_rows(pred_rows, qa_rows)
    _write_jsonl(out_jsonl, wide_rows)
    _write_csv(out_csv, wide_rows)

    rag_ready = sum(1 for r in wide_rows if str(r.get("rag_pred_answer", "")).strip())
    graphrag_ready = sum(
        1 for r in wide_rows if str(r.get("graphrag_pred_answer", "")).strip()
    )
    print("Done.")
    print(f"Input pred rows:   {len(pred_rows)}")
    print(f"Input qa rows:     {len(qa_rows)}")
    print(f"Wide output rows:  {len(wide_rows)}")
    print(f"Rows with RAG ans: {rag_ready}")
    print(f"Rows with GR ans:  {graphrag_ready}")
    print(f"Out JSONL:         {out_jsonl.resolve()}")
    print(f"Out CSV:           {out_csv.resolve()}")


if __name__ == "__main__":
    main()
