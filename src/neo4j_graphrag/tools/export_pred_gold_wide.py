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


def _extract_payload(raw_answer: Any) -> dict[str, Any] | None:
    if isinstance(raw_answer, dict):
        return raw_answer
    if raw_answer is None:
        return None
    text = str(raw_answer).strip()
    if not text:
        return None
    candidate = _strip_json_fence(text)
    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = (
                    item.get("evidence_text")
                    or item.get("text")
                    or item.get("quote")
                    or item.get("reference")
                )
                if text is not None:
                    parts.append(str(text).strip())
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item).strip())
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        text = (
            value.get("evidence_text")
            or value.get("text")
            or value.get("quote")
            or value.get("reference")
        )
        if text is not None:
            return str(text).strip()
    return json.dumps(value, ensure_ascii=False)


def _extract_pred_answer(raw_answer: Any) -> str:
    if raw_answer is None:
        return ""
    payload = _extract_payload(raw_answer)
    if payload is not None:
        for key in ("answer", "final_answer"):
            value = payload.get(key)
            normalized = _normalize_value(value)
            if normalized:
                return normalized
        return ""
    return str(raw_answer).strip()


def _extract_pred_evidence(raw_evidence: Any, raw_answer: Any) -> str:
    normalized_evidence = _normalize_value(raw_evidence)
    if normalized_evidence:
        return normalized_evidence
    payload = _extract_payload(raw_answer)
    if payload is None:
        return ""
    return _normalize_value(payload.get("evidence"))


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
                "gold_type": gold.get("gold_type", gold.get("question_type", gold.get("type"))),
                "gold_answer": gold.get("gold_answer", gold.get("answer")),
                "gold_reference": gold.get(
                    "gold_reference",
                    gold.get("gold_evidence", gold.get("references", gold.get("evidence"))),
                ),
                "rag_pred_answer": "",
                "rag_evidence": "",
                "graphrag_pred_answer": "",
                "graphrag_evidence": "",
            }

        w = pred_by_question[question]
        w[f"{mode}_pred_answer"] = _extract_pred_answer(row.get("answer"))
        w[f"{mode}_evidence"] = _extract_pred_evidence(row.get("evidence"), row.get("answer"))

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
        "rag_pred_answer",
        "rag_evidence",
        "graphrag_pred_answer",
        "graphrag_evidence",
        "gold_answer",
        "gold_reference",
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
