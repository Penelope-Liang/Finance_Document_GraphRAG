#!/usr/bin/env python3
"""Prepare the 14-question FinanceBench metrics-generated information-extraction subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the FinanceBench metrics-generated + Information extraction subset "
            "with exact question list, QA rows, and gold question->doc mapping."
        )
    )
    parser.add_argument(
        "--input-path",
        default="dataset/financebench/financebench_merged.jsonl",
        help="Path to financebench_merged.jsonl.",
    )
    parser.add_argument(
        "--out-dir",
        default="output/financebench",
        help="Directory for generated subset artifacts.",
    )
    parser.add_argument(
        "--prefix",
        default="metrics_extract",
        help="Output filename prefix.",
    )
    return parser.parse_args()


def _select_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("question_type") != "metrics-generated":
            continue
        if row.get("question_reasoning") != "Information extraction":
            continue
        selected.append(row)
    return selected


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _select_rows(_iter_jsonl(input_path))
    rows.sort(key=lambda r: (str(r.get("doc_name", "")), str(r.get("financebench_id", ""))))

    qa_path = out_dir / f"qa_{args.prefix}.jsonl"
    questions_path = out_dir / f"questions_{args.prefix}.txt"
    qmap_path = out_dir / f"question_doc_map_{args.prefix}.jsonl"

    qa_rows: list[dict[str, Any]] = []
    question_lines: list[str] = []
    qmap_rows: list[dict[str, Any]] = []

    for row in rows:
        question = str(row.get("question", "")).strip()
        doc_name = str(row.get("doc_name", "")).strip()
        answer = str(row.get("answer", "")).strip()
        financebench_id = str(row.get("financebench_id", "")).strip()
        if not question or not doc_name:
            continue

        evidence = row.get("evidence") or []
        gold_pages = sorted(
            {
                int(ev.get("evidence_page_num"))
                for ev in evidence
                if isinstance(ev, dict) and ev.get("evidence_page_num") is not None
            }
        )

        qa_rows.append(
            {
                "_id": financebench_id,
                "question": question,
                "answer": answer,
                "gold_type": "metrics-generated",
                "doc_name": doc_name,
                "question_reasoning": row.get("question_reasoning"),
                "gold_pages": gold_pages,
            }
        )
        question_lines.append(question)
        qmap_rows.append(
            {
                "question": question,
                "doc_ids": [f"{doc_name}.pdf", doc_name],
                "gold_pages": gold_pages,
            }
        )

    with qa_path.open("w", encoding="utf-8") as f:
        for row in qa_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    questions_path.write_text("\n".join(question_lines) + ("\n" if question_lines else ""), encoding="utf-8")
    with qmap_path.open("w", encoding="utf-8") as f:
        for row in qmap_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Done.")
    print(f"Input rows:   {len(rows)}")
    print(f"Questions:    {len(question_lines)}")
    print(f"Unique docs:  {len({str(r.get('doc_name', '')).strip() for r in rows if r.get('doc_name')})}")
    print(f"QA:           {qa_path}")
    print(f"Questions:    {questions_path}")
    print(f"Question map: {qmap_path}")


if __name__ == "__main__":
    main()
