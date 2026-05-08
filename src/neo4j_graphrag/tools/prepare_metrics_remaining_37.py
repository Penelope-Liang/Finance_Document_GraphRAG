#!/usr/bin/env python3
"""Prepare the remaining 37 metrics-generated FinanceBench questions after the 13 direct-extraction subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXCLUDED_DIRECT_13_IDS = {
    "financebench_id_03029",
    "financebench_id_04672",
    "financebench_id_08286",
    "financebench_id_03882",
    "financebench_id_05718",
    "financebench_id_04417",
    "financebench_id_07661",
    "financebench_id_10285",
    "financebench_id_04209",
    "financebench_id_04700",
    "financebench_id_03282",
    "financebench_id_03531",
    "financebench_id_04980",
}


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        payload = json.loads(raw)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the remaining 37 FinanceBench metrics-generated questions after "
            "removing the current 13 direct-extraction benchmark items."
        )
    )
    parser.add_argument(
        "--input-path",
        default="dataset/financebench/financebench_merged.jsonl",
        help="Path to financebench_merged.jsonl.",
    )
    parser.add_argument(
        "--output-path",
        default="scripts/financebench_metrics_remaining_37.jsonl",
        help="Output JSONL path.",
    )
    return parser.parse_args()


def _select_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("question_type") != "metrics-generated":
            continue
        financebench_id = str(row.get("financebench_id", "")).strip()
        if financebench_id in EXCLUDED_DIRECT_13_IDS:
            continue
        selected.append(row)
    return selected


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _select_rows(_iter_jsonl(input_path))
    rows.sort(key=lambda r: (str(r.get("doc_name", "")), str(r.get("financebench_id", ""))))

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        doc_name = str(row.get("doc_name", "")).strip()
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        financebench_id = str(row.get("financebench_id", "")).strip()
        if not doc_name or not question or not financebench_id:
            continue
        output_rows.append(
            {
                "doc_id": f"{doc_name}.pdf",
                "doc_name": doc_name,
                "question_id": financebench_id,
                "question": question,
                "gold_answer": answer,
                "question_reasoning": row.get("question_reasoning"),
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Done.")
    print(f"Questions:   {len(output_rows)}")
    print(f"Unique docs: {len({row['doc_name'] for row in output_rows})}")
    print(f"Output:      {output_path}")


if __name__ == "__main__":
    main()
