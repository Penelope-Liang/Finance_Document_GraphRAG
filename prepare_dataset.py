#!/usr/bin/env python3
"""Download a QA dataset from Hugging Face and export files for query/evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a HF QA dataset into questions.txt (for query_router) "
            "and qa.jsonl (for offline evaluation)."
        )
    )
    parser.add_argument(
        "--dataset",
        default="PatronusAI/financebench",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--split",
        default="",
        help="Dataset split name. If empty, prefer train then first available split.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/financebench",
        help="Output directory.",
    )
    parser.add_argument(
        "--questions-out",
        default="questions.txt",
        help="Questions file name (one question per line; no id prefix).",
    )
    parser.add_argument(
        "--qa-out",
        default="qa.jsonl",
        help="QA metadata JSONL file name.",
    )
    parser.add_argument(
        "--text-key",
        default="auto",
        help="Column name for question text.",
    )
    parser.add_argument(
        "--id-key",
        default="auto",
        help="Column name for question id.",
    )
    parser.add_argument(
        "--answer-key",
        default="auto",
        help="Column name for gold answer.",
    )
    parser.add_argument(
        "--reference-key",
        default="auto",
        help="Column name for gold reference/evidence.",
    )
    parser.add_argument(
        "--type-key",
        default="auto",
        help="Column name for gold question type.",
    )
    parser.add_argument(
        "--category-key",
        default="auto",
        help="Column name for category.",
    )
    parser.add_argument(
        "--reasoning-key",
        default="auto",
        help="Column name for reasoning flag.",
    )
    return parser.parse_args()


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _choose_split(ds_obj: Any, preferred: str) -> tuple[str, Any]:
    # DatasetDict-like
    if hasattr(ds_obj, "keys") and callable(ds_obj.keys):
        keys = list(ds_obj.keys())
        if not keys:
            raise ValueError("Dataset has no splits")
        if preferred:
            if preferred not in ds_obj:
                raise ValueError(f"Split '{preferred}' not found. Available splits: {keys}")
            return preferred, ds_obj[preferred]
        if "train" in ds_obj:
            return "train", ds_obj["train"]
        first = keys[0]
        return first, ds_obj[first]

    # Single Dataset
    split_name = preferred or "dataset"
    return split_name, ds_obj


def _pick_key(
    sample_row: dict[str, Any],
    *,
    user_value: str,
    candidates: list[str],
) -> str | None:
    if user_value and user_value != "auto":
        return user_value
    for key in candidates:
        if key in sample_row:
            return key
    return None


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Could not import datasets. Install with `python3 -m pip install datasets huggingface_hub`."
        ) from exc

    ds_obj = load_dataset(args.dataset)
    split_name, ds = _choose_split(ds_obj, args.split)
    if len(ds) == 0:
        raise RuntimeError(f"Dataset split is empty: {args.dataset}:{split_name}")

    sample_row = ds[0]
    if not isinstance(sample_row, dict):
        raise RuntimeError("Dataset row is not a dict-like record; cannot map columns.")

    text_key = _pick_key(
        sample_row,
        user_value=args.text_key,
        candidates=["question", "text", "query", "prompt", "instruction"],
    )
    if not text_key:
        raise RuntimeError(
            "Could not auto-detect question text key. "
            "Please pass --text-key explicitly."
        )
    id_key = _pick_key(
        sample_row,
        user_value=args.id_key,
        candidates=["_id", "id", "question_id", "uid"],
    )
    answer_key = _pick_key(
        sample_row,
        user_value=args.answer_key,
        candidates=[
            "answer",
            "gold_answer",
            "final_answer",
            "expected_answer",
            "ground_truth",
        ],
    )
    reference_key = _pick_key(
        sample_row,
        user_value=args.reference_key,
        candidates=[
            "references",
            "reference",
            "evidence",
            "gold_evidence",
            "gold_reference",
            "context",
            "source",
        ],
    )
    type_key = _pick_key(
        sample_row,
        user_value=args.type_key,
        candidates=["type", "gold_type", "question_type", "task_type"],
    )
    category_key = _pick_key(
        sample_row,
        user_value=args.category_key,
        candidates=["category", "topic", "domain", "sector"],
    )
    reasoning_key = _pick_key(
        sample_row,
        user_value=args.reasoning_key,
        candidates=["reasoning", "requires_reasoning", "reasoning_required"],
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions_path = out_dir / args.questions_out
    qa_path = out_dir / args.qa_out

    total_rows = 0
    written_rows = 0
    skipped_rows = 0

    with questions_path.open("w", encoding="utf-8") as fq, qa_path.open("w", encoding="utf-8") as fqa:
        for row in ds:
            total_rows += 1
            question = _as_text(row.get(text_key)).strip()
            if not question:
                skipped_rows += 1
                continue

            fq.write(question + "\n")

            qa_record = {
                "_id": _as_text(row.get(id_key)).strip() or f"{split_name}-{total_rows}",
                "question": question,
                "gold_answer": row.get(answer_key) if answer_key else None,
                "gold_evidence": row.get(reference_key) if reference_key else None,
                "gold_type": row.get(type_key) if type_key else None,
                "category": row.get(category_key) if category_key else None,
                "reasoning": row.get(reasoning_key) if reasoning_key else None,
                "source_dataset": args.dataset,
                "source_split": split_name,
            }
            fqa.write(json.dumps(qa_record, ensure_ascii=False) + "\n")
            written_rows += 1

    print("Done.")
    print(f"Dataset:   {args.dataset}")
    print(f"Split:     {split_name}")
    print(f"Rows total:{total_rows}")
    print(f"Rows wrote:{written_rows}")
    print(f"Rows skip: {skipped_rows}")
    print(f"Key map:   text={text_key}, id={id_key}, answer={answer_key}, reference={reference_key}, type={type_key}, category={category_key}, reasoning={reasoning_key}")
    print(f"Questions: {questions_path.resolve()}")
    print(f"QA JSONL:  {qa_path.resolve()}")


if __name__ == "__main__":
    main()
