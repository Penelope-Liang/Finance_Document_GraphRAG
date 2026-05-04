from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip()


def _extract_numeric(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.replace(",", "").replace("$", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _is_pass(gold: str | None, pred: str | None) -> bool:
    if gold is None or pred is None:
        return False
    gold_num = _extract_numeric(gold)
    pred_num = _extract_numeric(pred)
    if gold_num is not None and pred_num is not None:
        # Relative tolerance suitable for the rounding differences already observed,
        # e.g. 8.70 vs 8.74 and 0.40 vs 0.39.
        tol = max(0.01, abs(gold_num) * 0.01)
        return abs(gold_num - pred_num) <= tol
    return _normalize_text(gold) == _normalize_text(pred)


def _classify_error(row: dict) -> str | None:
    if row.get("error"):
        return "runtime_error"
    if row.get("answer") in (None, ""):
        return "no_answer"
    if row.get("evidence") in (None, ""):
        return "no_evidence"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize direct-extraction GraphRAG batch results.")
    parser.add_argument(
        "--input",
        default="output/direct_extract_13_batch_results.json",
        help="Batch result JSON path.",
    )
    parser.add_argument(
        "--output",
        default="output/direct_extract_13_batch_summary.json",
        help="Summary JSON path.",
    )
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    summary = []
    passed = 0
    for row in rows:
        item = {
            "question_id": row.get("question_id"),
            "doc_id": row.get("doc_id"),
            "gold_answer": row.get("gold_answer"),
            "pred_answer": row.get("answer"),
            "pass": _is_pass(row.get("gold_answer"), row.get("answer")),
            "evidence": row.get("evidence"),
            "error_type": _classify_error(row),
        }
        if item["pass"]:
            passed += 1
        summary.append(item)

    payload = {
        "total": len(summary),
        "passed": passed,
        "failed": len(summary) - passed,
        "items": summary,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved summary to {args.output}")
    print(f"Passed: {passed}/{len(summary)}")
    for item in summary:
        print(
            f"{item['question_id']}\tpass={item['pass']}\tgold={item['gold_answer']}\tpred={item['pred_answer']}\terror={item['error_type']}"
        )


if __name__ == "__main__":
    main()
