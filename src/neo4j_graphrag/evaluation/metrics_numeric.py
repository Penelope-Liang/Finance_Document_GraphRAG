#!/usr/bin/env python3
"""Numeric evaluation for FinanceBench-style metrics questions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


NUMBER_RE = re.compile(r"[-+]?\(?\d[\d,]*(?:\.\d+)?\)?")
OLLAMA_CONTENT_RE = re.compile(r"content='((?:\\'|[^'])*)'")

MONEY_UNIT_TO_DOLLARS = {
    "usd_thousands": 1e3,
    "usd_millions": 1e6,
    "usd_billions": 1e9,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate metrics-generated predictions with numeric parsing, unit normalization, "
            "and tolerance-based matching."
        )
    )
    parser.add_argument("--pred-path", required=True, help="Prediction JSON path.")
    parser.add_argument("--qa-path", required=True, help="Gold QA JSONL path.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["rag", "graphrag"],
        choices=["rag", "graphrag"],
        help="Which modes to evaluate.",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.02,
        help="Relative tolerance for numeric match.",
    )
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=0.05,
        help="Absolute tolerance in normalized target units.",
    )
    parser.add_argument(
        "--skip-empty-answer",
        action="store_true",
        help="Skip rows where model answer is empty/null.",
    )
    return parser.parse_args()


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
        return ""

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
            return ""
    except Exception:
        pass

    match = OLLAMA_CONTENT_RE.search(text)
    if match:
        content = match.group(1).replace("\\'", "'").strip()
        if content:
            return content

    return text


def _gold_index(qa_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    dup = 0
    for row in qa_rows:
        q = str(row.get("question", "")).strip()
        if not q:
            continue
        if q in idx:
            dup += 1
            continue
        idx[q] = row
    if dup:
        print(f"[warn] duplicated questions in gold set: {dup} (first occurrence used)")
    return idx


def _infer_target_unit(question: str) -> str:
    q = question.lower()
    if "usd billions" in q or "in usd billions" in q:
        return "usd_billions"
    if "usd millions" in q or "in usd millions" in q:
        return "usd_millions"
    if "usd thousands" in q or "in usd thousands" in q:
        return "usd_thousands"
    if "units of percents" in q or "in percents" in q or " as a %" in q or "percent" in q:
        return "percent"
    return "plain"


def _contains_any(text: str, patterns: list[str]) -> bool:
    return any(p in text for p in patterns)


def _infer_explicit_unit(text: str, target_unit: str) -> str:
    t = text.lower()
    if "%" in t or _contains_any(t, [" percent", " percents", " percentage "]):
        return "percent"
    if _contains_any(t, [" billion", " billions", " bn", " bn."]):
        return "usd_billions"
    if _contains_any(t, [" million", " millions", " mm", " mn"]):
        return "usd_millions"
    if _contains_any(t, [" thousand", " thousands", " k "]):
        return "usd_thousands"
    return target_unit


def _extract_first_number(text: str) -> float | None:
    match = NUMBER_RE.search(text)
    if not match:
        return None
    token = match.group(0).replace(",", "")
    negative = token.startswith("(") and token.endswith(")")
    token = token.strip("()")
    try:
        value = float(token)
    except ValueError:
        return None
    return -value if negative else value


def _normalize_numeric(text: str, question: str) -> tuple[float | None, str]:
    raw = (text or "").strip()
    if not raw:
        return (None, "empty")
    low = raw.lower()
    if any(x in low for x in ("insufficient evidence", "no relevant context", "null", "none")):
        return (None, "missing")

    value = _extract_first_number(raw)
    if value is None:
        return (None, "parse_failed")

    target_unit = _infer_target_unit(question)
    explicit_unit = _infer_explicit_unit(raw, target_unit)

    if target_unit in MONEY_UNIT_TO_DOLLARS:
        explicit_scale = MONEY_UNIT_TO_DOLLARS.get(explicit_unit, MONEY_UNIT_TO_DOLLARS[target_unit])
        target_scale = MONEY_UNIT_TO_DOLLARS[target_unit]
        value = (value * explicit_scale) / target_scale
    elif target_unit == "percent":
        value = value
    else:
        value = value

    return (value, target_unit)


def _within_tolerance(
    pred_value: float,
    gold_value: float,
    *,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    diff = abs(pred_value - gold_value)
    limit = max(abs_tol, rel_tol * max(abs(gold_value), 1.0))
    return diff <= limit


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def main() -> None:
    args = parse_args()

    pred_path = Path(args.pred_path)
    qa_path = Path(args.qa_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_rows = _read_json(pred_path)
    if not isinstance(pred_rows, list):
        raise ValueError(f"Prediction file must be a JSON array: {pred_path}")

    qa_rows = _read_jsonl(qa_path)
    gold_by_q = _gold_index(qa_rows)

    summary: dict[str, Any] = {
        "pred_path": str(pred_path),
        "qa_path": str(qa_path),
        "relative_tolerance": args.relative_tolerance,
        "absolute_tolerance": args.absolute_tolerance,
        "modes": {},
    }

    for mode in args.modes:
        total = 0
        missing_gold = 0
        non_metrics = 0
        used = 0
        empty_pred = 0
        parsed_pred = 0
        parsed_gold = 0
        exact_numeric = 0
        tolerance_match = 0
        detail_rows: list[dict[str, Any]] = []

        for pred in pred_rows:
            if pred.get("mode") != mode:
                continue
            total += 1

            question = str(pred.get("question", "")).strip()
            if not question:
                continue

            gold = gold_by_q.get(question)
            if not gold:
                missing_gold += 1
                continue

            if str(gold.get("gold_type") or "") != "metrics-generated":
                non_metrics += 1
                continue

            pred_answer = _extract_pred_answer(pred.get("answer"))
            if not pred_answer:
                empty_pred += 1
                if args.skip_empty_answer:
                    continue

            gold_answer = "" if gold.get("gold_answer") is None else str(gold.get("gold_answer"))
            pred_value, pred_unit = _normalize_numeric(pred_answer, question)
            gold_value, gold_unit = _normalize_numeric(gold_answer, question)

            used += 1
            if pred_value is not None:
                parsed_pred += 1
            if gold_value is not None:
                parsed_gold += 1

            is_exact = False
            is_tolerance = False
            if pred_value is not None and gold_value is not None:
                is_exact = abs(pred_value - gold_value) <= 1e-9
                is_tolerance = _within_tolerance(
                    pred_value,
                    gold_value,
                    rel_tol=args.relative_tolerance,
                    abs_tol=args.absolute_tolerance,
                )
                exact_numeric += int(is_exact)
                tolerance_match += int(is_tolerance)

            detail_rows.append(
                {
                    "mode": mode,
                    "question": question,
                    "pred_answer": pred_answer,
                    "gold_answer": gold_answer,
                    "pred_value": pred_value,
                    "gold_value": gold_value,
                    "pred_unit": pred_unit,
                    "gold_unit": gold_unit,
                    "exact_numeric_match": is_exact,
                    "tolerance_match": is_tolerance,
                    "error": pred.get("error"),
                }
            )

        mode_summary = {
            "total_pred_rows": total,
            "used_rows": used,
            "missing_gold": missing_gold,
            "non_metrics_skipped": non_metrics,
            "empty_prediction": empty_pred,
            "parsed_prediction": parsed_pred,
            "parsed_gold": parsed_gold,
            "parse_coverage": round(_safe_div(parsed_pred, used), 6),
            "exact_numeric_match": exact_numeric,
            "exact_numeric_accuracy": round(_safe_div(exact_numeric, used), 6),
            "tolerance_match": tolerance_match,
            "tolerance_accuracy": round(_safe_div(tolerance_match, used), 6),
        }
        summary["modes"][mode] = mode_summary

        (out_dir / f"metrics_numeric_{mode}_details.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in detail_rows) + ("\n" if detail_rows else ""),
            encoding="utf-8",
        )
        (out_dir / f"metrics_numeric_{mode}_summary.json").write_text(
            json.dumps(mode_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(
            f"[done] mode={mode} used={used} parsed={parsed_pred}/{used} "
            f"exact_acc={mode_summary['exact_numeric_accuracy']:.4f} "
            f"tol_acc={mode_summary['tolerance_accuracy']:.4f}"
        )

    (out_dir / "metrics_numeric_summary_all_modes.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
