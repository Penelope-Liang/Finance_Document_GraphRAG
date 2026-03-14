#!/usr/bin/env python3
"""Lightweight evaluation for query_router results using EM + token-level F1."""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate rag/graphrag predictions with Exact Match and token-level F1."
    )
    parser.add_argument(
        "--pred-path",
        default="output/finder/pred_50_compare.json",
        help="Prediction JSON path from query_router --mode compare.",
    )
    parser.add_argument(
        "--qa-path",
        default="output/finder/qa_50.jsonl",
        help="Gold QA JSONL path (must include question and gold_answer).",
    )
    parser.add_argument(
        "--out-dir",
        default="output/finder/f1",
        help="Output directory for EM/F1 artifacts.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["rag", "graphrag"],
        choices=["rag", "graphrag"],
        help="Which modes to evaluate.",
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
    t = re.sub(r"^```json\\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^```\\s*", "", t)
    t = re.sub(r"\\s*```$", "", t)
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

    return text


def _normalize_text(s: str) -> str:
    text = s.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    gold_tokens = _normalize_text(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def _exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if _normalize_text(prediction) == _normalize_text(ground_truth) else 0.0


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
        "modes": {},
    }

    for mode in args.modes:
        total = 0
        missing_gold = 0
        used = 0
        empty_pred = 0
        sum_f1 = 0.0
        sum_em = 0.0

        by_type: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "f1": 0.0, "em": 0.0})
        detail_rows: list[dict[str, Any]] = []

        for pred in pred_rows:
            if pred.get("mode") != mode:
                continue
            total += 1

            q = str(pred.get("question", "")).strip()
            if not q:
                continue

            gold = gold_by_q.get(q)
            if not gold:
                missing_gold += 1
                continue

            pred_answer = _extract_pred_answer(pred.get("answer"))
            if not pred_answer:
                empty_pred += 1
                if args.skip_empty_answer:
                    continue

            gold_answer = "" if gold.get("gold_answer") is None else str(gold.get("gold_answer"))
            gold_type = str(gold.get("gold_type") or "UNKNOWN")

            f1 = _token_f1(pred_answer, gold_answer)
            em = _exact_match(pred_answer, gold_answer)

            used += 1
            sum_f1 += f1
            sum_em += em

            by_type[gold_type]["n"] += 1
            by_type[gold_type]["f1"] += f1
            by_type[gold_type]["em"] += em

            detail_rows.append(
                {
                    "mode": mode,
                    "question": q,
                    "pred_answer": pred_answer,
                    "gold_answer": gold_answer,
                    "gold_type": gold_type,
                    "f1": round(f1, 6),
                    "em": round(em, 6),
                    "error": pred.get("error"),
                }
            )

        by_type_summary = {
            t: {
                "n": int(v["n"]),
                "avg_f1": round(_safe_div(v["f1"], v["n"]), 6),
                "avg_em": round(_safe_div(v["em"], v["n"]), 6),
            }
            for t, v in sorted(by_type.items(), key=lambda kv: kv[0])
        }

        mode_summary = {
            "total_pred_rows": total,
            "used_rows": used,
            "missing_gold": missing_gold,
            "empty_prediction": empty_pred,
            "avg_f1": round(_safe_div(sum_f1, used), 6),
            "avg_em": round(_safe_div(sum_em, used), 6),
            "by_type": by_type_summary,
        }
        summary["modes"][mode] = mode_summary

        (out_dir / f"f1_{mode}_details.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in detail_rows) + ("\n" if detail_rows else ""),
            encoding="utf-8",
        )
        (out_dir / f"f1_{mode}_summary.json").write_text(
            json.dumps(mode_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(
            f"[done] mode={mode} used={used} avg_f1={mode_summary['avg_f1']:.4f} "
            f"avg_em={mode_summary['avg_em']:.4f}"
        )

    (out_dir / "f1_summary_all_modes.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
