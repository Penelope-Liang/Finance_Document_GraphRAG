#!/usr/bin/env python3
"""Evaluate retrieval-only outputs against FinanceBench gold doc/page evidence."""

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


def _iter_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        return [obj] if isinstance(obj, dict) else []
    return _iter_jsonl(path)


def _doc_key(value: str) -> str:
    key = value.strip().lower()
    return key[:-4] if key.endswith(".pdf") else key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute doc/page retrieval hit-rate for retrieval-only run_compare outputs."
        )
    )
    parser.add_argument("--pred-path", required=True, help="Path to retrieval-only predictions JSON/JSONL.")
    parser.add_argument("--qa-path", required=True, help="Path to QA JSONL with doc_name and gold_pages.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Evaluate hits within top-k retrieved evidence rows.",
    )
    parser.add_argument(
        "--mode",
        default="rag",
        help="Optional mode filter (e.g. rag or graphrag). Empty means no filter.",
    )
    parser.add_argument(
        "--out-path",
        help="Optional JSON path for summary + per-question details.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred_path).expanduser().resolve()
    qa_path = Path(args.qa_path).expanduser().resolve()

    qa_rows = _iter_jsonl(qa_path)
    qa_by_question = {str(r.get("question", "")).strip(): r for r in qa_rows if r.get("question")}
    pred_rows = _iter_json_or_jsonl(pred_path)
    if args.mode:
        pred_rows = [r for r in pred_rows if str(r.get("mode", "")) == args.mode]

    details: list[dict[str, Any]] = []
    doc_hit = 0
    page_hit = 0
    used = 0

    for pred in pred_rows:
        question = str(pred.get("question", "")).strip()
        if not question or question not in qa_by_question:
            continue
        gold = qa_by_question[question]
        gold_doc = _doc_key(str(gold.get("doc_name", "")))
        gold_pages = {int(x) for x in (gold.get("gold_pages") or []) if x is not None}
        retrieved = list(pred.get("retrieved_evidence") or [])[: max(0, int(args.top_k))]
        hit_doc = False
        hit_page = False
        for ev in retrieved:
            doc_id = _doc_key(str(ev.get("doc_id", "")))
            page = ev.get("page")
            if doc_id == gold_doc:
                hit_doc = True
                if page is not None and gold_pages and int(page) in gold_pages:
                    hit_page = True
                    break
        used += 1
        doc_hit += int(hit_doc)
        page_hit += int(hit_page)
        details.append(
            {
                "question": question,
                "question_id": gold.get("_id"),
                "gold_doc_name": gold.get("doc_name"),
                "gold_pages": sorted(gold_pages),
                "hit_doc": hit_doc,
                "hit_page": hit_page,
                "top_k": args.top_k,
                "retrieval_status": pred.get("retrieval_status"),
                "retrieved_evidence": retrieved,
            }
        )

    summary = {
        "pred_path": str(pred_path),
        "qa_path": str(qa_path),
        "mode": args.mode,
        "top_k": args.top_k,
        "used": used,
        "doc_hit": doc_hit,
        "page_hit": page_hit,
        "doc_hit_rate": round(doc_hit / used, 4) if used else 0.0,
        "page_hit_rate": round(page_hit / used, 4) if used else 0.0,
    }

    if args.out_path:
        out_path = Path(args.out_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
