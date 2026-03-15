#!/usr/bin/env python3
"""Build question->doc_ids mapping for query.router --question-doc-map."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _tokens(text: str) -> set[str]:
    return set(_norm(text).split())


def _extract_years(text: str) -> set[str]:
    return set(re.findall(r"20\d{2}", text))


def _company_tokens_from_doc_id(doc_id: str) -> list[str]:
    # Examples:
    # 3M_2018_10K.pdf -> ["3m"]
    # JOHNSON_JOHNSON_2022Q4_EARNINGS.pdf -> ["johnson","johnson"]
    key = doc_id.strip().lower()
    if key.endswith(".pdf"):
        key = key[:-4]
    parts = [p for p in key.split("_") if p]
    company: list[str] = []
    for p in parts:
        if re.search(r"20\d{2}", p):
            break
        company.append(p)
    if not company and parts:
        company = [parts[0]]
    return company


def _doc_years(doc_id: str) -> set[str]:
    key = doc_id.strip().lower()
    return _extract_years(key)


def _pick_doc_ids_for_question(
    question: str,
    docs: list[str],
    max_docs: int,
) -> list[str]:
    q_norm = _norm(question)
    q_tokens = _tokens(question)
    q_years = _extract_years(question)

    scored: list[tuple[float, str]] = []
    for doc_id in docs:
        c_tokens = _company_tokens_from_doc_id(doc_id)
        c_phrase = " ".join(c_tokens).strip()
        if not c_tokens:
            continue

        overlap = len(set(c_tokens) & q_tokens)
        if overlap == 0 and c_phrase not in q_norm:
            continue

        score = 0.0
        score += overlap / max(1, len(set(c_tokens)))
        if c_phrase and c_phrase in q_norm:
            score += 1.5

        d_years = _doc_years(doc_id)
        if q_years:
            inter = len(q_years & d_years)
            if inter > 0:
                score += 1.0 + 0.3 * inter
            else:
                score -= 0.5

        scored.append((score, doc_id))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)

    selected: list[str] = []
    for score, doc_id in scored:
        if score <= 0:
            continue
        selected.append(doc_id)
        if len(selected) >= max_docs:
            break

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build question->doc_ids map from QA questions and FAISS mapping doc_id."
    )
    parser.add_argument(
        "--qa-path",
        default="output/financebench/qa_metrics.jsonl",
        help="QA JSONL path containing questions.",
    )
    parser.add_argument(
        "--mapping-path",
        default="output/run_20260313_160938/intid_to_chunkid.jsonl",
        help="FAISS mapping JSONL path containing doc_id.",
    )
    parser.add_argument(
        "--out-path",
        default="output/financebench/question_doc_map_metrics.jsonl",
        help="Output JSONL path for --question-doc-map.",
    )
    parser.add_argument(
        "--max-docs-per-question",
        type=int,
        default=5,
        help="Max number of candidate docs to keep per question.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qa_rows = _read_jsonl(Path(args.qa_path))
    map_rows = _read_jsonl(Path(args.mapping_path))

    doc_ids = sorted(
        {
            str(r.get("doc_id")).strip()
            for r in map_rows
            if isinstance(r.get("doc_id"), str) and str(r.get("doc_id")).strip()
        }
    )

    out_rows: list[dict[str, Any]] = []
    matched = 0
    for row in qa_rows:
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        doc_ids_for_q = _pick_doc_ids_for_question(
            question=question,
            docs=doc_ids,
            max_docs=max(1, int(args.max_docs_per_question)),
        )
        if doc_ids_for_q:
            matched += 1
        out_rows.append(
            {
                "question": question,
                "doc_ids": doc_ids_for_q,
            }
        )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Done.")
    print(f"QA rows:              {len(qa_rows)}")
    print(f"Unique doc_ids:       {len(doc_ids)}")
    print(f"Questions with match: {matched}")
    print(f"Out:                  {out_path.resolve()}")


if __name__ == "__main__":
    main()

