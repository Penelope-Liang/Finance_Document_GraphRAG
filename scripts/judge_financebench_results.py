from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from neo4j_graphrag.llm import OllamaLLM


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = _strip_think(text)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_prometheus_score(raw: str) -> int | None:
    text = _strip_think(raw)
    obj = _extract_json_object(text)
    if obj is not None:
        score = obj.get("score")
        try:
            parsed = int(score)
            return parsed if 1 <= parsed <= 5 else None
        except (TypeError, ValueError):
            pass

    strict_score = re.search(r"###Score:\s*([1-5])\b", text, flags=re.IGNORECASE)
    if strict_score:
        return int(strict_score.group(1))

    rationale_match = re.search(
        r"###Rationale:\s*(.*?)(?:\[END\]\s*)?$",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    score_text = rationale_match.group(1) if rationale_match else text

    patterns = (
        r"\boverall\s+score\s*(?:of|is|:|=)\s*([1-5])\b",
        r"\bscore\s*(?:of|is|:|=)\s*([1-5])\b",
        r"\breceive\s+a\s+score\s+of\s+([1-5])\b",
        r"\bwould\s+receive\s+(?:a\s+)?score\s+of\s+([1-5])\b",
        r"\boverall\s+score\s+is\s+([1-5])\b",
        r"\b([1-5])\s*/\s*5\b",
    )
    for pattern in patterns:
        match = re.search(pattern, score_text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    # Prometheus often emits only a rationale with rubric wording.
    lowered = score_text.lower()
    phrase_scores = (
        ("fully correct", 5),
        ("perfect match", 5),
        ("mostly correct", 4),
        ("essentially correct", 4),
        ("correctly identifies", 4),
        ("partially correct", 3),
        ("mostly incorrect", 2),
        ("candidate answer is missing", 1),
        ("response is missing", 1),
        ("answer is missing", 1),
        ("incorrect", 1),
    )
    for phrase, score in phrase_scores:
        if phrase in lowered:
            return score
    return None


def _extract_rationale(raw: str) -> str:
    text = _strip_think(raw).strip()
    text = re.sub(r"\[END\]\s*$", "", text).strip()
    match = re.search(
        r"###Rationale:\s*(.*?)(?:###Score:|$)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    obj = _extract_json_object(text)
    if obj is not None and obj.get("rationale"):
        return str(obj["rationale"])
    return text[:1200]


def _evidence_text(row: dict[str, Any], max_chars: int) -> str:
    chunks = []
    evidence = row.get("evidence")
    if evidence:
        chunks.append(f"Model evidence:\n{evidence}")
    retriever_result = row.get("retriever_result") or {}
    items = retriever_result.get("items") or []
    per_item = max(500, max_chars // max(1, min(len(items), 8)))
    for idx, item in enumerate(items[:8], start=1):
        content = str(item.get("content") or "")
        chunks.append(f"[{idx}]\n{content[:per_item]}")
    return "\n\n".join(chunks)[:max_chars]


def _judge_prompt(
    *,
    question: str,
    response: Any,
    reference_answer: Any,
    evidence: str,
) -> str:
    return f"""###Task Description:
An instruction, a response to evaluate, a reference answer, and retrieved evidence are given.
Evaluate whether the response correctly answers the instruction compared with the reference answer.
Allow equivalent wording and minor rounding differences. Penalize wrong company, wrong period,
unsupported claims, missing required reasoning, or answers contradicted by the reference/evidence.

###Instruction:
{question}

###Response:
{response}

###Reference Answer:
{reference_answer}

###Retrieved Evidence:
{evidence}

###Score Rubric:
5: Fully correct and adequately supported.
4: Mostly correct with only minor omissions.
3: Partially correct but missing an important detail.
2: Mostly incorrect with only a small relevant part.
1: Incorrect, unsupported, or missing.

You must output exactly this format:
###Rationale:
<brief explanation>

###Score:
<one integer from 1 to 5>

###Rationale:"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge existing FinanceBench QA results with an LLM-as-judge.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--judge-model", default=os.environ.get("OLLAMA_JUDGE_MODEL", "vicgalle/prometheus-7b-v2.0:latest"))
    parser.add_argument("--max-evidence-chars", type=int, default=6000)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--question-id",
        action="append",
        default=[],
        help="Question ID to judge. Can be passed multiple times. Defaults to all rows.",
    )
    args = parser.parse_args()

    host = os.environ.get("OLLAMA_HOST")
    ollama_kwargs = {"host": host} if host else {}
    judge = OllamaLLM(
        model_name=args.judge_model,
        model_params={"options": {"temperature": 0}},
        **ollama_kwargs,
    )

    rows = _load_rows(args.input)
    selected_ids = set(args.question_id)
    selected_rows = [row for row in rows if not selected_ids or row.get("question_id") in selected_ids]
    if args.limit > 0:
        selected_rows = selected_rows[: args.limit]
    total = len(selected_rows)
    for idx, row in enumerate(selected_rows, start=1):
        existing = row.get("llm_judge")
        if existing and existing.get("score") is not None and not args.overwrite:
            continue

        answer = row.get("answer")
        reference_answer = row.get("gold_answer") or row.get("reference_answer")
        question = row.get("question")
        print(f"[{idx}/{total}] {row.get('question_id')} answer={answer!r}")

        if not answer:
            row["llm_judge"] = {
                "score": 1,
                "pass": False,
                "rationale": "Candidate answer is missing.",
                "raw": None,
                "model": args.judge_model,
            }
            _save_rows(args.output, rows)
            continue

        prompt = _judge_prompt(
            question=str(question or ""),
            response=answer,
            reference_answer=reference_answer,
            evidence=_evidence_text(row, args.max_evidence_chars),
        )
        try:
            raw = judge.invoke(prompt).content
            score = _parse_prometheus_score(raw)
            row["llm_judge"] = {
                "score": score,
                "pass": bool(score is not None and score >= 4),
                "rationale": _extract_rationale(raw),
                "raw": raw,
                "model": args.judge_model,
            }
            print(f"  score={score} pass={row['llm_judge']['pass']}")
        except Exception as exc:
            row["llm_judge"] = {
                "score": None,
                "pass": None,
                "rationale": str(exc),
                "raw": None,
                "model": args.judge_model,
            }
            print(f"  error={exc}")

        _save_rows(args.output, rows)
        if args.sleep > 0:
            time.sleep(args.sleep)

    _save_rows(args.output, rows)
    print(f"Saved judged results to {args.output}")


if __name__ == "__main__":
    main()
