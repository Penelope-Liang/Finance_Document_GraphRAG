from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from neo4j_graphrag.llm import OllamaLLM

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for ad-hoc runs.
    tqdm = None


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
    reference_justification: Any = None,
    evidence: str = "",
) -> str:
    evidence_section = ""
    if evidence.strip():
        evidence_section = f"""
###Retrieved Evidence:
Evidence is optional auxiliary context only. Do not penalize the response for lacking citations
or explicit supporting evidence. Use evidence only to resolve ambiguity or detect contradiction.
{evidence}
"""
    return f"""###Task Description:
You are evaluating a FinanceBench answer. You are given an instruction/question,
a candidate response, a reference answer, and optionally a reference justification.

Judge whether the candidate response correctly answers the question in a good-faith
financial sense. Focus on financial substance, not wording or formatting.

First infer the answer type from the question:
1. Direct extraction:
   The question asks for a specific fact, entity, amount, date, filing item, segment, or line item.
2. Numerical / calculation reasoning:
   The question asks for a ratio, growth rate, margin, turnover, percentage change, comparison,
   increase/decrease, or other computed result.
3. Qualitative / logical finance judgment:
   The question asks for business interpretation, financial health judgment, metric applicability,
   trend explanation, risk explanation, or a yes/no conclusion based on financial evidence.
4. Open-ended multi-step reasoning:
   The question requires combining multiple facts across sections, periods, tables, or narrative disclosures.

Evaluation rules:
- Do not require citations, evidence references, or full calculation steps.
- A concise answer can be fully correct if it gives the correct final fact, amount, entity, direction, or conclusion.
- Allow equivalent wording, common financial abbreviations, equivalent units, and minor rounding differences.
- For numerical/calculation questions, the sign, direction, unit scale, metric, and period must be correct.
- For direct extraction questions, the requested entity/fact/amount must match the reference.
- For qualitative/logical questions, accept different wording or a different-but-valid explanation if the conclusion is consistent with the reference answer and not contradicted by it.
- For open-ended questions, the response must cover the main required components; missing important components should reduce the score.
- If the candidate includes extra reasoning, do not penalize it unless it contradicts the reference answer or changes the conclusion.
- If the reference justification is provided, use it only to clarify the reference answer; do not require identical wording.

Common failure cases:
- wrong company
- wrong fiscal year, quarter, period, or filing
- wrong sign or direction of change
- wrong entity, segment, metric, or line item
- amount off by a large order of magnitude
- missing a required component of a multi-part question
- answer is unsupported by or contradicted by the reference answer
- candidate says it cannot answer when the reference answer provides the answer

###Reference Answer:
{reference_answer}

###Reference Justification:
{reference_justification or "N/A"}

###Instruction:
{question}

###Response:
{response}
{evidence_section}
###Score Rubric:
5: Fully correct in financial substance. Equivalent wording, equivalent units, concise answers, and minor rounding differences are acceptable.
4: Mostly correct; the main answer is correct but there is a minor omission, minor imprecision, or non-critical extra detail.
3: Partially correct; contains some correct relevant information but misses an important required component or has a notable but not fatal error.
2: Mostly incorrect; only a small part is relevant/correct, or it refuses/claims missing information despite the reference answer being available.
1: Incorrect, hallucinated, wrong company/period/entity/metric, wrong direction/sign, contradicted by the reference, or missing candidate answer.

You must output exactly this format:
###Rationale:
<brief explanation>

###Score:
<one integer from 1 to 5>"""


def _build_ollama_judge(model_name: str) -> Any:
    host = os.environ.get("OLLAMA_HOST")
    ollama_kwargs = {"host": host} if host else {}
    return OllamaLLM(
        model_name=model_name,
        model_params={"options": {"temperature": 0}},
        **ollama_kwargs,
    )


def _invoke_judge(
    *,
    judge: Any,
    prompt: str,
    llm_as_judge: str,
    max_tokens: int,
) -> str:
    if llm_as_judge == "ollama":
        return judge.invoke(prompt).content

    response = judge.chat.completions.create(
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": "You are a precise FinanceBench evaluator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_completion_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def _build_azure_openai_judge() -> Any:
    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise ImportError("Azure OpenAI judge requires `pip install openai`.") from exc

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    if not endpoint:
        raise ValueError("Missing AZURE_OPENAI_ENDPOINT.")
    if not api_key:
        raise ValueError("Missing AZURE_OPENAI_API_KEY.")

    return AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge existing FinanceBench QA results with an LLM-as-judge.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--llm-as-judge",
        choices=["ollama", "azure-openai"],
        default=os.environ.get("LLM_AS_JUDGE", "ollama"),
        help="Judge backend. Use azure-openai for Azure OpenAI deployments.",
    )
    parser.add_argument("--judge-model", default=os.environ.get("OLLAMA_JUDGE_MODEL", "vicgalle/prometheus-7b-v2.0:latest"))
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum judge completion tokens for Azure OpenAI.",
    )
    parser.add_argument("--max-evidence-chars", type=int, default=6000)
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="Do not include retrieved evidence in the judge prompt (faster, recommended for correctness-only evaluation).",
    )
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output.")
    parser.add_argument(
        "--question-id",
        action="append",
        default=[],
        help="Question ID to judge. Can be passed multiple times. Defaults to all rows.",
    )
    args = parser.parse_args()

    if args.llm_as_judge == "ollama":
        judge = _build_ollama_judge(args.judge_model)
        judge_model = args.judge_model
    else:
        judge = _build_azure_openai_judge()
        judge_model = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

    rows = _load_rows(args.input)
    selected_ids = set(args.question_id)
    selected_rows = [row for row in rows if not selected_ids or row.get("question_id") in selected_ids]
    if args.limit > 0:
        selected_rows = selected_rows[: args.limit]
    total = len(selected_rows)
    iterator = enumerate(selected_rows, start=1)
    progress = None
    if tqdm is not None and not args.no_progress:
        progress = tqdm(iterator, total=total, desc="Judging", unit="question")
        iterable = progress
    else:
        iterable = iterator

    judged_count = 0
    elapsed_total = 0.0
    for idx, row in iterable:
        item_start = time.perf_counter()
        existing = row.get("llm_judge")
        if existing and existing.get("score") is not None and not args.overwrite:
            if progress is not None:
                progress.set_postfix_str("skipped existing")
            continue

        answer = row.get("answer")
        reference_answer = row.get("gold_answer") or row.get("reference_answer")
        reference_justification = (
            row.get("gold_justification")
            or row.get("justification")
            or row.get("reference_justification")
        )
        question = row.get("question")
        print(f"[{idx}/{total}] {row.get('question_id')} answer={answer!r}")

        if not answer:
            row["llm_judge"] = {
                "score": 1,
                "pass": False,
                "rationale": "Candidate answer is missing.",
                "raw": None,
                "model": judge_model,
                "judge_backend": args.llm_as_judge,
            }
            _save_rows(args.output, rows)
            continue

        evidence = "" if args.no_evidence else _evidence_text(row, args.max_evidence_chars)
        prompt = _judge_prompt(
            question=str(question or ""),
            response=answer,
            reference_answer=reference_answer,
            reference_justification=reference_justification,
            evidence=evidence,
        )
        try:
            raw = _invoke_judge(
                judge=judge,
                prompt=prompt,
                llm_as_judge=args.llm_as_judge,
                max_tokens=args.max_tokens,
            )
            score = _parse_prometheus_score(raw)
            row["llm_judge"] = {
                "score": score,
                "pass": bool(score is not None and score >= 4),
                "rationale": _extract_rationale(raw),
                "raw": raw,
                "model": judge_model,
                "judge_backend": args.llm_as_judge,
            }
            print(f"  score={score} pass={row['llm_judge']['pass']}")
        except Exception as exc:
            row["llm_judge"] = {
                "score": None,
                "pass": None,
                "rationale": str(exc),
                "raw": None,
                "model": judge_model,
                "judge_backend": args.llm_as_judge,
            }
            print(f"  error={exc}")

        _save_rows(args.output, rows)
        if args.sleep > 0:
            time.sleep(args.sleep)
        elapsed = time.perf_counter() - item_start
        judged_count += 1
        elapsed_total += elapsed
        avg = elapsed_total / judged_count if judged_count else 0.0
        timing = f"last={elapsed:.1f}s avg={avg:.1f}s judged={judged_count}/{total}"
        if progress is not None:
            progress.set_postfix_str(timing)
        else:
            print(f"  {timing}")

    _save_rows(args.output, rows)
    print(f"Saved judged results to {args.output}")


if __name__ == "__main__":
    main()
