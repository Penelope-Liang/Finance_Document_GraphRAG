from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from evaluate_correctness import _build_ragas_azure_components


TOKEN_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?|%")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.lower()
    text = text.replace("$", " ")
    text = text.replace(",", "")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("%", " % ")
    return text


def _tokens(value: Any) -> set[str]:
    return set(TOKEN_RE.findall(_normalize_text(value)))


def _f1(a: Any, b: Any) -> float | None:
    a_tokens = _tokens(a)
    b_tokens = _tokens(b)
    if not a_tokens or not b_tokens:
        return None
    overlap = len(a_tokens & b_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(a_tokens)
    recall = overlap / len(b_tokens)
    return 2 * precision * recall / (precision + recall)


def _recall(covered_by: Any, target: Any) -> float | None:
    source_tokens = _tokens(covered_by)
    target_tokens = _tokens(target)
    if not target_tokens:
        return None
    if not source_tokens:
        return 0.0
    return len(source_tokens & target_tokens) / len(target_tokens)


def _selected_rows(
    rows: list[dict[str, Any]],
    *,
    question_ids: set[str],
    limit: int,
) -> list[tuple[int, dict[str, Any]]]:
    selected = [
        (idx, row)
        for idx, row in enumerate(rows)
        if not question_ids or str(row.get("question_id")) in question_ids
    ]
    if limit > 0:
        selected = selected[:limit]
    return selected


def _context_text(context: Any) -> str:
    if isinstance(context, str):
        return context
    if isinstance(context, dict):
        for key in ("text", "content", "page_content"):
            value = context.get(key)
            if value:
                return str(value)
    return str(context or "")


def _context_texts(
    row: dict[str, Any],
    *,
    context_field: str,
    max_contexts: int,
    max_context_chars: int,
) -> list[str]:
    contexts = row.get(context_field)
    if contexts is None and context_field != "final_context":
        contexts = row.get("final_context")
    if contexts is None and context_field != "retrieved_contexts":
        contexts = row.get("retrieved_contexts")
    if not isinstance(contexts, list):
        return []
    output: list[str] = []
    for context in contexts[:max_contexts]:
        text = _context_text(context).strip()
        if not text:
            continue
        if max_context_chars > 0:
            text = text[:max_context_chars]
        output.append(text)
    return output


def _reference_text(row: dict[str, Any]) -> str:
    parts = [
        row.get("gold_answer"),
        row.get("gold_justification"),
        row.get("gold_evidence"),
    ]
    return "\n".join(str(part) for part in parts if part)


def _nan_to_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _evaluate_ragas_retrieval(samples: list[dict[str, Any]], *, chat_deployment: str,
    embedding_deployment: str, temperature: float) -> list[dict[str, float | None]]:
    if not samples:
        return []

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (answer_relevancy, context_precision, context_recall, faithfulness)
    except ImportError as exc:
        raise ImportError("Retrieval evaluation requires: pip install -e '.[evaluation]'") from exc

    llm, embeddings = _build_ragas_azure_components(
        chat_deployment=chat_deployment,
        embedding_deployment=embedding_deployment,
        temperature=temperature,
    )
    result = evaluate(
        Dataset.from_list(samples),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    frame = result.to_pandas()
    output: list[dict[str, float | None]] = []
    for _, row in frame.iterrows():
        output.append(
            {
                "faithfulness": _nan_to_none(row.get("faithfulness")),
                "answer_relevance": _nan_to_none(
                    row.get("answer_relevancy", row.get("answer_relevance"))
                ),
                "context_precision": _nan_to_none(row.get("context_precision")),
                "context_recall": _nan_to_none(row.get("context_recall")),
            }
        )
    return output


def _evaluate_lexical_proxy(
    *,
    question: Any,
    answer: Any,
    reference: str,
    contexts: list[str],
) -> dict[str, float | None]:
    context_text = "\n".join(contexts)
    relevance_values = [
        score
        for context in contexts
        if (score := _f1(context, reference)) is not None
    ]
    relevant = [score for score in relevance_values if score > 0]
    precision_at_relevant_ranks: list[float] = []
    relevant_so_far = 0
    for rank, score in enumerate(relevance_values, start=1):
        if score > 0:
            relevant_so_far += 1
            precision_at_relevant_ranks.append(relevant_so_far / rank)
    return {
        "faithfulness": _recall(context_text, answer),
        "answer_relevance": _f1(answer, question),
        "context_precision": (
            sum(precision_at_relevant_ranks) / len(precision_at_relevant_ranks)
            if precision_at_relevant_ranks
            else 0.0
        ),
        "context_recall": _recall(context_text, reference),
        "lexical_relevant_contexts": len(relevant),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluations = [
        row.get("retrieval_evaluation") or {}
        for row in rows
        if row.get("retrieval_evaluation") is not None
    ]
    metrics = [
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "context_recall",
    ]
    summary: dict[str, Any] = {"evaluated": len(evaluations)}
    for metric in metrics:
        values = [
            item.get(metric)
            for item in evaluations
            if item.get(metric) is not None
        ]
        summary[f"mean_{metric}"] = sum(values) / len(values) if values else None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RAG output quality with faithfulness, answer relevance, "
            "context precision, and context recall."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=["ragas", "lexical"],
        default="ragas",
        help=(
            "ragas uses LLM/embedding metrics for paper results; lexical is a "
            "cheap smoke-test proxy only."
        ),
    )
    parser.add_argument(
        "--context-field",
        default="retrieved_contexts",
        help="Field containing retrieved contexts. Falls back to final_context.",
    )
    parser.add_argument("--max-contexts", type=int, default=10)
    parser.add_argument("--max-context-chars", type=int, default=2000)
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        or os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
        or "gpt-4.1-mini",
        help="Azure OpenAI chat deployment used by RAGAS metrics.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", ""),
        help="Azure OpenAI embedding deployment used by answer relevance.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--question-id",
        action="append",
        default=[],
        help="Question ID to evaluate. Can be passed multiple times. Defaults to all rows.",
    )
    args = parser.parse_args()

    rows = _load_rows(args.input)
    selected = _selected_rows(
        rows,
        question_ids=set(args.question_id),
        limit=args.limit,
    )

    ragas_samples: list[dict[str, Any]] = []
    ragas_row_indices: list[int] = []

    for row_index, row in selected:
        if row.get("retrieval_evaluation") and not args.overwrite:
            continue

        contexts = _context_texts(
            row,
            context_field=args.context_field,
            max_contexts=args.max_contexts,
            max_context_chars=args.max_context_chars,
        )
        reference = _reference_text(row)
        row["retrieval_evaluation"] = {
            "faithfulness": None,
            "answer_relevance": None,
            "context_precision": None,
            "context_recall": None,
            "metric_backend": args.mode,
            "judge_model": args.judge_model if args.mode == "ragas" else None,
            "embedding_model": args.embedding_model if args.mode == "ragas" else None,
            "context_field": args.context_field,
            "contexts_used": len(contexts),
        }

        if args.mode == "lexical":
            row["retrieval_evaluation"].update(
                _evaluate_lexical_proxy(
                    question=row.get("question"),
                    answer=row.get("answer"),
                    reference=reference,
                    contexts=contexts,
                )
            )
            continue

        if not row.get("answer"):
            row["retrieval_evaluation"]["note"] = "Candidate answer is missing."
            continue
        if not contexts:
            row["retrieval_evaluation"]["note"] = "Retrieved contexts are missing."
            continue
        if not reference:
            row["retrieval_evaluation"]["note"] = (
                "Reference text is missing; context precision/recall may fail."
            )

        ragas_row_indices.append(row_index)
        ragas_samples.append(
            {
                "user_input": str(row.get("question") or ""),
                "response": str(row.get("answer") or ""),
                "retrieved_contexts": contexts,
                "reference": reference,
            }
        )

    if args.mode == "ragas" and ragas_samples:
        if not args.embedding_model:
            raise ValueError(
                "Missing --embedding-model or AZURE_OPENAI_EMBEDDING_DEPLOYMENT. "
                "RAGAS answer_relevancy requires embeddings."
            )
        scores = _evaluate_ragas_retrieval(
            ragas_samples,
            chat_deployment=args.judge_model,
            embedding_deployment=args.embedding_model,
            temperature=args.temperature,
        )
        for row_index, score in zip(ragas_row_indices, scores, strict=True):
            rows[row_index]["retrieval_evaluation"].update(score)

    _save_rows(args.output, rows)
    print(json.dumps(_summary(rows), indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Saved retrieval evaluation to {args.output}")


if __name__ == "__main__":
    main()
