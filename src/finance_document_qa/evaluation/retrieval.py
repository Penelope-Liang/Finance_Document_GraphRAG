import argparse
import asyncio
import json
import math
import os
from pathlib import Path

from finance_document_qa.evaluation.ragas_components import build_ragas_azure_components


def load_rows(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def context_text(context):
    if isinstance(context, str):
        return context.strip()
    if isinstance(context, dict):
        return str(context.get("text") or "").strip()
    return str(context or "").strip()


def get_retrieved_contexts(row, max_contexts, max_context_chars):
    contexts = row.get("retrieved_contexts") or []
    if not isinstance(contexts, list):
        return []

    output = []
    for context in contexts[:max_contexts]:
        text = context_text(context)
        if not text:
            continue
        if max_context_chars > 0:
            text = text[:max_context_chars]
        output.append(text)
    return output


def clean_score(value):
    if value is None:
        return None
    result = getattr(value, "value", value)
    if isinstance(result, float) and math.isnan(result):
        return None
    try:
        number = float(result)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


async def score_context_recall(scorer, question, gold_answer, retrieved):
    return await scorer.ascore(
        user_input=question,
        reference=gold_answer,
        retrieved_contexts=retrieved,
    )


async def score_context_precision(scorer, question, gold_answer, retrieved):
    return await scorer.ascore(
        user_input=question,
        reference=gold_answer,
        retrieved_contexts=retrieved,
    )


async def score_context_relevance(scorer, question, retrieved):
    return await scorer.ascore(
        user_input=question,
        retrieved_contexts=retrieved,
    )


async def evaluate_samples(samples, judge_model, temperature):
    try:
        from ragas.metrics.collections import (
            ContextPrecision,
            ContextRecall,
            ContextRelevance,
        )
    except ImportError as exc:
        raise ImportError("Retrieval evaluation requires ragas.") from exc

    llm = build_ragas_azure_components(judge_model, temperature=temperature)
    context_recall_scorer = ContextRecall(llm=llm)
    context_precision_scorer = ContextPrecision(llm=llm)
    context_relevance_scorer = ContextRelevance(llm=llm)

    results = []
    for sample in samples:
        context_recall_score = await score_context_recall(
            context_recall_scorer,
            question=sample["question"],
            gold_answer=sample["gold_answer"],
            retrieved=sample["retrieved_contexts"],
        )
        context_precision_score = await score_context_precision(
            context_precision_scorer,
            question=sample["question"],
            gold_answer=sample["gold_answer"],
            retrieved=sample["retrieved_contexts"],
        )
        context_relevance_score = await score_context_relevance(
            context_relevance_scorer,
            question=sample["question"],
            retrieved=sample["retrieved_contexts"],
        )
        results.append(
            {
                "context_recall": clean_score(context_recall_score),
                "context_precision": clean_score(context_precision_score),
                "context_relevance": clean_score(context_relevance_score),
            }
        )
    return results


def selected_rows(rows, question_ids, limit):
    selected = [
        (idx, row)
        for idx, row in enumerate(rows)
        if not question_ids or str(row.get("question_id")) in question_ids
    ]
    if limit > 0:
        selected = selected[:limit]
    return selected


def mean_metric(evaluations, key):
    values = [item.get(key) for item in evaluations if item.get(key) is not None]
    return sum(values) / len(values) if values else None


def summary(rows):
    evaluations = [
        row.get("retrieval_evaluation") or {}
        for row in rows
        if row.get("retrieval_evaluation") is not None
    ]
    return {
        "evaluated": len(evaluations),
        "mean_context_recall": mean_metric(evaluations, "context_recall"),
        "mean_context_precision": mean_metric(evaluations, "context_precision"),
        "mean_context_relevance": mean_metric(evaluations, "context_relevance"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval context recall, precision, and relevance with RAGAS metrics."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-contexts", type=int, default=10)
    parser.add_argument("--max-context-chars", type=int, default=2000)
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        or os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
        or "gpt-4.1-mini",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", ""),
        help="Ignored: retrieval metrics do not use embeddings. Kept for CLI compatibility.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--question-id", action="append", default=[])
    args = parser.parse_args()

    rows = load_rows(args.input)
    samples = []
    row_indices = []

    for row_index, row in selected_rows(rows, set(args.question_id), args.limit):
        if row.get("retrieval_evaluation") and not args.overwrite:
            continue

        question = str(row.get("question") or "").strip()
        gold_answer = str(row.get("gold_answer") or "").strip()
        retrieved = get_retrieved_contexts(
            row,
            max_contexts=args.max_contexts,
            max_context_chars=args.max_context_chars,
        )
        row["retrieval_evaluation"] = {
            "context_recall": None,
            "context_precision": None,
            "context_relevance": None,
            "context_recall_metric": "ContextRecall",
            "context_precision_metric": "ContextPrecision",
            "context_relevance_metric": "ContextRelevance",
            "judge_model": args.judge_model,
            "retrieved_contexts_used": len(retrieved),
        }

        notes = []
        if not question:
            notes.append("question is missing")
        if not gold_answer:
            notes.append("gold_answer is missing")
        if not retrieved:
            notes.append("retrieved_contexts is missing")
        if notes:
            row["retrieval_evaluation"]["note"] = "; ".join(notes)
            continue

        row_indices.append(row_index)
        samples.append(
            {
                "question": question,
                "gold_answer": gold_answer,
                "retrieved_contexts": retrieved,
            }
        )

    if samples:
        scores = asyncio.run(
            evaluate_samples(
                samples,
                judge_model=args.judge_model,
                temperature=args.temperature,
            )
        )
        for row_index, score in zip(row_indices, scores, strict=True):
            rows[row_index]["retrieval_evaluation"].update(score)

    save_rows(args.output, rows)
    print(json.dumps(summary(rows), indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Saved retrieval evaluation to {args.output}")


if __name__ == "__main__":
    main()
