import argparse
import asyncio
import json
import math
import os
import warnings
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


def evidence_text(value):
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        output = []
        for item in value:
            output.extend(evidence_text(item))
        return output
    if isinstance(value, dict):
        for key in ("evidence_text", "text", "content", "evidence", "gold_evidence"):
            texts = evidence_text(value.get(key))
            if texts:
                return texts
        return []
    text = str(value).strip()
    return [text] if text else []


def get_reference_contexts(row):
    return evidence_text(row.get("gold_evidence"))


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


async def score_llm_context_recall(scorer, question, gold_answer, retrieved):
    return await scorer.ascore(
        user_input=question,
        reference=gold_answer,
        retrieved_contexts=retrieved,
    )


async def score_llm_context_precision(scorer, question, gold_answer, retrieved):
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


async def score_nonllm_context_recall(scorer, retrieved, reference_contexts):
    from ragas.dataset_schema import SingleTurnSample

    sample = SingleTurnSample(
        retrieved_contexts=retrieved,
        reference_contexts=reference_contexts,
    )
    return await scorer.single_turn_ascore(sample)


async def score_nonllm_context_precision(scorer, retrieved, reference_contexts):
    from ragas.dataset_schema import SingleTurnSample

    sample = SingleTurnSample(
        retrieved_contexts=retrieved,
        reference_contexts=reference_contexts,
    )
    return await scorer.single_turn_ascore(sample)


async def evaluate_samples(samples, judge_model, temperature):
    try:
        from ragas.metrics.collections import (
            ContextPrecision,
            ContextRecall,
            ContextRelevance,
        )

        # NonLLMContextRecall only lives in the legacy namespace; silence its
        # deprecation warning. It is non-LLM (string distance) and needs no LLM.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.metrics import (
                NonLLMContextPrecisionWithReference,
                NonLLMContextRecall,
            )
    except ImportError as exc:
        raise ImportError("Retrieval evaluation requires ragas.") from exc

    llm = build_ragas_azure_components(judge_model, temperature=temperature)
    llm_recall_scorer = ContextRecall(llm=llm)
    llm_precision_scorer = ContextPrecision(llm=llm)
    context_relevance_scorer = ContextRelevance(llm=llm)
    nonllm_recall_scorer = NonLLMContextRecall()
    nonllm_precision_scorer = NonLLMContextPrecisionWithReference()

    results = []
    for sample in samples:
        llm_score = await score_llm_context_recall(
            llm_recall_scorer,
            question=sample["question"],
            gold_answer=sample["gold_answer"],
            retrieved=sample["retrieved_contexts"],
        )
        llm_precision_score = await score_llm_context_precision(
            llm_precision_scorer,
            question=sample["question"],
            gold_answer=sample["gold_answer"],
            retrieved=sample["retrieved_contexts"],
        )
        context_relevance_score = await score_context_relevance(
            context_relevance_scorer,
            question=sample["question"],
            retrieved=sample["retrieved_contexts"],
        )
        nonllm_recall_score = None
        nonllm_precision_score = None
        if sample["reference_contexts"]:
            nonllm_recall_score = await score_nonllm_context_recall(
                nonllm_recall_scorer,
                retrieved=sample["retrieved_contexts"],
                reference_contexts=sample["reference_contexts"],
            )
            nonllm_precision_score = await score_nonllm_context_precision(
                nonllm_precision_scorer,
                retrieved=sample["retrieved_contexts"],
                reference_contexts=sample["reference_contexts"],
            )
        results.append(
            {
                "llm_context_recall": clean_score(llm_score),
                "llm_context_precision": clean_score(llm_precision_score),
                "context_relevance": clean_score(context_relevance_score),
                "nonllm_context_recall": clean_score(nonllm_recall_score),
                "nonllm_context_precision": clean_score(nonllm_precision_score),
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
        "mean_llm_context_recall": mean_metric(evaluations, "llm_context_recall"),
        "mean_llm_context_precision": mean_metric(evaluations, "llm_context_precision"),
        "mean_context_relevance": mean_metric(evaluations, "context_relevance"),
        "mean_nonllm_context_recall": mean_metric(evaluations, "nonllm_context_recall"),
        "mean_nonllm_context_precision": mean_metric(
            evaluations, "nonllm_context_precision"
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval context recall and precision with LLM and non-LLM RAGAS metrics."
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
        reference_contexts = get_reference_contexts(row)

        row["retrieval_evaluation"] = {
            "llm_context_recall": None,
            "llm_context_precision": None,
            "context_relevance": None,
            "nonllm_context_recall": None,
            "nonllm_context_precision": None,
            "llm_recall_metric": "ContextRecall",
            "llm_precision_metric": "ContextPrecision",
            "context_relevance_metric": "ContextRelevance",
            "nonllm_recall_metric": "NonLLMContextRecall",
            "nonllm_precision_metric": "NonLLMContextPrecisionWithReference",
            "judge_model": args.judge_model,
            "retrieved_contexts_used": len(retrieved),
            "reference_contexts_used": len(reference_contexts),
        }

        notes = []
        if not question:
            notes.append("question is missing")
        if not gold_answer:
            notes.append("gold_answer is missing")
        if not retrieved:
            notes.append("retrieved_contexts is missing")
        if not reference_contexts:
            row["retrieval_evaluation"]["nonllm_note"] = (
                "gold_evidence is missing; non-LLM metrics skipped"
            )
        if notes:
            row["retrieval_evaluation"]["note"] = "; ".join(notes)
            continue

        row_indices.append(row_index)
        samples.append(
            {
                "question": question,
                "gold_answer": gold_answer,
                "retrieved_contexts": retrieved,
                "reference_contexts": reference_contexts,
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
