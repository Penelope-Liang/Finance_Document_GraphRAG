import argparse
import json
import math
import os
from pathlib import Path

from evaluate_correctness import _build_ragas_azure_components


def _load_rows(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _selected_rows(rows, question_ids, limit):
    selected = [
        (idx, row)
        for idx, row in enumerate(rows)
        if not question_ids or str(row.get("question_id")) in question_ids
    ]
    if limit > 0:
        selected = selected[:limit]
    return selected


def _context_text(context):
    if isinstance(context, str):
        return context
    if isinstance(context, dict):
        for key in ("text", "content", "page_content"):
            value = context.get(key)
            if value:
                return str(value)
    return str(context or "")


def _context_texts(row, context_field, max_contexts, max_context_chars):
    contexts = row.get(context_field)
    if contexts is None and context_field != "final_context":
        contexts = row.get("final_context")
    if contexts is None and context_field != "retrieved_contexts":
        contexts = row.get("retrieved_contexts")
    if not isinstance(contexts, list):
        return []

    output = []
    for context in contexts[:max_contexts]:
        text = _context_text(context).strip()
        if not text:
            continue
        if max_context_chars > 0:
            text = text[:max_context_chars]
        output.append(text)
    return output


def _nan_to_none(value):
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


def _evaluate_ragas_answer_quality(samples, chat_deployment, embedding_deployment, temperature):
    if not samples:
        return []

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics.collections import answer_accuracy, faithfulness
    except ImportError as exc:
        raise ImportError("Answer quality evaluation requires: pip install -e '.[evaluation]'") from exc

    llm, embeddings = _build_ragas_azure_components(
        chat_deployment=chat_deployment,
        embedding_deployment=embedding_deployment,
        temperature=temperature,
    )
    result = evaluate(
        Dataset.from_list(samples),
        metrics=[faithfulness, answer_accuracy],
        llm=llm,
        embeddings=embeddings,
    )
    frame = result.to_pandas()
    output = []
    for _, row in frame.iterrows():
        output.append(
            {
                "faithfulness_score": _nan_to_none(row.get("faithfulness")),
                "answer_accuracy": _nan_to_none(row.get("answer_accuracy")),
            }
        )
    return output


def _summary(rows):
    evaluations = [
        row.get("answer_quality_evaluation") or {}
        for row in rows
        if row.get("answer_quality_evaluation") is not None
    ]
    summary = {"evaluated": len(evaluations)}
    for metric in ("faithfulness_score", "answer_accuracy"):
        values = [
            item.get(metric)
            for item in evaluations
            if item.get(metric) is not None
        ]
        summary[f"mean_{metric}"] = sum(values) / len(values) if values else None
    summary["missing_answers"] = sum(1 for row in rows if not row.get("answer"))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated answer quality with RAGAS faithfulness "
            "and RAGAS answer accuracy."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
        help="Azure OpenAI embedding deployment used by RAGAS metrics when required.",
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
    selected = _selected_rows(rows, set(args.question_id), args.limit)

    ragas_samples = []
    ragas_row_indices = []

    for row_index, row in selected:
        if row.get("answer_quality_evaluation") and not args.overwrite:
            continue

        contexts = _context_texts(
            row,
            context_field=args.context_field,
            max_contexts=args.max_contexts,
            max_context_chars=args.max_context_chars,
        )
        answer = row.get("answer")
        row["answer_quality_evaluation"] = {
            "faithfulness_score": None,
            "answer_accuracy": None,
            "metric_backend": "ragas",
            "judge_model": args.judge_model,
            "embedding_model": args.embedding_model,
            "context_field": args.context_field,
            "contexts_used": len(contexts),
        }

        if not answer:
            row["answer_quality_evaluation"]["note"] = "Candidate answer is missing."
            continue
        if not contexts:
            row["answer_quality_evaluation"]["note"] = "Retrieved contexts are missing."
            continue

        ragas_row_indices.append(row_index)
        ragas_samples.append(
            {
                "user_input": str(row.get("question") or ""),
                "response": str(answer),
                "retrieved_contexts": contexts,
                "reference": str(row.get("gold_answer") or ""),
            }
        )

    if ragas_samples:
        if not args.embedding_model:
            raise ValueError(
                "Missing --embedding-model or AZURE_OPENAI_EMBEDDING_DEPLOYMENT."
            )
        scores = _evaluate_ragas_answer_quality(
            ragas_samples,
            chat_deployment=args.judge_model,
            embedding_deployment=args.embedding_model,
            temperature=args.temperature,
        )
        for row_index, score in zip(ragas_row_indices, scores, strict=True):
            rows[row_index]["answer_quality_evaluation"].update(score)

    _save_rows(args.output, rows)
    print(json.dumps(_summary(rows), indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Saved answer quality evaluation to {args.output}")


if __name__ == "__main__":
    main()
