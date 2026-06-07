import argparse
import json
import math
import os
import re
import string
from collections import Counter
from pathlib import Path

from ragas_components import build_ragas_azure_components


ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def _load_rows(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def normalize_evidence(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return ARTICLES_REGEX.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s or "")))))


def get_tokens(s):
    if not s:
        return []
    return normalize_evidence(s).split()


def context_token_recall_at_k(gold_evidence, contexts):
    gold_tokens = get_tokens(gold_evidence)
    if not gold_tokens:
        return None
    retrieved_tokens = get_tokens(" ".join(contexts))
    if not retrieved_tokens:
        return 0.0
    common = Counter(gold_tokens) & Counter(retrieved_tokens)
    overlap = sum(common.values())
    return overlap / len(gold_tokens)


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


def _reference_text(row):
    parts = [
        row.get("gold_answer"),
        row.get("gold_justification"),
        row.get("gold_evidence"),
    ]
    return "\n".join(str(part) for part in parts if part)


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


def _init_metric(metric):
    if hasattr(metric, "required_columns"):
        return metric
    if isinstance(metric, type):
        return metric()
    if callable(metric):
        candidate = metric()
        if hasattr(candidate, "required_columns"):
            return candidate
    return metric


def _evaluate_ragas_retrieval(samples, chat_deployment, embedding_deployment, temperature):
    if not samples:
        return []

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics.collections import context_precision, context_recall
    except ImportError as exc:
        raise ImportError("Retrieval evaluation requires: pip install -e '.[evaluation]'") from exc

    llm, embeddings = build_ragas_azure_components(
        chat_deployment=chat_deployment,
        embedding_deployment=embedding_deployment,
        temperature=temperature,
    )
    result = evaluate(
        Dataset.from_list(samples),
        metrics=[_init_metric(context_precision), _init_metric(context_recall)],
        llm=llm,
        embeddings=embeddings,
    )
    frame = result.to_pandas()
    output = []
    for _, row in frame.iterrows():
        output.append(
            {
                "context_precision": _nan_to_none(row.get("context_precision")),
                "context_recall": _nan_to_none(row.get("context_recall")),
            }
        )
    return output


def _evaluate_token_recall(reference, contexts):
    return {
        "context_token_recall": context_token_recall_at_k(reference, contexts),
    }


def _summary(rows):
    evaluations = [
        row.get("retrieval_evaluation") or {}
        for row in rows
        if row.get("retrieval_evaluation") is not None
    ]
    summary = {"evaluated": len(evaluations)}
    for metric in ("context_precision", "context_recall", "context_token_recall"):
        values = [
            item.get(metric)
            for item in evaluations
            if item.get(metric) is not None
        ]
        summary[f"mean_{metric}"] = sum(values) / len(values) if values else None
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval quality with RAGAS context precision/recall "
            "or lexical Context Token Recall@K."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=["ragas", "token_recall"],
        default="ragas",
        help="ragas uses LLM judge metrics; token_recall uses SQuAD-style token overlap.",
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
        help="Azure OpenAI embedding deployment used by RAGAS metrics.",
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
            "context_precision": None,
            "context_recall": None,
            "context_token_recall": None,
            "metric_backend": args.mode,
            "judge_model": args.judge_model if args.mode == "ragas" else None,
            "embedding_model": args.embedding_model if args.mode == "ragas" else None,
            "context_field": args.context_field,
            "contexts_used": len(contexts),
        }

        if args.mode == "token_recall":
            row["retrieval_evaluation"].update(_evaluate_token_recall(reference, contexts))
            continue

        if not contexts:
            row["retrieval_evaluation"]["note"] = "Retrieved contexts are missing."
            continue
        if not reference:
            row["retrieval_evaluation"]["note"] = "Reference text is missing."

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
                "Missing --embedding-model or AZURE_OPENAI_EMBEDDING_DEPLOYMENT."
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
