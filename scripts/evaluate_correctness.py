from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?|%")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _normalize_for_tokens(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.lower()
    text = text.replace("$", " ")
    text = text.replace(",", "")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("%", " % ")
    return text


def _tokenize(value: Any) -> list[str]:
    return TOKEN_RE.findall(_normalize_for_tokens(value))


def answer_recall(prediction: Any, reference: Any) -> float | None:
    """Token-level recall of reference-answer tokens covered by the prediction."""
    reference_tokens = _tokenize(reference)
    if not reference_tokens:
        return None
    prediction_tokens = _tokenize(prediction)
    if not prediction_tokens:
        return 0.0

    reference_counts = Counter(reference_tokens)
    prediction_counts = Counter(prediction_tokens)
    overlap = sum((reference_counts & prediction_counts).values())
    return overlap / sum(reference_counts.values())


def _build_ragas_azure_components(
    *,
    chat_deployment: str,
    embedding_deployment: str,
    temperature: float,
) -> tuple[Any, Any]:
    try:
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise ImportError(
            "RAGAS correctness evaluation requires: pip install ragas langchain-openai datasets"
        ) from exc

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    if not endpoint:
        raise ValueError("Missing AZURE_OPENAI_ENDPOINT.")
    if not api_key:
        raise ValueError("Missing AZURE_OPENAI_API_KEY.")

    chat_model = AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=chat_deployment,
        temperature=temperature,
    )
    embedding_model = AzureOpenAIEmbeddings(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=embedding_deployment,
    )
    return LangchainLLMWrapper(chat_model), LangchainEmbeddingsWrapper(embedding_model)


def _evaluate_ragas_correctness(
    samples: list[dict[str, str]],
    *,
    chat_deployment: str,
    embedding_deployment: str,
    temperature: float,
) -> list[float | None]:
    if not samples:
        return []

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_correctness
    except ImportError as exc:
        raise ImportError(
            "RAGAS correctness evaluation requires: pip install ragas langchain-openai datasets"
        ) from exc

    llm, embeddings = _build_ragas_azure_components(
        chat_deployment=chat_deployment,
        embedding_deployment=embedding_deployment,
        temperature=temperature,
    )
    dataset = Dataset.from_list(samples)
    result = evaluate(
        dataset,
        metrics=[answer_correctness],
        llm=llm,
        embeddings=embeddings,
    )
    frame = result.to_pandas()
    if "answer_correctness" not in frame:
        raise KeyError("RAGAS result does not contain answer_correctness.")
    scores: list[float | None] = []
    for value in frame["answer_correctness"].tolist():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            scores.append(None)
        else:
            scores.append(float(value))
    return scores


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


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [
        row.get("correctness_evaluation") or {}
        for row in rows
        if row.get("correctness_evaluation") is not None
    ]
    cc_values = [
        item.get("correctness_coefficient")
        for item in evaluated
        if item.get("correctness_coefficient") is not None
    ]
    recall_values = [
        item.get("answer_recall")
        for item in evaluated
        if item.get("answer_recall") is not None
    ]
    return {
        "evaluated": len(evaluated),
        "mean_correctness_coefficient": (
            sum(cc_values) / len(cc_values) if cc_values else None
        ),
        "mean_answer_recall": (
            sum(recall_values) / len(recall_values) if recall_values else None
        ),
        "missing_answers": sum(1 for row in rows if not row.get("answer")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FinanceBench answers with token recall and RAGAS answer_correctness."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        or os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
        or "gpt-4.1-mini",
        help="Azure OpenAI chat deployment used by RAGAS answer_correctness.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", ""),
        help="Azure OpenAI embedding deployment used by RAGAS answer_correctness.",
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
    parser.add_argument(
        "--recall-only",
        action="store_true",
        help="Compute only token-level answer recall without calling RAGAS.",
    )
    args = parser.parse_args()

    rows = _load_rows(args.input)
    selected = _selected_rows(
        rows,
        question_ids=set(args.question_id),
        limit=args.limit,
    )

    ragas_samples: list[dict[str, str]] = []
    ragas_row_indices: list[int] = []

    for row_index, row in selected:
        existing = row.get("correctness_evaluation")
        if existing and not args.overwrite:
            continue

        prediction = row.get("answer")
        reference = row.get("gold_answer") or row.get("reference_answer")
        recall = answer_recall(prediction, reference)

        row["correctness_evaluation"] = {
            "answer_recall": recall,
            "correctness_coefficient": None,
            "metric_backend": "ragas.answer_correctness",
            "judge_model": args.judge_model,
            "embedding_model": args.embedding_model or None,
        }

        if not prediction:
            row["correctness_evaluation"]["correctness_coefficient"] = 0.0
            row["correctness_evaluation"]["note"] = "Candidate answer is missing."
            continue

        if args.recall_only:
            continue

        if not reference:
            row["correctness_evaluation"]["note"] = "Reference answer is missing."
            continue

        ragas_row_indices.append(row_index)
        ragas_samples.append(
            {
                "user_input": str(row.get("question") or ""),
                "response": str(prediction),
                "reference": str(reference),
            }
        )

    if ragas_samples:
        if not args.embedding_model:
            raise ValueError(
                "Missing --embedding-model or AZURE_OPENAI_EMBEDDING_DEPLOYMENT. "
                "RAGAS answer_correctness requires embeddings."
            )
        scores = _evaluate_ragas_correctness(
            ragas_samples,
            chat_deployment=args.judge_model,
            embedding_deployment=args.embedding_model,
            temperature=args.temperature,
        )
        for row_index, score in zip(ragas_row_indices, scores, strict=True):
            rows[row_index]["correctness_evaluation"][
                "correctness_coefficient"
            ] = score

    _save_rows(args.output, rows)
    summary = _summary(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Saved correctness results to {args.output}")


if __name__ == "__main__":
    main()
