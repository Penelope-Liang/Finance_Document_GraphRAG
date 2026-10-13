import argparse
import asyncio
import json
import os
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

from finance_document_qa.evaluation.utils import (
    clean_score,
    get_retrieved_contexts,
    load_rows,
    mean_metric,
    save_rows,
    selected_rows,
)
from finance_document_qa.evaluation.ragas_components import build_ragas_azure_components


NUMBER_PATTERN = re.compile(
    r"(?P<prefix>[$€£])?\s*"
    r"(?P<negative_paren>\()?"
    r"(?P<number>[-+]?\d[\d,]*(?:\.\d+)?)"
    r"(?P<negative_paren_close>\))?"
    r"\s*(?P<unit>%|percent|percentage points?|bps|basis points?|millions?|"
    r"billions?|thousands?|bn|mm|m|b)?",
    re.IGNORECASE,
)

YES_NO_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def infer_rounding_decimals(question, gold_answer):
    text = str(question or "").lower()
    match = re.search(r"round(?:ed)?\s+to\s+(?:the\s+)?(\w+|\d+)\s+decimal", text)
    if match:
        word = match.group(1)
        words = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
        }
        if word.isdigit():
            return int(word)
        if word in words:
            return words[word]

    gold_match = NUMBER_PATTERN.search(str(gold_answer or ""))
    if gold_match:
        raw_number = gold_match.group("number").strip("()")
        if "." in raw_number:
            fractional = raw_number.split(".", 1)[1]
            return len(fractional.rstrip("0"))
    return 0


def question_scale(question):
    text = str(question or "").lower()
    if re.search(
        r"\b(?:usd|dollars?)\s+billions?\b|\bbillions?\s+of\s+dollars?\b", text
    ):
        return 1_000_000_000.0
    if re.search(
        r"\b(?:usd|dollars?)\s+millions?\b|\bmillions?\s+of\s+dollars?\b", text
    ):
        return 1_000_000.0
    if re.search(
        r"\b(?:usd|dollars?)\s+thousands?\b|\bthousands?\s+of\s+dollars?\b", text
    ):
        return 1_000.0
    return 1.0


def is_percent_question(question):
    text = str(question or "").lower()
    return bool(re.search(r"\bpercent(?:age)?s?\b|%", text))


def unit_scale(unit):
    normalized = (unit or "").lower()
    if normalized in {"billion", "billions", "bn", "b"}:
        return 1_000_000_000.0
    if normalized in {"million", "millions", "mm", "m"}:
        return 1_000_000.0
    if normalized in {"thousand", "thousands"}:
        return 1_000.0
    return None


def parse_numeric_candidates(text, question):
    """Return comparable numeric candidates from an answer string.

    Explicit million/billion units override the question unit. Otherwise, values
    inherit units requested in the question. Percent questions are compared in
    percentage-point space, while also accepting decimal-ratio equivalents.
    """
    candidates = []
    q_scale = question_scale(question)
    q_is_percent = is_percent_question(question)
    for match in NUMBER_PATTERN.finditer(str(text or "")):
        raw_number = match.group("number")
        try:
            value = float(raw_number.replace(",", ""))
        except ValueError:
            continue
        if match.group("negative_paren") and match.group("negative_paren_close"):
            value = -abs(value)

        unit = match.group("unit") or ""
        unit_lower = unit.lower()
        if unit_lower in {"bps", "basis point", "basis points"}:
            candidates.append(
                {
                    "value": value / 100.0,
                    "raw": match.group(0).strip(),
                    "scale": 1.0,
                }
            )
            continue

        scale = unit_scale(unit)
        if scale is not None:
            normalized = value * scale
        else:
            scale = q_scale
            normalized = value * scale

        variants = [normalized]
        variant_scale = scale
        if q_is_percent or unit_lower in {
            "%",
            "percent",
            "percentage point",
            "percentage points",
        }:
            variants = [value]
            variant_scale = 1.0
            if abs(value) <= 1:
                variants.append(value * 100.0)

        for candidate in variants:
            candidates.append(
                {
                    "value": candidate,
                    "raw": match.group(0).strip(),
                    "scale": variant_scale,
                }
            )
    return candidates


def extract_yes_no(text):
    match = YES_NO_PATTERN.search(str(text or ""))
    return match.group(1).lower() if match else None


def normalize_text(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def round_for_answer_comparison(value, decimals, scale):
    """Round a normalized value at the answer's requested precision."""
    try:
        decimal_value = Decimal(str(value))
        quantum = Decimal(str(scale)).normalize() * (Decimal("1").scaleb(-decimals))
        return decimal_value.quantize(quantum, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return None


def deterministic_answer_evaluation(question, answer, gold_answer):
    gold_yes_no = extract_yes_no(gold_answer)
    answer_yes_no = extract_yes_no(answer)
    if gold_yes_no:
        return {
            "deterministic_answer_accuracy": 1.0
            if answer_yes_no == gold_yes_no
            else 0.0,
            "deterministic_answer_match": answer_yes_no == gold_yes_no,
            "deterministic_answer_type": "yes_no",
            "deterministic_answer_reason": (
                f"answer={answer_yes_no}; gold={gold_yes_no}"
                if answer_yes_no
                else "could not extract yes/no from answer"
            ),
        }

    answer_numbers = parse_numeric_candidates(answer, question)
    gold_numbers = parse_numeric_candidates(gold_answer, question)
    if answer_numbers and gold_numbers:
        decimals = infer_rounding_decimals(question, gold_answer)
        abs_tolerance = 0.5 * (10**-decimals)
        best = None
        for answer_number in answer_numbers:
            for gold_number in gold_numbers:
                diff = abs(answer_number["value"] - gold_number["value"])
                scale = max(
                    answer_number.get("scale", 1.0),
                    gold_number.get("scale", 1.0),
                )
                tolerance = abs_tolerance * scale
                rounded_answer = round_for_answer_comparison(
                    answer_number["value"],
                    decimals,
                    scale,
                )
                rounded_gold = round_for_answer_comparison(
                    gold_number["value"],
                    decimals,
                    scale,
                )
                item = (
                    diff,
                    rounded_answer,
                    rounded_gold,
                    tolerance,
                    answer_number,
                    gold_number,
                )
                if best is None or diff < best[0]:
                    best = item
        diff, rounded_answer, rounded_gold, tolerance, answer_number, gold_number = best
        matched = (
            rounded_answer is not None
            and rounded_gold is not None
            and rounded_answer == rounded_gold
        )
        return {
            "deterministic_answer_accuracy": 1.0 if matched else 0.0,
            "deterministic_answer_match": matched,
            "deterministic_answer_type": "numeric",
            "deterministic_answer_reason": (
                f"answer={answer_number['raw']} ({answer_number['value']}); "
                f"gold={gold_number['raw']} ({gold_number['value']}); "
                f"rounded_answer={rounded_answer}; rounded_gold={rounded_gold}; "
                f"diff={diff}; tolerance_equivalent={tolerance}; "
                f"rounding_decimals={decimals}"
            ),
            "deterministic_answer_value": answer_number["value"],
            "deterministic_gold_value": gold_number["value"],
            "deterministic_rounded_answer_value": (
                float(rounded_answer) if rounded_answer is not None else None
            ),
            "deterministic_rounded_gold_value": (
                float(rounded_gold) if rounded_gold is not None else None
            ),
            "deterministic_tolerance": tolerance,
        }

    normalized_answer = normalize_text(answer)
    normalized_gold = normalize_text(gold_answer)
    if normalized_answer and normalized_gold:
        matched = normalized_answer == normalized_gold or (
            len(normalized_gold) >= 4 and normalized_gold in normalized_answer
        )
        return {
            "deterministic_answer_accuracy": 1.0 if matched else 0.0,
            "deterministic_answer_match": matched,
            "deterministic_answer_type": "text",
            "deterministic_answer_reason": "normalized exact/contains match",
        }

    return {
        "deterministic_answer_accuracy": None,
        "deterministic_answer_match": None,
        "deterministic_answer_type": None,
        "deterministic_answer_reason": "could not extract comparable answer",
    }


async def evaluate_samples(samples, judge_model, temperature):
    try:
        from ragas.metrics.collections import AnswerAccuracy, Faithfulness
    except ImportError as exc:
        raise ImportError("Answer evaluation requires ragas.") from exc

    llm = build_ragas_azure_components(judge_model, temperature=temperature)
    accuracy_scorer = AnswerAccuracy(llm=llm)
    faithfulness_scorer = Faithfulness(llm=llm)

    results = []
    for sample in samples:
        accuracy_score = await accuracy_scorer.ascore(
            user_input=sample["question"],
            response=sample["answer"],
            reference=sample["gold_answer"],
        )
        faithfulness_score = await faithfulness_scorer.ascore(
            user_input=sample["question"],
            response=sample["answer"],
            retrieved_contexts=sample["retrieved_contexts"],
        )
        results.append(
            {
                "answer_accuracy": clean_score(accuracy_score),
                "faithfulness": clean_score(faithfulness_score),
            }
        )
    return results


def summary(rows):
    evaluations = [
        row.get("answer_evaluation") or {}
        for row in rows
        if row.get("answer_evaluation") is not None
    ]
    return {
        "evaluated": len(evaluations),
        "mean_answer_accuracy": mean_metric(evaluations, "answer_accuracy"),
        "mean_deterministic_answer_accuracy": mean_metric(
            evaluations,
            "deterministic_answer_accuracy",
        ),
        "mean_faithfulness": mean_metric(evaluations, "faithfulness"),
        "missing_answers": sum(1 for row in rows if not row.get("answer")),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate answer accuracy and faithfulness with RAGAS metrics."
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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--question-id", action="append", default=[])
    args = parser.parse_args()

    rows = load_rows(args.input)
    samples = []
    row_indices = []

    for row_index, row in selected_rows(rows, set(args.question_id), args.limit):
        if row.get("answer_evaluation") and not args.overwrite:
            continue

        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        gold_answer = str(row.get("gold_answer") or "").strip()
        contexts = get_retrieved_contexts(
            row,
            max_contexts=args.max_contexts,
            max_context_chars=args.max_context_chars,
        )
        row["answer_evaluation"] = {
            "answer_accuracy": None,
            "faithfulness": None,
            "deterministic_answer_accuracy": None,
            "answer_accuracy_metric": "AnswerAccuracy",
            "faithfulness_metric": "Faithfulness",
            "deterministic_answer_metric": "numeric_or_yes_no_or_text_match",
            "judge_model": args.judge_model,
            "retrieved_contexts_used": len(contexts),
        }

        notes = []
        if not question:
            notes.append("question is missing")
        if not answer:
            notes.append("answer is missing")
        if not gold_answer:
            notes.append("gold_answer is missing")
        if notes:
            if not contexts:
                notes.append("retrieved_contexts is missing")
            row["answer_evaluation"]["note"] = "; ".join(notes)
            continue

        row["answer_evaluation"].update(
            deterministic_answer_evaluation(
                question=question,
                answer=answer,
                gold_answer=gold_answer,
            )
        )

        if not contexts:
            row["answer_evaluation"]["note"] = (
                "retrieved_contexts is missing; RAGAS metrics skipped"
            )
            continue

        row_indices.append(row_index)
        samples.append(
            {
                "question": question,
                "answer": answer,
                "gold_answer": gold_answer,
                "retrieved_contexts": contexts,
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
            rows[row_index]["answer_evaluation"].update(score)

    save_rows(args.output, rows)
    print(json.dumps(summary(rows), indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Saved answer evaluation to {args.output}")


if __name__ == "__main__":
    main()
