from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from neo4j_graphrag.llm import OllamaLLM

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for ad-hoc runs.
    tqdm = None


ROUTER_PROMPT = """You are routing FinanceBench questions into exactly one pipeline.

Choose exactly one label:

direct
- The question asks for one specific fact, entity, date, filing item, financial line item, or raw amount.
- No ratio, formula, percentage change, trend judgment, or multi-step business explanation is required.

calculation
- The question asks for a computed metric, ratio, margin, turnover, growth rate, percentage, year-over-year change, comparison, increase/decrease, average, ROA/ROE, quick ratio, current ratio, fixed asset turnover, free cash flow conversion, or formula-based answer.
- This includes yes/no questions where the answer depends mainly on computing or comparing financial values.

domain
- The question asks for finance/business judgment, financial health, metric applicability, capital intensity, liquidity profile, margin profile, risk, trend explanation, business meaning, whether a metric is useful/meaningful, what drove a change, or why something happened.
- These questions may require calculations, but the final answer is a business/financial interpretation or judgment.

novel
- The question is open-ended, multi-step, or cross-section reasoning not mainly a standard financial metric calculation.
- It may ask about policies, filings, governance, derivatives, customer retention, strategic events, acquisitions, business segments, or other document-specific reasoning.

Priority rules:
- If the question asks for metric applicability or business interpretation after a calculation, choose domain, not calculation.
- If it asks only for a formula result or numeric comparison, choose calculation.
- If it asks only for one fact or raw value, choose direct.
- If none of the above clearly fits and the question requires broad document reasoning, choose novel.

Return JSON only:
{{"label": "direct|calculation|domain|novel", "reason": "short reason"}}

Question:
{question}
"""


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing env var: {name}")
    return value


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


def _parse_route(raw: str | None) -> tuple[str | None, str | None]:
    if not raw:
        return None, None

    obj = _extract_json_object(raw)
    if obj is not None:
        label = str(obj.get("label") or obj.get("route") or obj.get("kind") or "").lower()
        reason = obj.get("reason")
        if label in {"direct", "calculation", "domain", "novel"}:
            return label, str(reason) if reason is not None else None

    text = _strip_think(raw).lower()
    match = re.search(r"\b(direct|calculation|domain|novel)\b", text)
    return (match.group(1), None) if match else (None, None)


def _gold_route(row: dict[str, Any]) -> str:
    question_type = row.get("question_type")
    reasoning = row.get("question_reasoning")
    if question_type == "domain-relevant":
        return "domain"
    if question_type == "novel-generated":
        return "novel"
    if question_type == "metrics-generated":
        return "direct" if reasoning == "Information extraction" else "calculation"
    return "unknown"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _build_ollama_client(model_name: str) -> Any:
    host = os.environ.get("OLLAMA_HOST")
    kwargs = {"host": host} if host else {}
    return OllamaLLM(
        model_name=model_name,
        model_params={"options": {"temperature": 0}},
        **kwargs,
    )


def _build_azure_client() -> Any:
    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise ImportError("Azure OpenAI router evaluation requires `pip install openai`.") from exc

    return AzureOpenAI(
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
        api_key=_require_env("AZURE_OPENAI_API_KEY"),
    )


def _invoke_router(
    *,
    client: Any,
    llm_backend: str,
    prompt: str,
    model_name: str,
    max_tokens: int,
) -> str:
    if llm_backend == "ollama":
        return client.invoke(prompt).content

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a precise FinanceBench routing classifier."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_completion_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("pred_route") is not None]
    correct = [row for row in valid if row.get("correct") is True]
    confusion: dict[str, dict[str, int]] = {}
    raw_confusion: defaultdict[str, Counter[str | None]] = defaultdict(Counter)
    for row in rows:
        raw_confusion[row.get("gold_route") or "unknown"][row.get("pred_route")] += 1
    for gold, counts in raw_confusion.items():
        confusion[gold] = {str(pred): count for pred, count in counts.items()}
    return {
        "rows": len(rows),
        "valid_predictions": len(valid),
        "none_predictions": len(rows) - len(valid),
        "correct": len(correct),
        "accuracy_valid_only": (len(correct) / len(valid)) if valid else None,
        "accuracy_all": (len(correct) / len(rows)) if rows else None,
        "gold_counts": dict(Counter(row.get("gold_route") for row in rows)),
        "pred_counts": dict(Counter(row.get("pred_route") for row in rows)),
        "confusion": confusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FinanceBench question router labels with a local or Azure LLM."
    )
    parser.add_argument("--input", type=Path, default=Path("dataset/financebench/financebench_merged.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("output/router_eval_llm_150.json"))
    parser.add_argument(
        "--llm",
        dest="llm_backend",
        choices=["ollama", "azure-openai"],
        default=os.environ.get("LLM_BACKEND", "ollama"),
        help="Router LLM backend.",
    )
    parser.add_argument(
        "--llm-backend",
        dest="llm_backend",
        choices=["ollama", "azure-openai"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--ollama-model", default=os.environ.get("OLLAMA_LLM_MODEL", "qwen3.6:27b"))
    parser.add_argument("--azure-deployment", default=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    rows = _load_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]

    if args.llm_backend == "ollama":
        client = _build_ollama_client(args.ollama_model)
        model_name = args.ollama_model
    else:
        client = _build_azure_client()
        model_name = args.azure_deployment

    results: list[dict[str, Any]] = []
    iterator = enumerate(rows, start=1)
    progress = None
    if tqdm is not None and not args.no_progress:
        progress = tqdm(iterator, total=len(rows), desc="Routing", unit="question")
        iterable = progress
    else:
        iterable = iterator

    started = time.perf_counter()
    for idx, row in iterable:
        item_started = time.perf_counter()
        question = row.get("question") or ""
        gold = _gold_route(row)
        raw = None
        error = None
        try:
            raw = _invoke_router(
                client=client,
                llm_backend=args.llm_backend,
                prompt=ROUTER_PROMPT.format(question=question),
                model_name=model_name,
                max_tokens=args.max_tokens,
            )
            pred, reason = _parse_route(raw)
        except Exception as exc:  # pragma: no cover - external service
            pred = None
            reason = None
            error = str(exc)

        result = {
            "question_id": row.get("financebench_id") or row.get("question_id") or row.get("id"),
            "doc_name": row.get("doc_name"),
            "question": question,
            "question_type": row.get("question_type"),
            "question_reasoning": row.get("question_reasoning"),
            "gold_route": gold,
            "pred_route": pred,
            "correct": pred == gold,
            "reason": reason,
            "raw": raw,
            "error": error,
            "model": model_name,
            "llm_backend": args.llm_backend,
        }
        results.append(result)
        _save_json(args.output, results)

        elapsed = time.perf_counter() - item_started
        avg = (time.perf_counter() - started) / idx
        status = f"last={elapsed:.1f}s avg={avg:.1f}s pred={pred} gold={gold}"
        if progress is not None:
            progress.set_postfix_str(status)
        else:
            print(f"[{idx}/{len(rows)}] {status} ok={result['correct']} id={result['question_id']}", flush=True)
        if args.sleep > 0:
            time.sleep(args.sleep)

    summary = _summary(results)
    _save_json(args.output.with_suffix(".summary.json"), summary)
    print("rows:", summary["rows"])
    print("valid_predictions:", summary["valid_predictions"])
    print("none_predictions:", summary["none_predictions"])
    print("correct:", summary["correct"])
    print(
        "accuracy_valid_only:",
        f"{summary['accuracy_valid_only']:.2%}" if summary["accuracy_valid_only"] is not None else "NA",
    )
    print("accuracy_all:", f"{summary['accuracy_all']:.2%}" if summary["accuracy_all"] is not None else "NA")
    print("gold_counts:", summary["gold_counts"])
    print("pred_counts:", summary["pred_counts"])
    print("confusion:", summary["confusion"])
    print("saved:", args.output)
    print("summary:", args.output.with_suffix(".summary.json"))


if __name__ == "__main__":
    main()
