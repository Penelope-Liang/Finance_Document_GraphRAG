#!/usr/bin/env python3
"""Evaluate query_router compare results with RAGAS using local Ollama models.

Expected inputs:
- predictions JSON from query_router --mode compare
- gold qa JSONL from FinDER export (question + gold_answer)
"""

from __future__ import annotations

import argparse
import json
import re
from inspect import signature
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate rag/graphrag predictions with RAGAS using local Ollama models."
    )
    parser.add_argument(
        "--pred-path",
        default="artifacts/finder/pred_50_compare.json",
        help="Prediction JSON path from query_router --mode compare.",
    )
    parser.add_argument(
        "--qa-path",
        default="artifacts/finder/qa_50.jsonl",
        help="Gold QA JSONL path (must include question and gold_answer).",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/finder/ragas",
        help="Output directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["rag", "graphrag"],
        choices=["rag", "graphrag"],
        help="Which modes to evaluate.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://127.0.0.1:11434",
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--ollama-api-key",
        default="ollama",
        help="API key for Ollama OpenAI-compatible endpoint (any non-empty string is typically accepted).",
    )
    parser.add_argument(
        "--llm-model",
        default="qwen2.5:7b-instruct",
        help="Local Ollama chat model for RAGAS judge.",
    )
    parser.add_argument(
        "--embed-model",
        default="bge-m3:latest",
        help="Local Ollama embedding model for RAGAS embedding metrics.",
    )
    parser.add_argument(
        "--skip-empty-answer",
        action="store_true",
        help="Skip rows where model answer is empty/null before evaluation.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```json\\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^```\\s*", "", t)
    t = re.sub(r"\\s*```$", "", t)
    return t.strip()


def _extract_answer(raw_answer: Any) -> str:
    if raw_answer is None:
        return ""
    if isinstance(raw_answer, dict):
        # Compatible with both old and new prompt schemas
        for key in ("answer", "final_answer"):
            value = raw_answer.get(key)
            if value is not None:
                value_str = str(value).strip()
                if value_str:
                    return value_str
        return json.dumps(raw_answer, ensure_ascii=False)

    text = str(raw_answer).strip()
    if not text:
        return ""

    candidate = _strip_json_fence(text)
    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            for key in ("answer", "final_answer"):
                value = payload.get(key)
                if value is not None:
                    value_str = str(value).strip()
                    if value_str:
                        return value_str
    except Exception:
        pass

    return text


def _extract_contexts(raw_context: Any) -> list[str]:
    if raw_context is None:
        return []
    if isinstance(raw_context, list):
        return [str(x) for x in raw_context if str(x).strip()]
    text = str(raw_context).strip()
    return [text] if text else []


def _build_gold_index(qa_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    for row in qa_rows:
        question = str(row.get("question", "")).strip()
        if question and question not in idx:
            idx[question] = row
    return idx


def _evaluation_rows_for_mode(
    pred_rows: list[dict[str, Any]],
    gold_by_question: dict[str, dict[str, Any]],
    mode: str,
    skip_empty_answer: bool,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    missing_gold = 0

    for pred in pred_rows:
        if pred.get("mode") != mode:
            continue

        question = str(pred.get("question", "")).strip()
        if not question:
            continue

        gold = gold_by_question.get(question)
        if not gold:
            missing_gold += 1
            continue

        answer = _extract_answer(pred.get("answer"))
        if skip_empty_answer and not answer:
            continue

        ground_truth = gold.get("gold_answer")
        ground_truth_str = "" if ground_truth is None else str(ground_truth).strip()

        rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": _extract_contexts(pred.get("context_preview")),
                "ground_truth": ground_truth_str,
            }
        )

    return rows, missing_gold


def _result_to_dict(result: Any) -> dict[str, Any]:
    # Works across ragas versions where EvaluationResult acts dict-like
    try:
        return {k: float(v) if isinstance(v, (int, float)) else v for k, v in result.items()}
    except Exception:
        pass

    if hasattr(result, "to_dict") and callable(result.to_dict):
        try:
            data = result.to_dict()
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # Fallback to string representation
    return {"result": str(result)}


def _normalize_openai_base_url(ollama_base_url: str) -> str:
    base = ollama_base_url.rstrip("/")
    if not base.endswith("/v1"):
        return f"{base}/v1"
    return base


def _call_with_supported_kwargs(factory: Any, kwargs: dict[str, Any], positional: tuple[Any, ...]) -> Any:
    """Call a factory by dropping unknown kwargs based on its signature."""
    params = signature(factory).parameters
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return factory(*positional, **filtered)


def _build_ragas_runtime(args: argparse.Namespace) -> tuple[Any, Any, str]:
    """Build (llm, embeddings, backend_name) with modern API first, fallback to legacy wrappers."""
    # Modern path: ragas factories + OpenAI-compatible client pointed to Ollama
    try:
        from openai import OpenAI
        from ragas.embeddings import embedding_factory
        from ragas.llms import llm_factory

        client = OpenAI(
            base_url=_normalize_openai_base_url(args.ollama_base_url),
            api_key=args.ollama_api_key,
        )

        llm = None
        llm_attempts = (
            ((), {"model": args.llm_model, "provider": "openai", "client": client}),
            ((args.llm_model,), {"provider": "openai", "client": client}),
            ((args.llm_model,), {"client": client}),
            ((), {"model": args.llm_model, "client": client}),
        )
        for pos_args, kwargs in llm_attempts:
            try:
                llm = _call_with_supported_kwargs(llm_factory, kwargs, pos_args)
                break
            except Exception:
                continue
        if llm is None:
            raise RuntimeError("Unable to initialize llm_factory for Ollama/OpenAI-compatible endpoint.")

        embeddings = None
        emb_attempts = (
            ((), {"provider": "openai", "model": args.embed_model, "client": client}),
            ((args.embed_model,), {"provider": "openai", "client": client}),
            ((args.embed_model,), {"client": client}),
            ((), {"model": args.embed_model, "client": client}),
        )
        for pos_args, kwargs in emb_attempts:
            try:
                embeddings = _call_with_supported_kwargs(embedding_factory, kwargs, pos_args)
                break
            except Exception:
                continue
        if embeddings is None:
            raise RuntimeError("Unable to initialize embedding_factory for Ollama/OpenAI-compatible endpoint.")

        return llm, embeddings, "factory-openai-compat"
    except Exception:
        pass

    # Legacy fallback path
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    llm = LangchainLLMWrapper(
        ChatOllama(
            model=args.llm_model,
            temperature=0,
            base_url=args.ollama_base_url,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=args.embed_model, base_url=args.ollama_base_url)
    )
    return llm, embeddings, "langchain-wrapper-fallback"


def main() -> None:
    args = parse_args()

    try:
        from datasets import Dataset
        from ragas import evaluate
        try:
            from ragas.metrics.collections import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except ImportError:
            from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise ImportError(
            "Missing dependency. Install with: "
            "python3 -m pip install ragas langchain-ollama datasets pandas openai"
        ) from exc

    pred_path = Path(args.pred_path)
    qa_path = Path(args.qa_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_rows = _read_json(pred_path)
    if not isinstance(pred_rows, list):
        raise ValueError(f"Prediction file must be a JSON array: {pred_path}")

    qa_rows = _read_jsonl(qa_path)
    gold_by_question = _build_gold_index(qa_rows)

    llm, embeddings, runtime_backend = _build_ragas_runtime(args)
    print(f"[runtime] backend={runtime_backend} llm={args.llm_model} embed={args.embed_model}")

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    global_summary: dict[str, Any] = {
        "pred_path": str(pred_path),
        "qa_path": str(qa_path),
        "ollama_base_url": args.ollama_base_url,
        "ollama_api_key": "***" if args.ollama_api_key else "",
        "llm_model": args.llm_model,
        "embed_model": args.embed_model,
        "runtime_backend": runtime_backend,
        "modes": {},
    }

    for mode in args.modes:
        eval_rows, missing_gold = _evaluation_rows_for_mode(
            pred_rows=pred_rows,
            gold_by_question=gold_by_question,
            mode=mode,
            skip_empty_answer=args.skip_empty_answer,
        )

        if not eval_rows:
            print(f"[skip] mode={mode}: no rows to evaluate")
            global_summary["modes"][mode] = {
                "rows": 0,
                "missing_gold": missing_gold,
                "scores": {},
            }
            continue

        ds = Dataset.from_list(eval_rows)
        result = evaluate(
            ds,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
        )

        scores = _result_to_dict(result)
        mode_summary = {
            "rows": len(eval_rows),
            "missing_gold": missing_gold,
            "scores": scores,
        }
        global_summary["modes"][mode] = mode_summary

        (out_dir / f"ragas_{mode}_summary.json").write_text(
            json.dumps(mode_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / f"ragas_{mode}_eval_input.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in eval_rows) + "\n",
            encoding="utf-8",
        )

        print(f"[done] mode={mode} rows={len(eval_rows)} missing_gold={missing_gold}")
        print(json.dumps(scores, ensure_ascii=False))

    (out_dir / "ragas_summary_all_modes.json").write_text(
        json.dumps(global_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
