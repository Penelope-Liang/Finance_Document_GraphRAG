from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from finance_document_qa.config import FinanceDocumentGraphRAGConfig
from finance_document_qa.pipeline.formatting import _extract_gold_evidence
from finance_document_qa.pipeline.utils import _load_questions, _save_results
from finance_document_qa.services.base import (
    BaseFinanceDocumentService,
    QueryOptions,
)
from finance_document_qa.services.graphrag import (
    FinanceDocumentGraphRAGService,
    FinanceDocumentText2CypherService,
)
from finance_document_qa.services.rag import FinanceDocumentRAGService

_SERVICE_BY_MODE: dict[str, type[BaseFinanceDocumentService]] = {
    "hybrid": FinanceDocumentRAGService,
    "dual": FinanceDocumentGraphRAGService,
    "text2cypher": FinanceDocumentText2CypherService,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FinanceBench QA with RAG or GraphRAG retrieval."
    )
    parser.add_argument(
        "--questions-path",
        default="dataset/financebench/financebench_direct_extract_13.jsonl",
        help="Path to the FinanceBench question set.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Optional cap on the number of questions to run (0 means all).",
    )
    parser.add_argument(
        "--output",
        default="output/direct_extract_13_batch_results.json",
        help="Path to save batch results.",
    )
    parser.add_argument(
        "--retriever-mode",
        choices=["text2cypher", "hybrid", "dual"],
        default="text2cypher",
        help="Retriever mode to use for the batch run.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Top K candidates for hybrid retrieval."
    )
    parser.add_argument(
        "--effective-search-ratio",
        type=int,
        default=10,
        help="Hybrid candidate expansion factor before final top-k.",
    )
    parser.add_argument(
        "--question-style",
        choices=["direct", "calculation", "domain", "novel", "auto", "llm-auto"],
        default="direct",
        help=(
            "Prompt and Text2Cypher template: direct (single-value), calculation (multi-input), "
            "domain (finance judgment), novel (open-ended), auto (rule-based classification per question), "
            "or llm-auto (LLM router per question; see question_style_* in output JSON)."
        ),
    )
    parser.add_argument(
        "--use-query-rewrite",
        action="store_true",
        help=(
            "FinanceBench_RAG-style retrieval rewrite: keep the original question for answering, "
            "but use a retrieval-friendly query with reported components in dual mode."
        ),
    )
    parser.add_argument(
        "--use-gold-doc-id",
        action="store_true",
        dest="use_gold_doc_id",
        help=(
            "Oracle evaluation mode: restrict retrieval to the dataset-provided "
            "gold doc_id instead of inferring the filing from the question."
        ),
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help=(
            "Enable the CrossEncoder reranker for hybrid/text2cypher retrieval "
            "and dual-mode evidence fusion."
        ),
    )
    parser.add_argument(
        "--reranker-model",
        default=os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        help="Sentence-transformers CrossEncoder model for reranking.",
    )
    parser.add_argument(
        "--reranker-top-k-retrieve",
        type=int,
        default=20,
        help="Number of candidates to fetch before reranking.",
    )
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Apply an LLM-based evidence filter after retrieval and before answer generation.",
    )
    parser.add_argument(
        "--llm-judge-max-chars",
        type=int,
        default=1200,
        help="Max chars per candidate item shown to the LLM judge.",
    )
    parser.add_argument(
        "--llm-judge-model",
        default=os.environ.get("OLLAMA_JUDGE_MODEL", "qwen3:14b"),
        help="Ollama model used only for evidence judging.",
    )
    parser.add_argument(
        "--llm-retries",
        type=int,
        default=6,
        help=(
            "Max retry rounds after a failed rag.search for transient Ollama errors "
            "(e.g. json: unsupported value: NaN / status 500). Each round retries the full search."
        ),
    )
    parser.add_argument(
        "--llm-retry-sleep",
        type=float,
        default=1.5,
        help="Base seconds between retryable LLM errors (scaled by attempt; NaN/encode errors wait longer).",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["ollama", "azure-openai"],
        default=os.environ.get("LLM_BACKEND", "ollama"),
        help="Answer-generation LLM backend. Defaults to local Ollama; use azure-openai for Azure OpenAI API.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("OLLAMA_LLM_MODEL", "qwen3:30b-a3b"),
        help="Ollama answer model when --llm-backend=ollama.",
    )
    parser.add_argument(
        "--ollama-num-ctx",
        type=int,
        default=None,
        help="Set Ollama options.num_ctx for local LLM calls.",
    )
    parser.add_argument(
        "--use-calculation-executor",
        action="store_true",
        help=(
            "For calculation questions only, ask a coder model for a JSON formula plan, "
            "safely evaluate it, and write the computed result back to answer."
        ),
    )
    parser.add_argument(
        "--use-direct-table-fallback",
        action="store_true",
        help=(
            "For direct questions only, when the LLM returns no answer, deterministically "
            "build a numeric answer from the top exact_cell/metric_cell TableCell evidence."
        ),
    )
    parser.add_argument(
        "--calculation-model",
        default=os.environ.get("OLLAMA_CALC_MODEL", "qwen2.5-coder:14b-instruct"),
        help="Ollama coder model used by --use-calculation-executor.",
    )
    return parser


def _build_service(args: argparse.Namespace) -> BaseFinanceDocumentService:
    service_cls = _SERVICE_BY_MODE[args.retriever_mode]
    return service_cls(
        FinanceDocumentGraphRAGConfig.from_env(),
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        ollama_num_ctx=args.ollama_num_ctx,
        use_reranker=args.use_reranker,
        reranker_model=args.reranker_model,
        reranker_top_k_retrieve=args.reranker_top_k_retrieve,
        use_llm_judge=args.use_llm_judge,
        llm_judge_model=args.llm_judge_model,
        llm_judge_max_chars=args.llm_judge_max_chars,
        use_calculation_executor=args.use_calculation_executor,
        calculation_model=args.calculation_model,
    )


def _query_options(args: argparse.Namespace) -> QueryOptions:
    return QueryOptions(
        top_k=args.top_k,
        effective_search_ratio=args.effective_search_ratio,
        question_style=args.question_style,
        use_query_rewrite=args.use_query_rewrite,
        use_gold_doc_id=args.use_gold_doc_id,
        use_direct_table_fallback=args.use_direct_table_fallback,
        llm_retries=args.llm_retries,
        llm_retry_sleep=args.llm_retry_sleep,
    )


def main() -> None:
    args = _build_parser().parse_args()

    questions = _load_questions(Path(args.questions_path))
    if args.max_questions > 0:
        questions = questions[: args.max_questions]
    available_doc_ids = sorted(
        {
            str(row.get("doc_id") or "").strip()
            for row in questions
            if str(row.get("doc_id") or "").strip()
        }
    )

    options = _query_options(args)
    output_path = Path(args.output)
    results: list[dict[str, Any]] = []

    with _build_service(args) as service:
        for idx, row in enumerate(questions, start=1):
            print(
                f"[{idx}/{len(questions)}] {row['doc_id']} :: {row['question_id']}",
                flush=True,
            )
            record = service.query(
                row["question"],
                question_id=row["question_id"],
                doc_id=row["doc_id"],
                doc_name=row["doc_name"],
                question_reasoning=row.get("question_reasoning"),
                available_doc_ids=available_doc_ids,
                options=options,
            )
            record["gold_answer"] = row["gold_answer"]
            record["gold_justification"] = row.get("gold_justification")
            record["gold_evidence"] = _extract_gold_evidence(row)
            results.append(record)
            _save_results(output_path, results)
            if record.get("error"):
                print(f"  error={record['error']}", flush=True)
            else:
                print(
                    f"  answer={record.get('answer')!r} gold={row['gold_answer']}",
                    flush=True,
                )

    _save_results(output_path, results)
    print(f"\nSaved results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
