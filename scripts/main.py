from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import neo4j
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.llm import AzureOpenAILLM, OllamaLLM
from neo4j_graphrag.rag import GraphRAG
from neo4j_graphrag.retrievers import (
    HybridCypherRetriever,
    RerankingRetriever,
    Text2CypherRetriever,
)

from financebench_pipeline.cypher_prompts import (
    FINANCIAL_GRAPH_SCHEMA,
    TEXT2CYPHER_EXAMPLES,
    TEXT2CYPHER_EXAMPLES_CALCULATION,
    TEXT2CYPHER_EXAMPLES_NOVEL,
    TEXT2CYPHER_PROMPT,
    TEXT2CYPHER_PROMPT_CALCULATION,
    TEXT2CYPHER_PROMPT_NOVEL,
)

from financebench_pipeline.question_routing import (
    classify_question_style_auto as _classify_question_style_auto,
    classify_question_style_llm as _classify_question_style_llm,
)
from financebench_pipeline.answer_prompts import build_answer_prompt as _build_answer_prompt
from financebench_pipeline.calculation_executor import (
    run_calculation_pipeline as _run_calculation_pipeline,
)
from financebench_pipeline.utils import (
    _load_questions,
    _parse_answer_payload,
    _sanitize_for_json,
    _save_results,
)
from financebench_pipeline.query_rewriting import _rewrite_metrics_query_for_retrieval
from financebench_pipeline.retrievers import (
    DualRetriever,
    LLMJudgeRetriever,
    MetricsTableCellRetriever,
    PrimaryStatementRetriever,
    StyleRoutingText2CypherRetriever,
    _rag_search_with_retry,
)
from financebench_pipeline.retrieval_scope import _build_retrieval_filter
from financebench_pipeline.retrieval_queries import CHUNK_RETRIEVAL_QUERY
from financebench_pipeline.formatting import (
    _compact_retriever_evidence,
    _extract_gold_evidence,
    _format_chunk_record,
    _format_table_record,
    _standardize_retrieved_contexts,
)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing env var: {name}")
    return value


def _build_answer_llm(
    *,
    backend: str,
    ollama_model: str,
    ollama_host: str | None,
    ollama_num_ctx: int | None,
) -> Any:
    if backend == "azure-openai":
        return AzureOpenAILLM(
            model_name=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"),
            model_params={"temperature": 0},
            azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
            api_key=_require_env("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )

    ollama_kwargs = {"host": ollama_host} if ollama_host else {}
    ollama_options: dict[str, Any] = {"temperature": 0}
    if ollama_num_ctx is not None:
        ollama_options["num_ctx"] = ollama_num_ctx
    return OllamaLLM(
        model_name=ollama_model,
        model_params={"options": ollama_options},
        **ollama_kwargs,
    )


def _build_run_config(
    args: argparse.Namespace,
    *,
    effective_style: str,
    embedding_model: str,
) -> dict[str, Any]:
    return {
        "retriever_mode": args.retriever_mode,
        "question_style": effective_style,
        "top_k": args.top_k,
        "effective_search_ratio": args.effective_search_ratio,
        "use_query_rewrite": bool(args.use_query_rewrite),
        "restrict_doc_id": bool(args.restrict_doc_id),
        "use_calculation_executor": bool(args.use_calculation_executor),
        "llm_backend": args.llm_backend,
        "llm_model": (
            os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
            if args.llm_backend == "azure-openai"
            else args.llm_model
        ),
        "embedding_model": embedding_model,
        "ollama_num_ctx": args.ollama_num_ctx if args.llm_backend == "ollama" else None,
        "calculation_model": args.calculation_model if args.use_calculation_executor else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinanceBench QA with RAG or GraphRAG retrieval.")
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
        "--top-k",
        type=int,
        default=5,
        help="Top K candidates for hybrid retrieval.",
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
            "or llm-auto (LLM router per question; "
            "see question_style_* in output JSON)."
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
        "--restrict-doc-id",
        action="store_true",
        help=(
            "Oracle evaluation mode: restrict retrieval to the dataset-provided "
            "target doc_id instead of inferring the filing from the question."
        ),
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Wrap the chosen retriever with the existing CrossEncoder reranker.",
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
        "--calculation-model",
        default=os.environ.get("OLLAMA_CALC_MODEL", "qwen2.5-coder:14b-instruct"),
        help="Ollama coder model used by --use-calculation-executor.",
    )
    args = parser.parse_args()

    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = _require_env("NEO4J_PASSWORD")
    database = os.environ.get("NEO4J_DATABASE")
    vector_index = os.environ.get("NEO4J_VECTOR_INDEX", "chunk_embedding_index")
    fulltext_index = os.environ.get("NEO4J_FULLTEXT_INDEX", "chunk_fulltext_index")
    embedding_model = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3:latest")
    ollama_host = os.environ.get("OLLAMA_HOST")

    ollama_kwargs = {}
    if ollama_host:
        ollama_kwargs["host"] = ollama_host

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
    llm = _build_answer_llm(
        backend=args.llm_backend,
        ollama_model=args.llm_model,
        ollama_host=ollama_host,
        ollama_num_ctx=args.ollama_num_ctx,
    )
    ollama_options: dict[str, Any] = {"temperature": 0}
    if args.ollama_num_ctx is not None:
        ollama_options["num_ctx"] = args.ollama_num_ctx
    calc_llm = (
        OllamaLLM(
            model_name=args.calculation_model,
            model_params={"options": ollama_options},
            **ollama_kwargs,
        )
        if args.use_calculation_executor
        else None
    )
    judge_llm = OllamaLLM(
        model_name=args.llm_judge_model,
        model_params={"options": ollama_options},
        **ollama_kwargs,
    )
    embedder = OllamaEmbeddings(model=embedding_model, **ollama_kwargs)

    output_path = Path(args.output)
    results = []
    with neo4j.GraphDatabase.driver(uri, auth=(user, password)) as driver:
        t2c_direct = Text2CypherRetriever(
            driver=driver,
            llm=llm,
            neo4j_database=database,
            neo4j_schema=FINANCIAL_GRAPH_SCHEMA,
            examples=TEXT2CYPHER_EXAMPLES,
            custom_prompt=TEXT2CYPHER_PROMPT,
            result_formatter=_format_table_record,
        )
        t2c_calculation = Text2CypherRetriever(
            driver=driver,
            llm=llm,
            neo4j_database=database,
            neo4j_schema=FINANCIAL_GRAPH_SCHEMA,
            examples=TEXT2CYPHER_EXAMPLES_CALCULATION,
            custom_prompt=TEXT2CYPHER_PROMPT_CALCULATION,
            result_formatter=_format_table_record,
        )
        t2c_novel = Text2CypherRetriever(
            driver=driver,
            llm=llm,
            neo4j_database=database,
            neo4j_schema=FINANCIAL_GRAPH_SCHEMA,
            examples=TEXT2CYPHER_EXAMPLES_NOVEL,
            custom_prompt=TEXT2CYPHER_PROMPT_NOVEL,
            result_formatter=_format_table_record,
        )
        table_router = StyleRoutingText2CypherRetriever(
            t2c_direct,
            t2c_calculation,
            t2c_novel,
        )
        if args.retriever_mode == "text2cypher":
            retriever = table_router
        elif args.retriever_mode == "hybrid":
            retriever = HybridCypherRetriever(
                driver=driver,
                vector_index_name=vector_index,
                fulltext_index_name=fulltext_index,
                neo4j_database=database,
                retrieval_query=CHUNK_RETRIEVAL_QUERY,
                embedder=embedder,
                result_formatter=_format_chunk_record,
            )
        else:
            chunk_retriever = HybridCypherRetriever(
                driver=driver,
                vector_index_name=vector_index,
                fulltext_index_name=fulltext_index,
                neo4j_database=database,
                retrieval_query=CHUNK_RETRIEVAL_QUERY,
                embedder=embedder,
                result_formatter=_format_chunk_record,
            )
            table_retriever = table_router
            primary_retriever = PrimaryStatementRetriever(
                driver=driver,
                neo4j_database=database,
            )
            metrics_retriever = MetricsTableCellRetriever(
                driver=driver,
                neo4j_database=database,
            )
            retriever = DualRetriever(
                driver=driver,
                chunk_retriever=chunk_retriever,
                table_retriever=table_retriever,
                primary_retriever=primary_retriever,
                metrics_retriever=metrics_retriever,
                neo4j_database=database,
            )
        if args.use_reranker:
            retriever = RerankingRetriever(
                inner_retriever=retriever,
                cross_encoder_model=args.reranker_model,
                top_k_retrieve=args.reranker_top_k_retrieve,
            )
        if args.use_llm_judge:
            retriever = LLMJudgeRetriever(
                inner_retriever=retriever,
                llm=judge_llm,
                judge_max_chars=args.llm_judge_max_chars,
                neo4j_database=database,
            )

        rag = GraphRAG(retriever=retriever, llm=llm)

        for idx, row in enumerate(questions, start=1):
            question_text = row["question"]
            if args.question_style == "auto":
                effective_style, q_style_conf, q_style_reason = _classify_question_style_auto(
                    question_text
                )
                q_style_raw = None
            elif args.question_style == "llm-auto":
                effective_style, q_style_conf, q_style_reason, q_style_raw = (
                    _classify_question_style_llm(llm, question_text)
                )
            else:
                effective_style = args.question_style
                q_style_conf = 1.0
                q_style_reason = "cli_fixed"
                q_style_raw = None
            style_fields = {
                "question_style_mode": args.question_style,
                "question_style": effective_style,
                "question_style_confidence": round(float(q_style_conf), 4),
                "question_style_reason": q_style_reason,
                "question_style_raw": q_style_raw,
            }
            query_rewrite_plan = (
                _rewrite_metrics_query_for_retrieval(
                    question_text,
                    question_style=effective_style,
                    doc_name=str(row.get("doc_name") or ""),
                    llm=llm,
                )
                if args.use_query_rewrite
                else {
                    "rewritten_query": question_text,
                    "company": None,
                    "company_source": None,
                    "company_doc_key": None,
                    "years": [],
                    "target_filing_year": None,
                    "operation": "disabled",
                    "components": [],
                    "statements": [],
                }
            )
            query_rewrite_fields = {
                "query_rewrite_applied": bool(args.use_query_rewrite),
                "retrieval_query": query_rewrite_plan["rewritten_query"],
                "query_rewrite_company": query_rewrite_plan["company"],
                "query_rewrite_company_source": query_rewrite_plan["company_source"],
                "query_rewrite_company_doc_key": query_rewrite_plan["company_doc_key"],
                "query_rewrite_years": query_rewrite_plan["years"],
                "query_rewrite_target_filing_year": query_rewrite_plan.get("target_filing_year"),
                "query_rewrite_operation": query_rewrite_plan["operation"],
                "query_rewrite_components": query_rewrite_plan["components"],
                "query_rewrite_statements": query_rewrite_plan["statements"],
            }
            print(
                f"[{idx}/{len(questions)}] {row['doc_id']} :: {row['question_id']}",
                flush=True,
            )
            if args.use_query_rewrite:
                print(f"  retrieval_query={query_rewrite_plan['rewritten_query']}", flush=True)
            retrieval_query_text = query_rewrite_plan["rewritten_query"]
            retrieval_filter = _build_retrieval_filter(
                row=row,
                query_rewrite_plan=query_rewrite_plan,
                use_query_rewrite=args.use_query_rewrite,
                restrict_doc_id=args.restrict_doc_id,
                available_doc_ids=available_doc_ids,
            )
            query_rewrite_fields["retrieval_doc_id"] = retrieval_filter.get("doc_id")
            query_rewrite_fields["retrieval_doc_ids"] = retrieval_filter.get("doc_ids")
            query_rewrite_fields["retrieval_doc_id_source"] = retrieval_filter.get("doc_id_source")
            query_rewrite_fields["retrieval_company_doc_key"] = retrieval_filter.get("company_doc_key")
            start = time.time()
            try:
                search_kwargs = {
                    "query_text": retrieval_query_text,
                    "return_context": True,
                    "examples": _build_answer_prompt(row["question"], effective_style),
                }
                retriever_config: dict[str, Any] = {}
                if args.retriever_mode in {"hybrid", "dual"}:
                    retriever_config = {
                        "top_k": args.top_k,
                        "effective_search_ratio": args.effective_search_ratio,
                        "query_params": retrieval_filter,
                    }
                if args.retriever_mode == "dual":
                    retriever_config["t2c_style"] = effective_style
                    if args.use_query_rewrite:
                        retriever_config["retrieval_query_text"] = retrieval_query_text
                        if effective_style in {"calculation", "domain"}:
                            retriever_config["metrics_doc_id"] = retrieval_filter["doc_id"]
                            retriever_config["metrics_doc_ids"] = retrieval_filter.get("doc_ids") or []
                            retriever_config["metrics_company_doc_key"] = retrieval_filter[
                                "company_doc_key"
                            ]
                            retriever_config["metrics_plan"] = query_rewrite_plan
                elif args.retriever_mode == "text2cypher":
                    retriever_config["t2c_style"] = effective_style
                if retriever_config:
                    search_kwargs["retriever_config"] = retriever_config

                result = _rag_search_with_retry(
                    rag,
                    max_retries=args.llm_retries,
                    retry_sleep=args.llm_retry_sleep,
                    **search_kwargs,
                )
                raw_answer = result.answer
                answer_payload = _parse_answer_payload(raw_answer)
                if answer_payload is not None:
                    answer = answer_payload.get("answer")
                    evidence = answer_payload.get("evidence")
                    missing_information = answer_payload.get("missing_information")
                else:
                    answer, evidence = raw_answer, None
                    missing_information = None
                retriever_result_dump = (
                    _sanitize_for_json(result.retriever_result.model_dump())
                    if result.retriever_result is not None
                    else None
                )
                retrieved_contexts = _standardize_retrieved_contexts(
                    retriever_result_dump,
                    question_id=str(row["question_id"]),
                )
                llm_answer_before_calculation = None
                calculation_executor = None
                if (
                    args.use_calculation_executor
                    and effective_style == "calculation"
                    and calc_llm is not None
                ):
                    llm_answer_before_calculation = answer
                    calculation_evidence = "\n\n".join(
                        part
                        for part in (
                            f"Current answer:\n{answer}" if answer is not None else "",
                            f"Answer evidence:\n{evidence}" if evidence else "",
                            _compact_retriever_evidence(retriever_result_dump),
                        )
                        if part
                    )
                    calculation_executor = _run_calculation_pipeline(
                        llm=calc_llm,
                        question=question_text,
                        current_answer=answer,
                        evidence=calculation_evidence,
                    )
                    if (
                        calculation_executor.get("calculation_success")
                        and calculation_executor.get("calculation_answer") is not None
                    ):
                        answer = calculation_executor["calculation_answer"]
                        evidence = (
                            f"{evidence}\n\n" if evidence else ""
                        ) + "Calculation executor applied."
                elapsed = round(time.time() - start, 2)
                results.append(
                    {
                        "question_id": row["question_id"],
                        "doc_id": row["doc_id"],
                        "doc_name": row["doc_name"],
                        "question": row["question"],
                        "gold_answer": row["gold_answer"],
                        "gold_justification": row.get("gold_justification"),
                        "gold_evidence": _extract_gold_evidence(row),
                        "retrieved_contexts": retrieved_contexts,
                        "final_context": retrieved_contexts,
                        "run_config": _build_run_config(
                            args,
                            effective_style=effective_style,
                            embedding_model=embedding_model,
                        ),
                        "answer": answer,
                        "evidence": evidence,
                        "missing_information": missing_information,
                        "raw_answer": raw_answer,
                        "llm_answer_before_calculation": llm_answer_before_calculation,
                        "calculation_executor": calculation_executor,
                        "retriever_result": retriever_result_dump,
                        "time_seconds": elapsed,
                        "error": None,
                        **style_fields,
                        **query_rewrite_fields,
                    }
                )
                _save_results(output_path, results)
                print(f"  answer={answer!r} gold={row['gold_answer']}", flush=True)
            except Exception as exc:
                elapsed = round(time.time() - start, 2)
                results.append(
                    {
                        "question_id": row["question_id"],
                        "doc_id": row["doc_id"],
                        "doc_name": row["doc_name"],
                        "question": row["question"],
                        "gold_answer": row["gold_answer"],
                        "gold_justification": row.get("gold_justification"),
                        "gold_evidence": _extract_gold_evidence(row),
                        "retrieved_contexts": [],
                        "final_context": [],
                        "run_config": _build_run_config(
                            args,
                            effective_style=effective_style,
                            embedding_model=embedding_model,
                        ),
                        "answer": None,
                        "evidence": None,
                        "missing_information": None,
                        "raw_answer": None,
                        "retriever_result": None,
                        "time_seconds": elapsed,
                        "error": str(exc),
                        **style_fields,
                        **query_rewrite_fields,
                    }
                )
                _save_results(output_path, results)
                print(f"  error={exc}", flush=True)

    _save_results(output_path, results)
    print(f"\nSaved results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
