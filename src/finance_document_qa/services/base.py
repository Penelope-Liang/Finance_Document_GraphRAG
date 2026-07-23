from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import neo4j
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.llm import AzureOpenAILLM, OllamaLLM
from neo4j_graphrag.rag import GraphRAG
from neo4j_graphrag.retrievers import RerankingRetriever, Text2CypherRetriever
from neo4j_graphrag.retrievers.base import Retriever

from finance_document_qa.config import FinanceDocumentGraphRAGConfig
from finance_document_qa.pipeline.answer_prompts import build_answer_prompt
from finance_document_qa.pipeline.calculation_executor import (
    run_calculation_pipeline,
)
from finance_document_qa.pipeline.cypher_prompts import (
    FINANCIAL_GRAPH_SCHEMA,
    TEXT2CYPHER_EXAMPLES,
    TEXT2CYPHER_EXAMPLES_CALCULATION,
    TEXT2CYPHER_EXAMPLES_NOVEL,
    TEXT2CYPHER_PROMPT,
    TEXT2CYPHER_PROMPT_CALCULATION,
    TEXT2CYPHER_PROMPT_NOVEL,
)
from finance_document_qa.pipeline.direct_fallback import (
    _direct_table_cell_fallback,
)
from finance_document_qa.pipeline.formatting import (
    _compact_retriever_evidence,
    _format_table_record,
    _standardize_retrieved_contexts,
)
from finance_document_qa.pipeline.query_rewriting import (
    _rewrite_metrics_query_for_retrieval,
)
from finance_document_qa.pipeline.question_routing import (
    classify_question_style_auto,
    classify_question_style_llm,
)
from finance_document_qa.pipeline.retrieval_filters import build_retrieval_filter
from finance_document_qa.pipeline.retrievers import (
    LLMJudgeRetriever,
    StyleRoutingText2CypherRetriever,
    _rag_search_with_retry,
)
from finance_document_qa.pipeline.utils import (
    _parse_answer_payload,
    _sanitize_for_json,
)


@dataclass
class QueryOptions:
    """Per-query knobs (the CLI-style flags that can change per question).

    Build-time settings (LLM backend, reranker, calculation executor, etc.) live
    on the service instance instead; these are only the switches that may vary
    from one request to the next.
    """

    top_k: int = 5
    effective_search_ratio: int = 10
    question_style: str = "direct"
    use_query_rewrite: bool = False
    use_gold_doc_id: bool = False
    use_direct_table_fallback: bool = False
    llm_retries: int = 6
    llm_retry_sleep: float = 1.5


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing env var: {name}")
    return value


class BaseFinanceDocumentService:
    """Callable FinanceBench QA service.

    Holds the expensive, build-once objects (Neo4j driver, answer/calc/judge
    LLMs, embedder, retriever, GraphRAG) and exposes a single ``query`` method
    that runs the full per-question pipeline. Subclasses only implement
    ``_build_core_retriever`` to decide which retriever the mode uses.
    """

    #: Overridden by subclasses ("hybrid", "dual", "text2cypher").
    retriever_mode: str = "dual"

    def __init__(
        self,
        config: FinanceDocumentGraphRAGConfig | None = None,
        *,
        llm_backend: str = "ollama",
        llm_model: str = "qwen3:30b-a3b",
        ollama_num_ctx: int | None = None,
        use_reranker: bool = False,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        reranker_top_k_retrieve: int = 20,
        use_llm_judge: bool = False,
        llm_judge_model: str = "qwen3:14b",
        llm_judge_max_chars: int = 1200,
        use_calculation_executor: bool = False,
        calculation_model: str = "qwen2.5-coder:14b-instruct",
        driver: neo4j.Driver | None = None,
    ) -> None:
        self.config = config or FinanceDocumentGraphRAGConfig.from_env()
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.ollama_num_ctx = ollama_num_ctx
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.reranker_top_k_retrieve = reranker_top_k_retrieve
        self.use_llm_judge = use_llm_judge
        self.llm_judge_model = llm_judge_model
        self.llm_judge_max_chars = llm_judge_max_chars
        self.use_calculation_executor = use_calculation_executor
        self.calculation_model = calculation_model
        self._available_documents: list[str] | None = None

        self.driver = driver or neo4j.GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, _require_env("NEO4J_PASSWORD")),
        )

        self.llm = self._build_answer_llm()
        self.embedder = OllamaEmbeddings(
            model=self.config.embedding_model, **self._ollama_kwargs()
        )
        self.calc_llm = (
            OllamaLLM(
                model_name=self.calculation_model,
                model_params={"options": self._ollama_options()},
                **self._ollama_kwargs(),
            )
            if self.use_calculation_executor
            else None
        )
        self.judge_llm = (
            OllamaLLM(
                model_name=self.llm_judge_model,
                model_params={"options": self._ollama_options()},
                **self._ollama_kwargs(),
            )
            if self.use_llm_judge
            else None
        )

        core = self._build_core_retriever()
        if self.use_reranker and self.retriever_mode != "dual":
            core = RerankingRetriever(
                inner_retriever=core,
                cross_encoder_model=self.reranker_model,
                top_k_retrieve=self.reranker_top_k_retrieve,
            )
        if self.use_llm_judge and self.judge_llm is not None:
            core = LLMJudgeRetriever(
                inner_retriever=core,
                llm=self.judge_llm,
                judge_max_chars=self.llm_judge_max_chars,
                neo4j_database=self.config.neo4j_database,
            )
        self.retriever = core
        self.rag = GraphRAG(retriever=self.retriever, llm=self.llm)

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------
    def _build_core_retriever(self) -> Retriever:
        raise NotImplementedError(
            "Subclasses must implement _build_core_retriever() to select the mode's retriever."
        )

    # ------------------------------------------------------------------
    # Shared builders
    # ------------------------------------------------------------------
    def _ollama_kwargs(self) -> dict[str, Any]:
        return {"host": self.config.ollama_host} if self.config.ollama_host else {}

    def _ollama_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": 0}
        if self.ollama_num_ctx is not None:
            options["num_ctx"] = self.ollama_num_ctx
        return options

    def _build_answer_llm(self) -> Any:
        if self.llm_backend == "azure-openai":
            return AzureOpenAILLM(
                model_name=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"),
                model_params={"temperature": 0},
                azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
                api_key=_require_env("AZURE_OPENAI_API_KEY"),
                api_version=os.environ.get(
                    "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
                ),
            )
        return OllamaLLM(
            model_name=self.llm_model,
            model_params={"options": self._ollama_options()},
            **self._ollama_kwargs(),
        )

    def _build_table_router(self) -> StyleRoutingText2CypherRetriever:
        """Shared Text2Cypher style router (used by dual and text2cypher modes)."""
        t2c_direct = Text2CypherRetriever(
            driver=self.driver,
            llm=self.llm,
            neo4j_database=self.config.neo4j_database,
            neo4j_schema=FINANCIAL_GRAPH_SCHEMA,
            examples=TEXT2CYPHER_EXAMPLES,
            custom_prompt=TEXT2CYPHER_PROMPT,
            result_formatter=_format_table_record,
        )
        t2c_calculation = Text2CypherRetriever(
            driver=self.driver,
            llm=self.llm,
            neo4j_database=self.config.neo4j_database,
            neo4j_schema=FINANCIAL_GRAPH_SCHEMA,
            examples=TEXT2CYPHER_EXAMPLES_CALCULATION,
            custom_prompt=TEXT2CYPHER_PROMPT_CALCULATION,
            result_formatter=_format_table_record,
        )
        t2c_novel = Text2CypherRetriever(
            driver=self.driver,
            llm=self.llm,
            neo4j_database=self.config.neo4j_database,
            neo4j_schema=FINANCIAL_GRAPH_SCHEMA,
            examples=TEXT2CYPHER_EXAMPLES_NOVEL,
            custom_prompt=TEXT2CYPHER_PROMPT_NOVEL,
            result_formatter=_format_table_record,
        )
        return StyleRoutingText2CypherRetriever(t2c_direct, t2c_calculation, t2c_novel)

    def _build_run_config(
        self, options: QueryOptions, *, effective_style: str
    ) -> dict[str, Any]:
        return {
            "retriever_mode": self.retriever_mode,
            "question_style": effective_style,
            "top_k": options.top_k,
            "effective_search_ratio": options.effective_search_ratio,
            "use_query_rewrite": bool(options.use_query_rewrite),
            "use_gold_doc_id": bool(options.use_gold_doc_id),
            "use_calculation_executor": bool(self.use_calculation_executor),
            "llm_backend": self.llm_backend,
            "llm_model": (
                os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
                if self.llm_backend == "azure-openai"
                else self.llm_model
            ),
            "embedding_model": self.config.embedding_model,
            "ollama_num_ctx": (
                self.ollama_num_ctx if self.llm_backend == "ollama" else None
            ),
            "calculation_model": (
                self.calculation_model if self.use_calculation_executor else None
            ),
        }

    @staticmethod
    def _requires_calculation_executor(
        question_reasoning: str | None, effective_style: str
    ) -> bool:
        if str(question_reasoning or "").strip().lower() == "numerical reasoning":
            return True
        return effective_style == "calculation"

    # ------------------------------------------------------------------
    # Graph inspection
    # ------------------------------------------------------------------
    def list_documents(self) -> list[str]:
        """Return the distinct ``doc_id`` values present in the graph."""
        cypher = (
            "MATCH (c:Chunk) WHERE c.doc_id IS NOT NULL "
            "RETURN DISTINCT c.doc_id AS doc_id ORDER BY doc_id"
        )
        with self.driver.session(database=self.config.neo4j_database) as session:
            return [str(record["doc_id"]) for record in session.run(cypher)]

    @property
    def available_documents(self) -> list[str]:
        if self._available_documents is None:
            self._available_documents = self.list_documents()
        return self._available_documents

    # ------------------------------------------------------------------
    # Core single-question pipeline
    # ------------------------------------------------------------------
    def _classify_style(
        self, question_text: str, question_reasoning: str | None, mode: str
    ) -> dict[str, Any]:
        if mode == "auto":
            effective_style, conf, reason = classify_question_style_auto(question_text)
            raw = None
        elif mode == "llm-auto":
            effective_style, conf, reason, raw = classify_question_style_llm(
                self.llm, question_text
            )
        else:
            effective_style, conf, reason, raw = mode, 1.0, "cli_fixed", None

        if str(question_reasoning or "").strip().lower() == "numerical reasoning":
            if effective_style != "calculation":
                reason = (
                    f"dataset_question_reasoning=numerical_reasoning; router={reason}"
                )
                conf = 1.0
            effective_style = "calculation"

        return {
            "effective_style": effective_style,
            "fields": {
                "question_style_mode": mode,
                "question_style": effective_style,
                "question_style_confidence": round(float(conf), 4),
                "question_style_reason": reason,
                "question_style_raw": raw,
            },
        }

    def query(
        self,
        question: str,
        *,
        question_id: str | None = None,
        doc_id: str | None = None,
        doc_name: str | None = None,
        question_reasoning: str | None = None,
        available_doc_ids: list[str] | None = None,
        options: QueryOptions | None = None,
    ) -> dict[str, Any]:
        """Run the full retrieval + answer pipeline for a single question."""
        options = options or QueryOptions()
        row = {
            "question": question,
            "doc_id": doc_id,
            "doc_name": doc_name or "",
            "question_reasoning": question_reasoning,
        }
        base_record: dict[str, Any] = {
            "question_id": question_id,
            "doc_id": doc_id,
            "doc_name": doc_name,
            "question": question,
        }

        start = time.time()
        try:
            doc_ids_scope = (
                available_doc_ids
                if available_doc_ids is not None
                else self.available_documents
            )

            style = self._classify_style(
                question, question_reasoning, options.question_style
            )
            effective_style = style["effective_style"]
            style_fields = style["fields"]

            query_rewrite_plan = (
                _rewrite_metrics_query_for_retrieval(
                    question,
                    question_style=effective_style,
                    doc_name=doc_name or "",
                    llm=self.llm,
                )
                if options.use_query_rewrite
                else {
                    "rewritten_query": question,
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
            query_rewrite_fields: dict[str, Any] = {
                "query_rewrite_applied": bool(options.use_query_rewrite),
                "retrieval_query": query_rewrite_plan["rewritten_query"],
                "query_rewrite_company": query_rewrite_plan["company"],
                "query_rewrite_company_source": query_rewrite_plan["company_source"],
                "query_rewrite_company_doc_key": query_rewrite_plan["company_doc_key"],
                "query_rewrite_years": query_rewrite_plan["years"],
                "query_rewrite_target_filing_year": query_rewrite_plan.get(
                    "target_filing_year"
                ),
                "query_rewrite_operation": query_rewrite_plan["operation"],
                "query_rewrite_components": query_rewrite_plan["components"],
                "query_rewrite_statements": query_rewrite_plan["statements"],
            }

            retrieval_query_text = query_rewrite_plan["rewritten_query"]
            retrieval_filter = build_retrieval_filter(
                row=row,
                query_rewrite_plan=query_rewrite_plan,
                use_query_rewrite=options.use_query_rewrite,
                use_gold_doc_id=options.use_gold_doc_id,
                available_doc_ids=doc_ids_scope,
            )
            query_rewrite_fields["retrieval_doc_id"] = retrieval_filter.get("doc_id")
            query_rewrite_fields["retrieval_doc_ids"] = retrieval_filter.get("doc_ids")
            query_rewrite_fields["retrieval_doc_id_source"] = retrieval_filter.get(
                "doc_id_source"
            )
            query_rewrite_fields["retrieval_company_doc_key"] = retrieval_filter.get(
                "company_doc_key"
            )

            base_record.update(
                {
                    "run_config": self._build_run_config(
                        options, effective_style=effective_style
                    ),
                    **style_fields,
                    **query_rewrite_fields,
                }
            )

            search_kwargs: dict[str, Any] = {
                "query_text": retrieval_query_text,
                "return_context": True,
                "examples": build_answer_prompt(question, effective_style),
            }
            retriever_config: dict[str, Any] = {}
            if self.retriever_mode in {"hybrid", "dual"}:
                retriever_config = {
                    "top_k": options.top_k,
                    "effective_search_ratio": options.effective_search_ratio,
                    "query_params": retrieval_filter,
                }
            if self.retriever_mode == "dual":
                retriever_config["t2c_style"] = effective_style
                if options.use_query_rewrite:
                    retriever_config["retrieval_query_text"] = retrieval_query_text
                    if effective_style in {"calculation", "domain"}:
                        retriever_config["metrics_doc_id"] = retrieval_filter["doc_id"]
                        retriever_config["metrics_doc_ids"] = (
                            retrieval_filter.get("doc_ids") or []
                        )
                        retriever_config["metrics_company_doc_key"] = retrieval_filter[
                            "company_doc_key"
                        ]
                        retriever_config["metrics_plan"] = query_rewrite_plan
            elif self.retriever_mode == "text2cypher":
                retriever_config["t2c_style"] = effective_style
            if retriever_config:
                search_kwargs["retriever_config"] = retriever_config

            result = _rag_search_with_retry(
                self.rag,
                max_retries=options.llm_retries,
                retry_sleep=options.llm_retry_sleep,
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
                question_id=str(question_id) if question_id is not None else "",
            )

            answer_source = "llm"
            llm_answer_before_calculation = None
            calculation_executor = None
            should_run_calculation_executor = self._requires_calculation_executor(
                question_reasoning, effective_style
            )
            if (
                self.use_calculation_executor
                and should_run_calculation_executor
                and self.calc_llm is not None
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
                calculation_executor = run_calculation_pipeline(
                    llm=self.calc_llm,
                    question=question,
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
                    answer_source = "calculation_executor"

            direct_table_fallback = None
            if (
                options.use_direct_table_fallback
                and answer is None
                and effective_style == "direct"
            ):
                fallback_answer, direct_table_fallback = _direct_table_cell_fallback(
                    question,
                    retrieved_contexts,
                )
                if fallback_answer is not None:
                    answer = fallback_answer
                    evidence = (
                        f"{evidence}\n\n" if evidence else ""
                    ) + "Deterministic direct extraction from TableCell."
                    missing_information = None
                    answer_source = "deterministic_table_cell"

            elapsed = round(time.time() - start, 2)
            base_record.update(
                {
                    "retrieved_contexts": retrieved_contexts,
                    "final_context": retrieved_contexts,
                    "answer": answer,
                    "answer_source": answer_source,
                    "direct_table_fallback": direct_table_fallback,
                    "evidence": evidence,
                    "missing_information": missing_information,
                    "raw_answer": raw_answer,
                    "llm_answer_before_calculation": llm_answer_before_calculation,
                    "calculation_executor": calculation_executor,
                    "retriever_result": retriever_result_dump,
                    "time_seconds": elapsed,
                    "error": None,
                }
            )
            return base_record
        except Exception as exc:  # noqa: BLE001 - mirror batch-runner robustness
            elapsed = round(time.time() - start, 2)
            base_record.update(
                {
                    "retrieved_contexts": [],
                    "final_context": [],
                    "answer": None,
                    "answer_source": None,
                    "direct_table_fallback": None,
                    "evidence": None,
                    "missing_information": None,
                    "raw_answer": None,
                    "llm_answer_before_calculation": None,
                    "calculation_executor": None,
                    "retriever_result": None,
                    "time_seconds": elapsed,
                    "error": str(exc),
                }
            )
            return base_record

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> "BaseFinanceDocumentService":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
