from __future__ import annotations

import json
import time
from typing import Any

import neo4j
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.rag import GraphRAG
from neo4j_graphrag.retrievers import HybridCypherRetriever, Text2CypherRetriever
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import RawSearchResult, RetrieverResult, RetrieverResultItem

from financebench_pipeline.filing_constraints import _extract_question_constraints
from financebench_pipeline.formatting import _format_chunk_record, _format_table_record
from financebench_pipeline.fusion import (
    _attach_fusion_scores,  # noqa: F401  # kept for commented-out soft-fusion fallback
    _evidence_bucket,
    _item_base_score,
)
from financebench_pipeline.query_rewriting import (
    _METRICS_TABLE_TARGETS,
    _metrics_component_targets,
)
from financebench_pipeline.question_routing import _detect_statement_type
from financebench_pipeline.retrieval_queries import METRICS_TABLE_CELL_QUERY, PRIMARY_STATEMENT_QUERY
from financebench_pipeline.utils import (
    _extract_json_object,
    _safe_float,
    _sanitize_retriever_item,
    _sanitize_retriever_result,
    _sanitize_value,
)




class MetricsTableCellRetriever(Retriever):
    VERIFY_NEO4J_VERSION = False

    def __init__(
        self,
        driver: neo4j.Driver,
        neo4j_database: str | None = None,
        limit: int = 80,
        per_target_limit: int = 12,
    ) -> None:
        super().__init__(driver, neo4j_database)
        self.limit = limit
        self.per_target_limit = per_target_limit

    def get_search_results(self, *args: Any, **kwargs: Any) -> RawSearchResult:
        raise NotImplementedError("MetricsTableCellRetriever uses a custom search implementation.")

    def search(
        self,
        query_text: str,
        doc_id: str | None = None,
        doc_ids: list[str] | None = None,
        company_doc_key: str | None = None,
        years: list[str] | None = None,
        plan: dict[str, Any] | None = None,
    ) -> RetrieverResult:
        target_names = _metrics_component_targets(query_text, plan)
        if not target_names:
            return RetrieverResult(
                items=[],
                metadata={
                    "__retriever": "MetricsTableCellRetriever",
                    "matched": False,
                    "reason": "missing targets",
                    "doc_id": doc_id,
                    "doc_ids": list(doc_ids or []),
                    "company_doc_key": company_doc_key,
                    "target_names": target_names,
                },
            )

        targets = [
            {
                "name": name,
                "aliases": list(_METRICS_TABLE_TARGETS[name]["aliases"]),
                "section_hints": list(_METRICS_TABLE_TARGETS[name]["section_hints"]),
            }
            for name in target_names
            if name in _METRICS_TABLE_TARGETS
        ]
        params = {
            "doc_id": doc_id,
            "doc_ids": list(doc_ids or []),
            "company_doc_key": company_doc_key,
            "years": list(years or []),
            "targets": targets,
            "limit": self.limit,
            "per_target_limit": self.per_target_limit,
        }
        with self.driver.session(database=self.neo4j_database) as session:
            records = list(session.run(METRICS_TABLE_CELL_QUERY, params))

        items: list[RetrieverResultItem] = []
        for record in records:
            item = _format_table_record(record)
            metadata = dict(item.metadata or {})
            metadata["source"] = "metrics_table"
            metadata["target_name"] = _sanitize_value(record.get("target_name"))
            metadata["section_hit"] = int(record.get("section_hit") or 0)
            metadata["row_hit"] = int(record.get("row_hit") or 0)
            metadata["year_hit"] = int(record.get("year_hit") or 0)
            content = f"Metric target: {record.get('target_name')}\n{item.content}"
            items.append(
                _sanitize_retriever_item(
                    RetrieverResultItem(content=content, metadata=metadata)
                )
            )

        return _sanitize_retriever_result(RetrieverResult(
            items=items,
            metadata={
                "__retriever": "MetricsTableCellRetriever",
                "matched": bool(items),
                "doc_id": doc_id,
                "doc_id_filter_applied": bool(doc_id),
                "doc_ids": list(doc_ids or []),
                "doc_ids_filter_applied": bool(doc_ids),
                "company_doc_key": company_doc_key,
                "company_filter_applied": bool(company_doc_key),
                "years": list(years or []),
                "target_names": target_names,
                "kept": len(items),
            },
        ))


class StyleRoutingText2CypherRetriever(Retriever):
    """Routes Text2Cypher retrieval by question style."""

    VERIFY_NEO4J_VERSION = False

    def __init__(
        self,
        retriever_direct: Text2CypherRetriever,
        retriever_calculation: Text2CypherRetriever,
        retriever_novel: Text2CypherRetriever | None = None,
    ) -> None:
        super().__init__(
            retriever_direct.driver,
            getattr(retriever_direct, "neo4j_database", None),
        )
        self._direct = retriever_direct
        self._calc = retriever_calculation
        self._novel = retriever_novel or retriever_direct

    def get_search_results(self, *args: Any, **kwargs: Any) -> RawSearchResult:
        raise NotImplementedError("StyleRoutingText2CypherRetriever delegates to inner retrievers.")

    def search(self, query_text: str, t2c_style: str = "direct", **kwargs: Any) -> RetrieverResult:
        if t2c_style == "calculation":
            inner = self._calc
        elif t2c_style in {"novel", "domain"}:
            inner = self._novel
        else:
            inner = self._direct
        return _sanitize_retriever_result(inner.search(query_text=query_text))


def _format_judge_candidate(index: int, item: RetrieverResultItem, max_chars: int = 1200) -> str:
    metadata = item.metadata or {}
    section_name = str(metadata.get("section_name") or "")
    source = str(metadata.get("source") or "")
    row_header = str(metadata.get("row_header") or "")
    col_header = str(metadata.get("col_header") or "")
    content = str(item.content or "")[:max_chars]
    return (
        f"[ID: {index}]\n"
        f"Source: {source}\n"
        f"Section: {section_name}\n"
        f"Row Header: {row_header}\n"
        f"Col Header: {col_header}\n"
        f"Content:\n{content}"
    )


def _judge_prompt(query_text: str, items: list[RetrieverResultItem], max_chars: int) -> str:
    constraints = _extract_question_constraints(query_text)
    candidates_text = "\n\n".join(
        _format_judge_candidate(i, item, max_chars=max_chars)
        for i, item in enumerate(items)
    )
    hints = []
    if constraints["section_hints"]:
        hints.append(f"Section/table hints: {', '.join(constraints['section_hints'])}")
    if constraints["years"]:
        hints.append(f"Years: {', '.join(constraints['years'])}")
    if constraints["quarters"]:
        hints.append(f"Quarters: {', '.join('Q' + q for q in constraints['quarters'])}")
    if constraints["months"]:
        hints.append(f"Months: {', '.join(constraints['months'])}")
    if constraints["period_hints"]:
        hints.append(f"Period hints: {', '.join(constraints['period_hints'])}")
    hint_block = "\n".join(f"- {line}" for line in hints) if hints else "- No explicit structured hints extracted."
    return f"""
You are filtering retrieved evidence for a financial question.

Task:
- Keep all candidates that are not clearly irrelevant.
- Prefer false positives over false negatives.
- Prioritize candidates matching the question's statement/section/table hints and time constraints.
- A candidate does not need to directly answer the question; keep it if it may help answer.

Question:
{query_text}

Extracted constraints:
{hint_block}

Candidate evidence:
{candidates_text}

Return only a JSON object with this exact schema:
{{"ids": [0, 2, 5]}}

Rules:
- IDs must be chosen only from the listed candidates.
- Return at least 5 IDs if 5 or more candidates are available, unless almost everything is clearly irrelevant.
- Return an empty list only if no candidate is plausibly useful.
""".strip()


class LLMJudgeRetriever(Retriever):
    VERIFY_NEO4J_VERSION = False

    def __init__(
        self,
        inner_retriever: Retriever,
        llm: OllamaLLM,
        judge_max_chars: int = 1200,
        neo4j_database: str | None = None,
    ) -> None:
        super().__init__(
            driver=inner_retriever.driver,
            neo4j_database=neo4j_database or getattr(inner_retriever, "neo4j_database", None),
        )
        self.inner_retriever = inner_retriever
        self.llm = llm
        self.judge_max_chars = judge_max_chars

    def get_search_results(self, *args: Any, **kwargs: Any) -> RawSearchResult:
        raise NotImplementedError("LLMJudgeRetriever wraps search(), not raw Neo4j records.")

    def search(self, *args: Any, **kwargs: Any) -> RetrieverResult:
        result = _sanitize_retriever_result(self.inner_retriever.search(*args, **kwargs))
        if not result.items:
            return result

        query_text = kwargs.get("query_text") or (args[0] if args else "")
        if not query_text:
            return _sanitize_retriever_result(RetrieverResult(
                items=result.items,
                metadata={**(result.metadata or {}), "llm_judge_applied": False},
            ))

        prompt = _judge_prompt(query_text, result.items, self.judge_max_chars)
        try:
            response = self.llm.invoke(prompt)
            payload_text = _extract_json_object(response.content) or response.content
            payload = json.loads(payload_text)
            ids = payload.get("ids", []) if isinstance(payload, dict) else []
            if not isinstance(ids, list):
                ids = []
            judged_items = [
                result.items[i]
                for i in ids
                if isinstance(i, int) and 0 <= i < len(result.items)
            ]
            if not judged_items:
                judged_items = result.items
            return _sanitize_retriever_result(RetrieverResult(
                items=judged_items,
                metadata={
                    **(result.metadata or {}),
                    "llm_judge_applied": True,
                    "llm_judge_kept": len(judged_items),
                },
            ))
        except Exception:
            return _sanitize_retriever_result(RetrieverResult(
                items=result.items,
                metadata={**(result.metadata or {}), "llm_judge_applied": False},
            ))


def _is_retryable_llm_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "unsupported value: nan",
            "json: unsupported value",
            "status code: 500",
            "failed to encode response",
            "internal server error",
            "connection reset",
            "broken pipe",
            "timeout",
        )
    )


def _rag_search_with_retry(
    rag: GraphRAG,
    *,
    max_retries: int,
    retry_sleep: float,
    **search_kwargs: Any,
) -> Any:
    """Retry whole RAG search (retrieval + answer LLM). Ollama sometimes returns 500
    ``json: unsupported value: NaN`` on flaky GPU runs; extra attempts often succeed."""
    attempt = 0
    while True:
        try:
            return rag.search(**search_kwargs)
        except Exception as exc:
            if attempt >= max_retries or not _is_retryable_llm_error(exc):
                raise
            attempt += 1
            print(f"  retryable LLM error ({attempt}/{max_retries}): {exc}", flush=True)
            if retry_sleep > 0:
                # Longer backoff for NaN/encode failures (often need GPU to settle).
                msg = str(exc).lower()
                mult = 2.0 if ("nan" in msg or "encode" in msg) else 1.0
                time.sleep(retry_sleep * mult * attempt)




class PrimaryStatementRetriever(Retriever):
    """Deterministically pulls primary financial statement chunks
    (balance sheet / income statement / cash flow / equity) by section_name
    pattern matching, bypassing vector similarity which often ranks MD&A
    discussions above the actual statements."""

    VERIFY_NEO4J_VERSION = False

    def __init__(
        self,
        driver: neo4j.Driver,
        neo4j_database: str | None = None,
        limit: int = 8,
    ) -> None:
        super().__init__(driver, neo4j_database)
        self.limit = limit

    def get_search_results(self, *args: Any, **kwargs: Any) -> RawSearchResult:
        raise NotImplementedError("PrimaryStatementRetriever uses a custom search implementation.")

    def search(
        self,
        query_text: str,
        query_params: dict[str, Any] | None = None,
    ) -> RetrieverResult:
        spec = _detect_statement_type(query_text)
        doc_id = (query_params or {}).get("doc_id") if query_params else None
        doc_ids = list((query_params or {}).get("doc_ids") or []) if query_params else []
        company_doc_key = (query_params or {}).get("company_doc_key") if query_params else None

        if spec is None:
            return RetrieverResult(
                items=[],
                metadata={
                    "__retriever": "PrimaryStatementRetriever",
                    "matched": False,
                    "reason": "no statement type detected",
                },
            )

        params = {
            "doc_id": doc_id,
            "doc_ids": doc_ids,
            "company_doc_key": company_doc_key,
            "statement_type": spec["name"],
            "includes": list(spec["includes"]),
            "excludes": list(spec["excludes"]),
            "limit": self.limit,
        }
        with self.driver.session(database=self.neo4j_database) as session:
            records = list(session.run(PRIMARY_STATEMENT_QUERY, params))

        items = []
        for record in records:
            item = _format_chunk_record(record)
            metadata = dict(item.metadata or {})
            metadata["source"] = "primary_statement"
            metadata["statement_type"] = spec["name"]
            metadata["chunk_statement_type"] = _sanitize_value(record.get("chunk_statement_type"))
            metadata["match_priority"] = int(record.get("match_priority") or 0)
            items.append(
                _sanitize_retriever_item(
                    RetrieverResultItem(content=item.content, metadata=metadata)
                )
            )

        return _sanitize_retriever_result(RetrieverResult(
            items=items,
            metadata={
                "__retriever": "PrimaryStatementRetriever",
                "matched": True,
                "statement_type": spec["name"],
                "doc_id": doc_id,
                "doc_id_filter_applied": bool(doc_id),
                "doc_ids": doc_ids,
                "doc_ids_filter_applied": bool(doc_ids),
                "company_doc_key": company_doc_key,
                "company_filter_applied": bool(company_doc_key),
                "kept": len(items),
            },
        ))


class DualRetriever(Retriever):
    VERIFY_NEO4J_VERSION = False

    def __init__(
        self,
        driver: neo4j.Driver,
        chunk_retriever: HybridCypherRetriever,
        table_retriever: Text2CypherRetriever | StyleRoutingText2CypherRetriever,
        primary_retriever: PrimaryStatementRetriever | None = None,
        metrics_retriever: MetricsTableCellRetriever | None = None,
        neo4j_database: str | None = None,
    ) -> None:
        super().__init__(driver, neo4j_database)
        self.chunk_retriever = chunk_retriever
        self.table_retriever = table_retriever
        self.primary_retriever = primary_retriever
        self.metrics_retriever = metrics_retriever

    def get_search_results(self, *args: Any, **kwargs: Any) -> RawSearchResult:
        raise NotImplementedError("DualRetriever uses a custom search implementation.")

    @staticmethod
    def _failed_result(retriever_name: str, exc: Exception) -> RetrieverResult:
        return RetrieverResult(
            items=[],
            metadata={
                "__retriever": retriever_name,
                "matched": False,
                "error": str(exc),
            },
        )

    @staticmethod
    def _filter_result_doc_ids(result: RetrieverResult, doc_ids: list[str]) -> RetrieverResult:
        if not doc_ids:
            return result
        allowed = set(doc_ids)
        kept = []
        dropped = 0
        for item in result.items:
            item_doc_id = (item.metadata or {}).get("doc_id")
            if item_doc_id and item_doc_id not in allowed:
                dropped += 1
                continue
            kept.append(item)
        metadata = dict(result.metadata or {})
        metadata["post_doc_ids_filter"] = list(doc_ids)
        metadata["post_doc_id_filter_dropped"] = dropped
        return RetrieverResult(items=kept, metadata=metadata)

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        effective_search_ratio: int = 10,
        query_params: dict[str, Any] | None = None,
        t2c_style: str = "direct",
        retrieval_query_text: str | None = None,
        metrics_doc_id: str | None = None,
        metrics_doc_ids: list[str] | None = None,
        metrics_company_doc_key: str | None = None,
        metrics_plan: dict[str, Any] | None = None,
    ) -> RetrieverResult:
        search_text = retrieval_query_text or query_text
        target_doc_id = (query_params or {}).get("doc_id") if query_params else None
        target_doc_ids = list((query_params or {}).get("doc_ids") or []) if query_params else []
        table_search_text = (
            f"{search_text}\nTarget filing doc_ids: {', '.join(target_doc_ids)}"
            if target_doc_ids
            else search_text
        )
        try:
            metrics_result = (
                self.metrics_retriever.search(
                    query_text=query_text,
                    doc_id=metrics_doc_id,
                    doc_ids=list(metrics_doc_ids or []),
                    company_doc_key=metrics_company_doc_key,
                    years=list((metrics_plan or {}).get("years") or []),
                    plan=metrics_plan,
                )
                if self.metrics_retriever is not None
                and t2c_style in {"calculation", "domain"}
                and metrics_plan is not None
                else RetrieverResult(
                    items=[],
                    metadata={"__retriever": "MetricsTableCellRetriever", "matched": False},
                )
            )
        except Exception as exc:
            metrics_result = self._failed_result("MetricsTableCellRetriever", exc)
        metrics_result = _sanitize_retriever_result(metrics_result)

        try:
            chunk_result = self.chunk_retriever.search(
                query_text=search_text,
                top_k=top_k,
                effective_search_ratio=effective_search_ratio,
                query_params=query_params,
            )
        except Exception as exc:
            chunk_result = self._failed_result("HybridCypherRetriever", exc)
        chunk_result = _sanitize_retriever_result(chunk_result)

        try:
            if isinstance(self.table_retriever, StyleRoutingText2CypherRetriever):
                table_result = self.table_retriever.search(
                    query_text=table_search_text, t2c_style=t2c_style
                )
            else:
                table_result = self.table_retriever.search(query_text=table_search_text)
        except Exception as exc:
            table_result = self._failed_result("Text2CypherRetriever", exc)
        table_result = _sanitize_retriever_result(table_result)
        table_result = self._filter_result_doc_ids(table_result, target_doc_ids)

        try:
            primary_result = (
                self.primary_retriever.search(query_text=search_text, query_params=query_params)
                if self.primary_retriever is not None
                else RetrieverResult(
                    items=[],
                    metadata={"__retriever": "PrimaryStatementRetriever", "matched": False},
                )
            )
        except Exception as exc:
            primary_result = self._failed_result("PrimaryStatementRetriever", exc)
        primary_result = _sanitize_retriever_result(primary_result)

        items: list[RetrieverResultItem] = []
        seen_contents: set[str] = set()

        # Order matters: structured metric cells first, then primary statements,
        # so deterministic evidence wins ties on dedup before semantic retrieval.
        for item in metrics_result.items + primary_result.items + table_result.items + chunk_result.items:
            content = str(item.content)
            if content in seen_contents:
                continue
            seen_contents.add(content)
            items.append(item)

        # --- OLD soft-fusion ranking (tunable magic-number weights); kept for ablation ---
        # fusion_question = query_text if search_text == query_text else f"{query_text}\n{search_text}"
        # items = _attach_fusion_scores(fusion_question, items)
        # items.sort(
        #     key=lambda item: _safe_float((item.metadata or {}).get("fused_score")),
        #     reverse=True,
        # )

        # --- NEW deterministic constraint-bucket ranking ---
        # Hard financial constraints (doc / metric / period / numeric cell) decide
        # the bucket (tier); a simple per-source base score breaks ties within a
        # bucket. No tunable fusion weights, no reranker. Items are demoted, never
        # dropped, so recall is preserved.
        requested_years = list((metrics_plan or {}).get("years") or [])
        if not requested_years:
            requested_years = list(_extract_question_constraints(search_text)["years"])

        ranked: list[tuple[int, int, float, RetrieverResultItem]] = []
        for item in items:
            priority, bucket, features = _evidence_bucket(
                item,
                target_doc_ids,
                requested_years=requested_years,
            )
            base_score = _safe_float(_item_base_score(item))
            source_priority = int(features.get("source_priority") or 0)
            metadata = dict(item.metadata or {})
            metadata["bucket"] = bucket
            metadata["bucket_priority"] = priority
            metadata["bucket_features"] = features
            metadata["source_priority"] = source_priority
            metadata["base_score"] = base_score
            ranked.append(
                (
                    priority,
                    source_priority,
                    base_score,
                    _sanitize_retriever_item(
                        RetrieverResultItem(content=item.content, metadata=metadata)
                    ),
                )
            )
        ranked.sort(key=lambda entry: (entry[0], entry[1], entry[2]), reverse=True)
        items = [item for _, _, _, item in ranked]

        return _sanitize_retriever_result(RetrieverResult(
            items=items,
            metadata={
                "__retriever": "DualRetriever",
                "metrics_retriever": metrics_result.metadata,
                "chunk_retriever": chunk_result.metadata,
                "table_retriever": table_result.metadata,
                "primary_retriever": primary_result.metadata,
                "retrieval_query_text": search_text,
                "ranking": {
                    "strategy": "constraint_bucket_then_base_score",
                    "bucket_priority": {
                        "exact_cell": 5,
                        "metric_cell": 4,
                        "statement": 3,
                        "document": 2,
                        "weak_or_mismatch": 1,
                    },
                    "tie_break": "source_priority_then_per_source_base_score",
                    "requested_years": requested_years,
                },
            },
        ))
