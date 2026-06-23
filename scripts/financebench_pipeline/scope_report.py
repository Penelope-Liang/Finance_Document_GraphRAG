from __future__ import annotations

from typing import Any


def build_planned_scope(
    *,
    args: Any,
    row: dict[str, Any],
    effective_style: str,
    query_rewrite_plan: dict[str, Any],
    retrieval_filter: dict[str, Any],
) -> dict[str, Any]:
    """Describe the intentional experimental boundary for this question."""
    return {
        "question_style": effective_style,
        "retriever_mode": args.retriever_mode,
        "evidence_boundary": {
            "dataset": "FinanceBench",
            "doc_id": str(row.get("doc_id") or ""),
            "doc_name": row.get("doc_name"),
            "restrict_doc_id": bool(args.restrict_doc_id),
            "retrieval_doc_id": retrieval_filter.get("doc_id"),
            "retrieval_doc_ids": retrieval_filter.get("doc_ids") or [],
            "doc_id_source": retrieval_filter.get("doc_id_source"),
            "company_doc_key": retrieval_filter.get("company_doc_key"),
        },
        "retrieval_boundary": {
            "top_k": args.top_k,
            "effective_search_ratio": args.effective_search_ratio,
            "use_query_rewrite": bool(args.use_query_rewrite),
            "rewritten_query": query_rewrite_plan.get("rewritten_query"),
            "operation": query_rewrite_plan.get("operation"),
            "components": query_rewrite_plan.get("components") or [],
            "statements": query_rewrite_plan.get("statements") or [],
            "years": query_rewrite_plan.get("years") or [],
            "target_filing_year": query_rewrite_plan.get("target_filing_year"),
        },
        "answer_boundary": {
            "use_calculation_executor": bool(args.use_calculation_executor),
            "use_direct_table_fallback": bool(args.use_direct_table_fallback),
            "source_policy": "retrieved_evidence_only",
        },
    }


def build_observed_limitations(
    *,
    planned_scope: dict[str, Any],
    retrieved_contexts: list[dict[str, Any]],
    answer: Any,
    answer_source: str | None,
    missing_information: Any,
    error: str | None = None,
) -> dict[str, Any]:
    """Describe how the planned boundary affected this observed result."""
    limitations: list[dict[str, Any]] = []
    future_work_hints: list[str] = []

    retrieval_boundary = planned_scope["retrieval_boundary"]
    evidence_boundary = planned_scope["evidence_boundary"]
    answer_boundary = planned_scope["answer_boundary"]

    if error:
        limitations.append(
            {
                "type": "execution_failure",
                "scope_choice": "pipeline execution",
                "observed_effect": error,
            }
        )
        future_work_hints.append("Add failure-specific retry or recovery logic.")

    if not retrieved_contexts:
        limitations.append(
            {
                "type": "no_retrieved_context",
                "scope_choice": "retrieval boundary",
                "observed_effect": "No evidence was available inside the configured retrieval scope.",
            }
        )
        future_work_hints.append("Broaden retrieval scope or add a fallback retriever.")

    if evidence_boundary["restrict_doc_id"]:
        limitations.append(
            {
                "type": "single_document_scope",
                "scope_choice": "restrict_doc_id=True",
                "observed_effect": "The answer cannot use evidence outside the dataset-provided filing.",
            }
        )
    elif not evidence_boundary["retrieval_doc_ids"] and not evidence_boundary["company_doc_key"]:
        limitations.append(
            {
                "type": "weak_document_constraint",
                "scope_choice": "unrestricted document retrieval",
                "observed_effect": "The retrieval scope was not anchored to a specific document or company key.",
            }
        )
        future_work_hints.append("Improve company and filing resolution before retrieval.")

    if retrieval_boundary["use_query_rewrite"] and not retrieval_boundary["components"]:
        limitations.append(
            {
                "type": "rewrite_coverage_gap",
                "scope_choice": "metric-component query rewrite",
                "observed_effect": "The rewrite did not identify explicit financial components.",
            }
        )
        future_work_hints.append("Extend rewrite rules or use structured intent extraction.")

    if missing_information:
        limitations.append(
            {
                "type": "insufficient_evidence",
                "scope_choice": "retrieved-evidence-only answer policy",
                "observed_effect": missing_information,
            }
        )
        future_work_hints.append("Retrieve additional statement/table evidence for missing inputs.")

    if answer is None:
        limitations.append(
            {
                "type": "no_answer",
                "scope_choice": "answer generation boundary",
                "observed_effect": "The system did not produce an answer within the available evidence.",
            }
        )

    if answer_source == "calculation_executor":
        limitations.append(
            {
                "type": "post_hoc_calculation",
                "scope_choice": "LLM answer followed by calculation executor",
                "observed_effect": "Final answer depends on deterministic post-processing after generation.",
            }
        )
    elif answer_source == "deterministic_table_cell":
        limitations.append(
            {
                "type": "fallback_answer",
                "scope_choice": "direct table-cell fallback",
                "observed_effect": "The final answer came from fallback extraction rather than the main LLM path.",
            }
        )

    if answer_boundary["source_policy"] == "retrieved_evidence_only":
        limitations.append(
            {
                "type": "closed_book_scope",
                "scope_choice": "retrieved evidence only",
                "observed_effect": "The result should be interpreted only within retrieved FinanceBench evidence.",
            }
        )

    return {
        "limitations": limitations,
        "future_work_hints": sorted(set(future_work_hints)),
    }
