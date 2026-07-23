from __future__ import annotations

from typing import Any

from finance_document_qa.pipeline.filing_constraints import (
    _candidate_filing_doc_ids,
)


def build_retrieval_filter(
    *,
    row: dict[str, Any],
    query_rewrite_plan: dict[str, Any],
    use_query_rewrite: bool,
    use_gold_doc_id: bool,
    available_doc_ids: list[str],
) -> dict[str, Any]:
    """Shared document/company constraints for every retrieval branch."""
    if use_gold_doc_id:
        gold_doc_id = str(row.get("doc_id") or "").strip()
        if not gold_doc_id:
            raise ValueError("use_gold_doc_id=True requires dataset doc_id.")
        return {
            "doc_id": gold_doc_id,
            "doc_ids": [gold_doc_id],
            "company_doc_key": None,
            "doc_id_source": "gold_doc_id",
        }

    candidate_doc_ids, company_doc_key, doc_id_source = _candidate_filing_doc_ids(
        question=str(row.get("question") or ""),
        query_rewrite_plan=query_rewrite_plan,
        available_doc_ids=available_doc_ids,
    )
    if candidate_doc_ids:
        return {
            "doc_id": candidate_doc_ids[0] if len(candidate_doc_ids) == 1 else None,
            "doc_ids": candidate_doc_ids,
            "company_doc_key": None,
            "doc_id_source": doc_id_source,
        }

    return {
        "doc_id": None,
        "doc_ids": [],
        "company_doc_key": (
            company_doc_key or query_rewrite_plan.get("company_doc_key")
            if use_query_rewrite
            else None
        ),
        "doc_id_source": None,
    }
