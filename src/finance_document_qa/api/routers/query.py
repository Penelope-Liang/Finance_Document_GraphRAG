from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from finance_document_qa.api.deps import get_service
from finance_document_qa.api.schemas import QueryRequest, QueryResponse
from finance_document_qa.services.base import BaseFinanceDocumentService, QueryOptions

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    request: Request,
    service: BaseFinanceDocumentService = Depends(get_service),
) -> QueryResponse:
    result = service.query(
        payload.question,
        question_id=payload.question_id,
        doc_id=payload.doc_id,
        doc_name=payload.doc_name,
        question_reasoning=payload.question_reasoning,
        options=QueryOptions(
            top_k=payload.top_k,
            effective_search_ratio=payload.effective_search_ratio,
            question_style=payload.question_style,
            use_query_rewrite=payload.use_query_rewrite,
        ),
    )
    return _to_response(result, request_id=getattr(request.state, "request_id", None))


def _to_response(result: dict[str, Any], *, request_id: str | None) -> QueryResponse:
    metadata = {
        "request_id": request_id,
        "answer_source": result.get("answer_source"),
        "missing_information": result.get("missing_information"),
        "run_config": result.get("run_config"),
        "time_seconds": result.get("time_seconds"),
    }
    return QueryResponse(
        answer=result.get("answer"),
        evidence=result.get("evidence"),
        retrieved_contexts=result.get("retrieved_contexts") or [],
        metadata={key: value for key, value in metadata.items() if value is not None},
        error=result.get("error"),
    )
