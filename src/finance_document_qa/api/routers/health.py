from __future__ import annotations

from fastapi import APIRouter, Depends

from finance_document_qa.api.deps import APISettings, get_settings
from finance_document_qa.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: APISettings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        profile=settings.app_profile,
        retriever_mode=settings.retriever_mode,
    )
