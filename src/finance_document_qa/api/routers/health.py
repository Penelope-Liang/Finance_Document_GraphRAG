from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Response

from finance_document_qa.api.deps import APISettings, get_settings
from finance_document_qa.api.schemas import HealthResponse
from finance_document_qa.api.health_checks import check_llm, check_neo4j, run_probe

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(response: Response, settings: APISettings = Depends(get_settings)) -> HealthResponse:
    neo4j, llm = await asyncio.gather(
        run_probe(check_neo4j), run_probe(lambda: check_llm(settings))
    )
    healthy = neo4j.status == llm.status == "ok"
    response.status_code = 200 if healthy else 503
    response.headers["Cache-Control"] = "no-store"
    return HealthResponse(
        status="ok" if healthy else "degraded",
        profile=settings.app_profile,
        retriever_mode=settings.retriever_mode,
        checks={"neo4j": neo4j, "llm": llm},
    )


@router.get("/health/live", response_model=HealthResponse)
def live(settings: APISettings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        profile=settings.app_profile,
        retriever_mode=settings.retriever_mode,
    )
