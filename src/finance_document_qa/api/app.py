from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from finance_document_qa.api.deps import APISettings
from finance_document_qa.api.middleware.logging import request_logging_middleware
from finance_document_qa.api.routers import health, query


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    service = app.state.service
    if service is not None:
        service.close()


def create_app() -> FastAPI:
    app = FastAPI(title="Finance Document QA", lifespan=lifespan)
    app.state.settings = APISettings.from_env()
    app.state.service = None
    app.middleware("http")(request_logging_middleware)
    app.include_router(health.router)
    app.include_router(query.router)
    return app


app = create_app()
