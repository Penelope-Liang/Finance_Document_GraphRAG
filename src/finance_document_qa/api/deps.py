from __future__ import annotations

import os
from dataclasses import dataclass

from fastapi import Request

from finance_document_qa.config import FinanceDocumentGraphRAGConfig
from finance_document_qa.services.base import BaseFinanceDocumentService
from finance_document_qa.services.graphrag import (
    FinanceDocumentGraphOnlyService,
    FinanceDocumentGraphRAGService,
    FinanceDocumentText2CypherService,
)
from finance_document_qa.services.rag import FinanceDocumentRAGService
from finance_document_qa.services.routed import FinanceDocumentRoutedService


_SERVICE_BY_MODE: dict[str, type[BaseFinanceDocumentService]] = {
    "auto": FinanceDocumentRoutedService,
    "graph": FinanceDocumentGraphOnlyService,
    "dual": FinanceDocumentGraphRAGService,
    "hybrid": FinanceDocumentRAGService,
    "text2cypher": FinanceDocumentText2CypherService,
}
_VALID_PROFILES = {"local", "cloud"}


@dataclass(frozen=True)
class APISettings:
    app_profile: str = "local"
    retriever_mode: str = "auto"
    llm_backend: str = "ollama"
    llm_model: str = "qwen3:30b-a3b"
    use_reranker: bool = False

    @classmethod
    def from_env(cls) -> "APISettings":
        profile = os.environ.get("APP_PROFILE", cls.app_profile)
        if profile not in _VALID_PROFILES:
            raise RuntimeError(f"APP_PROFILE must be one of {sorted(_VALID_PROFILES)}")
        retriever_mode = os.environ.get("RETRIEVER_MODE", cls.retriever_mode)
        if retriever_mode not in _SERVICE_BY_MODE:
            raise RuntimeError(
                f"RETRIEVER_MODE must be one of {sorted(_SERVICE_BY_MODE)}"
            )
        default_llm_backend = "azure-openai" if profile == "cloud" else "ollama"
        return cls(
            app_profile=profile,
            retriever_mode=retriever_mode,
            llm_backend=os.environ.get("LLM_BACKEND", default_llm_backend),
            llm_model=os.environ.get("OLLAMA_LLM_MODEL", cls.llm_model),
            use_reranker=_env_bool("USE_RERANKER", default=False),
        )


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_settings(request: Request) -> APISettings:
    return request.app.state.settings


def get_service(request: Request) -> BaseFinanceDocumentService:
    service = request.app.state.service
    if service is None:
        settings = get_settings(request)
        service_cls = _SERVICE_BY_MODE[settings.retriever_mode]
        service = service_cls(
            FinanceDocumentGraphRAGConfig.from_env(),
            llm_backend=settings.llm_backend,
            llm_model=settings.llm_model,
            use_reranker=settings.use_reranker,
        )
        request.app.state.service = service
    return service
