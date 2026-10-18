"""Bounded dependency probes, independent of retrieval/index initialization."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import os
import time

import httpx
from neo4j import AsyncGraphDatabase, Query, READ_ACCESS

from finance_document_qa.api.deps import APISettings
from finance_document_qa.api.schemas import DependencyHealth
from finance_document_qa.config import FinanceDocumentGraphRAGConfig


PROBE_TIMEOUT_SECONDS = 10.0


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Missing {name}")
    return value


async def check_neo4j() -> None:
    config = FinanceDocumentGraphRAGConfig.from_env()
    async with AsyncGraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_user, _required("NEO4J_PASSWORD")),
        connection_timeout=PROBE_TIMEOUT_SECONDS,
        connection_acquisition_timeout=PROBE_TIMEOUT_SECONDS,
        max_transaction_retry_time=0,
    ) as driver:
        async with driver.session(
            database=config.neo4j_database, default_access_mode=READ_ACCESS
        ) as session:
            result = await session.run(Query("RETURN 1 AS ok", timeout=PROBE_TIMEOUT_SECONDS))
            record = await result.single()
            if record is None or record["ok"] != 1:
                raise RuntimeError("Unexpected database response")


async def check_llm(settings: APISettings) -> None:
    messages = [{"role": "user", "content": "Reply OK."}]
    if settings.llm_backend == "azure-openai":
        from openai import AsyncAzureOpenAI

        async with AsyncAzureOpenAI(
            azure_endpoint=_required("AZURE_OPENAI_ENDPOINT"),
            api_key=_required("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            timeout=PROBE_TIMEOUT_SECONDS,
            max_retries=0,
        ) as client:
            result = await client.chat.completions.create(
                model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"),
                messages=messages,
                max_completion_tokens=1,
            )
            if not result.choices:
                raise RuntimeError("No model response")
    elif settings.llm_backend == "ollama":
        config = FinanceDocumentGraphRAGConfig.from_env()
        # Use the same host interpretation and authentication as the query SDK.
        from ollama import AsyncClient

        client = AsyncClient(host=config.ollama_host, timeout=PROBE_TIMEOUT_SECONDS)
        try:
            result = await client.chat(
                model=settings.llm_model,
                messages=messages,
                stream=False,
                options={"num_predict": 1},
            )
            if not result.done:
                raise RuntimeError("Incomplete model response")
        finally:
            await client._client.aclose()
    else:
        raise ValueError("Unsupported LLM backend")


async def run_probe(check: Callable[[], Awaitable[None]]) -> DependencyHealth:
    start = time.perf_counter()
    error = None
    try:
        await asyncio.wait_for(check(), timeout=PROBE_TIMEOUT_SECONDS)
    except (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException):
        error = "Dependency check timed out"
    except Exception as exc:
        # Exception messages may contain credentials, URLs or provider payloads.
        error = f"Dependency check failed ({type(exc).__name__})"
    return DependencyHealth(
        status="error" if error else "ok",
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
        error=error,
    )
