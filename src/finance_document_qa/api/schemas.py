from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DependencyHealth(BaseModel):
    status: Literal["ok", "error"]
    latency_ms: float
    error: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    profile: str
    retriever_mode: str
    checks: dict[str, DependencyHealth] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    question_id: str | None = None
    doc_id: str | None = None
    doc_name: str | None = None
    question_reasoning: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    effective_search_ratio: int = Field(default=10, ge=1, le=100)
    question_style: Literal[
        "direct", "calculation", "domain", "novel", "auto", "llm-auto"
    ] = "auto"
    use_query_rewrite: bool = False


class QueryResponse(BaseModel):
    answer: Any = None
    evidence: Any = None
    retrieved_contexts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
