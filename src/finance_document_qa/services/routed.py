from __future__ import annotations

import time
from dataclasses import replace
from threading import RLock
from typing import Any

from finance_document_qa.config import FinanceDocumentGraphRAGConfig
from finance_document_qa.services.base import BaseFinanceDocumentService, QueryOptions
from finance_document_qa.services.graphrag import FinanceDocumentGraphOnlyService
from finance_document_qa.services.rag import FinanceDocumentRAGService


class FinanceDocumentRoutedService(BaseFinanceDocumentService):
    """Open-book -> hybrid; numerical -> graph, with hybrid fallback on failure."""

    retriever_mode = "auto"

    def __init__(
        self, config: FinanceDocumentGraphRAGConfig | None = None, **kwargs: Any
    ) -> None:
        self.config = config or FinanceDocumentGraphRAGConfig.from_env()
        self._service_kwargs = kwargs
        self._services: dict[str, BaseFinanceDocumentService] = {}
        self._service_lock = RLock()

    def _service(self, mode: str) -> BaseFinanceDocumentService:
        with self._service_lock:
            if mode not in self._services:
                cls = (
                    FinanceDocumentGraphOnlyService if mode == "graph"
                    else FinanceDocumentRAGService
                )
                self._services[mode] = cls(self.config, **self._service_kwargs)
            return self._services[mode]

    @property
    def llm(self) -> Any:
        # Classification does not run retrieval or initialize a hybrid retriever.
        return self._service("graph").llm

    def list_documents(self) -> list[str]:
        return self._service("graph").list_documents()

    @property
    def available_documents(self) -> list[str]:
        return self._service("graph").available_documents

    def query(
        self,
        question: str,
        *,
        question_id: str | None = None,
        doc_id: str | None = None,
        doc_name: str | None = None,
        question_reasoning: str | None = None,
        available_doc_ids: list[str] | None = None,
        options: QueryOptions | None = None,
    ) -> dict[str, Any]:
        start = time.time()
        options = options or QueryOptions(question_style="auto")
        try:
            label = " ".join(
                str(question_reasoning or "").lower().replace("-", " ").replace("_", " ").split()
            )
            if label in {"numerical", "numerical reasoning"}:
                style, reason = "calculation", "question_reasoning=numerical"
            elif label in {"open", "open book", "information extraction"}:
                style, reason = "direct", "question_reasoning=open_book"
            else:
                classification = self._classify_style(
                    question, question_reasoning, options.question_style
                )
                style = classification["effective_style"]
                reason = classification["fields"]["question_style_reason"]
            mode = "graph" if style == "calculation" else "hybrid"
            initial_mode = mode

            def run(selected_mode: str) -> dict[str, Any]:
                try:
                    return self._service(selected_mode).query(
                        question,
                        question_id=question_id,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        question_reasoning=question_reasoning,
                        available_doc_ids=available_doc_ids,
                        options=replace(options, question_style=style),
                    )
                except Exception as exc:
                    return self.build_error_record(
                        question=question, question_id=question_id, doc_id=doc_id,
                        doc_name=doc_name, elapsed=round(time.time() - start, 2), error=exc,
                    )

            result = run(mode)
            fallback_reason = None
            if mode == "graph" and result.get("error"):
                fallback_reason = str(result["error"])
                mode = "hybrid"
                result = run(mode)
            result.setdefault("run_config", {}).update({
                "routing_mode": "auto",
                "retriever_mode": mode,
                "initial_retriever_mode": initial_mode,
                "fallback_applied": fallback_reason is not None,
                "fallback_reason": fallback_reason,
                "question_category": "numerical" if initial_mode == "graph" else "open_book",
                "routing_reason": reason,
            })
            result["time_seconds"] = round(time.time() - start, 2)
            return result
        except Exception as exc:
            return self.build_error_record(
                question=question, question_id=question_id, doc_id=doc_id,
                doc_name=doc_name, elapsed=round(time.time() - start, 2), error=exc,
            )

    def close(self) -> None:
        with self._service_lock:
            for service in self._services.values():
                service.close()
            self._services.clear()
