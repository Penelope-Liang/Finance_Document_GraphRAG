from __future__ import annotations

from typing import Any, Literal, TypedDict

from finance_document_qa.services.base import BaseFinanceDocumentService, QueryOptions


class FinanceRAGState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    question: str
    question_id: str | None
    doc_id: str | None
    doc_name: str | None
    question_reasoning: str | None
    available_doc_ids: list[str]
    options: QueryOptions
    effective_style: str
    question_style_fields: dict[str, Any]
    query_rewrite_plan: dict[str, Any]
    retrieval_query: str
    retrieval_filter: dict[str, Any]
    retriever_config: dict[str, Any]
    retrieval_path: Literal["rag", "graphrag"]
    base_record: dict[str, Any]
    rag_result: Any
    retriever_result: dict[str, Any] | None
    retrieved_contexts: list[dict[str, Any]]
    raw_answer: str
    answer: Any
    evidence: Any
    missing_information: Any
    answer_source: str
    calculation_executor: dict[str, Any] | None
    error: str | None
    result: dict[str, Any]


def build_finance_rag_graph(service: BaseFinanceDocumentService) -> Any:
    """Build a lightweight LangGraph orchestration layer around the existing QA service.

    The graph keeps the production retrievers and LLM implementation intact. LangGraph
    is used only to make state transitions and retrieval routing explicit.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph is required for the production query workflow."
        ) from exc

    def prepare_query(state: FinanceRAGState) -> dict[str, Any]:
        options = state.get("options") or QueryOptions()
        return service.build_query_plan(
            state["question"],
            question_id=state.get("question_id"),
            doc_id=state.get("doc_id"),
            doc_name=state.get("doc_name"),
            question_reasoning=state.get("question_reasoning"),
            available_doc_ids=state.get("available_doc_ids"),
            options=options,
        )

    def run_rag(state: FinanceRAGState) -> dict[str, Any]:
        result = service.run_rag_search(
            state["question"],
            effective_style=state["effective_style"],
            retrieval_query=state["retrieval_query"],
            retrieval_filter=state["retrieval_filter"],
            query_rewrite_plan=state["query_rewrite_plan"],
            options=state["options"],
        )
        return {
            "rag_result": result,
            "retriever_result": (
                result.retriever_result.model_dump()
                if result.retriever_result is not None
                else None
            ),
        }

    def finalize_answer(state: FinanceRAGState) -> dict[str, Any]:
        result = service.build_success_record(
            base_record=state["base_record"],
            result=state["rag_result"],
            question=state["question"],
            question_id=state.get("question_id"),
            question_reasoning=state.get("question_reasoning"),
            effective_style=state["effective_style"],
            options=state["options"],
        )
        return {
            "answer": result.get("answer"),
            "evidence": result.get("evidence"),
            "retrieved_contexts": result.get("retrieved_contexts") or [],
            "result": result,
        }

    builder = StateGraph(FinanceRAGState)
    builder.add_node("prepare_query", prepare_query)
    builder.add_node("run_rag", run_rag)
    builder.add_node("finalize_answer", finalize_answer)
    builder.add_edge(START, "prepare_query")
    builder.add_edge("prepare_query", "run_rag")
    builder.add_edge("run_rag", "finalize_answer")
    builder.add_edge("finalize_answer", END)
    return builder.compile()
