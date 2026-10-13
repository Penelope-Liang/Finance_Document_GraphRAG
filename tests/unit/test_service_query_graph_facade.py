from typing import Any

from finance_document_qa.services.base import BaseFinanceDocumentService, QueryOptions


class FakeGraph:
    def __init__(self) -> None:
        self.state: dict[str, Any] | None = None

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.state = state
        return {"result": {"answer": "ok", "retrieved_contexts": [], "error": None}}


def test_service_query_invokes_graph() -> None:
    graph = FakeGraph()
    service = object.__new__(BaseFinanceDocumentService)
    service._query_graph = graph

    result = service.query(
        "What is revenue?",
        question_id="q1",
        available_doc_ids=["doc1"],
        options=QueryOptions(top_k=3),
    )

    assert result["answer"] == "ok"
    assert graph.state is not None
    assert graph.state["question"] == "What is revenue?"
    assert graph.state["options"].top_k == 3
