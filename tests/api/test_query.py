from typing import Any

from fastapi.testclient import TestClient

from finance_document_qa.api.app import create_app
from finance_document_qa.api.deps import get_service


class FakeService:
    def query(self, question: str, **_: Any) -> dict[str, Any]:
        return {
            "answer": f"mock: {question}",
            "evidence": "mock evidence",
            "retrieved_contexts": [{"content": "mock context"}],
            "run_config": {"retriever_mode": "dual"},
            "time_seconds": 0.01,
            "error": None,
        }


def test_query_uses_service_override() -> None:
    app = create_app()
    app.dependency_overrides[get_service] = lambda: FakeService()
    client = TestClient(app)

    response = client.post("/query", json={"question": "What is revenue?"})

    assert response.status_code == 200
    assert "x-request-id" in response.headers
    body = response.json()
    assert body["answer"] == "mock: What is revenue?"
    assert body["metadata"]["request_id"]
    assert body["metadata"]["run_config"]["retriever_mode"] == "dual"
