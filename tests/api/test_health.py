import asyncio
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
import pytest

from finance_document_qa.api.app import create_app
from finance_document_qa.api import health_checks
from finance_document_qa.api.routers import health


@pytest.mark.parametrize("failed", [None, "neo4j", "llm", "both"])
def test_dependency_health(monkeypatch, failed) -> None:
    probes = {}
    for name in ("neo4j", "llm"):
        probe = AsyncMock()
        if failed in (name, "both"):
            probe.side_effect = RuntimeError("secret password and provider URL")
        monkeypatch.setattr(health, f"check_{name}", probe)
        probes[name] = probe
    with TestClient(create_app()) as client:
        response = client.get("/health")
    assert response.status_code == (503 if failed else 200)
    body = response.json()
    assert body["status"] == ("degraded" if failed else "ok")
    assert "secret password" not in response.text
    for name, probe in probes.items():
        probe.assert_awaited_once()
        assert body["checks"][name]["status"] == (
            "error" if failed in (name, "both") else "ok"
        )
        assert body["checks"][name]["latency_ms"] >= 0
    assert response.headers["Cache-Control"] == "no-store"


def test_liveness_does_not_connect(monkeypatch) -> None:
    probe = AsyncMock(side_effect=AssertionError("Must not connect"))
    monkeypatch.setattr(health, "check_neo4j", probe)
    monkeypatch.setattr(health, "check_llm", probe)
    with TestClient(create_app()) as client:
        response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    probe.assert_not_called()


def test_timeout_preserves_other_dependency_result(monkeypatch) -> None:
    async def slow():
        await asyncio.Event().wait()

    monkeypatch.setattr(health_checks, "PROBE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(health, "check_neo4j", slow)
    monkeypatch.setattr(health, "check_llm", AsyncMock())
    with TestClient(create_app()) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["checks"]["neo4j"]["error"] == "Dependency check timed out"
    assert response.json()["checks"]["llm"]["status"] == "ok"


def test_neo4j_probe_queries_selected_database(monkeypatch) -> None:
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
    monkeypatch.setenv("NEO4J_DATABASE", "financial")
    result = AsyncMock()
    result.single.return_value = {"ok": 1}
    session = AsyncMock()
    session.run.return_value = result
    driver = MagicMock()
    driver.session.return_value.__aenter__ = AsyncMock(return_value=session)
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=driver)
    monkeypatch.setattr(health_checks.AsyncGraphDatabase, "driver", factory)
    asyncio.run(health_checks.check_neo4j())
    assert driver.session.call_args.kwargs["database"] == "financial"
    assert session.run.call_args.args[0].text == "RETURN 1 AS ok"
    factory.return_value.__aexit__.assert_awaited_once()
    driver.session.return_value.__aexit__.assert_awaited_once()


def test_neo4j_accepts_aura_username(monkeypatch) -> None:
    from finance_document_qa.config import FinanceDocumentGraphRAGConfig

    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.setenv("NEO4J_USERNAME", "aura-user")
    assert FinanceDocumentGraphRAGConfig.from_env().neo4j_user == "aura-user"
    monkeypatch.setenv("NEO4J_USER", "explicit-user")
    assert FinanceDocumentGraphRAGConfig.from_env().neo4j_user == "explicit-user"


@pytest.mark.parametrize("backend", ["ollama", "azure-openai"])
@pytest.mark.parametrize("provider_status", [200, 401])
def test_llm_probe_calls_configured_model(monkeypatch, backend, provider_status) -> None:
    import json
    import httpx
    from finance_document_qa.api.deps import APISettings

    calls = []

    def respond(request):
        calls.append(request)
        if provider_status != 200:
            return httpx.Response(provider_status, json={"error": "private provider message"})
        if backend == "ollama":
            return httpx.Response(200, json={
                "model": "test-model", "message": {"role": "assistant", "content": "OK"},
                "done": True,
            })
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0,
            "model": "test-deployment", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "OK"},
                 "finish_reason": "length"}
            ],
        })

    transport = httpx.MockTransport(respond)
    clients = []
    if backend == "ollama":
        ollama = pytest.importorskip("ollama")
        original = ollama.AsyncClient
        monkeypatch.setenv("OLLAMA_HOST", "http://ollama.test:11434")

        def factory(**kwargs):
            client = original(**kwargs, transport=transport)
            clients.append(client._client)
            return client

        monkeypatch.setattr(ollama, "AsyncClient", factory)
    else:
        openai = pytest.importorskip("openai")
        original = openai.AsyncAzureOpenAI
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://azure.test")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "test-deployment")

        def factory(**kwargs):
            http_client = httpx.AsyncClient(transport=transport)
            clients.append(http_client)
            return original(**kwargs, http_client=http_client)

        monkeypatch.setattr(openai, "AsyncAzureOpenAI", factory)

    settings = APISettings(llm_backend=backend, llm_model="test-model")
    result = asyncio.run(health_checks.run_probe(lambda: health_checks.check_llm(settings)))
    assert result.status == ("ok" if provider_status == 200 else "error")
    assert "private provider message" not in (result.error or "")
    assert len(calls) == 1
    payload = json.loads(calls[0].content)
    if backend == "ollama":
        assert calls[0].url.host == "ollama.test"
        assert payload["model"] == "test-model"
        assert payload["options"]["num_predict"] == 1
    else:
        assert "/deployments/test-deployment/" in calls[0].url.path
        assert payload["max_completion_tokens"] == 1
    assert all(client.is_closed for client in clients)
