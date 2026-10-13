from finance_document_qa.api.deps import APISettings


def test_cloud_profile_defaults_to_azure_openai(monkeypatch) -> None:
    monkeypatch.setenv("APP_PROFILE", "cloud")
    monkeypatch.delenv("LLM_BACKEND", raising=False)

    settings = APISettings.from_env()

    assert settings.llm_backend == "azure-openai"
    assert settings.use_reranker is False


def test_explicit_llm_backend_overrides_profile(monkeypatch) -> None:
    monkeypatch.setenv("APP_PROFILE", "cloud")
    monkeypatch.setenv("LLM_BACKEND", "ollama")

    settings = APISettings.from_env()

    assert settings.llm_backend == "ollama"
