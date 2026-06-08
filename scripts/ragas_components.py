import os


def _azure_credentials():
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    if not endpoint:
        raise ValueError("Missing AZURE_OPENAI_ENDPOINT.")
    if not api_key:
        raise ValueError("Missing AZURE_OPENAI_API_KEY.")
    return endpoint, api_key, api_version


def build_ragas_v2_llm(chat_deployment, temperature=0.0):
    """Build a ragas v2 (collections) LLM via llm_factory + AsyncAzureOpenAI.

    Collections metrics (ragas.metrics.collections) require an
    InstructorBaseRagasLLM, not the legacy LangchainLLMWrapper.
    """
    try:
        from openai import AsyncAzureOpenAI
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise ImportError(
            "RAGAS v2 evaluation requires: pip install ragas openai instructor"
        ) from exc

    endpoint, api_key, api_version = _azure_credentials()
    client = AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
    return llm_factory(chat_deployment, client=client, temperature=temperature)


def build_ragas_azure_components(chat_deployment, embedding_deployment, temperature):
    try:
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise ImportError(
            "RAGAS evaluation requires: pip install ragas langchain-openai datasets"
        ) from exc

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    if not endpoint:
        raise ValueError("Missing AZURE_OPENAI_ENDPOINT.")
    if not api_key:
        raise ValueError("Missing AZURE_OPENAI_API_KEY.")

    chat_model = AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=chat_deployment,
        temperature=temperature,
    )
    embedding_model = AzureOpenAIEmbeddings(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=embedding_deployment,
    )
    return LangchainLLMWrapper(chat_model), LangchainEmbeddingsWrapper(embedding_model)
