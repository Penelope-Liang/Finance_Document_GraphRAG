import os

from finance_document_qa.pipeline.retrievers import DualRetriever


class _PoolConfig:
    user_agent = ""


class _Pool:
    pool_config = _PoolConfig()


class FakeDriver:
    _pool = _Pool()


def test_dual_retriever_does_not_enable_reranker_from_env() -> None:
    old_value = os.environ.get("RERANKER_MODEL")
    os.environ["RERANKER_MODEL"] = "BAAI/bge-reranker-v2-m3"
    try:
        retriever = DualRetriever(
            driver=FakeDriver(),
            chunk_retriever=None,  # type: ignore[arg-type]
            table_retriever=None,  # type: ignore[arg-type]
            reranker_model=None,
        )
        assert retriever.reranker_model is None
    finally:
        if old_value is None:
            os.environ.pop("RERANKER_MODEL", None)
        else:
            os.environ["RERANKER_MODEL"] = old_value
