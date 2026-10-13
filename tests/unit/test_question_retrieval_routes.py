from unittest.mock import MagicMock, Mock

import pytest

from finance_document_qa.config import FinanceDocumentGraphRAGConfig
from finance_document_qa.pipeline.retrievers import DualRetriever, MetricsTableCellRetriever
from finance_document_qa.services import graphrag, routed
from finance_document_qa.services.base import QueryOptions
from finance_document_qa.services.cli import _build_parser
from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem


@pytest.mark.parametrize(
    "question,reasoning,mode",
    [
        ("What is FY2023 revenue?", None, "hybrid"),
        ("What is the current ratio?", None, "graph"),
        ("What drove the revenue change?", None, "hybrid"),
        ("Find the answer.", "Numerical reasoning", "graph"),
        ("Find the answer.", "numerical", "graph"),
        ("Explain the ratio definition.", "open-book", "hybrid"),
        ("Find the amount.", "Information extraction", "hybrid"),
    ],
)
def test_auto_uses_only_selected_service(monkeypatch, question, reasoning, mode):
    hybrid = Mock()
    graph = Mock()
    for service in (hybrid, graph):
        service.query.return_value = {"answer": "ok", "error": None}
    hybrid_factory, graph_factory = Mock(return_value=hybrid), Mock(return_value=graph)
    monkeypatch.setattr(routed, "FinanceDocumentRAGService", hybrid_factory)
    monkeypatch.setattr(routed, "FinanceDocumentGraphOnlyService", graph_factory)
    service = routed.FinanceDocumentRoutedService()
    options = QueryOptions(question_style="auto")

    result = service.query(question, question_reasoning=reasoning, options=options)

    selected, unused = (graph, hybrid) if mode == "graph" else (hybrid, graph)
    selected.query.assert_called_once()
    unused.query.assert_not_called()
    (hybrid_factory if mode == "graph" else graph_factory).assert_not_called()
    assert result["run_config"]["retriever_mode"] == mode
    assert options.question_style == "auto"
    service.close()
    selected.close.assert_called_once()


@pytest.mark.parametrize("raises", [True, False])
@pytest.mark.parametrize("fallback_error", [None, "hybrid unavailable"])
def test_numerical_failure_falls_back_to_hybrid(monkeypatch, raises, fallback_error):
    graph, hybrid = Mock(), Mock()
    if raises:
        graph.query.side_effect = RuntimeError("graph unavailable")
    else:
        graph.query.return_value = {"error": "graph unavailable"}
    hybrid.query.return_value = {"answer": "fallback answer", "error": fallback_error}
    hybrid_factory = Mock(return_value=hybrid)
    monkeypatch.setattr(routed, "FinanceDocumentGraphOnlyService", Mock(return_value=graph))
    monkeypatch.setattr(routed, "FinanceDocumentRAGService", hybrid_factory)
    result = routed.FinanceDocumentRoutedService().query(
        "Compute result", question_reasoning="numerical"
    )
    assert result["error"] == fallback_error
    graph.query.assert_called_once()
    hybrid.query.assert_called_once()
    assert hybrid.query.call_args.kwargs["options"].question_style == "calculation"
    config = result["run_config"]
    assert config["retriever_mode"] == "hybrid"
    assert config["initial_retriever_mode"] == "graph"
    assert config["question_category"] == "numerical"
    assert config["fallback_applied"] is True
    assert config["fallback_reason"] == "graph unavailable"


def test_llm_router_does_not_search_graph_for_open_book(monkeypatch):
    graph, hybrid = Mock(), Mock()
    graph.llm.invoke.return_value.content = '{"label":"direct","reason":"lookup"}'
    hybrid.query.return_value = {"answer": "ok"}
    monkeypatch.setattr(routed, "FinanceDocumentGraphOnlyService", Mock(return_value=graph))
    monkeypatch.setattr(routed, "FinanceDocumentRAGService", Mock(return_value=hybrid))
    result = routed.FinanceDocumentRoutedService().query(
        "What is revenue?", options=QueryOptions(question_style="llm-auto")
    )
    assert result["run_config"]["retriever_mode"] == "hybrid"
    graph.query.assert_not_called()
    hybrid.query.assert_called_once()


def test_graph_service_does_not_construct_hybrid(monkeypatch):
    hybrid_factory = Mock(side_effect=AssertionError("hybrid forbidden"))
    monkeypatch.setattr(graphrag, "HybridCypherRetriever", hybrid_factory)
    monkeypatch.setattr(graphrag, "PrimaryStatementRetriever", Mock())
    monkeypatch.setattr(graphrag, "MetricsTableCellRetriever", Mock())
    fusion = Mock()
    monkeypatch.setattr(graphrag, "DualRetriever", fusion)
    service = object.__new__(graphrag.FinanceDocumentGraphOnlyService)
    service.driver = Mock()
    service.config = FinanceDocumentGraphRAGConfig()
    service.use_reranker = False
    service._build_table_router = Mock()
    service._build_core_retriever()
    assert fusion.call_args.kwargs["chunk_retriever"] is None
    hybrid_factory.assert_not_called()


@pytest.mark.parametrize("outcome", ["evidence", "partial", "empty", "error"])
def test_graph_only_fusion_executes_graph_branches(monkeypatch, outcome):
    # Exercise real fusion, rather than mocking the graph-only search itself.
    retriever = object.__new__(DualRetriever)
    retriever.chunk_retriever = None
    retriever.reranker_model = None
    retriever.table_retriever = Mock()
    retriever.primary_retriever = Mock()
    retriever.metrics_retriever = Mock()
    for index, branch in enumerate((
        retriever.table_retriever, retriever.primary_retriever, retriever.metrics_retriever
    )):
        branch.search.return_value = RetrieverResult(items=[
            RetrieverResultItem(content=f"evidence {index}", metadata={"source": "table"})
        ])
        if outcome == "empty" or (outcome == "partial" and index == 2):
            branch.search.return_value = RetrieverResult(items=[])
        elif outcome == "error":
            branch.search.side_effect = RuntimeError("database unavailable")

    # Connect the actual fusion implementation to the routing facade, so a
    # misplaced empty-result check cannot pass a mocked fallback test.
    def graph_query(*args, **kwargs):
        result = retriever.search("current ratio", t2c_style="calculation", metrics_plan={})
        assert len(result.items) == (2 if outcome == "partial" else 3)
        assert result.metadata["__retriever"] == "GraphEvidenceRetriever"
        assert result.metadata["chunk_retriever"]["enabled"] is False
        return {"answer": "graph answer", "error": None}

    graph, hybrid = Mock(), Mock()
    graph.query.side_effect = graph_query
    hybrid.query.return_value = {"answer": "hybrid answer", "error": None}
    monkeypatch.setattr(routed, "FinanceDocumentGraphOnlyService", Mock(return_value=graph))
    hybrid_factory = Mock(return_value=hybrid)
    monkeypatch.setattr(routed, "FinanceDocumentRAGService", hybrid_factory)
    result = routed.FinanceDocumentRoutedService().query(
        "What is the current ratio?", question_reasoning="numerical"
    )
    if outcome in {"empty", "error"}:
        assert result["answer"] == "hybrid answer"
        assert result["run_config"]["fallback_applied"] is True
        assert "Graph retrieval returned no evidence" in result["run_config"]["fallback_reason"]
        hybrid.query.assert_called_once()
    else:
        assert result["answer"] == "graph answer"
        assert result["run_config"]["fallback_applied"] is False
        hybrid_factory.assert_not_called()
    for branch in (retriever.table_retriever, retriever.primary_retriever, retriever.metrics_retriever):
        branch.search.assert_called_once()


def test_empty_metric_query_returns_normally():
    retriever = object.__new__(MetricsTableCellRetriever)
    retriever.driver = MagicMock()
    retriever.neo4j_database = None
    retriever.limit = 80
    retriever.per_target_limit = 12
    session = retriever.driver.session.return_value.__enter__.return_value
    session.run.return_value = []

    result = retriever.search("What is the current ratio?", doc_id="filing")

    session.run.assert_called_once()
    assert result.items == []
    assert result.metadata["matched"] is False


def test_cli_defaults_to_question_routing():
    args = _build_parser().parse_args([])
    assert args.retriever_mode == "auto"
    assert args.question_style == "auto"
