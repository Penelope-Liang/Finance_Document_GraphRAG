from __future__ import annotations

from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.retrievers.base import Retriever

from finance_document_qa.pipeline.formatting import _format_chunk_record
from finance_document_qa.pipeline.retrieval_queries import CHUNK_RETRIEVAL_QUERY
from finance_document_qa.pipeline.retrievers import (
    DualRetriever,
    MetricsTableCellRetriever,
    PrimaryStatementRetriever,
)
from finance_document_qa.services.base import BaseFinanceDocumentService


class FinanceDocumentGraphRAGService(BaseFinanceDocumentService):
    """GraphRAG: fuses chunk, Text2Cypher table, primary-statement, and metric
    table-cell evidence via :class:`DualRetriever`."""

    retriever_mode = "dual"

    def _build_core_retriever(self) -> Retriever:
        chunk_retriever = HybridCypherRetriever(
            driver=self.driver,
            vector_index_name=self.config.vector_index,
            fulltext_index_name=self.config.fulltext_index,
            neo4j_database=self.config.neo4j_database,
            retrieval_query=CHUNK_RETRIEVAL_QUERY,
            embedder=self.embedder,
            result_formatter=_format_chunk_record,
        )
        table_retriever = self._build_table_router()
        primary_retriever = PrimaryStatementRetriever(
            driver=self.driver,
            neo4j_database=self.config.neo4j_database,
        )
        metrics_retriever = MetricsTableCellRetriever(
            driver=self.driver,
            neo4j_database=self.config.neo4j_database,
        )
        return DualRetriever(
            driver=self.driver,
            chunk_retriever=chunk_retriever,
            table_retriever=table_retriever,
            primary_retriever=primary_retriever,
            metrics_retriever=metrics_retriever,
            neo4j_database=self.config.neo4j_database,
            reranker_model=self.reranker_model if self.use_reranker else None,
        )


class FinanceDocumentText2CypherService(BaseFinanceDocumentService):
    """Graph query retrieval only: style-routed Text2Cypher over table cells."""

    retriever_mode = "text2cypher"

    def _build_core_retriever(self) -> Retriever:
        return self._build_table_router()
