from __future__ import annotations

from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.retrievers.base import Retriever

from finance_document_qa.pipeline.formatting import _format_chunk_record
from finance_document_qa.pipeline.retrieval_queries import CHUNK_RETRIEVAL_QUERY
from finance_document_qa.services.base import BaseFinanceDocumentService


class FinanceDocumentRAGService(BaseFinanceDocumentService):
    """Text-only RAG baseline: hybrid vector + full-text chunk retrieval."""

    retriever_mode = "hybrid"

    def _build_core_retriever(self) -> Retriever:
        return HybridCypherRetriever(
            driver=self.driver,
            vector_index_name=self.config.vector_index,
            fulltext_index_name=self.config.fulltext_index,
            neo4j_database=self.config.neo4j_database,
            retrieval_query=CHUNK_RETRIEVAL_QUERY,
            embedder=self.embedder,
            result_formatter=_format_chunk_record,
        )
