from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class FinanceDocumentGraphRAGConfig:
    """Environment-backed runtime settings for finance document GraphRAG."""

    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_database: str | None = None
    vector_index: str = "chunk_embedding_index"
    fulltext_index: str = "chunk_fulltext_index"
    embedding_model: str = "bge-m3:latest"
    ollama_host: str | None = None

    @classmethod
    def from_env(cls) -> "FinanceDocumentGraphRAGConfig":
        return cls(
            neo4j_uri=os.environ.get("NEO4J_URI", cls.neo4j_uri),
            neo4j_user=os.environ.get("NEO4J_USER", cls.neo4j_user),
            neo4j_database=os.environ.get("NEO4J_DATABASE"),
            vector_index=os.environ.get("NEO4J_VECTOR_INDEX", cls.vector_index),
            fulltext_index=os.environ.get("NEO4J_FULLTEXT_INDEX", cls.fulltext_index),
            embedding_model=os.environ.get("OLLAMA_EMBED_MODEL", cls.embedding_model),
            ollama_host=os.environ.get("OLLAMA_HOST"),
        )
