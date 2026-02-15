#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Reranking retriever: wraps any Retriever and reranks results with a CrossEncoder.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import RawSearchResult, RetrieverResult

logger = logging.getLogger(__name__)


def _get_cross_encoder(model: Union[str, Any]):
    """Resolve CrossEncoder from model name or instance."""
    if hasattr(model, "predict"):
        return model
    try:
        from sentence_transformers import CrossEncoder as CE
        return CE(model)
    except ImportError as e:
        raise ImportError(
            "Reranking requires sentence-transformers. "
            "Install with: pip install neo4j-graphrag[sentence-transformers]"
        ) from e


class RerankingRetriever(Retriever):
    """
    Wraps another retriever and reranks its results using a CrossEncoder
    (query, document) scoring model before returning the top_k items.

    Example:

    .. code-block:: python

      from neo4j_graphrag.retrievers import HybridCypherRetriever, RerankingRetriever

      base = HybridCypherRetriever(driver, "v", "ft", retrieval_query, embedder)
      retriever = RerankingRetriever(
          inner_retriever=base,
          cross_encoder_model="BAAI/bge-reranker-v2-m3",
          top_k_retrieve=20,
      )
      retriever.search(query_text="...", top_k=10)  # fetches 20, reranks, returns 10
    """

    VERIFY_NEO4J_VERSION = False

    def __init__(
        self,
        inner_retriever: Retriever,
        cross_encoder_model: Union[str, Any],
        top_k_retrieve: Optional[int] = None,
        neo4j_database: Optional[str] = None,
    ):
        """
        Args:
            inner_retriever: The retriever to use for first-stage retrieval.
            cross_encoder_model: HuggingFace model name (e.g. "BAAI/bge-reranker-v2-m3")
                or an already-loaded CrossEncoder instance.
            top_k_retrieve: Number of items to fetch before reranking. If None,
                uses 2 * top_k passed to search().
            neo4j_database: Neo4j database name (passed to base; taken from inner if None).
        """
        super().__init__(
            driver=inner_retriever.driver,
            neo4j_database=neo4j_database or getattr(inner_retriever, "neo4j_database", None),
        )
        self.inner_retriever = inner_retriever
        self._cross_encoder_model_arg = cross_encoder_model
        self._ce = None  # lazy-loaded CrossEncoder
        self.top_k_retrieve = top_k_retrieve

    def _get_ce(self):
        if self._ce is None:
            self._ce = _get_cross_encoder(self._cross_encoder_model_arg)
        return self._ce

    def get_search_results(self, *args: Any, **kwargs: Any) -> RawSearchResult:
        """Delegate to inner retriever with expanded top_k (for compatibility)."""
        top_k = kwargs.get("top_k", 5)
        retrieve_k = self.top_k_retrieve if self.top_k_retrieve is not None else max(top_k * 2, 20)
        kwargs = {**kwargs, "top_k": retrieve_k}
        return self.inner_retriever.get_search_results(*args, **kwargs)

    def search(self, *args: Any, **kwargs: Any) -> RetrieverResult:
        """Retrieve more items, rerank with CrossEncoder, return top_k."""
        top_k = kwargs.get("top_k", 5)
        retrieve_k = self.top_k_retrieve if self.top_k_retrieve is not None else max(top_k * 2, 20)
        kwargs_retrieve = {**kwargs, "top_k": retrieve_k}

        result = self.inner_retriever.search(*args, **kwargs_retrieve)
        if not result.items:
            return result

        query_text = kwargs.get("query_text") or (args[0] if args else "")
        if not query_text:
            logger.warning("RerankingRetriever: no query_text in args/kwargs; returning un-reranked.")
            return RetrieverResult(
                items=result.items[:top_k],
                metadata={**(result.metadata or {}), "reranked": False},
            )

        pairs = [(query_text, item.content) for item in result.items]
        try:
            scores = self._get_ce().predict(pairs, show_progress_bar=False)
        except Exception as e:
            logger.warning("CrossEncoder predict failed: %s; returning un-reranked.", e)
            return RetrieverResult(
                items=result.items[:top_k],
                metadata={**(result.metadata or {}), "reranked": False},
            )

        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        indexed = list(zip(scores, result.items, strict=True))
        indexed.sort(key=lambda x: x[0], reverse=True)
        reranked_items = [item for _, item in indexed[:top_k]]
        return RetrieverResult(
            items=reranked_items,
            metadata={**(result.metadata or {}), "reranked": True},
        )
