"""Backward-compatible alias for :mod:`neo4j_graphrag.neo4j_indexes`."""

from __future__ import annotations

from neo4j_graphrag import neo4j_indexes as _neo4j_indexes

__all__ = getattr(
    _neo4j_indexes,
    "__all__",
    [name for name in dir(_neo4j_indexes) if not name.startswith("_")],
)

globals().update({name: getattr(_neo4j_indexes, name) for name in __all__})

