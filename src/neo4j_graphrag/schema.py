"""Backward-compatible alias for :mod:`neo4j_graphrag.graph_schema`."""

from __future__ import annotations

from neo4j_graphrag import graph_schema as _graph_schema

__all__ = getattr(
    _graph_schema,
    "__all__",
    [name for name in dir(_graph_schema) if not name.startswith("_")],
)

globals().update({name: getattr(_graph_schema, name) for name in __all__})

