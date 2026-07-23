from __future__ import annotations

from typing import Any

import neo4j
from neo4j_graphrag.types import RetrieverResultItem

from finance_document_qa.pipeline.utils import _sanitize_for_json, _sanitize_value


def _format_chunk_record(record: neo4j.Record) -> RetrieverResultItem:
    text = str(record.get("text") or "").strip()
    doc_id = str(record.get("doc_id") or "").strip()
    section_name = str(record.get("section_name") or "").strip()
    table_cells = _sanitize_for_json(record.get("table_cells") or [])
    lines = []
    if section_name:
        lines.append(f"Section: {section_name}")
    if text:
        lines.append(f"Chunk: {text}")
    if table_cells:
        lines.append("Related table cells:")
        for cell in table_cells[:10]:
            if not isinstance(cell, dict):
                continue
            row_header = cell.get("row_header")
            col_header = cell.get("col_header")
            cell_text = cell.get("text")
            numeric_value = cell.get("numeric_value")
            numeric_kind = cell.get("numeric_kind")
            scale = cell.get("scale")
            lines.append(
                f"- row={row_header!r}; col={col_header!r}; text={cell_text!r}; "
                f"numeric_value={numeric_value!r}; numeric_kind={numeric_kind!r}; scale={scale!r}"
            )
    if not lines:
        lines.append(str(record))
    return RetrieverResultItem(
        content="\n".join(lines),
        metadata=_sanitize_for_json(
            {
                "doc_id": doc_id or None,
                "section_name": section_name or None,
                "similarity_score": record.get("similarity_score"),
                "source": "chunk",
            }
        ),
    )


def _format_table_record(record: neo4j.Record) -> RetrieverResultItem:
    doc_id = _sanitize_value(record.get("doc_id"))
    section_name = str(record.get("section_name") or "").strip()
    row_header = _sanitize_value(record.get("row_header"))
    col_header = _sanitize_value(record.get("col_header"))
    text = _sanitize_value(record.get("text"))
    numeric_value = _sanitize_value(record.get("numeric_value"))
    numeric_kind = _sanitize_value(record.get("numeric_kind"))
    scale = _sanitize_value(record.get("scale"))
    content = (
        f"Section: {section_name}\n"
        f"Table evidence: row={row_header!r}; col={col_header!r}; text={text!r}; "
        f"numeric_value={numeric_value!r}; numeric_kind={numeric_kind!r}; scale={scale!r}"
    )
    return RetrieverResultItem(
        content=content,
        metadata=_sanitize_for_json(
            {
                "doc_id": doc_id,
                "section_name": section_name or None,
                "row_header": row_header,
                "col_header": col_header,
                "text": text,
                "numeric_value": numeric_value,
                "numeric_kind": numeric_kind,
                "scale": scale,
                "source": "table",
            }
        ),
    )


def _compact_retriever_evidence(
    retriever_result_dump: dict[str, Any] | None,
    *,
    max_items: int = 8,
    max_chars_per_item: int = 1200,
) -> str:
    if not retriever_result_dump:
        return ""
    items = retriever_result_dump.get("items") or []
    blocks: list[str] = []
    for index, item in enumerate(items[:max_items], start=1):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") or {}
        content = str(item.get("content") or "")[:max_chars_per_item]
        blocks.append(
            "\n".join(
                [
                    f"[{index}] source={metadata.get('source')}",
                    f"doc={metadata.get('doc_id')} section={metadata.get('section_name')}",
                    content,
                ]
            )
        )
    return "\n\n".join(blocks)


def _extract_gold_evidence(row: dict[str, Any]) -> Any:
    """Preserve FinanceBench evidence annotations when present in the input row."""
    return (
        row.get("gold_evidence")
        or row.get("evidence")
        or row.get("evidences")
        or row.get("evidence_annotations")
    )


def _context_score(metadata: dict[str, Any]) -> Any:
    for key in (
        "bucket_priority",
        "base_score",
        "fused_score",
        "score",
        "similarity_score",
        "normalized_score",
        "raw_score",
    ):
        value = metadata.get(key)
        if value is not None:
            return value
    return None


def _standardize_retrieved_contexts(
    retriever_result_dump: dict[str, Any] | None,
    *,
    question_id: str,
) -> list[dict[str, Any]]:
    """Expose retriever_result.items in an evaluation-friendly format.

    The original retriever_result is kept for debugging. This normalized list
    adds stable rank/context_id fields needed by retrieval metrics.
    """
    if not retriever_result_dump:
        return []
    items = retriever_result_dump.get("items") or []
    contexts: list[dict[str, Any]] = []
    for rank, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        text = str(item.get("content") or "")
        source = metadata.get("source")
        contexts.append(
            {
                "rank": rank,
                "context_id": f"{question_id}::ctx_{rank}",
                "source": source,
                "text": text,
                "score": _context_score(metadata),
                "doc_id": metadata.get("doc_id"),
                "doc_name": metadata.get("doc_name") or metadata.get("doc_id"),
                "section_name": metadata.get("section_name"),
                "metadata": _sanitize_for_json(metadata),
            }
        )
    return contexts
