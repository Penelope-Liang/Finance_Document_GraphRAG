from __future__ import annotations

from typing import Any

from neo4j_graphrag.preprocess.schema.clean_document import CleanDocument


BLOCK_TYPE_TO_CATEGORY = {
    "title": "Title",
    "paragraph": "NarrativeText",
    "list": "ListItem",
    "table": "Table",
}


def _flatten_table_text(grid: list[list[dict[str, Any]]] | None, fallback: str) -> str:
    if not grid:
        return fallback.strip()
    parts: list[str] = []
    for row in grid:
        row_text = " ".join(str(cell.get("text") or "").strip() for cell in row if str(cell.get("text") or "").strip())
        if row_text:
            parts.append(row_text)
    return "\n".join(parts).strip() or fallback.strip()


def clean_document_to_element_records(doc: CleanDocument) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, block in enumerate(doc.blocks):
        category = BLOCK_TYPE_TO_CATEGORY.get(block.type, "UncategorizedText")
        text = block.text.strip()
        if block.type == "table":
            text = _flatten_table_text(block.grid, text)
        record: dict[str, Any] = {
            "doc_id": doc.doc_id,
            "source_path": doc.source_path,
            "element_index": idx,
            "category": category,
            "text": text,
            "page": block.page,
            "section_hint": block.section or "",
            "bbox": block.bbox,
            "metadata": {
                "filename": doc.doc_id,
                "filetype": "application/pdf",
                "page_number": block.page,
                "category_depth": None,
                "parent_id": None,
                "languages": ["eng"],
                "element_id": block.block_id,
                "text_as_html": block.html if block.type == "table" else None,
                "parser_backend": doc.parser_backend,
                "clean_block_type": block.type,
                "clean_block_attrs": block.attrs,
            },
        }
        records.append(record)
    return records
