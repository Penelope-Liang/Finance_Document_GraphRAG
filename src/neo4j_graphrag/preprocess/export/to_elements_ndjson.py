from __future__ import annotations

from typing import Any

from neo4j_graphrag.preprocess.provenance import IDENTITY_FIELDS

from neo4j_graphrag.preprocess.schema.clean_document import CleanDocument


BLOCK_TYPE_TO_CATEGORY = {
    "title": "Title",
    "paragraph": "NarrativeText",
    "list": "ListItem",
    "table": "Table",
    "caption": "Caption",
    "footnote": "Footnote",
    "formula": "Formula",
    "code": "Code",
}


def clean_document_to_element_records(doc: CleanDocument) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, block in enumerate(doc.blocks):
        category = BLOCK_TYPE_TO_CATEGORY.get(block.type, "UncategorizedText")
        text = block.text.strip()
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
                "category_depth": block.attrs.get("heading_level"),
                "parent_id": None,
                "languages": ["eng"],
                "element_id": block.block_id,
                "parser_backend": doc.parser_backend,
                "clean_block_type": block.type,
                "clean_block_attrs": block.attrs,
                "intermediate_schema": "docling_json_elements_v1" if doc.parser_backend == "docling_json" else "docling_elements",
                "table_data": block.table_data,
                "table_context": block.attrs.get("table_context", []),
                "possible_continuation": block.attrs.get("possible_continuation", False),
                "continuation_reason": block.attrs.get("continuation_reason"),
                "provenance": block.provenance,
                "source_refs": block.source_refs,
                "pages": sorted({p["page_no"] for p in block.provenance if isinstance(p.get("page_no"), int)}),
                "section_paths": [block.section_path] if block.section_path else [],
                **{key: doc.attrs.get(key) for key in IDENTITY_FIELDS},
            },
        }
        records.append(record)
    return records
