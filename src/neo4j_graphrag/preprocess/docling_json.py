"""Map native Docling JSON to our IR without Markdown/HTML reconstruction.

Only referenced body children determine reading order. Raw JSON stays untouched.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterator

from neo4j_graphrag.preprocess.chunk_utils import normalize_section_display
from neo4j_graphrag.preprocess.provenance import PIPELINE_VERSION, unique
from neo4j_graphrag.preprocess.schema.clean_document import CleanBlock, CleanDocument

LABEL_TYPES = {
    "title": "title",
    "section_header": "title",
    "text": "paragraph",
    "paragraph": "paragraph",
    "handwritten_text": "paragraph",
    "list_item": "list",
    "caption": "caption",
    "footnote": "footnote",
    "formula": "formula",
    "code": "code",
    "reference": "paragraph",
    "table": "table",
}
SKIP_LABELS = {
    "page_header",
    "page_footer",
    "picture",
    "chart",
    "document_index",
    "checkbox_selected",
    "checkbox_unselected",
    "empty_value",
}
UNIT_RE = re.compile(
    r"(?:in|\$\s*(?:and shares)?\s*in)\s+(?:thousands|millions|billions)|except.*per.share",
    re.I,
)
ITEM_RE = re.compile(r"^item\s+\d+[a-z]?\b", re.I)


def _link_following_table_footnotes(blocks: list[CleanBlock]) -> None:
    """Link marked notes after a contiguous table group on the same page.

    Keep the original text blocks. Do not infer links across prose, headings,
    pages, or ambiguous repeated definitions of the same marker.
    """
    marker_re = re.compile(r"\(([a-z])\)")
    note_re = re.compile(r"^\(([a-z])\)\s+\S")
    for index, block in enumerate(blocks):
        if block.type != "table" or block.page is None:
            continue
        following = index + 1
        while following < len(blocks):
            candidate = blocks[following]
            if (candidate.type != "table" or candidate.page != block.page
                    or candidate.section != block.section):
                break
            following += 1
        notes: dict[str, list[CleanBlock]] = {}
        for candidate in blocks[following:]:
            match = note_re.match(candidate.text)
            if (candidate.type not in {"paragraph", "footnote"}
                    or candidate.page != block.page
                    or candidate.section != block.section or not match):
                break
            notes.setdefault(match[1], []).append(candidate)
        markers = set(marker_re.findall(block.text))
        context = block.attrs.setdefault("table_context", [])
        for marker, definitions in notes.items():
            if marker not in markers or len(definitions) != 1:
                continue
            note = definitions[0]
            if block.bbox and note.bbox:
                # A note in another column is not evidence for this table.
                if (min(block.bbox[1][0], note.bbox[1][0])
                        <= max(block.bbox[0][0], note.bbox[0][0])
                        or note.bbox[0][1] < block.bbox[1][1]):
                    continue
            if any(c.get("source_ref") in note.source_refs for c in context):
                continue
            context.append({
                "kind": "matched_footnote",
                "marker": f"({marker})",
                "text": note.text,
                "source_ref": note.source_refs[0],
                "provenance": note.provenance,
                "association_method": "following_same_page_marker",
            })


def artifact_hash(document: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            document, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def resolve(document: dict[str, Any], reference: Any) -> dict[str, Any]:
    ref = (
        reference
        if isinstance(reference, str)
        else reference.get("$ref", reference.get("cref", ""))
    )
    if not ref.startswith("#/"):
        raise ValueError(f"Invalid Docling reference: {ref!r}")
    value: Any = document
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    if not isinstance(value, dict):
        raise ValueError(f"Reference does not point to an object: {ref}")
    return value


def iter_body(document: dict[str, Any]) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()

    def walk(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
        ref = node.get("self_ref", "")
        if ref in seen:
            return
        seen.add(ref)
        label = node.get("label", "")
        if node.get("content_layer") == "furniture" or label in {
            "page_header",
            "page_footer",
        }:
            return
        if label in LABEL_TYPES or ref.startswith(
            ("#/texts/", "#/tables/", "#/pictures/")
        ):
            yield node
        # Picture OCR is not treated as body prose; preserve captions/footnotes.
        children = [] if label in {"picture", "chart"} else node.get("children", [])
        for child in children:
            yield from walk(resolve(document, child))
        for key in ("captions", "footnotes"):
            for child in node.get(key, []):
                yield from walk(resolve(document, child))

    yield from walk(document["body"])


def provenance_records(
    document: dict[str, Any], item: dict[str, Any], source_ref: str
) -> list[dict[str, Any]]:
    result = []
    for prov in item.get("prov", []):
        page_no = prov.get("page_no")
        size = document.get("pages", {}).get(str(page_no), {}).get("size") or {}
        bbox = prov.get("bbox")
        normalized = None
        if bbox:
            origin = bbox.get("coord_origin")
            if origin == "TOPLEFT":
                normalized = dict(bbox)
            elif origin == "BOTTOMLEFT" and size.get("height") is not None:
                normalized = {
                    **bbox,
                    "t": size["height"] - bbox["t"],
                    "b": size["height"] - bbox["b"],
                    "coord_origin": "TOPLEFT",
                }
        result.append(
            {
                "page_no": page_no,
                "bbox": normalized,
                "raw_bbox": bbox,
                "page_size": size,
                "charspan": prov.get("charspan"),
                "source_ref": source_ref,
            }
        )
    return result


def docling_json_to_clean_document(
    *,
    document: dict[str, Any],
    doc_id: str,
    source_path: str,
    parse_info: dict[str, Any] | None = None,
) -> CleanDocument:
    if document.get("schema_name") != "DoclingDocument" or "body" not in document:
        raise ValueError("Expected native DoclingDocument JSON with body references")
    if not str(document.get("version", "")).startswith("1."):
        raise ValueError(
            "Unsupported Docling JSON schema version; inspect before migrating"
        )
    info = dict(parse_info or {})
    aid = artifact_hash(document)
    if info.get("artifact_id") and info["artifact_id"] != aid:
        raise ValueError("Docling JSON does not match its parse-info artifact hash")
    info.update(
        pipeline_version=PIPELINE_VERSION,
        artifact_id=aid,
        docling_schema_version=document.get("version"),
    )
    blocks: list[CleanBlock] = []
    section = ""
    headings: list[tuple[int, str]] = []
    item_anchor = ""
    skipped: dict[str, int] = {}
    previous_table: CleanBlock | None = None
    for node in iter_body(document):
        label = str(node.get("label", ""))
        block_type = LABEL_TYPES.get(label)
        if block_type is None:
            skipped[label] = skipped.get(label, 0) + 1
            continue
        ref = node["self_ref"]
        source_ref = f"{doc_id}::{aid}::{ref}"
        prov = provenance_records(document, node, source_ref)
        pages = sorted(
            {p["page_no"] for p in prov if isinstance(p.get("page_no"), int)}
        )
        text = str(node.get("text") or "").strip()
        attrs: dict[str, Any] = {"docling_label": label, "self_ref": ref}
        if block_type == "title" and UNIT_RE.search(text) and len(text) < 200:
            block_type = "paragraph"
            attrs["mapping_reason"] = "unit_annotation_not_section"
        elif block_type == "paragraph" and ITEM_RE.match(text) and len(text) <= 160:
            block_type = "title"
            attrs["mapping_reason"] = "sec_item_heading_text"
        if block_type == "title":
            text = " ".join(text.split())
            section = normalize_section_display(text)
            level = node.get("level")
            attrs.update(heading_level=level, section_source="heading_text")
            if isinstance(level, int) and level > 0:
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, section))
            else:
                headings = [(1, section)]
            if ITEM_RE.match(section):
                item_anchor = section
        section_path = unique(
            ([item_anchor] if item_anchor else []) + [h[1] for h in headings]
        )
        table_data = None
        if block_type == "table":
            data = node.get("data") or {}
            table_data = {
                "num_rows": data.get("num_rows", 0),
                "num_cols": data.get("num_cols", 0),
                "table_cells": [],
            }
            for index, original_cell in enumerate(data.get("table_cells", [])):
                cell = dict(original_cell)
                cell_ref = f"{source_ref}/data/table_cells/{index}"
                cell["source_refs"] = [cell_ref]
                # Cell bbox has no page number: only assign one for a single-page table.
                if cell.get("bbox") and len(pages) == 1:
                    cell["provenance"] = provenance_records(
                        document,
                        {"prov": [{"page_no": pages[0], "bbox": cell["bbox"]}]},
                        cell_ref,
                    )
                    cell["provenance_scope"] = "cell"
                else:
                    cell["provenance"] = prov
                    cell["provenance_scope"] = "table"
                table_data["table_cells"].append(cell)
            # Derived display text only; financial extraction uses table_data.
            text_rows: dict[int, list[tuple[int, str]]] = {}
            for cell in table_data["table_cells"]:
                text_rows.setdefault(int(cell["start_row_offset_idx"]), []).append(
                    (int(cell["start_col_offset_idx"]), str(cell.get("text", "")))
                )
            text = "\n".join(
                " | ".join(value for _, value in sorted(text_rows[row]))
                for row in sorted(text_rows)
            )
            context = []
            for relation in ("captions", "footnotes"):
                for reference in node.get(relation, []):
                    linked = resolve(document, reference)
                    context.append(
                        {
                            "kind": relation,
                            "text": linked.get("text", ""),
                            "source_ref": f"{doc_id}::{aid}::{linked['self_ref']}",
                        }
                    )
            # At most two preceding body texts; do not carry units across tables/pages/sections.
            for prior in reversed(blocks[-2:]):
                if (
                    prior.type not in {"paragraph", "caption", "footnote"}
                    or prior.section != section
                ):
                    break
                if prior.page not in pages:
                    break
                if UNIT_RE.search(prior.text):
                    context.append(
                        {
                            "kind": "preceding_unit_text",
                            "text": prior.text,
                            "source_ref": prior.source_refs[0],
                        }
                    )
            attrs["table_context"] = context
            continued = bool(
                re.search(
                    r"\bcontinued\b",
                    section + " " + " ".join(c["text"] for c in context),
                    re.I,
                )
            )
            adjacent = bool(
                previous_table
                and previous_table.page is not None
                and pages
                and pages[0] == previous_table.page + 1
                and previous_table.section == section
                and (previous_table.table_data or {}).get("num_cols")
                == table_data["num_cols"]
            )
            attrs["possible_continuation"] = continued or adjacent or len(pages) > 1
            attrs["continuation_reason"] = (
                "explicit_continued"
                if continued
                else "adjacent_same_section_and_width"
                if adjacent
                else "multiple_provenance_pages"
                if len(pages) > 1
                else None
            )
            attrs["has_column_headers"] = any(
                c.get("column_header") for c in table_data["table_cells"]
            )
        bbox = prov[0].get("bbox") if len(prov) == 1 else None
        block = CleanBlock(
            block_id=f"{doc_id}::{ref}",
            type=block_type,
            text=text,
            page=pages[0] if pages else None,
            section=section,
            order=len(blocks),
            bbox=[[bbox["l"], bbox["t"]], [bbox["r"], bbox["b"]]] if bbox else None,
            attrs=attrs,
            provenance=prov,
            source_refs=[source_ref],
            section_path=section_path,
            table_data=table_data,
        )
        blocks.append(block)
        if block_type == "table":
            previous_table = block
    _link_following_table_footnotes(blocks)
    info["skipped_labels"] = skipped
    return CleanDocument(
        doc_id=doc_id,
        source_path=source_path,
        parser_backend="docling_json",
        blocks=blocks,
        attrs=info,
    )
