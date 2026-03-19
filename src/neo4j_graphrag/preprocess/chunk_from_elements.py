from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from neo4j_graphrag.preprocess.ingest_chunks import (
    compute_chunk_id,
    normalize_section_display,
)


KEEP_CATEGORIES = {"Title", "NarrativeText", "Table", "ListItem", "UncategorizedText"}


@dataclass
class ElementRow:
    doc_id: str
    element_index: int
    category: str
    text: str
    page: Optional[int]
    section_hint: str
    bbox: Optional[list[list[float]]]
    metadata: dict[str, Any]


@dataclass
class ChunkBuffer:
    doc_id: str
    section_name: str = ""
    texts: list[str] = field(default_factory=list)
    element_indices: list[int] = field(default_factory=list)
    element_types: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    bboxes: list[list[list[float]]] = field(default_factory=list)
    table_html: Optional[str] = None

    def add(self, row: ElementRow) -> None:
        self.texts.append(row.text.strip())
        self.element_indices.append(row.element_index)
        self.element_types.append(row.category)
        if row.page is not None:
            self.pages.append(row.page)
        if row.bbox:
            self.bboxes.append(row.bbox)
        html = row.metadata.get("text_as_html")
        if isinstance(html, str) and html.strip() and row.category == "Table":
            self.table_html = html

    @property
    def char_count(self) -> int:
        return sum(len(x) for x in self.texts) + max(0, len(self.texts) - 1) * 2

    @property
    def is_empty(self) -> bool:
        return not any(t.strip() for t in self.texts)

    @property
    def is_table(self) -> bool:
        return "Table" in self.element_types

    def build_text(self) -> str:
        return "\n\n".join(x for x in self.texts if x.strip()).strip()

    def build_metadata(self) -> dict[str, Any]:
        page_start = min(self.pages) if self.pages else None
        page_end = max(self.pages) if self.pages else None
        bbox_union = _merge_bboxes(self.bboxes)
        return {
            "page_start": page_start,
            "page_end": page_end,
            "element_indices": self.element_indices,
            "element_types": self.element_types,
            "is_table": self.is_table,
            "bbox_union": bbox_union,
            "table_html": self.table_html,
        }


def _read_elements(path: Path) -> list[ElementRow]:
    rows: list[ElementRow] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        payload = json.loads(raw)
        rows.append(
            ElementRow(
                doc_id=str(payload.get("doc_id") or ""),
                element_index=int(payload.get("element_index") or 0),
                category=str(payload.get("category") or "UncategorizedText"),
                text=str(payload.get("text") or ""),
                page=payload.get("page"),
                section_hint=str(payload.get("section_hint") or ""),
                bbox=payload.get("bbox"),
                metadata=payload.get("metadata") or {},
            )
        )
    rows.sort(key=lambda r: (r.doc_id, r.page or 0, r.element_index))
    return rows


def _merge_bboxes(all_bboxes: list[list[list[float]]]) -> Optional[list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for bbox in all_bboxes:
        for point in bbox:
            if isinstance(point, list) and len(point) == 2:
                try:
                    xs.append(float(point[0]))
                    ys.append(float(point[1]))
                except (TypeError, ValueError):
                    continue
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _is_noise(row: ElementRow, min_text_len: int) -> bool:
    text = row.text.strip()
    if not text:
        return True
    if row.category not in KEEP_CATEGORIES:
        return True
    if row.category == "UncategorizedText" and len(text) < min_text_len:
        return True
    return False


def _make_record(
    *,
    buffer: ChunkBuffer,
    chunk_index: int,
    chunk_id_version: str,
    source_type: str,
    chunk_engine: str,
) -> Optional[dict[str, Any]]:
    text = buffer.build_text()
    if not text:
        return None
    section_name = normalize_section_display(buffer.section_name)
    chunk_id = compute_chunk_id(
        version=chunk_id_version,
        doc_id=buffer.doc_id,
        section_name=section_name,
        chunk_index=chunk_index,
        text=text,
    )
    return {
        "chunk_id": chunk_id,
        "chunk_id_version": chunk_id_version,
        "doc_id": buffer.doc_id,
        "text": text,
        "chunk_index": chunk_index,
        "section_name": section_name,
        "page": min(buffer.pages) if buffer.pages else None,
        "source_type": source_type,
        "parser_engine": "unstructured_elements",
        "chunk_engine": chunk_engine,
        "metadata": buffer.build_metadata(),
    }


def _new_buffer(doc_id: str, section_name: str) -> ChunkBuffer:
    return ChunkBuffer(doc_id=doc_id, section_name=section_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build section-aware chunks from elements.ndjson."
    )
    parser.add_argument("--elements-path", required=True, help="Input elements NDJSON path.")
    parser.add_argument("--output", required=True, help="Output chunks NDJSON path.")
    parser.add_argument(
        "--chunk-id-version",
        default="v1",
        help="Version prefix embedded in chunk_id hash payload.",
    )
    parser.add_argument(
        "--target-max-chars",
        type=int,
        default=2048,
        help="Keep merging small elements until this size is reached, without breaking elements.",
    )
    parser.add_argument(
        "--min-uncategorized-len",
        type=int,
        default=12,
        help="Drop UncategorizedText shorter than this threshold.",
    )
    parser.add_argument(
        "--merge-table-fragments",
        action="store_true",
        help="Attach short same-page UncategorizedText immediately after a table back into that table chunk.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    elements_path = Path(args.elements_path).expanduser().resolve()
    if not elements_path.exists():
        print(f"Elements path does not exist: {elements_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_elements(elements_path)
    if not rows:
        print("No elements found.", file=sys.stderr)
        sys.exit(1)

    out_records: list[dict[str, Any]] = []
    current_doc: Optional[str] = None
    current_section = "UNSPECIFIED"
    buffer: Optional[ChunkBuffer] = None
    chunk_index = 0
    pending_table_record: Optional[dict[str, Any]] = None

    def flush_buffer() -> None:
        nonlocal buffer, chunk_index
        if buffer is None or buffer.is_empty:
            buffer = None
            return
        record = _make_record(
            buffer=buffer,
            chunk_index=chunk_index,
            chunk_id_version=args.chunk_id_version,
            source_type="pdf",
            chunk_engine="elements_structured",
        )
        if record is not None:
            out_records.append(record)
            chunk_index += 1
        buffer = None

    for row in rows:
        if _is_noise(row, args.min_uncategorized_len):
            continue

        if current_doc != row.doc_id:
            flush_buffer()
            current_doc = row.doc_id
            current_section = "UNSPECIFIED"
            buffer = _new_buffer(doc_id=row.doc_id, section_name=current_section)
            pending_table_record = None

        if row.category == "Title":
            flush_buffer()
            current_section = normalize_section_display(row.text) or row.section_hint or current_section
            buffer = _new_buffer(doc_id=row.doc_id, section_name=current_section)
            pending_table_record = None
            continue

        if row.category == "Table":
            flush_buffer()
            table_buffer = _new_buffer(doc_id=row.doc_id, section_name=current_section)
            table_buffer.add(row)
            record = _make_record(
                buffer=table_buffer,
                chunk_index=chunk_index,
                chunk_id_version=args.chunk_id_version,
                source_type="pdf",
                chunk_engine="elements_table",
            )
            if record is not None:
                out_records.append(record)
                pending_table_record = record
                chunk_index += 1
            buffer = _new_buffer(doc_id=row.doc_id, section_name=current_section)
            continue

        if (
            args.merge_table_fragments
            and pending_table_record is not None
            and row.category == "UncategorizedText"
            and row.page == pending_table_record.get("page")
            and len(row.text.strip()) <= max(80, args.min_uncategorized_len)
        ):
            pending_table_record["text"] = f"{pending_table_record['text']}\n{row.text.strip()}".strip()
            meta = pending_table_record.get("metadata") or {}
            meta["element_indices"] = list(meta.get("element_indices") or []) + [row.element_index]
            meta["element_types"] = list(meta.get("element_types") or []) + [row.category]
            pending_table_record["metadata"] = meta
            continue

        pending_table_record = None
        if buffer is None:
            buffer = _new_buffer(doc_id=row.doc_id, section_name=current_section)

        projected = buffer.char_count + len(row.text.strip()) + (2 if buffer.texts else 0)
        if buffer.texts and projected > args.target_max_chars:
            flush_buffer()
            buffer = _new_buffer(doc_id=row.doc_id, section_name=current_section)

        buffer.add(row)

    flush_buffer()

    with output_path.open("w", encoding="utf-8") as out:
        for record in out_records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Done.")
    print(f"Input:  {elements_path}")
    print(f"Output: {output_path}")
    print(f"Chunks: {len(out_records)}")


if __name__ == "__main__":
    main()
