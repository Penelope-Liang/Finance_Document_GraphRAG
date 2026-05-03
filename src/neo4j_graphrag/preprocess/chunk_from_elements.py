from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import re

from neo4j_graphrag.preprocess.ingest_chunks import (
    compute_chunk_id,
    normalize_section_display,
)


KEEP_CATEGORIES = {"Title", "NarrativeText", "Table", "ListItem"}
OPTIONAL_CATEGORIES = {"UncategorizedText"}


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
    title_text: str = ""
    texts: list[str] = field(default_factory=list)
    element_indices: list[int] = field(default_factory=list)
    element_types: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    bboxes: list[list[list[float]]] = field(default_factory=list)
    table_html: Optional[str] = None
    parser_engines: list[str] = field(default_factory=list)
    intermediate_schemas: list[str] = field(default_factory=list)

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
        parser_backend = str(row.metadata.get("parser_backend") or "").strip().lower()
        if parser_backend:
            self.parser_engines.append(parser_backend)
        intermediate_schema = str(row.metadata.get("intermediate_schema") or "").strip()
        if intermediate_schema:
            self.intermediate_schemas.append(intermediate_schema)

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
        preview_text = self.build_text()
        is_toc_like = _is_toc_like_text(preview_text, self.is_table)
        is_front_matter = _is_front_matter_chunk(
            section_name=self.section_name,
            text=preview_text,
            page_start=page_start,
            is_toc_like=is_toc_like,
        )
        is_appendix_like = _is_appendix_like_chunk(
            section_name=self.section_name,
            title_text=self.title_text,
            text=preview_text,
        )
        return {
            "page_start": page_start,
            "page_end": page_end,
            "element_indices": self.element_indices,
            "element_types": self.element_types,
            "is_table": self.is_table,
            "bbox_union": bbox_union,
            "table_html": self.table_html,
            "title_text": self.title_text or "",
            "is_toc_like": is_toc_like,
            "is_front_matter": is_front_matter,
            "is_appendix_like": is_appendix_like,
            "should_index": not (is_toc_like or is_front_matter or is_appendix_like),
            "parser_engine": _choose_label(self.parser_engines, default="unstructured"),
            "intermediate_schema": _choose_label(
                self.intermediate_schemas,
                default="unstructured_compatible",
            ),
            "chunker": "legacy_chunk_from_elements",
        }


def _choose_label(values: list[str], *, default: str) -> str:
    if not values:
        return default
    counts: dict[str, int] = {}
    for value in values:
        clean = value.strip()
        if not clean:
            continue
        counts[clean] = counts.get(clean, 0) + 1
    if not counts:
        return default
    return max(counts.items(), key=lambda item: item[1])[0]


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
    if row.category in KEEP_CATEGORIES:
        return False
    if row.category in OPTIONAL_CATEGORIES and len(text) < min_text_len:
        return True
    if row.category in OPTIONAL_CATEGORIES:
        return False
    return True


def _looks_like_substantive_uncategorized(text: str, min_text_len: int) -> bool:
    clean = text.strip()
    if len(clean) < min_text_len:
        return False
    if len(clean.split()) >= 4:
        return True
    return bool(re.search(r"[.:;,\d$%]", clean))


def _is_toc_like_text(text: str, is_table: bool) -> bool:
    clean = " ".join((text or "").split()).lower()
    if not clean:
        return False
    if "table of contents" in clean or "beginning page" in clean:
        return True
    continued_hits = sum(
        1
        for phrase in (
            "continued",
            "form 10-k summary",
            "financial statements and supplementary data",
            "index to financial statements",
        )
        if phrase in clean
    )
    item_hits = len(re.findall(r"\bitem\s+\d+[a-z]?\b", clean))
    note_hits = len(re.findall(r"\bnote\s+\d+\b", clean))
    page_number_hits = len(re.findall(r"\b\d{1,3}\b", clean))
    toc_label_hits = sum(
        clean.count(label)
        for label in (
            "directors, executive officers",
            "controls and procedures",
            "executive compensation",
            "exhibits, financial statement schedules",
            "certain relationships and related transactions",
            "principal accounting fees and services",
            "other information",
        )
    )
    if is_table and (
        item_hits >= 3
        or (note_hits >= 4 and page_number_hits >= 8)
        or (continued_hits >= 1 and page_number_hits >= 6)
        or (toc_label_hits >= 2 and page_number_hits >= 6)
    ):
        return True
    if item_hits >= 4 and page_number_hits >= 8:
        return True
    if note_hits >= 5 and page_number_hits >= 8:
        return True
    if continued_hits >= 1 and (item_hits + note_hits >= 2 or page_number_hits >= 6):
        return True
    if toc_label_hits >= 2 and page_number_hits >= 6:
        return True
    return False


def _is_front_matter_chunk(
    *,
    section_name: str,
    text: str,
    page_start: Optional[int],
    is_toc_like: bool,
) -> bool:
    clean = " ".join((text or "").split()).lower()
    section = (section_name or "").strip().lower()
    if is_toc_like:
        return True
    front_matter_signals = (
        "united states securities and exchange commission",
        "form 10-k",
        "commission file number",
        "documents incorporated by reference",
        "indicate by check mark",
    )
    if any(x in clean for x in front_matter_signals):
        return True
    if any(x in section for x in front_matter_signals):
        return True
    if page_start is not None and page_start <= 3:
        if "yes ☒ no ☐" in clean or "yes ☐ no ☒" in clean:
            return True
        if "securities registered pursuant" in clean:
            return True
    if page_start is not None and page_start <= 5:
        if "continued" in clean and (
            "financial statements and supplementary data" in clean
            or "form 10-k summary" in clean
            or "index to financial statements" in clean
        ):
            return True
    return False


def _is_appendix_like_chunk(*, section_name: str, title_text: str, text: str) -> bool:
    combined = " ".join(
        part.strip().lower()
        for part in (section_name or "", title_text or "", text or "")
        if part and part.strip()
    )
    if not combined:
        return False

    strong_signals = (
        "award agreement",
        "deferred stock unit award agreement",
        "taxation and tax withholding",
        "nonemployee director",
        "long-term incentive plan",
        "stock unit award agreement",
    )
    if any(sig in combined for sig in strong_signals):
        return True

    weak_hits = sum(
        1
        for sig in (
            "agreement",
            "appendix",
            "exhibit",
            "plan",
            "participant",
            "grant date",
            "vesting date",
            "tax-related items",
            "administrator",
        )
        if sig in combined
    )
    if weak_hits >= 3:
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
    metadata = buffer.build_metadata()
    return {
        "chunk_id": chunk_id,
        "chunk_id_version": chunk_id_version,
        "doc_id": buffer.doc_id,
        "text": text,
        "chunk_index": chunk_index,
        "section_name": section_name,
        "page": min(buffer.pages) if buffer.pages else None,
        "source_type": source_type,
        "parser_engine": metadata.get("parser_engine", "unstructured"),
        "chunk_engine": chunk_engine,
        "metadata": metadata,
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
    pending_title: Optional[ElementRow] = None

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
        if row.category == "UncategorizedText" and not _looks_like_substantive_uncategorized(
            row.text,
            args.min_uncategorized_len,
        ):
            continue

        if current_doc != row.doc_id:
            flush_buffer()
            current_doc = row.doc_id
            current_section = "UNSPECIFIED"
            buffer = _new_buffer(doc_id=row.doc_id, section_name=current_section)
            pending_table_record = None
            pending_title = None

        if row.category == "Title":
            flush_buffer()
            current_section = normalize_section_display(row.text) or row.section_hint or current_section
            buffer = _new_buffer(doc_id=row.doc_id, section_name=current_section)
            pending_table_record = None
            pending_title = row
            continue

        if row.category == "Table":
            flush_buffer()
            table_buffer = _new_buffer(doc_id=row.doc_id, section_name=current_section)
            if pending_title is not None:
                table_buffer.title_text = pending_title.text.strip()
                table_buffer.add(pending_title)
                pending_title = None
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
        if pending_title is not None and buffer.is_empty:
            buffer.title_text = pending_title.text.strip()
            buffer.add(pending_title)
            pending_title = None

        projected = buffer.char_count + len(row.text.strip()) + (2 if buffer.texts else 0)
        if buffer.texts and projected > args.target_max_chars:
            flush_buffer()
            buffer = _new_buffer(doc_id=row.doc_id, section_name=current_section)
            if pending_title is not None and buffer.is_empty:
                buffer.title_text = pending_title.text.strip()
                buffer.add(pending_title)
                pending_title = None

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
