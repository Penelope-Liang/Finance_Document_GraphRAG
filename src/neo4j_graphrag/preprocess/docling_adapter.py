from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from neo4j_graphrag.preprocess.ingest_chunks import iter_input_files, normalize_section_display
from neo4j_graphrag.preprocess.schema.clean_document import CleanBlock, CleanDocument
from neo4j_graphrag.preprocess.export.to_elements_ndjson import clean_document_to_element_records


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_LIST_RE = re.compile(r"^\s*[-*]\s+(.+?)\s*$")
_TABLE_RULE_RE = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")


def _normalize_space(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_cell_text(text: str) -> str:
    value = _normalize_space(text)
    value = re.sub(r"\s+%", "%", value)
    value = re.sub(r"\$\s+", "$ ", value)
    value = re.sub(r"\s+([,.;:])", r"\1", value)
    return value


def _strip_outer_pipes(line: str) -> str:
    value = line.strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|"):
        value = value[:-1]
    return value


def _split_markdown_row(line: str) -> list[str]:
    return [_clean_cell_text(part) for part in _strip_outer_pipes(line).split("|")]


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def _table_lines_to_grid(lines: list[str]) -> list[list[dict[str, Any]]]:
    rows: list[list[str]] = []
    for line in lines:
        if _TABLE_RULE_RE.match(line):
            continue
        rows.append(_split_markdown_row(line))
    if not rows:
        return []

    max_cols = max(len(row) for row in rows)
    grid: list[list[dict[str, Any]]] = []
    for row_idx, row in enumerate(rows):
        padded = row + [""] * (max_cols - len(row))
        grid.append(
            [
                {
                    "row": row_idx,
                    "col": col_idx,
                    "text": cell,
                    "kind": "HEADER" if row_idx == 0 else "",
                }
                for col_idx, cell in enumerate(padded)
            ]
        )
    return grid


def _table_grid_to_html(grid: list[list[dict[str, Any]]]) -> str:
    if not grid:
        return ""
    head = grid[0]
    body = grid[1:]
    head_html = "".join(f"<th>{html.escape(cell['text'])}</th>" for cell in head)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell['text'])}</td>" for cell in row) + "</tr>"
        for row in body
    )
    if body_html:
        return f"<table><thead><tr>{head_html}</tr></thead><tbody>{body_html}</tbody></table>"
    single_row_html = "".join(f"<td>{html.escape(cell['text'])}</td>" for cell in head)
    return f"<table><tbody><tr>{single_row_html}</tr></tbody></table>"


def _table_grid_to_text(grid: list[list[dict[str, Any]]]) -> str:
    lines: list[str] = []
    for row in grid:
        text = " ".join(cell["text"] for cell in row if cell["text"])
        if text:
            lines.append(text)
    return "\n".join(lines).strip()


def _flush_paragraph(
    *,
    paragraph_lines: list[str],
    blocks: list[CleanBlock],
    doc_id: str,
    current_section: str,
    order_counter: int,
) -> int:
    if not paragraph_lines:
        return order_counter
    text = _normalize_space(" ".join(paragraph_lines))
    if not text:
        return order_counter
    block_type = "paragraph"
    match = _LIST_RE.match(text)
    if match:
        block_type = "list"
        text = _normalize_space(match.group(1))
    blocks.append(
        CleanBlock(
            block_id=f"{doc_id}_b_{order_counter:04d}",
            type=block_type,
            text=text,
            page=None,
            section=current_section,
            order=order_counter,
        )
    )
    return order_counter + 1


def markdown_to_clean_document(*, markdown_text: str, doc_id: str, source_path: str) -> CleanDocument:
    lines = markdown_text.splitlines()
    blocks: list[CleanBlock] = []
    current_section = ""
    paragraph_lines: list[str] = []
    order_counter = 1
    i = 0

    while i < len(lines):
        raw = lines[i].rstrip()
        stripped = raw.strip()
        if not stripped:
            order_counter = _flush_paragraph(
                paragraph_lines=paragraph_lines,
                blocks=blocks,
                doc_id=doc_id,
                current_section=current_section,
                order_counter=order_counter,
            )
            paragraph_lines = []
            i += 1
            continue

        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            order_counter = _flush_paragraph(
                paragraph_lines=paragraph_lines,
                blocks=blocks,
                doc_id=doc_id,
                current_section=current_section,
                order_counter=order_counter,
            )
            paragraph_lines = []
            heading_text = _clean_cell_text(heading_match.group(2))
            current_section = normalize_section_display(heading_text)
            blocks.append(
                CleanBlock(
                    block_id=f"{doc_id}_b_{order_counter:04d}",
                    type="title",
                    text=heading_text,
                    page=None,
                    section=current_section,
                    order=order_counter,
                    attrs={"heading_level": len(heading_match.group(1))},
                )
            )
            order_counter += 1
            i += 1
            continue

        if _is_table_line(stripped):
            order_counter = _flush_paragraph(
                paragraph_lines=paragraph_lines,
                blocks=blocks,
                doc_id=doc_id,
                current_section=current_section,
                order_counter=order_counter,
            )
            paragraph_lines = []
            table_lines: list[str] = []
            while i < len(lines) and _is_table_line(lines[i].strip()):
                table_lines.append(lines[i].strip())
                i += 1
            grid = _table_lines_to_grid(table_lines)
            table_text = _table_grid_to_text(grid)
            table_html = _table_grid_to_html(grid)
            blocks.append(
                CleanBlock(
                    block_id=f"{doc_id}_b_{order_counter:04d}",
                    type="table",
                    text=table_text,
                    page=None,
                    section=current_section,
                    order=order_counter,
                    html=table_html,
                    grid=grid,
                    attrs={"source": "docling_markdown"},
                )
            )
            order_counter += 1
            continue

        paragraph_lines.append(stripped)
        i += 1

    order_counter = _flush_paragraph(
        paragraph_lines=paragraph_lines,
        blocks=blocks,
        doc_id=doc_id,
        current_section=current_section,
        order_counter=order_counter,
    )

    return CleanDocument(
        doc_id=doc_id,
        source_path=source_path,
        parser_backend="docling",
        blocks=blocks,
    )


def _convert_pdf(path: Path) -> str:
    try:
        from docling.document_converter import DocumentConverter
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Docling is not available. Install it in the current Python environment first."
        ) from exc

    converter = DocumentConverter()
    result = converter.convert(str(path))
    document = result.document
    return document.export_to_markdown()


def _iter_pdfs(path: Path) -> list[Path]:
    return [p for p in iter_input_files(path) if p.suffix.lower() == ".pdf"]


def _write_ndjson(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert PDFs with Docling into CleanDocument JSONL and optional elements.ndjson export."
    )
    parser.add_argument("--input-path", required=True, help="PDF file or directory.")
    parser.add_argument("--clean-output", required=True, help="Output path for clean_document JSONL.")
    parser.add_argument(
        "--elements-output",
        default="",
        help="Optional output path for elements.ndjson-compatible export.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail fast on first file error.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        print(f"Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    files = _iter_pdfs(input_path)
    if not files:
        print("No PDF files found.", file=sys.stderr)
        sys.exit(1)

    clean_output = Path(args.clean_output).expanduser().resolve()
    elements_output = Path(args.elements_output).expanduser().resolve() if args.elements_output else None

    clean_rows: list[dict[str, Any]] = []
    element_rows: list[dict[str, Any]] = []
    processed = 0
    errored = 0

    for file_path in files:
        rel_doc_id = file_path.relative_to(input_path).as_posix() if input_path.is_dir() else file_path.name
        try:
            markdown = _convert_pdf(file_path)
            clean_doc = markdown_to_clean_document(
                markdown_text=markdown,
                doc_id=rel_doc_id,
                source_path=str(file_path),
            )
            clean_rows.append(clean_doc.to_dict())
            if elements_output is not None:
                element_rows.extend(clean_document_to_element_records(clean_doc))
            processed += 1
            print(f"[ok] {file_path} -> {len(clean_doc.blocks)} blocks")
        except Exception as exc:
            errored += 1
            print(f"[error] {file_path}: {exc}", file=sys.stderr)
            if args.strict:
                raise

    _write_ndjson(clean_output, clean_rows)
    if elements_output is not None:
        _write_ndjson(elements_output, element_rows)

    print("\nDone.")
    print(f"Clean output: {clean_output}")
    if elements_output is not None:
        print(f"Elements output: {elements_output}")
    print(f"Files processed: {processed}")
    print(f"Files errored: {errored}")


if __name__ == "__main__":
    main()
