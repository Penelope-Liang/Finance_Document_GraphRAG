from __future__ import annotations

import argparse
import json
import math
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional


_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_NUMBER_RE = re.compile(r"[+-]?(?:\d[\d,]*\.?\d*|\.\d+)")
_SCALE_HINTS = (
    ("billion", 1_000_000_000.0),
    ("million", 1_000_000.0),
    ("thousand", 1_000.0),
)
_EMPTY_VALUES = {
    "",
    "-",
    "—",
    "–",
    "~",
    "n/a",
    "nm",
    "n.m.",
    "$—",
    "—%",
    "$ -",
    "$-",
    "-%",
    "%",
    "$",
}


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _safe_int(value: Optional[str], default: int = 1) -> int:
    try:
        return max(default, int(value or default))
    except (TypeError, ValueError):
        return default


def _is_empty_text(text: str) -> bool:
    return _normalize_space(text).casefold() in _EMPTY_VALUES


def _is_year(text: str) -> bool:
    clean = _normalize_space(text)
    return bool(clean and _YEAR_RE.fullmatch(clean))


def _extract_number(text: str) -> Optional[float]:
    clean = _normalize_space(text)
    if not clean:
        return None
    if _is_year(clean):
        return None
    stripped = clean.replace("$", "").replace("€", "").replace("£", "")
    stripped = stripped.replace("¥", "").replace(",", "").replace(" ", "")
    negative = stripped.startswith("(") and stripped.endswith(")")
    stripped = stripped.strip("()")
    match = _NUMBER_RE.search(stripped)
    if not match:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    if negative:
        value = -value
    if "%" in clean:
        value /= 100.0
    return value


def _infer_scale(texts: list[str]) -> tuple[str, float]:
    haystack = " ".join(_normalize_space(text).casefold() for text in texts if text).strip()
    for label, factor in _SCALE_HINTS:
        if f"in {label}" in haystack or label in haystack:
            return label, factor
    if "%" in haystack or "percent" in haystack:
        return "percent", 0.01
    return "", 1.0


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cells: list[dict[str, Any]] = []
        self._in_table = False
        self._row_index = -1
        self._open_cell: Optional[dict[str, Any]] = None
        self._capture: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "table":
            self._in_table = True
            return
        if not self._in_table:
            return
        if tag == "tr":
            self._row_index += 1
            return
        if tag in {"td", "th"}:
            self._open_cell = {
                "row": self._row_index,
                "colspan": _safe_int(attrs_dict.get("colspan"), 1),
                "rowspan": _safe_int(attrs_dict.get("rowspan"), 1),
                "is_header": tag == "th",
            }
            self._capture = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self._in_table = False
            return
        if not self._in_table:
            return
        if tag in {"td", "th"} and self._open_cell is not None:
            cell = dict(self._open_cell)
            cell["text"] = _normalize_space("".join(self._capture))
            self.cells.append(cell)
            self._open_cell = None
            self._capture = []

    def handle_data(self, data: str) -> None:
        if self._open_cell is not None:
            self._capture.append(data)


def _parse_table_html(table_html: str) -> list[list[dict[str, Any]]]:
    parser = _TableHTMLParser()
    parser.feed(table_html)
    if not parser.cells:
        return []

    grid: list[list[Optional[dict[str, Any]]]] = []
    open_spans: dict[tuple[int, int], dict[str, Any]] = {}

    def ensure_pos(row_idx: int, col_idx: int) -> None:
        while len(grid) <= row_idx:
            grid.append([])
        row = grid[row_idx]
        while len(row) <= col_idx:
            row.append(None)

    cells_by_row: dict[int, list[dict[str, Any]]] = {}
    for cell in parser.cells:
        cells_by_row.setdefault(int(cell["row"]), []).append(cell)

    max_row = max(cells_by_row) if cells_by_row else -1
    for row_idx in range(max_row + 1):
        row_cells = cells_by_row.get(row_idx, [])
        col_idx = 0
        for cell in row_cells:
            while (row_idx, col_idx) in open_spans:
                span_cell = open_spans[(row_idx, col_idx)]
                ensure_pos(row_idx, col_idx)
                grid[row_idx][col_idx] = dict(span_cell, is_span_ref=True)
                del open_spans[(row_idx, col_idx)]
                col_idx += 1

            base_cell = {
                "row_idx": row_idx,
                "col_idx": col_idx,
                "text": cell["text"],
                "is_header": bool(cell["is_header"]),
                "rowspan": int(cell["rowspan"]),
                "colspan": int(cell["colspan"]),
                "is_span_ref": False,
            }
            for r_off in range(base_cell["rowspan"]):
                for c_off in range(base_cell["colspan"]):
                    target_row = row_idx + r_off
                    target_col = col_idx + c_off
                    ensure_pos(target_row, target_col)
                    if r_off == 0 and c_off == 0:
                        grid[target_row][target_col] = base_cell
                    else:
                        span_ref = dict(base_cell, is_span_ref=True)
                        if target_row == row_idx:
                            grid[target_row][target_col] = span_ref
                        else:
                            open_spans[(target_row, target_col)] = span_ref
            col_idx += base_cell["colspan"]

        while (row_idx, col_idx) in open_spans:
            span_cell = open_spans[(row_idx, col_idx)]
            ensure_pos(row_idx, col_idx)
            grid[row_idx][col_idx] = dict(span_cell, is_span_ref=True)
            del open_spans[(row_idx, col_idx)]
            col_idx += 1

    return [[cell or {
        "row_idx": row_idx,
        "col_idx": col_idx,
        "text": "",
        "is_header": False,
        "rowspan": 1,
        "colspan": 1,
        "is_span_ref": False,
    } for col_idx, cell in enumerate(row)] for row_idx, row in enumerate(grid)]


def _classify_grid(
    grid: list[list[dict[str, Any]]],
    *,
    title_text: str,
    section_name: str,
    chunk_text: str,
) -> list[list[dict[str, Any]]]:
    if not grid:
        return grid

    scale_label, scale_factor = _infer_scale([title_text, section_name, chunk_text] + [cell["text"] for row in grid for cell in row])

    row_has_number: dict[int, bool] = {}
    col_has_number: dict[int, bool] = {}
    for row in grid:
        for cell in row:
            num = _extract_number(cell["text"])
            if num is not None:
                row_has_number[cell["row_idx"]] = True
                col_has_number[cell["col_idx"]] = True

    for row in grid:
        for cell in row:
            text = cell["text"]
            cell_type = "TEXT"
            numeric_value = None
            normalized_value = None

            if _is_empty_text(text):
                cell_type = "EMPTY"
            elif _is_year(text):
                cell_type = "YEAR"
            else:
                number = _extract_number(text)
                if number is not None:
                    cell_type = "NUMBER"
                    numeric_value = number
                    normalized_value = number * scale_factor if scale_factor not in {0.01, 1.0} else number
                elif cell["is_header"] or (cell["col_idx"] == 0 and row_has_number.get(cell["row_idx"], False)):
                    cell_type = "ROW_HEADER"
                elif cell["row_idx"] == 0 and col_has_number.get(cell["col_idx"], False):
                    cell_type = "COL_HEADER"
                elif col_has_number.get(cell["col_idx"], False):
                    cell_type = "COL_HEADER"
                elif row_has_number.get(cell["row_idx"], False):
                    cell_type = "ROW_HEADER"

            cell["cell_type"] = cell_type
            cell["numeric_value"] = numeric_value
            cell["scale"] = scale_label if numeric_value is not None else ""
            cell["normalized_value"] = normalized_value

    return grid


def _find_left_header(grid: list[list[dict[str, Any]]], row_idx: int, col_idx: int) -> str:
    row = grid[row_idx]
    for idx in range(col_idx - 1, -1, -1):
        cell = row[idx]
        if cell["cell_type"] in {"ROW_HEADER", "TEXT"} and not _is_empty_text(cell["text"]):
            return cell["text"]
    return ""


def _find_top_header(grid: list[list[dict[str, Any]]], row_idx: int, col_idx: int) -> str:
    for idx in range(row_idx - 1, -1, -1):
        cell = grid[idx][col_idx]
        if cell["cell_type"] in {"COL_HEADER", "YEAR", "TEXT"} and not _is_empty_text(cell["text"]):
            return cell["text"]
    return ""


def _build_table_structured(
    *,
    table_html: str,
    title_text: str,
    section_name: str,
    chunk_text: str,
) -> dict[str, Any]:
    grid = _parse_table_html(table_html)
    grid = _classify_grid(
        grid,
        title_text=title_text,
        section_name=section_name,
        chunk_text=chunk_text,
    )

    cells: list[dict[str, Any]] = []
    numeric_cells: list[dict[str, Any]] = []
    for row in grid:
        for cell in row:
            record = {
                "row_idx": cell["row_idx"],
                "col_idx": cell["col_idx"],
                "text": cell["text"],
                "cell_type": cell["cell_type"],
                "is_header": bool(cell["is_header"]),
                "is_span_ref": bool(cell["is_span_ref"]),
                "rowspan": int(cell["rowspan"]),
                "colspan": int(cell["colspan"]),
            }
            if cell["numeric_value"] is not None:
                record["numeric_value"] = cell["numeric_value"]
                record["scale"] = cell["scale"]
                record["normalized_value"] = cell["normalized_value"]
                record["row_header"] = _find_left_header(grid, cell["row_idx"], cell["col_idx"])
                record["col_header"] = _find_top_header(grid, cell["row_idx"], cell["col_idx"])
                numeric_cells.append(record)
            cells.append(record)

    rows = [
        [{"text": cell["text"], "cell_type": cell["cell_type"]} for cell in row]
        for row in grid
    ]
    return {
        "n_rows": len(grid),
        "n_cols": max((len(row) for row in grid), default=0),
        "rows": rows,
        "cells": cells,
        "numeric_cells": numeric_cells,
    }


def _iter_ndjson(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError(f"Line {line_no} is not a JSON object.")
        rows.append(obj)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attach structured table objects to chunks.ndjson table rows."
    )
    parser.add_argument("--chunks-path", required=True, help="Input chunks.ndjson path.")
    parser.add_argument("--output", required=True, help="Output ndjson path.")
    parser.add_argument(
        "--only-should-index",
        action="store_true",
        help="Only process rows whose metadata.should_index is true.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast if one table_html row cannot be parsed.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    chunks_path = Path(args.chunks_path).expanduser().resolve()
    if not chunks_path.exists():
        print(f"Chunks path does not exist: {chunks_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _iter_ndjson(chunks_path)
    processed = 0
    errors = 0

    with output_path.open("w", encoding="utf-8") as out:
        for row in rows:
            metadata = dict(row.get("metadata") or {})
            should_index = bool(metadata.get("should_index", True))
            if args.only_should_index and not should_index:
                row["metadata"] = metadata
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            table_html = str(metadata.get("table_html") or "").strip()
            is_table = bool(metadata.get("is_table")) or str(row.get("chunk_engine") or "") == "elements_table"
            if is_table and table_html:
                try:
                    metadata["table_structured"] = _build_table_structured(
                        table_html=table_html,
                        title_text=str(metadata.get("title_text") or ""),
                        section_name=str(row.get("section_name") or ""),
                        chunk_text=str(row.get("text") or ""),
                    )
                    processed += 1
                except Exception as exc:
                    errors += 1
                    metadata["table_structured_error"] = str(exc)
                    if args.strict:
                        raise

            row["metadata"] = metadata
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Done.")
    print(f"Input:      {chunks_path}")
    print(f"Output:     {output_path}")
    print(f"Processed:  {processed}")
    print(f"Errors:     {errors}")


if __name__ == "__main__":
    main()
