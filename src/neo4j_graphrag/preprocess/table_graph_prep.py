from __future__ import annotations

import argparse
from copy import deepcopy
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional


_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_NUMBER_RE = re.compile(r"[+-]?(?:\d[\d,]*\.?\d*|\.\d+)")
_NUMERIC_CELL_RE = re.compile(
    r"""
    ^\s*
    \(?
    [\$€£¥]?
    \s*
    [+-]?
    (?:
        \d[\d,]*\.?\d*
        |
        \.\d+
    )
    \)?
    \s*
    %?
    \s*[\$€£¥]?
    \s*$
    """,
    re.VERBOSE,
)
_SCALE_HINTS = (
    ("billion", 1_000_000_000.0),
    ("million", 1_000_000.0),
    ("thousand", 1_000.0),
)
_PERIOD_HEADER_HINTS = (
    "three months ended",
    "year ended",
    "months ended",
    "quarter ended",
    "quarters ended",
    "as of",
    "at december",
    "at january",
    "at march",
    "at june",
    "at september",
    "at october",
    "at november",
)
_PERCENT_CONTEXT_HINTS = (
    "%",
    "percent",
    "margin",
    "% of total",
    "%of total",
    "% change",
    "%change",
    # Removed: "change", "mix", "growth", "translation", "divestitures" —
    # these words appear on dollar-amount rows too and caused false PERCENT labels.
)
_ROW_PERCENT_HINTS = (
    "price",
    "volume",
    "volume-organic",
    "volume -organic",
    "volume - organic",
    "organic local-currency sales",
    "organic local currency sales",
    "organic sales change",
    "organic sales growth",
    "sales change",
    "local-currency sales change",
    "local currency sales change",
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


def _is_empty_text(text: str) -> bool:
    return _normalize_space(text).casefold() in _EMPTY_VALUES


def _is_year(text: str) -> bool:
    clean = _normalize_space(text)
    return bool(clean and _YEAR_RE.fullmatch(clean))


def _contains_alpha(text: str) -> bool:
    return any(char.isalpha() for char in _normalize_space(text))


def _is_period_header_text(text: str) -> bool:
    clean = _normalize_space(text).casefold()
    if not clean:
        return False
    if any(hint in clean for hint in _PERIOD_HEADER_HINTS):
        return True
    if _YEAR_RE.search(clean) and any(month in clean for month in (
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    )):
        return True
    return False


def _looks_numeric_cell(text: str) -> bool:
    clean = _normalize_space(text)
    if not clean or _is_year(clean):
        return False
    if _contains_alpha(clean) or _is_period_header_text(clean):
        return False
    if _NUMERIC_CELL_RE.fullmatch(clean):
        return True
    compact = (
        clean.replace("$", "")
        .replace("€", "")
        .replace("£", "")
        .replace("¥", "")
        .replace(",", "")
        .replace(" ", "")
    )
    if compact.startswith("(") and compact.endswith(")%"):
        compact = compact[1:-2] + "%"
    elif compact.startswith("(") and compact.endswith(")"):
        compact = compact[1:-1]
    elif compact.startswith("-(") and compact.endswith(")"):
        compact = compact[2:-1]
    return bool(_NUMBER_RE.fullmatch(compact.strip("%")))


def _extract_number(text: str) -> Optional[float]:
    clean = _normalize_space(text)
    if not clean:
        return None
    if _is_year(clean) or not _looks_numeric_cell(clean):
        return None
    stripped = clean.replace("$", "").replace("€", "").replace("£", "")
    stripped = stripped.replace("¥", "").replace(",", "").replace(" ", "")
    negative = False
    if stripped.startswith("(") and stripped.endswith(")%"):
        negative = True
        stripped = stripped[1:-2] + "%"
    elif stripped.startswith("(") and stripped.endswith(")"):
        negative = True
        stripped = stripped[1:-1]
    elif stripped.startswith("-(") and stripped.endswith(")"):
        negative = True
        stripped = stripped[2:-1]
    match = _NUMBER_RE.fullmatch(stripped.strip("%"))
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
        # Require the scale word to appear in an explicit unit annotation context,
        # not as a bare word anywhere in cell content (e.g. "5 billion customers").
        if (
            f"in {label}" in haystack
            or f"in {label}s" in haystack
            or f"({label}" in haystack
            or f"({label}s" in haystack
            or f"{label}s)" in haystack
            or f"{label})" in haystack
        ):
            return label, factor
    return "", 1.0


def _infer_numeric_kind(
    text: str,
    row_header: str = "",
    col_header: str = "",
    section_name: str = "",
    has_scale: bool = False,
) -> str:
    clean = _normalize_space(text)
    normalized_row_header = _normalize_space(row_header).casefold()
    context = " ".join(
        _normalize_space(part).casefold()
        for part in (row_header, col_header, section_name)
        if part
    )
    compact_context = context.replace(" ", "")
    compact_row_header = normalized_row_header.replace(" ", "")

    # Explicit PERCENT signals: % symbol, or unambiguous percentage context words.
    # Removed broad terms like "change", "growth", "translation", "divestitures"
    # which frequently appear on dollar-amount rows too.
    if "%" in clean or "rate (annualized)" in col_header.lower() or any((hint != "margin" or not has_scale) and (hint in context or hint.replace(" ", "") in compact_context) for hint in _PERCENT_CONTEXT_HINTS):
        return "PERCENT"
    if any(hint in normalized_row_header or hint.replace(" ", "") in compact_row_header for hint in _ROW_PERCENT_HINTS):
        return "PERCENT"

    # Share-count columns/groups are not per-share currency amounts.
    if (re.search(r"\bshares\b", normalized_row_header) and not re.search(r"\bper shares?\b", normalized_row_header)) or "shares used in computing" in normalized_row_header or re.search(r"\bshares\b", col_header.lower()):
        return "SHARES"

    # Explicit PLAIN signals: per-share or ratio metrics that should never be scaled.
    if any(token in context for token in ("per share", "eps", "diluted share", "basic share")):
        return "PLAIN"

    # Explicit currency symbol → always AMOUNT.
    if any(token in clean for token in ("$", "€", "£", "¥")):
        return "AMOUNT"

    # Known financial statement line items → AMOUNT.
    if any(token in context for token in ("sales", "income", "taxes", "assets", "debt", "cash", "revenue", "expense")):
        return "AMOUNT"

    # If the table carries a scale (millions/thousands/billions) and the cell is not
    # clearly PERCENT or PLAIN, treat it as a scaled monetary amount rather than a
    # bare number, to avoid silently dropping the scale factor.
    if has_scale:
        return "AMOUNT"

    return "PLAIN"


def _normalize_percent_value(text: str, value: float) -> float:
    clean = _normalize_space(text)
    if "%" in clean:
        return value
    return value / 100.0


def _grid_from_docling(data: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Expand spans for header lookup; each numeric anchor is emitted once."""
    n_rows, n_cols = int(data["num_rows"]), int(data["num_cols"])
    if n_rows < 0 or n_cols < 0 or n_rows * n_cols > 1_000_000:
        raise ValueError("Invalid or excessively large table dimensions")
    grid = [[{
        "row_idx": r, "col_idx": c, "text": "", "is_header": False,
        "rowspan": 1, "colspan": 1, "is_span_ref": False, "native_structure": True,
    } for c in range(n_cols)] for r in range(n_rows)]
    occupied: set[tuple[int, int]] = set()
    for cell in data.get("table_cells", []):
        r0, c0 = int(cell["start_row_offset_idx"]), int(cell["start_col_offset_idx"])
        r1, c1 = int(cell["end_row_offset_idx"]), int(cell["end_col_offset_idx"])
        if not (0 <= r0 < r1 <= n_rows and 0 <= c0 < c1 <= n_cols):
            raise ValueError(f"Invalid table cell offsets: {(r0, r1, c0, c1)}")
        if int(cell.get("row_span", r1-r0)) != r1-r0 or int(cell.get("col_span", c1-c0)) != c1-c0:
            raise ValueError("Cell offsets and span disagree")
        for r in range(r0, r1):
            for c in range(c0, c1):
                if (r, c) in occupied:
                    raise ValueError(f"Overlapping table cells at {(r, c)}")
                occupied.add((r, c))
                grid[r][c] = {
                    "row_idx": r, "col_idx": c, "text": _normalize_space(str(cell.get("text", ""))),
                    "is_header": bool(cell.get("column_header") or cell.get("row_header") or cell.get("row_section")),
                    "column_header": bool(cell.get("column_header")),
                    "docling_row_header": bool(cell.get("row_header")), "row_section": bool(cell.get("row_section")),
                    "rowspan": r1-r0, "colspan": c1-c0, "is_span_ref": (r,c) != (r0,c0),
                    "anchor_row_idx": r0, "anchor_col_idx": c0, "native_structure": True,
                    "source_refs": list(cell.get("source_refs") or []),
                    "provenance": deepcopy(cell.get("provenance") or []),
                    "provenance_scope": cell.get("provenance_scope", "unknown"),
                    "raw_bbox": cell.get("bbox"),
                }
    return grid


def _classify_grid(
    grid: list[list[dict[str, Any]]],
    *,
    title_text: str,
    section_name: str,
    chunk_text: str,
) -> list[list[dict[str, Any]]]:
    if not grid:
        return grid

    # Only use the table title and section name to infer scale — not cell content,
    # which may contain incidental occurrences of "billion"/"million" in non-unit contexts.
    scale_label, scale_factor = _infer_scale([title_text, section_name, chunk_text])

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
            elif cell.get("native_structure") and cell.get("column_header"):
                cell_type = "YEAR" if _is_year(text) else "COL_HEADER"
            elif cell.get("native_structure") and (cell.get("docling_row_header") or cell.get("row_section")):
                cell_type = "ROW_HEADER"
            elif _is_year(text) and not cell.get("native_structure"):
                cell_type = "YEAR"
            elif cell["is_header"] and (cell["col_idx"] > 0 or _is_period_header_text(text)):
                cell_type = "COL_HEADER"
            elif _is_period_header_text(text):
                cell_type = "COL_HEADER"
            else:
                number = _extract_number(text)
                if number is not None:
                    cell_type = "NUMBER"
                    numeric_value = number
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
            cell["normalized_value"] = normalized_value
            cell["scale"] = ""
            cell["numeric_kind"] = ""

    for row in grid:
        for cell in row:
            if cell["cell_type"] != "NUMBER":
                continue
            row_header = _find_left_header(grid, cell["row_idx"], cell["col_idx"])
            col_header = _find_top_header(grid, cell["row_idx"], cell["col_idx"])
            numeric_kind = _infer_numeric_kind(
                cell["text"],
                row_header=row_header,
                col_header=col_header,
                section_name=section_name,
                has_scale=scale_factor != 1.0,
            )
            normalized_value = cell["numeric_value"]
            applied_scale = ""
            if numeric_kind == "AMOUNT" and scale_factor != 1.0:
                normalized_value = cell["numeric_value"] * scale_factor
                applied_scale = scale_label
            elif numeric_kind == "SHARES":
                annotation = " ".join([title_text, section_name])
                share_unit = re.search(r"(?:number of shares|shares)[^.;]{0,65}?in (thousands|millions|billions)", annotation, re.I)
                share_label, share_factor = _infer_scale(["in " + share_unit.group(1)]) if share_unit else (scale_label, scale_factor)
                normalized_value = cell["numeric_value"] * share_factor
                applied_scale = share_label
            elif numeric_kind == "PERCENT":
                cell["numeric_value"] = _normalize_percent_value(cell["text"], cell["numeric_value"])
                normalized_value = cell["numeric_value"]
                applied_scale = "percent"
            cell["row_header"] = row_header
            cell["col_header"] = col_header
            cell["numeric_kind"] = numeric_kind
            cell["scale"] = applied_scale
            cell["normalized_value"] = normalized_value

    return grid


def _find_left_header(grid: list[list[dict[str, Any]]], row_idx: int, col_idx: int) -> str:
    row = grid[row_idx]
    for idx in range(col_idx - 1, -1, -1):
        cell = row[idx]
        if cell["cell_type"] in {"ROW_HEADER", "TEXT"} and not _is_empty_text(cell["text"]):
            header = cell["text"]
            if cell.get("native_structure") and header.strip().lower() in {"basic", "diluted"}:
                for above in reversed(grid[:row_idx]):
                    # A row-group label with otherwise empty cells scopes Basic/Diluted.
                    labels = [c["text"] for c in above if c["cell_type"] == "ROW_HEADER" and not c.get("is_span_ref")]
                    if labels and all(c["cell_type"] in {"ROW_HEADER", "EMPTY"} for c in above):
                        header = " | ".join(labels + [header])
                        break
            return header
    return ""


def _find_top_header(grid: list[list[dict[str, Any]]], row_idx: int, col_idx: int) -> str:
    parts: list[str] = []
    for idx in range(row_idx - 1, -1, -1):
        if col_idx >= len(grid[idx]):
            continue
        cell = grid[idx][col_idx]
        if cell["cell_type"] in {"COL_HEADER", "YEAR"} and not _is_empty_text(cell["text"]):
            text = cell["text"]
            if text not in parts:
                parts.append(text)
    return " | ".join(reversed(parts))


def _build_table_structured(
    *,
    table_data: dict[str, Any],
    table_context: list[dict[str, Any]] | None = None,
    title_text: str,
    section_name: str,
    chunk_text: str,
) -> dict[str, Any]:
    if not isinstance(table_data, dict):
        raise ValueError("Native Docling table_data is required; reparse legacy inputs as JSON.")
    grid = _grid_from_docling(table_data)
    # Marker-matched notes are retrieval evidence, not table-wide unit overrides.
    context_text = " ".join(
        str(c.get("text", "")) for c in (table_context or [])
        if c.get("kind") != "matched_footnote"
    )
    grid = _classify_grid(
        grid,
        title_text=title_text + " " + context_text,
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
            for key in ("source_refs", "provenance", "provenance_scope", "raw_bbox", "anchor_row_idx", "anchor_col_idx", "column_header", "docling_row_header", "row_section"):
                if key in cell:
                    record[key] = cell[key]
            if cell["numeric_value"] is not None:
                record["numeric_value"] = cell["numeric_value"]
                record["numeric_kind"] = cell["numeric_kind"]
                record["scale"] = cell["scale"]
                record["normalized_value"] = cell["normalized_value"]
                record["row_header"] = cell.get("row_header", "")
                record["col_header"] = cell.get("col_header", "")
                if not cell["is_span_ref"]:
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
        help="Fail fast if one structured JSON table cannot be parsed.",
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

            is_table = bool(metadata.get("is_table")) or str(row.get("chunk_engine") or "") == "elements_table"
            if is_table:
                try:
                    metadata["table_structured"] = _build_table_structured(
                        table_data=metadata.get("table_data"),
                        table_context=metadata.get("table_context"),
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
