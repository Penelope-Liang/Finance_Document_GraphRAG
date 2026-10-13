from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


from neo4j_graphrag.preprocess.provenance import graph_source_properties


def _import_neo4j():
    try:
        import neo4j
    except ImportError as exc:
        raise ImportError(
            "Could not import neo4j Python driver. Install with `uv pip install neo4j`."
        ) from exc
    return neo4j


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"Missing env var: {name}", file=sys.stderr)
        sys.exit(1)
    return value


@dataclass
class TableChunkRow:
    chunk_id: str
    doc_id: str
    chunk_index: int
    section_name: str
    page: int | None
    parser_engine: str
    table_structured: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def _load_ndjson(path: Path, *, strict: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                if strict:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
                print(f"[warn] skip invalid JSON {path}:{line_no}: {exc}", file=sys.stderr)
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _read_tables(path: Path, *, strict: bool) -> dict[str, list[TableChunkRow]]:
    rows = _load_ndjson(path, strict=strict)
    by_doc: dict[str, list[TableChunkRow]] = defaultdict(list)
    for idx, payload in enumerate(rows, start=1):
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            continue
        table_structured = metadata.get("table_structured")
        if not isinstance(table_structured, dict):
            continue
        chunk_id = payload.get("chunk_id")
        doc_id = payload.get("doc_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            if strict:
                raise ValueError(f"row {idx}: missing chunk_id")
            continue
        if not isinstance(doc_id, str) or not doc_id:
            if strict:
                raise ValueError(f"row {idx}: missing doc_id")
            continue
        try:
            chunk_index = int(payload.get("chunk_index", 0))
        except (TypeError, ValueError):
            chunk_index = 0
        page = payload.get("page")
        if page is None:
            page = metadata.get("page_start")
        by_doc[doc_id].append(
            TableChunkRow(
                chunk_id=chunk_id,
                doc_id=doc_id,
                chunk_index=chunk_index,
                section_name=str(payload.get("section_name") or "UNSPECIFIED"),
                page=int(page) if isinstance(page, (int, float)) else None,
                parser_engine=str(payload.get("parser_engine") or ""),
                table_structured=table_structured,
                metadata=metadata,
            )
        )
    for doc_id, doc_rows in by_doc.items():
        doc_rows.sort(key=lambda row: (row.chunk_index, row.chunk_id))
    return by_doc


def _build_table_payload(row: TableChunkRow, *, cell_mode: str) -> dict[str, Any]:
    table_id = row.chunk_id
    captions = [c for c in (row.metadata.get("table_context") or [])
                if c.get("kind") == "captions" and str(c.get("text") or "").strip()]
    titles = {" ".join(str(c["text"]).split()) for c in captions}
    # A chapter heading or ambiguous set of captions is not a confirmed table title.
    title = next(iter(titles)) if len(titles) == 1 else None
    title_refs = list(dict.fromkeys(c["source_ref"] for c in captions
                                   if title and c.get("source_ref")))
    cells = row.table_structured.get("numeric_cells") if cell_mode == "numeric" else row.table_structured.get("cells")
    if not isinstance(cells, list):
        cells = []
    cell_rows: list[dict[str, Any]] = []
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("is_span_ref"):
            continue
        cell_id = f"{table_id}:{cell.get('row_idx')}:{cell.get('col_idx')}"
        payload = {
            "cell_id": cell_id,
            "table_id": table_id,
            "row_idx": cell.get("row_idx"),
            "col_idx": cell.get("col_idx"),
            "text": cell.get("text"),
            "cell_type": cell.get("cell_type"),
            "row_header": cell.get("row_header", ""),
            "col_header": cell.get("col_header", ""),
            "is_header": bool(cell.get("is_header", False)),
            "is_span_ref": bool(cell.get("is_span_ref", False)),
            "rowspan": int(cell.get("rowspan", 1) or 1),
            "colspan": int(cell.get("colspan", 1) or 1),
        }
        if "numeric_value" in cell:
            payload["numeric_value"] = cell.get("numeric_value")
            payload["numeric_kind"] = cell.get("numeric_kind")
            payload["scale"] = cell.get("scale")
            payload["normalized_value"] = cell.get("normalized_value")
        sources = dict(row.metadata.get("table_sources") or row.metadata)
        for key in ("source_refs", "provenance"):
            if key in cell:
                sources[key] = cell[key]
        sources["pages"] = sorted({p["page_no"] for p in sources.get("provenance", []) if isinstance(p.get("page_no"), int)}) or sources.get("pages", [])
        payload["source_properties"] = graph_source_properties(sources)
        payload["source_properties"]["provenance_scope"] = cell.get("provenance_scope", "table")
        payload["source_properties"]["doc_id"] = row.doc_id
        payload["source_properties"]["raw_bbox_json"] = json.dumps(cell.get("raw_bbox"))
        cell_rows.append(payload)
    return {
        "table_id": table_id,
        "chunk_id": row.chunk_id,
        "doc_id": row.doc_id,
        "chunk_index": row.chunk_index,
        "section_name": row.section_name,
        "page": row.page,
        "parser_engine": row.parser_engine,
        "n_rows": row.table_structured.get("n_rows"),
        "n_cols": row.table_structured.get("n_cols"),
        "cell_rows": cell_rows,
        "source_properties": {
            **graph_source_properties(row.metadata.get("table_sources") or row.metadata),
            "title": title,
            "title_source": "docling_caption" if title else None,
            "title_source_refs": title_refs,
            "table_context_json": json.dumps(row.metadata.get("table_context") or [], ensure_ascii=False),
            "possible_continuation": bool(row.metadata.get("possible_continuation")),
            "continuation_reason": row.metadata.get("continuation_reason"),
        },
    }


def _create_schema(driver: Any, *, database: str | None) -> None:
    statements = [
        "CREATE CONSTRAINT table_id_unique IF NOT EXISTS FOR (t:Table) REQUIRE t.table_id IS UNIQUE",
        "CREATE CONSTRAINT table_cell_id_unique IF NOT EXISTS FOR (c:TableCell) REQUIRE c.cell_id IS UNIQUE",
        "CREATE INDEX table_chunk_id_idx IF NOT EXISTS FOR (t:Table) ON (t.chunk_id)",
        "CREATE INDEX table_cell_numeric_kind_idx IF NOT EXISTS FOR (c:TableCell) ON (c.numeric_kind)",
        "CREATE INDEX table_cell_headers_idx IF NOT EXISTS FOR (c:TableCell) ON (c.row_header, c.col_header)",
    ]
    for stmt in statements:
        driver.execute_query(stmt, database_=database)


def _purge_doc_tables_tx(tx: Any, *, doc_id: str) -> None:
    tx.run(
        """
        MATCH (c:Chunk {doc_id: $doc_id})-[:HAS_TABLE]->(t:Table)
        OPTIONAL MATCH (t)-[:HAS_CELL]->(cell:TableCell)
        WITH collect(DISTINCT cell) AS cells, collect(DISTINCT t) AS tables
        FOREACH (cell IN cells | DETACH DELETE cell)
        FOREACH (table IN tables | DETACH DELETE table)
        """,
        doc_id=doc_id,
    ).consume()


def _import_doc_tables_tx(tx: Any, *, rows: list[dict[str, Any]], purge_doc_id: str | None = None) -> None:
    if purge_doc_id is not None:
        _purge_doc_tables_tx(tx, doc_id=purge_doc_id)
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (chunk:Chunk {chunk_id: row.chunk_id})
        MERGE (table:Table {table_id: row.table_id})
        SET table.chunk_id = row.chunk_id,
            table.doc_id = row.doc_id,
            table.chunk_index = row.chunk_index,
            table.section_name = row.section_name,
            table.page = row.page,
            table.parser_engine = row.parser_engine,
            table.n_rows = row.n_rows,
            table.n_cols = row.n_cols
        SET table += row.source_properties
        MERGE (chunk)-[:HAS_TABLE]->(table)
        WITH row, table
        UNWIND row.cell_rows AS cell_row
        MERGE (cell:TableCell {cell_id: cell_row.cell_id})
        SET cell.table_id = cell_row.table_id,
            cell.row_idx = cell_row.row_idx,
            cell.col_idx = cell_row.col_idx,
            cell.text = cell_row.text,
            cell.cell_type = cell_row.cell_type,
            cell.row_header = cell_row.row_header,
            cell.col_header = cell_row.col_header,
            cell.is_header = cell_row.is_header,
            cell.is_span_ref = cell_row.is_span_ref,
            cell.rowspan = cell_row.rowspan,
            cell.colspan = cell_row.colspan,
            cell.numeric_value = cell_row.numeric_value,
            cell.numeric_kind = cell_row.numeric_kind,
            cell.scale = cell_row.scale,
            cell.normalized_value = cell_row.normalized_value
        SET cell += cell_row.source_properties
        MERGE (table)-[:HAS_CELL]->(cell)
        """,
        rows=rows,
    ).consume()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import table_structured rows into Neo4j alongside existing Chunk nodes."
    )
    parser.add_argument(
        "--chunks-path",
        required=True,
        help="Input chunks_table_structured.ndjson path.",
    )
    parser.add_argument(
        "--cell-mode",
        choices=["numeric", "all"],
        default="numeric",
        help="Import only numeric cells or all parsed cells.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on malformed rows.",
    )
    parser.add_argument(
        "--purge-doc-first",
        action="store_true",
        help="Delete existing Table/TableCell subgraph for each doc before import.",
    )
    parser.add_argument(
        "--neo4j-uri",
        default=os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"),
        help="Neo4j URI.",
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j username.",
    )
    parser.add_argument(
        "--neo4j-database",
        default=os.environ.get("NEO4J_DATABASE"),
        help="Neo4j database name (optional).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    chunks_path = Path(args.chunks_path).expanduser().resolve()
    if not chunks_path.exists():
        print(f"Input file not found: {chunks_path}", file=sys.stderr)
        sys.exit(1)

    by_doc = _read_tables(chunks_path, strict=args.strict)
    if not by_doc:
        print("No table_structured rows found.", file=sys.stderr)
        sys.exit(1)

    password = _require_env("NEO4J_PASSWORD")
    neo4j = _import_neo4j()

    imported_docs = 0
    imported_tables = 0
    imported_cells = 0
    with neo4j.GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, password)) as driver:
        _create_schema(driver, database=args.neo4j_database)
        with driver.session(database=args.neo4j_database) as session:
            for doc_id, doc_rows in sorted(by_doc.items()):
                payload_rows = [
                    _build_table_payload(row, cell_mode=args.cell_mode)
                    for row in doc_rows
                ]
                session.execute_write(_import_doc_tables_tx, rows=payload_rows, purge_doc_id=doc_id if args.purge_doc_first else None)
                imported_docs += 1
                imported_tables += len(payload_rows)
                imported_cells += sum(len(row["cell_rows"]) for row in payload_rows)
                print(
                    f"Imported doc={doc_id} tables={len(payload_rows)} cells={sum(len(row['cell_rows']) for row in payload_rows)}"
                )

    print("\nDone.")
    print(f"Imported docs:   {imported_docs}")
    print(f"Imported tables: {imported_tables}")
    print(f"Imported cells:  {imported_cells}")


if __name__ == "__main__":
    main()
