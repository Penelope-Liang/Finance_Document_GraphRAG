from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neo4j_graphrag.preprocess.provenance import graph_source_properties

from neo4j_graphrag.preprocess.statement_type import classify_statement_type

@dataclass
class ChunkRow:
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    section_name: str
    page: int | None
    title_text: str
    is_table: bool
    should_index: bool
    statement_type: str
    source_properties: dict[str, Any] = field(default_factory=dict)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"Missing env var: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def _import_neo4j():
    try:
        import neo4j
    except ImportError as exc:
        raise ImportError(
            "Could not import neo4j Python driver. Install with `uv pip install neo4j`."
        ) from exc
    return neo4j


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"updated_at_utc": _now_utc(), "docs": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"updated_at_utc": _now_utc(), "docs": {}}
    if not isinstance(payload, dict):
        return {"updated_at_utc": _now_utc(), "docs": {}}
    payload.setdefault("docs", {})
    return payload


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = _now_utc()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _doc_stem_to_company_id(doc_id: str) -> str:
    stem = Path(doc_id).stem.strip().upper()
    return stem or "UNKNOWN"


def _form_id(company_id: str | None, doc_id: str) -> str:
    if company_id:
        return f"{company_id}|{doc_id}"
    return f"UNK|{doc_id}"


def _read_chunks(path: Path, *, strict: bool) -> dict[str, list[ChunkRow]]:
    by_doc: dict[str, list[ChunkRow]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                if strict:
                    raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
                print(f"[warn] skip invalid JSON line {line_no}: {exc}", file=sys.stderr)
                continue

            chunk_id = payload.get("chunk_id")
            doc_id = payload.get("doc_id")
            text = payload.get("text")
            chunk_index = payload.get("chunk_index", 0)
            section_name = payload.get("section_name") or "UNSPECIFIED"
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            page = payload.get("page")
            if page is None:
                page = metadata.get("page_start")
            title_text = str(metadata.get("title_text") or "")
            is_table = bool(metadata.get("is_table"))
            should_index = bool(metadata.get("should_index", True))
            statement_type = str(metadata.get("statement_type") or "").strip() or classify_statement_type(
                section_name=str(section_name),
                title_text=title_text,
                text=str(text or ""),
                metadata=metadata,
            )

            if not isinstance(chunk_id, str) or not chunk_id:
                if strict:
                    raise ValueError(f"Missing/invalid chunk_id at line {line_no}")
                print(f"[warn] skip line {line_no}: invalid chunk_id", file=sys.stderr)
                continue
            if not isinstance(doc_id, str) or not doc_id:
                if strict:
                    raise ValueError(f"Missing/invalid doc_id at line {line_no}")
                print(f"[warn] skip line {line_no}: invalid doc_id", file=sys.stderr)
                continue
            if not isinstance(text, str) or not text:
                if strict:
                    raise ValueError(f"Missing/invalid text at line {line_no}")
                print(f"[warn] skip line {line_no}: invalid text", file=sys.stderr)
                continue
            try:
                chunk_index_int = int(chunk_index)
            except (TypeError, ValueError):
                if strict:
                    raise ValueError(f"Invalid chunk_index at line {line_no}: {chunk_index}")
                chunk_index_int = 0

            by_doc[doc_id].append(
                ChunkRow(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=text,
                    chunk_index=chunk_index_int,
                    section_name=str(section_name).strip() or "UNSPECIFIED",
                    page=int(page) if isinstance(page, (int, float)) else None,
                    title_text=title_text.strip(),
                    is_table=is_table,
                    should_index=should_index,
                    statement_type=statement_type,
                    source_properties=graph_source_properties(metadata),
                )
            )

    for doc_id, rows in by_doc.items():
        rows.sort(key=lambda r: (r.chunk_index, r.chunk_id))
    return by_doc


def _doc_fingerprint(rows: list[ChunkRow]) -> str:
    from dataclasses import asdict
    return hashlib.sha256(json.dumps([asdict(r) for r in rows], sort_keys=True).encode()).hexdigest()


def _create_constraints_and_indexes(
    driver: Any,
    *,
    database: str | None,
    create_fulltext_index: bool,
    fulltext_index_name: str,
    with_sections: bool,
) -> None:
    statements = [
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT form_id_unique IF NOT EXISTS FOR (f:Form) REQUIRE f.form_id IS UNIQUE",
        "CREATE CONSTRAINT company_id_unique IF NOT EXISTS FOR (c:Company) REQUIRE c.company_id IS UNIQUE",
        "CREATE CONSTRAINT statement_type_name_unique IF NOT EXISTS FOR (st:StatementType) REQUIRE st.name IS UNIQUE",
        "CREATE INDEX chunk_statement_type IF NOT EXISTS FOR (c:Chunk) ON (c.statement_type)",
    ]
    if with_sections:
        statements.append(
            "CREATE CONSTRAINT section_key_unique IF NOT EXISTS FOR (s:Section) REQUIRE (s.form_id, s.name) IS UNIQUE"
        )
    if create_fulltext_index:
        statements.append(
            f"CREATE FULLTEXT INDEX {fulltext_index_name} IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]"
        )

    for stmt in statements:
        driver.execute_query(
            stmt,
            database_=database,
        )


def _import_doc_tx(
    tx: Any,
    *,
    doc_id: str,
    form_id: str,
    company_id: str | None,
    rows_payload: list[dict[str, Any]],
    with_sections: bool,
    purge_doc_first: bool,
) -> None:
    if purge_doc_first:
        # Remove table descendants before deleting their parent chunks, including
        # tables left behind by older importers. Everything is one transaction.
        tx.run(
            """
            MATCH (t:Table {doc_id: $doc_id})
            OPTIONAL MATCH (t)-[:HAS_CELL]->(cell:TableCell)
            WITH collect(DISTINCT cell) AS cells, collect(DISTINCT t) AS tables
            FOREACH (cell IN cells | DETACH DELETE cell)
            FOREACH (table IN tables | DETACH DELETE table)
            """, doc_id=doc_id,
        ).consume()
        tx.run(
            """
            OPTIONAL MATCH (f:Form {form_id: $form_id})
            WITH f WHERE f IS NOT NULL
            OPTIONAL MATCH (f)<-[:IN_FORM]-(c:Chunk)
            WITH [x IN collect(DISTINCT c) WHERE x IS NOT NULL] AS chunks
            FOREACH (c IN chunks | DETACH DELETE c)
            """,
            form_id=form_id,
        ).consume()
        tx.run(
            """
            OPTIONAL MATCH (f:Form {form_id: $form_id})
            WITH f WHERE f IS NOT NULL
            OPTIONAL MATCH (f)-[:HAS_SECTION]->(s:Section)
            WITH [x IN collect(DISTINCT s) WHERE x IS NOT NULL] AS sections
            FOREACH (s IN sections | DETACH DELETE s)
            """,
            form_id=form_id,
        ).consume()

    if not purge_doc_first:
        stale = tx.run(
            "MATCH (c:Chunk {doc_id: $doc_id}) WHERE NOT c.chunk_id IN $ids RETURN count(c) AS count",
            doc_id=doc_id, ids=[row["chunk_id"] for row in rows_payload],
        ).single()
        if stale and stale["count"]:
            raise ValueError("Document chunk IDs changed; use --purge-doc-first to replace this document")

    if company_id:
        tx.run(
            """
            MERGE (co:Company {company_id: $company_id})
            MERGE (f:Form {form_id: $form_id})
            ON CREATE SET f.doc_id = $doc_id
            SET f.doc_id = $doc_id
            MERGE (co)-[:FILED]->(f)
            """,
            company_id=company_id,
            form_id=form_id,
            doc_id=doc_id,
        ).consume()
    else:
        tx.run(
            """
            MERGE (f:Form {form_id: $form_id})
            ON CREATE SET f.doc_id = $doc_id
            SET f.doc_id = $doc_id
            """,
            form_id=form_id,
            doc_id=doc_id,
        ).consume()

    if with_sections:
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (f:Form {form_id: $form_id})
            MERGE (s:Section {form_id: $form_id, name: row.section_name})
            MERGE (f)-[:HAS_SECTION]->(s)
            MERGE (c:Chunk {chunk_id: row.chunk_id})
            SET c.text = row.text,
                c.chunk_index = row.chunk_index,
                c.section_name = row.section_name,
                c.page = row.page,
                c.title_text = row.title_text,
                c.is_table = row.is_table,
                c.should_index = row.should_index,
                c.statement_type = row.statement_type,
                c.doc_id = $doc_id
            SET c += row.source_properties
            SET s.statement_type = coalesce(s.statement_type, row.statement_type)
            MERGE (c)-[:IN_SECTION]->(s)
            SET c += row.source_properties
            MERGE (c)-[:IN_FORM]->(f)
            MERGE (st:StatementType {name: row.statement_type})
            MERGE (c)-[:HAS_STATEMENT_TYPE]->(st)
            MERGE (s)-[:HAS_STATEMENT_TYPE]->(st)
            """,
            rows=rows_payload,
            form_id=form_id,
            doc_id=doc_id,
        ).consume()
    else:
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (f:Form {form_id: $form_id})
            MERGE (c:Chunk {chunk_id: row.chunk_id})
            SET c.text = row.text,
                c.chunk_index = row.chunk_index,
                c.section_name = row.section_name,
                c.page = row.page,
                c.title_text = row.title_text,
                c.is_table = row.is_table,
                c.should_index = row.should_index,
                c.statement_type = row.statement_type,
                c.doc_id = $doc_id
            SET c += row.source_properties
            MERGE (c)-[:IN_FORM]->(f)
            MERGE (st:StatementType {name: row.statement_type})
            MERGE (c)-[:HAS_STATEMENT_TYPE]->(st)
            """,
            rows=rows_payload,
            form_id=form_id,
            doc_id=doc_id,
        ).consume()

    tx.run(
        """
        UNWIND range(0, size($rows)-2) AS i
        WITH $rows[i] AS a, $rows[i+1] AS b
        MATCH (c1:Chunk {chunk_id: a.chunk_id})
        MATCH (c2:Chunk {chunk_id: b.chunk_id})
        MERGE (c1)-[:NEXT]->(c2)
        """,
        rows=rows_payload,
    ).consume()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import chunks.ndjson into Neo4j with resumable checkpoints "
            "(doc_id-level progress)."
        )
    )
    parser.add_argument(
        "--chunks-path",
        default="chunks.ndjson",
        help="Input chunks NDJSON generated during document preprocessing.",
    )
    parser.add_argument(
        "--state-path",
        default="output/graph_import/state.json",
        help="Checkpoint state file path.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Limit number of docs imported in this run (0 = all).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on malformed rows or per-doc import errors.",
    )
    parser.add_argument(
        "--company-mode",
        choices=["doc_stem", "none", "constant"],
        default="doc_stem",
        help="How to derive Company node id.",
    )
    parser.add_argument(
        "--company-id",
        default="",
        help="Used when --company-mode=constant.",
    )
    parser.add_argument(
        "--no-sections",
        action="store_true",
        help="Do not create Section nodes/relationships.",
    )
    parser.add_argument(
        "--no-fulltext-index",
        action="store_true",
        help="Do not create chunk fulltext index.",
    )
    parser.add_argument(
        "--purge-doc-first",
        action="store_true",
        help=(
            "Before importing each doc/form, delete its existing Chunk/Section subgraph "
            "to avoid stale nodes when re-ingesting the same doc with new chunking."
        ),
    )
    parser.add_argument(
        "--fulltext-index-name",
        default=os.environ.get("NEO4J_FULLTEXT_INDEX", "chunk_fulltext_index"),
        help="Chunk fulltext index name.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip docs already marked done in state file.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Ignore state done markers and re-import all docs.",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Start with an empty state file for this run.",
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

    state_path = Path(args.state_path).expanduser().resolve()
    state = {"updated_at_utc": _now_utc(), "docs": {}} if args.reset_state else _load_state(state_path)

    by_doc = _read_chunks(chunks_path, strict=args.strict)
    if not by_doc:
        print("No valid chunks to import.", file=sys.stderr)
        sys.exit(1)

    password = _require_env("NEO4J_PASSWORD")
    with_sections = not args.no_sections
    create_fulltext_index = not args.no_fulltext_index

    doc_ids = sorted(by_doc.keys())
    if args.resume:
        doc_ids = [d for d in doc_ids if state["docs"].get(d, {}).get("status") != "done" or state["docs"].get(d, {}).get("fingerprint") != _doc_fingerprint(by_doc[d])]
    if args.max_docs > 0:
        doc_ids = doc_ids[: args.max_docs]

    total_docs = len(doc_ids)
    if total_docs == 0:
        print("Nothing to import (all docs already done).")
        return

    print(f"Loaded docs: {len(by_doc)} | pending this run: {total_docs}")

    done = 0
    failed = 0
    neo4j = _import_neo4j()
    with neo4j.GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, password)) as driver:
        _create_constraints_and_indexes(
            driver,
            database=args.neo4j_database,
            create_fulltext_index=create_fulltext_index,
            fulltext_index_name=args.fulltext_index_name,
            with_sections=with_sections,
        )

        for idx, doc_id in enumerate(doc_ids, start=1):
            rows = by_doc[doc_id]
            if args.company_mode == "none":
                company_id = None
            elif args.company_mode == "constant":
                company_id = (args.company_id or "").strip().upper() or "UNKNOWN"
            else:
                company_id = _doc_stem_to_company_id(doc_id)
            form_id = _form_id(company_id, doc_id)

            rows_payload = [
                {
                    "chunk_id": r.chunk_id,
                    "text": r.text,
                    "chunk_index": r.chunk_index,
                    "section_name": r.section_name,
                    "page": r.page,
                    "title_text": r.title_text,
                    "is_table": r.is_table,
                    "should_index": r.should_index,
                    "statement_type": r.statement_type,
                    "source_properties": r.source_properties,
                }
                for r in rows
            ]

            print(f"[{idx}/{total_docs}] importing doc={doc_id} chunks={len(rows_payload)}")
            try:
                with driver.session(database=args.neo4j_database) as session:
                    session.execute_write(
                        _import_doc_tx,
                        doc_id=doc_id,
                        form_id=form_id,
                        company_id=company_id,
                        rows_payload=rows_payload,
                        with_sections=with_sections,
                        purge_doc_first=args.purge_doc_first,
                    )
                state["docs"][doc_id] = {
                    "status": "done",
                    "fingerprint": _doc_fingerprint(rows),
                    "updated_at_utc": _now_utc(),
                    "chunk_count": len(rows_payload),
                    "form_id": form_id,
                    "company_id": company_id,
                    "error": None,
                }
                _save_state(state_path, state)
                done += 1
            except Exception as exc:
                failed += 1
                error_text = str(exc)
                print(f"  [error] {error_text}", file=sys.stderr)
                state["docs"][doc_id] = {
                    "status": "failed",
                    "updated_at_utc": _now_utc(),
                    "chunk_count": len(rows_payload),
                    "form_id": form_id,
                    "company_id": company_id,
                    "error": error_text,
                }
                _save_state(state_path, state)
                if args.strict:
                    raise

    print("\nDone.")
    print(f"State: {state_path}")
    print(f"Imported docs: {done}")
    print(f"Failed docs:   {failed}")
    print(
        "Re-run the same command to resume: docs marked status=done will be skipped "
        "(unless --no-resume)."
    )


if __name__ == "__main__":
    main()
