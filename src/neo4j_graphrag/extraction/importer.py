from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _import_neo4j():
    try:
        import neo4j
    except ImportError as exc:
        raise ImportError(
            "Could not import neo4j Python driver. Install with `pip install neo4j`."
        ) from exc
    return neo4j


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"Missing env var: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def _load_ndjson(path: Path, *, strict: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
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
            if not isinstance(payload, dict):
                if strict:
                    raise ValueError(f"Row is not an object at {path}:{line_no}")
                continue
            rows.append(payload)
    return rows


def _sanitize_entities(rows: list[dict[str, Any]], *, strict: bool) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        entity_key = row.get("entity_key")
        title = row.get("title")
        entity_type = row.get("type")
        if not isinstance(entity_key, str) or not entity_key:
            if strict:
                raise ValueError(f"entities row {idx}: missing entity_key")
            continue
        if not isinstance(title, str) or not title:
            if strict:
                raise ValueError(f"entities row {idx}: missing title")
            continue
        text_unit_ids = row.get("text_unit_ids")
        if not isinstance(text_unit_ids, list):
            text_unit_ids = []
        clean.append(
            {
                "entity_key": entity_key,
                "title": title,
                "type": entity_type if isinstance(entity_type, str) else "UNKNOWN",
                "description": str(row.get("description") or ""),
                "frequency": int(row.get("frequency") or len(text_unit_ids)),
                "text_unit_ids": [str(x) for x in text_unit_ids if str(x)],
                "doc_ids": [str(x) for x in (row.get("doc_ids") or []) if str(x)],
                "source_method": str(row.get("source_method") or "llm"),
            }
        )
    return clean


def _sanitize_relationships(
    rows: list[dict[str, Any]], *, strict: bool
) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        source_key = row.get("source_key")
        target_key = row.get("target_key")
        if not isinstance(source_key, str) or not source_key:
            if strict:
                raise ValueError(f"relationships row {idx}: missing source_key")
            continue
        if not isinstance(target_key, str) or not target_key:
            if strict:
                raise ValueError(f"relationships row {idx}: missing target_key")
            continue
        if source_key == target_key:
            continue
        try:
            weight = float(row.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        text_unit_ids = row.get("text_unit_ids")
        if not isinstance(text_unit_ids, list):
            text_unit_ids = []
        clean.append(
            {
                "source": str(row.get("source") or ""),
                "target": str(row.get("target") or ""),
                "source_key": source_key,
                "target_key": target_key,
                "description": str(row.get("description") or ""),
                "weight": weight,
                "text_unit_ids": [str(x) for x in text_unit_ids if str(x)],
                "doc_ids": [str(x) for x in (row.get("doc_ids") or []) if str(x)],
                "source_method": str(row.get("source_method") or "llm"),
            }
        )
    return clean


def _create_schema(
    driver: Any,
    *,
    database: str | None,
) -> None:
    statements = [
        "CREATE CONSTRAINT entity_key_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_key IS UNIQUE",
        "CREATE INDEX entity_title_idx IF NOT EXISTS FOR (e:Entity) ON (e.title)",
        "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    ]
    for stmt in statements:
        driver.execute_query(stmt, database_=database)


def _chunks(lst: list[dict[str, Any]], size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _import_entity_batch(tx: Any, rows: list[dict[str, Any]]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (e:Entity {entity_key: row.entity_key})
        SET e.title = row.title,
            e.type = row.type,
            e.description = row.description,
            e.frequency = row.frequency,
            e.text_unit_ids = row.text_unit_ids,
            e.doc_ids = row.doc_ids,
            e.source_method = row.source_method
        WITH e, row
        UNWIND coalesce(row.text_unit_ids, []) AS cid
        OPTIONAL MATCH (c:Chunk {chunk_id: cid})
        FOREACH (_ IN CASE WHEN c IS NULL THEN [] ELSE [1] END |
            MERGE (e)-[:FROM_CHUNK]->(c)
        )
        """,
        rows=rows,
    ).consume()


def _import_relationship_batch(tx: Any, rows: list[dict[str, Any]]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (s:Entity {entity_key: row.source_key})
        MATCH (t:Entity {entity_key: row.target_key})
        MERGE (s)-[r:RELATED]->(t)
        SET r.source = row.source,
            r.target = row.target,
            r.weight = row.weight,
            r.description = row.description,
            r.text_unit_ids = row.text_unit_ids,
            r.doc_ids = row.doc_ids,
            r.source_method = row.source_method
        """,
        rows=rows,
    ).consume()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import extracted entities/relationships into Neo4j and link them to Chunk nodes."
    )
    parser.add_argument(
        "--entities-path",
        default="output/entities/entities.ndjson",
        help="Merged entities NDJSON path.",
    )
    parser.add_argument(
        "--relationships-path",
        default="output/entities/relationships.ndjson",
        help="Merged relationships NDJSON path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Batch size for Neo4j writes.",
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
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on malformed rows.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    entities_path = Path(args.entities_path).expanduser().resolve()
    relationships_path = Path(args.relationships_path).expanduser().resolve()

    if not entities_path.exists():
        print(f"Entities file not found: {entities_path}", file=sys.stderr)
        sys.exit(1)
    if not relationships_path.exists():
        print(f"Relationships file not found: {relationships_path}", file=sys.stderr)
        sys.exit(1)

    raw_entities = _load_ndjson(entities_path, strict=args.strict)
    raw_relationships = _load_ndjson(relationships_path, strict=args.strict)
    entities = _sanitize_entities(raw_entities, strict=args.strict)
    relationships = _sanitize_relationships(raw_relationships, strict=args.strict)

    neo4j = _import_neo4j()
    password = _require_env("NEO4J_PASSWORD")
    with neo4j.GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, password)) as driver:
        _create_schema(driver, database=args.neo4j_database)
        imported_entities = 0
        imported_relationships = 0

        with driver.session(database=args.neo4j_database) as session:
            for batch in _chunks(entities, max(1, args.batch_size)):
                session.execute_write(_import_entity_batch, batch)
                imported_entities += len(batch)

            for batch in _chunks(relationships, max(1, args.batch_size)):
                session.execute_write(_import_relationship_batch, batch)
                imported_relationships += len(batch)

    print("\nDone.")
    print(f"Entities imported:      {imported_entities}")
    print(f"Relationships imported: {imported_relationships}")
    print(f"Entities file:          {entities_path}")
    print(f"Relationships file:     {relationships_path}")


if __name__ == "__main__":
    main()
