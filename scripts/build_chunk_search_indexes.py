from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import neo4j
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.neo4j_indexes import create_fulltext_index, create_vector_index


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"Missing env var: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def _load_doc_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    doc_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row.get("doc_id")
            if doc_id:
                doc_ids.add(str(doc_id))
    return sorted(doc_ids)


def _fetch_batch(
    session: neo4j.Session,
    *,
    doc_ids: list[str] | None,
    embedding_property: str,
    limit: int,
) -> list[dict[str, Any]]:
    query = f"""
    MATCH (c:Chunk)
    WHERE c.`{embedding_property}` IS NULL
      AND ($doc_ids IS NULL OR c.doc_id IN $doc_ids)
      AND c.text IS NOT NULL
    RETURN c.chunk_id AS chunk_id,
           c.doc_id AS doc_id,
           c.section_name AS section_name,
           c.text AS text
    ORDER BY c.doc_id, c.chunk_index
    LIMIT $limit
    """
    rows = session.run(query, doc_ids=doc_ids, limit=limit)
    return [dict(row) for row in rows]


def _embedding_text(row: dict[str, Any]) -> str:
    parts = []
    if row.get("doc_id"):
        parts.append(f"Document: {row['doc_id']}")
    if row.get("section_name"):
        parts.append(f"Section: {row['section_name']}")
    parts.append(str(row.get("text") or ""))
    return "\n".join(parts)


def _write_batch(
    session: neo4j.Session,
    *,
    rows: list[dict[str, Any]],
    id_property: str,
    embedding_property: str,
) -> int:
    query = f"""
    UNWIND $rows AS row
    MATCH (c:Chunk {{`{id_property}`: row.chunk_id}})
    SET c.`{embedding_property}` = row.embedding
    RETURN count(c) AS updated
    """
    record = session.run(query, rows=rows).single()
    return int(record["updated"] or 0) if record else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate missing Chunk embeddings and create chunk vector/fulltext indexes."
    )
    parser.add_argument("--questions-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--embed-model", default=os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3:latest"))
    parser.add_argument("--vector-index", default=os.environ.get("NEO4J_VECTOR_INDEX", "chunk_embedding_index"))
    parser.add_argument("--fulltext-index", default=os.environ.get("NEO4J_FULLTEXT_INDEX", "chunk_fulltext_index"))
    parser.add_argument("--embedding-property", default="embedding")
    parser.add_argument("--id-property", default="chunk_id")
    parser.add_argument("--similarity-fn", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    uri = _require_env("NEO4J_URI")
    user = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME")
    if not user:
        raise SystemExit("Missing env var: NEO4J_USER or NEO4J_USERNAME")
    password = _require_env("NEO4J_PASSWORD")
    database = os.environ.get("NEO4J_DATABASE")
    ollama_host = os.environ.get("OLLAMA_HOST")
    ollama_kwargs = {"host": ollama_host} if ollama_host else {}

    doc_ids = _load_doc_ids(args.questions_path)
    embedder = OllamaEmbeddings(model=args.embed_model, **ollama_kwargs)

    started = time.perf_counter()
    processed = 0
    dimension = 0

    with neo4j.GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            while True:
                if args.limit and processed >= args.limit:
                    break
                batch_limit = args.batch_size
                if args.limit:
                    batch_limit = min(batch_limit, args.limit - processed)

                batch = _fetch_batch(
                    session,
                    doc_ids=doc_ids,
                    embedding_property=args.embedding_property,
                    limit=batch_limit,
                )
                if not batch:
                    break

                write_rows = []
                for row in batch:
                    vector = embedder.embed_query(_embedding_text(row))
                    if not dimension:
                        dimension = len(vector)
                    elif len(vector) != dimension:
                        raise ValueError(
                            f"Embedding dimension changed: expected {dimension}, got {len(vector)}"
                        )
                    write_rows.append(
                        {
                            "chunk_id": row["chunk_id"],
                            "embedding": [float(x) for x in vector],
                        }
                    )
                updated = _write_batch(
                    session,
                    rows=write_rows,
                    id_property=args.id_property,
                    embedding_property=args.embedding_property,
                )
                processed += updated
                if args.progress_every > 0 and processed % args.progress_every == 0:
                    elapsed = time.perf_counter() - started
                    print(f"  embedded {processed} chunks ({elapsed:.1f}s)")

        if not dimension:
            with driver.session(database=database) as session:
                record = session.run(
                    f"""
                    MATCH (c:Chunk)
                    WHERE c.`{args.embedding_property}` IS NOT NULL
                    RETURN size(c.`{args.embedding_property}`) AS dimension
                    LIMIT 1
                    """
                ).single()
                dimension = int(record["dimension"] or 0) if record else 0
        if not dimension:
            raise SystemExit("No embeddings found or generated; cannot create vector index.")

        create_vector_index(
            driver=driver,
            name=args.vector_index,
            label="Chunk",
            embedding_property=args.embedding_property,
            dimensions=dimension,
            similarity_fn=args.similarity_fn,
            fail_if_exists=False,
            neo4j_database=database,
        )
        create_fulltext_index(
            driver=driver,
            name=args.fulltext_index,
            label="Chunk",
            node_properties=["text", "section_name", "doc_id"],
            fail_if_exists=False,
            neo4j_database=database,
        )

    elapsed = time.perf_counter() - started
    print("Done.")
    print(f"Embedded chunks: {processed}")
    print(f"Embedding dim:   {dimension}")
    print(f"Vector index:    {args.vector_index}")
    print(f"Fulltext index:  {args.fulltext_index}")
    print(f"Elapsed seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
