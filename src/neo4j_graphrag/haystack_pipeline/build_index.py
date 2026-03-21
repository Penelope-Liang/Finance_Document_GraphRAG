from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from neo4j_graphrag.haystack_pipeline.io_utils import (
    ChunkRow,
    format_duration_minsec,
    read_chunks_ndjson,
)


def _import_haystack_document_and_embedder():
    try:
        from haystack import Document
        from haystack.components.embedders import SentenceTransformersDocumentEmbedder
    except ImportError as exc:
        raise ImportError(
            "Haystack components not found. Install with `pip install haystack-ai sentence-transformers`."
        ) from exc
    return Document, SentenceTransformersDocumentEmbedder


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a Haystack-ready local index (documents + embeddings) from chunks.ndjson."
    )
    parser.add_argument(
        "--chunks-path",
        required=True,
        help="Path to chunks.ndjson.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for Haystack index artifacts.",
    )
    parser.add_argument(
        "--embedder-model",
        default="BAAI/bge-m3",
        help="Sentence-transformers model used by Haystack document embedder.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N chunks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max chunks to index (0 means all).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on malformed rows.",
    )
    parser.add_argument(
        "--use-enriched-text",
        action="store_true",
        help="Append enrichment metadata (title/summary/keywords/table description/prefix) into embedding text.",
    )
    parser.add_argument(
        "--skip-non-indexable",
        action="store_true",
        help="Skip rows whose metadata.should_index is explicitly false.",
    )
    return parser


def _chunk_to_meta(row: ChunkRow) -> dict[str, Any]:
    return {
        "chunk_id": row.chunk_id,
        "doc_id": row.doc_id,
        "chunk_index": row.chunk_index,
        "section_name": row.section_name,
        "page": row.page,
        "title_text": str(row.metadata.get("title_text") or ""),
        "should_index": bool(row.metadata.get("should_index", True)),
    }


def _compose_embedding_text(row: ChunkRow, *, use_enriched_text: bool) -> str:
    if not use_enriched_text:
        return row.text

    metadata = row.metadata or {}
    parts: list[str] = []

    title_text = str(metadata.get("title_text") or "").strip()
    prefix_text = str(metadata.get("prefix_text") or "").strip()
    summary = str(metadata.get("summary") or "").strip()
    table_description = str(metadata.get("table_description") or "").strip()

    keywords_value = metadata.get("keywords")
    keywords: list[str] = []
    if isinstance(keywords_value, list):
        keywords = [str(x).strip() for x in keywords_value if str(x).strip()]

    if title_text:
        parts.append(f"Title: {title_text}")
    if row.section_name and row.section_name != "UNSPECIFIED":
        parts.append(f"Section: {row.section_name}")
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")
    if summary:
        parts.append(f"Summary: {summary}")
    if table_description:
        parts.append(f"Table description: {table_description}")
    if prefix_text:
        parts.append(f"Prefix: {prefix_text}")
    parts.append(f"Content: {row.text}")
    return "\n".join(parts)


def _embed_batch(
    *,
    batch_rows: list[ChunkRow],
    Document: Any,
    embedder: Any,
    use_enriched_text: bool,
) -> list[dict[str, Any]]:
    documents = [
        Document(
            id=row.chunk_id,
            content=_compose_embedding_text(row, use_enriched_text=use_enriched_text),
            meta=_chunk_to_meta(row),
        )
        for row in batch_rows
    ]
    result = embedder.run(documents=documents)
    embedded_docs = result.get("documents") if isinstance(result, dict) else None
    if not isinstance(embedded_docs, list):
        raise RuntimeError("Unexpected embedder output: missing 'documents' list.")

    out_rows: list[dict[str, Any]] = []
    for row, doc in zip(batch_rows, embedded_docs, strict=False):
        embedding = getattr(doc, "embedding", None)
        if embedding is None:
            raise RuntimeError(f"Embedder returned null embedding for chunk_id={row.chunk_id}")
        out_rows.append(
            {
                "id": row.chunk_id,
                "content": _compose_embedding_text(row, use_enriched_text=use_enriched_text),
                "embedding": [float(x) for x in embedding],
                "meta": _chunk_to_meta(row),
            }
        )
    return out_rows


def main() -> None:
    args = _build_parser().parse_args()
    started = time.perf_counter()

    chunks_path = Path(args.chunks_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_out = out_dir / "documents_with_embeddings.jsonl"
    meta_out = out_dir / "index_meta.json"

    if not chunks_path.exists():
        print(f"chunks path not found: {chunks_path}", file=sys.stderr)
        sys.exit(1)

    rows = read_chunks_ndjson(chunks_path, strict=args.strict)
    if args.skip_non_indexable:
        rows = [row for row in rows if bool(row.metadata.get("should_index", True))]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        print("No chunks to index.", file=sys.stderr)
        sys.exit(1)

    Document, SentenceTransformersDocumentEmbedder = _import_haystack_document_and_embedder()
    embedder = SentenceTransformersDocumentEmbedder(model=args.embedder_model)
    if hasattr(embedder, "warm_up"):
        embedder.warm_up()

    total = len(rows)
    count = 0
    dim = 0
    with docs_out.open("w", encoding="utf-8") as f:
        for start in range(0, total, args.batch_size):
            batch = rows[start : start + args.batch_size]
            embedded_rows = _embed_batch(
                batch_rows=batch,
                Document=Document,
                embedder=embedder,
                use_enriched_text=args.use_enriched_text,
            )
            for doc_row in embedded_rows:
                if not dim:
                    dim = len(doc_row["embedding"])
                f.write(json.dumps(doc_row, ensure_ascii=False) + "\n")
            count += len(embedded_rows)
            if args.progress_every > 0 and count % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                print(f"  indexed {count}/{total} chunks ({format_duration_minsec(elapsed)})")

    elapsed = time.perf_counter() - started
    meta = {
        "chunks_path": str(chunks_path),
        "documents_path": str(docs_out),
        "embedder_model": args.embedder_model,
        "use_enriched_text": bool(args.use_enriched_text),
        "skip_non_indexable": bool(args.skip_non_indexable),
        "count": count,
        "dimension": dim,
        "elapsed_seconds": round(elapsed, 2),
        "elapsed": format_duration_minsec(elapsed),
    }
    meta_out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Done.")
    print(f"Documents: {docs_out}")
    print(f"Meta:      {meta_out}")
    print(f"Indexed:   {count} chunks | dim={dim} | elapsed={format_duration_minsec(elapsed)}")


if __name__ == "__main__":
    main()
