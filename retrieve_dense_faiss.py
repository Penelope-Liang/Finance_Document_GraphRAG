from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "Could not import faiss. Install with `uv pip install faiss-cpu` "
            "or on GPU machine use `faiss-gpu`."
        ) from exc
    return faiss


def _import_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Could not import numpy. Install with `uv pip install numpy`.") from exc
    return np


def _create_embedder(backend: str, model: str):
    if backend == "ollama":
        from neo4j_graphrag.embeddings import OllamaEmbeddings

        return OllamaEmbeddings(model=model)
    if backend == "sentence-transformers":
        from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings

        return SentenceTransformerEmbeddings(model=model)
    raise ValueError(f"Unsupported backend: {backend}")


def _close_embedder(embedder: Any) -> None:
    if hasattr(embedder, "client") and hasattr(embedder.client, "_client"):
        embedder.client._client.close()


def _load_mapping(path: Path) -> dict[int, dict[str, Any]]:
    mapping: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            int_id = payload.get("int_id")
            if int_id is None:
                raise ValueError(f"Missing int_id in mapping at line {line_no}")
            mapping[int(int_id)] = payload
    return mapping


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dense retrieve from saved FAISS index and return chunk_id hits."
    )
    parser.add_argument(
        "-q",
        "--query",
        required=True,
        help="Query text.",
    )
    parser.add_argument(
        "--index-path",
        default="artifacts/faiss/chunks.index",
        help="Path to FAISS index file.",
    )
    parser.add_argument(
        "--mapping-path",
        default="artifacts/faiss/intid_to_chunkid.jsonl",
        help="Path to int_id -> chunk_id mapping JSONL.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top K results.")
    parser.add_argument(
        "--embedder-backend",
        choices=["ollama", "sentence-transformers"],
        default="ollama",
        help="Embedding backend.",
    )
    parser.add_argument(
        "--embedder-model",
        default=os.environ.get("OLLAMA_EMBEDDER_MODEL", "bge-m3:latest"),
        help="Embedding model identifier.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization before IndexFlatIP search.",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    faiss = _import_faiss()
    np = _import_numpy()

    index_path = Path(args.index_path)
    mapping_path = Path(args.mapping_path)
    if not index_path.exists():
        print(f"Index file not found: {index_path}", file=sys.stderr)
        sys.exit(1)
    if not mapping_path.exists():
        print(f"Mapping file not found: {mapping_path}", file=sys.stderr)
        sys.exit(1)

    index = faiss.read_index(str(index_path))
    mapping = _load_mapping(mapping_path)

    embedder = _create_embedder(args.embedder_backend, args.embedder_model)
    try:
        query_vec = np.asarray(embedder.embed_query(args.query), dtype="float32").reshape(1, -1)
    finally:
        _close_embedder(embedder)

    if not args.no_normalize:
        faiss.normalize_L2(query_vec)

    scores, ids = index.search(query_vec, args.top_k)
    results: list[dict[str, Any]] = []
    for score, int_id in zip(scores[0], ids[0]):
        if int(int_id) < 0:
            continue
        row = mapping.get(int(int_id), {})
        results.append(
            {
                "int_id": int(int_id),
                "chunk_id": row.get("chunk_id"),
                "score": float(score),
                "doc_id": row.get("doc_id"),
                "chunk_index": row.get("chunk_index"),
                "section_name": row.get("section_name"),
            }
        )

    if args.output_format == "json":
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if not results:
        print("No results.")
        return

    for rank, row in enumerate(results, start=1):
        print(
            f"{rank:>2}. score={row['score']:.6f} "
            f"chunk_id={row.get('chunk_id')} doc_id={row.get('doc_id')} "
            f"chunk_index={row.get('chunk_index')} section={row.get('section_name')}"
        )


if __name__ == "__main__":
    main()
