from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from neo4j_graphrag.embeddings import OllamaEmbeddings


def _iter_input_paths(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            files.extend(sorted(path.glob("*_chunks.ndjson")))
        elif path.exists():
            files.append(path)
        else:
            print(f"[warn] input path not found: {path}", file=sys.stderr)
    return files


def _load_existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            chunk_id = payload.get("id")
            if isinstance(chunk_id, str) and chunk_id:
                ids.add(chunk_id)
    return ids


def _iter_chunks(paths: Iterable[Path], *, only_should_index: bool, strict: bool) -> Iterable[dict[str, Any]]:
    seen: set[str] = set()
    for path in paths:
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
                    print(f"[warn] skip invalid JSON at {path}:{line_no}: {exc}", file=sys.stderr)
                    continue

                chunk_id = payload.get("chunk_id")
                text = payload.get("text")
                metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
                if not isinstance(chunk_id, str) or not chunk_id:
                    if strict:
                        raise ValueError(f"Missing chunk_id at {path}:{line_no}")
                    continue
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                if not isinstance(text, str) or not text.strip():
                    if strict:
                        raise ValueError(f"Missing text for chunk {chunk_id} at {path}:{line_no}")
                    continue
                if only_should_index and metadata.get("should_index") is False:
                    continue

                yield payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build documents_with_embeddings.jsonl from chunks.ndjson files. "
            "The output is compatible with tools/import_chunk_vectors.py."
        )
    )
    parser.add_argument(
        "--chunks-path",
        action="append",
        required=True,
        help=(
            "Input chunks.ndjson file or directory containing *_chunks.ndjson. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output documents_with_embeddings.jsonl path.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3:latest"),
        help="Ollama embedding model.",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST"),
        help="Ollama host URL, e.g. http://127.0.0.1:11434.",
    )
    parser.add_argument(
        "--include-non-indexed",
        action="store_true",
        help="Also embed chunks with metadata.should_index=false.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip chunk ids already present in the output file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of new chunks to embed (0 = all).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N embedded chunks.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on malformed chunk rows.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    started = time.perf_counter()

    chunk_paths = _iter_input_paths(args.chunks_path)
    if not chunk_paths:
        print("No chunk files found.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = _load_existing_ids(output_path) if args.resume else set()
    embedder_kwargs: dict[str, Any] = {}
    if args.ollama_host:
        embedder_kwargs["host"] = args.ollama_host
    embedder = OllamaEmbeddings(model=args.model, **embedder_kwargs)

    scanned = 0
    skipped_existing = 0
    embedded = 0
    with output_path.open("a" if args.resume else "w", encoding="utf-8") as out:
        for chunk in _iter_chunks(
            chunk_paths,
            only_should_index=not args.include_non_indexed,
            strict=args.strict,
        ):
            scanned += 1
            chunk_id = str(chunk["chunk_id"])
            if chunk_id in existing_ids:
                skipped_existing += 1
                continue
            if args.limit > 0 and embedded >= args.limit:
                break

            text = str(chunk["text"])
            embedding = embedder.embed_query(text)
            row = {
                "id": chunk_id,
                "content": text,
                "metadata": {
                    "doc_id": chunk.get("doc_id"),
                    "section_name": chunk.get("section_name"),
                    "chunk_index": chunk.get("chunk_index"),
                    "page": chunk.get("page"),
                    "source_type": chunk.get("source_type"),
                    "parser_engine": chunk.get("parser_engine"),
                    "chunk_engine": chunk.get("chunk_engine"),
                },
                "embedding": embedding,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            embedded += 1

            if args.progress_every > 0 and embedded % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  embedded {embedded} chunks "
                    f"(scanned={scanned}, skipped_existing={skipped_existing}, elapsed={elapsed:.1f}s)"
                )

    elapsed = time.perf_counter() - started
    print("Done.")
    print(f"Chunk files:       {len(chunk_paths)}")
    print(f"Scanned chunks:    {scanned}")
    print(f"Embedded chunks:   {embedded}")
    print(f"Skipped existing:  {skipped_existing}")
    print(f"Output:            {output_path}")
    print(f"Model:             {args.model}")
    print(f"Elapsed seconds:   {elapsed:.2f}")


if __name__ == "__main__":
    main()
