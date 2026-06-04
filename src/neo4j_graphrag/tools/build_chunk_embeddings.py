from __future__ import annotations

import argparse
import json
import math
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


def _count_candidate_chunks(
    paths: Iterable[Path], *, only_should_index: bool, strict: bool
) -> int:
    return sum(
        1
        for _ in _iter_chunks(
            paths,
            only_should_index=only_should_index,
            strict=strict,
        )
    )


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60:.2f}h"


def _chunk_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": chunk.get("doc_id"),
        "section_name": chunk.get("section_name"),
        "chunk_index": chunk.get("chunk_index"),
        "page": chunk.get("page"),
        "source_type": chunk.get("source_type"),
        "parser_engine": chunk.get("parser_engine"),
        "chunk_engine": chunk.get("chunk_engine"),
    }


def _write_failed_chunk(
    failed_handle: Any,
    *,
    chunk: dict[str, Any],
    error: Exception,
    model: str,
) -> None:
    text = str(chunk.get("text") or "")
    row = {
        "chunk_id": chunk.get("chunk_id"),
        "doc_id": chunk.get("doc_id"),
        "section_name": chunk.get("section_name"),
        "chunk_index": chunk.get("chunk_index"),
        "model": model,
        "error_type": type(error).__name__,
        "error": str(error),
        "text_preview": text[:500],
    }
    failed_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    failed_handle.flush()


def _validate_embedding(embedding: Any) -> list[float]:
    if not isinstance(embedding, list) or not embedding:
        length = len(embedding) if isinstance(embedding, list) else "n/a"
        raise ValueError(f"invalid embedding container (len={length})")
    values: list[float] = []
    for index, value in enumerate(embedding):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"invalid embedding value at index {index}: {value!r}")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"non-finite embedding value at index {index}: {value!r}")
        values.append(numeric_value)
    return values


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
        "--failed-output",
        default=None,
        help=(
            "Path for failed embedding records. Defaults to failed_embeddings.jsonl "
            "next to --output."
        ),
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
    failed_output_path = (
        Path(args.failed_output).expanduser()
        if args.failed_output
        else output_path.with_name("failed_embeddings.jsonl")
    )
    failed_output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = _load_existing_ids(output_path) if args.resume else set()
    total_candidates = _count_candidate_chunks(
        chunk_paths,
        only_should_index=not args.include_non_indexed,
        strict=args.strict,
    )
    embedder_kwargs: dict[str, Any] = {}
    if args.ollama_host:
        embedder_kwargs["host"] = args.ollama_host
    embedder = OllamaEmbeddings(model=args.model, **embedder_kwargs)

    scanned = 0
    skipped_existing = 0
    embedded = 0
    failed = 0
    remaining_target = max(total_candidates - len(existing_ids), 0) if args.resume else total_candidates
    if args.limit > 0:
        remaining_target = min(remaining_target, args.limit)

    print("Starting chunk embedding build.", flush=True)
    print(f"Start time:         {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Chunk files:        {len(chunk_paths)}", flush=True)
    print(f"Candidate chunks:   {total_candidates}", flush=True)
    print(f"Existing embeddings: {len(existing_ids)}", flush=True)
    print(f"Remaining target:   {remaining_target}", flush=True)
    print(f"Output:             {output_path}", flush=True)
    print(f"Failed output:      {failed_output_path}", flush=True)
    print(f"Model:              {args.model}", flush=True)

    with (
        output_path.open("a" if args.resume else "w", encoding="utf-8") as out,
        failed_output_path.open("a" if args.resume else "w", encoding="utf-8") as failed_out,
    ):
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
            try:
                embedding = _validate_embedding(embedder.embed_query(text))
            except Exception as exc:
                failed += 1
                _write_failed_chunk(
                    failed_out,
                    chunk=chunk,
                    error=exc,
                    model=args.model,
                )
                print(
                    "[embed-error] "
                    f"failed={failed} chunk_id={chunk_id} "
                    f"doc_id={chunk.get('doc_id')} "
                    f"section={chunk.get('section_name')} "
                    f"error={type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.strict:
                    raise
                continue
            row = {
                "id": chunk_id,
                "content": text,
                "metadata": _chunk_metadata(chunk),
                "embedding": embedding,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            embedded += 1

            if args.progress_every > 0 and embedded % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                rate = embedded / elapsed if elapsed > 0 else 0.0
                remaining = max(remaining_target - embedded - failed, 0)
                eta = remaining / rate if rate > 0 else None
                print(
                    f"  embedded={embedded} failed={failed} scanned={scanned} "
                    f"skipped_existing={skipped_existing} "
                    f"elapsed={_format_duration(elapsed)} rate={rate:.2f}/s "
                    f"ETA={_format_duration(eta)}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    print("Done.")
    print(f"End time:          {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Chunk files:       {len(chunk_paths)}")
    print(f"Candidate chunks:  {total_candidates}")
    print(f"Scanned chunks:    {scanned}")
    print(f"Embedded chunks:   {embedded}")
    print(f"Failed chunks:     {failed}")
    print(f"Skipped existing:  {skipped_existing}")
    print(f"Output:            {output_path}")
    print(f"Failed output:     {failed_output_path}")
    print(f"Model:             {args.model}")
    print(f"Elapsed:           {_format_duration(elapsed)}")
    print(f"Elapsed seconds:   {elapsed:.2f}")


if __name__ == "__main__":
    main()
