from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_MODULE = _THIS_DIR / "prompts.py"
DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"


@dataclass
class ChunkRow:
    chunk_id: str
    text: str
    doc_id: str


@dataclass
class RawEntity:
    title: str
    entity_type: str
    description: str
    source_id: str
    doc_id: str


@dataclass
class RawRelationship:
    source: str
    target: str
    description: str
    weight: float
    source_id: str
    doc_id: str


def _import_ollama_client():
    try:
        from ollama import Client
    except ImportError as exc:
        raise ImportError(
            "Could not import ollama client. Install with `pip install ollama`."
        ) from exc
    return Client


def _normalize_space(text: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _clean_field(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    return _normalize_space(text)


def _sha1_hex(payload: str) -> str:
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _entity_key(title: str, entity_type: str) -> str:
    return _sha1_hex(f"{title}|{entity_type}")


def _normalize_record_type(value: Any) -> str:
    return _clean_field(value).strip().strip('"').strip("'").lower()


def _load_prompt_bundle(prompt_file: str | None) -> tuple[str, str, str]:
    if prompt_file:
        prompt_path = Path(prompt_file).expanduser().resolve()
    else:
        prompt_path = DEFAULT_PROMPT_MODULE
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    payload = runpy.run_path(str(prompt_path))
    extraction_prompt = payload.get("GRAPH_EXTRACTION_PROMPT")
    continue_prompt = payload.get("CONTINUE_PROMPT")
    loop_prompt = payload.get("LOOP_PROMPT")
    if not isinstance(extraction_prompt, str):
        raise ValueError(f"GRAPH_EXTRACTION_PROMPT missing in {prompt_path}")
    if not isinstance(continue_prompt, str):
        continue_prompt = (
            "MANY entities and relationships were missed in the last extraction. "
            "Add them below using the same format:\n"
        )
    if not isinstance(loop_prompt, str):
        loop_prompt = (
            "Answer Y if there are still missing entities or relationships; "
            "otherwise answer N."
        )
    return extraction_prompt, continue_prompt, loop_prompt


def _response_content(response: Any) -> str:
    # Supports both dict-style and object-style ollama responses.
    if isinstance(response, dict):
        message = response.get("message") or {}
        if isinstance(message, dict):
            return str(message.get("content", ""))
    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content is not None:
            return str(content)
    content = getattr(response, "content", None)
    if content is not None:
        return str(content)
    return str(response)


def _chat(client: Any, model: str, messages: list[dict[str, str]], temperature: float) -> str:
    response = client.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
    )
    return _response_content(response)


def _extract_for_text(
    *,
    client: Any,
    model: str,
    text: str,
    entity_types: list[str],
    extraction_prompt: str,
    continue_prompt: str,
    loop_prompt: str,
    max_gleanings: int,
    temperature: float,
) -> str:
    prompt_text = extraction_prompt.format(
        input_text=text,
        entity_types=",".join(entity_types),
    )
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt_text}]
    first = _chat(client, model, messages, temperature)
    messages.append({"role": "assistant", "content": first})
    result = first

    if max_gleanings <= 0:
        return result

    for idx in range(max_gleanings):
        messages.append({"role": "user", "content": continue_prompt})
        addition = _chat(client, model, messages, temperature)
        messages.append({"role": "assistant", "content": addition})
        result += addition
        if idx >= max_gleanings - 1:
            break

        messages.append({"role": "user", "content": loop_prompt})
        decision = _chat(client, model, messages, temperature)
        messages.append({"role": "assistant", "content": decision})
        if _clean_field(decision).upper() != "Y":
            break

    return result


def _parse_result(
    *,
    result: str,
    source_id: str,
    doc_id: str,
) -> tuple[list[RawEntity], list[RawRelationship]]:
    entities: list[RawEntity] = []
    relationships: list[RawRelationship] = []
    records = [part.strip() for part in result.split(DEFAULT_RECORD_DELIMITER)]
    for raw in records:
        cleaned = re.sub(r"^\(|\)$", "", raw.strip())
        if not cleaned or cleaned == DEFAULT_COMPLETION_DELIMITER:
            continue
        attrs = cleaned.split(DEFAULT_TUPLE_DELIMITER)
        record_type = _normalize_record_type(attrs[0])

        if record_type == "entity" and len(attrs) >= 4:
            title = _clean_field(attrs[1]).upper()
            entity_type = _clean_field(attrs[2]).upper()
            description = _clean_field(attrs[3])
            if not title:
                continue
            entities.append(
                RawEntity(
                    title=title,
                    entity_type=entity_type or "UNKNOWN",
                    description=description,
                    source_id=source_id,
                    doc_id=doc_id,
                )
            )
            continue

        if record_type == "relationship" and len(attrs) >= 5:
            source = _clean_field(attrs[1]).upper()
            target = _clean_field(attrs[2]).upper()
            description = _clean_field(attrs[3])
            if not source or not target:
                continue
            try:
                weight = float(_clean_field(attrs[-1]))
            except ValueError:
                weight = 1.0
            relationships.append(
                RawRelationship(
                    source=source,
                    target=target,
                    description=description,
                    weight=weight,
                    source_id=source_id,
                    doc_id=doc_id,
                )
            )
    return entities, relationships


def _read_chunks(
    path: Path,
    *,
    chunk_id_key: str,
    text_key: str,
    doc_id_key: str,
    strict: bool,
) -> list[ChunkRow]:
    rows: list[ChunkRow] = []
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
            chunk_id = payload.get(chunk_id_key)
            text = payload.get(text_key)
            doc_id = payload.get(doc_id_key) or "UNKNOWN_DOC"
            if not isinstance(chunk_id, str) or not chunk_id:
                if strict:
                    raise ValueError(f"Missing chunk id in line {line_no}")
                continue
            if not isinstance(text, str) or not text.strip():
                if strict:
                    raise ValueError(f"Missing text in line {line_no}")
                continue
            rows.append(ChunkRow(chunk_id=chunk_id, text=text, doc_id=str(doc_id)))
    return rows


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _merge_entities(raw_entities: list[RawEntity]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[RawEntity]] = defaultdict(list)
    for row in raw_entities:
        grouped[(row.title, row.entity_type)].append(row)

    merged: list[dict[str, Any]] = []
    for (title, entity_type), rows in grouped.items():
        descriptions = _ordered_unique([r.description for r in rows if r.description])
        text_unit_ids = _ordered_unique([r.source_id for r in rows])
        doc_ids = _ordered_unique([r.doc_id for r in rows])
        merged.append(
            {
                "entity_key": _entity_key(title, entity_type),
                "title": title,
                "type": entity_type,
                "description": " ".join(descriptions).strip(),
                "descriptions": descriptions,
                "text_unit_ids": text_unit_ids,
                "doc_ids": doc_ids,
                "frequency": len(text_unit_ids),
                "source_method": "llm",
            }
        )
    merged.sort(key=lambda x: (x["title"], x["type"]))
    return merged


def _merge_relationships(
    raw_relationships: list[RawRelationship],
    merged_entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not raw_relationships or not merged_entities:
        return []

    title_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in merged_entities:
        title_candidates[row["title"]].append(row)

    def best_entity_for_title(title: str) -> dict[str, Any] | None:
        candidates = title_candidates.get(title, [])
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x.get("frequency", 0), x.get("type", "")), reverse=True)
        return candidates[0]

    grouped: dict[tuple[str, str], list[RawRelationship]] = defaultdict(list)
    for row in raw_relationships:
        grouped[(row.source, row.target)].append(row)

    merged: list[dict[str, Any]] = []
    for (source, target), rows in grouped.items():
        source_entity = best_entity_for_title(source)
        target_entity = best_entity_for_title(target)
        if source_entity is None or target_entity is None:
            continue
        descriptions = _ordered_unique([r.description for r in rows if r.description])
        text_unit_ids = _ordered_unique([r.source_id for r in rows])
        doc_ids = _ordered_unique([r.doc_id for r in rows])
        merged.append(
            {
                "source": source,
                "target": target,
                "source_key": source_entity["entity_key"],
                "target_key": target_entity["entity_key"],
                "description": " ".join(descriptions).strip(),
                "descriptions": descriptions,
                "text_unit_ids": text_unit_ids,
                "doc_ids": doc_ids,
                "weight": float(sum(r.weight for r in rows)),
                "source_method": "llm",
            }
        )
    merged.sort(key=lambda x: (x["source"], x["target"]))
    return merged


def _write_ndjson(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract entity/relationship graph from chunks.ndjson using an LLM. "
            "This reuses GraphRAG tuple-format extraction semantics."
        )
    )
    parser.add_argument(
        "--chunks-path",
        default="chunks.ndjson",
        help="Input chunks NDJSON path.",
    )
    parser.add_argument(
        "--entities-out",
        default="output/entities/entities.ndjson",
        help="Merged entities output path.",
    )
    parser.add_argument(
        "--relationships-out",
        default="output/entities/relationships.ndjson",
        help="Merged relationships output path.",
    )
    parser.add_argument(
        "--raw-entities-out",
        default="",
        help="Optional raw entities output path.",
    )
    parser.add_argument(
        "--raw-relationships-out",
        default="",
        help="Optional raw relationships output path.",
    )
    parser.add_argument(
        "--chunk-id-key",
        default="chunk_id",
        help="Field name for chunk id in input NDJSON.",
    )
    parser.add_argument(
        "--text-key",
        default="text",
        help="Field name for chunk text in input NDJSON.",
    )
    parser.add_argument(
        "--doc-id-key",
        default="doc_id",
        help="Field name for document id in input NDJSON.",
    )
    parser.add_argument(
        "--entity-types",
        default="organization,person,geo,event",
        help="Comma-separated entity types for extraction prompt.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:72b",
        help="Ollama model used for entity extraction.",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://127.0.0.1:11434",
        help="Ollama host URL.",
    )
    parser.add_argument(
        "--prompt-file",
        default="",
        help="Optional prompt python file that defines GRAPH_EXTRACTION_PROMPT.",
    )
    parser.add_argument(
        "--max-gleanings",
        type=int,
        default=1,
        help="Extra extraction passes per chunk.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit chunks to process (0 = all).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Log every N processed chunks.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on any parsing or model error.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    chunks_path = Path(args.chunks_path).expanduser().resolve()
    if not chunks_path.exists():
        print(f"Input file not found: {chunks_path}", file=sys.stderr)
        sys.exit(1)

    chunks = _read_chunks(
        chunks_path,
        chunk_id_key=args.chunk_id_key,
        text_key=args.text_key,
        doc_id_key=args.doc_id_key,
        strict=args.strict,
    )
    if not chunks:
        print("No valid chunks found.", file=sys.stderr)
        sys.exit(1)
    if args.limit > 0:
        chunks = chunks[: args.limit]

    extraction_prompt, continue_prompt, loop_prompt = _load_prompt_bundle(
        args.prompt_file or None
    )
    entity_types = [part.strip() for part in args.entity_types.split(",") if part.strip()]
    if not entity_types:
        print("No valid entity types provided.", file=sys.stderr)
        sys.exit(1)

    Client = _import_ollama_client()
    client = Client(host=args.ollama_host)

    raw_entities: list[RawEntity] = []
    raw_relationships: list[RawRelationship] = []
    for idx, chunk in enumerate(chunks, start=1):
        try:
            completion = _extract_for_text(
                client=client,
                model=args.model,
                text=chunk.text,
                entity_types=entity_types,
                extraction_prompt=extraction_prompt,
                continue_prompt=continue_prompt,
                loop_prompt=loop_prompt,
                max_gleanings=args.max_gleanings,
                temperature=args.temperature,
            )
            entities, relationships = _parse_result(
                result=completion,
                source_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
            )
            raw_entities.extend(entities)
            raw_relationships.extend(relationships)
        except Exception as exc:
            if args.strict:
                raise
            print(
                f"[warn] chunk {idx}/{len(chunks)} failed (chunk_id={chunk.chunk_id}): {exc}",
                file=sys.stderr,
            )

        if idx % max(1, args.progress_every) == 0 or idx == len(chunks):
            print(f"  processed {idx}/{len(chunks)} chunks")

    merged_entities = _merge_entities(raw_entities)
    merged_relationships = _merge_relationships(raw_relationships, merged_entities)

    entities_out = Path(args.entities_out).expanduser().resolve()
    relationships_out = Path(args.relationships_out).expanduser().resolve()
    _write_ndjson(entities_out, merged_entities)
    _write_ndjson(relationships_out, merged_relationships)

    if args.raw_entities_out:
        raw_entities_out = Path(args.raw_entities_out).expanduser().resolve()
        _write_ndjson(raw_entities_out, [e.__dict__ for e in raw_entities])
    if args.raw_relationships_out:
        raw_relationships_out = Path(args.raw_relationships_out).expanduser().resolve()
        _write_ndjson(raw_relationships_out, [r.__dict__ for r in raw_relationships])

    print("\nDone.")
    print(f"Chunks processed:    {len(chunks)}")
    print(f"Raw entities:        {len(raw_entities)}")
    print(f"Raw relationships:   {len(raw_relationships)}")
    print(f"Merged entities:     {len(merged_entities)}")
    print(f"Merged relationships:{len(merged_relationships)}")
    print(f"Entities out:        {entities_out}")
    print(f"Relationships out:   {relationships_out}")


if __name__ == "__main__":
    main()
