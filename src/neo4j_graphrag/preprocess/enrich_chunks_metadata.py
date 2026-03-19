from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def _import_ollama_client():
    try:
        from ollama import Client
    except ImportError as exc:
        raise ImportError(
            "Could not import ollama client. Install with `pip install ollama`."
        ) from exc
    return Client


def _response_content(response: Any) -> str:
    if isinstance(response, dict):
        message = response.get("message") or {}
        if isinstance(message, dict):
            return str(message.get("content", "") or "")
    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content is not None:
            return str(content)
    content = getattr(response, "content", None)
    if content is not None:
        return str(content)
    return str(response)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _first_two_sentences(text: str) -> str:
    clean = (text or "").strip()
    if not clean:
        return ""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", clean) if p.strip()]
    if not parts:
        return clean
    return " ".join(parts[:2]).strip()


def _extract_json_object(raw: str) -> dict[str, Any]:
    candidate = (raw or "").strip()
    if not candidate:
        raise ValueError("Empty LLM response.")
    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not find JSON object in LLM response.")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Parsed JSON is not an object.")
    return payload


def _normalize_keywords(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = [x.strip() for x in re.split(r"[,;\n]", value) if x.strip()]
    elif isinstance(value, list):
        parts = [_normalize_space(str(x)) for x in value if _normalize_space(str(x))]
    else:
        parts = []

    out: list[str] = []
    seen: set[str] = set()
    for item in parts:
        norm = _normalize_space(item).strip('"').strip("'")
        key = norm.casefold()
        if not norm or key in seen:
            continue
        seen.add(key)
        out.append(norm)
        if len(out) >= 6:
            break
    return out


def _build_prompt(*, section_name: str, text: str, is_table: bool) -> str:
    return (
        "You are enriching a chunk from a financial report for indexing and retrieval.\n"
        "Return JSON only with exactly these keys:\n"
        '{\n'
        '  "keywords": ["kw1", "kw2"],\n'
        '  "summary": "short summary",\n'
        '  "table_description": "caption-style description or empty string"\n'
        '}\n\n'
        "Rules:\n"
        "- keywords: at most 6 concise phrases.\n"
        "- summary: 1-2 short sentences, factual, no speculation.\n"
        "- table_description: only fill this if the chunk is a table; otherwise return an empty string.\n"
        "- Do not include markdown.\n"
        "- Do not include any text outside the JSON object.\n\n"
        f"Section: {section_name or 'UNSPECIFIED'}\n"
        f"Is table: {'yes' if is_table else 'no'}\n"
        "Chunk text:\n"
        f"{text}"
    )


def _chat(client: Any, *, model: str, prompt: str, temperature: float) -> str:
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return _response_content(response)


def _read_ndjson(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"Line {line_no} is not a JSON object.")
        rows.append(payload)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enrich chunks.ndjson with LLM-generated indexing metadata."
    )
    parser.add_argument("--chunks-path", required=True, help="Input chunks NDJSON path.")
    parser.add_argument("--output", required=True, help="Output enriched NDJSON path.")
    parser.add_argument("--llm-model", required=True, help="Ollama model name.")
    parser.add_argument(
        "--ollama-host",
        default="http://127.0.0.1:11434",
        help="Ollama host URL.",
    )
    parser.add_argument(
        "--ollama-timeout-seconds",
        type=float,
        default=None,
        help="Optional Ollama client timeout in seconds.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for metadata generation.",
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=5000,
        help="Truncate chunk text to this many characters before prompting the model.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start enriching from this chunk index in the input file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only enrich at most this many rows.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows that already have keywords and summary in metadata.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on first LLM/parsing error.",
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

    rows = _read_ndjson(chunks_path)
    if not rows:
        print("No chunks found.", file=sys.stderr)
        sys.exit(1)

    Client = _import_ollama_client()
    if args.ollama_timeout_seconds is None:
        client = Client(host=args.ollama_host)
    else:
        client = Client(host=args.ollama_host, timeout=args.ollama_timeout_seconds)

    start = max(0, args.start_index)
    end = len(rows) if args.limit is None else min(len(rows), start + max(0, args.limit))

    processed = 0
    enriched = 0
    skipped = 0
    errors = 0

    for idx, row in enumerate(rows):
        if idx < start or idx >= end:
            continue
        processed += 1
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
            row["metadata"] = metadata

        if args.skip_existing and metadata.get("keywords") and metadata.get("summary"):
            skipped += 1
            continue

        text = str(row.get("text") or "").strip()
        section_name = str(row.get("section_name") or "")
        is_table = bool(metadata.get("is_table")) or str(row.get("chunk_engine") or "") == "elements_table"
        prefix_text = "" if is_table else _first_two_sentences(text)
        prompt = _build_prompt(
            section_name=section_name,
            text=_truncate(text, args.max_input_chars),
            is_table=is_table,
        )

        try:
            raw = _chat(
                client,
                model=args.llm_model,
                prompt=prompt,
                temperature=args.temperature,
            )
            payload = _extract_json_object(raw)
            keywords = _normalize_keywords(payload.get("keywords"))
            summary = _normalize_space(str(payload.get("summary") or ""))
            table_description = _normalize_space(str(payload.get("table_description") or ""))

            metadata["keywords"] = keywords
            metadata["summary"] = summary
            metadata["prefix_text"] = prefix_text
            metadata["table_description"] = table_description if is_table else ""
            metadata["metadata_enrichment"] = {
                "provider": "ollama",
                "model": args.llm_model,
                "temperature": args.temperature,
                "input_char_count": len(text),
                "prompt_input_char_count": min(len(text), args.max_input_chars),
                "is_table": is_table,
                "version": "v1",
            }
            enriched += 1
            print(
                f"[{processed}/{max(0, end - start)}] enriched chunk_index={row.get('chunk_index')} "
                f"doc_id={row.get('doc_id')} keywords={len(keywords)}"
            )
        except Exception as exc:
            errors += 1
            metadata["metadata_enrichment_error"] = str(exc)
            print(
                f"[error] chunk_index={row.get('chunk_index')} doc_id={row.get('doc_id')}: {exc}",
                file=sys.stderr,
            )
            if args.strict:
                raise

    with output_path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nDone.")
    print(f"Input:     {chunks_path}")
    print(f"Output:    {output_path}")
    print(f"Rows:      {len(rows)}")
    print(f"Processed: {processed}")
    print(f"Enriched:  {enriched}")
    print(f"Skipped:   {skipped}")
    print(f"Errors:    {errors}")


if __name__ == "__main__":
    main()
