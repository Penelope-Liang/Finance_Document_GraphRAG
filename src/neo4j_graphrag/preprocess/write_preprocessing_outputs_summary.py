from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iter_ndjson(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(payload)
    return rows


def _summarize_artifact(name: str, path: Path) -> dict[str, Any]:
    rows = _iter_ndjson(path)
    doc_ids = sorted({str(row.get("doc_id")) for row in rows if row.get("doc_id")})
    parser_engines = sorted(
        {
            str(row.get("parser_engine") or "")
            for row in rows
            if row.get("parser_engine")
        }
    )
    chunk_engines = sorted(
        {
            str(row.get("chunk_engine") or "")
            for row in rows
            if row.get("chunk_engine")
        }
    )
    table_rows = 0
    for row in rows:
        metadata = row.get("metadata")
        if isinstance(metadata, dict) and metadata.get("table_structured"):
            table_rows += 1
    return {
        "name": name,
        "path": str(path),
        "line_count": len(rows),
        "doc_ids": doc_ids,
        "parser_engines": parser_engines,
        "chunk_engines": chunk_engines,
        "table_structured_rows": table_rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a reproducible manifest for frozen Docling pipeline artifacts."
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Artifact in name=path form. May be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON manifest path.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not args.artifact:
        raise ValueError("At least one --artifact name=path entry is required.")

    artifacts: list[dict[str, Any]] = []
    for item in args.artifact:
        if "=" not in item:
            raise ValueError(f"Invalid artifact spec '{item}'. Expected name=path.")
        name, raw_path = item.split("=", 1)
        artifact_path = Path(raw_path).expanduser().resolve()
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")
        artifacts.append(_summarize_artifact(name.strip(), artifact_path))

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": _now_utc(),
        "artifacts": artifacts,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote manifest: {output_path}")
    print(f"Artifacts: {len(artifacts)}")


if __name__ == "__main__":
    main()
