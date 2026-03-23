from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from neo4j_graphrag.preprocess.statement_type import classify_statement_type


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate chunks NDJSON with metadata.statement_type."
    )
    parser.add_argument("--chunks-path", required=True, help="Input chunks/enriched chunks NDJSON.")
    parser.add_argument("--output", required=True, help="Output NDJSON path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing metadata.statement_type values.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N rows.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    chunks_path = Path(args.chunks_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        print(f"Input file not found: {chunks_path}", file=sys.stderr)
        sys.exit(1)

    total = 0
    updated = 0
    counts: Counter[str] = Counter()

    with chunks_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_no, line in enumerate(src, start=1):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_no} is not a JSON object.")

            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                payload["metadata"] = metadata

            if args.overwrite or not str(metadata.get("statement_type") or "").strip():
                metadata["statement_type"] = classify_statement_type(
                    section_name=str(payload.get("section_name") or ""),
                    title_text=str(metadata.get("title_text") or ""),
                    text=str(payload.get("text") or ""),
                    metadata=metadata,
                )
                updated += 1

            statement_type = str(metadata.get("statement_type") or "other").strip() or "other"
            counts[statement_type] += 1
            dst.write(json.dumps(payload, ensure_ascii=False) + "\n")
            total += 1

            if args.progress_every > 0 and total % args.progress_every == 0:
                print(f"  annotated {total} rows")

    print("Done.")
    print(f"Input:    {chunks_path}")
    print(f"Output:   {output_path}")
    print(f"Rows:     {total}")
    print(f"Updated:  {updated}")
    print("Counts:")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
