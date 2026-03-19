from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _bool_flag(enabled: bool, flag: str) -> list[str]:
    return [flag] if enabled else []


def _run_step(cmd: list[str]) -> None:
    print(f"\n[run] {' '.join(shlex.quote(part) for part in cmd)}")
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the element-based PDF pipeline: extract -> chunk -> optional enrich."
    )
    parser.add_argument("--input-path", required=True, help="PDF file or directory.")
    parser.add_argument("--elements-output", required=True, help="Output path for elements NDJSON.")
    parser.add_argument("--chunks-output", required=True, help="Output path for structured chunks NDJSON.")
    parser.add_argument(
        "--enriched-output",
        default="",
        help="Optional output path for enriched chunks NDJSON. If omitted, enrichment is skipped.",
    )

    parser.add_argument("--strategy", default="hi_res", help="partition_pdf strategy.")
    parser.add_argument(
        "--hi-res-model-name",
        default="yolox",
        help="High-resolution layout model for extract_elements.",
    )
    parser.add_argument(
        "--infer-table-structure",
        action="store_true",
        help="Preserve table structure during extraction.",
    )
    parser.add_argument(
        "--extract-image-block-types",
        default="",
        help="Optional comma-separated image block types for extract_elements.",
    )
    parser.add_argument(
        "--skip-infer-table-types",
        default="",
        help="Optional comma-separated element types to skip for table inference.",
    )
    parser.add_argument(
        "--strict-extract",
        action="store_true",
        help="Fail fast on extract_elements file errors.",
    )

    parser.add_argument(
        "--chunk-id-version",
        default="v1",
        help="Version prefix embedded in chunk_id payload.",
    )
    parser.add_argument(
        "--target-max-chars",
        type=int,
        default=2048,
        help="Target max chars for merging element groups.",
    )
    parser.add_argument(
        "--min-uncategorized-len",
        type=int,
        default=12,
        help="Drop short UncategorizedText below this threshold.",
    )
    parser.add_argument(
        "--merge-table-fragments",
        action="store_true",
        help="Attach short same-page table fragments back to the table chunk.",
    )

    parser.add_argument("--llm-model", default="", help="Ollama model for metadata enrichment.")
    parser.add_argument(
        "--ollama-host",
        default="http://127.0.0.1:11434",
        help="Ollama host URL for enrichment.",
    )
    parser.add_argument(
        "--ollama-timeout-seconds",
        type=float,
        default=None,
        help="Optional Ollama timeout in seconds for enrichment.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for enrichment.",
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=5000,
        help="Truncate chunk text before metadata prompting.",
    )
    parser.add_argument(
        "--enrich-start-index",
        type=int,
        default=0,
        help="Start enrichment from this row index.",
    )
    parser.add_argument(
        "--enrich-limit",
        type=int,
        default=None,
        help="Only enrich this many rows.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip enrichment for rows already containing keywords and summary.",
    )
    parser.add_argument(
        "--only-should-index",
        action="store_true",
        help="Only enrich rows whose metadata.should_index is true.",
    )
    parser.add_argument(
        "--strict-enrich",
        action="store_true",
        help="Fail fast on enrichment errors.",
    )
    parser.add_argument(
        "--skip-enrich",
        action="store_true",
        help="Run only extract + chunk even if enriched output is provided.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    elements_output = Path(args.elements_output).expanduser().resolve()
    chunks_output = Path(args.chunks_output).expanduser().resolve()
    enriched_output = Path(args.enriched_output).expanduser().resolve() if args.enriched_output else None

    elements_output.parent.mkdir(parents=True, exist_ok=True)
    chunks_output.parent.mkdir(parents=True, exist_ok=True)
    if enriched_output is not None:
        enriched_output.parent.mkdir(parents=True, exist_ok=True)

    extract_cmd = [
        sys.executable,
        "-u",
        "-m",
        "neo4j_graphrag.preprocess.extract_elements",
        "--input-path",
        args.input_path,
        "--output",
        str(elements_output),
        "--strategy",
        args.strategy,
        "--hi-res-model-name",
        args.hi_res_model_name,
        * _bool_flag(args.infer_table_structure, "--infer-table-structure"),
        * (_bool_flag(bool(args.extract_image_block_types), "--extract-image-block-types") + ([args.extract_image_block_types] if args.extract_image_block_types else [])),
        * (_bool_flag(bool(args.skip_infer_table_types), "--skip-infer-table-types") + ([args.skip_infer_table_types] if args.skip_infer_table_types else [])),
        * _bool_flag(args.strict_extract, "--strict"),
    ]
    _run_step(extract_cmd)

    chunk_cmd = [
        sys.executable,
        "-u",
        "-m",
        "neo4j_graphrag.preprocess.chunk_from_elements",
        "--elements-path",
        str(elements_output),
        "--output",
        str(chunks_output),
        "--chunk-id-version",
        args.chunk_id_version,
        "--target-max-chars",
        str(args.target_max_chars),
        "--min-uncategorized-len",
        str(args.min_uncategorized_len),
        * _bool_flag(args.merge_table_fragments, "--merge-table-fragments"),
    ]
    _run_step(chunk_cmd)

    if args.skip_enrich:
        print("\n[done] enrichment skipped by --skip-enrich")
        return

    if enriched_output is None:
        print("\n[done] enrichment skipped because --enriched-output was not provided")
        return

    if not args.llm_model:
        raise SystemExit("--llm-model is required when --enriched-output is provided.")

    enrich_cmd = [
        sys.executable,
        "-u",
        "-m",
        "neo4j_graphrag.preprocess.enrich_chunks_metadata",
        "--chunks-path",
        str(chunks_output),
        "--output",
        str(enriched_output),
        "--llm-model",
        args.llm_model,
        "--ollama-host",
        args.ollama_host,
        "--temperature",
        str(args.temperature),
        "--max-input-chars",
        str(args.max_input_chars),
        "--start-index",
        str(args.enrich_start_index),
        * _bool_flag(args.skip_existing, "--skip-existing"),
        * _bool_flag(args.only_should_index, "--only-should-index"),
        * _bool_flag(args.strict_enrich, "--strict"),
    ]
    if args.ollama_timeout_seconds is not None:
        enrich_cmd.extend(["--ollama-timeout-seconds", str(args.ollama_timeout_seconds)])
    if args.enrich_limit is not None:
        enrich_cmd.extend(["--limit", str(args.enrich_limit)])
    _run_step(enrich_cmd)

    print("\nDone.")
    print(f"Elements: {elements_output}")
    print(f"Chunks:   {chunks_output}")
    print(f"Enriched: {enriched_output}")


if __name__ == "__main__":
    main()
