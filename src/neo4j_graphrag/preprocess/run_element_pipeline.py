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
        description="Run the element-based PDF pipeline: extract -> chunk."
    )
    parser.add_argument("--input-path", required=True, help="PDF file or directory.")
    parser.add_argument("--elements-output", required=True, help="Output path for elements NDJSON.")
    parser.add_argument("--chunks-output", required=True, help="Output path for structured chunks NDJSON.")

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

    return parser


def main() -> None:
    args = build_parser().parse_args()

    elements_output = Path(args.elements_output).expanduser().resolve()
    chunks_output = Path(args.chunks_output).expanduser().resolve()

    elements_output.parent.mkdir(parents=True, exist_ok=True)
    chunks_output.parent.mkdir(parents=True, exist_ok=True)

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

    print("\nDone.")
    print(f"Elements: {elements_output}")
    print(f"Chunks:   {chunks_output}")


if __name__ == "__main__":
    main()
