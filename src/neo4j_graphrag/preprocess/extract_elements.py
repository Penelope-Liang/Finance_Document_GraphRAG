from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from neo4j_graphrag.preprocess.ingest_chunks import (
    ensure_unstructured_importable,
    iter_input_files,
    normalize_section_display,
)


def _metadata_value(metadata: Any, key: str) -> Any:
    if metadata is None:
        return None
    return getattr(metadata, key, None)


def _bbox_from_metadata(metadata: Any) -> list[list[float]] | None:
    coords = _metadata_value(metadata, "coordinates")
    points = getattr(coords, "points", None)
    if not points:
        return None
    out: list[list[float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        try:
            out.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            continue
    return out or None


def _element_record(*, doc_id: str, source_path: Path, element: Any, element_index: int) -> dict[str, Any]:
    metadata = getattr(element, "metadata", None)
    text = (getattr(element, "text", "") or "").strip()
    category = str(getattr(element, "category", "") or "UncategorizedText")
    page_number = _metadata_value(metadata, "page_number")
    try:
        page = int(page_number) if page_number is not None else None
    except (TypeError, ValueError):
        page = None

    section_hint = normalize_section_display(
        _metadata_value(metadata, "section_name")
        or _metadata_value(metadata, "section")
        or _metadata_value(metadata, "page_name")
        or (text if category == "Title" else "")
    )

    record = {
        "doc_id": doc_id,
        "source_path": str(source_path),
        "element_index": element_index,
        "category": category,
        "text": text,
        "page": page,
        "section_hint": section_hint,
        "bbox": _bbox_from_metadata(metadata),
        "metadata": {
            "filename": _metadata_value(metadata, "filename"),
            "filetype": _metadata_value(metadata, "filetype"),
            "page_number": page,
            "category_depth": _metadata_value(metadata, "category_depth"),
            "parent_id": _metadata_value(metadata, "parent_id"),
            "languages": _metadata_value(metadata, "languages"),
            "element_id": _metadata_value(metadata, "element_id"),
            "text_as_html": _metadata_value(metadata, "text_as_html"),
        },
    }
    return record


def _partition_pdf(path: Path, args: argparse.Namespace) -> list[Any]:
    ensure_unstructured_importable()
    from unstructured.partition.pdf import partition_pdf

    kwargs: dict[str, Any] = {
        "filename": str(path),
        "strategy": args.strategy,
        "infer_table_structure": args.infer_table_structure,
    }
    if args.extract_image_block_types:
        kwargs["extract_image_block_types"] = [x.strip() for x in args.extract_image_block_types.split(",") if x.strip()]
    if args.hi_res_model_name:
        kwargs["hi_res_model_name"] = args.hi_res_model_name
    if args.skip_infer_table_types:
        kwargs["skip_infer_table_types"] = [x.strip() for x in args.skip_infer_table_types.split(",") if x.strip()]
    return partition_pdf(**kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract PDF layout elements to NDJSON using Unstructured partition_pdf."
    )
    parser.add_argument("--input-path", required=True, help="PDF file or directory.")
    parser.add_argument("--output", required=True, help="Output elements NDJSON path.")
    parser.add_argument(
        "--strategy",
        default="hi_res",
        help="partition_pdf strategy. Use hi_res for layout-aware extraction.",
    )
    parser.add_argument(
        "--hi-res-model-name",
        default="yolox",
        help=(
            "High-resolution layout model name passed to partition_pdf. "
            "For local open-source Unstructured, prefer yolox or detectron2_onnx."
        ),
    )
    parser.add_argument(
        "--infer-table-structure",
        action="store_true",
        help="Ask Unstructured to preserve table structure when available.",
    )
    parser.add_argument(
        "--extract-image-block-types",
        default="",
        help="Optional comma-separated image block types for partition_pdf.",
    )
    parser.add_argument(
        "--skip-infer-table-types",
        default="",
        help="Optional comma-separated element types to skip for table inference.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on first file error.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        print(f"Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    files = [p for p in iter_input_files(input_path) if p.suffix.lower() == ".pdf"]
    if not files:
        print("No PDF files found.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_elements = 0
    error_files = 0

    with output_path.open("w", encoding="utf-8") as out:
        for file_path in files:
            total_files += 1
            rel_doc_id = (
                file_path.relative_to(input_path).as_posix()
                if input_path.is_dir()
                else file_path.name
            )
            try:
                elements = _partition_pdf(file_path, args)
            except Exception as exc:
                error_files += 1
                print(f"[error] {file_path}: {exc}", file=sys.stderr)
                if args.strict:
                    raise
                continue

            kept = 0
            for idx, element in enumerate(elements):
                text = (getattr(element, "text", "") or "").strip()
                if not text:
                    continue
                record = _element_record(
                    doc_id=rel_doc_id,
                    source_path=file_path,
                    element=element,
                    element_index=idx,
                )
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
                total_elements += 1

            print(f"[ok] {file_path} -> {kept} elements")

    print("\nDone.")
    print(f"Output: {output_path}")
    print(f"Files processed: {total_files}")
    print(f"Files errored: {error_files}")
    print(f"Total elements: {total_elements}")


if __name__ == "__main__":
    main()
