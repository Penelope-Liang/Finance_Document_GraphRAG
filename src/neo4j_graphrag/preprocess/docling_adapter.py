from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from importlib.metadata import version
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from neo4j_graphrag.preprocess.docling_json import artifact_hash, docling_json_to_clean_document
from neo4j_graphrag.preprocess.provenance import PIPELINE_VERSION
from neo4j_graphrag.preprocess.chunk_utils import iter_input_files
from neo4j_graphrag.preprocess.export.to_elements_ndjson import clean_document_to_element_records


def _build_converter(*, do_ocr: bool = True, device: str = "cpu", num_threads: int = 4) -> Any:
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.accelerator_options import AcceleratorOptions
    except ImportError as exc:
        raise RuntimeError("Install requirements/docling.txt in the parsing environment.") from exc
    return DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=PdfPipelineOptions(do_ocr=do_ocr, accelerator_options=AcceleratorOptions(device=device, num_threads=num_threads)))
    })


def _convert_pdf(path: Path, *, converter: Any = None, page_range: tuple[int, int] | None = None) -> Any:
    converter = converter or _build_converter()
    result = converter.convert(str(path), **({"page_range": page_range} if page_range else {}))
    if str(getattr(result.status, "value", result.status)) != "success":
        raise RuntimeError(f"Docling conversion incomplete: {result.status}")
    return result.document


def _iter_pdfs(path: Path) -> list[Path]:
    return [p for p in iter_input_files(path) if p.suffix.lower() == ".pdf"]


def _write_ndjson(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert PDFs with Docling into CleanDocument JSONL and optional elements.ndjson export."
    )
    parser.add_argument("--input-path", required=True, help="PDF or native Docling JSON file/directory.")
    parser.add_argument("--input-format", choices=["pdf", "json"], default="pdf")
    parser.add_argument("--raw-output-dir", help="Original JSON and parse-info directory (default: alongside clean output).")
    parser.add_argument("--device", choices=["cpu", "auto", "mps", "cuda"], default="cpu")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR for born-digital PDFs.")
    parser.add_argument("--page-range", nargs=2, type=int, metavar=("FIRST", "LAST"), help="Optional 1-based inclusive sample range; recorded in parse info.")
    parser.add_argument("--clean-output", required=True, help="Output path for clean_document JSONL.")
    parser.add_argument(
        "--elements-output",
        default="",
        help="Optional output path for elements.ndjson-compatible export.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail fast on first file error.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        print(f"Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.input_format == "json":
        files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*.docling.json"))
    else:
        files = _iter_pdfs(input_path)
    if args.page_range and (args.page_range[0] < 1 or args.page_range[1] < args.page_range[0]):
        raise ValueError("page-range must be positive and ordered")
    if not files:
        print("No input files found.", file=sys.stderr)
        sys.exit(1)

    clean_output = Path(args.clean_output).expanduser().resolve()
    elements_output = Path(args.elements_output).expanduser().resolve() if args.elements_output else None

    raw_dir = Path(args.raw_output_dir) if args.raw_output_dir else clean_output.parent / "docling_raw"
    converter = _build_converter(do_ocr=not args.no_ocr, device=args.device, num_threads=args.num_threads) if args.input_format == "pdf" else None
    clean_rows: list[dict[str, Any]] = []
    element_rows: list[dict[str, Any]] = []
    processed = 0
    errored = 0

    for file_path in files:
        rel_doc_id = file_path.relative_to(input_path).as_posix() if input_path.is_dir() else file_path.name
        try:
            if args.input_format == "pdf":
                doc = _convert_pdf(file_path, converter=converter, page_range=tuple(args.page_range) if args.page_range else None)
                raw = doc.export_to_dict()
                info = {
                    "doc_id": rel_doc_id, "source_path": str(file_path),
                    "source_sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
                    "artifact_id": artifact_hash(raw), "pipeline_version": PIPELINE_VERSION,
                    "docling_version": version("docling"), "docling_core_version": version("docling-core"),
                    "parsed_at": datetime.now(timezone.utc).isoformat(),
                    "page_range": args.page_range, "do_ocr": not args.no_ocr,
                    "device": args.device, "num_threads": args.num_threads,
                    "docling_ibm_models_version": version("docling-ibm-models"),
                    "docling_parse_version": version("docling-parse"),
                }
                # Content-addressed names avoid collisions and preserve earlier parses.
                basename = f"{file_path.name}.{info['artifact_id'][:16]}"
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / f"{basename}.docling.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                (raw_dir / f"{basename}.parse_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
                clean_doc = docling_json_to_clean_document(document=raw, doc_id=rel_doc_id, source_path=str(file_path), parse_info=info)
            else:
                raw = json.loads(file_path.read_text(encoding="utf-8"))
                info_path = file_path.with_name(file_path.name.removesuffix(".docling.json") + ".parse_info.json")
                if not info_path.exists():
                    raise ValueError(f"Replay requires parse-info file: {info_path}")
                info = json.loads(info_path.read_text(encoding="utf-8"))
                clean_doc = docling_json_to_clean_document(document=raw, doc_id=info["doc_id"], source_path=info["source_path"], parse_info=info)
            clean_rows.append(clean_doc.to_dict())
            if elements_output is not None:
                element_rows.extend(clean_document_to_element_records(clean_doc))
            processed += 1
            print(f"[ok] {file_path} -> {len(clean_doc.blocks)} blocks")
        except Exception as exc:
            errored += 1
            print(f"[error] {file_path}: {exc}", file=sys.stderr)
            if args.strict:
                raise

    _write_ndjson(clean_output, clean_rows)
    if elements_output is not None:
        _write_ndjson(elements_output, element_rows)

    print("\nDone.")
    print(f"Clean output: {clean_output}")
    if elements_output is not None:
        print(f"Elements output: {elements_output}")
    print(f"Files processed: {processed}")
    print(f"Files errored: {errored}")


if __name__ == "__main__":
    main()
