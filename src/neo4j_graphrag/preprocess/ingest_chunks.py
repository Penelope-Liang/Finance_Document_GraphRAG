from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import sys


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


SUPPORTED_STRUCTURED_EXTS = {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xls", ".xlsx"}
SUPPORTED_UNSTRUCTURED_EXTS = {
    ".pdf",
    ".docx",
    ".html",
    ".htm",
    ".eml",
    ".msg",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".gif",
    ".webp",
}
SUPPORTED_TEXT_EXTS = {".txt", ".md"}


@dataclass
class ParsedChunk:
    text: str
    section_name: str
    page: Optional[int]
    metadata: dict[str, Any]
    chunk_engine: str
    parser_engine: str


ITEM_10K_HEADER_RE = re.compile(
    r"(?i)^\s*(?:part\s+[ivx]+\s+)?item\s+"
    r"(\d+(?:\.\d+)?[a-c]?)"
    r"\s*[\.\-:\)]?\s*(.*)$"
)


# ---------------------------------------------------------------------------
# FakeElement: wraps a PyMuPDF page so it looks like an unstructured element.
# Only parse_and_chunk_item10k needs .text and .metadata.page_number.
# ---------------------------------------------------------------------------
class _FakeMeta:
    def __init__(self, page_number: int, filename: str) -> None:
        self.page_number = page_number
        self.filename = filename
        self.filetype = "application/pdf"


class FakeElement:
    def __init__(self, text: str, page_number: int, filename: str = "") -> None:
        self.text = text
        self.category = "NarrativeText"
        self.metadata = _FakeMeta(page_number=page_number, filename=filename)


def normalize_for_id(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def normalize_section_display(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_chunk_id(
    *,
    version: str,
    doc_id: str,
    section_name: str,
    chunk_index: int,
    text: str,
) -> str:
    section_name_norm = normalize_for_id(section_name)
    normalized_text = normalize_for_id(text)
    payload = f"{version}|{doc_id}|{section_name_norm}|{chunk_index}|{normalized_text}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files: list[Path] = []
    for path in input_path.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            files.append(path)
    return sorted(files)


def flatten_json(obj: Any, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(flatten_json(value, next_prefix))
        return lines
    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]"
            lines.extend(flatten_json(value, next_prefix))
        return lines
    if obj is None:
        return lines
    text = str(obj)
    if text.strip():
        lines.append(f"{prefix}: {text}" if prefix else text)
    return lines


def parse_csv_like(path: Path, delimiter: str) -> str:
    rows: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames:
            for row_idx, row in enumerate(reader):
                pairs = [
                    f"{k}={v}"
                    for k, v in row.items()
                    if k is not None and v is not None and str(v).strip()
                ]
                if pairs:
                    rows.append(f"row[{row_idx}]: " + " | ".join(pairs))
        else:
            f.seek(0)
            raw_reader = csv.reader(f, delimiter=delimiter)
            for row_idx, row in enumerate(raw_reader):
                values = [cell for cell in row if cell.strip()]
                if values:
                    rows.append(f"row[{row_idx}]: " + " | ".join(values))
    return "\n".join(rows)


def parse_json_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    return "\n".join(flatten_json(data))


def parse_jsonl_file(path: Path) -> str:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            flattened = flatten_json(record)
            if flattened:
                lines.append(f"line[{idx}]")
                lines.extend(flattened)
    return "\n".join(lines)


def parse_excel_file(path: Path) -> str:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Excel parsing requires pandas/openpyxl. Install and retry."
        ) from exc

    sheets = pd.read_excel(path, sheet_name=None)
    lines: list[str] = []
    for sheet_name, df in sheets.items():
        lines.append(f"sheet: {sheet_name}")
        for row_idx, row in enumerate(df.fillna("").astype(str).to_dict(orient="records")):
            pairs = [f"{k}={v}" for k, v in row.items() if str(v).strip()]
            if pairs:
                lines.append(f"row[{row_idx}]: " + " | ".join(pairs))
    return "\n".join(lines)


def parse_native_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".csv":
        return parse_csv_like(path, ",")
    if ext == ".tsv":
        return parse_csv_like(path, "\t")
    if ext == ".json":
        return parse_json_file(path)
    if ext in {".jsonl", ".ndjson"}:
        return parse_jsonl_file(path)
    if ext in {".xls", ".xlsx"}:
        return parse_excel_file(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def split_fixed_size(
    text: str, chunk_size: int, chunk_overlap: int, approximate: bool = True
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    chunks: list[str] = []
    text_length = len(text)
    step = chunk_size - chunk_overlap

    for start in range(0, text_length, step):
        end = min(start + chunk_size, text_length)
        if approximate and end < text_length:
            candidate = end
            while candidate > start and not text[candidate - 1].isspace():
                candidate -= 1
            if candidate > start:
                end = candidate

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def build_native_splitter(args: argparse.Namespace) -> Callable[[str], list[str]]:
    splitter_name = args.native_splitter
    if splitter_name == "fixed":
        def _fixed_split(text: str) -> list[str]:
            return split_fixed_size(
                text=text,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                approximate=not args.disable_approximate,
            )

        return _fixed_split

    if splitter_name == "langchain":
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter, SpacyTextSplitter
        except ImportError as exc:
            raise RuntimeError(
                "LangChain splitter selected but dependencies are missing. "
                "Install langchain-text-splitters."
            ) from exc

        if args.langchain_mode == "spacy":
            lc_splitter = SpacyTextSplitter(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                pipeline=args.spacy_pipeline,
            )
        else:
            lc_splitter = RecursiveCharacterTextSplitter(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )

        def _langchain_split(text: str) -> list[str]:
            return [chunk for chunk in lc_splitter.split_text(text) if chunk.strip()]

        return _langchain_split

    if splitter_name == "llamaindex":
        try:
            from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
        except ImportError as exc:
            raise RuntimeError(
                "LlamaIndex splitter selected but dependencies are missing. "
                "Install llama-index-core."
            ) from exc

        if args.llamaindex_mode == "token":
            li_splitter = TokenTextSplitter(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        else:
            li_splitter = SentenceSplitter(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )

        def _llama_split(text: str) -> list[str]:
            return [chunk for chunk in li_splitter.split_text(text) if chunk.strip()]

        return _llama_split

    raise ValueError(f"Unsupported native splitter: {splitter_name}")


def ensure_unstructured_importable() -> None:
    try:
        import unstructured.chunking  # noqa: F401
        import unstructured.partition.auto  # noqa: F401
    except ImportError:
        local_candidates = [
            REPO_ROOT / "third_party_unstructured_src",
            REPO_ROOT / "unstructured",  # backward compatibility
        ]
        for local_unstructured_repo in local_candidates:
            if local_unstructured_repo.exists() and str(local_unstructured_repo) not in sys.path:
                sys.path.insert(0, str(local_unstructured_repo))
                break
        for module_name in list(sys.modules.keys()):
            if module_name == "unstructured" or module_name.startswith("unstructured."):
                sys.modules.pop(module_name, None)
        import unstructured.chunking  # noqa: F401
        import unstructured.partition.auto  # noqa: F401


def extract_section_name_from_element(element: Any) -> str:
    metadata = getattr(element, "metadata", None)
    if metadata is not None:
        for key in ("section_name", "section", "page_name"):
            value = getattr(metadata, key, None)
            cleaned = normalize_section_display(value)
            if cleaned:
                return cleaned
        orig_elements = getattr(metadata, "orig_elements", None) or []
        for item in orig_elements:
            if getattr(item, "category", "") == "Title":
                cleaned = normalize_section_display(getattr(item, "text", ""))
                if cleaned:
                    return cleaned
    if getattr(element, "category", "") == "Title":
        return normalize_section_display(getattr(element, "text", ""))
    return ""


def metadata_subset(element: Any) -> dict[str, Any]:
    metadata = getattr(element, "metadata", None)
    result: dict[str, Any] = {
        "element_category": getattr(element, "category", None),
    }
    if metadata is None:
        return result
    for key in (
        "filename",
        "filetype",
        "page_number",
        "page_name",
        "category_depth",
        "parent_id",
        "languages",
        "url",
        "last_modified",
    ):
        value = getattr(metadata, key, None)
        if value is not None:
            result[key] = value
    return result


def _first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ""


def _item10k_section_from_text(text: str) -> tuple[Optional[str], Optional[str]]:
    first_line = _first_non_empty_line(text)
    if not first_line:
        return None, None
    match = ITEM_10K_HEADER_RE.match(first_line)
    if not match:
        return None, None
    item_number = match.group(1).upper()
    title = normalize_section_display(match.group(2) or "")
    return f"ITEM {item_number}", title or None


def _split_section_text_for_item10k(
    text: str,
    *,
    max_characters: int,
    new_after_n_chars: int,
    overlap: int,
) -> list[str]:
    target_size = max(1, min(max_characters, new_after_n_chars))
    safe_overlap = max(0, min(overlap, target_size - 1))
    return split_fixed_size(
        text=text,
        chunk_size=target_size,
        chunk_overlap=safe_overlap,
        approximate=True,
    )


def parse_and_chunk_item10k(elements: list[Any], args: argparse.Namespace) -> list[ParsedChunk]:
    grouped_sections: list[dict[str, Any]] = []
    current_section = "UNSPECIFIED"
    current_title: Optional[str] = None
    buffer: list[str] = []
    page: Optional[int] = None
    base_metadata: dict[str, Any] = {}
    found_item_header = False

    def flush() -> None:
        nonlocal buffer, page, base_metadata, current_title
        text = "\n\n".join(part.strip() for part in buffer if part and part.strip()).strip()
        if not text:
            buffer = []
            page = None
            base_metadata = {}
            return
        grouped_sections.append(
            {
                "section_name": current_section,
                "section_title": current_title,
                "text": text,
                "page": page,
                "metadata": dict(base_metadata),
            }
        )
        buffer = []
        page = None
        base_metadata = {}

    for element in elements:
        element_text = (getattr(element, "text", "") or "").strip()
        if not element_text:
            continue
        detected_section, detected_title = _item10k_section_from_text(element_text)
        if detected_section:
            found_item_header = True
            flush()
            current_section = detected_section
            current_title = detected_title
        if page is None:
            candidate_page = getattr(getattr(element, "metadata", None), "page_number", None)
            page = candidate_page if isinstance(candidate_page, int) else None
        if not base_metadata:
            base_metadata = metadata_subset(element)
            if current_title:
                base_metadata["item_title"] = current_title
        buffer.append(element_text)
    flush()

    if not grouped_sections or not found_item_header:
        return []

    parsed_chunks: list[ParsedChunk] = []
    for section in grouped_sections:
        chunks = _split_section_text_for_item10k(
            section["text"],
            max_characters=args.unstructured_max_characters,
            new_after_n_chars=args.unstructured_new_after_n_chars,
            overlap=args.unstructured_overlap,
        )
        for chunk_text in chunks:
            if not chunk_text.strip():
                continue
            parsed_chunks.append(
                ParsedChunk(
                    text=chunk_text,
                    section_name=section["section_name"],
                    page=section["page"],
                    metadata=section["metadata"],
                    chunk_engine="item10k",
                    parser_engine="pymupdf",  # changed from "unstructured"
                )
            )
    return parsed_chunks


def route_type_for_ext(ext: str) -> Optional[str]:
    if ext in SUPPORTED_STRUCTURED_EXTS:
        return "native"
    if ext in SUPPORTED_UNSTRUCTURED_EXTS:
        return "unstructured"
    if ext in SUPPORTED_TEXT_EXTS:
        return "native"
    return None


def parse_and_chunk_native(
    *,
    path: Path,
    splitter_fn: Callable[[str], list[str]],
    native_splitter_name: str,
) -> list[ParsedChunk]:
    raw_text = parse_native_text(path)
    chunks = splitter_fn(raw_text)
    return [
        ParsedChunk(
            text=chunk_text,
            section_name="",
            page=None,
            metadata={"source_path": str(path)},
            chunk_engine=native_splitter_name,
            parser_engine="native",
        )
        for chunk_text in chunks
    ]


def parse_and_chunk_unstructured(path: Path, args: argparse.Namespace) -> list[ParsedChunk]:
    # --- CHANGED: use PyMuPDFLoader instead of unstructured partition ---
    # PyMuPDFLoader correctly handles tables (converts to text with structure)
    # and does not require tesseract OCR for text-based PDFs.
    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            from langchain_community.document_loaders import PyMuPDFLoader
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF loader not available. Install with: "
                "pip install langchain-community pymupdf"
            ) from exc

        pages = PyMuPDFLoader(str(path)).load()
        # Split each page into lines so _item10k_section_from_text can detect
        # Item headers on any line, not just the first line of a page.
        elements: list[Any] = []
        for p in pages:
            page_num = int(p.metadata.get("page", 0)) + 1
            for line in p.page_content.splitlines():
                if line.strip():
                    elements.append(FakeElement(
                        text=line.strip(),
                        page_number=page_num,
                        filename=path.name,
                    ))
    else:
        # Non-PDF unstructured files: fall back to original unstructured partition
        ensure_unstructured_importable()
        from unstructured.partition.auto import partition
        elements = partition(filename=str(path))

    if not elements:
        return []

    strategy = args.unstructured_chunking

    # item10k strategy: group by Item headers then split
    if strategy == "item10k":
        item10k_chunks = parse_and_chunk_item10k(elements, args)
        if item10k_chunks:
            return item10k_chunks
        if args.unstructured_fallback_basic:
            strategy = "basic"
        else:
            return []

    # by_title / basic strategy for non-PDF or fallback
    # For PDFs we already have FakeElements so we do a simple fixed split
    if ext == ".pdf":
        # Flatten all page text and split with fixed size
        full_text = "\n\n".join(
            (getattr(e, "text", "") or "").strip()
            for e in elements
            if (getattr(e, "text", "") or "").strip()
        )
        chunks_text = split_fixed_size(
            full_text,
            chunk_size=args.unstructured_max_characters,
            chunk_overlap=args.unstructured_overlap,
            approximate=True,
        )
        return [
            ParsedChunk(
                text=chunk_text,
                section_name="",
                page=None,
                metadata={"filename": path.name},
                chunk_engine="fixed",
                parser_engine="pymupdf",
            )
            for chunk_text in chunks_text
            if chunk_text.strip()
        ]

    # Original unstructured by_title/basic path for non-PDF files
    ensure_unstructured_importable()
    from unstructured.chunking.dispatch import chunk as chunk_dispatch

    has_title = any(
        getattr(element, "category", "") == "Title"
        and normalize_section_display(getattr(element, "text", ""))
        for element in elements
    )

    if strategy == "by_title" and not has_title and args.unstructured_fallback_basic:
        strategy = "basic"

    try:
        chunked_elements = chunk_dispatch(
            elements,
            strategy,
            max_characters=args.unstructured_max_characters,
            new_after_n_chars=args.unstructured_new_after_n_chars,
            overlap=args.unstructured_overlap,
        )
    except Exception:
        if strategy == "by_title" and args.unstructured_fallback_basic:
            chunked_elements = chunk_dispatch(
                elements,
                "basic",
                max_characters=args.unstructured_max_characters,
                new_after_n_chars=args.unstructured_new_after_n_chars,
                overlap=args.unstructured_overlap,
            )
            strategy = "basic"
        else:
            raise

    parsed_chunks: list[ParsedChunk] = []
    for element in chunked_elements:
        text = getattr(element, "text", "") or ""
        if not text.strip():
            continue
        page = getattr(getattr(element, "metadata", None), "page_number", None)
        parsed_chunks.append(
            ParsedChunk(
                text=text,
                section_name=extract_section_name_from_element(element),
                page=page if isinstance(page, int) else None,
                metadata=metadata_subset(element),
                chunk_engine=strategy,
                parser_engine="unstructured",
            )
        )
    return parsed_chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Route files by type and generate unified chunks.ndjson with stable chunk_id "
            "(v1 hashing)."
        )
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="File or directory to ingest.",
    )
    parser.add_argument(
        "--output",
        default="chunks.ndjson",
        help="Output NDJSON path.",
    )
    parser.add_argument(
        "--chunk-id-version",
        default="v1",
        help="Version prefix embedded in chunk_id hash payload.",
    )
    parser.add_argument(
        "--native-splitter",
        choices=["fixed", "langchain", "llamaindex"],
        default="fixed",
        help="Splitter engine for native route.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size for native splitters.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap for native splitters.",
    )
    parser.add_argument(
        "--disable-approximate",
        action="store_true",
        help="Disable approximate word-boundary behavior in FixedSizeSplitter.",
    )
    parser.add_argument(
        "--langchain-mode",
        choices=["recursive", "spacy"],
        default="recursive",
        help="LangChain splitter type when --native-splitter=langchain.",
    )
    parser.add_argument(
        "--spacy-pipeline",
        default="en_core_web_sm",
        help="SpaCy pipeline name for LangChain SpacyTextSplitter.",
    )
    parser.add_argument(
        "--llamaindex-mode",
        choices=["sentence", "token"],
        default="sentence",
        help="LlamaIndex splitter type when --native-splitter=llamaindex.",
    )
    parser.add_argument(
        "--unstructured-chunking",
        choices=["by_title", "basic", "item10k"],
        default="by_title",
        help="Default unstructured chunking strategy.",
    )
    parser.add_argument(
        "--no-unstructured-fallback-basic",
        action="store_false",
        dest="unstructured_fallback_basic",
        help="Disable fallback to basic when by_title has no title structure or errors.",
    )
    parser.add_argument(
        "--unstructured-max-characters",
        type=int,
        default=900,
        help="max_characters for unstructured chunking.",
    )
    parser.add_argument(
        "--unstructured-new-after-n-chars",
        type=int,
        default=750,
        help="new_after_n_chars for unstructured chunking.",
    )
    parser.add_argument(
        "--unstructured-overlap",
        type=int,
        default=80,
        help="overlap for unstructured chunking.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on first file error.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        print(f"Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    files = iter_input_files(input_path)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    native_splitter_fn = build_native_splitter(args)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_chunks = 0
    skipped_files = 0
    error_files = 0

    with output_path.open("w", encoding="utf-8") as out:
        for file_path in files:
            total_files += 1
            ext = file_path.suffix.lower()
            route_type = route_type_for_ext(ext)
            if route_type is None:
                skipped_files += 1
                print(f"[skip] unsupported extension: {file_path}")
                continue

            rel_path = (
                file_path.relative_to(input_path).as_posix()
                if input_path.is_dir()
                else file_path.name
            )
            doc_id = rel_path

            try:
                if route_type == "native":
                    parsed_chunks = parse_and_chunk_native(
                        path=file_path,
                        splitter_fn=native_splitter_fn,
                        native_splitter_name=args.native_splitter,
                    )
                else:
                    parsed_chunks = parse_and_chunk_unstructured(file_path, args)
            except Exception as exc:
                error_files += 1
                print(f"[error] {file_path}: {exc}", file=sys.stderr)
                if args.strict:
                    raise
                continue

            for idx, parsed_chunk in enumerate(parsed_chunks):
                section_name = normalize_section_display(parsed_chunk.section_name)
                chunk_id = compute_chunk_id(
                    version=args.chunk_id_version,
                    doc_id=doc_id,
                    section_name=section_name,
                    chunk_index=idx,
                    text=parsed_chunk.text,
                )
                record = {
                    "chunk_id": chunk_id,
                    "chunk_id_version": args.chunk_id_version,
                    "doc_id": doc_id,
                    "text": parsed_chunk.text,
                    "chunk_index": idx,
                    "section_name": section_name,
                    "page": parsed_chunk.page,
                    "source_type": ext.lstrip("."),
                    "parser_engine": parsed_chunk.parser_engine,
                    "chunk_engine": parsed_chunk.chunk_engine,
                    "metadata": parsed_chunk.metadata,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

            print(
                f"[ok] {file_path} -> {len(parsed_chunks)} chunks "
                f"(route={route_type})"
            )

    print("\nDone.")
    print(f"Output: {output_path}")
    print(f"Files processed: {total_files}")
    print(f"Files skipped: {skipped_files}")
    print(f"Files errored: {error_files}")
    print(f"Total chunks: {total_chunks}")


if __name__ == "__main__":
    main()