from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Optional


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
