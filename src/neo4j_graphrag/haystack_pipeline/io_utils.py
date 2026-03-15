from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass
class ChunkRow:
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    section_name: str
    page: Optional[int]


def format_duration_minsec(seconds: float) -> str:
    total = max(0.0, float(seconds))
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = total % 60
    if hours > 0:
        return f"{hours}h {minutes}min {secs:.1f}s"
    return f"{minutes}min {secs:.1f}s"


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def build_context_preview(rows: list[dict[str, Any]], preview_items: int, preview_chars: int) -> str:
    content = "\n---\n".join(str(row.get("text_excerpt", "")) for row in rows[:preview_items])
    if len(content) <= preview_chars:
        return content
    return f"{content[:preview_chars].rstrip()}..."


def strip_json_fence(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```json\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^```\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_answer_and_evidence(raw: Any) -> tuple[str, str]:
    if raw is None:
        return ("", "")

    if isinstance(raw, dict):
        answer = raw.get("answer", raw.get("final_answer"))
        evidence = raw.get("evidence")
        return (normalize_value(answer), normalize_value(evidence))

    text = str(raw).strip()
    if not text:
        return ("", "")

    candidate = strip_json_fence(text)
    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            answer = payload.get("answer", payload.get("final_answer"))
            evidence = payload.get("evidence")
            return (normalize_value(answer), normalize_value(evidence))
    except Exception:
        pass
    return (text, "")


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        text = (
            value.get("evidence_text")
            or value.get("text")
            or value.get("quote")
            or value.get("reference")
        )
        if text is not None:
            return str(text).strip()
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.append(normalize_value(item))
        return "\n".join(x for x in parts if x)
    return json.dumps(value, ensure_ascii=False)


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def read_questions(
    *,
    questions_file: Optional[str],
    inline_questions: Optional[Iterable[str]],
) -> list[str]:
    rows: list[str] = []
    if questions_file:
        with open(questions_file, "r", encoding="utf-8") as f:
            rows.extend(
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            )
    if inline_questions:
        rows.extend(q.strip() for q in inline_questions if q and q.strip())
    return rows


def doc_key(doc_id: str) -> str:
    value = doc_id.strip().lower()
    if value.endswith(".pdf"):
        value = value[:-4]
    return value


def extract_docs_from_row(row: dict[str, Any]) -> set[str]:
    docs: set[str] = set()
    for key in ("doc_id", "doc_name", "form_id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            docs.add(value.strip())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    docs.add(item.strip())

    for key in ("doc_ids", "doc_names", "form_ids"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            docs.add(value.strip())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    docs.add(item.strip())

    refs = row.get("gold_reference") or row.get("gold_evidence") or row.get("evidence")
    if isinstance(refs, dict):
        doc_name = refs.get("doc_name")
        if isinstance(doc_name, str) and doc_name.strip():
            docs.add(doc_name.strip())
    elif isinstance(refs, list):
        for item in refs:
            if not isinstance(item, dict):
                continue
            doc_name = item.get("doc_name")
            if isinstance(doc_name, str) and doc_name.strip():
                docs.add(doc_name.strip())
    return docs


def load_question_doc_map(path: Optional[str]) -> dict[str, set[str]]:
    if not path:
        return {}
    rows = read_json_or_jsonl(Path(path))
    out: dict[str, set[str]] = {}
    for row in rows:
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        docs = extract_docs_from_row(row)
        if not docs:
            continue
        out.setdefault(question, set()).update(docs)
    return out


def read_chunks_ndjson(path: Path, *, strict: bool = True) -> list[ChunkRow]:
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
                    raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
                continue

            chunk_id = payload.get("chunk_id")
            doc_id = payload.get("doc_id")
            text = payload.get("text")
            chunk_index = payload.get("chunk_index", 0)
            section_name = payload.get("section_name") or "UNSPECIFIED"
            page = payload.get("page")
            if page is None:
                page = payload.get("page_number")

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                if strict:
                    raise ValueError(f"Missing/invalid chunk_id at {path}:{line_no}")
                continue
            if not isinstance(doc_id, str) or not doc_id.strip():
                if strict:
                    raise ValueError(f"Missing/invalid doc_id at {path}:{line_no}")
                continue
            if not isinstance(text, str) or not text.strip():
                if strict:
                    raise ValueError(f"Missing/invalid text at {path}:{line_no}")
                continue

            try:
                chunk_index_int = int(chunk_index)
            except (TypeError, ValueError):
                chunk_index_int = 0

            page_int: Optional[int] = None
            if page is not None:
                try:
                    page_int = int(page)
                except (TypeError, ValueError):
                    page_int = None

            rows.append(
                ChunkRow(
                    chunk_id=chunk_id.strip(),
                    doc_id=doc_id.strip(),
                    text=text,
                    chunk_index=chunk_index_int,
                    section_name=str(section_name).strip() or "UNSPECIFIED",
                    page=page_int,
                )
            )
    return rows


def save_results(
    *,
    output_path: str,
    output_format: str,
    rows: list[dict[str, Any]],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    if output_format == "jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    headers = [
        "mode",
        "retrieval",
        "question_id",
        "gold_type",
        "question",
        "answer",
        "evidence",
        "latency",
        "latency_seconds",
        "retriever",
        "retrieval_status",
        "retrieved_chunk_count",
        "retrieved_item_count",
        "retrieved_chunk_ids",
        "retrieved_evidence",
        "doc_filter",
        "embedding_model",
        "reranker_model",
        "generator_model",
        "context_preview",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            csv_row = row.copy()
            csv_row["retrieved_chunk_ids"] = "|".join(row.get("retrieved_chunk_ids", []))
            csv_row["retrieved_evidence"] = json.dumps(
                row.get("retrieved_evidence", []), ensure_ascii=False
            )
            csv_row["doc_filter"] = "|".join(row.get("doc_filter", []))
            writer.writerow(csv_row)
