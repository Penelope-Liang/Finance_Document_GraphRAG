"""Source metadata shared by preprocessing, import and evidence formatting."""

from __future__ import annotations

import json
from typing import Any

PIPELINE_VERSION = "docling-json-v1"
IDENTITY_FIELDS = (
    "pipeline_version",
    "source_sha256",
    "artifact_id",
    "docling_version",
    "docling_core_version",
)


def unique(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result = []
    for value in values:
        key = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def merge_sources(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("provenance", "source_refs", "section_paths"):
        result[key] = unique([v for record in records for v in record.get(key, [])])
    result["pages"] = sorted(
        {
            p
            for record in records
            for p in record.get("pages", [])
            if isinstance(p, int) and not isinstance(p, bool)
        }
        | {
            p["page_no"]
            for p in result["provenance"]
            if isinstance(p.get("page_no"), int)
        }
    )
    for key in IDENTITY_FIELDS:
        values = {record[key] for record in records if record.get(key)}
        if len(values) > 1:
            raise ValueError(f"Cannot merge different {key} values in one chunk")
        if values:
            result[key] = next(iter(values))
    return result


def graph_source_properties(metadata: dict[str, Any]) -> dict[str, Any]:
    """Only Neo4j-property-safe scalars/flat arrays; rich structures are JSON."""
    props = {key: metadata.get(key) for key in IDENTITY_FIELDS}
    props.update(
        {
            "pages": list(metadata.get("pages") or []),
            "source_refs": list(metadata.get("source_refs") or []),
            "provenance_json": json.dumps(
                metadata.get("provenance") or [], ensure_ascii=False
            ),
            "section_paths_json": json.dumps(
                metadata.get("section_paths") or [], ensure_ascii=False
            ),
        }
    )
    return props


def evidence_sources(record: Any) -> dict[str, Any]:
    raw = record.get("provenance_json")
    try:
        provenance = json.loads(raw) if isinstance(raw, str) else []
    except (TypeError, ValueError):
        provenance = []
    if not isinstance(provenance, list):
        provenance = []
    try:
        section_paths = json.loads(record.get("section_paths_json") or "[]")
    except (TypeError, ValueError):
        section_paths = []
    pages = record.get("pages") or []
    if not pages and record.get("page") is not None:
        pages = [record.get("page")]
    return {
        "pages": pages,
        "source_refs": record.get("source_refs") or [],
        "provenance": provenance,
        "provenance_scope": record.get("provenance_scope"),
        "section_paths": section_paths if isinstance(section_paths, list) else [],
        **{key: record.get(key) for key in IDENTITY_FIELDS},
    }
