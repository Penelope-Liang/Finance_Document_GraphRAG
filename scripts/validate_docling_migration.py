"""Offline audit of native Docling JSON against PDF-checked financial cells.

Run with PYTHONPATH=src. Does not connect to Neo4j or call an LLM.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from neo4j_graphrag.preprocess.docling_json import docling_json_to_clean_document
from neo4j_graphrag.preprocess.chunk_from_elements import _read_elements, build_chunks
from neo4j_graphrag.preprocess.export.to_elements_ndjson import (
    clean_document_to_element_records,
)
from neo4j_graphrag.preprocess.table_graph_prep import prepare_element
from neo4j_graphrag.tools.build_graph import _read_chunks
from neo4j_graphrag.tools.import_tables import _read_tables, _build_table_payload


def process(doc, output: Path):
    elements = output.with_suffix(".elements.ndjson")
    elements.write_text(
        "\n".join(json.dumps(prepare_element(x)) for x in clean_document_to_element_records(doc)),
        encoding="utf-8",
    )
    chunks = build_chunks(_read_elements(elements))
    cells = []
    for row in chunks:
        m = row["metadata"]
        if m.get("is_table") or m.get("table_data") is not None:
            cells.extend(
                {**c, "doc_id": doc.doc_id, "page": row["page"]}
                for c in m["table_structured"]["numeric_cells"]
            )
    output.write_text("\n".join(json.dumps(x) for x in chunks), encoding="utf-8")
    return chunks, cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-dir", type=Path, default=Path("output/docling_migration/samples")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/docling_migration/audit")
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=Path("tests/fixtures/docling/expected_evidence.json"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_cells = []
    for file in sorted(args.samples_dir.glob("*.docling.json")):
        stem = file.name.removesuffix(".docling.json")
        info = json.loads((args.samples_dir / (stem + ".parse_info.json")).read_text())
        raw = json.loads(file.read_text())
        doc = docling_json_to_clean_document(
            document=raw,
            doc_id=info["doc_id"],
            source_path=info["source_path"],
            parse_info=info,
        )
        path = args.output_dir / (stem + ".json.chunks.ndjson")
        chunks, cells = process(doc, path)
        all_cells.extend(cells)
        imports = _read_chunks(path, strict=True)
        table_payloads = [
            _build_table_payload(r, cell_mode="numeric")
            for rows in _read_tables(path, strict=True).values()
            for r in rows
        ]
        summary = {
            "doc_id": doc.doc_id,
            "page_range": info.get("page_range"),
            "blocks": len(doc.blocks),
            "chunks": len(chunks),
            "tables": len(table_payloads),
            "numeric_cells": len(cells),
            "import_chunks": sum(map(len, imports.values())),
            "cells_with_provenance": sum(bool(c.get("provenance")) for c in cells),
            "section_names": sorted({b.section for b in doc.blocks}),
        }
        summaries.append(summary)
    if not summaries:
        raise SystemExit("No sample JSON files found")
    checks = []
    for expected in json.loads(args.expected.read_text()):
        matches = [
            c
            for c in all_cells
            if c["doc_id"] == expected["doc_id"]
            and c["page"] == expected["page"]
            and c["row_header"] == expected["row_header"]
            and all(p in c["col_header"] for p in expected["period"])
        ]
        passed = len(matches) == 1
        if passed:
            actual = matches[0]
            passed = (
                all(
                    math.isclose(actual[k], expected[k], rel_tol=1e-10, abs_tol=1e-10)
                    for k in ("numeric_value", "normalized_value")
                )
                and all(actual[k] == expected[k] for k in ("scale", "numeric_kind"))
                and bool(actual.get("provenance"))
            )
        checks.append(
            {
                "expected": expected,
                "passed": passed,
                "matches": [
                    {
                        k: c.get(k)
                        for k in (
                            "row_header",
                            "col_header",
                            "numeric_value",
                            "normalized_value",
                            "numeric_kind",
                            "scale",
                        )
                    }
                    for c in matches
                ],
            }
        )
    report = {
        "documents": summaries,
        "golden_checks": checks,
        "passed": all(c["passed"] for c in checks),
        "neo4j_live_test": False,
        "scope": "Selected real PDF page ranges, not complete filings; values checked against rendered PDF pages.",
    }
    target = args.output_dir / "report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "report": str(target),
                "documents": len(summaries),
                "golden_passed": sum(c["passed"] for c in checks),
                "golden_total": len(checks),
                "passed": report["passed"],
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
