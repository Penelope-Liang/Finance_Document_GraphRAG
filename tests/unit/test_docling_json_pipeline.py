from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from finance_document_qa.pipeline.formatting import (
    _format_chunk_record,
    _format_table_record,
)
from neo4j_graphrag.preprocess.docling_json import (
    artifact_hash,
    docling_json_to_clean_document,
)
from neo4j_graphrag.preprocess.docling_adapter import (
    main as parse_main,
)
from neo4j_graphrag.preprocess.export.to_elements_ndjson import (
    clean_document_to_element_records,
)
from neo4j_graphrag.preprocess.chunk_from_elements import _read_elements, build_chunks
from neo4j_graphrag.preprocess.table_graph_prep import _build_table_structured, prepare_element
from neo4j_graphrag.tools.build_graph import (
    _read_chunks,
    _import_doc_tx,
    _doc_fingerprint,
)
from neo4j_graphrag.tools.import_tables import (
    _read_tables,
    _build_table_payload,
    _import_doc_tables_tx,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "docling"


def load_apple():
    return json.loads((FIXTURES / "APPLE_2018_10K.pdf.page41.json").read_text())


def adapt(raw, doc_id="APPLE_2018_10K.pdf"):
    return docling_json_to_clean_document(
        document=raw,
        doc_id=doc_id,
        source_path=doc_id,
        parse_info={
            "source_sha256": "source-hash",
            "docling_version": "2.73.1",
            "docling_core_version": "2.101.1",
        },
    )


def pipeline(doc, tmp_path):
    elements = tmp_path / "elements.ndjson"
    elements.write_text(
        "\n".join(json.dumps(prepare_element(x)) for x in clean_document_to_element_records(doc))
    )
    chunks = build_chunks(_read_elements(elements))
    path = tmp_path / "tables.ndjson"
    path.write_text("\n".join(json.dumps(x) for x in chunks))
    return chunks, path


def test_actual_json_reading_order_and_furniture():
    raw = load_apple()
    doc = adapt(raw)
    assert doc.blocks[0].text == "Apple Inc."
    assert doc.blocks[1].text == "CONSOLIDATED STATEMENTS OF OPERATIONS"
    assert not any(b.attrs["docling_label"] == "page_footer" for b in doc.blocks)
    table = next(b for b in doc.blocks if b.type == "table")
    assert table.page == 41
    assert "html" not in table.to_dict()
    assert table.section == "CONSOLIDATED STATEMENTS OF OPERATIONS"
    assert "In millions" in table.attrs["table_context"][0]["text"]
    prov = table.provenance[0]
    assert prov["raw_bbox"]["coord_origin"] == "BOTTOMLEFT"
    assert prov["bbox"]["coord_origin"] == "TOPLEFT"
    assert prov["bbox"]["t"] == pytest.approx(
        prov["page_size"]["height"] - prov["raw_bbox"]["t"]
    )
    assert all(c["provenance_scope"] == "cell" for c in table.table_data["table_cells"])
    # Follow body references, not the order in the texts array.
    raw["body"]["children"][0:2] = reversed(raw["body"]["children"][0:2])
    assert adapt(raw).blocks[0].text == "CONSOLIDATED STATEMENTS OF OPERATIONS"


def test_text_boundaries_and_multiple_provenance_are_preserved(tmp_path):
    raw = load_apple()
    text_indices = [i for i, x in enumerate(raw["texts"]) if x["label"] == "text"]
    first, second = [raw["texts"][i] for i in text_indices[:2]]
    first["text"] = "First paragraph.\nStill the same element."
    second["text"] = "Second paragraph."
    first["prov"].append({**deepcopy(first["prov"][0]), "page_no": 42})
    raw["pages"]["42"] = deepcopy(raw["pages"]["41"])
    raw["pages"]["42"]["page_no"] = 42
    raw["body"]["children"] = [
        {"$ref": first["self_ref"]},
        {"$ref": second["self_ref"]},
    ]
    doc = adapt(raw)
    assert len(doc.blocks) == 2
    assert len(doc.blocks[0].provenance) == 2
    chunks, _ = pipeline(doc, tmp_path)
    assert len(chunks) == 1
    assert "element.\n\nSecond paragraph" in chunks[0]["text"]
    metadata = chunks[0]["metadata"]
    assert metadata["pages"] == [41, 42]
    assert len(metadata["source_refs"]) == 2
    assert len(metadata["provenance"]) == 3
    assert metadata["bbox_union"] is None  # never union boxes from different pages


def test_missing_provenance_and_unknown_labels():
    raw = load_apple()
    for item in raw["texts"] + raw["tables"]:
        item["prov"] = []
    raw["texts"][0]["label"] = "future_unknown_label"
    doc = adapt(raw)
    assert doc.attrs["skipped_labels"]["future_unknown_label"] == 1
    assert all(b.page is None and not b.provenance for b in doc.blocks)
    assert all(
        c["provenance_scope"] == "table"
        for b in doc.blocks
        if b.table_data
        for c in b.table_data["table_cells"]
    )


def test_flat_levels_keep_item_anchor_and_heading_text():
    raw = load_apple()
    raw["texts"][0]["text"] = "Item 8. Financial Statements"
    doc = adapt(raw)
    table = next(b for b in doc.blocks if b.type == "table")
    assert table.section_path == [
        "Item 8. Financial Statements",
        "CONSOLIDATED STATEMENTS OF OPERATIONS",
    ]
    assert table.section == "CONSOLIDATED STATEMENTS OF OPERATIONS"


def test_real_table_values_spans_and_unit_exceptions(tmp_path):
    chunks, path = pipeline(adapt(load_apple()), tmp_path)
    table = next(c for c in chunks if c["metadata"].get("table_structured"))
    structured = table["metadata"]["table_structured"]
    cells = structured["numeric_cells"]

    def find(row):
        return next(
            c for c in cells if c["row_header"] == row and "2018" in c["col_header"]
        )

    assert find("Net sales")["numeric_value"] == 265595
    assert find("Net sales")["normalized_value"] == 265595000000
    assert find("Gross margin")["numeric_kind"] == "AMOUNT"
    assert find("Earnings per share: | Basic")["normalized_value"] == 12.01
    shares = find("Shares used in computing earnings per share: | Basic")
    assert shares["numeric_kind"] == "SHARES"
    assert shares["scale"] == "thousand"
    assert shares["normalized_value"] == 4955377000
    assert any(c["colspan"] == 3 for c in structured["cells"])
    assert all(not c["is_span_ref"] for c in cells)
    graph_rows = _read_chunks(path, strict=True)
    assert (
        graph_rows[table["doc_id"]][0].source_properties["pipeline_version"]
        == "docling-json-v1"
    )
    rows = _read_tables(path, strict=True)[table["doc_id"]]
    payload = _build_table_payload(rows[0], cell_mode="all")
    assert len({c["cell_id"] for c in payload["cell_rows"]}) == len(
        payload["cell_rows"]
    )
    for props in [payload["source_properties"]] + [
        c["source_properties"] for c in payload["cell_rows"]
    ]:
        for value in props.values():
            assert (
                value is None
                or isinstance(value, (str, bool, int, float))
                or (
                    isinstance(value, list)
                    and all(isinstance(x, (str, bool, int, float)) for x in value)
                )
            )
    cell = next(c for c in payload["cell_rows"] if c.get("row_header") == "Net sales")
    evidence = _format_table_record(
        {**cell, **cell["source_properties"], "section_name": rows[0].section_name}
    )
    assert evidence.metadata["pages"] == [41]
    assert evidence.metadata["provenance_scope"] == "cell"
    assert evidence.metadata["section_paths"]
    assert evidence.metadata["source_refs"]
    assert evidence.metadata["provenance"][0]["bbox"]["coord_origin"] == "TOPLEFT"


def test_quarter_and_half_year_headers_remain_distinct(tmp_path):
    raw = json.loads((FIXTURES / "BESTBUY_2024Q2_10Q.pdf.page4.json").read_text())
    chunks, _ = pipeline(adapt(raw, "BESTBUY_2024Q2_10Q.pdf"), tmp_path)
    cells = next(
        c["metadata"]["table_structured"]["numeric_cells"]
        for c in chunks
        if c["metadata"].get("table_structured")
    )
    revenues = [
        c for c in cells if c["row_header"] == "Revenue" and "2023" in c["col_header"]
    ]
    assert len(revenues) == 2
    assert (
        next(c for c in revenues if "Three Months" in c["col_header"])["numeric_value"]
        == 9583
    )
    assert (
        next(c for c in revenues if "Six Months" in c["col_header"])["numeric_value"]
        == 19050
    )
    negative = next(
        c
        for c in cells
        if c["row_header"] == "Interest expense"
        and "Three Months" in c["col_header"]
        and "2023" in c["col_header"]
    )
    assert negative["normalized_value"] == -12000000


def test_json_replay_matches_direct_adapter_and_checks_hash(tmp_path, monkeypatch):
    raw = load_apple()
    path = tmp_path / "sample.docling.json"
    path.write_text(json.dumps(raw))
    info = {
        "doc_id": "sample.pdf",
        "source_path": "sample.pdf",
        "artifact_id": artifact_hash(raw),
    }
    (tmp_path / "sample.parse_info.json").write_text(json.dumps(info))
    output = tmp_path / "clean.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "adapter",
            "--input-path",
            str(path),
            "--input-format",
            "json",
            "--clean-output",
            str(output),
            "--strict",
        ],
    )
    parse_main()
    expected = docling_json_to_clean_document(
        document=raw, doc_id="sample.pdf", source_path="sample.pdf", parse_info=info
    )
    assert json.loads(output.read_text()) == expected.to_dict()
    raw["texts"][0]["text"] = "changed"
    with pytest.raises(ValueError, match="artifact hash"):
        docling_json_to_clean_document(
            document=raw, doc_id="sample.pdf", source_path="", parse_info=info
        )


def test_legacy_missing_source_fields_degrade_gracefully():
    chunk = _format_chunk_record({"text": "old content", "doc_id": "old.pdf"})
    cell = _format_table_record(
        {"text": "100", "row_header": "Revenue", "provenance_json": "invalid json"}
    )
    assert chunk.metadata["pages"] == []
    assert cell.metadata["provenance"] == []
    assert "PDF pages" not in chunk.content


def test_chunk_ids_repeat_and_are_independent_of_batch(tmp_path):
    doc = adapt(load_apple())
    chunks, path = pipeline(doc, tmp_path)
    again, _ = pipeline(doc, tmp_path)
    assert [c["chunk_id"] for c in chunks] == [c["chunk_id"] for c in again]
    other = adapt(load_apple(), "AAAA.pdf")
    elements = tmp_path / "batch.ndjson"
    elements.write_text(
        "\n".join(
            json.dumps(r)
            for d in [other, doc]
            for r in clean_document_to_element_records(d)
        )
    )
    batched = build_chunks(_read_elements(elements))
    assert [c["chunk_id"] for c in batched if c["doc_id"] == doc.doc_id] == [
        c["chunk_id"] for c in chunks
    ]
    rows = _read_chunks(path, strict=True)[doc.doc_id]
    original = _doc_fingerprint(rows)
    rows[0].source_properties["pipeline_version"] = "another-version"
    assert _doc_fingerprint(rows) != original


def test_purge_removes_tables_before_chunks_and_reimport_guard():
    tx = Mock()
    _import_doc_tx(
        tx,
        doc_id="a",
        form_id="f",
        company_id="c",
        rows_payload=[],
        with_sections=True,
        purge_doc_first=True,
    )
    queries = [c.args[0] for c in tx.run.call_args_list]
    assert "Table" in queries[0] and "DETACH DELETE" in queries[0]
    assert "Chunk" in queries[1]
    tx = Mock()
    tx.run.return_value.single.return_value = {"count": 1}
    with pytest.raises(ValueError, match="purge-doc-first"):
        _import_doc_tx(
            tx,
            doc_id="a",
            form_id="f",
            company_id=None,
            rows_payload=[],
            with_sections=False,
            purge_doc_first=False,
        )
    tx = Mock()
    _import_doc_tables_tx(tx, rows=[], purge_doc_id="a")
    assert "DETACH DELETE" in tx.run.call_args_list[0].args[0]
    assert "MERGE" in tx.run.call_args_list[1].args[0]


def test_multi_page_table_does_not_guess_cell_page_or_merge_tables():
    raw = load_apple()
    t = raw["tables"][0]
    t["prov"].append({**deepcopy(t["prov"][0]), "page_no": 42})
    doc = adapt(raw)
    table = next(b for b in doc.blocks if b.type == "table")
    assert table.attrs["possible_continuation"]
    assert table.attrs["continuation_reason"] == "multiple_provenance_pages"
    assert len(table.provenance) == 2
    assert all(
        c["provenance_scope"] == "table" for c in table.table_data["table_cells"]
    )


def test_overlapping_cells_are_rejected():
    table = next(b for b in adapt(load_apple()).blocks if b.type == "table")
    table.table_data["table_cells"].append(deepcopy(table.table_data["table_cells"][0]))
    with pytest.raises(ValueError, match="Overlapping"):
        _build_table_structured(
            table_data=table.table_data, title_text="", section_name="", table_text=""
        )


def test_element_mapping_and_explicit_table_footnote_links():
    raw = load_apple()
    text = raw["texts"][2]
    labels = ["list_item", "caption", "footnote", "formula", "code", "page_header"]
    refs = []
    for label in labels:
        item = deepcopy(text)
        item["self_ref"] = f"#/texts/{len(raw['texts'])}"
        item["label"] = label
        item["text"] = f"{label} content"
        raw["texts"].append(item)
        refs.append({"$ref": item["self_ref"]})
    raw["body"]["children"].extend(refs)
    raw["tables"][0]["captions"] = [refs[1]]
    raw["tables"][0]["footnotes"] = [refs[2]]
    doc = adapt(raw)
    elements = clean_document_to_element_records(doc)
    for category in ["ListItem", "Caption", "Footnote", "Formula", "Code"]:
        assert any(e["category"] == category for e in elements)
    assert not any("page_header content" in e["text"] for e in elements)
    assert sum(b.text == "footnote content" for b in doc.blocks) == 1
    table = next(b for b in doc.blocks if b.type == "table")
    assert any(
        c["kind"] == "footnotes" and c["source_ref"].endswith(refs[2]["$ref"])
        for c in table.attrs["table_context"]
    )


def test_numeric_span_emits_one_anchor_and_rate_without_percent_symbol():
    data = {
        "num_rows": 2,
        "num_cols": 2,
        "table_cells": [
            {
                "start_row_offset_idx": 0,
                "end_row_offset_idx": 1,
                "start_col_offset_idx": 0,
                "end_col_offset_idx": 2,
                "row_span": 1,
                "col_span": 2,
                "text": "Rate (annualized)",
                "column_header": True,
            },
            {
                "start_row_offset_idx": 1,
                "end_row_offset_idx": 2,
                "start_col_offset_idx": 0,
                "end_col_offset_idx": 2,
                "row_span": 1,
                "col_span": 2,
                "text": "4.63",
            },
        ],
    }
    table = _build_table_structured(
        table_data=data,
        title_text="in millions, except rates",
        section_name="",
        table_text="",
    )
    assert len(table["numeric_cells"]) == 1
    assert table["numeric_cells"][0]["numeric_kind"] == "PERCENT"
    assert table["numeric_cells"][0]["normalized_value"] == pytest.approx(0.0463)


def test_missing_page_dimensions_keeps_raw_bbox_without_fake_conversion():
    raw = load_apple()
    raw["pages"] = {}
    table = next(b for b in adapt(raw).blocks if b.type == "table")
    assert table.provenance[0]["bbox"] is None
    assert table.provenance[0]["raw_bbox"]["coord_origin"] == "BOTTOMLEFT"


def test_title_that_is_only_units_does_not_replace_section():
    raw = load_apple()
    raw["texts"][2]["label"] = "section_header"
    raw["texts"][2]["level"] = 1
    table = next(b for b in adapt(raw).blocks if b.type == "table")
    assert table.section == "CONSOLIDATED STATEMENTS OF OPERATIONS"
    assert table.attrs["table_context"][0]["kind"] == "preceding_unit_text"


def load_jpmorgan():
    return json.loads((FIXTURES / "JPMORGAN_2023Q2_10Q.pdf.page164.json").read_text())


def test_shared_table_footnotes_reach_graph_payload_and_evidence(tmp_path):
    raw = load_jpmorgan()
    original = deepcopy(raw)
    doc = adapt(raw, "JPMORGAN_2023Q2_10Q.pdf")
    assert raw == original
    tables = [b for b in doc.blocks if b.type == "table"]
    assert len(tables) == 2
    for table in tables:
        context = table.attrs["table_context"]
        assert [c["marker"] for c in context] == ["(a)", "(b)"]
        assert all(c["provenance"][0]["page_no"] == 164 for c in context)
        assert all(c["source_ref"].endswith(("#/texts/10", "#/texts/11")) for c in context)
    assert any(b.text.startswith("(a)") and b.type == "paragraph" for b in doc.blocks)
    _, path = pipeline(doc, tmp_path)
    rows = [row for group in _read_tables(path, strict=True).values() for row in group]
    assert len(rows) == 2
    for row in rows:
        payload = _build_table_payload(row, cell_mode="numeric")
        stored = payload["source_properties"]["table_context_json"]
        record = {"doc_id": doc.doc_id, "table_context_json": stored}
        evidence = _format_table_record(record)
        assert "quarterly average" in evidence.content
        assert "prior year" in evidence.content
        assert len(evidence.metadata["table_context"]) == 2
        chunk_evidence = _format_chunk_record({"table_cells": [record, record]})
        assert chunk_evidence.content.count("(a) Reflects") == 1
        assert len(chunk_evidence.metadata["table_context"]) == 2


@pytest.mark.parametrize("boundary", ["unmatched_marker", "different_page", "heading", "prose", "different_column", "ambiguous_marker"])
def test_table_footnotes_do_not_cross_boundaries(boundary):
    raw = load_jpmorgan()
    note = raw["texts"][10]
    if boundary == "unmatched_marker":
        note["text"] = note["text"].replace("(a)", "(z)")
    elif boundary == "different_page":
        note["prov"][0]["page_no"] = 165
    elif boundary == "heading":
        note["label"] = "section_header"
    elif boundary == "prose":
        note["text"] = "Unrelated body paragraph."
    elif boundary == "different_column":
        note["prov"][0]["bbox"].update(l=0, r=10)
    else:
        raw["texts"][11]["text"] = "(a) Another conflicting definition."
    tables = [b for b in adapt(raw).blocks if b.type == "table"]
    assert all(not any(c.get("marker") == "(a)" for c in t.attrs["table_context"]) for t in tables)


def test_table_context_missing_or_invalid_is_backward_compatible():
    for value in (None, "not JSON", "{}", "[null, 1]"):
        evidence = _format_table_record({"table_context_json": value})
        assert evidence.metadata["table_context"] == []
        assert "Table context:" not in evidence.content


def test_explicit_table_caption_becomes_title_in_graph_and_evidence(tmp_path):
    raw = load_apple()
    raw['texts'][0]['text'] = 'Consolidated\n  Statements\t of Operations'
    raw['tables'][0]['captions'] = [{'$ref': raw['texts'][0]['self_ref']}]
    original = deepcopy(raw)
    _, path = pipeline(adapt(raw), tmp_path)
    row = next(iter(_read_tables(path, strict=True).values()))[0]
    properties = _build_table_payload(row, cell_mode='numeric')['source_properties']
    assert properties['title'] == 'Consolidated Statements of Operations'
    assert properties['title_source'] == 'docling_caption'
    assert properties['title_source_refs'][0].endswith('::#/texts/0')
    assert raw == original
    record = {
        'table_title': properties['title'],
        'table_title_source': properties['title_source'],
        'table_title_source_refs': properties['title_source_refs'],
    }
    evidence = _format_table_record(record)
    assert 'Table title: Consolidated Statements of Operations' in evidence.content
    assert evidence.metadata['table_title_source_refs'] == properties['title_source_refs']
    assert 'Table title: Consolidated Statements of Operations' in _format_chunk_record({'table_cells': [record]}).content


@pytest.mark.parametrize('captions', [[], [
    {'kind': 'captions', 'text': 'First title', 'source_ref': 'a'},
    {'kind': 'captions', 'text': 'Different title', 'source_ref': 'b'},
]])
def test_unknown_or_ambiguous_table_title_does_not_use_section(tmp_path, captions):
    _, path = pipeline(adapt(load_apple()), tmp_path)
    row = next(iter(_read_tables(path, strict=True).values()))[0]
    row.metadata['table_context'] = captions
    assert row.section_name
    properties = _build_table_payload(row, cell_mode='numeric')['source_properties']
    assert properties['title'] is None
    assert properties['title_source'] is None
    assert properties['title_source_refs'] == []
    assert 'Table title:' not in _format_table_record({}).content


def test_json_only_pipeline_rejects_missing_table_data_and_retired_cli():
    from neo4j_graphrag.preprocess.docling_adapter import build_parser

    with pytest.raises(ValueError, match="Native Docling table_data is required"):
        _build_table_structured(
            table_data=None, title_text="", section_name="", table_text="legacy table"
        )
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([
            '--input-path', 'sample.pdf', '--clean-output', 'clean.jsonl',
            '--legacy-markdown',
        ])
    assert exc.value.code == 2


def test_table_preparation_is_independent_of_chunking(tmp_path):
    records = clean_document_to_element_records(adapt(load_apple()))
    original = deepcopy(records)
    prepared = [prepare_element(row) for row in records]
    assert records == original
    table = next(row for row in prepared if row['category'] == 'Table')
    assert 'chunk_id' not in table
    assert table['metadata']['table_structured']['numeric_cells']
    for before, after in zip(records, prepared):
        if before['category'] != 'Table':
            assert before == after
        else:
            assert before['metadata']['table_data'] == after['metadata']['table_data']
            assert before['metadata']['provenance'] == after['metadata']['provenance']
    path = tmp_path / 'prepared.ndjson'
    path.write_text('\n'.join(json.dumps(row) for row in prepared))
    for budget in (100, 2048, 10000):
        chunks = build_chunks(_read_elements(path), target_max_chars=budget)
        chunk = next(row for row in chunks if row['metadata']['is_table'])
        assert chunk['metadata']['table_structured'] == table['metadata']['table_structured']
        assert chunk['metadata']['table_context'] == table['metadata']['table_context']


def test_table_preparation_cli_and_invalid_structure(tmp_path, monkeypatch):
    from neo4j_graphrag.preprocess.table_graph_prep import main

    records = clean_document_to_element_records(adapt(load_apple()))
    table = next(row for row in records if row['category'] == 'Table')
    path = tmp_path / 'elements.ndjson'
    output = tmp_path / 'prepared.ndjson'
    def write_input():
        path.write_text('\n'.join(json.dumps(row) for row in records))
    write_input()
    monkeypatch.setattr('sys.argv', [
        'table_graph_prep', '--elements-path', str(path), '--output', str(output), '--strict'
    ])
    main()
    prepared = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(prepared) == len(records)
    assert next(row for row in prepared if row['category'] == 'Table')['metadata']['table_structured']
    # Validation happens before a chunk exists, and strict failure does not
    # publish a partial output or overwrite the last successful result.
    previous_output = output.read_text()
    table['metadata']['table_data']['table_cells'][0]['end_row_offset_idx'] = 99999
    write_input()
    with pytest.raises(ValueError, match='Invalid table cell offsets'):
        main()
    assert output.read_text() == previous_output


def test_failed_reprepare_clears_stale_cells_and_import_rejects(tmp_path):
    records = clean_document_to_element_records(adapt(load_apple()))
    table = next(row for row in records if row['category'] == 'Table')
    table = prepare_element(table)
    table['metadata']['table_data']['table_cells'].append(
        deepcopy(table['metadata']['table_data']['table_cells'][0])
    )
    failed = prepare_element(table, strict=False)
    assert 'table_structured' not in failed['metadata']
    assert 'Overlapping' in failed['metadata']['table_structured_error']
    path = tmp_path / 'failed.ndjson'
    path.write_text(json.dumps(failed))
    chunks = build_chunks(_read_elements(path))
    path.write_text('\n'.join(json.dumps(row) for row in chunks))
    with pytest.raises(ValueError, match='table is not prepared'):
        _read_tables(path, strict=True)
    assert _read_tables(path, strict=False) == {}
    # Repairing and reprocessing also removes the old failure marker.
    failed['metadata']['table_data']['table_cells'].pop()
    repaired = prepare_element(failed)
    assert 'table_structured_error' not in repaired['metadata']
    assert repaired['metadata']['table_structured']['numeric_cells']


def test_table_processor_rejects_chunk_input_and_old_cli():
    from neo4j_graphrag.preprocess.table_graph_prep import build_parser

    with pytest.raises(ValueError, match='before chunking'):
        prepare_element({'chunk_id': 'old-chunk', 'metadata': {}})
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--chunks-path', 'chunks.ndjson', '--output', 'out.ndjson'])
