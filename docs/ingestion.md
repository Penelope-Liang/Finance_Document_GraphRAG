# PDF ingestion

Run these commands from the repository root after completing the README setup.
Use the same Neo4j database and embedding model for ingestion and retrieval.

```text
PDF → Docling JSON → CleanDocument → elements.ndjson
    → table preparation → prepared_elements.ndjson → chunks.ndjson
    → Neo4j graph → chunk embeddings → vector index
```

## 1. Parse PDFs

Docling runs in a separate environment with pinned parser dependencies. It may
download model weights on first use.

```bash
uv venv .venv-docling --python 3.13
uv pip install --python .venv-docling/bin/python -r requirements/docling.txt

PYTHONPATH=src .venv-docling/bin/python \
  -m neo4j_graphrag.preprocess.docling_adapter \
  --input-path dataset/financebench/pdfs/BESTBUY_2024Q2_10Q.pdf \
  --no-ocr \
  --clean-output output/preprocess/clean.jsonl \
  --elements-output output/preprocess/elements.ndjson --strict
```

`--input-path` accepts a PDF or directory. Omit `--no-ocr` for OCR.
`--page-range FIRST LAST` limits parsing to an inclusive range of PDF pages,
numbered from 1; omit it for complete filings.

Original `.docling.json` files and companion `.parse_info.json` files are saved
under `output/preprocess/docling_raw/`. Parse information records the source hash,
parser versions, OCR/device options, page range, and JSON content hash. Keep each
pair together; the content hash identifies a specific parsing result.

## 2. Prepare tables, then build chunks

```bash
uv run --env-file .env python \
  -m neo4j_graphrag.preprocess.table_graph_prep \
  --elements-path output/preprocess/elements.ndjson \
  --output output/preprocess/prepared_elements.ndjson --strict

uv run --env-file .env python \
  -m neo4j_graphrag.preprocess.chunk_from_elements \
  --elements-path output/preprocess/prepared_elements.ndjson \
  --output output/preprocess/chunks.ndjson
```

Table preparation reads each Table Element directly: native cells, section context,
captions, units and footnotes. It validates spans, associates headers, and normalizes
financial values before chunking. Original `table_data` stays intact; derived cells
are stored in `metadata.table_structured`. Non-table elements pass through unchanged.
The processor does not require chunk IDs. Chunking carries these results into a
separate table chunk; graph import binds Table and TableCell IDs to that chunk.
Use `--strict` to stop on malformed tables. Without it, failed tables retain an
error marker and no structured result; strict table import rejects those records.
The old table-preparation `--chunks-path` and `--only-should-index` options are
removed: all table elements are prepared before chunk-level indexing decisions.

The adapter follows Docling body references in reading order. Each text item
becomes an input block; chunking groups blocks by headings, tables, and character
budget. Table cells are read directly from JSON, including row/column spans.
Page headers and footers are excluded from retrieval text.

SEC Item text helps identify sections when heading levels are flat. Titles have
normalized whitespace. Captions, explicit footnotes, and up to two adjacent
preceding unit paragraphs provide table context. Matching `(a)`/`(b)`-style notes
can be shared by consecutive same-page tables. Links stop at intervening prose
or headings; ambiguous markers are not linked.

## 3. Import the graph

Configure `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, and `NEO4J_DATABASE` in `.env`.
The project uses `NEO4J_USER` for the username.

```bash
uv run --env-file .env python -m neo4j_graphrag.tools.build_graph \
  --chunks-path output/preprocess/chunks.ndjson --strict

uv run --env-file .env python -m neo4j_graphrag.tools.import_tables \
  --chunks-path output/preprocess/chunks.ndjson --cell-mode numeric --strict
```

The chunk importer creates the full-text index by default. The table importer
adds tables and numerical cells to their parent chunks.

```text
Company ─FILED→ Form ─HAS_SECTION→ Section
Chunk ─IN_FORM→ Form
Chunk ─IN_SECTION→ Section
Chunk ─NEXT→ Chunk
Chunk ─HAS_STATEMENT_TYPE→ StatementType
Chunk ─HAS_TABLE→ Table ─HAS_CELL→ TableCell
```

TableCell properties include `row_header`, `col_header`, `numeric_value`,
`numeric_kind`, `scale`, and `normalized_value`. Cells obtain the table title
through their parent Table. `title`, `title_source`, and `title_source_refs` come
from a unique explicitly linked caption; a section heading is not a substitute.

Page numbers and source references are stored as flat arrays. Rich provenance,
section paths, and table context use JSON strings (`provenance_json`,
`section_paths_json`, `table_context_json`). Coordinates are normalized to a
top-left origin when page dimensions are known; raw coordinates are retained.

## 4. Build and import embeddings

Run Ollama and install the model selected by `OLLAMA_EMBED_MODEL` before this
step. Retrieval must use the same model and vector dimension.

```bash
uv run --env-file .env --extra ollama python \
  -m neo4j_graphrag.tools.build_chunk_embeddings \
  --chunks-path output/preprocess/chunks.ndjson \
  --output output/preprocess/embeddings.jsonl

uv run --env-file .env python -m neo4j_graphrag.tools.import_chunk_vectors \
  --documents-path output/preprocess/embeddings.jsonl
```

The vector importer creates the vector index. `NEO4J_VECTOR_INDEX` and
`NEO4J_FULLTEXT_INDEX` must match the indexes used by the query service.

## Replay saved JSON

Replay does not need Docling or PDF model loading:

```bash
uv run python -m neo4j_graphrag.preprocess.docling_adapter \
  --input-format json --input-path output/preprocess/docling_raw \
  --clean-output output/preprocess/replayed.jsonl \
  --elements-output output/preprocess/replayed.elements.ndjson --strict
```

## Replace an existing filing

Replaying unchanged input produces deterministic chunk IDs. If new chunk
boundaries would leave obsolete nodes, the importer requires explicit replacement.
Use `build_graph --purge-doc-first --no-resume`, followed by
`import_tables --purge-doc-first`, with the complete replacement filing. Each
importer uses a separate transaction per document. Rebuild embeddings for new IDs.
A page-range sample must not replace a complete filing in an existing database.

## Limits

Docling can misread text and table headers. Cross-page tables are not automatically
merged, and unmarked or cross-page footnotes are not inferred. A text item may be
only part of a natural paragraph. Table titles remain unset without a confirmed
caption. For cells in multi-page tables, provenance may be table-level rather
than a precise cell location.
