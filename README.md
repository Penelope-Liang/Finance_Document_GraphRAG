# FinanceBench Financial Document GraphRAG

This repository contains a FinanceBench question-answering pipeline that
compares a text-only RAG baseline with a Financial Document GraphRAG system over
public financial filings.

The project focuses on evidence construction for financial reports. It parses
PDF filings into section-aware chunks, extracts structured table-cell evidence,
imports the resulting document graph into Neo4j, and answers FinanceBench
questions using either hybrid chunk retrieval or graph-augmented dual retrieval.

## Main Components

- `scripts/run_document.py` : thin CLI entrypoint for RAG and GraphRAG
  experiments.
- `scripts/evaluate_answer.py` : answer evaluator using RAGAS `faithfulness` and
  `answer_accuracy` .
- `scripts/evaluate_retrieval.py` : retrieval evaluator using RAGAS retrieval
  metrics.
- `src/finance_document_qa/` : FinanceBench-specific services, retrieval
  pipeline, prompts, and evaluation code.
- `src/neo4j_graphrag/preprocess/` : PDF element processing, section-aware
  chunking, and table-cell extraction.
- `src/neo4j_graphrag/tools/build_graph.py` : imports chunks into the Neo4j
  financial document graph.
- `src/neo4j_graphrag/tools/import_tables.py` : imports structured table and
  table-cell nodes into Neo4j.
- `src/neo4j_graphrag/tools/build_chunk_embeddings.py` : builds chunk embedding
  JSONL files for vector indexing.
- `src/neo4j_graphrag/tools/import_chunk_vectors.py` : imports chunk embeddings
  into Neo4j and creates the vector index.
- `src/neo4j_graphrag/retrievers/` : hybrid, vector, Text2Cypher, and reranking
  retriever implementations.

## PDF preprocessing: native Docling JSON

The ingestion path is:

```text
PDF -> DoclingDocument -> original .docling.json + .parse_info.json
                      -> CleanDocument -> elements.ndjson -> chunks
                      -> structured table cells -> Neo4j import
```

The JSON file preserves the parser output. The adjacent **parse-info file** is a
small explanation of how it was generated: source PDF SHA-256, original document
ID/path, Docling/Core/model/parser versions, device, OCR setting, selected page
range, timestamp, and a canonical JSON content hash (`artifact_id`). Raw files
are named with the artifact hash so another parse does not overwrite an earlier
one. Replay checks that hash when present. Keep the JSON and its parse-info file
together.

The adapter walks body references in reading order, rather than iterating the
`texts` array directly. One text item remains one input block; Docling text
items are not guaranteed to be complete natural paragraphs. ChunkBuffer combines
input blocks by headings, tables and character budget. Table display text is
derived row by row from JSON for retrieval; cell extraction uses the structured
data, never that display text or reconstructed HTML.

| Docling element                                 | Internal handling                                              |
| ----------------------------------------------- | -------------------------------------------------------------- |
| title / section_header                          | Title; retain supplied level and use heading text for sections |
| text / paragraph / reference / handwritten_text | NarrativeText                                                  |
| list_item                                       | ListItem                                                       |
| caption / footnote                              | Caption / Footnote; keep explicit table links where provided   |
| formula / code                                  | Formula / Code text; no formula execution during parsing       |
| page_header / page_footer / furniture           | Excluded from retrieval body; original JSON retained           |
| picture / chart                                 | Image content excluded; referenced captions/footnotes retained |
| unknown labels                                  | Skipped and counted in document metadata for review            |

Heading levels can be flat. SEC Item heading text provides an additional section
anchor; short unit-only headings are treated as unit annotations instead of
replacing the financial statement section. This is heuristic, not a
reconstruction of every SEC section hierarchy. Captions, explicit footnotes, and
up to two immediately preceding same-page/same-section unit paragraphs are
retained as table context. Lowercase letter markers such as `(a)` and `(b)` also
link immediately following notes to matching tables in a contiguous
same-page/same-section group. Shared notes can belong to both tables; unmatched
or ambiguous markers, intervening prose or headings, and different columns are
excluded. Original note blocks remain intact. These inferred links retain source
references and provenance in `table_context_json` , which the metrics and
primary-statement retrieval formatters include in evidence. They do not override
table-wide numeric units. Cross-page and unmarked footnote inference is not
supported.

Table nodes expose `title` , `title_source` , and `title_source_refs` . A
unique, explicitly linked Docling caption supplies the title (with whitespace
normalized); missing or conflicting captions leave it unset. `section_name`
remains separate and is never substituted for a table title. TableCells access
this title through their parent Table; metrics and primary-statement retrieval
return it with evidence.

Every block/cell keeps a list of provenance records. Bboxes are normalized to
**TOPLEFT page coordinates** when page dimensions permit; original bbox/origin
and page size remain available. Unknown coordinates stay unknown. A cell bbox is
only assigned a page when its parent table has a single source page; otherwise
its provenance is explicitly table-level. Chunk metadata carries page sets and
source references, and never unions bounding boxes across different pages.

### Run and replay

Use a separate parsing environment with the pinned versions in
[`requirements/docling.txt`](requirements/docling.txt) . Docling may download
model weights on first use. This environment is independent of the online QA
dependencies.

```bash
uv venv .venv-docling --python 3.13
uv pip install --python .venv-docling/bin/python -r requirements/docling.txt

PYTHONPATH=src .venv-docling/bin/python -m neo4j_graphrag.preprocess.docling_adapter \
  --input-path dataset/financebench/pdfs/BESTBUY_2024Q2_10Q.pdf \
  --no-ocr \
  --clean-output output/preprocess/clean.jsonl \
  --elements-output output/preprocess/elements.ndjson --strict

# Replay saved JSON without loading PDF models (Docling is not needed for replay).
PYTHONPATH=src python -m neo4j_graphrag.preprocess.docling_adapter \
  --input-format json --input-path output/preprocess/docling_raw \
  --clean-output output/preprocess/replayed.jsonl \
  --elements-output output/preprocess/replayed.elements.ndjson --strict

PYTHONPATH=src python -m neo4j_graphrag.preprocess.chunk_from_elements \
  --elements-path output/preprocess/elements.ndjson \
  --output output/preprocess/chunks.ndjson

PYTHONPATH=src python -m neo4j_graphrag.preprocess.table_graph_prep \
  --chunks-path output/preprocess/chunks.ndjson \
  --output output/preprocess/tables.ndjson --strict
```

`--no-ocr` disables OCR for born-digital filings; omit it for OCR.
`--page-range FIRST LAST` selects **1-based inclusive PDF pages** for
inspection; leave it out when processing a complete filing. The pipeline uses
native Docling JSON for document structure and table cells. The native JSON
mapping follows
[Docling's structured document model](https://github.com/docling-project/docling/blob/main/docs/v2.md)
.

### Neo4j storage and replacement

Chunk, Table and TableCell source properties include `pipeline_version` ,
`source_sha256` , `artifact_id` , and parser versions. `pages` and `source_refs`
are flat arrays; richer provenance and section paths are JSON strings
(`provenance_json`, `section_paths_json` ), not nested Neo4j properties. Table
context is stored as `table_context_json` . Retrieval formatting exposes source
pages and provenance when available and leaves them empty for old records.
Text2Cypher is instructed to return these optional fields; model-generated
queries can still omit them.

JSON-derived chunks default to ID version `docling-json-v1` . Replaying the same
input is deterministic. Changes to chunk boundaries can produce new IDs. Import
checkpoints fingerprint content and provenance rather than skipping any
previously seen document ID. An import that would leave obsolete chunks refuses
without `--purge-doc-first` .

To replace a previously ingested complete filing, use
`build_graph --purge-doc-first --no-resume` , then
`import_tables --purge-doc-first` , with the new structured chunk file. The
chunk importer deletes old tables/cells before deleting old chunks; replacement
is transactional per document within each importer. The two import tools are
separate transactions, so rebuild in an isolated database or maintenance window.
Regenerate embeddings for new chunk IDs before switching retrieval to the new
data. Do not use a page-range sample to replace a complete filing in an existing
database.

### Parsing boundaries

Docling output can contain recognition errors. Adjacent tables may be marked as
possible continuations but are not automatically merged. Cross-page table
reconstruction and unmarked footnote association are not performed. Table titles
require explicitly linked captions; missing titles remain unset.

## Graph Representation

The Financial Document Graph stores financial filings as connected evidence
objects:

- `Company`
- `Form`
- `Section`
- `Chunk`
- `StatementType`
- `Table`
- `TableCell`

Important relationships include:

- `Company -[:FILED]-> Form`
- `Form -[:HAS_SECTION]-> Section`
- `Chunk -[:IN_FORM]-> Form`
- `Chunk -[:IN_SECTION]-> Section`
- `Chunk -[:NEXT]-> Chunk`
- `Chunk -[:HAS_STATEMENT_TYPE]-> StatementType`
- `Chunk -[:HAS_TABLE]-> Table`
- `Table -[:HAS_CELL]-> TableCell`

`TableCell` nodes store row and column context as properties, including
`row_header` , `col_header` , `numeric_value` , `numeric_kind` , `scale` , and
`normalized_value` .

## Retrieval Modes

The API and CLI default to `auto` routing:

- Open-book questions use `hybrid` .
- Numerical questions first use `graph` , which queries table cells, metric
  inputs, and primary statements without hybrid search. If the graph path
  returns an error or all graph branches return no evidence, routing falls back
  to hybrid. An empty individual branch does not trigger fallback if another has
  evidence.

Dataset `question_reasoning` labels `numerical` / `Numerical reasoning` select
graph; `open` / `open-book` / `Information extraction` select hybrid. Otherwise,
the question-style classifier routes `calculation` to graph and other styles to
hybrid. The default classifier is rule-based; `--question-style llm-auto` uses
the model classifier. Check classifications for ambiguous questions.

Use `--retriever-mode auto` for CLI routing or `RETRIEVER_MODE=auto` for the API
(existing environment overrides still apply). Results record the initial and
actual retriever modes, question category, and fallback flag/reason in
`run_config` . Numerical fallback preserves the calculation style and options.

Explicit retrieval modes are also available:

- `hybrid` : text-only RAG baseline using vector search and full-text search
  over `Chunk.text` .
- `text2cypher` : graph query retrieval using LLM-generated Cypher.
- `graph` : graph-only retrieval; use `auto` for the hybrid fallback policy.
- `dual` : GraphRAG retrieval combining hybrid chunk retrieval with graph
  evidence from document structure, primary financial statements, and table
  cells.

`auto` selects `hybrid` or `graph` ; it does not select the mixed `dual` mode.

## Production Architecture

The project exposes one production query path:

```text
API / CLI / batch
  -> FinanceDocument*Service.query()
  -> LangGraph workflow
  -> service query-planning, retrieval, and result-building steps
  -> Neo4j + LLM backend
```

FastAPI provides `/health` and `/query` for real-time serving. The CLI keeps the
same `service.query()` path for batch runs, so API and offline execution do not
drift.

Runtime behavior is controlled by environment profile:

```text
APP_PROFILE=local   -> Ollama by default
APP_PROFILE=cloud   -> Azure OpenAI by default, reranker off by default
```

Secrets are read from environment variables and should be supplied through a
local `.env` file, shell environment, or cloud secret store. Real credentials
must not be committed.

## Environment

The pipeline expects:

- Python environment with the dependencies from `pyproject.toml` .
- Neo4j or Neo4j Aura instance containing the imported financial document graph.
- Ollama for local answer generation and embeddings.
- Optional Azure OpenAI credentials for RAGAS answer correctness evaluation.

Install the evaluation dependencies only when running the correctness evaluator:

```bash
pip install -e '.[evaluation]'
```

Common environment variables:

```bash
export NEO4J_URI='neo4j+s://...databases.neo4j.io'
export NEO4J_USER='...'
export NEO4J_PASSWORD='...'
export NEO4J_DATABASE='...'

export OLLAMA_HOST='http://127.0.0.1:11434'
export OLLAMA_LLM_MODEL='qwen3.6:27b'
export OLLAMA_EMBED_MODEL='bge-m3:latest'
```

For Azure OpenAI correctness evaluation:

```bash
export AZURE_OPENAI_ENDPOINT='https://...openai.azure.com/'
export AZURE_OPENAI_API_KEY='...'
export AZURE_OPENAI_API_VERSION='2024-12-01-preview'
export AZURE_OPENAI_DEPLOYMENT='gpt-4.1-mini'
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT='text-embedding-3-small'
```

Do not commit real credentials.

## Quick Start

Run the API service:

```bash
uv run --extra app python scripts/run_api.py
```

Check the shallow health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

The production path is `API/CLI -> FinanceDocument*Service.query() -> LangGraph`
. `service.query()` remains the stable external interface; LangGraph owns the
internal orchestration.

Run the text-only RAG baseline:

```bash
PYTHONPATH=src python -u scripts/run_document.py \
  --questions-path dataset/financebench/financebench_metrics_remaining_37.jsonl \
  --retriever-mode hybrid \
  --question-style llm-auto \
  --use-query-rewrite \
  --top-k 10 \
  --effective-search-ratio 20 \
  --output output/rag_results.json
```

Run GraphRAG with dual retrieval:

```bash
PYTHONPATH=src python -u scripts/run_document.py \
  --questions-path dataset/financebench/financebench_metrics_remaining_37.jsonl \
  --retriever-mode dual \
  --question-style llm-auto \
  --use-query-rewrite \
  --top-k 10 \
  --effective-search-ratio 20 \
  --use-calculation-executor \
  --calculation-model qwen2.5-coder:14b-instruct \
  --output output/graphrag_results.json
```

Evaluate generated answers:

```bash
PYTHONPATH=src python -u scripts/evaluate_answer.py \
  --input output/graphrag_results.json \
  --output output/graphrag_results_answer_eval.json \
  --judge-model gpt-4.1-mini \
  --overwrite
```

This evaluator writes `answer_evaluation` to each result row. The primary
metrics are RAGAS `faithfulness` , which checks whether the answer is supported
by retrieved context, RAGAS `answer_accuracy` , which compares the generated
answer against the reference answer, and a deterministic answer accuracy check
for numeric, rounded, unit-scaled, and yes/no answers.

## Notes

- Runtime outputs are written under `output/` and are ignored by Git.
- Local logs are ignored by Git.
- The repository keeps the original Neo4j license and notice files because this
  project builds on the Neo4j GraphRAG Python package.
