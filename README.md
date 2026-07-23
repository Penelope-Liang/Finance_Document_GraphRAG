# FinanceBench Financial Document GraphRAG

This repository contains a FinanceBench question-answering pipeline that compares a text-only RAG baseline with a Financial Document GraphRAG system over public financial filings.

The project focuses on evidence construction for financial reports. It parses PDF filings into section-aware chunks, extracts structured table-cell evidence, imports the resulting document graph into Neo4j, and answers FinanceBench questions using either hybrid chunk retrieval or graph-augmented dual retrieval.

## Main Components

- `scripts/run_document.py`: thin CLI entrypoint for RAG and GraphRAG experiments.
- `scripts/evaluate_answer.py`: answer evaluator using RAGAS `faithfulness` and `answer_accuracy`.
- `scripts/evaluate_retrieval.py`: retrieval evaluator using RAGAS retrieval metrics.
- `src/finance_document_qa/`: FinanceBench-specific services, retrieval pipeline, prompts, and evaluation code.
- `src/neo4j_graphrag/preprocess/`: PDF element processing, section-aware chunking, and table-cell extraction.
- `src/neo4j_graphrag/tools/build_graph.py`: imports chunks into the Neo4j financial document graph.
- `src/neo4j_graphrag/tools/import_tables.py`: imports structured table and table-cell nodes into Neo4j.
- `src/neo4j_graphrag/tools/build_chunk_embeddings.py`: builds chunk embedding JSONL files for vector indexing.
- `src/neo4j_graphrag/tools/import_chunk_vectors.py`: imports chunk embeddings into Neo4j and creates the vector index.
- `src/neo4j_graphrag/retrievers/`: hybrid, vector, Text2Cypher, and reranking retriever implementations.

## Graph Representation

The Financial Document Graph stores financial filings as connected evidence objects:

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

`TableCell` nodes store row and column context as properties, including `row_header`, `col_header`, `numeric_value`, `numeric_kind`, `scale`, and `normalized_value`.

## Retrieval Modes

The QA runner supports three retrieval modes:

- `hybrid`: text-only RAG baseline using vector search and full-text search over `Chunk.text`.
- `text2cypher`: graph query retrieval using LLM-generated Cypher.
- `dual`: GraphRAG retrieval combining hybrid chunk retrieval with graph evidence from document structure, primary financial statements, and table cells.

In the reported experiments, the RAG baseline uses `--retriever-mode hybrid`, while GraphRAG uses `--retriever-mode dual`.

## Environment

The pipeline expects:

- Python environment with the dependencies from `pyproject.toml`.
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

This evaluator writes `answer_evaluation` to each result row. The primary metrics are RAGAS `faithfulness`, which checks whether the answer is supported by retrieved context, RAGAS `answer_accuracy`, which compares the generated answer against the reference answer, and a deterministic answer accuracy check for numeric, rounded, unit-scaled, and yes/no answers.

## Archived Preprocessing and Graph Import

Historical batch scripts used during thesis experiments are archived under `_archive/scripts/`:

- `_archive/scripts/run_docling_direct_extract_13.sh`
- `_archive/scripts/run_docling_metrics_remaining_37.sh`
- `_archive/scripts/import_direct_extract_13_to_neo4j.sh`

These scripts parse selected FinanceBench PDFs with Docling, construct section-aware chunks, extract structured table cells, and import the result into Neo4j. They are retained only for reproducibility of old experiment batches; the active QA workflow assumes the graph has already been imported.

## Notes

- Runtime outputs are written under `output/` and are ignored by Git.
- Local logs are ignored by Git.
- `_archive/` contains old experiments and upstream package remnants and is ignored by Git by default.
- The repository keeps the original Neo4j license and notice files because this project builds on the Neo4j GraphRAG Python package.
