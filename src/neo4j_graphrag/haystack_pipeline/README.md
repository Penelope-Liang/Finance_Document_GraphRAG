# Haystack Pipeline (Isolated)

This folder provides an isolated Haystack-based retrieval/QA flow without changing:
- `src/neo4j_graphrag/query/router.py`
- `src/neo4j_graphrag/dense_retrieval/faiss_builder.py`
- extraction/build-graph pipeline

## Install

```bash
pip install haystack-ai sentence-transformers
# optional generation backend
pip install ollama-haystack
```

## 1) Build index

```bash
python -m neo4j_graphrag.haystack_pipeline.build_index \
  --chunks-path output/run_20260313_160938/chunks.ndjson \
  --out-dir output/run_20260313_160938/haystack_index \
  --embedder-model BAAI/bge-m3 \
  --batch-size 64
```

Artifacts:
- `documents_with_embeddings.jsonl`
- `index_meta.json`

## 2) Run compare (RAG vs GraphRAG)

```bash
python -m neo4j_graphrag.haystack_pipeline.run_compare \
  --mode compare \
  --retrieval vector \
  --index-path output/run_20260313_160938/haystack_index/documents_with_embeddings.jsonl \
  --questions-file output/financebench/questions_metrics.txt \
  --question-doc-map output/financebench/question_doc_map_metrics.jsonl \
  --top-k 20 \
  --effective-search-ratio 5 \
  --output output/run_20260313_160938/pred_compare_haystack_vector.json
```

## Retrieval-only diagnostics

```bash
python -m neo4j_graphrag.haystack_pipeline.run_compare \
  --mode compare \
  --retrieval hybrid \
  --retrieval-only \
  --index-path output/run_20260313_160938/haystack_index/documents_with_embeddings.jsonl \
  --questions-file output/financebench/questions_metrics.txt \
  --question-doc-map output/financebench/question_doc_map_metrics.jsonl \
  --top-k 20 \
  --effective-search-ratio 10 \
  --output output/run_20260313_160938/pred_compare_haystack_hybrid_retrieval_only.json
```

