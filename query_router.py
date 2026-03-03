from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import neo4j
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.rag import GraphRAG
from neo4j_graphrag.retrievers import (
    HybridCypherRetriever,
    HybridRetriever,
    VectorCypherRetriever,
    VectorRetriever,
)
from neo4j_graphrag.types import RetrieverResultItem


warnings.filterwarnings("ignore", category=DeprecationWarning, module="neo4j_graphrag")

DEFAULT_CYPHER_RETRIEVAL_QUERY = """
WITH node AS chunk, score
OPTIONAL MATCH (chunk)<-[:FROM_CHUNK]-(entity)
WITH chunk, score, collect(DISTINCT {
    label: labels(entity)[1],
    name: entity.name
}) AS entities
RETURN chunk.text AS text,
       score AS similarity_score,
       coalesce(chunk.chunk_id, chunk.id, elementId(chunk)) AS chunk_id,
       entities
"""


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"Missing env var: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def _load_questions(args: argparse.Namespace) -> list[str]:
    questions: list[str] = []
    if args.questions_file:
        with open(args.questions_file, "r", encoding="utf-8") as f:
            questions.extend(
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            )
    if args.question:
        questions.extend(q.strip() for q in args.question if q.strip())
    if args.questions:
        questions.extend(q.strip() for q in args.questions if q.strip())
    return questions


def _read_text_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return Path(path).read_text(encoding="utf-8")


def _resolve_mode_list(mode: str) -> list[str]:
    return ["rag", "graphrag"] if mode == "compare" else [mode]


def _node_result_formatter(record: neo4j.Record) -> RetrieverResultItem:
    node = record.get("node")
    score = record.get("score")
    metadata: dict[str, Any] = {"score": score}
    content = None

    if node is not None:
        chunk_id = node.get("chunk_id") or node.get("id")
        if chunk_id is not None:
            metadata["chunk_id"] = str(chunk_id)
        content = node.get("text")

    return RetrieverResultItem(
        content=content if content is not None else str(node) if node is not None else str(record),
        metadata=metadata,
    )


def _cypher_result_formatter(record: neo4j.Record) -> RetrieverResultItem:
    metadata: dict[str, Any] = {}
    for key in ("chunk_id", "similarity_score", "score", "entities", "form_id", "company_id"):
        value = record.get(key)
        if value is not None:
            metadata[key] = value

    if "score" not in metadata and "similarity_score" in metadata:
        metadata["score"] = metadata["similarity_score"]

    return RetrieverResultItem(
        content=record.get("text") or str(record),
        metadata=metadata or None,
    )


def _build_retriever(
    *,
    mode: str,
    retrieval: str,
    driver: neo4j.Driver,
    vector_index_name: str,
    fulltext_index_name: str,
    embedder: OllamaEmbeddings,
    retrieval_query: str,
    database: Optional[str],
):
    if mode == "rag":
        if retrieval == "vector":
            return VectorRetriever(
                driver=driver,
                index_name=vector_index_name,
                embedder=embedder,
                result_formatter=_node_result_formatter,
                neo4j_database=database,
            )
        return HybridRetriever(
            driver=driver,
            vector_index_name=vector_index_name,
            fulltext_index_name=fulltext_index_name,
            embedder=embedder,
            result_formatter=_node_result_formatter,
            neo4j_database=database,
        )

    if retrieval == "vector":
        return VectorCypherRetriever(
            driver=driver,
            index_name=vector_index_name,
            retrieval_query=retrieval_query,
            embedder=embedder,
            result_formatter=_cypher_result_formatter,
            neo4j_database=database,
        )
    return HybridCypherRetriever(
        driver=driver,
        vector_index_name=vector_index_name,
        fulltext_index_name=fulltext_index_name,
        retrieval_query=retrieval_query,
        embedder=embedder,
        result_formatter=_cypher_result_formatter,
        neo4j_database=database,
    )


def _extract_chunk_ids(items: list[RetrieverResultItem]) -> list[str]:
    ids: list[str] = []
    for item in items:
        metadata = item.metadata or {}
        chunk_id = (
            metadata.get("chunk_id")
            or metadata.get("id")
            or metadata.get("elementId")
            or metadata.get("element_id")
        )
        if chunk_id is None:
            continue
        value = str(chunk_id)
        if value not in ids:
            ids.append(value)
    return ids


def _build_context_preview(
    items: list[RetrieverResultItem], preview_items: int, preview_chars: int
) -> str:
    content = "\n---\n".join(str(item.content) for item in items[:preview_items])
    if len(content) > preview_chars:
        return f"{content[:preview_chars].rstrip()}..."
    return content


def _save_results(
    output_path: str,
    output_format: str,
    rows: list[dict[str, Any]],
) -> None:
    if output_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        return

    if output_format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    headers = [
        "mode",
        "retrieval",
        "question",
        "answer",
        "latency_ms",
        "retriever",
        "retrieved_chunk_ids",
        "context_preview",
        "error",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            csv_row = row.copy()
            csv_row["retrieved_chunk_ids"] = "|".join(row["retrieved_chunk_ids"])
            writer.writerow(csv_row)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified RAG/GraphRAG query runner with selectable retriever mode."
    )
    parser.add_argument("questions", nargs="*", help="Inline question text")
    parser.add_argument(
        "-q",
        "--question",
        action="append",
        help="Inline question (can be repeated)",
    )
    parser.add_argument(
        "--questions-file",
        help="File containing questions (one per line, '#' for comments)",
    )
    parser.add_argument(
        "--mode",
        choices=["rag", "graphrag", "compare"],
        default="graphrag",
        help="Run plain RAG, GraphRAG, or both (compare).",
    )
    parser.add_argument(
        "--retrieval",
        choices=["vector", "hybrid"],
        default="hybrid",
        help="Retriever family used in the selected mode(s).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top K to retrieve.")
    parser.add_argument(
        "--effective-search-ratio",
        type=int,
        default=1,
        help="Multiplier for vector candidate pool size.",
    )
    parser.add_argument(
        "--hybrid-ranker",
        choices=["naive", "linear"],
        default="naive",
        help="Hybrid ranker (only applies when --retrieval=hybrid).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Weight for vector score when --hybrid-ranker=linear (0..1).",
    )
    parser.add_argument(
        "--output",
        default="query_results.json",
        help="Output path for structured run results.",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "jsonl", "csv"],
        default="json",
        help="Output file format.",
    )
    parser.add_argument(
        "--preview-items",
        type=int,
        default=3,
        help="Number of retrieved items included in context preview.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        help="Max character length of context preview.",
    )
    parser.add_argument(
        "--retrieval-query",
        help="Cypher retrieval query used in GraphRAG mode.",
    )
    parser.add_argument(
        "--retrieval-query-file",
        help="Read retrieval query from a file path.",
    )
    parser.add_argument(
        "--neo4j-uri",
        default=os.environ.get("NEO4J_URI", "neo4j+s://998e2e0a.databases.neo4j.io"),
        help="Neo4j URI.",
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j user.",
    )
    parser.add_argument(
        "--neo4j-database",
        default=os.environ.get("NEO4J_DATABASE"),
        help="Neo4j database name (optional).",
    )
    parser.add_argument(
        "--vector-index",
        default=os.environ.get("NEO4J_VECTOR_INDEX", "chunk_embedding_index"),
        help="Neo4j vector index name.",
    )
    parser.add_argument(
        "--fulltext-index",
        default=os.environ.get("NEO4J_FULLTEXT_INDEX", "chunk_fulltext_index"),
        help="Neo4j fulltext index name.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:72b"),
        help="LLM model name for Ollama.",
    )
    parser.add_argument(
        "--embedder-model",
        default=os.environ.get("OLLAMA_EMBEDDER_MODEL", "bge-m3:latest"),
        help="Embedding model name for Ollama.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("OLLAMA_TEMPERATURE", "0")),
        help="LLM temperature.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    questions = _load_questions(args)
    if not questions:
        print("No questions provided. Use --questions-file or -q.", file=sys.stderr)
        sys.exit(1)

    if args.hybrid_ranker == "linear" and args.alpha is None:
        print("--alpha is required when --hybrid-ranker=linear", file=sys.stderr)
        sys.exit(1)

    if args.retrieval_query and args.retrieval_query_file:
        print("Use either --retrieval-query or --retrieval-query-file, not both.", file=sys.stderr)
        sys.exit(1)

    password = _require_env("NEO4J_PASSWORD")

    file_query = _read_text_file(args.retrieval_query_file)
    retrieval_query = args.retrieval_query or file_query or DEFAULT_CYPHER_RETRIEVAL_QUERY

    modes = _resolve_mode_list(args.mode)
    run_rows: list[dict[str, Any]] = []

    llm = OllamaLLM(
        model_name=args.llm_model,
        model_params={"options": {"temperature": args.temperature}},
    )
    embedder = OllamaEmbeddings(model=args.embedder_model)

    try:
        with neo4j.GraphDatabase.driver(
            args.neo4j_uri, auth=(args.neo4j_user, password)
        ) as driver:
            rag_by_mode: dict[str, GraphRAG] = {}
            for mode in modes:
                retriever = _build_retriever(
                    mode=mode,
                    retrieval=args.retrieval,
                    driver=driver,
                    vector_index_name=args.vector_index,
                    fulltext_index_name=args.fulltext_index,
                    embedder=embedder,
                    retrieval_query=retrieval_query,
                    database=args.neo4j_database,
                )
                rag_by_mode[mode] = GraphRAG(retriever=retriever, llm=llm)

            total_runs = len(questions) * len(modes)
            done = 0
            for question in questions:
                for mode in modes:
                    done += 1
                    print(f"[{done}/{total_runs}] mode={mode} retrieval={args.retrieval} | {question}")
                    retriever_config: dict[str, Any] = {
                        "top_k": args.top_k,
                        "effective_search_ratio": args.effective_search_ratio,
                    }
                    if args.retrieval == "hybrid":
                        retriever_config["ranker"] = args.hybrid_ranker
                        if args.hybrid_ranker == "linear":
                            retriever_config["alpha"] = args.alpha

                    start = time.perf_counter()
                    answer = None
                    error = None
                    context_preview = ""
                    chunk_ids: list[str] = []
                    retriever_name = ""
                    try:
                        result = rag_by_mode[mode].search(
                            query_text=question,
                            retriever_config=retriever_config,
                            return_context=True,
                        )
                        answer = result.answer
                        retriever_result = result.retriever_result
                        retriever_name = (
                            (retriever_result.metadata or {}).get("__retriever", "")
                            if retriever_result
                            else ""
                        )
                        items = retriever_result.items if retriever_result else []
                        chunk_ids = _extract_chunk_ids(items)
                        context_preview = _build_context_preview(
                            items, args.preview_items, args.preview_chars
                        )
                        print(f"  answer: {answer[:120]}..." if len(answer) > 120 else f"  answer: {answer}")
                    except Exception as exc:
                        error = str(exc)
                        print(f"  error: {error}")

                    latency_ms = round((time.perf_counter() - start) * 1000, 2)
                    run_rows.append(
                        {
                            "mode": mode,
                            "retrieval": args.retrieval,
                            "question": question,
                            "answer": answer,
                            "latency_ms": latency_ms,
                            "retriever": retriever_name,
                            "retrieved_chunk_ids": chunk_ids,
                            "context_preview": context_preview,
                            "error": error,
                        }
                    )
                    print(f"  latency_ms: {latency_ms}")

    finally:
        if hasattr(llm, "client") and hasattr(llm.client, "_client"):
            llm.client._client.close()
        if hasattr(embedder, "client") and hasattr(embedder.client, "_client"):
            embedder.client._client.close()

    _save_results(args.output, args.output_format, run_rows)
    print(f"Saved structured results to {args.output}")


if __name__ == "__main__":
    main()
