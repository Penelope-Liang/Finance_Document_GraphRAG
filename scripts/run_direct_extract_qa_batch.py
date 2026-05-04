from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import neo4j
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.rag import GraphRAG
from neo4j_graphrag.retrievers import HybridCypherRetriever, Text2CypherRetriever

from neo4j_graphrag.tools.batch_ask import (
    FINANCIAL_GRAPH_SCHEMA,
    TEXT2CYPHER_EXAMPLES,
    TEXT2CYPHER_PROMPT,
)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing env var: {name}")
    return value


def _load_questions(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def _parse_answer(raw_answer: str) -> tuple[str | None, str | None]:
    try:
        payload = json.loads(raw_answer)
    except json.JSONDecodeError:
        return raw_answer, None
    if isinstance(payload, dict):
        return payload.get("answer"), payload.get("evidence")
    return raw_answer, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 13 direct-extraction FinanceBench questions via GraphRAG.")
    parser.add_argument(
        "--questions-path",
        default="scripts/financebench_direct_extract_13.jsonl",
        help="Path to the direct-extraction question set.",
    )
    parser.add_argument(
        "--output",
        default="output/direct_extract_13_batch_results.json",
        help="Path to save batch results.",
    )
    parser.add_argument(
        "--retriever-mode",
        choices=["text2cypher", "hybrid"],
        default="text2cypher",
        help="Retriever mode to use for the batch run.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top K candidates for hybrid retrieval.",
    )
    args = parser.parse_args()

    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = _require_env("NEO4J_PASSWORD")
    database = os.environ.get("NEO4J_DATABASE")
    vector_index = os.environ.get("NEO4J_VECTOR_INDEX", "chunk_embedding_index")
    fulltext_index = os.environ.get("NEO4J_FULLTEXT_INDEX", "chunk_fulltext_index")
    llm_model = os.environ.get("OLLAMA_LLM_MODEL", "qwen3:30b-a3b")
    embedding_model = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3:latest")
    ollama_host = os.environ.get("OLLAMA_HOST")

    ollama_kwargs = {}
    if ollama_host:
        ollama_kwargs["host"] = ollama_host

    questions = _load_questions(Path(args.questions_path))
    llm = OllamaLLM(
        model_name=llm_model,
        model_params={"options": {"temperature": 0}},
        **ollama_kwargs,
    )
    embedder = OllamaEmbeddings(model=embedding_model, **ollama_kwargs)

    retrieval_query = """
    WITH node AS chunk, score
    OPTIONAL MATCH (chunk)<-[:FROM_CHUNK]-(entity)
    WITH chunk, score, collect(DISTINCT {
        label: coalesce(entity.type, head(labels(entity))),
        name: coalesce(entity.title, entity.name, elementId(entity))
    }) AS entities
    RETURN chunk.text AS text,
           score AS similarity_score,
           entities
    """

    results = []
    with neo4j.GraphDatabase.driver(uri, auth=(user, password)) as driver:
        if args.retriever_mode == "text2cypher":
            retriever = Text2CypherRetriever(
                driver=driver,
                llm=llm,
                neo4j_database=database,
                neo4j_schema=FINANCIAL_GRAPH_SCHEMA,
                examples=TEXT2CYPHER_EXAMPLES,
                custom_prompt=TEXT2CYPHER_PROMPT,
            )
        else:
            retriever = HybridCypherRetriever(
                driver=driver,
                vector_index_name=vector_index,
                fulltext_index_name=fulltext_index,
                neo4j_database=database,
                retrieval_query=retrieval_query,
                embedder=embedder,
            )
        rag = GraphRAG(retriever=retriever, llm=llm)

        for idx, row in enumerate(questions, start=1):
            question_text = f"[doc_id={row['doc_id']}] {row['question']}"
            print(f"[{idx}/{len(questions)}] {row['doc_id']} :: {row['question_id']}")
            start = time.time()
            try:
                search_kwargs = {
                    "query_text": question_text,
                    "return_context": True,
                }
                if args.retriever_mode == "hybrid":
                    search_kwargs["retriever_config"] = {"top_k": args.top_k}

                result = rag.search(**search_kwargs)
                raw_answer = result.answer
                answer, evidence = _parse_answer(raw_answer)
                elapsed = round(time.time() - start, 2)
                results.append(
                    {
                        "question_id": row["question_id"],
                        "doc_id": row["doc_id"],
                        "doc_name": row["doc_name"],
                        "question": row["question"],
                        "gold_answer": row["gold_answer"],
                        "answer": answer,
                        "evidence": evidence,
                        "raw_answer": raw_answer,
                        "retriever_result": (
                            result.retriever_result.model_dump()
                            if result.retriever_result is not None
                            else None
                        ),
                        "time_seconds": elapsed,
                        "error": None,
                    }
                )
                print(f"  answer={answer!r} gold={row['gold_answer']}")
            except Exception as exc:
                elapsed = round(time.time() - start, 2)
                results.append(
                    {
                        "question_id": row["question_id"],
                        "doc_id": row["doc_id"],
                        "doc_name": row["doc_name"],
                        "question": row["question"],
                        "gold_answer": row["gold_answer"],
                        "answer": None,
                        "evidence": None,
                        "raw_answer": None,
                        "retriever_result": None,
                        "time_seconds": elapsed,
                        "error": str(exc),
                    }
                )
                print(f"  error={exc}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
