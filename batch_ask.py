from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

# Suppress deprecation warnings from neo4j_graphrag
warnings.filterwarnings("ignore", category=DeprecationWarning, module="neo4j_graphrag")

import neo4j
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import HybridCypherRetriever


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"Missing env var: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ask questions via GraphRAG.")
    parser.add_argument("questions_file", nargs="?", help="File with questions (one per line)")
    parser.add_argument("-q", "--question", action="append", help="Inline question (can use multiple times)")
    parser.add_argument("--top-k", type=int, default=5, help="Top K chunks to retrieve")
    parser.add_argument("-o", "--output", default="batch_results.json", help="Output file")
    args = parser.parse_args()

    # Collect questions
    questions = []
    if args.questions_file:
        with open(args.questions_file, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if args.question:
        questions.extend(args.question)
    
    if not questions:
        print("No questions provided. Use a file or -q flag.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(questions)} questions")

    # Setup
    uri = os.environ.get("NEO4J_URI", "neo4j+s://998e2e0a.databases.neo4j.io")
    password = _require_env("NEO4J_PASSWORD")
    vector_index = os.environ.get("NEO4J_VECTOR_INDEX", "chunk_embedding_index")
    fulltext_index = "chunk_fulltext_index"

    llm = OllamaLLM(
        model_name="qwen2.5:72b",
        model_params={"options": {"temperature": 0}},
    )

    embedder = OllamaEmbeddings(model="bge-m3:latest")

    RETRIEVAL_QUERY = """
    WITH node AS chunk, score
    OPTIONAL MATCH (chunk)<-[:FROM_CHUNK]-(entity)
    WITH chunk, score, collect(DISTINCT {
        label: labels(entity)[1],
        name: entity.name
    }) AS entities
    RETURN chunk.text AS text,
           score AS similarity_score,
           entities
    """

    results = []

    with neo4j.GraphDatabase.driver(uri, auth=("neo4j", password)) as driver:
        retriever = HybridCypherRetriever(
            driver=driver,
            vector_index_name=vector_index,
            fulltext_index_name=fulltext_index,
            retrieval_query=RETRIEVAL_QUERY,
            embedder=embedder,
        )
        rag = GraphRAG(retriever=retriever, llm=llm)

        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] {question}")
            start = time.time()
            
            try:
                result = rag.search(
                    query_text=question,
                    retriever_config={"top_k": args.top_k},
                    return_context=False,
                )
                answer = result.answer
                elapsed = time.time() - start
                print(f"  Answer: {answer[:100]}..." if len(answer) > 100 else f"  Answer: {answer}")
                print(f"  Time: {elapsed:.1f}s")
                
                results.append({
                    "question": question,
                    "answer": answer,
                    "time_seconds": round(elapsed, 2),
                    "error": None,
                })
            except Exception as e:
                print(f"  Error: {e}")
                results.append({
                    "question": question,
                    "answer": None,
                    "time_seconds": None,
                    "error": str(e),
                })

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"Results saved to {args.output}")
    
    # Summary
    success = sum(1 for r in results if r["error"] is None)
    total_time = sum(r["time_seconds"] or 0 for r in results)
    print(f"Success: {success}/{len(results)}")
    print(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
