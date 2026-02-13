"""
Evaluate RAG on Medical Questions

Reads questions from medical_questions.json, queries the RAG system,
and saves results for comparison with ground truth answers.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

# Load .env file if exists
env_file = Path(__file__).parents[2] / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

import neo4j
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.rag import GraphRAG
from neo4j_graphrag.rag.prompts import RagTemplate
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.retrievers import HybridCypherRetriever, VectorRetriever, VectorCypherRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Neo4j connection settings
URI = os.getenv("NEO4J_URI", "neo4j+s://998e2e0a.databases.neo4j.io")
AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# File paths
QUESTIONS_FILE = os.path.join(os.path.dirname(__file__), "medical_questions.json")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "eval_results.json")

# Retrieval query for graph-enhanced context
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


def load_questions(file_path: str, limit: int = None) -> list[dict]:
    """Load questions from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    if limit:
        questions = questions[:limit]
    
    logger.info(f"Loaded {len(questions)} questions")
    return questions


def setup_rag(driver: neo4j.Driver) -> GraphRAG:
    """Set up the RAG pipeline."""
    # Embedder
    embedder = OllamaEmbeddings(model="bge-m3:latest")
    
    # LLM with reduced context for speed
    llm = OllamaLLM(
        model_name="qwen2.5:72b",
        model_params={"options": {"temperature": 0, "num_ctx": 8192}},
    )
    
    # Custom prompt template for medical QA
    prompt_template = RagTemplate(
        template="""You are a medical question answering assistant.

Based on the provided context, answer the question accurately and concisely.

Rules:
1. If the context contains the answer, cite the relevant evidence.
2. If the context is partially relevant, use it and clearly state what is supported vs. inferred.
3. If the context does not contain relevant information, say "The provided context does not contain sufficient information to answer this question."
4. Do not guess or use information not present in the context.

Context:
{context}

Question:
{query_text}
{examples}
Answer:
""",
    )
    
    # Graph-enhanced retrieval query:
    # 1. Hybrid search (vector + fulltext) finds relevant chunks
    # 2. Graph traversal finds related entities from the KG
    retrieval_query = """
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
    
    # Retriever: Hybrid (vector + fulltext) + Graph expansion
    retriever = HybridCypherRetriever(
        driver=driver,
        vector_index_name="chunk_embedding_index",
        fulltext_index_name="chunk_fulltext_index",
        embedder=embedder,
        retrieval_query=retrieval_query,
    )
    
    # RAG pipeline with custom prompt
    rag = GraphRAG(retriever=retriever, llm=llm, prompt_template=prompt_template)
    
    return rag


def evaluate(questions: list[dict], rag: GraphRAG, top_k: int = 10) -> list[dict]:
    """Run RAG on each question and collect results."""
    results = []
    
    for i, q in enumerate(questions):
        logger.info(f"Processing question {i+1}/{len(questions)}: {q['question'][:50]}...")
        
        start_time = time.time()
        try:
            result = rag.search(
                query_text=q['question'],
                retriever_config={"top_k": top_k},
                return_context=False,
            )
            rag_answer = result.answer
            error = None
        except Exception as e:
            rag_answer = None
            error = str(e)
            logger.error(f"Error on question {q['id']}: {e}")
        
        elapsed = time.time() - start_time
        
        results.append({
            "id": q['id'],
            "question": q['question'],
            "ground_truth": q['answer'],
            "rag_answer": rag_answer,
            "question_type": q.get('question_type'),
            "evidence": q.get('evidence'),
            "elapsed_seconds": round(elapsed, 2),
            "error": error,
        })
        
        if rag_answer:
            logger.info(f"  Answer: {rag_answer[:100]}...")
    
    return results


def save_results(results: list[dict], file_path: str):
    """Save results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(results),
        "successful": sum(1 for r in results if r['rag_answer']),
        "failed": sum(1 for r in results if r['error']),
        "results": results,
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {file_path}")


def main(limit: int = None, top_k: int = 10):
    """Main evaluation function."""
    # Load questions
    questions = load_questions(QUESTIONS_FILE, limit=limit)
    
    # Connect to Neo4j and run evaluation
    with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
        logger.info("Connected to Neo4j")
        
        # Setup RAG
        rag = setup_rag(driver)
        
        # Run evaluation
        results = evaluate(questions, rag, top_k=top_k)
    
    # Save results
    save_results(results, RESULTS_FILE)
    
    # Print summary
    successful = sum(1 for r in results if r['rag_answer'])
    print(f"\n=== Evaluation Complete ===")
    print(f"Total: {len(results)}, Successful: {successful}, Failed: {len(results) - successful}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG on Medical Questions")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--top-k", type=int, default=10, help="Top K chunks to retrieve")
    
    args = parser.parse_args()
    main(limit=args.limit, top_k=args.top_k)
