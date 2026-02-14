"""
Evaluate RAG on Medical Questions

Reads questions from medical_questions.json, queries the RAG system,
and saves results for comparison with ground truth answers.
"""

import argparse
import json
import logging
import os
import random
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

# Use project/scratch cache for Hugging Face to avoid "No space left on device" on small home.
# Must set TMPDIR too: downloads write to temp dir first, often /tmp (on full disk).
_cache_dir = Path(__file__).parents[2] / ".cache" / "huggingface"
_cache_dir.mkdir(parents=True, exist_ok=True)
_tmp_dir = _cache_dir / "tmp"
_tmp_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_cache_dir))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_cache_dir))
os.environ.setdefault("HF_HUB_CACHE", str(_cache_dir))
os.environ.setdefault("TMPDIR", str(_tmp_dir))
os.environ.setdefault("TEMP", str(_tmp_dir))
os.environ.setdefault("TMP", str(_tmp_dir))

import neo4j
from neo4j_graphrag.embeddings import OllamaEmbeddings, SentenceTransformerEmbeddings
from neo4j_graphrag.rag import GraphRAG
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.retrievers import HybridCypherRetriever, VectorRetriever, VectorCypherRetriever

# Import custom prompts
from prompts import MEDICAL_RAG_TEMPLATE

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

# Max length for embed input to avoid Ollama/bge-m3 NaN issues
MAX_EMBED_QUERY_LENGTH = 8000


def sanitize_question(text: str) -> str:
    """Sanitize question text to reduce Ollama embed NaN/500 errors."""
    if not text or not isinstance(text, str):
        return "No question text"
    # Strip and normalize whitespace
    s = " ".join(text.strip().split())
    # Remove null bytes and other control chars that can break embed
    s = "".join(c for c in s if ord(c) >= 32 or c in "\n\t")
    # Truncate to avoid very long inputs that may trigger NaN in bge-m3
    if len(s) > MAX_EMBED_QUERY_LENGTH:
        s = s[:MAX_EMBED_QUERY_LENGTH].rsplit(" ", 1)[0] + "."
    return s if s else "No question text"


def load_questions(
    file_path: str,
    limit: int = None,
    question_type: str = None,
    sample_size: int = None,
    seed: int = None,
) -> list[dict]:
    """Load questions from JSON file. Optionally filter by question_type and random sample."""
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # Filter by question_type if specified
    if question_type is not None:
        questions = [q for q in questions if q.get('question_type') == question_type]
        logger.info(f"Filtered to question_type={question_type!r}: {len(questions)} questions")

    # Random sample if sample_size specified
    if sample_size is not None and len(questions) > sample_size:
        if seed is not None:
            random.seed(seed)
        questions = random.sample(questions, sample_size)
        logger.info(f"Randomly sampled {sample_size} questions")

    # Original behavior: limit to first N (when not using question_type/sample_size)
    # if limit:
    #     questions = questions[:limit]

    if limit and question_type is None and sample_size is None:
        questions = questions[:limit]

    logger.info(f"Loaded {len(questions)} questions")
    return questions


def setup_rag(driver: neo4j.Driver, embedder_type: str = "sentence-transformers") -> GraphRAG:
    """Set up the RAG pipeline.
    embedder_type: 'sentence-transformers' (default, no Ollama NaN) or 'ollama'
    """
    if embedder_type == "sentence-transformers":
        # Same model as bge-m3:latest; avoids Ollama embed NaN/500 entirely
        embedder = SentenceTransformerEmbeddings(model="BAAI/bge-m3")
        logger.info("Using SentenceTransformerEmbeddings (BAAI/bge-m3) for retrieval")
    else:
        embedder = OllamaEmbeddings(model="bge-m3:latest")
        logger.info("Using OllamaEmbeddings (bge-m3:latest) for retrieval")
    
    # LLM with reduced context for speed
    llm = OllamaLLM(
        model_name="qwen2.5:72b",
        model_params={"options": {"temperature": 0, "num_ctx": 8192}},
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
    
    # RAG pipeline with custom prompt from prompts.py
    rag = GraphRAG(retriever=retriever, llm=llm, prompt_template=MEDICAL_RAG_TEMPLATE)
    
    return rag


def evaluate(
    questions: list[dict],
    rag: GraphRAG,
    top_k: int = 10,
    results_file: str = RESULTS_FILE,
    for_evaluation: bool = False,
) -> list[dict]:
    """Run RAG on each question and collect results. Saves after each question.

    If for_evaluation=True: also returns context and saves generated_answer/context
    so the output can be fed to Evaluation/generation_eval.py and retrieval_eval.py.
    """
    results = []

    for i, q in enumerate(questions):
        raw_question = q["question"]
        query_text = sanitize_question(raw_question)
        logger.info(f"Processing question {i+1}/{len(questions)}: {query_text[:50]}...")

        start_time = time.time()
        rag_answer = None
        error = None
        retrieved_context = None  # for Evaluation (context list)
        for attempt in range(2):  # retry once on embed NaN/500
            try:
                # for_evaluation: return_context=True so we can save context for Evaluation/
                result = rag.search(
                    query_text=query_text,
                    retriever_config={"top_k": top_k},
                    return_context=for_evaluation,
                )
                rag_answer = result.answer
                if for_evaluation and hasattr(result, "context") and result.context:
                    retrieved_context = result.context if isinstance(result.context, list) else [result.context]
                break
            except Exception as e:
                error = str(e)
                if "NaN" in error or "500" in error:
                    logger.warning(f"Attempt {attempt + 1}/2 failed (embed error), retrying...")
                else:
                    logger.error(f"Error on question {q['id']}: {e}")
                    break
        if error and not rag_answer:
            logger.error(f"Error on question {q['id']}: {error}")

        elapsed = time.time() - start_time

        # Base fields (always saved)
        record = {
            "id": q["id"],
            "question": q["question"],
            "ground_truth": q["answer"],
            "rag_answer": rag_answer,
            "question_type": q.get("question_type"),
            "evidence": q.get("evidence"),
            "elapsed_seconds": round(elapsed, 2),
            "error": error,
        }
        # For Evaluation/: same record + generated_answer + context (list) so generation_eval/retrieval_eval can read it
        # if for_evaluation:
        #     record["generated_answer"] = rag_answer
        #     record["context"] = retrieved_context or []
        if for_evaluation:
            record["generated_answer"] = rag_answer
            record["context"] = retrieved_context if retrieved_context is not None else []

        results.append(record)

        if rag_answer:
            logger.info(f"  Answer: {rag_answer[:100]}...")

        # Save after each question (real-time progress)
        save_results(results, results_file, for_evaluation=for_evaluation)

    return results


def save_results(results: list[dict], file_path: str, for_evaluation: bool = False):
    """Save results to JSON file. If for_evaluation=True, also write a list-only file for Evaluation/."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(results),
        "successful": sum(1 for r in results if r.get("rag_answer")),
        "failed": sum(1 for r in results if r.get("error")),
        "results": results,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {file_path}")

    # Evaluation/generation_eval.py expects data_file to be a top-level list of items
    # if for_evaluation:
    #     eval_file = file_path.replace(".json", "_for_evaluation.json")
    #     with open(eval_file, "w", encoding="utf-8") as f:
    #         json.dump(results, f, indent=2, ensure_ascii=False)
    #     logger.info(f"Evaluation-ready list saved to {eval_file}")
    if for_evaluation:
        eval_file = file_path.replace(".json", "_for_evaluation.json")
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation-ready list saved to {eval_file}")


def main(
    limit: int = None,
    top_k: int = 10,
    question_type: str = None,
    sample_size: int = None,
    seed: int = None,
    embedder_type: str = "sentence-transformers",
    for_evaluation: bool = False,
):
    """Main evaluation function."""
    # Default: 20 random "Complex Reasoning". With --limit only: original behavior (first N).
    # questions = load_questions(QUESTIONS_FILE, limit=limit)  # original: first N
    if limit is not None and question_type is None:
        questions = load_questions(QUESTIONS_FILE, limit=limit)
    else:
        questions = load_questions(
            QUESTIONS_FILE,
            question_type=question_type or "Complex Reasoning",
            sample_size=sample_size if sample_size is not None else 20,
            seed=seed,
        )

    # Connect to Neo4j and run evaluation
    with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
        logger.info("Connected to Neo4j")

        # Setup RAG
        rag = setup_rag(driver, embedder_type=embedder_type)

        # Run evaluation (for_evaluation=True: save context + generated_answer for Evaluation/)
        results = evaluate(questions, rag, top_k=top_k, for_evaluation=for_evaluation)

    # Save results (for_evaluation already saved inside evaluate; here just ensure final save)
    save_results(results, RESULTS_FILE, for_evaluation=for_evaluation)
    
    # Print summary
    successful = sum(1 for r in results if r['rag_answer'])
    print(f"\n=== Evaluation Complete ===")
    print(f"Total: {len(results)}, Successful: {successful}, Failed: {len(results) - successful}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG on Medical Questions")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions (used when no --question-type)")
    parser.add_argument("--top-k", type=int, default=10, help="Top K chunks to retrieve")
    parser.add_argument("--question-type", type=str, default=None, help='Filter by question_type (default: "Complex Reasoning")')
    parser.add_argument("--sample", type=int, default=None, help="Random sample size (default: 20 when --question-type used)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    parser.add_argument(
        "--embedder",
        type=str,
        choices=["sentence-transformers", "ollama"],
        default="sentence-transformers",
        help="Embedder for retrieval: sentence-transformers (default, avoids Ollama NaN) or ollama",
    )
    parser.add_argument(
        "--for-evaluation",
        action="store_true",
        help="Save context + generated_answer and write eval_results_for_evaluation.json (list) for Evaluation/generation_eval.py",
    )

    args = parser.parse_args()
    main(
        limit=args.limit,
        top_k=args.top_k,
        question_type=args.question_type,
        sample_size=args.sample,
        seed=args.seed,
        embedder_type=args.embedder,
        for_evaluation=args.for_evaluation,
    )
