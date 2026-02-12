"""
CUAD Dataset Test Pipeline
===========================
Tests GraphRAG on first contract from CUAD dataset:
1. Build knowledge graph using SimpleKGPipeline (LLM auto-extraction, no schema)
2. Create vector index for retrieval
3. Run GraphRAG Q&A on all questions for the contract
4. Evaluate results using Jaccard similarity
5. Save results to JSONL

Usage:
    python cuad_dataset/test_pipeline.py

Environment variables required:
    NEO4J_PASSWORD - Neo4j database password
"""

import asyncio
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import neo4j
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.rag import GraphRAG
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.kg_builder.pipeline.kg_builder import SimpleKGPipeline

# Import evaluation functions
sys.path.insert(0, str(Path(__file__).parent.parent))
from cuad_dataset.evaluate import (
    evaluate_single_question,
    save_rag_results,
    compute_rag_summary,
)

# ==================== Configuration ====================
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j+s://998e2e0a.databases.neo4j.io")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
NEO4J_DATABASE = "neo4j"

# Vector index settings
VECTOR_INDEX_NAME = "chunk_vector_index"
EMBEDDING_DIMENSION = 1024  # bge-m3 dimension

# Data paths
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "data" / "train_separate_questions.json"
OUTPUT_DIR = SCRIPT_DIR / "results"

# LLM settings
LLM_MODEL = "qwen2.5:72b"
EMBEDDER_MODEL = "bge-m3:latest"


def require_env(name: str) -> str:
    """Get required environment variable or exit."""
    value = os.environ.get(name)
    if not value:
        print(f"Error: Missing environment variable: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def load_first_contract():
    """Load the first contract from CUAD dataset."""
    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    
    first_contract = data["data"][0]
    title = first_contract["title"]
    paragraph = first_contract["paragraphs"][0]
    context = paragraph["context"]
    qas = paragraph["qas"]
    
    print(f"Contract: {title}")
    print(f"Context length: {len(context)} characters")
    print(f"Questions: {len(qas)}")
    
    return title, context, qas


async def build_knowledge_graph(driver, llm, embedder, context: str, title: str):
    """Build knowledge graph from contract text using SimpleKGPipeline."""
    print("\n" + "=" * 50)
    print("Step 1: Building Knowledge Graph")
    print("=" * 50)
    
    # Create pipeline without schema (LLM will auto-extract entities/relations)
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        # No schema - let LLM extract freely
        from_pdf=False,
        neo4j_database=NEO4J_DATABASE,
    )
    
    print("Running KG builder pipeline...")
    start_time = time.time()
    
    result = await kg_builder.run_async(text=context)
    
    elapsed = time.time() - start_time
    print(f"KG built in {elapsed:.1f} seconds")
    print(f"Pipeline result: {result}")
    
    return result


def create_chunk_vector_index(driver, embedder):
    """Create vector index on Chunk nodes for retrieval."""
    print("\n" + "=" * 50)
    print("Step 2: Creating Vector Index")
    print("=" * 50)
    
    try:
        create_vector_index(
            driver,
            name=VECTOR_INDEX_NAME,
            label="Chunk",
            embedding_property="embedding",
            dimensions=EMBEDDING_DIMENSION,
            similarity_fn="cosine",
        )
        print(f"Created vector index: {VECTOR_INDEX_NAME}")
    except Exception as e:
        if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
            print(f"Vector index {VECTOR_INDEX_NAME} already exists, skipping...")
        else:
            raise


def setup_graphrag(driver, llm, embedder):
    """Set up GraphRAG with VectorCypherRetriever."""
    print("\n" + "=" * 50)
    print("Step 3: Setting up GraphRAG")
    print("=" * 50)
    
    # Retrieval query: get chunk text and related entities
    RETRIEVAL_QUERY = """
    WITH node AS chunk, score
    OPTIONAL MATCH (chunk)<-[:FROM_CHUNK]-(entity)
    WITH chunk, score, collect(DISTINCT {
        label: labels(entity)[0],
        name: entity.name
    }) AS entities
    RETURN chunk.text AS text,
           score AS similarity_score,
           entities
    """
    
    retriever = VectorCypherRetriever(
        driver=driver,
        index_name=VECTOR_INDEX_NAME,
        embedder=embedder,
        retrieval_query=RETRIEVAL_QUERY,
        neo4j_database=NEO4J_DATABASE,
    )
    
    rag = GraphRAG(retriever=retriever, llm=llm)
    print("GraphRAG initialized")
    
    return rag


def parse_llm_response(response: str) -> tuple:
    """Parse LLM response to extract answer and evidence.
    
    Expected format:
    {
        "final_answer": "...",
        "evidence": "..."
    }
    """
    try:
        # Try to parse as JSON
        data = json.loads(response)
        answer = data.get("final_answer", "")
        evidence = data.get("evidence", "")
        return answer, evidence
    except json.JSONDecodeError:
        # Fallback: treat entire response as answer
        return response.strip(), ""


def run_qa_evaluation(rag, qas: list, context: str, top_k: int = 5):
    """Run Q&A on all questions and evaluate results."""
    print("\n" + "=" * 50)
    print("Step 4: Running Q&A Evaluation")
    print("=" * 50)
    
    results = []
    total = len(qas)
    
    for i, qa in enumerate(qas, 1):
        question = qa["question"]
        print(f"\n[{i}/{total}] {question[:80]}...")
        
        start_time = time.time()
        try:
            # Run GraphRAG search
            result = rag.search(
                query_text=question,
                retriever_config={"top_k": top_k},
                return_context=False,
            )
            raw_answer = result.answer
            elapsed = time.time() - start_time
            
            # Parse response
            llm_answer, llm_evidence = parse_llm_response(raw_answer)
            
            # Evaluate
            eval_result = evaluate_single_question(
                qa=qa,
                context=context,
                llm_answer=llm_answer,
                llm_evidence=llm_evidence,
            )
            results.append(eval_result)
            
            # Print result
            status = "✓" if eval_result["is_correct"] else "✗"
            print(f"  {status} Answer: {llm_answer[:60]}..." if len(llm_answer) > 60 else f"  {status} Answer: {llm_answer}")
            print(f"  Gold: {eval_result['gold_answers'][:2]}")
            print(f"  Jaccard: {eval_result['jaccard_score']:.3f}, Time: {elapsed:.1f}s")
            
        except Exception as e:
            print(f"  Error: {e}")
            # Record error as incorrect
            eval_result = evaluate_single_question(
                qa=qa,
                context=context,
                llm_answer=f"ERROR: {e}",
                llm_evidence="",
            )
            results.append(eval_result)
    
    return results


def clear_database(driver):
    """Clear all nodes and relationships from the database."""
    print("Clearing existing data from Neo4j...")
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("Database cleared")


async def main():
    """Main pipeline execution."""
    print("=" * 60)
    print("CUAD GraphRAG Test Pipeline")
    print("=" * 60)
    
    # Check environment
    password = require_env("NEO4J_PASSWORD")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Initialize components
    print("\nInitializing LLM and Embedder...")
    llm = OllamaLLM(
        model_name=LLM_MODEL,
        model_params={"options": {"temperature": 0}},
    )
    embedder = OllamaEmbeddings(model=EMBEDDER_MODEL)
    print(f"LLM: {LLM_MODEL}")
    print(f"Embedder: {EMBEDDER_MODEL}")
    
    # Load data
    title, context, qas = load_first_contract()
    
    # Connect to Neo4j and run pipeline
    with neo4j.GraphDatabase.driver(NEO4J_URI, auth=("neo4j", password)) as driver:
        # Verify connection
        driver.verify_connectivity()
        print(f"\nConnected to Neo4j: {NEO4J_URI}")
        
        # Clear existing data
        clear_database(driver)
        
        # Step 1: Build KG
        await build_knowledge_graph(driver, llm, embedder, context, title)
        
        # Step 2: Create vector index
        create_chunk_vector_index(driver, embedder)
        
        # Wait a bit for index to be ready
        print("Waiting for index to be ready...")
        time.sleep(5)
        
        # Step 3: Setup GraphRAG
        rag = setup_graphrag(driver, llm, embedder)
        
        # Step 4: Run Q&A and evaluate
        results = run_qa_evaluation(rag, qas, context)
    
    # Step 5: Save results
    print("\n" + "=" * 50)
    print("Step 5: Saving Results")
    print("=" * 50)
    
    summary = save_rag_results(results, str(OUTPUT_DIR))
    
    # Print summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Total questions: {summary['total_questions']}")
    print(f"Correct (Jaccard >= 0.5): {summary['correct_count']} ({summary['accuracy']*100:.1f}%)")
    print(f"Exact match: {summary['exact_match_count']} ({summary['exact_match_rate']*100:.1f}%)")
    
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
