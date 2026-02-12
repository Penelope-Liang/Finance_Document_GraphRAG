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
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import neo4j
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
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
# Note: SimpleKGPipeline auto-creates index named "chunk_embedding_index"
VECTOR_INDEX_NAME = "chunk_embedding_index"
EMBEDDING_DIMENSION = 1024  # bge-m3 dimension

# Data paths
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "data" / "train_separate_questions.json"
OUTPUT_DIR = SCRIPT_DIR / "results"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
LIVE_JSONL_FILE = OUTPUT_DIR / "live_results.jsonl"  # Real-time output

# LLM settings
LLM_MODEL = "qwen2.5:72b"
EMBEDDER_MODEL = "bge-m3:latest"

# Checkpoint settings
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N questions


def require_env(name: str) -> str:
    """Get required environment variable or exit."""
    value = os.environ.get(name)
    if not value:
        print(f"Error: Missing environment variable: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def load_first_contract():
    """Load the first contract from CUAD dataset."""
    logger.info(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    
    first_contract = data["data"][0]
    title = first_contract["title"]
    paragraph = first_contract["paragraphs"][0]
    context = paragraph["context"]
    qas = paragraph["qas"]
    
    logger.info(f"Contract: {title}")
    logger.info(f"Context length: {len(context):,} characters")
    logger.info(f"Questions: {len(qas)}")
    
    return title, context, qas


def save_checkpoint(results: list, start_idx: int, stage: str):
    """Save checkpoint to resume later."""
    checkpoint = {
        "stage": stage,
        "start_idx": start_idx,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)
    logger.info(f"Checkpoint saved: stage={stage}, processed={len(results)} questions")


def load_checkpoint():
    """Load checkpoint if exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            checkpoint = json.load(f)
        logger.info(f"Found checkpoint: stage={checkpoint['stage']}, processed={len(checkpoint['results'])} questions")
        return checkpoint
    return None


async def build_knowledge_graph(driver, llm, embedder, context: str, title: str):
    """Build knowledge graph from contract text using SimpleKGPipeline."""
    logger.info("=" * 50)
    logger.info("Step 1: Building Knowledge Graph")
    logger.info("=" * 50)
    
    # Create pipeline with schema="FREE" to skip auto-extraction
    # This avoids SchemaExtractionError when LLM returns invalid JSON
    logger.info("Creating SimpleKGPipeline with schema='FREE'...")
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        schema="FREE",  # Skip auto schema extraction, let LLM extract freely
        from_pdf=False,
        neo4j_database=NEO4J_DATABASE,
    )
    logger.info("SimpleKGPipeline created successfully")
    
    logger.info(f"Running KG builder on {len(context):,} chars of text...")
    start_time = time.time()
    
    try:
        result = await kg_builder.run_async(text=context)
        elapsed = time.time() - start_time
        logger.info(f"KG built successfully in {elapsed:.1f} seconds")
        logger.info(f"Pipeline result: {result}")
        
        # Verify nodes were created
        with driver.session(database=NEO4J_DATABASE) as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
            chunk_count = session.run("MATCH (n:Chunk) RETURN count(n) AS cnt").single()["cnt"]
            logger.info(f"Database now has {node_count} nodes, {chunk_count} Chunk nodes")
        
        return result
    except Exception as e:
        logger.error(f"KG build failed after {time.time() - start_time:.1f}s: {e}")
        raise


def create_chunk_vector_index(driver, embedder):
    """Verify vector index exists (SimpleKGPipeline creates it automatically)."""
    logger.info("=" * 50)
    logger.info("Step 2: Verifying Vector Index")
    logger.info("=" * 50)
    
    # SimpleKGPipeline already creates chunk_embedding_index
    # Just verify it exists and is ready
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(
            "SHOW INDEXES WHERE name = $name",
            name=VECTOR_INDEX_NAME
        ).single()
        if result:
            logger.info(f"Index '{VECTOR_INDEX_NAME}' found, status: {result.get('state', 'unknown')}")
        else:
            # Fallback: create if not exists
            logger.info(f"Index '{VECTOR_INDEX_NAME}' not found, creating...")
            try:
                create_vector_index(
                    driver,
                    name=VECTOR_INDEX_NAME,
                    label="Chunk",
                    embedding_property="embedding",
                    dimensions=EMBEDDING_DIMENSION,
                    similarity_fn="cosine",
                )
                logger.info(f"Created vector index: {VECTOR_INDEX_NAME}")
                time.sleep(3)  # Wait for index to populate
            except Exception as e:
                if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
                    logger.info(f"Vector index already exists")
                else:
                    raise


def setup_graphrag(driver, llm, embedder):
    """Set up GraphRAG with VectorCypherRetriever."""
    logger.info("=" * 50)
    logger.info("Step 3: Setting up GraphRAG")
    logger.info("=" * 50)
    
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
    
    logger.info("Creating VectorCypherRetriever...")
    retriever = VectorCypherRetriever(
        driver=driver,
        index_name=VECTOR_INDEX_NAME,
        embedder=embedder,
        retrieval_query=RETRIEVAL_QUERY,
        neo4j_database=NEO4J_DATABASE,
    )
    
    logger.info("Creating GraphRAG...")
    rag = GraphRAG(retriever=retriever, llm=llm)
    logger.info("GraphRAG initialized successfully")
    
    # Test retrieval with a simple query
    logger.info("Testing retrieval...")
    try:
        test_result = retriever.search(query_text="agreement", top_k=1)
        logger.info(f"Retrieval test: found {len(test_result.items)} items")
    except Exception as e:
        logger.warning(f"Retrieval test failed: {e}")
    
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


def run_qa_evaluation(rag, qas: list, context: str, top_k: int = 5, start_idx: int = 0, existing_results: list = None):
    """Run Q&A on all questions and evaluate results."""
    logger.info("=" * 50)
    logger.info("Step 4: Running Q&A Evaluation")
    logger.info("=" * 50)
    
    results = existing_results if existing_results else []
    total = len(qas)
    correct_count = sum(1 for r in results if r.get("is_correct", False))
    
    # Initialize live JSONL file (clear if starting fresh, keep if resuming)
    if start_idx == 0:
        LIVE_JSONL_FILE.write_text("")  # Clear file
        logger.info(f"Live results will be written to: {LIVE_JSONL_FILE}")
    else:
        logger.info(f"Resuming from question {start_idx + 1}, already processed {len(results)} questions")
    
    # Use tqdm for progress bar
    pbar = tqdm(enumerate(qas[start_idx:], start_idx + 1), 
                total=total, 
                initial=start_idx,
                desc="Q&A Progress",
                unit="q")
    
    for i, qa in pbar:
        question = qa["question"]
        q_type = qa["id"].split("/")[-1] if "/" in qa["id"] else "unknown"
        
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
            
            # Update stats
            if eval_result["is_correct"]:
                correct_count += 1
            
            # Write to live JSONL immediately
            with open(LIVE_JSONL_FILE, "a") as f:
                f.write(json.dumps(eval_result, ensure_ascii=False) + "\n")
            
            # Update progress bar
            accuracy = correct_count / len(results) * 100
            pbar.set_postfix({
                "acc": f"{accuracy:.1f}%",
                "time": f"{elapsed:.1f}s",
                "type": q_type[:15]
            })
            
            # Log every question
            status = "✓" if eval_result["is_correct"] else "✗"
            logger.debug(f"[{i}/{total}] {status} J={eval_result['jaccard_score']:.2f} | {question[:50]}...")
            
        except Exception as e:
            logger.error(f"[{i}/{total}] Error: {e}")
            # Record error as incorrect
            eval_result = evaluate_single_question(
                qa=qa,
                context=context,
                llm_answer=f"ERROR: {e}",
                llm_evidence="",
            )
            results.append(eval_result)
            # Also write error to live JSONL
            with open(LIVE_JSONL_FILE, "a") as f:
                f.write(json.dumps(eval_result, ensure_ascii=False) + "\n")
        
        # Save checkpoint periodically
        if i % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(results, i, "qa_evaluation")
    
    pbar.close()
    
    # Final stats
    logger.info(f"Q&A complete: {len(results)} questions, {correct_count} correct ({correct_count/len(results)*100:.1f}%)")
    
    return results


def clear_database(driver):
    """Clear all nodes and relationships from the database."""
    logger.info("Clearing existing data from Neo4j...")
    with driver.session(database=NEO4J_DATABASE) as session:
        # Count before
        count_before = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
        logger.info(f"Found {count_before} existing nodes")
        
        if count_before > 0:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")
        else:
            logger.info("Database already empty")


async def main():
    """Main pipeline execution."""
    logger.info("=" * 60)
    logger.info("CUAD GraphRAG Test Pipeline")
    logger.info(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # Check environment
    password = require_env("NEO4J_PASSWORD")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint()
    start_idx = 0
    existing_results = None
    skip_kg_build = False
    
    if checkpoint:
        stage = checkpoint.get("stage", "")
        if stage == "qa_evaluation":
            logger.info("Found Q&A checkpoint, will resume from there")
            start_idx = checkpoint["start_idx"]
            existing_results = checkpoint["results"]
            skip_kg_build = True
        elif stage in ("kg_build_complete", "index_created"):
            logger.info(f"Found checkpoint at stage '{stage}', skipping KG build")
            skip_kg_build = True
    
    # Initialize components
    logger.info("Initializing LLM and Embedder...")
    llm = OllamaLLM(
        model_name=LLM_MODEL,
        model_params={"options": {"temperature": 0}},
    )
    embedder = OllamaEmbeddings(model=EMBEDDER_MODEL)
    logger.info(f"LLM: {LLM_MODEL}")
    logger.info(f"Embedder: {EMBEDDER_MODEL}")
    
    # Test LLM connection
    logger.info("Testing LLM connection...")
    try:
        test_response = llm.invoke("Say 'OK' if you're working.")
        logger.info(f"LLM test response: {test_response.content[:50]}...")
    except Exception as e:
        logger.error(f"LLM connection failed: {e}")
        sys.exit(1)
    
    # Load data
    title, context, qas = load_first_contract()
    
    # Connect to Neo4j and run pipeline
    logger.info(f"Connecting to Neo4j: {NEO4J_URI}")
    with neo4j.GraphDatabase.driver(NEO4J_URI, auth=("neo4j", password)) as driver:
        # Verify connection
        try:
            driver.verify_connectivity()
            logger.info("Neo4j connection verified")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            sys.exit(1)
        
        if not skip_kg_build:
            # Clear existing data
            clear_database(driver)
            
            # Step 1: Build KG
            save_checkpoint([], 0, "kg_build_start")
            await build_knowledge_graph(driver, llm, embedder, context, title)
            save_checkpoint([], 0, "kg_build_complete")
            
            # Step 2: Create vector index
            create_chunk_vector_index(driver, embedder)
            save_checkpoint([], 0, "index_created")
            
            # Wait a bit for index to be ready
            logger.info("Waiting 5s for index to be ready...")
            time.sleep(5)
        else:
            logger.info("Skipping KG build (resuming from checkpoint)")
        
        # Step 3: Setup GraphRAG
        rag = setup_graphrag(driver, llm, embedder)
        
        # Step 4: Run Q&A and evaluate
        results = run_qa_evaluation(rag, qas, context, start_idx=start_idx, existing_results=existing_results)
    
    # Step 5: Save results
    logger.info("=" * 50)
    logger.info("Step 5: Saving Results")
    logger.info("=" * 50)
    
    summary = save_rag_results(results, str(OUTPUT_DIR))
    
    # Delete checkpoint file after successful completion
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint file removed (pipeline complete)")
    
    # Print summary
    logger.info("=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total questions: {summary['total_questions']}")
    logger.info(f"Correct (Jaccard >= 0.5): {summary['correct_count']} ({summary['accuracy']*100:.1f}%)")
    logger.info(f"Exact match: {summary['exact_match_count']} ({summary['exact_match_rate']*100:.1f}%)")
    logger.info(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
