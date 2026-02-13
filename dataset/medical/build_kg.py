"""
Medical Knowledge Graph Builder

This script builds a Knowledge Graph from the medical corpus using SimpleKGPipeline.
The corpus contains medical text about skin cancer and other medical topics.

Prerequisites:
- Neo4j database running
- Ollama running with qwen2.5:72b and bge-m3:latest models
"""

import asyncio
import json
import logging
import os
import time
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
from neo4j_graphrag.kg_builder.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.kg_builder.pipeline.pipeline import PipelineResult
from neo4j_graphrag.kg_builder.pipeline.types.schema import (
    EntityInputType,
    RelationInputType,
)
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.llm import OllamaLLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("neo4j_graphrag").setLevel(logging.DEBUG)

# Neo4j connection settings
URI = os.getenv("NEO4J_URI", "neo4j+s://998e2e0a.databases.neo4j.io")
AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Corpus file path
CORPUS_FILE = os.path.join(os.path.dirname(__file__), "Datasets_Corpus_medical.txt")

# Maximum text length per chunk for processing (to avoid LLM token limits)
MAX_CHUNK_SIZE = 50000  # characters


# Define medical entity types
NODE_TYPES: list[EntityInputType] = [
    {"label": "Disease", "description": "A medical condition or illness, including cancer types"},
    {"label": "Symptom", "description": "Signs or symptoms of a disease or condition"},
    {"label": "Treatment", "description": "Medical treatments, therapies, or procedures"},
    {"label": "Medication", "description": "Drugs or medicines used for treatment"},
    {"label": "BodyPart", "description": "Parts of the human body, organs, or anatomical structures"},
    {"label": "Cell", "description": "Types of cells in the body"},
    {"label": "RiskFactor", "description": "Factors that increase the risk of a disease"},
    {"label": "DiagnosticTest", "description": "Medical tests or examinations used for diagnosis"},
    {"label": "Stage", "description": "Stages or grades of a disease"},
    {"label": "Organization", "description": "Medical organizations, hospitals, or institutions"},
]

# Define medical relationship types
RELATIONSHIP_TYPES: list[RelationInputType] = [
    {"label": "CAUSES", "description": "One entity causes another (e.g., risk factor causes disease)"},
    {"label": "SYMPTOM_OF", "description": "A symptom is associated with a disease"},
    {"label": "TREATS", "description": "A treatment or medication treats a disease"},
    {"label": "LOCATED_IN", "description": "An entity is located in a body part"},
    {"label": "AFFECTS", "description": "A disease or condition affects a body part"},
    {"label": "DIAGNOSES", "description": "A diagnostic test diagnoses a disease"},
    {"label": "HAS_STAGE", "description": "A disease has a specific stage"},
    {"label": "DEVELOPS_INTO", "description": "One condition develops into another"},
    {"label": "CONTAINS", "description": "A body part contains cells or other structures"},
    {"label": "RECOMMENDS", "description": "An organization recommends a treatment or guideline"},
    {"label": "RISK_FOR", "description": "A risk factor increases risk for a disease"},
    {"label": "TYPE_OF", "description": "One entity is a type or subtype of another"},
]

# Define allowed relationship patterns
PATTERNS = [
    ("RiskFactor", "CAUSES", "Disease"),
    ("RiskFactor", "RISK_FOR", "Disease"),
    ("Symptom", "SYMPTOM_OF", "Disease"),
    ("Treatment", "TREATS", "Disease"),
    ("Medication", "TREATS", "Disease"),
    ("Disease", "LOCATED_IN", "BodyPart"),
    ("Disease", "AFFECTS", "BodyPart"),
    ("Cell", "LOCATED_IN", "BodyPart"),
    ("BodyPart", "CONTAINS", "Cell"),
    ("DiagnosticTest", "DIAGNOSES", "Disease"),
    ("Disease", "HAS_STAGE", "Stage"),
    ("Disease", "DEVELOPS_INTO", "Disease"),
    ("Organization", "RECOMMENDS", "Treatment"),
    ("Disease", "TYPE_OF", "Disease"),
    ("Cell", "TYPE_OF", "Cell"),
]


def load_corpus(file_path: str) -> str:
    """Load the medical corpus from JSON file."""
    logger.info(f"Loading corpus from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract context from the JSON structure
    if isinstance(data, list) and len(data) > 0:
        context = data[0].get('context', '')
        logger.info(f"Loaded corpus: {len(context)} characters")
        return context
    else:
        raise ValueError("Unexpected corpus format")


def chunk_text(text: str, chunk_size: int = MAX_CHUNK_SIZE, overlap: int = 1000) -> list[str]:
    """Split text into overlapping chunks for processing."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # overlap for context continuity
    
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks


async def build_kg_from_chunk(
    chunk: str,
    chunk_index: int,
    neo4j_driver: neo4j.Driver,
    llm: LLMInterface,
) -> PipelineResult:
    """Build KG from a single text chunk."""
    logger.info(f"Processing chunk {chunk_index + 1}, length: {len(chunk)} chars")
    
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=neo4j_driver,
        embedder=OllamaEmbeddings(model="bge-m3:latest"),
        schema={
            "node_types": NODE_TYPES,
            "relationship_types": RELATIONSHIP_TYPES,
            "patterns": PATTERNS,
            "additional_node_types": True,
            "additional_relationship_types": True,
            "additional_patterns": True,
        },
        from_pdf=False,
        neo4j_database=DATABASE,
    )
    
    return await kg_builder.run_async(
        text=chunk,
        document_metadata={
            "source": "medical_corpus",
            "chunk_index": str(chunk_index),
        },
    )


async def main(max_chunks: int = None) -> None:
    """
    Main function to build the medical Knowledge Graph.
    
    Args:
        max_chunks: Maximum number of chunks to process (None for all)
    """
    # Load corpus
    corpus_text = load_corpus(CORPUS_FILE)
    
    # Split into chunks
    chunks = chunk_text(corpus_text)
    
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
        logger.info(f"Processing only first {max_chunks} chunks")
    
    # Initialize LLM (Ollama + Qwen 72B)
    # num_ctx=8192: reduce context length to lower VRAM usage and speed up inference
    # 8192 is sufficient for chunk extraction (~4000 chars per chunk + prompt)
    llm = OllamaLLM(
        model_name="qwen2.5:72b",
        model_params={"options": {"temperature": 0, "num_ctx": 8192}},
    )
    
    # Connect to Neo4j and process chunks
    total_start_time = time.time()
    
    with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
        logger.info("Connected to Neo4j")
        
        for i, chunk in enumerate(chunks):
            chunk_start_time = time.time()
            try:
                result = await build_kg_from_chunk(chunk, i, driver, llm)
                chunk_time = time.time() - chunk_start_time
                logger.info(f"Chunk {i + 1}/{len(chunks)} processed successfully in {chunk_time:.2f}s")
            except Exception as e:
                logger.error(f"Error processing chunk {i + 1}: {e}")
                continue
    
    total_time = time.time() - total_start_time
    logger.info(f"Knowledge Graph building completed in {total_time:.2f}s ({total_time/60:.2f} min)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Medical Knowledge Graph")
    parser.add_argument(
        "--max-chunks", 
        type=int, 
        default=None,
        help="Maximum number of chunks to process (default: all)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=MAX_CHUNK_SIZE,
        help=f"Size of each text chunk in characters (default: {MAX_CHUNK_SIZE})"
    )
    
    args = parser.parse_args()
    
    if args.chunk_size != MAX_CHUNK_SIZE:
        MAX_CHUNK_SIZE = args.chunk_size
    
    asyncio.run(main(max_chunks=args.max_chunks))
