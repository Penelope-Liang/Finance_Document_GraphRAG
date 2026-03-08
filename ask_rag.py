"""Ask questions over the PDF-built graph using GraphRAG.

For Ollama (local):
- Make sure Ollama is running with qwen2.5:72b and bge-m3:latest

For Azure OpenAI, set env vars:
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_ENDPOINT
- OPENAI_API_VERSION

Required env vars:
- NEO4J_PASSWORD

Optional env vars:
- NEO4J_URI (default: bolt://127.0.0.1:7687)
- NEO4J_VECTOR_INDEX (default: chunk_embedding_index)
"""

from __future__ import annotations
import asyncio
import argparse
import os
import sys
import warnings
import neo4j
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.rag import GraphRAG
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever, HybridCypherRetriever

# Suppress deprecation warnings from neo4j_graphrag
warnings.filterwarnings("ignore", category=DeprecationWarning, module="neo4j_graphrag")

# === Original Azure OpenAI imports (commented out) ===
# from neo4j_graphrag.embeddings import AzureOpenAIEmbeddings
# from neo4j_graphrag.llm import AzureOpenAILLM



def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"Missing env var: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question via GraphRAG.")
    parser.add_argument("question", nargs="*", help="Question text")
    parser.add_argument("--top-k", type=int, default=5, help="Top K chunks to retrieve")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question and sys.stdin.isatty():
        question = input("Question: ").strip()
    if not question:
        print("No question provided.", file=sys.stderr)
        sys.exit(1)

    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    password = _require_env("NEO4J_PASSWORD")
    index_name = os.environ.get("NEO4J_VECTOR_INDEX", "chunk_embedding_index")

    # === Original Azure OpenAI LLM (commented out) ===
    # llm = AzureOpenAILLM(
    #     model_name=os.environ.get("AZURE_OPENAI_LLM_DEPLOYMENT", "Pathfinder_gpt-4"),
    #     api_key=_require_env("AZURE_OPENAI_API_KEY"),
    #     azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
    #     api_version=_require_env("OPENAI_API_VERSION"),
    #     model_params={"temperature": 0},
    # )

    # === Ollama LLM ===
    llm = OllamaLLM(
        model_name="qwen2.5:72b",
        model_params={
            "options": {
                "temperature": 0,
            },
        },
    )

    # === Original Azure OpenAI Embeddings (commented out) ===
    # embedder = AzureOpenAIEmbeddings(
    #     model=os.environ.get(
    #         "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
    #     ),
    #     api_key=_require_env("AZURE_OPENAI_API_KEY"),
    #     azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
    #     api_version=_require_env("OPENAI_API_VERSION"),
    # )

    # === Ollama Embeddings ===
    embedder = OllamaEmbeddings(
        model="bge-m3:latest",
    )

    # === Graph-enhanced retrieval query ===
    # After vector search finds Chunk, traverse graph to get related entities
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

    fulltext_index_name = "chunk_fulltext_index"

    try:
        with neo4j.GraphDatabase.driver(uri, auth=("neo4j", password)) as driver:
            # === VectorRetriever (vector only, no graph) ===
            # retriever = VectorRetriever(driver, index_name, embedder)
            
            # === VectorCypherRetriever (vector + graph traversal) ===
            # retriever = VectorCypherRetriever(
            #     driver=driver,
            #     index_name=index_name,
            #     embedder=embedder,
            #     retrieval_query=RETRIEVAL_QUERY,
            # )

            # === HybridCypherRetriever (vector + fulltext + graph traversal) ===
            retriever = HybridCypherRetriever(
                driver=driver,
                vector_index_name=index_name,
                fulltext_index_name=fulltext_index_name,
                retrieval_query=RETRIEVAL_QUERY,
                embedder=embedder,
            )
            
            rag = GraphRAG(retriever=retriever, llm=llm)
            result = rag.search(
                query_text=question,
                retriever_config={"top_k": args.top_k},
                return_context=False,  # Explicitly set to suppress deprecation warning
            )
            print(result.answer)
    finally:
        # Close Ollama client connections to avoid ResourceWarning
        if hasattr(llm, 'client') and hasattr(llm.client, '_client'):
            llm.client._client.close()
        if hasattr(embedder, 'client') and hasattr(embedder.client, '_client'):
            embedder.client._client.close()


if __name__ == "__main__":
    main()
