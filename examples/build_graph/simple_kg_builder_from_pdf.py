"""This example illustrates how to get started easily with the SimpleKGPipeline
and ingest PDF into a Neo4j Knowledge Graph.

This example assumes a Neo4j db is up and running. Update the credentials below
if needed.

For Ollama (local):
- Make sure Ollama is running: `ollama serve`
- Pull the models: `ollama pull llama3.2` and `ollama pull nomic-embed-text`

Azure OpenAI env vars:
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_ENDPOINT
- OPENAI_API_VERSION
"""

import asyncio
import logging
import os
from pathlib import Path

import neo4j

# Enable logging to see pipeline progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# === Original Azure OpenAI imports (commented out) ===
# from neo4j_graphrag.embeddings import AzureOpenAIEmbeddings
# from neo4j_graphrag.llm import AzureOpenAILLM
# from neo4j_graphrag.utils.rate_limit import RetryRateLimitHandler

# === Ollama imports ===
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.llm import OllamaLLM

from neo4j_graphrag.kg_builder.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.kg_builder.pipeline.pipeline import PipelineResult
from neo4j_graphrag.kg_builder.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.llm import LLMInterfaceV2

# === Entity Resolution imports ===
# 方式1: 精确匹配（默认）
# from neo4j_graphrag.kg_builder.components.resolver import SinglePropertyExactMatchResolver
# 方式2: 模糊匹配（需要 rapidfuzz）
# from neo4j_graphrag.kg_builder.components.resolver import FuzzyMatchResolver
# 方式3: 语义匹配（需要 spacy）
from neo4j_graphrag.kg_builder.components.resolver import SpaCySemanticMatchResolver

# Neo4j db infos
URI = os.environ.get("NEO4J_URI", "neo4j+s://998e2e0a.databases.neo4j.io")
AUTH = ("neo4j", os.environ.get("NEO4J_PASSWORD", ""))
DATABASE = "neo4j"


root_dir = Path(__file__).parents[1]
file_path = root_dir / "data" / "wipo_pub_rn2021_18e.pdf"


# Instantiate NodeType and RelationshipType objects. This defines the
# entities and relations the LLM will be looking for in the text.
NODE_TYPES = ["Person", "Organization", "Location"]
RELATIONSHIP_TYPES = ["SITUATED_AT", "INTERACTS", "LED_BY"]
PATTERNS = [
    ("Person", "SITUATED_AT", "Location"),
    ("Person", "INTERACTS", "Person"),
    ("Organization", "LED_BY", "Person"),
]


async def define_and_run_pipeline(
    neo4j_driver: neo4j.Driver,
    llm: LLMInterfaceV2,
) -> PipelineResult:
    print("[1/4] Initializing knowledge graph builder...")
    
    # Create an instance of the SimpleKGPipeline
    # 禁用默认的精确匹配实体消解，后面单独用语义匹配
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=neo4j_driver,
        # === Original Azure OpenAI Embeddings (commented out) ===
        # embedder=AzureOpenAIEmbeddings(
        #     model="text-embedding-3-large",
        #     api_key=os.environ["AZURE_OPENAI_API_KEY"],
        #     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        #     api_version=os.environ["OPENAI_API_VERSION"],
        #     rate_limit_handler=RetryRateLimitHandler(
        #                             max_attempts=10,
        #                             min_wait=1.0,
        #                             max_wait=60.0,
        #                             multiplier=2.0,
        #                             jitter=True,
        #                         ),
        # ),
        # === Ollama Embeddings ===
        embedder=OllamaEmbeddings(
            model="bge-m3:latest",
        ),
        # Use smaller chunk size for Ollama embedding models
        text_splitter=FixedSizeSplitter(chunk_size=512, chunk_overlap=50),
        schema={
            "node_types": NODE_TYPES,
            "relationship_types": RELATIONSHIP_TYPES,
            "patterns": PATTERNS,
        },
        neo4j_database=DATABASE,
        # === Entity Resolution 配置 ===
        # 设为 False，禁用默认的精确匹配，后面用语义匹配
        perform_entity_resolution=False,
        # 如果想用默认的精确匹配，改为 True
        # perform_entity_resolution=True,
    )
    
    print("[2/4] Running knowledge graph pipeline (this may take a while)...")
    
    result = await kg_builder.run_async(
        file_path=str(file_path),
        # optional, add document metadata, each item will
        # be saved as a property of the Document node
        # document_metadata={"author": "J. K. Rowling"},
    )
    
    print("[3/4] Running semantic entity resolution...")
    
    # === 语义匹配实体消解 ===
    # 使用 SpaCy 语义相似度合并相似实体
    resolver = SpaCySemanticMatchResolver(
        driver=neo4j_driver,
        neo4j_database=DATABASE,
        resolve_property="name",  # 基于 name 属性做语义匹配
        similarity_threshold=0.9,  # 相似度阈值（0-1，越高越严格）
    )
    resolution_result = await resolver.run()
    print(f"    Entity resolution: {resolution_result}")
    
    # === 其他实体消解方式（注释掉的备选方案） ===
    # 方式1: 精确匹配
    # resolver = SinglePropertyExactMatchResolver(
    #     driver=neo4j_driver,
    #     neo4j_database=DATABASE,
    # )
    
    # 方式2: 模糊匹配
    # resolver = FuzzyMatchResolver(
    #     driver=neo4j_driver,
    #     neo4j_database=DATABASE,
    #     threshold=0.85,  # 字符串相似度阈值
    # )
    
    print("[4/4] Processing completed!")
    
    return result


async def main() -> PipelineResult:
    # === Original Azure OpenAI LLM (commented out) ===
    # llm = AzureOpenAILLM(
    #     # Azure deployment name (not the model name)
    #     model_name="Pathfinder_gpt-4",
    #     model_params={
    #         "max_tokens": 2000,
    #         "response_format": {"type": "json_object"},
    #     },
    #     api_key=os.environ["AZURE_OPENAI_API_KEY"],
    #     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    #     api_version=os.environ["OPENAI_API_VERSION"],
    #     rate_limit_handler=RetryRateLimitHandler(
    #         max_attempts=10,
    #         min_wait=60.0,
    #         max_wait=300.0,
    #         multiplier=1.5,
    #     ),
    # )

    # === Ollama LLM (local model) ===
    # Using qwen2.5:72b
    llm = OllamaLLM(
        model_name="qwen2.5:72b",
        model_params={
            "options": {
                "temperature": 0,
            },
            "format": "json",
        },
    )

    with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
        res = await define_and_run_pipeline(driver, llm)
    # Ollama AsyncClient doesn't need explicit close
    return res


if __name__ == "__main__":
    res = asyncio.run(main())
    print(res)
