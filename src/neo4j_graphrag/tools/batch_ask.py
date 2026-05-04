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
from neo4j_graphrag.rag import GraphRAG
from neo4j_graphrag.retrievers import HybridCypherRetriever, Text2CypherRetriever


FINANCIAL_GRAPH_SCHEMA = """
Nodes:
- Chunk {chunk_id, doc_id, section_name, text, chunk_index, parser_engine, chunk_engine}
- Table {table_id, chunk_id, doc_id, section_name}
- TableCell {cell_id, chunk_id, doc_id, section_name, row_header, col_header, text, numeric_value, numeric_kind, scale, normalized_value}

Relationships:
- (:Chunk)-[:HAS_TABLE]->(:Table)
- (:Table)-[:HAS_CELL]->(:TableCell)

Important data conventions:
- Financial statement answers often live on TableCell nodes.
- Use doc_id to restrict to the target filing, e.g. "3M_2018_10K.pdf" or "AMAZON_2019_10K.pdf".
- Prefer primary statement sections over notes when ranking evidence:
  - balance sheet
  - statements of operations / income statement
  - cash flows / cash flow statement
- Typical table retrieval pattern:
  MATCH (ch:Chunk {doc_id: "...pdf"})-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
  WHERE ...
  RETURN ch.section_name, cell.row_header, cell.col_header, cell.text, cell.numeric_value, cell.numeric_kind, cell.scale
- For direct extraction questions, favor exact row_header matches and exact year/period col_header matches.
- Use LIMIT 1 when the question asks for one final answer.
"""


TEXT2CYPHER_PROMPT = """
Task: Generate a Cypher statement for answering a financial-report question over a Neo4j graph.

Schema:
{schema}

Examples:
{examples}

Rules:
- Return only Cypher. No prose.
- Prefer TableCell evidence over Chunk text for metrics questions.
- Always filter to the target filing using doc_id when the filing can be inferred from the question.
- For financial statements, prefer primary statement sections over note sections when possible.
- Favor exact row_header matches first. Use col_header to match the requested fiscal year or period.
- Return the minimal evidence needed to answer the question.
- For direct extraction questions, return:
  ch.section_name AS section_name,
  cell.row_header AS row_header,
  cell.col_header AS col_header,
  cell.text AS text,
  cell.numeric_value AS numeric_value,
  cell.numeric_kind AS numeric_kind,
  cell.scale AS scale
- Use LIMIT 1 unless the question clearly requires multiple rows.

Question:
{query_text}

Cypher query:
"""


TEXT2CYPHER_EXAMPLES = [
    """User: What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement.
Cypher:
MATCH (ch:Chunk {doc_id: "3M_2018_10K.pdf"})-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WHERE cell.row_header = "Purchases of property, plant and equipment (PP&E)"
  AND cell.col_header CONTAINS "2018"
RETURN ch.section_name AS section_name,
       cell.row_header AS row_header,
       cell.col_header AS col_header,
       cell.text AS text,
       cell.numeric_value AS numeric_value,
       cell.numeric_kind AS numeric_kind,
       cell.scale AS scale
LIMIT 1""",
    """User: Assume that you are a public equities analyst. Answer the following question by primarily using information that is shown in the balance sheet: what is the year end FY2018 net PPNE for 3M? Answer in USD billions.
Cypher:
MATCH (ch:Chunk {doc_id: "3M_2018_10K.pdf"})-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WHERE replace(toLower(cell.row_header), " ", "") CONTAINS "property,plantandequipment-net"
  AND cell.col_header CONTAINS "2018"
RETURN ch.section_name AS section_name,
       cell.row_header AS row_header,
       cell.col_header AS col_header,
       cell.text AS text,
       cell.numeric_value AS numeric_value,
       cell.numeric_kind AS numeric_kind,
       cell.scale AS scale
LIMIT 1""",
    """User: By drawing conclusions from the information stated only in the income statement, what is Amazon's FY2019 net income attributable to shareholders (in USD millions)?
Cypher:
MATCH (ch:Chunk {doc_id: "AMAZON_2019_10K.pdf"})-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WHERE cell.row_header = "Net income"
  AND cell.col_header CONTAINS "2019"
  AND toLower(ch.section_name) CONTAINS "statements of operations"
RETURN ch.section_name AS section_name,
       cell.row_header AS row_header,
       cell.col_header AS col_header,
       cell.text AS text,
       cell.numeric_value AS numeric_value,
       cell.numeric_kind AS numeric_kind,
       cell.scale AS scale
LIMIT 1""",
]


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
    parser.add_argument(
        "--retriever-mode",
        choices=["hybrid", "text2cypher"],
        default="hybrid",
        help="Retriever mode to use. 'text2cypher' is recommended for financial table extraction.",
    )
    parser.add_argument(
        "--doc-id",
        help="Optional filing doc_id hint, e.g. 3M_2018_10K.pdf. Added to each question in text2cypher mode.",
    )
    parser.add_argument(
        "--return-context",
        action="store_true",
        help="Include retriever context in the output JSON for inspection.",
    )
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
    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = _require_env("NEO4J_PASSWORD")
    database = os.environ.get("NEO4J_DATABASE")
    vector_index = os.environ.get("NEO4J_VECTOR_INDEX", "chunk_embedding_index")
    fulltext_index = "chunk_fulltext_index"
    llm_model = os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:72b")
    embedding_model = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3:latest")
    ollama_host = os.environ.get("OLLAMA_HOST")

    ollama_kwargs = {}
    if ollama_host:
        ollama_kwargs["host"] = ollama_host

    llm = OllamaLLM(
        model_name=llm_model,
        model_params={"options": {"temperature": 0}},
        **ollama_kwargs,
    )

    embedder = OllamaEmbeddings(model=embedding_model, **ollama_kwargs)

    RETRIEVAL_QUERY = """
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

    if args.doc_id and args.retriever_mode == "text2cypher":
        questions = [f"[doc_id={args.doc_id}] {q}" for q in questions]

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
                    return_context=args.return_context,
                )
                answer = result.answer
                elapsed = time.time() - start
                print(f"  Answer: {answer[:100]}..." if len(answer) > 100 else f"  Answer: {answer}")
                print(f"  Time: {elapsed:.1f}s")

                payload = {
                    "question": question,
                    "answer": answer,
                    "time_seconds": round(elapsed, 2),
                    "error": None,
                }
                if args.return_context and result.retriever_result is not None:
                    payload["retriever_result"] = result.retriever_result.model_dump()
                results.append(payload)
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
