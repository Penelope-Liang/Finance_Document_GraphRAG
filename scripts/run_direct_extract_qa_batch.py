from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import neo4j
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.rag import GraphRAG
from neo4j_graphrag.retrievers import (
    HybridCypherRetriever,
    RerankingRetriever,
    Text2CypherRetriever,
)
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import RawSearchResult, RetrieverResult, RetrieverResultItem

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


def _format_chunk_record(record: neo4j.Record) -> RetrieverResultItem:
    text = str(record.get("text") or "").strip()
    section_name = str(record.get("section_name") or "").strip()
    table_cells = record.get("table_cells") or []
    lines = []
    if section_name:
        lines.append(f"Section: {section_name}")
    if text:
        lines.append(f"Chunk: {text}")
    if table_cells:
        lines.append("Related table cells:")
        for cell in table_cells[:10]:
            if not isinstance(cell, dict):
                continue
            row_header = cell.get("row_header")
            col_header = cell.get("col_header")
            cell_text = cell.get("text")
            numeric_value = cell.get("numeric_value")
            numeric_kind = cell.get("numeric_kind")
            scale = cell.get("scale")
            lines.append(
                f"- row={row_header!r}; col={col_header!r}; text={cell_text!r}; "
                f"numeric_value={numeric_value!r}; numeric_kind={numeric_kind!r}; scale={scale!r}"
            )
    if not lines:
        lines.append(str(record))
    return RetrieverResultItem(
        content="\n".join(lines),
        metadata={
            "section_name": section_name or None,
            "similarity_score": record.get("similarity_score"),
            "source": "chunk",
        },
    )


def _format_table_record(record: neo4j.Record) -> RetrieverResultItem:
    section_name = str(record.get("section_name") or "").strip()
    row_header = record.get("row_header")
    col_header = record.get("col_header")
    text = record.get("text")
    numeric_value = record.get("numeric_value")
    numeric_kind = record.get("numeric_kind")
    scale = record.get("scale")
    content = (
        f"Section: {section_name}\n"
        f"Table evidence: row={row_header!r}; col={col_header!r}; text={text!r}; "
        f"numeric_value={numeric_value!r}; numeric_kind={numeric_kind!r}; scale={scale!r}"
    )
    return RetrieverResultItem(
        content=content,
        metadata={
            "section_name": section_name or None,
            "row_header": row_header,
            "col_header": col_header,
            "source": "table",
        },
    )


class DualRetriever(Retriever):
    VERIFY_NEO4J_VERSION = False

    def __init__(
        self,
        driver: neo4j.Driver,
        chunk_retriever: HybridCypherRetriever,
        table_retriever: Text2CypherRetriever,
        neo4j_database: str | None = None,
    ) -> None:
        super().__init__(driver, neo4j_database)
        self.chunk_retriever = chunk_retriever
        self.table_retriever = table_retriever

    def get_search_results(self, *args: Any, **kwargs: Any) -> RawSearchResult:
        raise NotImplementedError("DualRetriever uses a custom search implementation.")

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        effective_search_ratio: int = 10,
        query_params: dict[str, Any] | None = None,
    ) -> RetrieverResult:
        chunk_result = self.chunk_retriever.search(
            query_text=query_text,
            top_k=top_k,
            effective_search_ratio=effective_search_ratio,
            query_params=query_params,
        )
        table_result = self.table_retriever.search(query_text=query_text)

        items: list[RetrieverResultItem] = []
        seen_contents: set[str] = set()

        for item in table_result.items + chunk_result.items:
            content = str(item.content)
            if content in seen_contents:
                continue
            seen_contents.add(content)
            items.append(item)

        return RetrieverResult(
            items=items,
            metadata={
                "__retriever": "DualRetriever",
                "chunk_retriever": chunk_result.metadata,
                "table_retriever": table_result.metadata,
            },
        )


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
        choices=["text2cypher", "hybrid", "dual"],
        default="text2cypher",
        help="Retriever mode to use for the batch run.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top K candidates for hybrid retrieval.",
    )
    parser.add_argument(
        "--effective-search-ratio",
        type=int,
        default=10,
        help="Hybrid candidate expansion factor before final top-k.",
    )
    parser.add_argument(
        "--restrict-doc-id",
        action="store_true",
        help="Diagnostic mode: restrict hybrid evidence to the target doc_id.",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Wrap the chosen retriever with the existing CrossEncoder reranker.",
    )
    parser.add_argument(
        "--reranker-model",
        default=os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        help="Sentence-transformers CrossEncoder model for reranking.",
    )
    parser.add_argument(
        "--reranker-top-k-retrieve",
        type=int,
        default=20,
        help="Number of candidates to fetch before reranking.",
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

    chunk_retrieval_query = """
    WITH node AS chunk, score
    WHERE ($doc_id IS NULL OR chunk.doc_id = $doc_id)
    OPTIONAL MATCH (chunk)-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
    OPTIONAL MATCH (chunk)<-[:FROM_CHUNK]-(entity)
    WITH chunk, score,
         collect(DISTINCT {
             row_header: cell.row_header,
             col_header: cell.col_header,
             text: cell.text,
             numeric_value: cell.numeric_value,
             numeric_kind: cell.numeric_kind,
             scale: cell.scale
         })[0..20] AS table_cells,
         collect(DISTINCT {
             label: coalesce(entity.type, head(labels(entity))),
             name: coalesce(entity.title, entity.name, elementId(entity))
         }) AS entities
    RETURN chunk.text AS text,
           chunk.section_name AS section_name,
           score AS similarity_score,
           table_cells,
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
                result_formatter=_format_table_record,
            )
        elif args.retriever_mode == "hybrid":
            retriever = HybridCypherRetriever(
                driver=driver,
                vector_index_name=vector_index,
                fulltext_index_name=fulltext_index,
                neo4j_database=database,
                retrieval_query=chunk_retrieval_query,
                embedder=embedder,
                result_formatter=_format_chunk_record,
            )
        else:
            chunk_retriever = HybridCypherRetriever(
                driver=driver,
                vector_index_name=vector_index,
                fulltext_index_name=fulltext_index,
                neo4j_database=database,
                retrieval_query=chunk_retrieval_query,
                embedder=embedder,
                result_formatter=_format_chunk_record,
            )
            table_retriever = Text2CypherRetriever(
                driver=driver,
                llm=llm,
                neo4j_database=database,
                neo4j_schema=FINANCIAL_GRAPH_SCHEMA,
                examples=TEXT2CYPHER_EXAMPLES,
                custom_prompt=TEXT2CYPHER_PROMPT,
                result_formatter=_format_table_record,
            )
            retriever = DualRetriever(
                driver=driver,
                chunk_retriever=chunk_retriever,
                table_retriever=table_retriever,
                neo4j_database=database,
            )
        if args.use_reranker:
            retriever = RerankingRetriever(
                inner_retriever=retriever,
                cross_encoder_model=args.reranker_model,
                top_k_retrieve=args.reranker_top_k_retrieve,
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
                if args.retriever_mode in {"hybrid", "dual"}:
                    retriever_config = {
                        "top_k": args.top_k,
                        "effective_search_ratio": args.effective_search_ratio,
                        "query_params": {
                            "doc_id": row["doc_id"] if args.restrict_doc_id else None
                        },
                    }
                    search_kwargs["retriever_config"] = retriever_config

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
