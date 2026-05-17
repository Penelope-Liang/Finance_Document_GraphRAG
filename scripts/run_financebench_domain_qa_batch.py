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
from neo4j_graphrag.retrievers import HybridCypherRetriever, Text2CypherRetriever
from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem

import run_direct_extract_qa_batch as direct
from neo4j_graphrag.tools.batch_ask import FINANCIAL_GRAPH_SCHEMA


DOMAIN_TEXT2CYPHER_EXAMPLES = direct.TEXT2CYPHER_EXAMPLES_CALCULATION + direct.TEXT2CYPHER_EXAMPLES_NOVEL


DOMAIN_TEXT2CYPHER_PROMPT = """
Task: Generate a Cypher statement for retrieving evidence for a domain-relevant FinanceBench question over a Neo4j graph.

Schema:
{schema}

Examples:
{examples}

Rules:
- Return only Cypher. No prose.
- Domain-relevant questions may require direct extraction, calculation, comparison, trend explanation, metric applicability judgment, or mixed numeric/business reasoning.
- Use constraints from the question only when present: company, fiscal year, quarter, filing type, section, business segment, or named metric.
- Prefer same-company evidence using doc_id prefixes when a company is identifiable.
- For ratio, margin, turnover, liquidity, CAPEX, ROA, growth, increase/decrease, or multi-period comparison questions, retrieve all required table-cell inputs and relevant narrative chunks when possible.
- For metric-applicability or business-explanation questions, retrieve concise chunks from MD&A, business overview, risk factors, notes, segment discussion, liquidity, or the named section.
- Do not retrieve only a single final number when the question requires a formula or comparison.
- Return evidence in this shape when possible:
  ch.section_name AS section_name,
  cell.row_header AS row_header,
  cell.col_header AS col_header,
  cell.text AS text,
  cell.numeric_value AS numeric_value,
  cell.numeric_kind AS numeric_kind,
  cell.scale AS scale
- If retrieving narrative chunks without table cells, return null for row_header/col_header/numeric fields and ch.text AS text.
- Use LIMIT 12-30 depending on whether the question needs multiple evidence rows. Avoid LIMIT 1.

Question:
{query_text}

Cypher query:
"""


def _load_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_name = row.get("doc_name") or str(row.get("doc_id") or "").removesuffix(".pdf")
            row["doc_name"] = doc_name
            row["doc_id"] = row.get("doc_id") or f"{doc_name}.pdf"
            row["question_id"] = row.get("question_id") or row.get("financebench_id")
            row["gold_answer"] = row.get("gold_answer") or row.get("answer")
            row["gold_justification"] = row.get("gold_justification") or row.get("justification")
            rows.append(row)
    return rows


def _parse_domain_answer(raw_answer: str) -> tuple[str | None, str | None, str | None]:
    cleaned = direct._strip_think(raw_answer)
    payload: Any
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        json_str = direct._extract_json_object(cleaned)
        if not json_str:
            return raw_answer, None, None
        try:
            payload = json.loads(json_str)
        except json.JSONDecodeError:
            return raw_answer, None, None
    if not isinstance(payload, dict):
        return raw_answer, None, None
    return (
        payload.get("answer"),
        payload.get("evidence"),
        payload.get("missing_information"),
    )


def _build_domain_answer_guidance(question: str) -> str:
    constraints = direct._extract_question_constraints(question)
    lines = [
        "Domain-relevant FinanceBench guidance:",
        "- Use retrieved evidence as the only source of truth; do not use outside knowledge.",
        "- Return JSON only with keys 'answer', 'evidence', and 'missing_information'.",
        "- The answer field should be concise: 1-3 sentences. Include the final conclusion and essential explanation if needed.",
        "- Put formulas, raw input values, calculations, and detailed justification in evidence.",
        "- For finance judgment questions, do not answer only based on the company type, industry, or large dollar amounts.",
        "- First find the relevant numbers in the evidence, calculate the needed ratios, and then make the judgment.",
        "- If the ratios point to a different conclusion than the company type or industry, follow the ratios.",
        "- If the evidence does not contain the required numbers, set answer=null and explain missing_information instead of making an industry-based judgment.",
        "- If the question asks for a ratio, margin, turnover, growth, liquidity, ROA, CAPEX intensity, or comparison, identify the required inputs and compute the result.",
        "- If the question asks whether a metric is useful or meaningful, first judge applicability from the evidence and company context, then answer or explain why not.",
        "- If the question asks what drove a change, identify the direction and the main drivers from evidence.",
        "- If the question asks direct information extraction, return the requested fact without unnecessary explanation.",
        "- Respect company, fiscal year, quarter, period, filing, section, and segment constraints; do not mix periods or companies.",
        "- If required inputs are missing or ambiguous, set answer=null and explain missing_information instead of guessing.",
        "- evidence may be null only when no separate formula/justification is needed.",
    ]
    if constraints["section_hints"]:
        lines.append(f"- Extracted section/table hints: {', '.join(constraints['section_hints'])}")
    if constraints["years"]:
        lines.append(f"- Extracted years: {', '.join(constraints['years'])}")
    if constraints["quarters"]:
        lines.append(f"- Extracted quarters: {', '.join('Q' + q for q in constraints['quarters'])}")
    if constraints["months"]:
        lines.append(f"- Extracted months/dates context: {', '.join(constraints['months'])}")
    if constraints["period_hints"]:
        lines.append(f"- Extracted period hints: {', '.join(constraints['period_hints'])}")
    return "\n".join(lines)


def _domain_rewrite_query(question: str, doc_name: str | None) -> dict[str, Any]:
    plan = direct._rewrite_metrics_query_for_retrieval(
        question,
        question_style="calculation",
        doc_name=doc_name,
    )
    clean = direct._normalize_text(question)
    components = list(plan.get("components") or [])
    statements = list(plan.get("statements") or [])

    def add_component(value: str) -> None:
        if value not in components:
            components.append(value)

    def add_statement(value: str) -> None:
        if value not in statements:
            statements.append(value)

    if any(token in clean for token in ("useful", "meaningful", "relevant", "not something", "not useful")):
        add_component("metric applicability")
        add_component("business model")
        add_component("industry context")
        add_statement("business")
        add_statement("management discussion")
    if any(token in clean for token in ("drove", "driver", "driving", "due to", "primarily due", "change")):
        add_component("drivers")
        add_component("margin change")
        add_component("operating margin")
        add_component("gross margin")
        add_statement("management discussion and analysis")
    if "quick ratio" in clean or "liquidity" in clean:
        add_component("cash and cash equivalents")
        add_component("accounts receivable")
        add_component("inventories")
        add_component("total current assets")
        add_component("total current liabilities")
        add_statement("consolidated balance sheet")
        add_statement("liquidity")
    if "capital-intensive" in clean or "capital intensive" in clean or "capex" in clean:
        add_component("capital expenditures")
        add_component("property plant and equipment")
        add_component("total assets")
        add_component("return on assets")
        add_statement("cash flows")
        add_statement("balance sheet")
        add_statement("statement of operations")
    if "stable" in clean or "trend" in clean:
        add_component("trend")
        add_component("multi-period comparison")
    if "acquisition" in clean or "acquired" in clean:
        add_component("acquisitions")
        add_statement("business combinations")
    if "industry" in clean or "primarily operate" in clean:
        add_component("industry")
        add_component("business overview")
        add_statement("business")

    plan["operation"] = "domain_relevant"
    plan["components"] = components
    plan["statements"] = statements
    parts = []
    if plan.get("company"):
        parts.append(str(plan["company"]))
    if plan.get("years"):
        parts.append(f"fiscal years {', '.join(plan['years'])}")
    parts.append("; ".join(components))
    if statements:
        parts.append("; ".join(statements))
    parts.append(question)
    plan["rewritten_query"] = "; ".join(part for part in parts if part).strip()
    return plan


class DomainDualRetriever(direct.DualRetriever):
    """Dual retriever variant that enables deterministic metric-cell retrieval for domain questions."""

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        effective_search_ratio: int = 10,
        query_params: dict[str, Any] | None = None,
        t2c_style: str = "domain",
        retrieval_query_text: str | None = None,
        metrics_doc_id: str | None = None,
        metrics_company_doc_key: str | None = None,
        metrics_plan: dict[str, Any] | None = None,
    ) -> RetrieverResult:
        search_text = retrieval_query_text or query_text
        metrics_result = (
            self.metrics_retriever.search(
                query_text=query_text,
                doc_id=metrics_doc_id,
                company_doc_key=metrics_company_doc_key,
                years=list((metrics_plan or {}).get("years") or []),
                plan=metrics_plan,
            )
            if self.metrics_retriever is not None and metrics_plan is not None
            else RetrieverResult(items=[], metadata={"__retriever": "MetricsTableCellRetriever", "matched": False})
        )
        metrics_result = direct._sanitize_retriever_result(metrics_result)
        chunk_result = direct._sanitize_retriever_result(self.chunk_retriever.search(
            query_text=search_text,
            top_k=top_k,
            effective_search_ratio=effective_search_ratio,
            query_params=query_params,
        ))
        table_result = direct._sanitize_retriever_result(self.table_retriever.search(query_text=search_text))
        primary_result = (
            direct._sanitize_retriever_result(self.primary_retriever.search(query_text=search_text, query_params=query_params))
            if self.primary_retriever is not None
            else RetrieverResult(items=[], metadata={"__retriever": "PrimaryStatementRetriever", "matched": False})
        )

        items: list[RetrieverResultItem] = []
        seen_contents: set[str] = set()
        for item in metrics_result.items + primary_result.items + table_result.items + chunk_result.items:
            content = str(item.content)
            if content in seen_contents:
                continue
            seen_contents.add(content)
            items.append(item)

        fusion_question = query_text if search_text == query_text else f"{query_text}\n{search_text}"
        items = direct._attach_fusion_scores(fusion_question, items)
        items.sort(
            key=lambda item: direct._safe_float((item.metadata or {}).get("fused_score")),
            reverse=True,
        )
        return direct._sanitize_retriever_result(RetrieverResult(
            items=items,
            metadata={
                "__retriever": "DomainDualRetriever",
                "metrics_retriever": metrics_result.metadata,
                "chunk_retriever": chunk_result.metadata,
                "table_retriever": table_result.metadata,
                "primary_retriever": primary_result.metadata,
                "retrieval_query_text": search_text,
            },
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run domain-relevant FinanceBench questions via a temporary domain runner.")
    parser.add_argument("--questions-path", default="scripts/financebench_domain_relevant.jsonl")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--output", default="output/financebench_domain_dual_qwen36.json")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--effective-search-ratio", type=int, default=20)
    parser.add_argument("--restrict-doc-id", action="store_true")
    parser.add_argument("--metrics-table-no-doc-id", action="store_true")
    parser.add_argument("--metrics-table-no-filter", action="store_true")
    parser.add_argument("--llm-retries", type=int, default=4)
    parser.add_argument("--llm-retry-sleep", type=float, default=1.5)
    args = parser.parse_args()

    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = direct._require_env("NEO4J_PASSWORD")
    database = os.environ.get("NEO4J_DATABASE")
    vector_index = os.environ.get("NEO4J_VECTOR_INDEX", "chunk_embedding_index")
    fulltext_index = os.environ.get("NEO4J_FULLTEXT_INDEX", "chunk_fulltext_index")
    llm_model = os.environ.get("OLLAMA_LLM_MODEL", "qwen3.6:27b")
    embedding_model = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3:latest")
    ollama_host = os.environ.get("OLLAMA_HOST")
    ollama_kwargs = {"host": ollama_host} if ollama_host else {}

    questions = _load_questions(Path(args.questions_path))
    if args.max_questions > 0:
        questions = questions[: args.max_questions]

    llm = OllamaLLM(
        model_name=llm_model,
        model_params={"options": {"temperature": 0}},
        **ollama_kwargs,
    )
    embedder = OllamaEmbeddings(model=embedding_model, **ollama_kwargs)

    chunk_retrieval_query = """
    WITH node AS chunk, score
    WHERE ($doc_id IS NULL OR chunk.doc_id = $doc_id)
      AND ($company_doc_key IS NULL OR toLower(chunk.doc_id) STARTS WITH $company_doc_key)
    OPTIONAL MATCH (chunk)-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
    WITH chunk, score,
         collect(DISTINCT {
             row_header: cell.row_header,
             col_header: cell.col_header,
             text: cell.text,
             numeric_value: cell.numeric_value,
             numeric_kind: cell.numeric_kind,
             scale: cell.scale
         })[0..20] AS table_cells
    RETURN chunk.text AS text,
           chunk.doc_id AS doc_id,
           chunk.section_name AS section_name,
           score AS similarity_score,
           table_cells,
           [] AS entities
    """

    output_path = Path(args.output)
    results: list[dict[str, Any]] = []
    with neo4j.GraphDatabase.driver(uri, auth=(user, password)) as driver:
        table_retriever = Text2CypherRetriever(
            driver=driver,
            llm=llm,
            neo4j_database=database,
            neo4j_schema=FINANCIAL_GRAPH_SCHEMA,
            examples=DOMAIN_TEXT2CYPHER_EXAMPLES,
            custom_prompt=DOMAIN_TEXT2CYPHER_PROMPT,
            result_formatter=direct._format_table_record,
        )
        chunk_retriever = HybridCypherRetriever(
            driver=driver,
            vector_index_name=vector_index,
            fulltext_index_name=fulltext_index,
            neo4j_database=database,
            retrieval_query=chunk_retrieval_query,
            embedder=embedder,
            result_formatter=direct._format_chunk_record,
        )
        retriever = DomainDualRetriever(
            driver=driver,
            chunk_retriever=chunk_retriever,
            table_retriever=table_retriever,
            primary_retriever=direct.PrimaryStatementRetriever(driver=driver, neo4j_database=database),
            metrics_retriever=direct.MetricsTableCellRetriever(driver=driver, neo4j_database=database),
            neo4j_database=database,
        )
        rag = GraphRAG(retriever=retriever, llm=llm)

        for idx, row in enumerate(questions, start=1):
            question = row["question"]
            plan = _domain_rewrite_query(question, str(row.get("doc_name") or ""))
            print(f"[{idx}/{len(questions)}] {row['doc_id']} :: {row['question_id']}", flush=True)
            print(f"  retrieval_query={plan['rewritten_query']}", flush=True)
            start = time.time()
            try:
                if args.metrics_table_no_filter:
                    metrics_doc_id = None
                    metrics_company_doc_key = None
                elif args.metrics_table_no_doc_id:
                    metrics_doc_id = None
                    metrics_company_doc_key = plan.get("company_doc_key")
                else:
                    metrics_doc_id = row["doc_id"]
                    metrics_company_doc_key = None

                result = direct._rag_search_with_retry(
                    rag,
                    max_retries=args.llm_retries,
                    retry_sleep=args.llm_retry_sleep,
                    query_text=question,
                    return_context=True,
                    examples=_build_domain_answer_guidance(question),
                    retriever_config={
                        "top_k": args.top_k,
                        "effective_search_ratio": args.effective_search_ratio,
                        "query_params": {
                            "doc_id": row["doc_id"] if args.restrict_doc_id else None,
                            "company_doc_key": (
                                None if args.restrict_doc_id else plan.get("company_doc_key")
                            ),
                        },
                        "retrieval_query_text": plan["rewritten_query"],
                        "metrics_doc_id": metrics_doc_id,
                        "metrics_company_doc_key": metrics_company_doc_key,
                        "metrics_plan": plan,
                        "t2c_style": "domain",
                    },
                )
                raw_answer = result.answer
                answer, evidence, missing_information = _parse_domain_answer(raw_answer)
                retriever_result_dump = (
                    direct._sanitize_for_json(result.retriever_result.model_dump())
                    if result.retriever_result is not None
                    else None
                )
                elapsed = round(time.time() - start, 2)
                results.append({
                    "question_id": row["question_id"],
                    "doc_id": row["doc_id"],
                    "doc_name": row["doc_name"],
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "gold_justification": row.get("gold_justification"),
                    "answer": answer,
                    "evidence": evidence,
                    "missing_information": missing_information,
                    "raw_answer": raw_answer,
                    "retriever_result": retriever_result_dump,
                    "time_seconds": elapsed,
                    "error": None,
                    "question_style": "domain",
                    "retrieval_query": plan["rewritten_query"],
                    "query_rewrite_company": plan.get("company"),
                    "query_rewrite_company_doc_key": plan.get("company_doc_key"),
                    "query_rewrite_years": plan.get("years"),
                    "query_rewrite_components": plan.get("components"),
                    "query_rewrite_statements": plan.get("statements"),
                })
                direct._save_results(output_path, results)
                print(f"  answer={answer!r} gold={row['gold_answer']}", flush=True)
            except Exception as exc:
                elapsed = round(time.time() - start, 2)
                results.append({
                    "question_id": row["question_id"],
                    "doc_id": row["doc_id"],
                    "doc_name": row["doc_name"],
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "gold_justification": row.get("gold_justification"),
                    "answer": None,
                    "evidence": None,
                    "missing_information": None,
                    "raw_answer": None,
                    "retriever_result": None,
                    "time_seconds": elapsed,
                    "error": str(exc),
                    "question_style": "domain",
                    "retrieval_query": plan["rewritten_query"],
                    "query_rewrite_company": plan.get("company"),
                    "query_rewrite_company_doc_key": plan.get("company_doc_key"),
                    "query_rewrite_years": plan.get("years"),
                    "query_rewrite_components": plan.get("components"),
                    "query_rewrite_statements": plan.get("statements"),
                })
                direct._save_results(output_path, results)
                print(f"  error={exc}", flush=True)

    direct._save_results(output_path, results)
    print(f"\nSaved results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
