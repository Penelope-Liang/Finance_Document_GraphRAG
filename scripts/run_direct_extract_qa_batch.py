from __future__ import annotations

import argparse
import json
import math
import os
import re
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


def _extract_json_object(text: str) -> str | None:
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    return match.group(0) if match else None


_MONTH_NAMES = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _extract_question_constraints(question: str) -> dict[str, Any]:
    clean = _normalize_text(question)

    years = sorted(set(re.findall(r"\b(?:fy)?((?:19|20)\d{2})\b", clean)))
    quarter_match = re.findall(r"\bq([1-4])\b", clean)
    month_hits = [month for month in _MONTH_NAMES if month in clean]
    period_hints = [
        hint
        for hint in (
            "year end",
            "year ended",
            "three months ended",
            "six months ended",
            "nine months ended",
            "quarter ended",
            "quarters ended",
            "as of",
        )
        if hint in clean
    ]

    explicit_section_hints = [
        phrase
        for phrase in (
            "income statement",
            "statement of income",
            "statements of income",
            "statement of operations",
            "statements of operations",
            "statement of earnings",
            "statements of earnings",
            "balance sheet",
            "balance sheets",
            "statement of financial position",
            "cash flow statement",
            "cash flow statements",
            "statement of cash flows",
            "statements of cash flows",
            "statement of stockholders' equity",
            "statement of shareholders' equity",
            "segment information",
            "quarterly results",
            "geographic information",
        )
        if phrase in clean
    ]

    generic_table_hints = []
    for pattern in (
        r"(note\s+\d+[a-z]?)",
        r"(table\s+\d+[a-z]?)",
        r"(item\s+\d+[a-z]?)",
    ):
        generic_table_hints.extend(re.findall(pattern, clean))

    return {
        "years": years,
        "quarters": quarter_match,
        "months": month_hits,
        "period_hints": period_hints,
        "section_hints": explicit_section_hints + generic_table_hints,
    }


def _build_answer_guidance(question: str) -> str:
    constraints = _extract_question_constraints(question)
    lines = [
        "Answering guidance:",
        "- Prefer evidence whose section/table wording matches the question's report or section hint.",
        "- If the question mentions a specific report, section, note, table name, or similar label, prioritize exact matches first and then closely related phrasing.",
        "- If the exact wording does not appear, pay attention to semantically similar section/table names rather than ignoring the constraint.",
        "- Respect time constraints strictly: fiscal year, quarter, period-ended phrasing, and explicit dates must match the chosen evidence.",
        "- If candidate evidence conflicts on statement type or time period, prefer the exact match and avoid mixing periods.",
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


def _candidate_bonus(question: str, item: RetrieverResultItem) -> float:
    constraints = _extract_question_constraints(question)
    content = _normalize_text(str(item.content))
    metadata = item.metadata or {}
    section_name = _normalize_text(str(metadata.get("section_name") or ""))
    haystack = " ".join(part for part in (content, section_name) if part)

    score = 0.0
    for hint in constraints["section_hints"]:
        if hint in haystack:
            score += 2.0
        elif any(token in haystack for token in hint.split()):
            score += 0.6

    for year in constraints["years"]:
        if year in haystack:
            score += 1.5

    for quarter in constraints["quarters"]:
        q = f"q{quarter}"
        if q in haystack:
            score += 1.5
        if quarter == "1" and "three months ended" in haystack:
            score += 0.6
        if quarter == "2" and "six months ended" in haystack:
            score += 0.6

    for month in constraints["months"]:
        if month in haystack:
            score += 0.8

    for period_hint in constraints["period_hints"]:
        if period_hint in haystack:
            score += 1.0

    if metadata.get("source") == "table" and (
        constraints["section_hints"] or constraints["years"] or constraints["period_hints"]
    ):
        score += 0.4

    return score


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _table_base_score(item: RetrieverResultItem) -> float:
    metadata = item.metadata or {}
    score = 0.0
    if metadata.get("row_header"):
        score += 1.0
    if metadata.get("col_header"):
        score += 0.8
    if metadata.get("section_name"):
        score += 0.5

    content = _normalize_text(str(item.content or ""))
    if "numeric_value=" in content:
        score += 1.2
    if "table evidence:" in content:
        score += 0.5
    return score


def _item_base_score(item: RetrieverResultItem) -> float:
    metadata = item.metadata or {}
    source = metadata.get("source")
    if source == "chunk":
        return _safe_float(metadata.get("similarity_score"))
    if source == "table":
        return _table_base_score(item)
    return 0.0


def _softmax_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps)
    if total <= 0:
        return [0.0 for _ in scores]
    return [value / total for value in exps]


def _attach_fusion_scores(
    question: str,
    items: list[RetrieverResultItem],
    chunk_weight: float = 0.5,
    table_weight: float = 0.5,
    bonus_weight: float = 0.15,
) -> list[RetrieverResultItem]:
    by_source: dict[str, list[int]] = {}
    base_scores: list[float] = []
    bonuses: list[float] = []

    for idx, item in enumerate(items):
        source = str((item.metadata or {}).get("source") or "unknown")
        by_source.setdefault(source, []).append(idx)
        base_scores.append(_item_base_score(item))
        bonuses.append(_candidate_bonus(question, item))

    normalized_by_idx = [0.0] * len(items)
    for source, indices in by_source.items():
        source_scores = [base_scores[i] for i in indices]
        normalized = _softmax_normalize(source_scores)
        for idx, norm_score in zip(indices, normalized):
            normalized_by_idx[idx] = norm_score

    fused_items: list[RetrieverResultItem] = []
    for idx, item in enumerate(items):
        metadata = dict(item.metadata or {})
        source = str(metadata.get("source") or "unknown")
        source_weight = chunk_weight if source == "chunk" else table_weight if source == "table" else 1.0
        fused_score = source_weight * normalized_by_idx[idx] + bonus_weight * bonuses[idx]
        metadata.update(
            {
                "raw_score": base_scores[idx],
                "normalized_score": normalized_by_idx[idx],
                "constraint_bonus": bonuses[idx],
                "source_weight": source_weight,
                "fused_score": fused_score,
            }
        )
        fused_items.append(
            RetrieverResultItem(
                content=item.content,
                metadata=metadata,
            )
        )
    return fused_items


def _format_judge_candidate(index: int, item: RetrieverResultItem, max_chars: int = 1200) -> str:
    metadata = item.metadata or {}
    section_name = str(metadata.get("section_name") or "")
    source = str(metadata.get("source") or "")
    row_header = str(metadata.get("row_header") or "")
    col_header = str(metadata.get("col_header") or "")
    content = str(item.content or "")[:max_chars]
    return (
        f"[ID: {index}]\n"
        f"Source: {source}\n"
        f"Section: {section_name}\n"
        f"Row Header: {row_header}\n"
        f"Col Header: {col_header}\n"
        f"Content:\n{content}"
    )


def _judge_prompt(query_text: str, items: list[RetrieverResultItem], max_chars: int) -> str:
    constraints = _extract_question_constraints(query_text)
    candidates_text = "\n\n".join(
        _format_judge_candidate(i, item, max_chars=max_chars)
        for i, item in enumerate(items)
    )
    hints = []
    if constraints["section_hints"]:
        hints.append(f"Section/table hints: {', '.join(constraints['section_hints'])}")
    if constraints["years"]:
        hints.append(f"Years: {', '.join(constraints['years'])}")
    if constraints["quarters"]:
        hints.append(f"Quarters: {', '.join('Q' + q for q in constraints['quarters'])}")
    if constraints["months"]:
        hints.append(f"Months: {', '.join(constraints['months'])}")
    if constraints["period_hints"]:
        hints.append(f"Period hints: {', '.join(constraints['period_hints'])}")
    hint_block = "\n".join(f"- {line}" for line in hints) if hints else "- No explicit structured hints extracted."
    return f"""
You are filtering retrieved evidence for a financial question.

Task:
- Keep all candidates that are not clearly irrelevant.
- Prefer false positives over false negatives.
- Prioritize candidates matching the question's statement/section/table hints and time constraints.
- A candidate does not need to directly answer the question; keep it if it may help answer.

Question:
{query_text}

Extracted constraints:
{hint_block}

Candidate evidence:
{candidates_text}

Return only a JSON object with this exact schema:
{{"ids": [0, 2, 5]}}

Rules:
- IDs must be chosen only from the listed candidates.
- Return at least 5 IDs if 5 or more candidates are available, unless almost everything is clearly irrelevant.
- Return an empty list only if no candidate is plausibly useful.
""".strip()


class LLMJudgeRetriever(Retriever):
    VERIFY_NEO4J_VERSION = False

    def __init__(
        self,
        inner_retriever: Retriever,
        llm: OllamaLLM,
        judge_max_chars: int = 1200,
        neo4j_database: str | None = None,
    ) -> None:
        super().__init__(
            driver=inner_retriever.driver,
            neo4j_database=neo4j_database or getattr(inner_retriever, "neo4j_database", None),
        )
        self.inner_retriever = inner_retriever
        self.llm = llm
        self.judge_max_chars = judge_max_chars

    def get_search_results(self, *args: Any, **kwargs: Any) -> RawSearchResult:
        raise NotImplementedError("LLMJudgeRetriever wraps search(), not raw Neo4j records.")

    def search(self, *args: Any, **kwargs: Any) -> RetrieverResult:
        result = self.inner_retriever.search(*args, **kwargs)
        if not result.items:
            return result

        query_text = kwargs.get("query_text") or (args[0] if args else "")
        if not query_text:
            return RetrieverResult(
                items=result.items,
                metadata={**(result.metadata or {}), "llm_judge_applied": False},
            )

        prompt = _judge_prompt(query_text, result.items, self.judge_max_chars)
        try:
            response = self.llm.invoke(prompt)
            payload_text = _extract_json_object(response.content) or response.content
            payload = json.loads(payload_text)
            ids = payload.get("ids", []) if isinstance(payload, dict) else []
            if not isinstance(ids, list):
                ids = []
            judged_items = [
                result.items[i]
                for i in ids
                if isinstance(i, int) and 0 <= i < len(result.items)
            ]
            if not judged_items:
                judged_items = result.items
            return RetrieverResult(
                items=judged_items,
                metadata={
                    **(result.metadata or {}),
                    "llm_judge_applied": True,
                    "llm_judge_kept": len(judged_items),
                },
            )
        except Exception:
            return RetrieverResult(
                items=result.items,
                metadata={**(result.metadata or {}), "llm_judge_applied": False},
            )


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

        items = _attach_fusion_scores(query_text, items)
        items.sort(
            key=lambda item: _safe_float((item.metadata or {}).get("fused_score")),
            reverse=True,
        )

        return RetrieverResult(
            items=items,
            metadata={
                "__retriever": "DualRetriever",
                "chunk_retriever": chunk_result.metadata,
                "table_retriever": table_result.metadata,
                "fusion": {
                    "strategy": "source_softmax_plus_constraint_bonus",
                    "chunk_weight": 0.5,
                    "table_weight": 0.5,
                    "bonus_weight": 0.15,
                },
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
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Apply an LLM-based evidence filter after retrieval and before answer generation.",
    )
    parser.add_argument(
        "--llm-judge-max-chars",
        type=int,
        default=1200,
        help="Max chars per candidate item shown to the LLM judge.",
    )
    parser.add_argument(
        "--llm-judge-model",
        default=os.environ.get("OLLAMA_JUDGE_MODEL", "qwen3:14b"),
        help="Ollama model used only for evidence judging.",
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
    judge_llm = OllamaLLM(
        model_name=args.llm_judge_model,
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
        if args.use_llm_judge:
            retriever = LLMJudgeRetriever(
                inner_retriever=retriever,
                llm=judge_llm,
                judge_max_chars=args.llm_judge_max_chars,
                neo4j_database=database,
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
                    "examples": _build_answer_guidance(row["question"]),
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
