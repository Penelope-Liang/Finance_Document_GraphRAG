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
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None


_DOC_PREFIX_TO_COMPANY = {
    "3M": "3M",
    "ADOBE": "Adobe",
    "AMCOR": "Amcor",
    "AMD": "AMD",
    "AMERICANEXPRESS": "American Express",
    "BESTBUY": "Best Buy",
    "BOEING": "Boeing",
    "FOOTLOCKER": "Foot Locker",
    "JOHNSON_JOHNSON": "Johnson & Johnson",
    "JPMORGAN": "JPMorgan",
    "MGMRESORTS": "MGM Resorts",
    "PEPSICO": "PepsiCo",
    "PFIZER": "Pfizer",
    "ULTABEAUTY": "Ulta Beauty",
    "VERIZON": "Verizon",
}

_COMPANY_ALIASES = {
    "3m": "3M",
    "adobe": "Adobe",
    "amcor": "Amcor",
    "amd": "AMD",
    "american express": "American Express",
    "best buy": "Best Buy",
    "boeing": "Boeing",
    "foot locker": "Foot Locker",
    "johnson & johnson": "Johnson & Johnson",
    "johnson and johnson": "Johnson & Johnson",
    "jpmorgan": "JPMorgan",
    "jp morgan": "JPMorgan",
    "jpmorgan chase": "JPMorgan",
    "mgm resorts": "MGM Resorts",
    "pepsico": "PepsiCo",
    "pepsi": "PepsiCo",
    "pfizer": "Pfizer",
    "ulta beauty": "Ulta Beauty",
    "verizon": "Verizon",
}

_COMPANY_TO_DOC_KEY = {
    "3M": "3m",
    "Adobe": "adobe",
    "Amcor": "amcor",
    "AMD": "amd",
    "American Express": "americanexpress",
    "Best Buy": "bestbuy",
    "Boeing": "boeing",
    "Foot Locker": "footlocker",
    "Johnson & Johnson": "johnson_johnson",
    "JPMorgan": "jpmorgan",
    "MGM Resorts": "mgmresorts",
    "PepsiCo": "pepsico",
    "Pfizer": "pfizer",
    "Ulta Beauty": "ultabeauty",
    "Verizon": "verizon",
}

_STOPWORDS = {
    "what",
    "which",
    "when",
    "where",
    "why",
    "how",
    "the",
    "and",
    "for",
    "from",
    "with",
    "that",
    "this",
    "does",
    "did",
    "was",
    "were",
    "are",
    "is",
    "in",
    "of",
    "to",
    "a",
    "an",
    "as",
    "on",
    "by",
    "or",
    "it",
    "its",
    "their",
    "company",
    "fiscal",
    "year",
    "quarter",
    "report",
    "according",
    "based",
    "provided",
}


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing env var: {name}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sanitize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if np is not None and isinstance(value, np.floating):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if np is not None and isinstance(value, np.integer):
        return int(value)
    return value


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(v) for v in obj]
    return _sanitize_value(obj)


def _save_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _sanitize_for_json(rows)
    path.write_text(
        json.dumps(safe, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = _strip_think(text)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(cleaned[start : idx + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _doc_name(row: dict[str, Any]) -> str:
    doc_id = str(row.get("doc_id") or row.get("doc_name") or "")
    return doc_id.removesuffix(".pdf")


def _company_from_doc(row: dict[str, Any]) -> str | None:
    prefix = _doc_name(row).split("_20", 1)[0].split("_202", 1)[0]
    return _DOC_PREFIX_TO_COMPANY.get(prefix.upper())


def _company_from_question(question: str) -> str | None:
    q = " ".join((question or "").lower().replace("’", "'").split())
    for alias, company in sorted(_COMPANY_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", q):
            return company
    return None


def _extract_years(question: str) -> list[str]:
    return sorted(set(re.findall(r"\b(?:fy)?((?:19|20)\d{2})\b", question or "", re.I)))


def _important_terms(question: str, limit: int = 14) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z&'-]{2,}|\b\d{4}\b", question or "")
    terms: list[str] = []
    for word in words:
        key = word.lower().strip("'")
        if key in _STOPWORDS:
            continue
        if key not in {t.lower() for t in terms}:
            terms.append(word)
        if len(terms) >= limit:
            break
    return terms


def _build_retrieval_query(question: str, company: str | None, years: list[str]) -> str:
    parts = []
    if company:
        parts.append(company)
    if years:
        parts.append(" ".join(years))
    parts.extend(_important_terms(question))
    parts.append(question)
    return "; ".join(part for part in parts if part)


def _format_chunk_record(record: neo4j.Record) -> RetrieverResultItem:
    text = str(record.get("text") or "")
    table_cells = [
        cell
        for cell in (record.get("table_cells") or [])
        if cell and any(cell.get(key) is not None for key in ("row_header", "col_header", "text", "numeric_value"))
    ]
    entities = [
        entity
        for entity in (record.get("entities") or [])
        if entity and entity.get("name")
    ]
    lines = [
        f"Document: {record.get('doc_id')}",
        f"Section: {record.get('section_name')}",
        f"Chunk: {text}",
    ]
    if table_cells:
        lines.append("Related table cells:")
        for cell in table_cells[:12]:
            lines.append(
                "- "
                f"row={cell.get('row_header')!r}; "
                f"col={cell.get('col_header')!r}; "
                f"text={cell.get('text')!r}; "
                f"numeric_value={cell.get('numeric_value')!r}; "
                f"scale={cell.get('scale')!r}"
            )
    if entities:
        names = []
        for entity in entities[:12]:
            label = entity.get("label")
            name = entity.get("name")
            names.append(f"{label}:{name}" if label else str(name))
        lines.append("Graph entities: " + "; ".join(names))

    return RetrieverResultItem(
        content="\n".join(lines),
        metadata=_sanitize_for_json(
            {
                "doc_id": record.get("doc_id"),
                "section_name": record.get("section_name"),
                "similarity_score": record.get("similarity_score"),
                "table_cells": table_cells[:12],
                "entities": entities[:12],
            }
        ),
    )


def _evidence_text(result: RetrieverResult, max_chars_per_item: int) -> str:
    blocks = []
    for idx, item in enumerate(result.items or [], start=1):
        content = str(item.content or "")
        if len(content) > max_chars_per_item:
            content = content[:max_chars_per_item] + "\n[truncated]"
        blocks.append(f"[{idx}]\n{content}")
    return "\n\n".join(blocks)


def _answer_prompt(question: str, evidence: str) -> str:
    return f"""You are answering a FinanceBench novel-generated question.

Use only the provided evidence. The question may require direct extraction, comparison,
business explanation, or light numerical reasoning. Do not use outside knowledge.

Rules:
- Give a concise financial analyst answer.
- If numbers are needed, show the key calculation in the evidence field.
- If the evidence is insufficient, set answer to null and explain missing_information.
- Cite evidence by item id using used_evidence_ids.
- Return valid JSON only. No markdown.

JSON schema:
{{
  "answer": string or null,
  "evidence": string or null,
  "used_evidence_ids": [integer],
  "missing_information": string or null
}}

Question:
{question}

Evidence:
{evidence}
"""


def _judge_prompt(question: str, gold_answer: Any, candidate_answer: Any, evidence: str) -> str:
    return f"""You are an impartial evaluator for FinanceBench answers.

Grade whether the candidate answer correctly answers the question compared with the reference answer.
Allow minor wording differences and minor rounding differences. Penalize unsupported claims,
wrong financial period/company, wrong direction of change, or missing required explanation.

Score rubric:
5 = fully correct and adequately supported.
4 = essentially correct with small omissions.
3 = partially correct but misses an important detail.
2 = mostly wrong with a small relevant part.
1 = wrong or unsupported.

Return valid JSON only:
{{"score": integer, "pass": boolean, "rationale": "short reason"}}

Question:
{question}

Reference answer:
{gold_answer}

Candidate answer:
{candidate_answer}

Evidence available to candidate:
{evidence[:6000]}
"""


def _invoke_with_retries(llm: OllamaLLM, prompt: str, retries: int) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return llm.invoke(prompt).content
        except Exception as exc:  # pragma: no cover - external service
            last_error = exc
            print(f"  retryable LLM error ({attempt}/{retries}): {exc}")
            time.sleep(min(2 * attempt, 8))
    raise last_error or RuntimeError("LLM call failed")


def _run_judge(
    judge_llm: OllamaLLM,
    question: str,
    gold_answer: Any,
    answer: Any,
    evidence: str,
    retries: int,
) -> dict[str, Any]:
    raw = _invoke_with_retries(
        judge_llm,
        _judge_prompt(question, gold_answer, answer, evidence),
        retries=retries,
    )
    parsed = _extract_json_object(raw)
    if not parsed:
        return {"raw": raw, "score": None, "pass": None, "rationale": "judge_json_parse_failed"}
    score = parsed.get("score")
    try:
        score = int(score) if score is not None else None
    except (TypeError, ValueError):
        score = None
    passed = parsed.get("pass")
    if passed is None and score is not None:
        passed = score >= 4
    return {
        "raw": raw,
        "score": score,
        "pass": bool(passed) if passed is not None else None,
        "rationale": parsed.get("rationale"),
    }


def _build_retriever(
    driver: neo4j.Driver,
    database: str | None,
    embedder: OllamaEmbeddings,
    vector_index: str,
    fulltext_index: str,
) -> HybridCypherRetriever:
    retrieval_query = """
    WITH node AS chunk, score
    WHERE ($doc_id IS NULL OR chunk.doc_id = $doc_id)
      AND ($company_doc_key IS NULL OR toLower(chunk.doc_id) STARTS WITH $company_doc_key)
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
         })[0..20] AS entities
    RETURN chunk.text AS text,
           chunk.doc_id AS doc_id,
           chunk.section_name AS section_name,
           score AS similarity_score,
           table_cells,
           entities
    """
    return HybridCypherRetriever(
        driver=driver,
        vector_index_name=vector_index,
        fulltext_index_name=fulltext_index,
        retrieval_query=retrieval_query,
        embedder=embedder,
        result_formatter=_format_chunk_record,
        neo4j_database=database,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-path", type=Path, default=Path("scripts/financebench_novel_generated.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("output/financebench_novel_generated_results.json"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--effective-search-ratio", type=int, default=10)
    parser.add_argument("--max-evidence-chars-per-item", type=int, default=2800)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--judge-retries", type=int, default=2)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--restrict-doc-id", action="store_true")
    parser.add_argument("--no-company-filter", action="store_true")
    parser.add_argument("--vector-index", default=os.environ.get("NEO4J_VECTOR_INDEX", "chunk_embedding_index"))
    parser.add_argument("--fulltext-index", default=os.environ.get("NEO4J_FULLTEXT_INDEX", "chunk_fulltext_index"))
    parser.add_argument("--llm-model", default=os.environ.get("OLLAMA_LLM_MODEL", "qwen3.6:27b"))
    parser.add_argument("--judge-model", default=os.environ.get("OLLAMA_JUDGE_MODEL", "vicgalle/prometheus-7b-v2.0"))
    parser.add_argument("--embed-model", default=os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3:latest"))
    args = parser.parse_args()

    uri = _require_env("NEO4J_URI")
    user = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME")
    password = _require_env("NEO4J_PASSWORD")
    if not user:
        raise SystemExit("Missing env var: NEO4J_USER or NEO4J_USERNAME")
    database = os.environ.get("NEO4J_DATABASE")
    ollama_host = os.environ.get("OLLAMA_HOST")
    ollama_kwargs = {"host": ollama_host} if ollama_host else {}

    rows = _load_jsonl(args.questions_path)
    if args.max_questions is not None:
        rows = rows[: args.max_questions]

    embedder = OllamaEmbeddings(model=args.embed_model, **ollama_kwargs)
    llm = OllamaLLM(
        model_name=args.llm_model,
        model_params={"options": {"temperature": 0}},
        **ollama_kwargs,
    )
    judge_llm = OllamaLLM(
        model_name=args.judge_model,
        model_params={"options": {"temperature": 0}},
        **ollama_kwargs,
    )

    driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
    retriever = _build_retriever(
        driver=driver,
        database=database,
        embedder=embedder,
        vector_index=args.vector_index,
        fulltext_index=args.fulltext_index,
    )

    results: list[dict[str, Any]] = []
    try:
        for idx, row in enumerate(rows, start=1):
            started = time.time()
            question = row["question"]
            question_id = row.get("question_id") or row.get("financebench_id") or row.get("id")
            doc_id = row.get("doc_id")
            company = _company_from_question(question) or _company_from_doc(row)
            company_doc_key = None if args.no_company_filter else _COMPANY_TO_DOC_KEY.get(company or "")
            years = _extract_years(question)
            retrieval_query = _build_retrieval_query(question, company, years)
            print(f"[{idx}/{len(rows)}] {doc_id} :: {question_id}")
            print(f"  company={company} company_filter={company_doc_key} doc_id_filter={args.restrict_doc_id}")
            print(f"  retrieval_query={retrieval_query[:220]}")

            out_row = {
                **row,
                "question_id": question_id,
                "query_company": company,
                "query_company_doc_key": company_doc_key,
                "query_years": years,
                "retrieval_query": retrieval_query,
                "answer": None,
                "raw_answer": None,
                "evidence": None,
                "used_evidence_ids": None,
                "missing_information": None,
                "retriever_result": None,
                "judge": None,
                "error": None,
            }
            try:
                retriever_result = retriever.search(
                    query_text=retrieval_query,
                    top_k=args.top_k,
                    effective_search_ratio=args.effective_search_ratio,
                    query_params={
                        "doc_id": doc_id if args.restrict_doc_id else None,
                        "company_doc_key": company_doc_key,
                    },
                )
                retriever_result = _sanitize_for_json(retriever_result.model_dump())
                evidence = _evidence_text(
                    RetrieverResult(**retriever_result),
                    max_chars_per_item=args.max_evidence_chars_per_item,
                )
                raw_answer = _invoke_with_retries(
                    llm,
                    _answer_prompt(question, evidence),
                    retries=args.llm_retries,
                )
                parsed = _extract_json_object(raw_answer) or {}
                answer = parsed.get("answer") if parsed else _strip_think(raw_answer)
                used_ids = parsed.get("used_evidence_ids")
                answer_evidence = parsed.get("evidence")
                missing = parsed.get("missing_information")
                judge = None
                if not args.skip_judge:
                    gold = row.get("gold_answer") or row.get("answer")
                    judge = _run_judge(
                        judge_llm=judge_llm,
                        question=question,
                        gold_answer=gold,
                        answer=answer,
                        evidence=evidence,
                        retries=args.judge_retries,
                    )
                out_row.update(
                    {
                        "answer": answer,
                        "raw_answer": raw_answer,
                        "evidence": answer_evidence,
                        "used_evidence_ids": used_ids,
                        "missing_information": missing,
                        "retriever_result": retriever_result,
                        "judge": judge,
                    }
                )
                judge_text = ""
                if judge:
                    judge_text = f" judge_score={judge.get('score')} judge_pass={judge.get('pass')}"
                print(f"  answer={answer!r}{judge_text}")
            except Exception as exc:
                out_row["error"] = str(exc)
                print(f"  error={exc}")
            out_row["time_seconds"] = round(time.time() - started, 2)
            results.append(_sanitize_for_json(out_row))
            _save_results(args.output, results)
    finally:
        driver.close()

    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
