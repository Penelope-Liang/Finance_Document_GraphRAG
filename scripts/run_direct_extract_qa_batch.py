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


TEXT2CYPHER_EXAMPLES_CALCULATION = [
    """User: What is the year over year change in revenue for Amazon from FY2016 to FY2017? Answer as a percentage.
Cypher:
MATCH (ch:Chunk {doc_id: "AMAZON_2017_10K.pdf"})-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WHERE cell.row_header IN ["Net sales", "Total net sales", "Net revenues", "Total revenues", "Revenue"]
  AND (cell.col_header CONTAINS "2017" OR cell.col_header CONTAINS "2016")
  AND toLower(ch.section_name) CONTAINS "statements of operations"
RETURN ch.section_name AS section_name,
       cell.row_header AS row_header,
       cell.col_header AS col_header,
       cell.text AS text,
       cell.numeric_value AS numeric_value,
       cell.numeric_kind AS numeric_kind,
       cell.scale AS scale
ORDER BY cell.col_header DESC""",
    """User: What was Block's revenue growth rate from FY2019 to FY2020? Answer as a percentage.
Cypher:
MATCH (ch:Chunk {doc_id: "BLOCK_2020_10K.pdf"})-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WHERE cell.row_header IN ["Total net revenue", "Total net revenues", "Net revenue", "Total revenue", "Total revenues", "Revenue", "Revenues"]
  AND (cell.col_header CONTAINS "2020" OR cell.col_header CONTAINS "2019")
  AND toLower(ch.section_name) CONTAINS "statement of operations"
RETURN ch.section_name AS section_name,
       cell.row_header AS row_header,
       cell.col_header AS col_header,
       cell.text AS text,
       cell.numeric_value AS numeric_value,
       cell.numeric_kind AS numeric_kind,
       cell.scale AS scale
ORDER BY cell.col_header DESC""",
    """User: What is the three-year average net profit margin for Best Buy over FY2015 to FY2017?
Cypher:
MATCH (ch:Chunk {doc_id: "BESTBUY_2017_10K.pdf"})-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WHERE cell.row_header IN ["Revenue", "Net earnings", "Net income"]
  AND (cell.col_header CONTAINS "2017" OR cell.col_header CONTAINS "2016" OR cell.col_header CONTAINS "2015")
  AND toLower(ch.section_name) CONTAINS "statement of earnings"
RETURN ch.section_name AS section_name,
       cell.row_header AS row_header,
       cell.col_header AS col_header,
       cell.text AS text,
       cell.numeric_value AS numeric_value,
       cell.numeric_kind AS numeric_kind,
       cell.scale AS scale
ORDER BY cell.row_header, cell.col_header DESC""",
    """User: What is PepsiCo's FY2022 unadjusted EBITDA less capital expenditure in USD millions?
Cypher:
MATCH (ch:Chunk {doc_id: "PEPSICO_2022_10K.pdf"})-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WHERE cell.row_header IN ["Net cash provided by operating activities", "Capital spending", "Capital expenditures"]
  AND cell.col_header CONTAINS "2022"
RETURN ch.section_name AS section_name,
       cell.row_header AS row_header,
       cell.col_header AS col_header,
       cell.text AS text,
       cell.numeric_value AS numeric_value,
       cell.numeric_kind AS numeric_kind,
       cell.scale AS scale
ORDER BY cell.row_header""",
]


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


def _save_results(output_path: Path, results: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def _strip_think(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _parse_answer(raw_answer: str) -> tuple[str | None, str | None]:
    cleaned = _strip_think(raw_answer)
    # First try parsing the cleaned text directly
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to extracting the first {...} block
        json_str = _extract_json_object(cleaned)
        if json_str:
            try:
                payload = json.loads(json_str)
            except json.JSONDecodeError:
                return raw_answer, None
        else:
            return raw_answer, None
    if isinstance(payload, dict):
        return payload.get("answer"), payload.get("evidence")
    return raw_answer, None


def _extract_json_object(text: str) -> str | None:
    # Strip think blocks first so the greedy .* doesn't span across them
    cleaned = _strip_think(text)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
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


def _build_direct_extract_answer_guidance(question: str) -> str:
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


def _build_calculation_answer_guidance(question: str) -> str:
    constraints = _extract_question_constraints(question)
    lines = [
        "Calculation guidance:",
        "- This is a metrics calculation question, not a direct single-value extraction question.",
        "- First identify every required input value before answering.",
        "- Use only evidence that matches the question's requested statement, section/table hint, and time period.",
        "- For ratios, margins, averages, YoY changes, turnover, DPO, CCC, or CAGR, gather all required numerator/denominator or multi-period inputs before computing.",
        "- Do not answer from a single value if the metric requires multiple inputs.",
        "- If multiple candidate values are available, prefer the exact statement and period match.",
        "- If any required input is missing or period alignment is ambiguous, return insufficient information instead of guessing.",
        "- Keep the computation simple and explicit: identify inputs, apply the formula, then provide the final answer.",
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


def _build_answer_guidance(question: str, question_style: str) -> str:
    if question_style == "calculation":
        return _build_calculation_answer_guidance(question)
    return _build_direct_extract_answer_guidance(question)


def _select_text2cypher_examples(question_style: str) -> list[str]:
    if question_style == "calculation":
        return TEXT2CYPHER_EXAMPLES_CALCULATION
    return TEXT2CYPHER_EXAMPLES


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
    if source == "primary_statement":
        # Primary statements are deterministically matched, all are equally relevant
        # to the question's statement-type hint, so give them a uniform high base.
        return 1.0
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
        if source == "chunk":
            source_weight = chunk_weight
        elif source == "table":
            source_weight = table_weight
        elif source == "primary_statement":
            # Differentiated weight by match_priority:
            #   3 = both statement_type and section_name match (strongest signal)
            #   2 = only statement_type matches (PepsiCo-style: section_name lost during ingest)
            #   1 = only section_name matches (footnote / supplementary table)
            priority = int((item.metadata or {}).get("match_priority") or 2)
            source_weight = {3: 1.6, 2: 1.3, 1: 1.0}.get(priority, 1.3)
        else:
            source_weight = 1.0
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


_DOC_ID_RE = re.compile(r"\[doc_id=([^\]]+)\]")

# Statement-type detection: hint terms in the question → section_name patterns
# in Neo4j.  Each entry has:
#   - hints: substrings to look for in the question
#   - includes: section_name substring patterns (CONTAINS, case-insensitive)
#   - excludes: section_name substring patterns to filter OUT false positives
_PRIMARY_STATEMENT_TYPES = (
    {
        "name": "balance_sheet",
        "hints": (
            "balance sheet",
            "balance sheets",
            "statement of financial position",
            "statements of financial position",
        ),
        "includes": (
            "balance sheet",
            "balance sheets",
            "statement of financial position",
            "statements of financial position",
            "financial position",
        ),
        "excludes": (
            "off-balance sheet",
            "off balance sheet",
            "balance sheet classification",
            "balance sheet arrangements",
            "balance sheet hedge",
        ),
    },
    {
        "name": "income_statement",
        "hints": (
            "income statement",
            "statement of income",
            "statements of income",
            "statement of operations",
            "statements of operations",
            "statement of earnings",
            "statements of earnings",
            "statement of comprehensive income",
            "statements of comprehensive income",
        ),
        "includes": (
            "income statement",
            "income statements",
            "statement of income",
            "statements of income",
            "statement of operations",
            "statements of operations",
            "statement of earnings",
            "statements of earnings",
            "comprehensive income",
        ),
        "excludes": (
            "other income",
            "income tax",
            "interest income",
            "comprehensive income reclassif",
        ),
    },
    {
        "name": "cash_flow",
        "hints": (
            "cash flow statement",
            "cash flow statements",
            "statement of cash flows",
            "statements of cash flows",
            "cash flows statement",
        ),
        "includes": (
            "cash flow statement",
            "cash flow statements",
            "statement of cash flows",
            "statements of cash flows",
            "cash flows from",
        ),
        "excludes": (
            "cash flow hedge",
            "cash flow hedges",
            "free cash flow",
        ),
    },
    {
        "name": "equity_statement",
        "hints": (
            "stockholders' equity",
            "stockholders equity",
            "shareholders' equity",
            "shareholders equity",
            "statement of equity",
            "statements of equity",
            "changes in equity",
        ),
        "includes": (
            "stockholders' equity",
            "stockholders equity",
            "shareholders' equity",
            "shareholders equity",
            "statement of equity",
            "statements of equity",
            "changes in equity",
        ),
        "excludes": (
            "equity method",
            "equity investments",
        ),
    },
)


def _detect_statement_type(query_text: str) -> dict[str, Any] | None:
    clean = _normalize_text(query_text)
    for spec in _PRIMARY_STATEMENT_TYPES:
        if any(hint in clean for hint in spec["hints"]):
            return spec
    return None


def _extract_doc_id(query_text: str) -> str | None:
    match = _DOC_ID_RE.search(query_text or "")
    return match.group(1).strip() if match else None


_PRIMARY_STATEMENT_QUERY = """
MATCH (chunk:Chunk)
WHERE ($doc_id IS NULL OR chunk.doc_id = $doc_id)
  AND chunk.should_index = true
  AND (
    chunk.statement_type = $statement_type
    OR (
      chunk.section_name IS NOT NULL
      AND ANY(pat IN $includes WHERE toLower(chunk.section_name) CONTAINS pat)
      AND NONE(pat IN $excludes WHERE toLower(chunk.section_name) CONTAINS pat)
    )
  )
WITH chunk,
     (CASE WHEN chunk.statement_type = $statement_type THEN 2 ELSE 0 END
      + CASE WHEN chunk.section_name IS NOT NULL
                  AND ANY(pat IN $includes WHERE toLower(chunk.section_name) CONTAINS pat)
                  AND NONE(pat IN $excludes WHERE toLower(chunk.section_name) CONTAINS pat)
             THEN 1 ELSE 0 END) AS match_priority
OPTIONAL MATCH (chunk)-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WITH chunk, match_priority,
     collect(DISTINCT {
         row_header: cell.row_header,
         col_header: cell.col_header,
         text: cell.text,
         numeric_value: cell.numeric_value,
         numeric_kind: cell.numeric_kind,
         scale: cell.scale
     })[0..40] AS table_cells
RETURN chunk.text AS text,
       chunk.section_name AS section_name,
       chunk.statement_type AS chunk_statement_type,
       1.0 AS similarity_score,
       table_cells,
       [] AS entities,
       match_priority
ORDER BY match_priority DESC
LIMIT $limit
"""


class PrimaryStatementRetriever(Retriever):
    """Deterministically pulls primary financial statement chunks
    (balance sheet / income statement / cash flow / equity) by section_name
    pattern matching, bypassing vector similarity which often ranks MD&A
    discussions above the actual statements."""

    VERIFY_NEO4J_VERSION = False

    def __init__(
        self,
        driver: neo4j.Driver,
        neo4j_database: str | None = None,
        limit: int = 8,
    ) -> None:
        super().__init__(driver, neo4j_database)
        self.limit = limit

    def get_search_results(self, *args: Any, **kwargs: Any) -> RawSearchResult:
        raise NotImplementedError("PrimaryStatementRetriever uses a custom search implementation.")

    def search(
        self,
        query_text: str,
        query_params: dict[str, Any] | None = None,
    ) -> RetrieverResult:
        spec = _detect_statement_type(query_text)
        doc_id = (query_params or {}).get("doc_id") if query_params else None
        if doc_id is None:
            doc_id = _extract_doc_id(query_text)

        if spec is None:
            return RetrieverResult(
                items=[],
                metadata={
                    "__retriever": "PrimaryStatementRetriever",
                    "matched": False,
                    "reason": "no statement type detected",
                },
            )

        params = {
            "doc_id": doc_id,
            "statement_type": spec["name"],
            "includes": list(spec["includes"]),
            "excludes": list(spec["excludes"]),
            "limit": self.limit,
        }
        with self.driver.session(database=self.neo4j_database) as session:
            records = list(session.run(_PRIMARY_STATEMENT_QUERY, params))

        items = []
        for record in records:
            item = _format_chunk_record(record)
            metadata = dict(item.metadata or {})
            metadata["source"] = "primary_statement"
            metadata["statement_type"] = spec["name"]
            metadata["chunk_statement_type"] = record.get("chunk_statement_type")
            metadata["match_priority"] = int(record.get("match_priority") or 0)
            items.append(
                RetrieverResultItem(content=item.content, metadata=metadata)
            )

        return RetrieverResult(
            items=items,
            metadata={
                "__retriever": "PrimaryStatementRetriever",
                "matched": True,
                "statement_type": spec["name"],
                "doc_id": doc_id,
                "kept": len(items),
            },
        )


class DualRetriever(Retriever):
    VERIFY_NEO4J_VERSION = False

    def __init__(
        self,
        driver: neo4j.Driver,
        chunk_retriever: HybridCypherRetriever,
        table_retriever: Text2CypherRetriever,
        primary_retriever: PrimaryStatementRetriever | None = None,
        neo4j_database: str | None = None,
    ) -> None:
        super().__init__(driver, neo4j_database)
        self.chunk_retriever = chunk_retriever
        self.table_retriever = table_retriever
        self.primary_retriever = primary_retriever

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
        primary_result = (
            self.primary_retriever.search(query_text=query_text, query_params=query_params)
            if self.primary_retriever is not None
            else RetrieverResult(items=[], metadata={"__retriever": "PrimaryStatementRetriever", "matched": False})
        )

        items: list[RetrieverResultItem] = []
        seen_contents: set[str] = set()

        # Order matters: primary statements first so they win ties on dedup.
        for item in primary_result.items + table_result.items + chunk_result.items:
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
                "primary_retriever": primary_result.metadata,
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
        "--question-style",
        choices=["direct", "calculation"],
        default="direct",
        help="Prompt style for direct extraction vs metrics calculation questions.",
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

    output_path = Path(args.output)
    results = []
    with neo4j.GraphDatabase.driver(uri, auth=(user, password)) as driver:
        text2cypher_examples = _select_text2cypher_examples(args.question_style)
        if args.retriever_mode == "text2cypher":
            retriever = Text2CypherRetriever(
                driver=driver,
                llm=llm,
                neo4j_database=database,
                neo4j_schema=FINANCIAL_GRAPH_SCHEMA,
                examples=text2cypher_examples,
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
                examples=text2cypher_examples,
                custom_prompt=TEXT2CYPHER_PROMPT,
                result_formatter=_format_table_record,
            )
            primary_retriever = PrimaryStatementRetriever(
                driver=driver,
                neo4j_database=database,
            )
            retriever = DualRetriever(
                driver=driver,
                chunk_retriever=chunk_retriever,
                table_retriever=table_retriever,
                primary_retriever=primary_retriever,
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
            print(
                f"[{idx}/{len(questions)}] {row['doc_id']} :: {row['question_id']}",
                flush=True,
            )
            start = time.time()
            try:
                search_kwargs = {
                    "query_text": question_text,
                    "return_context": True,
                    "examples": _build_answer_guidance(row["question"], args.question_style),
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
                _save_results(output_path, results)
                print(f"  answer={answer!r} gold={row['gold_answer']}", flush=True)
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
                _save_results(output_path, results)
                print(f"  error={exc}", flush=True)

    _save_results(output_path, results)
    print(f"\nSaved results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
