from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from neo4j_graphrag.haystack_pipeline.io_utils import (
    build_context_preview,
    doc_key,
    format_duration_minsec,
    load_question_doc_map,
    parse_answer_and_evidence,
    read_json_or_jsonl,
    read_questions,
    save_results,
    strip_json_fence,
    truncate_text,
)


@dataclass
class IndexedDoc:
    id: str
    content: str
    embedding: np.ndarray
    doc_id: str
    chunk_index: int
    section_name: str
    page: Optional[int]
    token_set: set[str]


@dataclass
class IndexData:
    docs: list[IndexedDoc]
    matrix: np.ndarray
    positions_by_doc_key: dict[str, list[int]]
    neighbor_by_pos: dict[int, tuple[Optional[int], Optional[int]]]
    pos_by_doc_key: dict[str, list[int]]
    pos_ordinal_by_pos: dict[int, int]


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ANSWER_PREFIX_RE = re.compile(r"^\s*answer\s*[:：]\s*", flags=re.IGNORECASE)

# Table-aware context expansion (GraphRAG).
#
# Goal: when retrieval hits a financial table chunk that was cut mid-table,
# automatically pull adjacent chunks from the same document until the table ends.
#
# This is intentionally heuristic and biased toward low false positives:
# - Require multi-column numeric rows with clear column separation (2+ spaces / tabs)
# - Require multiple such rows per chunk
_NUM_TOKEN_RE = re.compile(
    r"(?:\(\s*)?\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?%?(?:\s*\))?",
    flags=re.IGNORECASE,
)
_COL_SEP_RE = re.compile(r"(?:\t| {2,})")
_LIKELY_NOISE_RE = re.compile(
    r"(?i)\b(page|item\s+\d+|form\s+10-?k|form\s+10-?q|table of contents|commission file number)\b"
)

_BALANCE_SHEET_PATTERNS = [
    "balance sheet",
    "balance sheets",
    "statement of financial position",
    "statements of financial position",
]
_CASH_FLOW_PATTERNS = [
    "cash flow statement",
    "cash flow statements",
    "statement of cash flows",
    "statements of cash flows",
]
_INCOME_STATEMENT_PATTERNS = [
    "income statement",
    "income statements",
    "statement of income",
    "statements of income",
    "statement of operations",
    "statements of operations",
    "p&l statement",
    "p&l",
    "profit and loss",
]
_LOW_VALUE_SECTION_PATTERNS = [
    "results of operations",
    "segment",
    "guidance",
    "contractual obligations",
    "streaming content",
    "opinion on the financial statements",
    "opinion",
    "exhibit",
    "table of contents",
    "five-year financial highlights",
    "selected financial data",
    "north america",
]


def _count_table_rows(text: str) -> int:
    rows = 0
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if _LIKELY_NOISE_RE.search(s):
            continue
        nums = _NUM_TOKEN_RE.findall(s)
        # Prefer explicit column separators (tabs / 2+ spaces), but allow very
        # number-dense lines without separators (common in PDF-extracted tables).
        has_sep = bool(_COL_SEP_RE.search(s))
        if (has_sep and len(nums) >= 2) or (len(nums) >= 4):
            rows += 1
    return rows


def _is_table_like_chunk(text: str) -> bool:
    # Use a moderate threshold so we actually trigger on statement fragments.
    # False positives are contained by the neighbor-expansion stop condition.
    return _count_table_rows(text) >= 3


def _contains_any(text: str, patterns: list[str]) -> bool:
    return any(p in text for p in patterns)


def _infer_target_statement(question: str) -> str | None:
    q = question.lower()
    if _contains_any(q, _BALANCE_SHEET_PATTERNS):
        return "balance_sheet"
    if _contains_any(q, _CASH_FLOW_PATTERNS):
        return "cash_flow"
    if _contains_any(q, _INCOME_STATEMENT_PATTERNS):
        return "income_statement"
    return None


def _statement_priority(*, question: str, row: dict[str, Any]) -> int:
    target = _infer_target_statement(question)
    if target is None:
        return 0

    section = str(row.get("section_name") or "").lower()
    text = str(row.get("text") or "").lower()
    head = f"{section}\n{text[:1200]}"

    if _contains_any(section, _LOW_VALUE_SECTION_PATTERNS):
        penalty = -3
    elif _contains_any(head, ["note ", "notes to", "exhibit", "table of contents"]):
        penalty = -2
    else:
        penalty = 0

    if target == "balance_sheet":
        if _contains_any(head, _BALANCE_SHEET_PATTERNS):
            return 6
        if (
            "assets" in head
            and "liabilities" in head
            and (
                "total current assets" in head
                or "total current liabilities" in head
                or "accounts payable" in head
                or "property, plant and equipment" in head
            )
        ):
            return 5
        if _contains_any(head, ["current assets", "current liabilities", "accounts payable", "total assets"]):
            return 3 + penalty
        return penalty

    if target == "cash_flow":
        if _contains_any(head, _CASH_FLOW_PATTERNS):
            return 6
        if _contains_any(
            head,
            [
                "cash flows from operating activities",
                "cash flows from investing activities",
                "cash flows from financing activities",
                "net cash provided by operating activities",
                "dividends paid",
                "purchases of property, plant and equipment",
            ],
        ):
            return 5
        return penalty

    if target == "income_statement":
        if _contains_any(head, _INCOME_STATEMENT_PATTERNS):
            return 6
        if _contains_any(
            head,
            [
                "net sales",
                "total revenues",
                "cost of sales",
                "operating income",
                "net income attributable",
                "income before income taxes",
            ],
        ):
            return 5
        return penalty

    return 0


def _apply_statement_aware_ranking(*, question: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not docs:
        return docs
    target = _infer_target_statement(question)
    if target is None:
        return docs

    scored: list[tuple[tuple[float, float, float], dict[str, Any]]] = []
    for row in docs:
        priority = _statement_priority(question=question, row=row)
        base = row.get("fused_score")
        if base is None:
            base = row.get("dense_score")
        base_score = float(base or 0.0)
        sparse_score = float(row.get("sparse_score") or 0.0)
        scored.append(((float(priority), base_score, sparse_score), row))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored]


def _expand_table_block(
    *,
    index_data: IndexData,
    pos: int,
    max_neighbors_each_side: int = 6,
) -> list[int]:
    """Return positions for a contiguous table region around pos within same doc."""
    if pos not in index_data.pos_ordinal_by_pos:
        return [pos]
    doc = index_data.docs[pos]
    ordered = index_data.positions_by_doc_key.get(doc_key(doc.doc_id), [])
    if not ordered:
        return [pos]
    center_ord = index_data.pos_ordinal_by_pos[pos]
    # Expand outwards while chunks remain table-like (or near table-like).
    left = center_ord
    right = center_ord
    # Walk left
    steps = 0
    while left - 1 >= 0 and steps < max_neighbors_each_side:
        cand_pos = ordered[left - 1]
        cand_text = index_data.docs[cand_pos].content
        if _is_table_like_chunk(cand_text) or _count_table_rows(cand_text) >= 2:
            left -= 1
            steps += 1
            continue
        break
    # Walk right
    steps = 0
    while right + 1 < len(ordered) and steps < max_neighbors_each_side:
        cand_pos = ordered[right + 1]
        cand_text = index_data.docs[cand_pos].content
        if _is_table_like_chunk(cand_text) or _count_table_rows(cand_text) >= 2:
            right += 1
            steps += 1
            continue
        break
    # Optionally include one header chunk immediately before the table region.
    if left - 1 >= 0:
        header_pos = ordered[left - 1]
        header_text = index_data.docs[header_pos].content
        if not _is_table_like_chunk(header_text) and len(header_text) <= 900:
            left -= 1
    return ordered[left : right + 1]

# Previous generic answer-only prompt is intentionally kept for reference.
# _ANSWER_ONLY_SYSTEM_INSTRUCTION = (
#     "You are a financial QA assistant. Use only the provided context. "
#     "Return only the final answer text, no JSON."
# )
#
# _ANSWER_ONLY_TEMPLATE = """You are a financial QA expert.
# Use only the provided financial context.
#
# Question:
# {query_text}
#
# Context:
# {context}
#
# Rules:
# 1. Return only the final answer text in a single concise line.
# 2. Do not output JSON, markdown, labels, or extra explanation.
# 3. For metrics questions, compute from context when needed and include units.
# 4. Do not default to "incomplete context".
# 5. If at least part of required numeric evidence exists, compute a best-supported numeric answer.
# 6. Only when no relevant value can be found at all, return exactly: "insufficient evidence".
#
# Final answer:
# """

_METRICS_ONLY_ANSWER_SYSTEM_INSTRUCTION = (
    "You are a financial QA assistant specialized in metrics-generated questions. "
    "Use only provided context. Follow the output format exactly."
)

_METRICS_ONLY_ANSWER_TEMPLATE = """Task: Answer a FinanceBench metrics-generated question using ONLY the provided context.

Question:
{query_text}

Context:
{context}

=== OUTPUT FORMAT (follow exactly) ===

STEP 1 - Identify line items:
List each numeric value you need. For EACH value write exactly one line:
  ITEM: <line item name> | YEAR: <fiscal year> | VALUE: <number> | UNIT: <USD millions / USD billions / percent / ratio / plain>

STEP 2 - Formula (skip if direct extraction):
Write the formula using the item names above, then show the arithmetic step by step.

STEP 3 - Final answer:
FINAL_ANSWER: <single number, optionally with $ or %>

=== RULES ===

1. Extract values ONLY for the specific company asked about. Ignore other companies in context.
2. Match the EXACT fiscal year(s) requested. FY2018 means the fiscal year ending in 2018.
3. Pay attention to units in the question:
   - "in USD millions" means your answer should be in millions (e.g. 5409, not 5.409 billion).
   - "in USD billions" means your answer should be in billions (e.g. 4.6, not 4600 million).
   - "in units of percents" means output a percentage number (e.g. 39.7, not 0.397).
   - Conversion reminder (SEC tables often state the table is in "USD millions"):
     - If the evidence/table says USD millions but the question asks USD billions: divide by 1000.
     - If the evidence/table says USD billions but the question asks USD millions: multiply by 1000.
     - If the evidence/table says USD thousands but the question asks USD millions: divide by 1000.
4. CRITICAL - Output format for ratios vs percentages:
   - If the question asks for a RATIO (e.g. "working capital ratio", "operating cash flow ratio", "payout ratio"),
     output as a plain decimal (e.g. 1.73, 0.66, 0.80). Do NOT multiply by 100.
   - If the question asks for a PERCENTAGE or "% margin" or "as a %" or "in units of percents",
     output as a percentage number (e.g. 39.7, 5.2). This IS multiplied by 100.
   - ROA, ROE: if the question says "ROA" without specifying units, output as a decimal ratio (e.g. 0.01, not 1.0).
     If the question says "ROA in %" or "ROA as a percentage", then multiply by 100.
   - Dividend payout ratio = Total dividends / Net income. This is a RATIO, output as decimal (e.g. 0.80, not 80).
   - CAGR: if question says "in units of percents", output percentage (e.g. 0.4). Otherwise decimal.
5. Formula definitions (use these EXACTLY when the question references them):
   - COGS % margin = COGS / Revenue * 100
   - Gross profit margin = (Revenue - COGS) / Revenue * 100.  NOTE: COGS margin != Gross margin.
   - Net profit margin = Net income / Revenue * 100
   - ROA = Net income / Average total assets. Average = (FY_prev + FY_current) / 2.
   - Operating cash flow ratio = Cash from operations / Total current liabilities
   - Fixed asset turnover = Revenue / Average PP&E
   - DSO = 365 * Average accounts receivable / Revenue
   - DIO = 365 * Average inventory / COGS
   - DPO = 365 * Average accounts payable / (COGS + change in inventory)
   - CCC = DIO + DSO - DPO. NOTE: CCC can be negative.
   - YoY change = (New - Old) / |Old| * 100
   - CAGR = ((End / Start) ^ (1 / years) - 1) * 100  (if answer in percents)
   - Working capital ratio = Total current assets / Total current liabilities
   - Dividend payout ratio = Total cash dividends paid / Net income (this is a ratio, NOT a percentage)
   - EBITDA = Operating income + Depreciation & Amortization
   - EBITDA % margin = EBITDA / Revenue * 100
6. For multi-year averages: compute each year's value separately, then average them.
7. SANITY CHECK (mandatory before giving final answer):
   - Margins (COGS%, gross%, net%, EBITDA%): must be between -50% and 100%. If outside, recheck.
   - Ratios (working capital, current, cash flow): typically 0.1 to 10. If >50 or <0, recheck.
   - ROA/ROE as decimal: typically -0.5 to 0.5. If >5, you likely forgot to divide by total assets.
   - Dividend payout ratio: typically 0 to 2.0. If >10, you likely multiplied by 100 when you shouldn't.
   - YoY change %: typically -80% to +200%. If >500%, recheck values.
   - Revenue/assets in millions: typically 100 to 500,000. If <1 or >10,000,000, check units.
   - If your answer fails sanity check, go back to STEP 1 and verify each extracted value.
8. If the question says "round to X decimal places", do so.
9. When extracting from balance sheets: "year end FY2019" means the balance as of the END of fiscal year 2019.
   For companies with Jan fiscal year end (e.g. Best Buy, Walmart), FY2019 ends in Jan/Feb 2020.
   Look for the LATER date column in the balance sheet.
10. For "net" values (net AR, net PP&E): use the value AFTER deducting allowances/depreciation, not gross.
11. Return "FINAL_ANSWER: insufficient evidence" only when the context contains NO relevant numeric data at all.

=== WORKED EXAMPLE A (ratio) ===

Question: What is the FY2019 fixed asset turnover ratio for Activision Blizzard?
Defined as: FY2019 revenue / (average PP&E between FY2018 and FY2019). Round to two decimal places.

STEP 1 - Identify line items:
ITEM: Total net revenues | YEAR: FY2019 | VALUE: 6489 | UNIT: USD millions
ITEM: Property and equipment, net | YEAR: FY2018 | VALUE: 253 | UNIT: USD millions
ITEM: Property and equipment, net | YEAR: FY2019 | VALUE: 282 | UNIT: USD millions

STEP 2 - Formula:
Fixed asset turnover = Revenue / Average PP&E
Average PP&E = (253 + 282) / 2 = 267.5
Ratio = 6489 / 267.5 = 24.26
Sanity check: fixed asset turnover of 24.26 is high but plausible for a software company with little PP&E. OK.

STEP 3 - Final answer:
FINAL_ANSWER: 24.26

=== WORKED EXAMPLE B (YoY % change) ===

Question: What is the year-over-year change in revenue from FY2016 to FY2017 (in units of percents)?

STEP 1 - Identify line items:
ITEM: Revenue | YEAR: FY2016 | VALUE: 135987 | UNIT: USD millions
ITEM: Revenue | YEAR: FY2017 | VALUE: 177866 | UNIT: USD millions

STEP 2 - Formula:
YoY change = (New - Old) / |Old| * 100
= (177866 - 135987) / 135987 * 100
= 41879 / 135987 * 100 = 30.8%
Sanity check: 30.8% revenue growth is plausible. OK.

STEP 3 - Final answer:
FINAL_ANSWER: 30.8%

=== NOW ANSWER THE QUESTION ===
"""


def _load_rag_prompt_template() -> tuple[str, str]:
    prompts_path = Path(__file__).resolve().parents[1] / "rag" / "prompts.py"
    payload = runpy.run_path(str(prompts_path))
    rag_template_cls = payload.get("RagTemplate")
    if rag_template_cls is None:
        raise RuntimeError(f"RagTemplate is not found in {prompts_path}")
    template = rag_template_cls()
    return str(template.system_instructions), str(template.template)


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


_FINAL_ANSWER_RE = re.compile(
    r"FINAL[_ ]?ANSWER\s*[:：]\s*(.+)",
    flags=re.IGNORECASE,
)


def _parse_answer_only(raw: Any) -> str:
    if raw is None:
        return ""
    text = strip_json_fence(str(raw).strip())
    if not text:
        return ""

    # Prefer FINAL_ANSWER: line if present (structured prompt output).
    for line in reversed(text.splitlines()):
        m = _FINAL_ANSWER_RE.search(line.strip())
        if m:
            val = m.group(1).strip()
            if val.lower() in {"insufficient evidence", "none", "n/a"}:
                return "insufficient evidence"
            val = re.sub(r"\s+", " ", val).strip()
            if val:
                return val

    # Allow fallback if the model still returns JSON.
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            answer = payload.get("answer", payload.get("final_answer"))
            if answer is not None:
                return re.sub(r"\s+", " ", str(answer)).strip()
    except Exception:
        pass

    text = _ANSWER_PREFIX_RE.sub("", text).strip()
    text = re.sub(r"\s*\(incomplete context\)\s*$", "", text, flags=re.IGNORECASE).strip()
    if text.lower() in {"incomplete context", "(incomplete context)"}:
        return "insufficient evidence"
    return re.sub(r"\s+", " ", text).strip()


def _clip_for_generation(text: str, max_chars: int) -> str:
    limit = max(1, int(max_chars))
    return truncate_text(text, limit)


def _import_haystack_text_embedder():
    try:
        from haystack.components.embedders import SentenceTransformersTextEmbedder
    except ImportError as exc:
        raise ImportError(
            "Haystack text embedder is unavailable. Install with `pip install haystack-ai sentence-transformers`."
        ) from exc
    return SentenceTransformersTextEmbedder


def _load_index(path: Path) -> IndexData:
    docs: list[IndexedDoc] = []
    matrix_rows: list[np.ndarray] = []
    positions_by_doc_key: dict[str, list[int]] = {}
    pos_by_doc_key: dict[str, list[int]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            doc_id = str((payload.get("meta") or {}).get("doc_id") or "").strip()
            if not doc_id:
                raise ValueError(f"Missing meta.doc_id in {path}:{line_no}")
            chunk_id = str(payload.get("id") or (payload.get("meta") or {}).get("chunk_id") or "").strip()
            if not chunk_id:
                raise ValueError(f"Missing id/chunk_id in {path}:{line_no}")
            content = str(payload.get("content") or "")
            embedding = payload.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise ValueError(f"Missing embedding in {path}:{line_no}")
            emb = np.asarray(embedding, dtype=np.float32)
            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm
            meta = payload.get("meta") or {}
            chunk_index = int(meta.get("chunk_index") or 0)
            section_name = str(meta.get("section_name") or "UNSPECIFIED")
            page = meta.get("page")
            page_int: Optional[int] = None
            if page is not None:
                try:
                    page_int = int(page)
                except (TypeError, ValueError):
                    page_int = None

            pos = len(docs)
            docs.append(
                IndexedDoc(
                    id=chunk_id,
                    content=content,
                    embedding=emb,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    section_name=section_name,
                    page=page_int,
                    token_set=_tokenize(content),
                )
            )
            matrix_rows.append(emb)
            key = doc_key(doc_id)
            positions_by_doc_key.setdefault(key, []).append(pos)
            pos_by_doc_key.setdefault(key, []).append(pos)

    if not docs:
        raise ValueError(f"No documents loaded from {path}")

    neighbor_by_pos: dict[int, tuple[Optional[int], Optional[int]]] = {}
    pos_ordinal_by_pos: dict[int, int] = {}
    for key, positions in positions_by_doc_key.items():
        ordered = sorted(positions, key=lambda p: (docs[p].chunk_index, docs[p].id))
        positions_by_doc_key[key] = ordered
        for idx, pos in enumerate(ordered):
            prev_pos = ordered[idx - 1] if idx > 0 else None
            next_pos = ordered[idx + 1] if idx < len(ordered) - 1 else None
            neighbor_by_pos[pos] = (prev_pos, next_pos)
            pos_ordinal_by_pos[pos] = idx

    matrix = np.vstack(matrix_rows).astype(np.float32)
    return IndexData(
        docs=docs,
        matrix=matrix,
        positions_by_doc_key=positions_by_doc_key,
        neighbor_by_pos=neighbor_by_pos,
        pos_by_doc_key=pos_by_doc_key,
        pos_ordinal_by_pos=pos_ordinal_by_pos,
    )


def _embed_query(embedder: Any, query: str) -> np.ndarray:
    result = embedder.run(text=query)
    embedding = None
    if isinstance(result, dict):
        embedding = result.get("embedding")
        if embedding is None:
            embeddings = result.get("embeddings")
            if isinstance(embeddings, list) and embeddings:
                embedding = embeddings[0]
    if embedding is None:
        raise RuntimeError("Could not read embedding from Haystack text embedder output.")
    vec = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec


def _rrf_fuse(
    *,
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    top_k: int,
    rrf_k: int,
) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for source_name, hits in (("dense", dense_hits), ("sparse", sparse_hits)):
        for rank, hit in enumerate(hits, start=1):
            pos = int(hit["pos"])
            row = merged.setdefault(
                pos,
                {
                    "pos": pos,
                    "dense_score": None,
                    "sparse_score": None,
                    "fused_score": 0.0,
                    "sources": [],
                },
            )
            row["fused_score"] += 1.0 / (rrf_k + rank)
            if source_name not in row["sources"]:
                row["sources"].append(source_name)
            if hit.get("dense_score") is not None:
                row["dense_score"] = hit["dense_score"]
            if hit.get("sparse_score") is not None:
                row["sparse_score"] = hit["sparse_score"]
    return sorted(merged.values(), key=lambda x: x["fused_score"], reverse=True)[:top_k]


def _linear_fuse(
    *,
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    top_k: int,
    alpha: float,
) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for hit in dense_hits:
        pos = int(hit["pos"])
        merged[pos] = {
            "pos": pos,
            "dense_score": hit.get("dense_score"),
            "sparse_score": None,
            "fused_score": 0.0,
            "sources": ["dense"],
        }
    for hit in sparse_hits:
        pos = int(hit["pos"])
        row = merged.setdefault(
            pos,
            {
                "pos": pos,
                "dense_score": None,
                "sparse_score": None,
                "fused_score": 0.0,
                "sources": [],
            },
        )
        row["sparse_score"] = hit.get("sparse_score")
        if "sparse" not in row["sources"]:
            row["sources"].append("sparse")

    dense_vals = [float(h.get("dense_score") or 0.0) for h in dense_hits]
    sparse_vals = [float(h.get("sparse_score") or 0.0) for h in sparse_hits]
    d_min = min(dense_vals) if dense_vals else 0.0
    d_max = max(dense_vals) if dense_vals else 0.0
    s_min = min(sparse_vals) if sparse_vals else 0.0
    s_max = max(sparse_vals) if sparse_vals else 0.0

    def _scale(value: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 1.0 if value > 0 else 0.0
        return (value - lo) / (hi - lo)

    for row in merged.values():
        d = float(row.get("dense_score") or 0.0)
        s = float(row.get("sparse_score") or 0.0)
        row["fused_score"] = alpha * _scale(d, d_min, d_max) + (1 - alpha) * _scale(
            s, s_min, s_max
        )
    return sorted(merged.values(), key=lambda x: x["fused_score"], reverse=True)[:top_k]


def _safe_top_indices(scores: np.ndarray, top_n: int) -> np.ndarray:
    if top_n <= 0:
        return np.array([], dtype=np.int32)
    top_n = min(top_n, scores.shape[0])
    if top_n == scores.shape[0]:
        return np.argsort(-scores)
    part = np.argpartition(-scores, top_n - 1)[:top_n]
    return part[np.argsort(-scores[part])]


def _vector_retrieve(
    *,
    index_data: IndexData,
    query_vec: np.ndarray,
    candidate_pos: np.ndarray,
    top_k: int,
    effective_search_ratio: int,
) -> list[dict[str, Any]]:
    sub = index_data.matrix[candidate_pos]
    dense_scores = np.dot(sub, query_vec)
    search_k = max(top_k, top_k * max(1, effective_search_ratio))
    top_idx_local = _safe_top_indices(dense_scores, search_k)
    hits: list[dict[str, Any]] = []
    for local_pos in top_idx_local:
        global_pos = int(candidate_pos[int(local_pos)])
        hits.append(
            {
                "pos": global_pos,
                "dense_score": float(dense_scores[int(local_pos)]),
                "sources": ["dense"],
            }
        )
    return hits


def _sparse_retrieve(
    *,
    index_data: IndexData,
    query_tokens: set[str],
    candidate_pos: np.ndarray,
    top_k: int,
    effective_search_ratio: int,
) -> list[dict[str, Any]]:
    if not query_tokens:
        return []
    search_k = max(top_k, top_k * max(1, effective_search_ratio))
    scores: list[tuple[int, float]] = []
    qn = float(len(query_tokens))
    for pos in candidate_pos.tolist():
        tokens = index_data.docs[int(pos)].token_set
        if not tokens:
            continue
        overlap = len(query_tokens.intersection(tokens))
        if overlap <= 0:
            continue
        scores.append((int(pos), overlap / qn))
    scores.sort(key=lambda x: x[1], reverse=True)
    scores = scores[:search_k]
    return [
        {"pos": pos, "sparse_score": score, "sources": ["sparse"]}
        for pos, score in scores
    ]


def _try_haystack_rerank(
    *,
    query: str,
    docs: list[dict[str, Any]],
    reranker_model: Optional[str],
    rerank_top_k: int,
    enabled: bool,
) -> list[dict[str, Any]]:
    if not enabled or not reranker_model or rerank_top_k <= 0 or not docs:
        return docs
    try:
        from haystack import Document
        try:
            from haystack.components.rankers import SentenceTransformersSimilarityRanker as _Ranker
        except ImportError:
            from haystack.components.rankers import TransformersSimilarityRanker as _Ranker
    except Exception:
        return docs

    try:
        ranker = _Ranker(model=reranker_model, top_k=rerank_top_k)
        if hasattr(ranker, "warm_up"):
            ranker.warm_up()
        in_docs = [
            Document(
                id=row["chunk_id"],
                content=row["text"],
                meta={"_pos": row["pos"]},
            )
            for row in docs
        ]
        result = ranker.run(query=query, documents=in_docs, top_k=rerank_top_k)
        out_docs = result.get("documents") if isinstance(result, dict) else None
        if not isinstance(out_docs, list):
            return docs
        pos_rank = [int((getattr(d, "meta", {}) or {}).get("_pos")) for d in out_docs]
        score_by_pos = {row["pos"]: row for row in docs}
        reranked: list[dict[str, Any]] = []
        seen: set[int] = set()
        for pos in pos_rank:
            if pos in seen:
                continue
            row = score_by_pos.get(pos)
            if row is None:
                continue
            seen.add(pos)
            reranked.append(row)
        for row in docs:
            if row["pos"] not in seen:
                reranked.append(row)
        return reranked[: max(1, rerank_top_k)]
    except Exception:
        return docs


def _init_generator(args: argparse.Namespace) -> tuple[Optional[Any], str]:
    if args.retrieval_only or args.generator_backend == "none":
        return (None, "none")

    if args.ollama_timeout_seconds is not None:
        try:
            from ollama import Client

            client = Client(host=args.ollama_host, timeout=args.ollama_timeout_seconds)
            return (client, args.llm_model)
        except Exception as exc:
            raise RuntimeError(
                "Could not initialize Ollama client with timeout. "
                "Install `ollama` and ensure `ollama serve` is reachable."
            ) from exc

    # Preferred: Haystack Ollama integration.
    for import_path in (
        "haystack_integrations.components.generators.ollama",
        "ollama_haystack",
    ):
        try:
            module = __import__(import_path, fromlist=["OllamaGenerator"])
            klass = getattr(module, "OllamaGenerator")
            last_exc: Optional[Exception] = None
            for kwargs in (
                {
                    "model": args.llm_model,
                    "url": args.ollama_host,
                    "generation_kwargs": {"temperature": args.temperature},
                },
                {
                    "model": args.llm_model,
                    "base_url": args.ollama_host,
                    "generation_kwargs": {"temperature": args.temperature},
                },
                {
                    "model": args.llm_model,
                    "generation_kwargs": {"temperature": args.temperature},
                },
            ):
                try:
                    return (klass(**kwargs), args.llm_model)
                except Exception as exc:
                    last_exc = exc
            if last_exc is not None:
                raise last_exc
        except Exception:
            continue

    # Fallback: raw Ollama client, keeps command flow working.
    try:
        from ollama import Client

        client = Client(host=args.ollama_host)
        return (client, args.llm_model)
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize generator. Install `ollama-haystack` (preferred) "
            "or `ollama`, and ensure `ollama serve` is reachable."
        ) from exc


def _generate(
    *,
    generator: Any,
    prompt: str,
    system_instruction: str,
    args: argparse.Namespace,
) -> str:
    # Haystack generator path
    if generator is not None and generator.__class__.__name__ == "OllamaGenerator":
        result = generator.run(prompt=prompt)
        if isinstance(result, dict):
            replies = result.get("replies")
            if isinstance(replies, list) and replies:
                return str(replies[0])
            generations = result.get("generations")
            if isinstance(generations, list) and generations:
                first = generations[0]
                if isinstance(first, str):
                    return first
                if isinstance(first, dict):
                    text = first.get("text")
                    if text is not None:
                        return str(text)
        return str(result)

    # Raw ollama client fallback path
    if generator is not None and generator.__class__.__name__ == "Client":
        response = generator.chat(
            model=args.llm_model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": args.temperature},
        )
        if isinstance(response, dict):
            message = response.get("message") or {}
            if isinstance(message, dict):
                return str(message.get("content", ""))
        message = getattr(response, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if content is not None:
                return str(content)
        return str(response)

    raise RuntimeError("Generator is not initialized.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Haystack-based compare runner. Uses local indexed embeddings and supports "
            "retrieval-only diagnostics."
        )
    )
    parser.add_argument("questions", nargs="*", help="Inline questions.")
    parser.add_argument("-q", "--question", action="append", help="Inline question (repeatable).")
    parser.add_argument("--questions-file", help="Text file with one question per line.")
    parser.add_argument(
        "--mode",
        choices=["rag", "graphrag", "compare"],
        default="compare",
        help="Run rag, graphrag, or both.",
    )
    parser.add_argument(
        "--retrieval",
        choices=["vector", "hybrid"],
        default="vector",
        help="Retrieval mode.",
    )
    parser.add_argument("--index-path", required=True, help="Path to documents_with_embeddings.jsonl.")
    parser.add_argument(
        "--index-meta-path",
        help="Optional index meta path (defaults to sibling index_meta.json).",
    )
    parser.add_argument("--output", required=True, help="Output path for results.")
    parser.add_argument(
        "--output-format",
        choices=["json", "jsonl", "csv"],
        default="json",
        help="Output format.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Final top-k for answer context.")
    parser.add_argument(
        "--effective-search-ratio",
        type=int,
        default=1,
        help="Multiplier for dense/sparse candidate pools.",
    )
    parser.add_argument(
        "--hybrid-ranker",
        choices=["naive", "linear"],
        default="naive",
        help="Fusion strategy in hybrid mode.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Linear fusion weight for dense score (only used with --hybrid-ranker=linear).",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF k constant when --hybrid-ranker=naive.",
    )
    parser.add_argument(
        "--question-doc-map",
        help="Optional JSON/JSONL question->doc mapping for retrieval filtering.",
    )
    parser.add_argument(
        "--qa-path",
        help="Optional QA JSONL path, used to populate question_id and gold_type in output.",
    )
    parser.add_argument(
        "--embedding-model",
        help="Override query embedding model. Defaults to model stored in index meta.",
    )
    parser.add_argument(
        "--reranker-model",
        help="Optional Haystack ranker model. If missing, rerank step is skipped.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=0,
        help="If >0 and reranker is available, rerank retrieved docs to this top-k.",
    )
    parser.add_argument(
        "--disable-rerank",
        action="store_true",
        help="Disable reranker even when model is provided.",
    )
    parser.add_argument(
        "--statement-aware-rerank",
        action="store_true",
        help=(
            "Apply query-aware post-ranking that boosts chunks matching the statement "
            "type explicitly requested in the question (balance sheet / cash flow / income statement)."
        ),
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Run retrieval diagnostics only; skip generation.",
    )
    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="Use answer-only prompt and store a plain final answer (no JSON expected).",
    )
    parser.add_argument(
        "--generator-backend",
        choices=["ollama", "none"],
        default="ollama",
        help="Generation backend (ignored when --retrieval-only).",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:72b"),
        help="LLM model for generation.",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama host URL.",
    )
    parser.add_argument(
        "--ollama-timeout-seconds",
        type=float,
        default=(
            float(os.environ["OLLAMA_TIMEOUT_SECONDS"])
            if os.environ.get("OLLAMA_TIMEOUT_SECONDS")
            else None
        ),
        help=(
            "Optional Ollama request timeout in seconds. "
            "When set, run_compare uses the raw Ollama Python client so the timeout is enforced."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--preview-items", type=int, default=3, help="Context preview item count.")
    parser.add_argument("--preview-chars", type=int, default=500, help="Context preview char limit.")
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=48000,
        help="Maximum characters passed to the generator as context.",
    )
    parser.add_argument(
        "--graphrag-block-chars",
        type=int,
        default=2000,
        help=(
            "Per-item character cap for GraphRAG expanded blocks "
            "(previous/chunk/next) before joining final context."
        ),
    )
    parser.add_argument(
        "--evidence-items",
        type=int,
        default=20,
        help="Number of evidence rows stored in output.",
    )
    parser.add_argument(
        "--evidence-chars",
        type=int,
        default=400,
        help="Character limit per evidence excerpt.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    questions = read_questions(
        questions_file=args.questions_file,
        inline_questions=(args.question or []) + (args.questions or []),
    )
    if not questions:
        print("No questions provided. Use --questions-file or -q.", file=sys.stderr)
        sys.exit(1)

    index_path = Path(args.index_path).expanduser().resolve()
    if not index_path.exists():
        print(f"Index not found: {index_path}", file=sys.stderr)
        sys.exit(1)

    index_meta_path = (
        Path(args.index_meta_path).expanduser().resolve()
        if args.index_meta_path
        else index_path.parent / "index_meta.json"
    )
    index_meta: dict[str, Any] = {}
    if index_meta_path.exists():
        index_meta = json.loads(index_meta_path.read_text(encoding="utf-8"))

    embedder_model = args.embedding_model or index_meta.get("embedder_model")
    if not embedder_model:
        print(
            "Embedding model is not specified. Use --embedding-model or provide index_meta.json.",
            file=sys.stderr,
        )
        sys.exit(1)

    index_data = _load_index(index_path)
    question_doc_map = load_question_doc_map(args.question_doc_map)

    qa_by_question: dict[str, dict[str, Any]] = {}
    if args.qa_path:
        for row in read_json_or_jsonl(Path(args.qa_path).expanduser().resolve()):
            q = str(row.get("question", "")).strip()
            if q and q not in qa_by_question:
                qa_by_question[q] = row

    SentenceTransformersTextEmbedder = _import_haystack_text_embedder()
    text_embedder = SentenceTransformersTextEmbedder(model=embedder_model)
    if hasattr(text_embedder, "warm_up"):
        text_embedder.warm_up()

    generator, generator_model = _init_generator(args)
    if args.answer_only:
        system_instruction = _METRICS_ONLY_ANSWER_SYSTEM_INSTRUCTION
        prompt_template_text = _METRICS_ONLY_ANSWER_TEMPLATE
    else:
        system_instruction, prompt_template_text = _load_rag_prompt_template()
    modes = ["rag", "graphrag"] if args.mode == "compare" else [args.mode]

    run_rows: list[dict[str, Any]] = []
    total_runs = len(questions) * len(modes)
    done = 0
    for question in questions:
        gold_row = qa_by_question.get(question, {})
        question_id = gold_row.get("_id")
        gold_type = gold_row.get("gold_type")
        allowed_doc_ids = sorted(question_doc_map.get(question, set()))
        allowed_doc_keys = {doc_key(x) for x in allowed_doc_ids if isinstance(x, str) and x.strip()}

        if allowed_doc_keys:
            candidate_positions = []
            for key in allowed_doc_keys:
                candidate_positions.extend(index_data.pos_by_doc_key.get(key, []))
            candidate_pos = np.asarray(candidate_positions, dtype=np.int32)
        else:
            candidate_pos = np.arange(len(index_data.docs), dtype=np.int32)

        for mode in modes:
            done += 1
            print(
                f"[{done}/{total_runs}] mode={mode} retrieval={args.retrieval} "
                f"index=haystack_local | {question}"
            )
            started = time.perf_counter()

            answer = ""
            evidence = ""
            error: Optional[str] = None
            retriever_name = ""
            retrieved_rows: list[dict[str, Any]] = []
            evidence_rows: list[dict[str, Any]] = []
            chunk_ids: list[str] = []
            context_preview = ""
            try:
                if candidate_pos.size == 0:
                    raise RuntimeError("No candidate chunks after doc filter.")

                query_vec = _embed_query(text_embedder, question)
                dense_hits = _vector_retrieve(
                    index_data=index_data,
                    query_vec=query_vec,
                    candidate_pos=candidate_pos,
                    top_k=args.top_k,
                    effective_search_ratio=args.effective_search_ratio,
                )
                if args.retrieval == "vector":
                    fused_hits = dense_hits[: args.top_k]
                    retriever_name = "HaystackVectorRetriever"
                else:
                    sparse_hits = _sparse_retrieve(
                        index_data=index_data,
                        query_tokens=_tokenize(question),
                        candidate_pos=candidate_pos,
                        top_k=args.top_k,
                        effective_search_ratio=args.effective_search_ratio,
                    )
                    if args.hybrid_ranker == "linear":
                        fused_hits = _linear_fuse(
                            dense_hits=dense_hits,
                            sparse_hits=sparse_hits,
                            top_k=args.top_k,
                            alpha=float(args.alpha),
                        )
                        retriever_name = "HaystackHybridLinearRetriever"
                    else:
                        fused_hits = _rrf_fuse(
                            dense_hits=dense_hits,
                            sparse_hits=sparse_hits,
                            top_k=args.top_k,
                            rrf_k=args.rrf_k,
                        )
                        retriever_name = "HaystackHybridRRFRetriever"

                for row in fused_hits:
                    pos = int(row["pos"])
                    doc = index_data.docs[pos]
                    retrieved_rows.append(
                        {
                            "pos": pos,
                            "chunk_id": doc.id,
                            "doc_id": doc.doc_id,
                            "section_name": doc.section_name,
                            "chunk_index": doc.chunk_index,
                            "page": doc.page,
                            "dense_score": row.get("dense_score"),
                            "sparse_score": row.get("sparse_score"),
                            "fused_score": row.get("fused_score"),
                            "sources": row.get("sources", []),
                            "text": doc.content,
                        }
                    )

                retrieved_rows = _try_haystack_rerank(
                    query=question,
                    docs=retrieved_rows,
                    reranker_model=args.reranker_model,
                    rerank_top_k=args.rerank_top_k,
                    enabled=not args.disable_rerank,
                )
                if args.statement_aware_rerank:
                    retrieved_rows = _apply_statement_aware_ranking(
                        question=question,
                        docs=retrieved_rows,
                    )

                chunk_ids = [row["chunk_id"] for row in retrieved_rows]

                evidence_rows = []
                graphrag_context_blocks: list[str] = []
                for rank, row in enumerate(retrieved_rows[: max(0, args.evidence_items)], start=1):
                    pos = int(row["pos"])
                    text_for_mode = row["text"]
                    if mode == "graphrag":
                        parts: list[str] = []
                        center_text = row["text"]
                        if _is_table_like_chunk(center_text):
                            expanded_positions = _expand_table_block(index_data=index_data, pos=pos)
                            for ppos in expanded_positions:
                                tag = "Chunk" if ppos == pos else "Neighbor"
                                pdoc = index_data.docs[ppos]
                                parts.append(f"[{tag} doc={pdoc.doc_id} idx={pdoc.chunk_index}]\n{pdoc.content}")
                        else:
                            prev_pos, next_pos = index_data.neighbor_by_pos.get(pos, (None, None))
                            if prev_pos is not None:
                                parts.append(f"[Previous]\n{index_data.docs[prev_pos].content}")
                            parts.append(f"[Chunk]\n{center_text}")
                            if next_pos is not None:
                                parts.append(f"[Next]\n{index_data.docs[next_pos].content}")
                        text_for_mode = "\n\n".join(parts) if parts else center_text
                        graphrag_context_blocks.append(
                            _clip_for_generation(text_for_mode, args.graphrag_block_chars)
                        )

                    evidence_rows.append(
                        {
                            "rank": rank,
                            "chunk_id": row["chunk_id"],
                            "doc_id": row["doc_id"],
                            "score": row.get("fused_score")
                            if row.get("fused_score") is not None
                            else row.get("dense_score"),
                            "dense_score": row.get("dense_score"),
                            "sparse_score": row.get("sparse_score"),
                            "fused_score": row.get("fused_score"),
                            "section_name": row.get("section_name"),
                            "chunk_index": row.get("chunk_index"),
                            "page": row.get("page"),
                            "text_excerpt": truncate_text(
                                text_for_mode, max(1, args.evidence_chars)
                            ),
                        }
                    )

                context_preview = build_context_preview(
                    evidence_rows, args.preview_items, args.preview_chars
                )

                if not args.retrieval_only and retrieved_rows:
                    if mode == "rag":
                        context_text = "\n".join(row["text"] for row in retrieved_rows)
                    else:
                        mode_parts = graphrag_context_blocks or [row["text"] for row in retrieved_rows]
                        context_text = "\n\n".join(mode_parts)
                    context_text = _clip_for_generation(context_text, args.max_context_chars)

                    prompt = prompt_template_text.format(
                        query_text=question,
                        context=context_text,
                        examples="",
                    )
                    raw = _generate(
                        generator=generator,
                        prompt=prompt,
                        system_instruction=system_instruction,
                        args=args,
                    )
                    if args.answer_only:
                        answer = _parse_answer_only(raw)
                        evidence = ""
                    else:
                        answer, evidence = parse_answer_and_evidence(raw)
                    if not evidence:
                        evidence = "\n".join(
                            r["text_excerpt"] for r in evidence_rows[:3] if r.get("text_excerpt")
                        )
            except Exception as exc:
                error = str(exc)
                print(f"  error: {error}")

            latency_seconds = round(time.perf_counter() - started, 2)
            latency = format_duration_minsec(latency_seconds)
            retrieval_status = (
                "error"
                if error
                else "no_hits"
                if not retrieved_rows
                else "hits"
            )

            row = {
                "mode": mode,
                "retrieval": args.retrieval,
                "question_id": question_id,
                "question": question,
                "gold_type": gold_type,
                "answer": answer,
                "evidence": evidence,
                "latency": latency,
                "latency_seconds": latency_seconds,
                "retriever": retriever_name,
                "retrieval_status": retrieval_status,
                "retrieved_chunk_count": len(chunk_ids),
                "retrieved_item_count": len(retrieved_rows),
                "retrieved_chunk_ids": chunk_ids,
                "retrieved_evidence": [
                    {
                        "rank": e["rank"],
                        "chunk_id": e["chunk_id"],
                        "score": e.get("score"),
                        "dense_score": e.get("dense_score"),
                        "sparse_score": e.get("sparse_score"),
                        "fused_score": e.get("fused_score"),
                        "section_name": e.get("section_name"),
                        "chunk_index": e.get("chunk_index"),
                        "page": e.get("page"),
                        "text_excerpt": e.get("text_excerpt"),
                        "doc_id": e.get("doc_id"),
                    }
                    for e in evidence_rows
                ],
                "doc_filter": allowed_doc_ids,
                "embedding_model": embedder_model,
                "reranker_model": args.reranker_model or "",
                "generator_model": generator_model,
                "context_preview": context_preview,
                "error": error,
            }
            run_rows.append(row)

            print(
                f"  retrieved: status={retrieval_status} chunks={len(chunk_ids)} items={len(retrieved_rows)}"
            )
            print(f"  latency: {latency}")

    save_results(output_path=args.output, output_format=args.output_format, rows=run_rows)
    print(f"Saved structured results to {args.output}")


if __name__ == "__main__":
    main()
