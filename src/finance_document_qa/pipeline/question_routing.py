from __future__ import annotations

import json
import re
from typing import Any, Callable

from finance_document_qa.pipeline.utils import _normalize_text


ROUTER_PROMPT = """You are routing FinanceBench questions into exactly one QA pipeline.

Choose exactly one label:

direct
- The question asks for one specific fact, entity, date, filing item, financial line item, or raw amount.
- No ratio, formula, percentage change, trend judgment, or multi-step business explanation is required.

calculation
- The question asks for a computed metric, ratio, margin, turnover, growth rate, percentage, year-over-year change, comparison, increase/decrease, average, ROA/ROE, quick ratio, current ratio, fixed asset turnover, free cash flow conversion, or formula-based answer.
- This includes yes/no questions where the answer depends mainly on computing or comparing financial values.

domain
- The question asks for finance/business judgment, financial health, metric applicability, capital intensity, liquidity profile, margin profile, risk, trend explanation, business meaning, whether a metric is useful/meaningful, what drove a change, or why something happened.
- These questions may require calculations, but the final answer is a business/financial interpretation or judgment.

novel
- The question is open-ended, multi-step, or cross-section reasoning not mainly a standard financial metric calculation.
- It may ask about policies, filings, governance, derivatives, customer retention, strategic events, acquisitions, business segments, or other document-specific reasoning.

Priority rules:
- If the question asks for metric applicability or business interpretation after a calculation, choose domain, not calculation.
- If it asks only for a formula result or numeric comparison, choose calculation.
- If it asks only for one fact or raw value, choose direct.
- If none of the above clearly fits and the question requires broad document reasoning, choose novel.

Return JSON only:
{{"label": "direct|calculation|domain|novel", "reason": "short reason"}}

Question:
{question}
"""


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def extract_json_object(text: str) -> str | None:
    cleaned = strip_think(text)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    return match.group(0) if match else None


def classify_question_style_auto(question: str) -> tuple[str, float, str]:
    """Rule-only router for FinanceBench question styles."""
    q = (question or "").lower()
    domain_hits: list[str] = []
    hits: list[str] = []

    def hit(tag: str) -> None:
        if tag not in hits:
            hits.append(tag)

    def domain_hit(tag: str) -> None:
        if tag not in domain_hits:
            domain_hits.append(tag)

    if "capital-intensive" in q or "capital intensive" in q:
        domain_hit("capital_intensity")
    if "what drove" in q or "what drives" in q or "driven by" in q:
        domain_hit("drivers")
    if re.search(r"\bwhy\b", q) or "due to" in q or "primarily due" in q:
        domain_hit("causal_explanation")
    if any(
        phrase in q
        for phrase in (
            "useful metric",
            "meaningful metric",
            "relevant metric",
            "not a useful metric",
        )
    ):
        domain_hit("metric_applicability")
    if any(
        phrase in q
        for phrase in (
            "business based on",
            "financial health",
            "liquidity profile",
            "margin profile",
        )
    ):
        domain_hit("business_judgment")

    if domain_hits:
        conf = min(0.98, 0.78 + 0.04 * len(domain_hits))
        return "domain", conf, "matched: " + ", ".join(domain_hits)

    if "defined as" in q:
        hit("defined_as")
    if "cagr" in q:
        hit("cagr")
    if re.search(r"\byoy\b", q) or "year-over-year" in q or "year over year" in q:
        hit("yoy")
    if re.search(r"\bratio\b", q):
        hit("ratio")
    if re.search(r"\bmargins?\b", q):
        hit("margin")
    if re.search(r"\baverage\b", q):
        hit("average")
    if re.search(r"\bgrowth\b", q):
        hit("growth")
    if re.search(r"\bturnover\b", q):
        hit("turnover")
    if re.search(r"\bdpo\b", q) or re.search(r"\bccc\b", q):
        hit("dpo_ccc")
    if re.search(r"\broa\b", q) or "return on assets" in q or "return on asset" in q:
        hit("roa")
    if "% of revenue" in q or "percent of revenue" in q or "percentage of revenue" in q:
        hit("pct_of_revenue")
    if re.search(r"between\s+fy", q):
        hit("between_fy")
    if re.search(r"fy\s*\d{4}\s*[-–]\s*fy\s*\d{4}", q):
        hit("multi_fy_dash")
    if re.search(r"from\s+fy\s*\d{4}\s+to\s+fy\s*\d{4}", q):
        hit("from_fy_to_fy")
    if "/" in q and ("average" in q or "pp&e" in q or "ppe" in q):
        hit("slash_formula")

    if hits:
        conf = min(0.98, 0.72 + 0.06 * len(hits))
        return "calculation", conf, "matched: " + ", ".join(hits)
    return "direct", 0.88, "no_calculation_heuristics; default_direct"


def parse_router_label(raw: str | None) -> tuple[str | None, str | None]:
    if not raw:
        return None, None

    cleaned = strip_think(raw)
    payload_text = extract_json_object(cleaned) or cleaned
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        label = str(payload.get("label") or payload.get("route") or "").strip().lower()
        reason = payload.get("reason")
        if label in {"direct", "calculation", "domain", "novel"}:
            return label, str(reason) if reason is not None else None

    match = re.search(r"\b(direct|calculation|domain|novel)\b", cleaned.lower())
    return (match.group(1), None) if match else (None, None)


def map_router_label_to_question_style(label: str | None) -> str | None:
    if label in {"direct", "calculation", "domain", "novel"}:
        return label
    return None


def classify_question_style_llm(
    llm: Any,
    question: str,
    *,
    fallback: Callable[[str], tuple[str, float, str]] = classify_question_style_auto,
) -> tuple[str, float, str, str | None]:
    prompt = ROUTER_PROMPT.format(question=question)
    raw = llm.invoke(prompt).content
    label, reason = parse_router_label(raw)
    effective_style = map_router_label_to_question_style(label)
    if effective_style is None:
        fallback_style, fallback_conf, fallback_reason = fallback(question)
        return (
            fallback_style,
            fallback_conf,
            f"llm_parse_failed; fallback={fallback_reason}",
            raw,
        )

    return effective_style, 1.0, f"llm_router={label}; {reason or 'no_reason'}", raw


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
