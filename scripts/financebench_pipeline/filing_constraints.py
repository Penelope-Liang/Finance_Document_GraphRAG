from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from financebench_pipeline.utils import _normalize_text


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
_MONTH_ALIASES = {
    "january": ("january", "jan"),
    "february": ("february", "feb"),
    "march": ("march", "mar"),
    "april": ("april", "apr"),
    "may": ("may",),
    "june": ("june", "jun"),
    "july": ("july", "jul"),
    "august": ("august", "aug"),
    "september": ("september", "sept", "sep"),
    "october": ("october", "oct"),
    "november": ("november", "nov"),
    "december": ("december", "dec"),
}

_MIN_FILING_YEAR = 2010
_MAX_FILING_YEAR = 2026
_YEAR_TOKEN_PATTERN = r"(?:fy\s*['’]?(?:(?:19|20)\d{2}|\d{2})|(?:19|20)\d{2})"
_COMPANY_EXTRACTION_PROMPT = """Extract the company mentioned in the question.

Return JSON only:
{{"company": "<company name or null>"}}

Rules:
- Extract only the company explicitly mentioned in the question.
- If the question uses a ticker-like abbreviation or short company name, keep that exact company form.
  Examples:
  - AMD -> AMD
  - AES -> AES
  - JPM -> JPM
  - MGM -> MGM
  - 3M -> 3M
- Expand only common non-official abbreviations when the intended company is unambiguous.
  Examples:
  - JnJ -> Johnson & Johnson
  - AMEX -> American Express
- If the question mentions a longer legal name, keep the meaningful company name but remove legal suffixes such as Inc., Corp., Corporation, Co., Company, Ltd., LLC, PLC.
  Examples:
  - AES Corporation -> AES
  - Advanced Micro Devices, Inc. -> Advanced Micro Devices
  - JPMorgan Chase & Co. -> JPMorgan Chase
  - MGM Resorts International -> MGM Resorts
- Do not infer a company if none is mentioned.
- Do not include fiscal years, quarters, metrics, or explanations.

Question:
{question}
"""
_LEGAL_COMPANY_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "llc",
    "plc",
    "international",
}


def _normalize_year_token(token: str) -> str | None:
    digits = re.findall(r"\d+", token)
    if not digits:
        return None
    value = digits[-1]
    if len(value) == 2:
        year = 2000 + int(value)
    elif len(value) == 4:
        year = int(value)
    else:
        return None
    if _MIN_FILING_YEAR <= year <= _MAX_FILING_YEAR:
        return str(year)
    return None


def _extract_years(text: str) -> list[str]:
    clean = _normalize_text(text)
    years: set[str] = set()
    range_pattern = re.compile(
        rf"\b({_YEAR_TOKEN_PATTERN})\s*(?:-|‐|‑|‒|–|—|\bto\b)\s*({_YEAR_TOKEN_PATTERN})\b"
    )
    range_spans: list[tuple[int, int]] = []
    for match in range_pattern.finditer(clean):
        start_year = _normalize_year_token(match.group(1))
        end_year = _normalize_year_token(match.group(2))
        if start_year and end_year:
            years.add(str(max(int(start_year), int(end_year))))
            range_spans.append(match.span())

    if range_spans:
        chars = list(clean)
        for start, end in range_spans:
            chars[start:end] = " " * (end - start)
        clean = "".join(chars)

    for match in re.finditer(r"\bfy\s*['’]?((?:19|20)\d{2}|\d{2})\b", clean):
        year = _normalize_year_token(match.group(0))
        if year:
            years.add(year)

    for match in re.finditer(r"\bq[1-4]\s*(?:of\s*)?(?:fy\s*)?((?:19|20)\d{2})\b", clean):
        year = _normalize_year_token(match.group(1))
        if year:
            years.add(year)

    for match in re.finditer(r"\b((?:19|20)\d{2})\b", clean):
        year = _normalize_year_token(match.group(1))
        if year:
            years.add(year)
    return sorted(years)


def _extract_months(text: str) -> list[str]:
    hits: list[str] = []
    for month, aliases in _MONTH_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", text):
                hits.append(month)
                break
    return hits


def _extract_question_constraints(question: str) -> dict[str, Any]:
    clean = _normalize_text(question)

    years = _extract_years(question)
    quarter_match = re.findall(r"\bq([1-4])\b", clean)
    month_hits = _extract_months(clean)
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


_DOC_PREFIX_TO_COMPANY_NAME = {
    "ACTIVISIONBLIZZARD": "Activision Blizzard",
    "ADOBE": "Adobe",
    "AES": "AES",
    "AMAZON": "Amazon",
    "AMCOR": "Amcor",
    "AMD": "AMD",
    "AMERICANEXPRESS": "American Express",
    "AMERICANWATERWORKS": "American Water Works",
    "BESTBUY": "Best Buy",
    "BLOCK": "Block",
    "BOEING": "Boeing",
    "COCACOLA": "Coca-Cola",
    "CORNING": "Corning",
    "COSTCO": "Costco",
    "CVSHEALTH": "CVS Health",
    "GENERALMILLS": "General Mills",
    "KRAFTHEINZ": "Kraft Heinz",
    "LOCKHEEDMARTIN": "Lockheed Martin",
    "MGMRESORTS": "MGM Resorts",
    "MICROSOFT": "Microsoft",
    "NETFLIX": "Netflix",
    "NIKE": "Nike",
    "PAYPAL": "PayPal",
    "PEPSICO": "PepsiCo",
    "PFIZER": "Pfizer",
    "FOOTLOCKER": "Foot Locker",
    "JOHNSON_JOHNSON": "Johnson & Johnson",
    "JPMORGAN": "JPMorgan",
    "ULTABEAUTY": "Ulta Beauty",
    "VERIZON": "Verizon",
    "WALMART": "Walmart",
    "3M": "3M",
}


def _company_name_from_doc_name(doc_name: str | None) -> str | None:
    if not doc_name:
        return None
    stem = Path(str(doc_name)).stem.upper()
    prefix = re.split(r"_(?:19|20)\d{2}", stem, maxsplit=1)[0]
    return _DOC_PREFIX_TO_COMPANY_NAME.get(prefix)
def _company_name_from_question(question: str | None, llm: Any | None = None) -> str | None:
    if not question or llm is None:
        return None
    try:
        raw = llm.invoke(_COMPANY_EXTRACTION_PROMPT.format(question=question)).content.strip()
    except Exception:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    company = payload.get("company")
    if company is None:
        return None
    company_text = str(company).strip()
    if not company_text or company_text.lower() == "null":
        return None
    return company_text


def _company_match_tokens(company: str | None) -> list[str]:
    if not company:
        return []
    normalized = str(company).lower().replace("&", " and ")
    tokens = re.findall(r"[a-z0-9]+", normalized)
    while tokens and tokens[-1] in _LEGAL_COMPANY_SUFFIXES:
        tokens.pop()
    return tokens


def _company_match_keys(company: str | None) -> set[str]:
    tokens = _company_match_tokens(company)
    keys: set[str] = set()
    if not tokens:
        return keys
    for end in range(len(tokens), 0, -1):
        subset = tokens[:end]
        keys.add("".join(subset))
        if len(subset) > 1:
            keys.add("".join(token[0] for token in subset if token))
    return keys


def _company_doc_key_from_company_name(company: str | None) -> str | None:
    if not company:
        return None
    company_keys = _company_match_keys(company)
    if not company_keys:
        return None
    for prefix, name in _DOC_PREFIX_TO_COMPANY_NAME.items():
        registry_keys = _company_match_keys(name) | {_normalize_text(prefix).replace("_", "")}
        if company_keys & registry_keys:
            return prefix.lower()
    for prefix, name in _DOC_PREFIX_TO_COMPANY_NAME.items():
        registry_keys = _company_match_keys(name) | {_normalize_text(prefix).replace("_", "")}
        for company_key in company_keys:
            if len(company_key) >= 3 and any(
                registry_key.startswith(company_key) or company_key.startswith(registry_key)
                for registry_key in registry_keys
                if len(registry_key) >= 3
            ):
                return prefix.lower()
    return None


def _doc_company_prefix(doc_id: str) -> str:
    stem = Path(str(doc_id)).stem.upper()
    return re.split(r"_(?:19|20)\d{2}", stem, maxsplit=1)[0].lower()


def _infer_filing_doc_id(
    *,
    question: str,
    query_rewrite_plan: dict[str, Any],
    available_doc_ids: list[str],
) -> tuple[str | None, str | None]:
    company_doc_key = query_rewrite_plan.get("company_doc_key")
    if not company_doc_key:
        company_doc_key = _company_doc_key_from_company_name(_company_name_from_question(question))
    if not company_doc_key:
        return None, None

    constraints = _extract_question_constraints(question)
    years = list(dict.fromkeys(list(query_rewrite_plan.get("years") or []) + constraints["years"]))
    quarters = constraints["quarters"]
    clean = _normalize_text(question)
    wants_quarterly = bool(quarters) or "10-q" in clean or "10q" in clean or "quarter" in clean
    wants_annual = (
        "10-k" in clean
        or "10k" in clean
        or "annual report" in clean
        or (bool(years) and not wants_quarterly)
    )

    # For multi-year annual questions (ratios, averages, YoY, CAGR) the target
    # filing is the latest year's 10-K, because a 10-K carries prior-year
    # comparative columns (e.g. the FY2017 10-K contains FY2015-FY2017 income
    # statements). Earlier filings can never contain later years' data.
    latest_year = str(max(int(year) for year in years)) if years else None

    scored: list[tuple[float, str]] = []
    for doc_id in available_doc_ids:
        if _doc_company_prefix(doc_id) != str(company_doc_key).lower():
            continue
        stem = Path(doc_id).stem.upper()
        is_annual_doc = "_10K" in stem or "ANNUALREPORT" in stem
        score = 0.0

        if quarters:
            for quarter in quarters:
                for year in years:
                    if f"{year}Q{quarter}" in stem:
                        score += 8.0
                if f"Q{quarter}" in stem:
                    score += 1.0

        for year in years:
            if year in stem:
                score += 2.0

        if wants_quarterly and ("_10Q" in stem or re.search(r"(?:19|20)\d{2}Q[1-4]", stem)):
            score += 1.0
        if wants_annual and is_annual_doc:
            score += 1.0

        # Latest-year tie-breaker: only fires for annual questions with no
        # quarter, and only when the latest year's annual filing actually
        # exists in the candidate list, so missing-filing cases stay ambiguous.
        if (
            wants_annual
            and not quarters
            and latest_year
            and is_annual_doc
            and latest_year in stem
        ):
            score += 3.0

        if not years and not quarters:
            continue
        if score > 0:
            scored.append((score, doc_id))

    if not scored:
        return None, None

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    top_score, top_doc_id = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else -1.0

    # A year+quarter match is high-confidence. A company+year match is also
    # sufficient when it uniquely identifies one filing; many questions mention
    # only fiscal year and company, without an explicit 10-K/10-Q label.
    if quarters and top_score >= 8.0:
        return top_doc_id, "question_company_period"
    if top_score >= 2.0 and top_score > runner_up:
        return top_doc_id, "question_company_period"
    return None, None


def _extract_year_window(question: str) -> list[str]:
    clean = _normalize_text(question)
    years = _extract_years(question)
    if not years:
        return []
    numeric_years = sorted(int(year) for year in years)
    # FinanceBench_RAG uses a small year window.  Include adjacent years because
    # average-balance and YoY formulas often need the prior period.
    needs_prior = any(
        token in clean
        for token in ("average", "year-over-year", "year over year", "yoy", "growth")
    )
    start = (
        min(numeric_years)
        if len(numeric_years) > 1 or not needs_prior
        else max(1900, min(numeric_years) - 1)
    )
    end = max(numeric_years)
    return [str(year) for year in range(start, end + 1)]
