from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from financebench_pipeline.utils import _normalize_text


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
_MONTH_TO_NUMBER = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
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
_TARGET_FILING_YEAR_PROMPT = """You are extracting year constraints from a finance question.

Return JSON only:
{{"mentioned_years": ["<year>", "..."], "target_filing_year": "<year or null>", "reason": "<brief reason>"}}

Rules:
- The provided regex_years are the only candidate years. Do not invent other years.
- Use regex_years as mentioned_years unless a normalized equivalent is needed.
- target_filing_year is the year that should be used to restrict retrieval to a filing or report.
- Use a year as target_filing_year when it refers to a reporting period, filing period, fiscal-year data, or historical financial period.
- If the question contains one suitable reporting year, use it.
- If the question contains a historical multi-year range or period comparison, use the latest suitable reporting year.
- Do not use a year as target_filing_year if it is only a forecast year, guidance year, expected future period, future payment year, event date, transaction date, AGM date, or income-impact/event year.
- If no suitable filing/reporting year is present, return null.

Examples:
- Question: What is the FY2017-FY2019 3 year average capital expenditure ratio?
  regex_years: ["2019"]
  Output: {{"mentioned_years": ["2019"], "target_filing_year": "2019", "reason": "Historical multi-year range; use the latest reporting year."}}

- Question: Was there any change between Q2 of FY2024 and FY2023?
  regex_years: ["2024"]
  Output: {{"mentioned_years": ["2024"], "target_filing_year": "2024", "reason": "Period comparison; use the latest reporting year."}}

- Question: What production changes is the company forecasting for FY2023?
  regex_years: ["2023"]
  Output: {{"mentioned_years": ["2023"], "target_filing_year": null, "reason": "FY2023 is only a forecast period."}}

- Question: As of FY2021, how much did the company expect to pay in 2024?
  regex_years: ["2021", "2024"]
  Output: {{"mentioned_years": ["2021", "2024"], "target_filing_year": "2021", "reason": "FY2021 is the as-of reporting year; 2024 is a future payment year."}}

- Question: Were there events that substantially increased net income in 2019?
  regex_years: ["2019"]
  Output: {{"mentioned_years": ["2019"], "target_filing_year": null, "reason": "2019 is an event or income-impact year, not an explicit filing year."}}

Question:
{question}

regex_years:
{regex_years}
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

    for match in re.finditer(r"\bfy\s*['’]?((?:19|20)\d{2}|\d{2})\s*q[1-4]\b", clean):
        year = _normalize_year_token(match.group(1))
        if year:
            years.add(year)

    for match in re.finditer(r"\b((?:19|20)\d{2})\s*q[1-4]\b", clean):
        year = _normalize_year_token(match.group(1))
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
    """Extract month-level temporal cues only; calendar days are ignored."""
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


def _parse_json_object(raw: str) -> dict[str, Any] | None:
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
    return payload if isinstance(payload, dict) else None


def _company_name_from_question(question: str | None, llm: Any | None = None) -> str | None:
    if not question or llm is None:
        return None
    try:
        raw = llm.invoke(_COMPANY_EXTRACTION_PROMPT.format(question=question)).content.strip()
    except Exception:
        return None
    payload = _parse_json_object(raw)
    if not payload:
        return None
    company = payload.get("company")
    if company is None:
        return None
    company_text = str(company).strip()
    if not company_text or company_text.lower() == "null":
        return None
    return company_text


def _target_filing_year_from_question(
    question: str,
    *,
    years: list[str] | None = None,
    llm: Any | None = None,
) -> str | None:
    regex_years = list(dict.fromkeys(years or _extract_years(question)))
    if not regex_years or llm is None:
        return None
    try:
        raw = llm.invoke(
            _TARGET_FILING_YEAR_PROMPT.format(
                question=question,
                regex_years=json.dumps(regex_years, ensure_ascii=False),
            )
        ).content.strip()
    except Exception:
        return None
    payload = _parse_json_object(raw)
    if not payload:
        return None
    target = payload.get("target_filing_year")
    if target is None:
        return None
    target_year = _normalize_year_token(str(target))
    if target_year and target_year in set(regex_years):
        return target_year
    return None


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


def _doc_matches_year(doc_id: str, year: str | None) -> bool:
    return not year or str(year) in Path(str(doc_id)).stem


def _doc_matches_quarter(doc_id: str, quarters: list[str]) -> bool:
    if not quarters:
        return True
    stem = Path(str(doc_id)).stem.upper()
    return any(f"Q{quarter}" in stem for quarter in quarters)


def _doc_matches_month(doc_id: str, months: list[str]) -> bool:
    if not months:
        return True
    stem = Path(str(doc_id)).stem
    return any(
        month in _MONTH_TO_NUMBER and re.search(rf"[-_]{_MONTH_TO_NUMBER[month]}[-_]\d{{2}}\b", stem)
        for month in months
    )


def _candidate_filing_doc_ids(
    *,
    question: str,
    query_rewrite_plan: dict[str, Any],
    available_doc_ids: list[str],
) -> tuple[list[str], str | None, str | None]:
    company_doc_key = query_rewrite_plan.get("company_doc_key")

    constraints = _extract_question_constraints(question)
    target_year = query_rewrite_plan.get("target_filing_year")
    quarters = constraints["quarters"]
    months = constraints["months"]

    # Apply whatever was extracted jointly; absent constraints simply pass through.
    # Company is no longer mandatory.  Month alone is too weak to trigger a hard,
    # cross-company doc filter, so it does not on its own narrow the corpus.
    if not (company_doc_key or target_year or quarters):
        return [], company_doc_key, None

    base_candidates = [
        doc_id
        for doc_id in available_doc_ids
        if (not company_doc_key or _doc_company_prefix(doc_id) == str(company_doc_key).lower())
        and _doc_matches_year(doc_id, target_year)
        and _doc_matches_quarter(doc_id, quarters)
    ]
    candidates = [doc_id for doc_id in base_candidates if _doc_matches_month(doc_id, months)]
    if months and not candidates:
        candidates = base_candidates
    if not candidates:
        return [], company_doc_key, None

    source_parts = []
    if company_doc_key:
        source_parts.append("question_company")
    if target_year:
        source_parts.append("target_year")
    if quarters:
        source_parts.append("quarter")
    if months and candidates != base_candidates:
        source_parts.append("month")
    return sorted(candidates), company_doc_key, "_".join(source_parts) or None


def _extract_year_window(question: str) -> list[str]:
    clean = _normalize_text(question)
    years = set(_extract_years(question))
    range_pattern = re.compile(
        rf"\b({_YEAR_TOKEN_PATTERN})\s*(?:-|‐|‑|‒|–|—|\bto\b)\s*({_YEAR_TOKEN_PATTERN})\b"
    )
    for match in range_pattern.finditer(clean):
        start_year = _normalize_year_token(match.group(1))
        end_year = _normalize_year_token(match.group(2))
        if start_year and end_year:
            years.add(str(min(int(start_year), int(end_year))))
            years.add(str(max(int(start_year), int(end_year))))
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
