from __future__ import annotations

import re
from typing import Any

STATEMENT_TYPES = {
    "balance_sheet",
    "cash_flow",
    "income_statement",
    "equity_statement",
    "notes",
    "md&a",
    "other",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _snippet(text: str, max_chars: int = 1600) -> str:
    clean = _normalize(text)
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars]


def classify_statement_type(
    *,
    section_name: str = "",
    title_text: str = "",
    text: str = "",
    metadata: dict[str, Any] | None = None,
) -> str:
    metadata = metadata or {}

    section = _normalize(section_name)
    title = _normalize(title_text or str(metadata.get("title_text") or ""))
    body = _snippet(text)
    combined = " | ".join(part for part in (title, section, body) if part)

    if not combined:
        return "other"

    # Strong heading matches first.
    if any(
        phrase in combined
        for phrase in (
            "consolidated balance sheet",
            "consolidated balance sheets",
            "balance sheet",
            "balance sheets",
            "statement of financial position",
            "statements of financial position",
        )
    ):
        return "balance_sheet"

    if any(
        phrase in combined
        for phrase in (
            "statement of cash flows",
            "statements of cash flows",
            "consolidated statement of cash flows",
            "consolidated statements of cash flows",
            "cash flows from operating activities",
            "net cash provided by operating activities",
            "net cash used in investing activities",
            "net cash provided by financing activities",
        )
    ):
        return "cash_flow"

    if any(
        phrase in combined
        for phrase in (
            "statement of income",
            "statements of income",
            "statement of operations",
            "statements of operations",
            "consolidated statement of income",
            "consolidated statements of income",
            "consolidated statement of operations",
            "consolidated statements of operations",
        )
    ):
        return "income_statement"

    if any(
        phrase in combined
        for phrase in (
            "statement of stockholders' equity",
            "statements of stockholders' equity",
            "statement of shareholders' equity",
            "statements of shareholders' equity",
            "statement of equity",
            "statements of equity",
            "statement of changes in equity",
            "statements of changes in equity",
        )
    ):
        return "equity_statement"

    if any(
        phrase in combined
        for phrase in (
            "management's discussion and analysis",
            "management discussion and analysis",
            "md&a",
            "item 7.",
            "item 7 ",
            "results of operations",
            "liquidity and capital resources",
            "critical accounting policies",
        )
    ):
        return "md&a"

    if (
        title.startswith("note ")
        or section.startswith("note ")
        or "notes to consolidated financial statements" in combined
        or "notes to the consolidated financial statements" in combined
        or "see note " in combined
    ):
        return "notes"

    # Table-content heuristics when the heading is weak or absent.
    if (
        "current assets" in body
        and "total current assets" in body
        and (
            "total current liabilities" in body
            or ("liabilities" in body and "equity" in body)
            or "total assets" in body
        )
    ):
        return "balance_sheet"

    if (
        "cash flows from operating activities" in body
        or "net cash provided by operating activities" in body
        or (
            "operating activities" in body
            and "investing activities" in body
            and "financing activities" in body
        )
    ):
        return "cash_flow"

    if (
        "net sales" in body
        and ("operating income" in body or "income before income taxes" in body)
        and "net income" in body
    ):
        return "income_statement"

    if (
        "retained earnings" in body
        and ("accumulated other comprehensive" in body or "treasury stock" in body)
    ):
        return "equity_statement"

    return "other"
