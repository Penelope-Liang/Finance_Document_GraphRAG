from __future__ import annotations

from typing import Any

from financebench_pipeline.filing_constraints import (
    _company_doc_key_from_company_name,
    _company_name_from_doc_name,
    _company_name_from_question,
    _extract_year_window,
)
from financebench_pipeline.utils import _normalize_text


def _rewrite_metrics_query_for_retrieval(
    question: str,
    *,
    question_style: str,
    doc_name: str | None = None,
    llm: Any | None = None,
) -> dict[str, Any]:
    """FinanceBench_RAG-style retrieval rewrite.

    Keep the user-facing question unchanged, but rewrite derived metric wording
    into reported financial statement components for retrieval.
    """
    company_from_question = _company_name_from_question(question, llm=llm)
    company_from_doc = _company_name_from_doc_name(doc_name)
    company = company_from_question or company_from_doc
    years = _extract_year_window(question)
    clean = _normalize_text(question)
    components: list[str] = []
    statements: list[str] = []
    operation = "direct"

    def add_component(value: str) -> None:
        if value not in components:
            components.append(value)

    def add_statement(value: str) -> None:
        if value not in statements:
            statements.append(value)

    if question_style == "calculation":
        operation = "calculation"
        if "fixed asset turnover" in clean:
            operation = "divide"
            add_component("consolidated revenue or net revenues")
            add_component("property and equipment net")
            add_component("property plant and equipment net")
            add_statement("consolidated statement of operations")
            add_statement("consolidated balance sheet")
        if "operating cash flow ratio" in clean:
            operation = "divide"
            add_component("net cash provided by operating activities")
            add_component("cash from operations")
            add_component("total current liabilities")
            add_statement("consolidated statement of cash flows")
            add_statement("consolidated balance sheet")
        if "capex" in clean or "capital expenditure" in clean or "capital expenditures" in clean:
            add_component("capital expenditures")
            add_component("capital expenditure")
            add_statement("consolidated statement of cash flows")
        if "% of revenue" in clean or "percent of revenue" in clean or "percentage of revenue" in clean:
            operation = "divide_or_average"
            add_component("consolidated net revenues")
            add_component("total net revenues")
            add_component("revenue")
            add_statement("consolidated statement of operations")
        if "year-over-year" in clean or "year over year" in clean or "yoy" in clean:
            operation = "change_ratio"
        if "operating income" in clean or "income from operations" in clean:
            add_component("operating income")
            add_component("income from operations")
            add_statement("consolidated statement of operations")
        if "average" in clean:
            operation = "average" if operation == "calculation" else operation
        if "net profit margin" in clean:
            operation = "divide_or_average"
            add_component("net income")
            add_component("net earnings")
            add_component("revenue")
            add_statement("consolidated statement of operations")
        if "working capital ratio" in clean or "current ratio" in clean:
            operation = "divide"
            add_component("total current assets")
            add_component("total current liabilities")
            add_statement("consolidated balance sheet")
        if "return on assets" in clean or "roa" in clean:
            operation = "divide"
            add_component("net income")
            add_component("total assets")
            add_statement("consolidated statement of operations")
            add_statement("consolidated balance sheet")
        if "ebitda" in clean:
            add_component("EBITDA")
            add_component("adjusted EBITDA")
        if "cogs" in clean or "cost of goods sold" in clean or "cost of sales" in clean:
            add_component("cost of goods sold")
            add_component("cost of sales")
            add_statement("consolidated statement of operations")
    else:
        # Direct extraction is already working well; keep its retrieval query close
        # to the original wording and only prepend stable company/year hints.
        operation = "information_extraction"

    if not components:
        # Fall back to the original question if no derived metric was recognized.
        components.append(question)

    if question_style in {"novel", "domain"}:
        operation = "open_ended" if question_style == "novel" else "domain_relevant"
        if "segment" in clean or "business segment" in clean:
            add_component("business segment")
            add_component("segment sales")
            add_component("organic sales change")
            add_statement("business segments")
            add_statement("performance by business segment")
        if "dividend" in clean:
            add_component("dividends")
            add_component("dividend distribution")
            add_component("cash dividends")
        if "free cash" in clean or "cashflow conversion" in clean or "cash flow conversion" in clean:
            add_component("free cash flow")
            add_component("net cash provided by operating activities")
            add_component("capital expenditures")
            add_component("net income")
            add_statement("cash flow")
        if "8k" in clean or "8-k" in clean or "filing" in clean:
            add_component("filing agenda")
            add_component("item")
            add_component("shareholder vote")
        if "cyclical" in clean or "cyclicality" in clean:
            add_component("cyclicality")
            add_component("economic conditions")
            add_statement("risk factors")
            add_statement("business")
        if "card member" in clean or "cardmember" in clean or "retain" in clean or "retention" in clean:
            add_component("card members")
            add_component("retention")
            add_component("customer engagement")
        if question_style == "domain":
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
            if "industry" in clean or "primarily operate" in clean:
                add_component("industry")
                add_component("business overview")
                add_statement("business")

    statement_text = "; ".join(statements)
    component_text = "; ".join(components)
    year_text = ", ".join(years)
    parts = []
    if company:
        parts.append(company)
    if year_text:
        parts.append(f"fiscal years {year_text}")
    parts.append(component_text)
    if statement_text:
        parts.append(statement_text)

    rewritten_query = "; ".join(part for part in parts if part).strip()
    return {
        "rewritten_query": rewritten_query or question,
        "company": company,
        "company_source": "question" if company_from_question else ("doc_name" if company_from_doc else None),
        "company_doc_key": _company_doc_key_from_company_name(company),
        "years": years,
        "operation": operation,
        "components": components,
        "statements": statements,
    }


_METRICS_TABLE_TARGETS = {
    "revenue": {
        "aliases": (
            "total consolidated net revenues",
            "total consolidated revenues",
            "total net revenues",
            "total net revenue",
            "total revenues",
            "total revenue",
            "net revenues",
            "net revenue",
            "net sales",
            "revenue",
            "revenues",
        ),
        "section_hints": (
            "statement of operations",
            "statements of operations",
            "statement of income",
            "statements of income",
            "statement of earnings",
            "income statement",
            "segment",
        ),
    },
    "ppne": {
        "aliases": (
            "property and equipment, net",
            "property, plant and equipment, net",
            "property plant and equipment net",
            "property and equipment",
            "property, plant and equipment",
            "net property, plant and equipment",
            "net pp&e",
            "pp&e",
        ),
        "section_hints": (
            "balance sheet",
            "balance sheets",
            "financial position",
        ),
    },
    "operating_cash_flow": {
        "aliases": (
            "net cash provided by operating activities",
            "net cash provided by operations",
            "cash provided by operating activities",
            "cash flow from operating activities",
            "cash flows from operating activities",
            "cash from operations",
            "net cash from operating activities",
        ),
        "section_hints": (
            "cash flow",
            "cash flows",
            "liquidity and capital resources",
        ),
    },
    "current_liabilities": {
        "aliases": (
            "total current liabilities",
            "current liabilities",
        ),
        "section_hints": (
            "balance sheet",
            "balance sheets",
            "financial position",
        ),
    },
    "current_assets": {
        "aliases": (
            "total current assets",
            "current assets",
        ),
        "section_hints": (
            "balance sheet",
            "balance sheets",
            "financial position",
        ),
    },
    "total_assets": {
        "aliases": (
            "total assets",
        ),
        "section_hints": (
            "balance sheet",
            "balance sheets",
            "financial position",
        ),
    },
    "capex": {
        "aliases": (
            "capital expenditures",
            "capital expenditure",
            "capital spending",
            "purchases of property, plant and equipment",
            "purchases of property and equipment",
            "additions to property and equipment",
        ),
        "section_hints": (
            "cash flow",
            "cash flows",
            "liquidity and capital resources",
        ),
    },
    "operating_income": {
        "aliases": (
            "operating income",
            "income from operations",
            "operating profit",
        ),
        "section_hints": (
            "statement of operations",
            "statements of operations",
            "statement of income",
            "statements of income",
            "income statement",
        ),
    },
    "net_income": {
        "aliases": (
            "net income attributable to shareholders",
            "net income attributable to common shareholders",
            "net income attributable to common stockholders",
            "net income",
            "net earnings",
        ),
        "section_hints": (
            "statement of operations",
            "statements of operations",
            "statement of income",
            "statements of income",
            "income statement",
        ),
    },
    "cogs": {
        "aliases": (
            "cost of goods sold",
            "cost of sales",
            "cost of revenue",
            "cost of revenues",
            "cost of products sold",
        ),
        "section_hints": (
            "statement of operations",
            "statements of operations",
            "statement of income",
            "statements of income",
            "income statement",
        ),
    },
    "inventory": {
        "aliases": (
            "inventories",
            "inventory",
        ),
        "section_hints": (
            "balance sheet",
            "balance sheets",
            "financial position",
        ),
    },
    "accounts_payable": {
        "aliases": (
            "accounts payable",
            "accounts payable and accrued expenses",
        ),
        "section_hints": (
            "balance sheet",
            "balance sheets",
            "financial position",
        ),
    },
    "accounts_receivable": {
        "aliases": (
            "accounts receivable",
            "accounts receivable, net",
            "trade accounts receivable",
        ),
        "section_hints": (
            "balance sheet",
            "balance sheets",
            "financial position",
        ),
    },
    "dividends_paid": {
        "aliases": (
            "cash dividends paid",
            "dividends paid",
            "dividends",
        ),
        "section_hints": (
            "cash flow",
            "cash flows",
        ),
    },
    "depreciation_amortization": {
        "aliases": (
            "depreciation and amortization",
            "depreciation & amortization",
            "depreciation",
        ),
        "section_hints": (
            "cash flow",
            "cash flows",
        ),
    },
}


def _metrics_component_targets(question: str, plan: dict[str, Any] | None) -> list[str]:
    clean = _normalize_text(question)
    targets: list[str] = []

    def add(name: str) -> None:
        if name not in targets:
            targets.append(name)

    if "fixed asset turnover" in clean:
        add("revenue")
        add("ppne")
    if "operating cash flow ratio" in clean:
        add("operating_cash_flow")
        add("current_liabilities")
    if "capex" in clean or "capital expenditure" in clean or "capital expenditures" in clean:
        add("capex")
    if "% of revenue" in clean or "percent of revenue" in clean or "percentage of revenue" in clean:
        add("revenue")
    if "revenue growth" in clean or "change in revenue" in clean or "revenue cagr" in clean:
        add("revenue")
    if "operating income" in clean or "income from operations" in clean:
        add("operating_income")
    if "working capital ratio" in clean or "current ratio" in clean:
        add("current_assets")
        add("current_liabilities")
    if "net working capital" in clean:
        add("current_assets")
        add("current_liabilities")
    if "return on assets" in clean or "roa" in clean:
        add("net_income")
        add("total_assets")
    if "net profit margin" in clean:
        add("net_income")
        add("revenue")
    if "inventory turnover" in clean:
        add("cogs")
        add("inventory")
    if "dpo" in clean or "days payable outstanding" in clean:
        add("accounts_payable")
        add("cogs")
        add("inventory")
    if "ccc" in clean or "cash conversion cycle" in clean:
        add("inventory")
        add("accounts_receivable")
        add("accounts_payable")
        add("cogs")
        add("revenue")
    if "dividend payout" in clean or "retention ratio" in clean:
        add("dividends_paid")
        add("net_income")
    if "ebitda" in clean:
        add("operating_income")
        add("depreciation_amortization")
    if "free cash flow" in clean or "fcf" in clean:
        add("operating_cash_flow")
        add("capex")

    # Fall back to query-rewriter components if no explicit formula pattern fired.
    if not targets and plan:
        component_text = _normalize_text(" ".join(str(c) for c in plan.get("components") or []))
        for name, spec in _METRICS_TABLE_TARGETS.items():
            if any(alias in component_text for alias in spec["aliases"]):
                add(name)
    return targets
