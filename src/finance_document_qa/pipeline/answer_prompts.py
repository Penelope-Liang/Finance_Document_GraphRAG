from __future__ import annotations

from typing import Any

from finance_document_qa.pipeline.filing_constraints import (
    _extract_question_constraints,
)


def _append_constraint_lines(lines: list[str], constraints: dict[str, Any]) -> str:
    if constraints["section_hints"]:
        lines.append(
            f"- Extracted section/table hints: {', '.join(constraints['section_hints'])}"
        )
    if constraints["years"]:
        lines.append(f"- Extracted years: {', '.join(constraints['years'])}")
    if constraints["quarters"]:
        lines.append(
            f"- Extracted quarters: {', '.join('Q' + q for q in constraints['quarters'])}"
        )
    if constraints["months"]:
        lines.append(
            f"- Extracted month-level temporal cues: {', '.join(constraints['months'])}"
        )
    if constraints["period_hints"]:
        lines.append(
            f"- Extracted period hints: {', '.join(constraints['period_hints'])}"
        )
    return "\n".join(lines)


def build_information_extraction_prompt(question: str) -> str:
    constraints = _extract_question_constraints(question)
    lines = [
        "Information extraction prompt:",
        "- Prefer evidence whose section/table wording matches the question's report or section hint.",
        "- If the question mentions a specific report, section, note, table name, or similar label, prioritize exact matches first and then closely related phrasing.",
        "- If the exact wording does not appear, pay attention to semantically similar section/table names rather than ignoring the constraint.",
        "- Respect time constraints strictly: fiscal year, quarter, period-ended phrasing, and explicit dates must match the chosen evidence.",
        "- If candidate evidence conflicts on statement type or time period, prefer the exact match and avoid mixing periods.",
        "- Structured Table evidence is valid financial statement evidence.",
        "- A Table evidence item is sufficient when its row header, column header, section name, and period match the question.",
        "- If a matching Table evidence item contains text or numeric_value, use it directly to answer the question.",
        "- Do not return answer=null merely because the full original table is not shown; the Table evidence item is extracted from that table.",
        "- Use the scale field for unit conversion, such as converting million to billion when the question asks for USD billions.",
        "Example:",
        "Question: What is the year-end net PP&E in USD billions?",
        "Evidence: Table evidence row='Property, plant and equipment, net'; col='December 31, 20XX'; text='8,000'; numeric_value=8000.0; scale='million'",
        'Answer: {"answer": "$8.0 billion", "evidence": "Used Property, plant and equipment, net for the requested year; converted 8,000 million to 8.0 billion.", "missing_information": null}',
    ]
    return _append_constraint_lines(lines, constraints)


def build_numerical_reasoning_prompt(question: str) -> str:
    constraints = _extract_question_constraints(question)
    lines = [
        "Numerical reasoning prompt:",
        "- This is a numerical reasoning question, not a direct single-value extraction question.",
        "- The 'answer' field must contain the final computed result only, using the unit and rounding requested by the question.",
        "- Do not put raw input values, a list of source numbers, or intermediate values in the 'answer' field.",
        "- Put raw input values, formula, unit conversions, and intermediate arithmetic in the 'evidence' field.",
        "- First identify every required input value before answering.",
        "- Use only evidence that matches the question's requested statement, section/table hint, and time period.",
        "- For ratios, margins, averages, YoY changes, turnover, DPO, CCC, or CAGR, gather all required numerator/denominator or multi-period inputs before computing.",
        "- If the required inputs are present, you must perform the calculation and return the computed final value; do not stop after listing the inputs.",
        "- Do not answer from a single value if the metric requires multiple inputs.",
        "- If multiple candidate values are available, prefer the exact statement and period match.",
        "- If any required input is missing or period alignment is ambiguous, return insufficient information instead of guessing.",
        "- Keep the computation simple and explicit in evidence: identify inputs, apply the formula, show the arithmetic, then provide only the final result in answer.",
        "- Return JSON with keys 'answer' and 'evidence'. In evidence, briefly name the rows/periods used in the calculation.",
    ]
    return _append_constraint_lines(lines, constraints)


def build_domain_relevant_prompt(question: str) -> str:
    constraints = _extract_question_constraints(question)
    lines = [
        "Domain-relevant FinanceBench prompt:",
        "- Use retrieved evidence as the only source of truth; do not use outside knowledge.",
        "- Return JSON only with keys 'answer', 'evidence', and 'missing_information'.",
        "- The answer field should be concise: 1-3 sentences. Include the final conclusion and essential explanation if needed.",
        "- For yes/no finance judgment questions, start the answer with a clear 'Yes' or 'No', then explain briefly.",
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
    return _append_constraint_lines(lines, constraints)


def build_novel_generated_prompt(question: str) -> str:
    constraints = _extract_question_constraints(question)
    lines = [
        "Novel-generated FinanceBench prompt:",
        "- Use the retrieved evidence as the only source of truth; do not use outside knowledge.",
        "- Answer the business question directly and concisely. If the answer requires an explanation, include the key reason.",
        "- If the question asks for a comparison, trend, increase/decrease, stability, or best/worst segment, compare the relevant periods/categories explicitly.",
        "- If the question requires numbers, identify the values used and show the calculation briefly in the evidence field.",
        "- Prefer evidence matching the same company, filing type, date/period, section, and business segment when those constraints are present in the question.",
        "- Company, period, filing type, and section constraints are optional: use them only if they appear in the question or are clearly implied by retrieved evidence.",
        "- If candidate evidence conflicts on company, period, filing, or section, prefer the exact match and avoid mixing periods or filings.",
        "- If the evidence is insufficient, return answer=null and explain the missing information rather than guessing.",
        "- Return JSON with keys 'answer', 'evidence', 'missing_information'.",
    ]
    return _append_constraint_lines(lines, constraints)


def build_answer_prompt(question: str, question_style: str) -> str:
    if question_style == "calculation":
        return build_numerical_reasoning_prompt(question)
    if question_style == "domain":
        return build_domain_relevant_prompt(question)
    if question_style == "novel":
        return build_novel_generated_prompt(question)
    return build_information_extraction_prompt(question)
