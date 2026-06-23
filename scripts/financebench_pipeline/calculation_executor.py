from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any


_CALCULATION_PLAN_PROMPT = """You are a financial calculation planner.

Given a FinanceBench numerical reasoning question, the model's current answer,
and retrieved evidence, create a JSON calculation plan that computes the final
answer requested by the question.

Rules:
1. Output valid JSON only. No markdown.
2. "required_inputs" is the single source of truth for the calculation. Every
   numeric value used by the formula must appear there as a number (not a string).
   Give each required input a unique "variable": a short identifier made of
   letters, digits, and underscores (e.g. revenue_2022) that the formula will
   reference.
3. List every required input in "required_inputs", including inputs that are
   missing. Each required input must include name, variable, value, unit, period,
   and source_text. Use value=null when the input is missing.
4. For any input that is missing or ambiguous, set its value=null and set
   formula=null, and do not invent substitute values.
5. If source_text contains a clear numeric value for an input, put that number
   in value. Do not set value=null for an input whose numeric value is plainly
   present in source_text.
6. Do not use derived values such as average inventory, average accounts payable,
   change in inventory, or change in receivables as required_inputs unless that
   derived value is explicitly reported in the evidence. Instead, extract the
   underlying period values separately and compute the average or change inside
   the formula.
   If the question defines "change in X between FYA and FYB", required_inputs
   must be X_FYA and X_FYB, and the formula must compute X_FYB - X_FYA.
7. For example, if a calculation needs average working capital, do not create one
   required input named average_working_capital unless that exact value is
   reported. Instead, create current_assets_2022, current_liabilities_2022,
   current_assets_2021, and current_liabilities_2021, then compute the average
   in the formula from those period-specific values.
8. Put the arithmetic expression in "formula". It may only reference the
   "variable" names you defined in "required_inputs" and use +, -, *, /, **,
   parentheses, and round/abs/min/max.
9. The formula must compute the final answer requested by the question, not just
   restate one input value.
10. Use the rounding requested by the question. If not stated, use 2 decimals.
11. If required values are missing or ambiguous, leave formula=null and do not
   invent values.
12. Do not use unrelated values even if they are numerically available in the
   evidence. The input name, period, statement context, and unit must match the
   question.
13. Do not return a list of source values as the final answer. Return only the
   computed final metric requested by the question.
14. If all required input values are present (no value is null), formula must be
   non-null: write the arithmetic expression that computes the answer.

CRITICAL unit rules:
- unit="percent" ONLY when the question explicitly asks for "percent", "percentage",
  "% margin", "% of revenue", or similar. Multiply by 100 in the formula.
- unit="plain" for: ratio, turnover, ROA, ROE, current ratio, quick ratio, working
  capital ratio, operating cash flow ratio, DPO, DSO, DIO, CCC, EPS, P/E, times
  interest earned, debt-to-equity, asset turnover, fixed asset turnover, inventory
  turnover, receivables turnover. These are dimensionless ratios, NOT percents.
- If the question says "margin" or "as % of revenue/net sales", the denominator
  MUST be revenue or net sales, and unit="percent". Formula: numerator / revenue * 100.
- If monetary values have different scales (thousands vs millions vs billions),
  normalize them to the same scale before computing.

When deriving the formula, think in terms of simple Python assignments, but do
NOT output Python code. Output the JSON schema below only. The examples below
are fictional and only show how to translate financial questions into
executable arithmetic formulas:

Example 1:
Question: What is Horizon Parts' FY2024 free cash flow in USD millions? Free
cash flow is defined as cash from operations minus capital expenditures.
Evidence: Net cash provided by operating activities was $1,240 million in
FY2024. Capital expenditures were $315 million in FY2024.
Formula derivation:
cash_from_operations_2024 = 1240
capital_expenditures_2024 = 315
ans = cash_from_operations_2024 - capital_expenditures_2024
JSON plan:
{{
  "required_inputs": [
    {{"name": "FY2024 cash from operations", "variable": "cash_from_operations_2024", "value": 1240, "unit": "million", "period": "FY2024", "source_text": "Net cash provided by operating activities was $1,240 million in FY2024."}},
    {{"name": "FY2024 capital expenditures", "variable": "capital_expenditures_2024", "value": 315, "unit": "million", "period": "FY2024", "source_text": "Capital expenditures were $315 million in FY2024."}}
  ],
  "formula": "cash_from_operations_2024 - capital_expenditures_2024",
  "round_to": 2,
  "unit": "usd_millions",
  "reason": "Free cash flow equals cash from operations minus capital expenditures, with both inputs in millions."
}}

Example 2:
Question: What is Northstar Retail's FY2023 inventory turnover ratio? Inventory
turnover is defined as FY2023 cost of sales divided by average inventory between
FY2022 and FY2023. Round to two decimal places.
Evidence: Cost of sales was $8,400 million in FY2023. Inventories were $1,180
million in FY2023 and $1,220 million in FY2022.
Formula derivation:
cost_of_sales_2023 = 8400
inventory_2023 = 1180
inventory_2022 = 1220
average_inventory = (inventory_2023 + inventory_2022) / 2
ans = cost_of_sales_2023 / average_inventory
JSON plan:
{{
  "required_inputs": [
    {{"name": "FY2023 cost of sales", "variable": "cost_of_sales_2023", "value": 8400, "unit": "million", "period": "FY2023", "source_text": "Cost of sales was $8,400 million in FY2023."}},
    {{"name": "FY2023 inventory", "variable": "inventory_2023", "value": 1180, "unit": "million", "period": "FY2023", "source_text": "Inventories were $1,180 million in FY2023."}},
    {{"name": "FY2022 inventory", "variable": "inventory_2022", "value": 1220, "unit": "million", "period": "FY2022", "source_text": "Inventories were $1,220 million in FY2022."}}
  ],
  "formula": "cost_of_sales_2023 / ((inventory_2023 + inventory_2022) / 2)",
  "round_to": 2,
  "unit": "plain",
  "reason": "Inventory turnover is a plain ratio, not a percent, and average inventory is computed inside the formula."
}}

Example 3:
Question: What is Meridian Software's FY2024 EBITDA margin? Define EBITDA as
operating income plus depreciation and amortization. Answer in percents and
round to one decimal place.
Evidence: Revenue was $5,600 million in FY2024. Operating income was $840
million. Depreciation and amortization was $210 million.
Formula derivation:
revenue_2024 = 5600
operating_income_2024 = 840
depreciation_and_amortization_2024 = 210
ebitda_2024 = operating_income_2024 + depreciation_and_amortization_2024
ans = ebitda_2024 / revenue_2024 * 100
JSON plan:
{{
  "required_inputs": [
    {{"name": "FY2024 revenue", "variable": "revenue_2024", "value": 5600, "unit": "million", "period": "FY2024", "source_text": "Revenue was $5,600 million in FY2024."}},
    {{"name": "FY2024 operating income", "variable": "operating_income_2024", "value": 840, "unit": "million", "period": "FY2024", "source_text": "Operating income was $840 million."}},
    {{"name": "FY2024 depreciation and amortization", "variable": "depreciation_and_amortization_2024", "value": 210, "unit": "million", "period": "FY2024", "source_text": "Depreciation and amortization was $210 million."}}
  ],
  "formula": "(operating_income_2024 + depreciation_and_amortization_2024) / revenue_2024 * 100",
  "round_to": 1,
  "unit": "percent",
  "reason": "EBITDA margin is EBITDA divided by revenue, multiplied by 100 because the question asks for percents."
}}

Return this schema:
{{
  "required_inputs": [
    {{
      "name": "human-readable input name",
      "variable": "variable_name_or_null",
      "value": number or null,
      "unit": "million|billion|thousand|percent|plain|other|null",
      "period": "fiscal year or period label or null",
      "source_text": "short evidence quote or null"
    }}
  ],
  "formula": "arithmetic expression or null",
  "round_to": integer,
  "unit": "plain|percent|usd_millions|usd_billions|usd_thousands|other",
  "reason": "brief explanation of formula, unit normalization, and rounding"
}}

Question:
{question}

Current answer:
{current_answer}

Retrieved evidence:
{evidence}
"""


_ALLOWED_FUNCTIONS = {
    "round": round,
    "abs": abs,
    "min": min,
    "max": max,
}

_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Call,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)


@dataclass
class FormulaResult:
    success: bool
    answer: float | int | None
    formula: str | None
    inputs: dict[str, float]
    round_to: int
    unit: str
    required_inputs: list[dict[str, Any]]
    missing_inputs: list[str]
    reason: str | None = None
    error: str | None = None


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _strip_json_line_comments(text: str) -> str:
    """Remove // comments outside JSON strings."""
    output: list[str] = []
    index = 0
    in_string = False
    escape = False
    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < len(text) and text[index + 1] == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue
        output.append(char)
        index += 1
    return "".join(output)


def _infer_round_to(question: str, fallback: int = 2) -> int:
    clean = question.lower()
    word_to_num = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
    }
    match = re.search(r"round(?:ed)? to (\d+) decimal", clean)
    if match:
        return int(match.group(1))
    match = re.search(r"round(?:ed)? to (zero|one|two|three|four) decimal", clean)
    if match:
        return word_to_num[match.group(1)]
    return fallback


_PLAIN_RATIO_KEYWORDS = frozenset({
    "ratio",
    "turnover",
    "roa",
    "roe",
    "roi",
    "eps",
    "p/e",
    "pe ratio",
    "dpo",
    "dso",
    "dio",
    "ccc",
    "times interest",
    "debt-to-equity",
    "debt to equity",
    "current ratio",
    "quick ratio",
    "working capital ratio",
    "operating cash flow ratio",
    "asset turnover",
    "fixed asset turnover",
    "inventory turnover",
    "receivables turnover",
})


def _explicit_percent_request(question: str) -> bool:
    clean = question.lower()
    return (
        "%" in clean
        or "percent" in clean
        or "percents" in clean
        or "percentage" in clean
    )


def _is_margin_question(question: str) -> bool:
    clean = question.lower()
    return (
        "margin" in clean
        or "% of revenue" in clean
        or "% of net sales" in clean
        or "as a percent of revenue" in clean
        or "as a percent of net sales" in clean
        or "as a percentage of revenue" in clean
        or "as a percentage of net sales" in clean
    )


def _is_plain_ratio_question(question: str) -> bool:
    clean = question.lower()
    if _explicit_percent_request(question) or _is_margin_question(question):
        return False
    for keyword in _PLAIN_RATIO_KEYWORDS:
        if keyword in clean:
            return True
    return False


def _infer_unit(question: str, fallback: str = "plain") -> str:
    clean = question.lower()
    if _is_plain_ratio_question(question):
        return "plain"
    if _is_margin_question(question) or _explicit_percent_request(question):
        return "percent"
    if "usd billions" in clean or "in billions" in clean:
        return "usd_billions"
    if "usd millions" in clean or "in millions" in clean:
        return "usd_millions"
    if "usd thousands" in clean or "in thousands" in clean:
        return "usd_thousands"
    return fallback


def _normalize_required_inputs(raw_required_inputs: Any) -> list[dict[str, Any]]:
    if raw_required_inputs is None:
        return []
    if not isinstance(raw_required_inputs, list):
        raise ValueError("required_inputs must be a list")
    normalized: list[dict[str, Any]] = []
    for item in raw_required_inputs:
        if not isinstance(item, dict):
            raise ValueError("each required input must be an object")
        normalized.append(
            {
                "name": item.get("name"),
                "variable": item.get("variable"),
                "value": item.get("value"),
                "unit": item.get("unit"),
                "period": item.get("period"),
                "source_text": item.get("source_text"),
            }
        )
    return normalized


def _validate_formula_ast(node: ast.AST, *, allowed_names: set[str]) -> None:
    for child in ast.walk(node):
        if not isinstance(child, _ALLOWED_NODE_TYPES):
            raise ValueError(f"disallowed syntax: {type(child).__name__}")
        if isinstance(child, ast.Name):
            if child.id not in allowed_names and child.id not in _ALLOWED_FUNCTIONS:
                raise ValueError(f"unknown variable/function: {child.id}")
        if isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Name) or child.func.id not in _ALLOWED_FUNCTIONS:
                raise ValueError("only round/abs/min/max calls are allowed")
            if child.keywords:
                raise ValueError("keyword arguments are not allowed")
        if isinstance(child, ast.Constant):
            if isinstance(child.value, bool) or not isinstance(child.value, (int, float)):
                raise ValueError("only numeric constants are allowed")


def _derive_inputs_from_required(required_inputs: list[dict[str, Any]]) -> dict[str, float]:
    """Rebuild the executable variable map from per-input values the model already
    extracted, for when it forgot to fill the separate ``inputs`` object."""
    derived: dict[str, float] = {}
    for item in required_inputs:
        variable = item.get("variable")
        value = item.get("value")
        if not isinstance(variable, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", variable):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        derived[variable] = float(value)
    return derived


def _missing_required_input_names(required_inputs: list[dict[str, Any]]) -> list[str]:
    """Genuinely-missing inputs are those whose value is null/non-numeric."""
    names: list[str] = []
    for item in required_inputs:
        value = item.get("value")
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            names.append(str(item.get("name") or item.get("variable") or "unnamed input"))
    return names


_NUMERIC_TOKEN = re.compile(r"\(?\$?\s*\d[\d,]*(?:\.\d+)?\s*\)?")


def _significant_digits(text: str) -> str:
    """Reduce a numeric string to its significant digits: drop everything that is
    not a digit (``$``, commas, decimal point, sign, parentheses) then strip
    leading and trailing zeros so scale differences (e.g. millions vs billions)
    do not cause a mismatch."""
    return re.sub(r"\D", "", text).lstrip("0").rstrip("0")


def _value_significant_digits(value: float) -> str:
    return _significant_digits(f"{abs(float(value)):.6f}")


def _value_supported_by_source(value: float, source_text: Any) -> bool:
    """Conservative anti-hallucination check: a value is supported when its
    significant digits appear in (or are a rounding-prefix of) some number quoted
    in source_text. When source_text has no number to compare against, we cannot
    prove fabrication and do not penalize the value."""
    if not isinstance(source_text, str) or not source_text.strip():
        return True
    tokens = _NUMERIC_TOKEN.findall(source_text)
    if not tokens:
        return True
    target = _value_significant_digits(value)
    if not target:  # value is zero; nothing meaningful to verify
        return True
    for token in tokens:
        candidate = _significant_digits(token)
        if not candidate:
            continue
        if target == candidate:
            return True
        shorter, longer = (target, candidate) if len(target) <= len(candidate) else (candidate, target)
        if len(shorter) >= 2 and longer.startswith(shorter):
            return True
    return False


def _unsupported_required_input_names(required_inputs: list[dict[str, Any]]) -> list[str]:
    """Inputs that carry a numeric value which does not appear in their own
    source_text quote (likely hallucinated)."""
    names: list[str] = []
    for item in required_inputs:
        value = item.get("value")
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not _value_supported_by_source(value, item.get("source_text")):
            names.append(str(item.get("name") or item.get("variable") or "unnamed input"))
    return names


def evaluate_formula_plan(plan: dict[str, Any], *, question: str) -> FormulaResult:
    required_inputs = _normalize_required_inputs(plan.get("required_inputs"))
    reason = str(plan.get("reason") or "").strip() or None
    round_to_default = _infer_round_to(question)
    unit_default = _infer_unit(question)

    # required_inputs is the single source of truth. We never trust the model's
    # self-reported "success", "missing_inputs", or "inputs" map; everything below
    # is derived from required_inputs. Missing data = required inputs with a
    # null/non-numeric value.
    missing_inputs = sorted(_missing_required_input_names(required_inputs))
    # Anti-hallucination: a value must actually appear in its own source_text quote.
    unsupported_inputs = sorted(_unsupported_required_input_names(required_inputs))
    if missing_inputs or unsupported_inputs:
        error_parts = []
        if missing_inputs:
            error_parts.append("missing required inputs: " + ", ".join(missing_inputs))
        if unsupported_inputs:
            error_parts.append("inputs not found in source_text: " + ", ".join(unsupported_inputs))
        return FormulaResult(
            success=False,
            answer=None,
            formula=None,
            inputs={},
            round_to=round_to_default,
            unit=unit_default,
            required_inputs=required_inputs,
            missing_inputs=missing_inputs,
            reason=reason,
            error="; ".join(error_parts),
        )

    # The executable variable map is always rebuilt from required_inputs, so the
    # values used for computation are exactly the ones that were extracted and
    # source-verified above.
    inputs = _derive_inputs_from_required(required_inputs)
    if not inputs:
        return FormulaResult(
            success=False,
            answer=None,
            formula=None,
            inputs={},
            round_to=round_to_default,
            unit=unit_default,
            required_inputs=required_inputs,
            missing_inputs=missing_inputs,
            reason=reason,
            error="no usable numeric inputs",
        )

    formula = plan.get("formula")
    if not isinstance(formula, str) or not formula.strip():
        # Inputs are present but the model produced no executable formula. Signal
        # this distinctly so the caller can attempt a focused formula repair.
        return FormulaResult(
            success=False,
            answer=None,
            formula=None,
            inputs=inputs,
            round_to=round_to_default,
            unit=unit_default,
            required_inputs=required_inputs,
            missing_inputs=missing_inputs,
            reason=reason,
            error="missing formula",
        )
    formula = formula.strip()

    round_to_raw = plan.get("round_to")
    round_to = (
        int(round_to_raw)
        if isinstance(round_to_raw, (int, float)) and not isinstance(round_to_raw, bool)
        else round_to_default
    )
    round_to = max(0, min(round_to, 6))
    llm_unit = str(plan.get("unit") or "").strip().lower()
    if _explicit_percent_request(question) or _is_margin_question(question):
        unit = "percent"
    elif _is_plain_ratio_question(question):
        unit = "plain"
    elif llm_unit in {"plain", "percent", "usd_millions", "usd_billions", "usd_thousands", "other"}:
        unit = llm_unit
    else:
        unit = unit_default

    parsed = ast.parse(formula, mode="eval")
    _validate_formula_ast(parsed, allowed_names=set(inputs))
    safe_globals = {"__builtins__": {}, **_ALLOWED_FUNCTIONS}
    value = eval(compile(parsed, "<financebench_formula>", "eval"), safe_globals, dict(inputs))  # noqa: S307

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("formula result must be numeric")

    return FormulaResult(
        success=True,
        answer=float(value),
        formula=formula,
        inputs=inputs,
        round_to=round_to,
        unit=unit,
        required_inputs=required_inputs,
        missing_inputs=missing_inputs,
        reason=reason,
        error=None,
    )


def format_calculation_answer(result: FormulaResult) -> str | None:
    if not result.success or result.answer is None:
        return None
    value = round(float(result.answer), result.round_to)
    if result.round_to == 0:
        text = str(int(value))
    else:
        text = f"{value:.{result.round_to}f}".rstrip("0").rstrip(".")
    if result.unit == "percent":
        return f"{text}%"
    return text


def generate_calculation_prompt(question: str, current_answer: Any, evidence: str) -> str:
    return _CALCULATION_PLAN_PROMPT.format(
        question=question,
        current_answer=current_answer,
        evidence=evidence,
    )


_FORMULA_REPAIR_PROMPT = """You are a financial calculation planner. The numeric inputs have ALREADY been extracted and verified. Do NOT re-extract, change, add, or remove any value.

Your only job: write the arithmetic formula that combines the given variables into the final answer the question asks for.

Rules:
- "formula" may reference ONLY the provided variable names and use + - * / ** ( ) and round/abs/min/max.
- The formula must compute the final answer requested by the question.
- For a percentage / margin / "% of revenue" question, multiply by 100 and set unit="percent".
- For a plain ratio (e.g. current ratio, x times), set unit="plain".
- Otherwise keep the monetary unit of the inputs.
- Use the rounding the question asks for; if unstated, use 2 decimals.

Return JSON ONLY:
{{
  "formula": "arithmetic expression using the variable names",
  "round_to": integer,
  "unit": "plain|percent|usd_millions|usd_billions|usd_thousands|other"
}}

Question:
{question}

Available variables (name = value):
{variables}
"""


def _repair_missing_formula(
    llm: Any,
    question: str,
    plan: dict[str, Any],
    result: FormulaResult,
) -> tuple[FormulaResult | None, str | None]:
    """Focused second call: inputs are present but no executable formula was
    produced. Ask the model for the formula only, reusing the extracted values."""
    variables = "\n".join(f"{name} = {value}" for name, value in result.inputs.items())
    repair_prompt = _FORMULA_REPAIR_PROMPT.format(question=question, variables=variables)
    response = llm.invoke(repair_prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    repair_json = _extract_json_object(raw or "")
    if not repair_json:
        return None, None
    repair_plan = json.loads(_strip_json_line_comments(repair_json))
    formula = repair_plan.get("formula")
    if not isinstance(formula, str) or not formula.strip():
        return None, None
    # evaluate_formula_plan re-derives inputs/missing_inputs from required_inputs,
    # so we only need to supply the freshly repaired formula and presentation hints.
    merged_plan = {
        "required_inputs": plan.get("required_inputs"),
        "formula": formula,
        "round_to": repair_plan.get("round_to", plan.get("round_to")),
        "unit": repair_plan.get("unit", plan.get("unit")),
        "reason": plan.get("reason"),
    }
    repaired = evaluate_formula_plan(merged_plan, question=question)
    return repaired, format_calculation_answer(repaired)


def run_calculation_pipeline(
    llm: Any,
    question: str,
    current_answer: Any,
    evidence: str,
) -> dict[str, Any]:
    prompt = generate_calculation_prompt(question, current_answer, evidence)
    response = llm.invoke(prompt)
    raw_response = response.content if hasattr(response, "content") else str(response)
    json_text = _extract_json_object(raw_response or "")
    if not json_text:
        return {
            "calculation_success": False,
            "calculation_answer": None,
            "calculation_result": None,
            "calculation_plan": None,
            "calculation_error": "missing JSON calculation plan",
            "calculation_raw_response": raw_response,
        }

    try:
        plan = json.loads(_strip_json_line_comments(json_text))
        result = evaluate_formula_plan(plan, question=question)
        answer = format_calculation_answer(result)
    except Exception as exc:
        return {
            "calculation_success": False,
            "calculation_answer": None,
            "calculation_result": None,
            "calculation_plan": json_text,
            "calculation_error": str(exc),
            "calculation_raw_response": raw_response,
        }

    # Inputs were extracted but the model produced no executable formula: try one
    # focused repair call rather than discarding an otherwise-complete extraction.
    if not result.success and result.error == "missing formula" and result.inputs:
        try:
            repaired, repaired_answer = _repair_missing_formula(llm, question, plan, result)
            if repaired is not None and repaired.success and repaired_answer is not None:
                result = repaired
                answer = repaired_answer
        except Exception:
            pass  # keep the original "missing formula" failure

    return {
        "calculation_success": bool(result.success and answer is not None),
        "calculation_answer": answer,
        "calculation_result": result.answer,
        "calculation_plan": {
            "required_inputs": result.required_inputs,
            "missing_inputs": result.missing_inputs,
            "inputs": result.inputs,
            "formula": result.formula,
            "round_to": result.round_to,
            "unit": result.unit,
            "reason": result.reason,
        },
        "calculation_error": result.error,
        "calculation_raw_response": raw_response,
    }
