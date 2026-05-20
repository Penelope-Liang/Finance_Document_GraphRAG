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
2. Put all numeric inputs in "inputs" as numbers, not strings.
3. Put the arithmetic expression in "formula". It may only reference variables
   defined in "inputs" and use +, -, *, /, **, parentheses, and round/abs/min/max.
4. The formula must compute the final answer requested by the question, not just
   restate one input value.
5. If the question asks for percents, set the formula to return the percent
   value, e.g. 65.4, not 0.654.
6. Use the rounding requested by the question. If not stated, use 2 decimals.
7. If required values are missing or ambiguous, return success=false with a
   short error and do not invent values.

Return this schema:
{{
  "success": boolean,
  "inputs": {{"variable_name": number}},
  "formula": "arithmetic expression or null",
  "round_to": integer,
  "unit": "plain|percent|usd_millions|usd_billions|usd_thousands|other",
  "error": "string or null"
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


def _infer_unit(question: str, fallback: str = "plain") -> str:
    clean = question.lower()
    if "percent" in clean or "percentage" in clean or "%" in clean:
        return "percent"
    if "usd billions" in clean or "in billions" in clean:
        return "usd_billions"
    if "usd millions" in clean or "in millions" in clean:
        return "usd_millions"
    if "usd thousands" in clean or "in thousands" in clean:
        return "usd_thousands"
    return fallback


def _normalize_inputs(raw_inputs: Any) -> dict[str, float]:
    if not isinstance(raw_inputs, dict):
        raise ValueError("inputs must be an object")
    inputs: dict[str, float] = {}
    for key, value in raw_inputs.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", str(key)):
            raise ValueError(f"invalid input variable name: {key}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"input {key} must be numeric")
        inputs[str(key)] = float(value)
    return inputs


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


def evaluate_formula_plan(plan: dict[str, Any], *, question: str) -> FormulaResult:
    if not bool(plan.get("success")):
        return FormulaResult(
            success=False,
            answer=None,
            formula=None,
            inputs={},
            round_to=_infer_round_to(question),
            unit=_infer_unit(question),
            error=str(plan.get("error") or "calculation plan marked unsuccessful"),
        )

    inputs = _normalize_inputs(plan.get("inputs"))
    formula = plan.get("formula")
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("formula must be a non-empty string")
    formula = formula.strip()

    round_to_raw = plan.get("round_to")
    round_to = int(round_to_raw) if isinstance(round_to_raw, (int, float)) else _infer_round_to(question)
    round_to = max(0, min(round_to, 6))
    unit = str(plan.get("unit") or _infer_unit(question)).strip() or "plain"

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
        plan = json.loads(json_text)
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

    return {
        "calculation_success": bool(result.success and answer is not None),
        "calculation_answer": answer,
        "calculation_result": result.answer,
        "calculation_plan": {
            "inputs": result.inputs,
            "formula": result.formula,
            "round_to": result.round_to,
            "unit": result.unit,
        },
        "calculation_error": result.error,
        "calculation_raw_response": raw_response,
    }
