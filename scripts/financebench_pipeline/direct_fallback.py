from __future__ import annotations

import re
from typing import Any

# Keywords that indicate the question asks for an amount/magnitude, where a
# parenthesised/negative table value (e.g. cash outflow "(1,577)") should be
# reported as a positive figure. Anything else keeps its sign so that genuine
# negatives (e.g. net loss) are not flipped.
_AMOUNT_KEYWORDS = (
    "capex",
    "capital expenditure",
    "capital expenditures",
    "purchases of property",
    "amount",
    "paid",
    "spent",
    "dividends paid",
    "repayments",
)

# Convert any cell scale to a common "millions" base.
_SCALE_TO_MILLIONS = {
    "thousand": 0.001,
    "thousands": 0.001,
    "million": 1.0,
    "millions": 1.0,
    "billion": 1000.0,
    "billions": 1000.0,
}

_UNIT_WORD = {"thousand": "thousand", "million": "million", "billion": "billion"}


def _requested_unit(question: str) -> str | None:
    """Return the unit the question explicitly asks for, if any."""
    text = str(question or "").lower()
    if re.search(r"\b(?:usd|us\$|dollars?|\$)\s*billions?\b|\bbillions?\s+of\s+dollars?\b", text):
        return "billion"
    if re.search(r"\b(?:usd|us\$|dollars?|\$)\s*millions?\b|\bmillions?\s+of\s+dollars?\b", text):
        return "million"
    if re.search(r"\b(?:usd|us\$|dollars?|\$)\s*thousands?\b|\bthousands?\s+of\s+dollars?\b", text):
        return "thousand"
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _format_money(value: float, unit: str | None) -> str:
    num = f"{value:,.4f}".rstrip("0").rstrip(".")
    return f"${num} {unit}" if unit else f"${num}"


def _direct_table_cell_fallback(
    question: str,
    retrieved_contexts: list[dict[str, Any]],
    *,
    max_evidence: int = 5,
) -> tuple[str | None, dict[str, Any] | None]:
    """Deterministically build a numeric answer from the top table-cell evidence.

    Only consider the highest-ranked evidence whose source is a table cell and
    whose constraint bucket is ``exact_cell`` or ``metric_cell`` and that exposes
    a numeric value. Returns ``(answer, info)`` or ``(None, None)`` when no
    suitable cell is found, so the caller can leave the LLM result untouched.
    """
    text = str(question or "").lower()
    wants_amount = any(keyword in text for keyword in _AMOUNT_KEYWORDS)
    requested_unit = _requested_unit(question)

    for ctx in (retrieved_contexts or [])[:max_evidence]:
        metadata = ctx.get("metadata") if isinstance(ctx, dict) else None
        metadata = metadata or {}
        if metadata.get("source") not in {"table", "metrics_table"}:
            continue
        if metadata.get("bucket") not in {"exact_cell", "metric_cell"}:
            continue
        numeric_value = _coerce_float(metadata.get("numeric_value"))
        if numeric_value is None:
            continue

        scale = str(metadata.get("scale") or "").lower().strip()
        value_in_millions = numeric_value * _SCALE_TO_MILLIONS.get(scale, 1.0)

        # Prefer the unit the question asks for; otherwise keep the cell's own
        # unit so we do not invent a scale the question never mentioned.
        unit = requested_unit or (scale if scale in _SCALE_TO_MILLIONS else None)
        if unit in _SCALE_TO_MILLIONS:
            value_in_unit = value_in_millions / _SCALE_TO_MILLIONS[unit]
            unit_word = _UNIT_WORD.get(unit.rstrip("s"))
        else:
            value_in_unit = numeric_value
            unit_word = None

        if wants_amount:
            value_in_unit = abs(value_in_unit)

        formatted_answer = _format_money(value_in_unit, unit_word)
        info = {
            "applied": True,
            "row_header": metadata.get("row_header"),
            "col_header": metadata.get("col_header"),
            "numeric_value": numeric_value,
            "scale": scale or None,
            "requested_unit": requested_unit,
            "sign_normalized": wants_amount,
            "formatted_answer": formatted_answer,
            "context_rank": ctx.get("rank"),
            "bucket": metadata.get("bucket"),
            "source": metadata.get("source"),
        }
        return formatted_answer, info

    return None, None
