"""Guardrail (plain Python, no Gemini): every ₹ amount and km figure in the composer's reply must appear in
the tool results stored in agent_outputs. First failure -> composer fixes it once; second -> safe reply."""

import re
from dataclasses import dataclass

import structlog
from langchain_core.messages import AIMessage

log = structlog.get_logger()

NUM = r"(\d[\d,]*(?:\.\d+)?)"
MONEY_RE = re.compile(rf"(?:₹|(?<![a-z])rs\.?|(?<![a-z])inr)\s*{NUM}\s*(lakhs?|lacs?|crores?|cr|l|k)?\b", re.I)
WORD_MONEY_RE = re.compile(rf"{NUM}\s*(lakhs?|lacs?|crores?)\b", re.I)
KM_RE = re.compile(rf"{NUM}\s*km\b", re.I)  # "kmpl" doesn't match: no word boundary after "km"
ANY_NUMBER_RE = re.compile(NUM)

MULTIPLIERS = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "l": 1e5,
               "crore": 1e7, "crores": 1e7, "cr": 1e7, "k": 1e3}
ROUNDED_TOLERANCE = 0.005  # 0.5% for "₹12.92 lakh"-style rounded figures

SAFE_REPLY = ("Sorry, I couldn't double-check the figures for this answer, so I'd rather not give you numbers "
              "that might be wrong. Would you like a Skoda team member to call you back? Just share your "
              "name and mobile number.")


@dataclass(frozen=True)
class Figure:
    text: str       # as written in the reply
    value: float
    kind: str       # "money" or "km"
    rounded: bool   # written with lakh/crore/k, so allow small rounding


def _to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


def extract_figures(reply: str) -> list[Figure]:
    figures, taken = [], []

    def overlaps(span):
        return any(span[0] < t[1] and t[0] < span[1] for t in taken)

    for rx, kind in ((MONEY_RE, "money"), (WORD_MONEY_RE, "money"), (KM_RE, "km")):
        for m in rx.finditer(reply):
            if overlaps(m.span()):
                continue
            taken.append(m.span())
            unit = (m.group(2) or "").lower() if m.lastindex and m.lastindex >= 2 else ""
            value = _to_float(m.group(1)) * MULTIPLIERS.get(unit, 1)
            figures.append(Figure(m.group(0).strip(), value, kind, bool(unit)))
    return figures


def allowed_numbers(agent_outputs: dict) -> list[float]:
    numbers = []
    for output in (agent_outputs or {}).values():
        for result in output.get("tool_results", []):
            for raw in ANY_NUMBER_RE.findall(str(result.get("content", ""))):
                try:
                    numbers.append(_to_float(raw))
                except ValueError:
                    continue
    return numbers


def is_supported(fig: Figure, allowed: list[float]) -> bool:
    for a in allowed:
        if fig.kind == "money":
            if fig.rounded:
                if a and abs(fig.value - a) / a <= ROUNDED_TOLERANCE:
                    return True
            elif abs(fig.value - a) <= 1:
                return True
        elif abs(fig.value - a) <= max(0.05, ROUNDED_TOLERANCE * a):
            return True
    return False


def unsupported_figures(reply: str, agent_outputs: dict) -> list[str]:
    allowed = allowed_numbers(agent_outputs)
    return [f.text for f in extract_figures(reply) if not is_supported(f, allowed)]


def guardrail(state: dict) -> dict:
    messages = state.get("messages") or []
    reply = messages[-1] if messages else None
    if not isinstance(reply, AIMessage):
        return {"guardrail_feedback": None}
    bad = unsupported_figures(reply.text, state.get("agent_outputs") or {})
    if not bad:
        return {"guardrail_feedback": None, "strikes": 0}  # a good reply: the hand-off counter starts again
    retries = state.get("guardrail_retries") or 0
    log.warning("guardrail mismatch", figures=bad, retry=retries)
    if retries == 0:
        return {"guardrail_feedback": f"these figures are not in the tool results: {', '.join(bad)}",
                "guardrail_retries": 1}
    return {"guardrail_feedback": None, "messages": [AIMessage(content=SAFE_REPLY, id=reply.id)],
            "strikes": (state.get("strikes") or 0) + 1}  # couldn't give a checked answer: counts toward a hand-off
