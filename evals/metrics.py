"""Deterministic eval metrics (no LLM). Each function scores one case from what the graph actually did."""

import json
import re

from app.graph.nodes.guardrail import unsupported_figures
from app.tools.lead_tools import normalize_indian_mobile

SEARCH_TOOLS = {"search_brochures", "search_service_docs"}


def _norm(text: str) -> str:
    """casefold, drop thousands separators, unify spaces: '₹12,92,125' and '₹1,292,125' both -> '₹1292125'."""
    text = (text or "").replace(" ", " ").replace(" ", " ").casefold()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return re.sub(r"\s+", " ", text)


def _contains(haystack: str, needle: str) -> bool:
    """needle in haystack, not as part of a longer word or number ('250' doesn't match '2500')."""
    n = re.escape(_norm(needle).strip())
    left = r"(?<![\w.])" if re.match(r"\w", needle.strip()[:1] or " ") else ""
    right = r"(?![\w])" if re.match(r"\w", needle.strip()[-1:] or " ") else ""
    return re.search(left + n + right, _norm(haystack)) is not None


# ---- routing, slots, tools, flow -------------------------------------------------

def routing(case: dict, intents: list[str], agents: list[str]) -> dict:
    """Right intents (as a set) and the right agents (as a set: the non-lead agents run in parallel)."""
    options = case.get("expected_intents_any") or [case["expected_intents"]]
    intents_ok = any(set(opt) == set(intents) for opt in options)
    agents_ok = case.get("expected_agents") is None or set(agents) == set(case["expected_agents"])
    return {"pass": intents_ok and agents_ok, "intents_ok": intents_ok, "agents_ok": agents_ok,
            "actual_intents": intents, "actual_agents": agents}


def _slot_equal(key: str, expected, actual) -> bool:
    if expected is None:  # "must not be set"
        return actual in (None, "")
    if key == "phone":
        return normalize_indian_mobile(str(expected)) == normalize_indian_mobile(str(actual or ""))
    return _norm(str(expected)).strip() == _norm(str(actual or "")).strip()


def slots(case: dict, actual: dict) -> dict | None:
    expected = case.get("expected_slots")
    if not expected:
        return None
    results = {k: _slot_equal(k, v, (actual or {}).get(k)) for k, v in expected.items()}
    return {"correct": sum(results.values()), "total": len(results),
            "wrong": {k: {"expected": expected[k], "actual": (actual or {}).get(k)} for k, ok in results.items() if not ok}}


def tools_called(agent_outputs: dict) -> list[str]:
    return [r["tool"] for out in (agent_outputs or {}).values() for r in out.get("tool_results", [])]


def tool_use(case: dict, agent_outputs: dict) -> dict | None:
    """Every expected tool (or one of its 'a|b' alternatives) was requested. [] means: no tool should run."""
    if "expected_tools" not in case:
        return None
    called = set(tools_called(agent_outputs))
    expected = case["expected_tools"]
    if not expected:
        return {"pass": not called, "missing": [], "called": sorted(called)}
    missing = [t for t in expected if not set(t.split("|")) & called]
    return {"pass": not missing, "missing": missing, "called": sorted(called)}


def flow(case: dict, nodes_run: list[str], paused: bool) -> dict:
    """Clarify and Confirm/Cancel pause happened exactly when expected (default: not expected)."""
    clarified = "clarify" in nodes_run
    want_clarify, want_pause = case.get("expect_clarify", False), case.get("expect_pause", False)
    return {"pass": clarified == want_clarify and paused == want_pause,
            "clarify": {"expected": want_clarify, "actual": clarified},
            "pause": {"expected": want_pause, "actual": paused}}


# ---- retrieval -------------------------------------------------------------------

def retrieved_passages(agent_outputs: dict) -> list[dict]:
    passages = []
    for out in (agent_outputs or {}).values():
        for r in out.get("tool_results", []):
            if r["tool"] not in SEARCH_TOOLS or not str(r["content"]).startswith("{"):
                continue
            try:
                passages += json.loads(r["content"]).get("results", [])
            except json.JSONDecodeError:
                continue
    return passages


def retrieval_hit(case: dict, agent_outputs: dict) -> dict | None:
    """hit@k: an expected source file + section is among the passages the search tools returned."""
    expected = case.get("expected_sources")
    if not expected:
        return None
    passages = retrieved_passages(agent_outputs)
    hit = any(p.get("source") == e["source"] and _norm(e["section"]) in _norm(p.get("section", ""))
              for p in passages for e in expected)
    return {"hit": hit, "retrieved": [f"{p.get('source')} | {p.get('section')}" for p in passages]}


# ---- correctness and refusals ----------------------------------------------------

def key_facts(case: dict, reply: str) -> dict | None:
    """Each key fact (any of its 'a|b' alternatives) appears in the reply. Numbers are compared without commas."""
    facts = case.get("key_facts")
    if not facts:
        return None
    missing = [f for f in facts if not any(_contains(reply, alt) for alt in f.split("|"))]
    return {"pass": not missing, "found": len(facts) - len(missing), "total": len(facts), "missing": missing}


def invented_figures(reply: str, agent_outputs: dict) -> list[str]:
    """₹ / km figures in the reply that no tool returned (same check as the guardrail)."""
    return unsupported_figures(reply, agent_outputs)
