"""Run one turn of the graph safely. Used by the CLI and (Phase 5) the API.

If anything fails mid-turn (e.g. Gemini unavailable after all retries), the full error goes to the logs,
the thread's State is put back exactly as it was before the turn, and the customer gets a friendly message.
So a failed turn never leaves State half-updated and the next message works normally.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import structlog
from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

from app.graph.build_graph import AGENT_NAMES
from app.graph.nodes.guardrail import SAFE_REPLY
from app.memory.file_store import RESTORE_AS_NODE
from app.observability import turn_context

log = structlog.get_logger()

FRIENDLY_ERROR = "Sorry, I'm having trouble right now. Please try again in a moment."

FIELD_DEFAULTS = {"intents": [], "slots": {}, "active_action": "", "guardrail_feedback": None, "guardrail_retries": 0,
                  "summary": "", "escalation": None, "strikes": 0, "escalated": False}


@dataclass
class TurnResult:
    status: Literal["reply", "paused", "error"]
    reply: str | None = None      # the checked reply (status "reply") or FRIENDLY_ERROR (status "error")
    pending: dict | None = None   # interrupt payload when paused for approval


def rollback(graph, config: dict, before_values: dict) -> None:
    """Make the thread's latest checkpoint equal to the pre-turn State, with nothing pending."""
    restore = {k: before_values.get(k, default) for k, default in FIELD_DEFAULTS.items()}
    for key in ("messages", "action_messages"):
        restore[key] = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *before_values.get(key, [])]
    restore["agent_outputs"] = None  # the merge reducer would keep this turn's entries; None resets to {}
    graph.update_state(config, restore, as_node=RESTORE_AS_NODE)
    if before_values.get("agent_outputs"):
        graph.update_state(config, {"agent_outputs": before_values["agent_outputs"]}, as_node=RESTORE_AS_NODE)


def run_turn(graph, config: dict, payload, on_node: Callable[[str], None] | None = None) -> TurnResult:
    """payload: {"messages": [HumanMessage]} for a new question, or Command(resume=...) after a pause.
    on_node is called with each node name as it finishes (for progress messages).

    Every log line of the turn carries turn_id/thread_id, and the turn ends with one "turn" summary line
    (see app/observability.py). config may carry LangSmith callbacks from tracing_callbacks()."""
    thread_id = config["configurable"]["thread_id"]
    with turn_context(thread_id) as (turn_id, stats):
        config = {**config, "run_name": "skoda_turn",
                  "metadata": {**(config.get("metadata") or {}), "turn_id": turn_id}}
        before = graph.get_state(config).values
        pending, nodes, node_seconds = None, [], {}
        start = last = time.perf_counter()
        try:
            for chunk in graph.stream(payload, config, stream_mode="updates"):
                now = time.perf_counter()
                for node, _update in chunk.items():
                    if node == "__interrupt__":
                        pending = _update[0].value
                        continue
                    nodes.append(node)
                    node_seconds[node] = round(node_seconds.get(node, 0.0) + (now - last), 2)
                    if on_node:
                        on_node(node)
                last = now
        except Exception as exc:
            log.exception("turn failed; state rolled back")
            try:
                rollback(graph, config, before)
                # the turn is undone, but the failure still counts toward a hand-off to a person
                graph.update_state(config, {"strikes": (before.get("strikes") or 0) + 1}, as_node=RESTORE_AS_NODE)
            except Exception:
                log.exception("rollback failed")
            log_turn_summary("error", payload, nodes, node_seconds, before, stats, start,
                             error=type(exc).__name__)
            return TurnResult("error", reply=FRIENDLY_ERROR)

        values = graph.get_state(config).values
        if pending is not None:
            log_turn_summary("paused", payload, nodes, node_seconds, values, stats, start,
                             awaiting=[a["tool"] for a in pending.get("actions", [])])
            return TurnResult("paused", pending=pending)
        messages = values.get("messages", [])
        reply = next((m.text for m in reversed(messages) if isinstance(m, AIMessage)), None)
        log_turn_summary("reply", payload, nodes, node_seconds, values, stats, start,
                         guardrail=_guardrail_result(nodes, values, reply))
        return TurnResult("reply", reply=reply)


def _guardrail_result(nodes: list[str], values: dict, reply: str | None) -> str | None:
    if "guardrail" not in nodes:
        return None  # e.g. the turn ended at clarify
    if reply == SAFE_REPLY:
        return "safe_reply"
    return "retried" if values.get("guardrail_retries") else "ok"


def log_turn_summary(status: str, payload, nodes: list[str], node_seconds: dict, values: dict, stats,
                     start: float, **extra) -> None:
    """One line per turn. Names and counts only: no message text, no slot values."""
    agents = list(dict.fromkeys(n for n in nodes if n in AGENT_NAMES))
    outputs = values.get("agent_outputs") or {}
    tools = [r["tool"] for a in agents for r in (outputs.get(a) or {}).get("tool_results", [])]
    log.info("turn", status=status, resumed=isinstance(payload, Command),
             total_s=round(time.perf_counter() - start, 2), nodes=nodes, node_seconds=node_seconds,
             intents=values.get("intents") or [], agents=agents, tools=tools,
             # parallel agents overlap, so their own timers (not node_seconds) show how long each took
             agent_seconds={a: (outputs.get(a) or {}).get("seconds") for a in agents if a in outputs},
             agent_status={a: (outputs.get(a) or {}).get("status") for a in agents if a in outputs},
             slots_filled=sorted(k for k, v in (values.get("slots") or {}).items()
                                 if v not in (None, "", [], {}) and k not in ("cars", "active_cars", "car_intents")),
             cars=len((values.get("slots") or {}).get("active_cars") or []),
             clarify="clarify" in nodes, escalated="escalate" in nodes, escalation=values.get("escalation"),
             strikes=values.get("strikes") or 0, **stats.as_dict(), **extra)
