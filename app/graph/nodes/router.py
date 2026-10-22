"""Routing functions for conditional edges. Plain Python: they read State, never call Gemini, never write State."""

from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.types import Send

from app.agents import CATALOG, Catalog
from app.config import get_settings
from app.graph.nodes.agent_factory import tool_rounds
from app.graph.nodes.supervisor import missing_slots

SUMMARY_TRIGGER = 20  # summarize when messages holds more than this many


def needs_person(state: dict) -> bool:
    """Hand off to a human: the supervisor flagged frustration / safety / a dispute, or the bot has failed or
    got stuck ESCALATE_AFTER_STRIKES turns in a row."""
    return bool(state.get("escalation")) or (state.get("strikes") or 0) >= get_settings().escalate_after_strikes


def route_after_supervisor(state: dict, catalog: Catalog = CATALOG):
    """A person is needed -> escalate (checked first: no more questions or automation). Missing required slot ->
    clarify (no agent runs). Otherwise start every parallel agent at once (one Send each, each with its own copy of
    what it needs); only action agents -> the first one; none -> composer."""
    from app.graph.nodes.worker import worker_input  # here to avoid an import cycle

    if needs_person(state):
        return "escalate"
    intents = state.get("intents") or []
    if missing_slots(intents, state.get("slots") or {}, catalog):
        return "clarify"
    parallel = [a for a in catalog.agents_for(intents) if a in catalog.workers]
    if parallel:
        return [Send(a, worker_input(state, a)) for a in parallel]
    return catalog.next_action_agent(intents) or "composer"


def after_agents(state: dict, catalog: Catalog = CATALOG) -> str:
    """Runs once, after every parallel agent has finished: the first asked action agent, or the composer."""
    return catalog.next_action_agent(state.get("intents") or []) or "composer"


def after_action(state: dict, catalog: Catalog = CATALOG) -> str:
    """After save_output: the next asked action agent (one at a time, each may pause), or the composer."""
    return catalog.next_action_agent(state.get("intents") or [], after=state.get("active_action")) or "composer"


def route_back(state: dict) -> str:
    """After the tools node: back to the action agent that asked for the tools."""
    return state["active_action"]


def agent_router(state: dict) -> str:
    """Action agent: tool calls within the round limit -> approval; final answer or limit -> save_output."""
    scratch = state.get("action_messages") or []
    last = scratch[-1] if scratch else None
    if isinstance(last, AIMessage) and last.tool_calls:
        if tool_rounds(scratch) > get_settings().max_tool_rounds:
            return "save_output"
        return "approval"
    return "save_output"


def after_guardrail(state: dict) -> str:
    """Feedback set -> composer (fix once); more than 20 messages -> summarize; otherwise -> END."""
    if state.get("guardrail_feedback"):
        return "composer"
    if len(state.get("messages") or []) > SUMMARY_TRIGGER:
        return "summarize"
    return END
