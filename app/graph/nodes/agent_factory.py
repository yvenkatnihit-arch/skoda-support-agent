"""Agent Gemini calls. Each agent calls Gemini with its own system prompt and tools (added only to the request,
never stored in State). The parallel agents use this inside their worker subgraph (app/graph/nodes/worker.py);
the action agents (lead, support...) run one at a time in the main graph with action_messages as scratchpad."""

import json
from datetime import date

import structlog
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from app.agents.base import AgentSpec
from app.graph.car_slots import agent_view
from app.graph.nodes.context import clip, recent_dialogue, summary_text
from app.prompts.agents_common import AGENT_HEADER, OTHER_FINDINGS_BLOCK

log = structlog.get_logger()

RECENT_MESSAGES = 4
AI_MESSAGE_MAX_CHARS = 800
OTHER_ANSWER_MAX_CHARS = 800      # lead's view of each other agent's answer
OTHER_TOOL_RESULT_MAX_CHARS = 1500
CANCELLED = "NOT DONE: cancelled. The customer asked something else instead of confirming."


def tool_rounds(scratch: list) -> int:
    """Number of tool-call rounds so far = AIMessages with tool calls in the scratchpad."""
    return sum(1 for m in scratch if isinstance(m, AIMessage) and m.tool_calls)


def patch_unanswered_tool_calls(scratch: list) -> list:
    """Add a 'NOT DONE: cancelled' ToolMessage for any tool call without a result, so the request stays valid.
    Used for the Gemini request only; State is not changed."""
    answered = {m.tool_call_id for m in scratch if isinstance(m, ToolMessage)}
    patched = []
    for m in scratch:
        patched.append(m)
        if isinstance(m, AIMessage) and m.tool_calls:
            for call in m.tool_calls:
                if call["id"] not in answered:
                    patched.append(ToolMessage(content=CANCELLED, tool_call_id=call["id"], name=call["name"]))
    return patched


def call_agent(spec: AgentSpec, llm_with_tools, *, slots: dict, summary: str, dialogue: list, scratch: list,
               extra: str = ""):
    """One agent step: system prompt + recent dialogue + this agent's scratchpad -> Gemini -> AIMessage."""
    filled = {k: v for k, v in (slots or {}).items() if v not in (None, "")}
    system = SystemMessage(
        AGENT_HEADER.format(role=spec.role, today=date.today().isoformat(),
                            slots=json.dumps(filled, ensure_ascii=False), summary=summary)
        + "\n" + spec.prompt + extra)
    reply = llm_with_tools.invoke([system, *dialogue, *patch_unanswered_tool_calls(scratch)])
    log.info("agent", agent=spec.name, tool_calls=[c["name"] for c in getattr(reply, "tool_calls", [])])
    return reply


def other_findings(agent_outputs: dict, exclude: str) -> str:
    """The other agents' results this turn, compact, for an action agent's prompt (e.g. nearest showroom)."""
    others = {}
    for name, out in (agent_outputs or {}).items():
        if name == exclude:
            continue
        others[name] = {
            "status": out.get("status"),
            "answer": clip(out.get("answer", ""), OTHER_ANSWER_MAX_CHARS),
            "tool_results": [{"tool": r.get("tool"), "content": clip(str(r.get("content", "")),
                                                                     OTHER_TOOL_RESULT_MAX_CHARS)}
                             for r in out.get("tool_results", [])],
        }
    return OTHER_FINDINGS_BLOCK.format(findings=json.dumps(others, ensure_ascii=False)) if others else ""


def make_action_node(spec: AgentSpec, llm_with_tools):
    """An action agent in the main graph: scratchpad action_messages; sees the other agents' findings."""

    def action_agent(state: dict) -> dict:
        reply = call_agent(
            spec, llm_with_tools, slots=agent_view(state.get("slots") or {}, spec.intent), summary=summary_text(state),
            dialogue=recent_dialogue(state.get("messages", []), spec.history_messages, AI_MESSAGE_MAX_CHARS),
            scratch=state.get("action_messages") or [],
            extra=other_findings(state.get("agent_outputs"), exclude=spec.name))
        return {"action_messages": [reply], "active_action": spec.name}

    action_agent.__name__ = spec.name
    return action_agent
