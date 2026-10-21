"""Parallel agent workers. Each non-lead agent (car_info, quotation, service, locator) is a small subgraph with
its own private scratchpad, so several can run at the same time without mixing their tool calls:

    agent (Gemini) -> worker_router -> approval (own read-only tools only) -> tools -> agent ... -> finish

finish returns only {"agent_outputs": {name: output}}; the merge_outputs reducer adds it to the others.
If Gemini stays unavailable after all retries, this worker reports status "failed" and the others carry on.
"""

import time

import structlog
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from app.agents.base import AgentSpec
from app.config import get_settings
from app.graph.car_slots import agent_view
from app.graph.nodes.agent_factory import AI_MESSAGE_MAX_CHARS, RECENT_MESSAGES, call_agent, tool_rounds
from app.graph.nodes.approval import pending_calls, rejected_calls
from app.graph.nodes.context import recent_dialogue, summary_text
from app.graph.nodes.save_output import build_output
from app.graph.state import WorkerOutput, WorkerState
from app.llm import LLMUnavailableError

log = structlog.get_logger()


def worker_input(state: dict, agent: str) -> dict:
    """What a worker gets from the main State when the router starts it (via Send)."""
    from app.agents import AGENTS

    spec = AGENTS.get(agent)
    history = spec.history_messages if spec else RECENT_MESSAGES
    return {
        "agent": agent,
        "dialogue": recent_dialogue(state.get("messages", []), history, AI_MESSAGE_MAX_CHARS),
        "known_slots": agent_view(state.get("slots") or {}, spec.intent if spec else None),  # only its cars
        "conversation_summary": summary_text(state),
        "started": time.perf_counter(),
    }


def make_worker(spec: AgentSpec, llm_with_tools):
    """Compiled worker subgraph for one agent. llm_with_tools: Gemini with .bind_tools(spec.tools)."""

    def agent(ws: dict) -> dict:
        try:
            reply = call_agent(spec, llm_with_tools, slots=ws.get("known_slots") or {},
                               summary=ws.get("conversation_summary") or "(none)",
                               dialogue=ws.get("dialogue") or [], scratch=ws.get("scratch") or [])
        except LLMUnavailableError as exc:
            log.warning("agent failed; the other agents continue", agent=spec.name, error=str(exc)[:200])
            return {"error": type(exc).__name__}
        return {"scratch": [reply]}

    def worker_router(ws: dict) -> str:
        """Tool calls within the round limit -> approval; final answer, limit reached or failure -> finish."""
        if ws.get("error"):
            return "finish"
        scratch = ws.get("scratch") or []
        last = scratch[-1] if scratch else None
        if isinstance(last, AIMessage) and last.tool_calls and tool_rounds(scratch) <= get_settings().max_tool_rounds:
            return "approval"
        return "finish"

    def approval(ws: dict) -> Command:
        """Workers never pause: only the agent's own read-only tools may run."""
        rejected = rejected_calls(spec, pending_calls(ws.get("scratch") or []), allow_actions=False)
        if rejected:
            return Command(goto="agent", update={"scratch": rejected})
        return Command(goto="tools")

    def finish(ws: dict) -> dict:
        started = ws.get("started")
        seconds = time.perf_counter() - started if started else None
        return {"agent_outputs": {spec.name: build_output(ws.get("scratch") or [], failed=bool(ws.get("error")),
                                                          seconds=seconds)}}

    g = StateGraph(WorkerState, output_schema=WorkerOutput)
    g.add_node("agent", agent)
    g.add_node("approval", approval, destinations=("agent", "tools"))
    g.add_node("tools", ToolNode(spec.tools, messages_key="scratch"))
    g.add_node("finish", finish)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", worker_router, {"approval": "approval", "finish": "finish"})
    g.add_edge("tools", "agent")
    g.add_edge("finish", END)
    return g.compile(name=spec.name)
