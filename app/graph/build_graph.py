"""Adds every node and edge (exactly as in ROADMAP.md section 4) and compiles with MemorySaver.

The non-lead agents are worker subgraphs that the router starts in parallel (Send); they all lead to the
after_agents join, which runs once when the slowest has finished. Lead runs after them in the main graph.

    python -m app.graph.build_graph     # writes the Mermaid diagram to docs/graph.md
"""

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.agents import CATALOG, Catalog
from app.config import PROJECT_ROOT, get_settings
from app.graph.nodes.agent_factory import make_action_node
from app.graph.nodes.approval import make_approval, make_ask_user
from app.graph.nodes.clarify import make_clarify
from app.graph.nodes.composer import make_composer
from app.graph.nodes.guardrail import guardrail
from app.graph.nodes.router import (after_action, after_agents, after_guardrail, agent_router, route_after_supervisor,
                                    route_back)
from app.graph.nodes.save_output import save_output
from app.graph.nodes.summarize import make_summarize
from app.graph.nodes.supervisor import make_supervisor, router_output_model
from app.graph.nodes.worker import make_worker
from app.graph.state import State

AGENT_NAMES = CATALOG.names      # every agent, in reply order (from app/agents/)
WORKER_NAMES = CATALOG.workers   # the ones that run in parallel (all but the action agents)
# Main graph: supervisor, the parallel group (one step), after_agents, composer x2, guardrail x2, summarize, and
# one spare. Each action agent adds up to 4 steps per tool round + its final step + save_output. Workers count
# their own steps (3 per round + final agent + finish), which is fewer.
FIXED_STEPS = 9
SAFETY_MARGIN = 10


@dataclass
class GraphLLMs:
    """The Gemini clients the graph uses. Tests pass fakes; production uses from_settings()."""
    supervisor: object  # with_structured_output(router_output_model(catalog.intents))
    clarify: object
    composer: object
    summarize: object
    agents: dict = field(default_factory=dict)  # agent name -> bind_tools(own tools)

    @classmethod
    def from_settings(cls, catalog: Catalog = CATALOG) -> "GraphLLMs":
        from app.llm import STRUCTURED_OUTPUT_METHOD, get_chat_model, logged

        return cls(
            supervisor=logged("supervisor", get_chat_model("supervisor").with_structured_output(
                router_output_model(catalog.intents), method=STRUCTURED_OUTPUT_METHOD, include_raw=True)),
            clarify=logged("clarify", get_chat_model("clarify")),
            composer=logged("composer", get_chat_model("composer")),
            summarize=logged("summarize", get_chat_model("summarize")),
            agents={name: logged(name, get_chat_model(name).bind_tools(spec.tools))
                    for name, spec in catalog.agents.items()},
        )


def recursion_limit(max_tool_rounds: int | None = None, catalog: Catalog = CATALOG) -> int:
    """The main graph's longest turn: fixed steps + every action agent's loop (4 steps per tool round: agent,
    approval, ask_user, tools; plus its final step and save_output) + margin. Workers run fewer steps."""
    rounds = max_tool_rounds if max_tool_rounds is not None else get_settings().max_tool_rounds
    return FIXED_STEPS + len(catalog.action_agents) * (4 * rounds + 2) + SAFETY_MARGIN


def build_graph(llms: GraphLLMs | None = None, checkpointer=None, catalog: Catalog = CATALOG, *,
                memory: bool = True):
    """Every agent without action tools becomes a parallel worker; the action agents (lead, support...) run after
    them one at a time, sharing approval / ask_user / tools. Tests can pass their own catalog.
    memory=False compiles without a checkpointer, for hosts that bring their own (LangGraph Studio's dev server)."""
    llms = llms or GraphLLMs.from_settings(catalog)
    actions = list(catalog.action_agents)
    g = StateGraph(State)

    g.add_node("supervisor", make_supervisor(llms.supervisor, catalog))
    g.add_node("clarify", make_clarify(llms.clarify, catalog))
    for name in catalog.workers:
        g.add_node(name, make_worker(catalog.agents[name], llms.agents[name]))
    g.add_node("after_agents", lambda _state: {})  # join: runs once, after every parallel agent finished
    g.add_node("composer", make_composer(llms.composer, catalog))
    g.add_node("guardrail", guardrail)
    g.add_node("summarize", make_summarize(llms.summarize))

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", partial(route_after_supervisor, catalog=catalog),
                            ["clarify", "composer", *catalog.workers, *actions])
    g.add_edge("clarify", END)
    for name in catalog.workers:
        g.add_edge(name, "after_agents")
    g.add_conditional_edges("after_agents", partial(after_agents, catalog=catalog), ["composer", *actions])

    if actions:
        tools = list({t.name: t for a in actions for t in catalog.agents[a].tools}.values())
        for name in actions:
            g.add_node(name, make_action_node(catalog.agents[name], llms.agents[name]))
            g.add_conditional_edges(name, agent_router, {"approval": "approval", "save_output": "save_output"})
        g.add_node("approval", make_approval(catalog), destinations=("tools", "ask_user", *actions))
        g.add_node("ask_user", make_ask_user(catalog), destinations=("tools", *actions))
        g.add_node("tools", ToolNode(tools, messages_key="action_messages"))  # approval checks whose tool it is
        g.add_node("save_output", save_output)
        g.add_conditional_edges("tools", route_back, actions)
        g.add_conditional_edges("save_output", partial(after_action, catalog=catalog), ["composer", *actions])

    g.add_edge("composer", "guardrail")
    g.add_conditional_edges("guardrail", after_guardrail,
                            {"composer": "composer", "summarize": "summarize", END: END})
    g.add_edge("summarize", END)

    if not memory:
        return g.compile()
    return g.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())


def write_mermaid(graph, path: Path = PROJECT_ROOT / "docs" / "graph.md") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Graph (generated by `python -m app.graph.build_graph`)\n\n"
                    "Compare with the flowchart and ROADMAP.md section 4.\n\n"
                    f"```mermaid\n{graph.get_graph().draw_mermaid()}```\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    from unittest.mock import MagicMock

    # Drawing needs only the structure, so no Gemini clients (or API key) are required.
    fake = GraphLLMs(MagicMock(), MagicMock(), MagicMock(), MagicMock(), {a: MagicMock() for a in AGENT_NAMES})
    print(f"Wrote {write_mermaid(build_graph(fake))}")
    print(f"recursion_limit = {recursion_limit()}")
