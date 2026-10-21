"""AgentSpec: everything about one agent, in that agent's own file.

To add an agent, add a file in app/agents/ that defines AGENT = AgentSpec(...), give its tools agents=[name]
in app/tools/registry.py, and add its intent to the eval set. Nothing else needs editing: the catalog finds the
file, and the supervisor, router, clarify, graph, composer and CLI read everything from it.
"""

from dataclasses import dataclass, field

from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class AgentSpec:
    name: str            # graph node name, letters/digits/underscores
    intent: str          # the supervisor's label that triggers this agent
    intent_help: str     # how the supervisor recognises the intent: description + examples (one bullet)
    role: str            # how the agent is introduced in its prompt
    prompt: str          # agent-specific instructions, appended to AGENT_HEADER
    reply_order: int     # position of this agent's part in the final reply (lower comes first)
    status_text: str = "Looking that up..."  # progress message shown while it runs
    # Slots that must be known before the agent runs; otherwise clarify asks first. A tuple means "any one of
    # these" (a tuple inside it means "all of these together", e.g. ("lat", "long")).
    required_slots: tuple = field(default_factory=tuple)
    # Car details (model, variant, transmission, fuel) needed for EACH of the agent's cars before it runs.
    required_car_slots: tuple = field(default_factory=tuple)
    # With two or more cars (a comparison) neither required_slots nor required_car_slots are needed: the agent
    # answers the comparison with what it has (quotation compares starting prices).
    comparison_relaxes: bool = False
    # How many recent customer/assistant messages the agent sees. More for agents that gather details over
    # several turns (support), at the cost of a larger request.
    history_messages: int = 4

    @property
    def tools(self) -> list[BaseTool]:
        from app.tools.registry import tools_for

        return tools_for(self.name)

    @property
    def tool_names(self) -> set[str]:
        return {t.name for t in self.tools}

    @property
    def has_action_tools(self) -> bool:
        from app.tools.registry import TOOL_REGISTRY

        return any(TOOL_REGISTRY[name].action for name in self.tool_names)
