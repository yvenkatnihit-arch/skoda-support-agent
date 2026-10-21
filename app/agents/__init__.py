"""The agent catalog: every agent file in this package (one AgentSpec named AGENT per file), checked at startup.

Adding an agent = adding a file here (plus register() lines for its tools). The supervisor's intents, the
required slots, which agents run in parallel, the reply order and the CLI progress messages all come from the
catalog, so no other file needs editing.

Agents with action tools (anything that saves something and needs Confirm/Cancel) are "action agents": they run
after the parallel group, one at a time in reply order, because each may pause for the customer.
"""

import importlib
import pkgutil
import re
from dataclasses import dataclass

from app.agents.base import AgentSpec
from app.graph.slots import CAR_KEYS, SLOT_KEYS

# Intents with no agent: the composer answers them directly.
RESERVED_INTENTS = {
    "small_talk": 'greetings, thanks, chit-chat. Examples: "Hi!" / "Thanks a lot" / "How are you?"',
    "out_of_scope": ("anything that is not about Škoda cars or Škoda services in India.\n"
                     '  Examples: "Who will win the IPL?" / "Hyundai Creta price" / "Write me a poem"'),
}
# Graph nodes an agent can't be named after.
RESERVED_NODE_NAMES = {"supervisor", "clarify", "after_agents", "save_output", "approval", "ask_user", "tools",
                       "composer", "guardrail", "summarize"}
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class CatalogError(ValueError):
    pass


@dataclass(frozen=True)
class Catalog:
    agents: dict[str, AgentSpec]        # in reply order
    action_agents: tuple[str, ...]      # agents with action tools, in reply order: run one at a time, may pause

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.agents)

    @property
    def workers(self) -> tuple[str, ...]:
        """Agents that run in parallel (everyone without action tools)."""
        return tuple(n for n in self.agents if n not in self.action_agents)

    def next_action_agent(self, intents: list[str], after: str | None = None) -> str | None:
        """The first asked action agent (after `after`, in reply order), or None when there are no more."""
        asked = [a for a in self.agents_for(intents) if a in self.action_agents]
        if after in asked:
            asked = asked[asked.index(after) + 1:]
        return asked[0] if asked else None

    @property
    def intent_to_agent(self) -> dict[str, str]:
        return {a.intent: a.name for a in self.agents.values()}

    @property
    def intents(self) -> tuple[str, ...]:
        return (*(a.intent for a in self.agents.values()), *RESERVED_INTENTS)

    def required_slots(self, intent: str) -> tuple:
        agent = self.intent_to_agent.get(intent)
        return self.agents[agent].required_slots if agent else ()

    def agents_for(self, intents: list[str]) -> list[str]:
        """Agents for these intents, without duplicates: parallel agents in intent order, then the action
        agents in reply order (the order they run in)."""
        out: list[str] = []
        for intent in intents:
            agent = self.intent_to_agent.get(intent)
            if agent and agent not in out:
                out.append(agent)
        return [a for a in out if a not in self.action_agents] + [a for a in self.action_agents if a in out]

    def intents_help(self) -> str:
        """The supervisor prompt's list of intents."""
        lines = [f"- {a.intent}: {a.intent_help}" for a in self.agents.values()]
        lines += [f"- {intent}: {text}" for intent, text in RESERVED_INTENTS.items()]
        return "\n".join(lines)


def _slot_names(requirement) -> list[str]:
    if isinstance(requirement, str):
        return [requirement]
    return [name for part in requirement for name in _slot_names(part)]


def build_catalog(specs: list[AgentSpec]) -> Catalog:
    """Check the agents against each other and the tool registry; refuse to start with a clear list of problems."""
    from app.tools.registry import TOOL_REGISTRY

    errors: list[str] = []
    names = [s.name for s in specs]
    intents = [s.intent for s in specs]
    for s in specs:
        if not NAME_RE.match(s.name):
            errors.append(f"agent name '{s.name}': use lower-case letters, digits and underscores")
        if s.name in RESERVED_NODE_NAMES:
            errors.append(f"agent name '{s.name}' is a built-in graph node")
        if s.intent in RESERVED_INTENTS:
            errors.append(f"agent '{s.name}': intent '{s.intent}' is reserved (answered by the composer)")
        if not s.tools:
            errors.append(f"agent '{s.name}' has no tools: register some with agents=['{s.name}']")
        unknown = [k for req in s.required_slots for k in _slot_names(req) if k not in SLOT_KEYS]
        if unknown:
            errors.append(f"agent '{s.name}': unknown required slots {unknown} (known: {', '.join(SLOT_KEYS)})")
        unknown_car = [k for k in s.required_car_slots if k not in CAR_KEYS]
        if unknown_car:
            errors.append(f"agent '{s.name}': unknown required car slots {unknown_car} (known: {', '.join(CAR_KEYS)})")
    errors += [f"agent name '{n}' is used twice" for n in sorted({n for n in names if names.count(n) > 1})]
    errors += [f"intent '{i}' belongs to two agents" for i in sorted({i for i in intents if intents.count(i) > 1})]
    for entry in TOOL_REGISTRY.values():
        for agent in entry.agents:
            if agent not in names:
                errors.append(f"tool '{entry.tool.name}' is registered for unknown agent '{agent}'")
    if errors:
        raise CatalogError("Agent setup problems:\n  - " + "\n  - ".join(errors))
    ordered = sorted(specs, key=lambda s: (s.reply_order, s.name))
    return Catalog(agents={s.name: s for s in ordered},
                   action_agents=tuple(s.name for s in ordered if s.has_action_tools))


def discover() -> list[AgentSpec]:
    """Every module in app/agents/ (except base) that defines AGENT."""
    specs = []
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        if info.name == "base" or info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        spec = getattr(module, "AGENT", None)
        if not isinstance(spec, AgentSpec):
            raise CatalogError(f"app/agents/{info.name}.py must define AGENT = AgentSpec(...)")
        specs.append(spec)
    return specs


CATALOG = build_catalog(discover())
AGENTS = CATALOG.agents  # name -> AgentSpec, in reply order
