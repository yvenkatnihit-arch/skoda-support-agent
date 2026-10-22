"""Supervisor: one Gemini call per customer question -> intents + slots. Which agents run is decided in Python."""

import json
from functools import lru_cache
from typing import Literal

import structlog
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import BaseModel, Field, create_model

from app.agents import CATALOG, Catalog
from app.data.repository import get_repository
from app.graph.car_slots import cars_for, merge_slots, public_view  # noqa: F401 (merge_slots re-exported)
from app.graph.nodes.context import recent_dialogue, summary_text
from app.graph.slots import Slots
from app.prompts.supervisor import SUPERVISOR_PROMPT

log = structlog.get_logger()

# Every intent, required slot and agent comes from the agent catalog (one file per agent in app/agents/).
HISTORY_MESSAGES = 6         # routing needs the recent turns only; older context is in the summary and slots
AI_MESSAGE_MAX_CHARS = 600


Escalation = Literal["frustration", "safety", "dispute"]  # why a person should take over (ROADMAP section 13)


@lru_cache
def router_output_model(intents: tuple[str, ...]) -> type[BaseModel]:
    """The supervisor's structured output, with the intent labels limited to the catalog's intents."""
    intent_type = Literal[intents]
    return create_model(
        "RouterOutput",
        intents=(list[intent_type], Field(min_length=1, description="All intents in the customer's latest message")),
        slots=(Slots, Field(default_factory=Slots, description="Details stated by the customer; null if not stated")),
        escalation=(Escalation | None, Field(default=None, description=(
            "Set ONLY when a person should take over: frustration (the customer is angry or fed up with the "
            "assistant), safety (accident, injury, fire, brakes or steering failing), dispute (legal action, "
            "consumer court, a refund or compensation dispute). Otherwise null."))),
    )


RouterOutput = router_output_model(CATALOG.intents)
INTENTS = CATALOG.intents


def agents_for(intents: list[str], catalog: Catalog = CATALOG) -> list[str]:
    """Agents for these intents, in intent order, without duplicates; lead always last."""
    return catalog.agents_for(intents)


def _filled(slots: dict, key) -> bool:
    if isinstance(key, tuple):
        return all(slots.get(k) not in (None, "") for k in key)
    return slots.get(key) not in (None, "")


def _gaps(slots: dict, requirements) -> list[str]:
    gaps = []
    for req in requirements:
        if isinstance(req, tuple):
            if not any(_filled(slots, option) for option in req):
                gaps.append("area, pincode or city")
        elif not _filled(slots, req):
            gaps.append(req)
    return gaps


def missing_slots(intents: list[str], slots: dict, catalog: Catalog = CATALOG) -> dict[str, list[str]]:
    """{intent: [missing slot descriptions]}. Any gap means clarify first, and no agent runs this turn.
    Shared details come from the agent's required_slots (a tuple = "any one of these"; lat + long count together),
    car details from required_car_slots, checked for each of the agent's cars. An agent with comparison_relaxes
    needs neither when two or more cars are compared (it compares starting prices instead)."""
    missing: dict[str, list[str]] = {}
    for intent in intents:
        agent = catalog.intent_to_agent.get(intent)
        if agent is None:
            continue
        spec = catalog.agents[agent]
        cars = cars_for(slots, intent)
        if spec.comparison_relaxes and len(cars) >= 2:
            continue
        gaps = _gaps(slots, spec.required_slots)
        if spec.required_car_slots:
            if not cars:
                gaps += list(spec.required_car_slots)
            for car in cars:
                for key in spec.required_car_slots:
                    if not car.get(key):
                        gaps.append(f"{key} (for the {car['model']})" if car.get("model") and len(cars) > 1
                                    else key)
        if gaps:
            missing[intent] = list(dict.fromkeys(gaps))
    return missing


def last_assistant_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m.text[:500]
    return "(none)"


def make_supervisor(router_llm, catalog: Catalog = CATALOG):
    """router_llm: Gemini with .with_structured_output(router_output_model(catalog.intents))."""
    output_model = router_output_model(catalog.intents)

    def supervisor(state: dict) -> dict:
        messages = state.get("messages", [])
        slots = state.get("slots") or {}
        system = SystemMessage(SUPERVISOR_PROMPT.format(
            intents=catalog.intents_help(),
            models=", ".join(m.name for m in get_repository().list_models()),
            slots=json.dumps({k: v for k, v in public_view(slots).items() if v not in (None, [], {})},
                             ensure_ascii=False),
            last_assistant=last_assistant_text(messages),
            summary=summary_text(state),
        ))
        history = recent_dialogue(messages, HISTORY_MESSAGES, AI_MESSAGE_MAX_CHARS)
        result = router_llm.invoke([system, *history])
        if not isinstance(result, output_model):
            result = output_model.model_validate(result if isinstance(result, dict) else result.model_dump())

        intents = list(dict.fromkeys(result.intents))
        latest = next((m.text for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        merged = merge_slots(slots, result.slots, latest)
        log.info("supervisor", intents=intents, agents=catalog.agents_for(intents),
                 slots={k: v for k, v in public_view(merged).items() if v})
        update = {
            "intents": intents, "slots": merged,
            # per-turn reset: nothing from the previous question leaks into this one (None -> {} in the reducer)
            "agent_outputs": None, "guardrail_feedback": None, "guardrail_retries": 0,
        }
        update["active_action"] = ""
        update["escalation"] = getattr(result, "escalation", None)
        if state.get("action_messages"):
            update["action_messages"] = [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
        return update

    return supervisor
