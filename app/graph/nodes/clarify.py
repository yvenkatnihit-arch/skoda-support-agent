"""Clarify: ask ONE short question for the missing slot(s); the turn then ends."""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents import CATALOG, Catalog
from app.graph.car_slots import public_view
from app.graph.nodes.supervisor import missing_slots
from app.prompts.clarify import CLARIFY_PROMPT


def make_clarify(llm, catalog: Catalog = CATALOG):
    def clarify(state: dict) -> dict:
        slots = state.get("slots") or {}
        missing = missing_slots(state.get("intents") or [], slots, catalog)
        wanted = sorted({gap for gaps in missing.values() for gap in gaps})
        system = SystemMessage(CLARIFY_PROMPT.format(
            missing=", ".join(wanted),
            intents=", ".join(missing),
            slots=json.dumps({k: v for k, v in public_view(slots).items() if v and k != "phone"}, ensure_ascii=False),
        ))
        last_question = next((m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)), None)
        reply = llm.invoke([system, *([last_question] if last_question else [])])
        # asking again counts toward a hand-off: after ESCALATE_AFTER_STRIKES the router offers a person instead
        return {"messages": [AIMessage(content=reply.text)], "strikes": (state.get("strikes") or 0) + 1}

    return clarify
