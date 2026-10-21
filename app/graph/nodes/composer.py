"""Composer: one Gemini call (no tools) that writes the single reply to the customer from agent_outputs.
On a guardrail fix it reuses the rejected reply's message id, so the reply is replaced, not duplicated."""

import json
import uuid
from datetime import date

from langchain_core.messages import AIMessage, SystemMessage

from app.agents import CATALOG, Catalog
from app.graph.car_slots import public_view
from app.graph.nodes.context import compact_findings, recent_dialogue
from app.prompts.composer import COMPOSER_PROMPT, FEEDBACK_BLOCK

RECENT_MESSAGES = 4
AI_MESSAGE_MAX_CHARS = 800


def in_reply_order(agent_outputs: dict, catalog: Catalog = CATALOG) -> dict:
    """Parallel agents finish in any order; the reply presents their parts in the catalog's reply_order."""
    outputs = agent_outputs or {}
    rank = {a: i for i, a in enumerate(catalog.names)}
    return {a: outputs[a] for a in sorted(outputs, key=lambda a: rank.get(a, len(rank)))}


def make_composer(llm, catalog: Catalog = CATALOG):
    def composer(state: dict) -> dict:
        messages = state.get("messages", [])
        feedback = state.get("guardrail_feedback")
        slots = {k: v for k, v in public_view(state.get("slots") or {}).items()
                 if v not in (None, "", [], {}) and k != "phone"}
        system = SystemMessage(COMPOSER_PROMPT.format(
            today=date.today().isoformat(),
            slots=json.dumps(slots, ensure_ascii=False),
            findings=json.dumps(compact_findings(in_reply_order(state.get("agent_outputs"), catalog)), ensure_ascii=False),
            feedback=FEEDBACK_BLOCK.format(feedback=feedback) if feedback else "",
        ))
        if feedback and messages and isinstance(messages[-1], AIMessage):
            reply_id = messages[-1].id      # replace the rejected reply
            history = messages[:-1]
        else:
            reply_id = str(uuid.uuid4())
            history = messages
        recent = recent_dialogue(history, RECENT_MESSAGES, AI_MESSAGE_MAX_CHARS)
        reply = llm.invoke([system, *recent])
        return {"messages": [AIMessage(content=reply.text, id=reply_id)]}

    return composer
