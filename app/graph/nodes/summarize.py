"""Summarize: when messages holds more than 20 entries, fold everything before the last 2 customer
questions into the running summary and remove those messages. Only cuts at a HumanMessage; slots untouched."""

import structlog
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage

from app.llm import LLMUnavailableError
from app.log import mask_phone
from app.prompts.summarize import SUMMARIZE_INPUT, SUMMARIZE_PROMPT

log = structlog.get_logger()

KEEP_QUESTIONS = 2


def split_for_summary(messages: list) -> tuple[list, list]:
    """(older, kept): kept starts at the 2nd-last HumanMessage. older is empty if there's nothing to cut."""
    human_idx = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_idx) < KEEP_QUESTIONS:
        return [], messages
    cut = human_idx[-KEEP_QUESTIONS]
    return messages[:cut], messages[cut:]


def _transcript(messages: list) -> str:
    lines = []
    for m in messages:
        role = "Customer" if isinstance(m, HumanMessage) else "Assistant" if isinstance(m, AIMessage) else None
        if role:
            lines.append(f"{role}: {mask_phone(m.text)}")
    return "\n".join(lines)


def make_summarize(llm):
    def summarize(state: dict) -> dict:
        older, _ = split_for_summary(state.get("messages") or [])
        if not older:
            return {}
        request = [SystemMessage(SUMMARIZE_PROMPT),
                   HumanMessage(SUMMARIZE_INPUT.format(summary=state.get("summary") or "(none)",
                                                       transcript=_transcript(older)))]
        try:
            new_summary = llm.invoke(request).text.strip()
        except LLMUnavailableError:
            # The reply is already written and checked: don't fail the turn. Keep the messages; retry next turn.
            log.warning("summary failed; messages kept, will retry next turn", exc_info=True)
            return {}
        if not new_summary:
            log.warning("summary came back empty; messages kept, will retry next turn")
            return {}
        return {"summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in older if m.id]}

    return summarize
