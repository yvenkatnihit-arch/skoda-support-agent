"""Keep Gemini requests small: recent dialogue only, long assistant replies and summaries clipped."""

from langchain_core.messages import AIMessage, HumanMessage

SUMMARY_MAX_CHARS = 1200
TOOL_RESULT_MAX_CHARS = 6000  # per tool result in the composer's findings


def clip(text: str, max_chars: int) -> str:
    text = text or ""
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + " …[shortened]"


def recent_dialogue(messages: list, n: int, ai_max_chars: int) -> list:
    """The last n customer/assistant messages. Customer messages are kept whole (the latest question must be
    exact); earlier assistant replies are clipped because a full quote table isn't needed to route a follow-up."""
    dialogue = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))][-n:]
    out = []
    for m in dialogue:
        if isinstance(m, AIMessage) and len(m.text) > ai_max_chars:
            out.append(AIMessage(content=clip(m.text, ai_max_chars), id=m.id))
        else:
            out.append(m)
    return out


def summary_text(state: dict) -> str:
    return clip(state.get("summary") or "", SUMMARY_MAX_CHARS) or "(none)"


def compact_findings(agent_outputs: dict) -> dict:
    """agent_outputs for the composer prompt, with each tool result capped (the guardrail still checks the full ones)."""
    return {
        agent: {**out, "tool_results": [{**r, "content": clip(str(r.get("content", "")), TOOL_RESULT_MAX_CHARS)}
                                        for r in out.get("tool_results", [])]}
        for agent, out in (agent_outputs or {}).items()
    }
