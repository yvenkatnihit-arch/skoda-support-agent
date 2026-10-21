"""Turning an agent's scratchpad into its agent_outputs entry (answer + tool results + status).

build_output is shared by the parallel workers' finish step and by save_output, which does it for the action
agent that just finished (active_action) and then clears action_messages."""

from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage

INCOMPLETE_ANSWER = ("Sorry, I couldn't complete this part of your request (too many lookups were needed). "
                     "A Skoda team member can call you back to help.")
FAILED_ANSWER = ("Sorry, I couldn't look this part up right now (the assistant service is busy). "
                 "Please try again in a moment, or a Skoda team member can call you back.")


def collect_tool_results(scratch: list) -> list[dict]:
    args_by_id = {call["id"]: call for m in scratch if isinstance(m, AIMessage) for call in m.tool_calls}
    results = []
    for m in scratch:
        if isinstance(m, ToolMessage):
            call = args_by_id.get(m.tool_call_id, {})
            content = m.content if isinstance(m.content, str) else str(m.content)
            results.append({"tool": m.name or call.get("name"), "args": call.get("args", {}), "content": content})
    return results


def build_output(scratch: list, *, failed: bool = False, seconds: float | None = None) -> dict:
    last = scratch[-1] if scratch else None
    if failed:
        answer, status = FAILED_ANSWER, "failed"
    elif isinstance(last, AIMessage) and last.tool_calls:  # stopped by the tool-round limit
        answer, status = INCOMPLETE_ANSWER, "incomplete"
    elif isinstance(last, AIMessage) and last.text.strip():
        answer, status = last.text, "complete"
    else:
        answer, status = "No answer was produced.", "incomplete"
    out = {"answer": answer, "tool_results": collect_tool_results(scratch), "status": status}
    if seconds is not None:
        out["seconds"] = round(seconds, 2)
    return out


def save_output(state: dict) -> dict:
    """An action agent finished: add its entry (the reducer keeps the others) and clear the scratchpad for the next."""
    scratch = state.get("action_messages") or []
    return {
        "agent_outputs": {state["active_action"]: build_output(scratch)},
        "action_messages": [RemoveMessage(id=m.id) for m in scratch if m.id],
    }
