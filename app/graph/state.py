"""Graph State. See ROADMAP.md section 3 for who reads and writes each field.

The non-lead agents run in parallel as worker subgraphs with their own private WorkerState; each returns only
its own entry in agent_outputs, which the merge_outputs reducer adds to the others.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


def merge_outputs(old: dict | None, new: dict | None) -> dict:
    """Each agent adds its own entry; None (written by the supervisor or a rollback) resets to {}."""
    if new is None:
        return {}
    return {**(old or {}), **new}


class State(TypedDict, total=False):
    # Customer chat only: HumanMessages, clarify questions, composer replies.
    # add_messages: appends; a message with an existing id replaces it; RemoveMessage deletes.
    messages: Annotated[list, add_messages]

    # Gemini's labels for the current question. Replaced every turn by the supervisor.
    intents: list[str]

    # model, variant, fuel, transmission, city, state, area_or_pincode, lat, long, name, phone.
    # Merged in Python by the supervisor (new non-empty values overwrite, missing ones are kept).
    slots: dict

    # Scratchpad of the action agent now running (lead, support...): tool-call AIMessages + ToolMessages.
    # Action agents take turns, so one is enough; cleared by save_output and by the supervisor's per-turn reset.
    # (The parallel agents keep theirs inside their own WorkerState.)
    action_messages: Annotated[list, add_messages]

    # Which action agent is running, so approval / ask_user / tools return to it. "" when none.
    active_action: str

    # {agent: {"answer": str, "tool_results": [{"tool", "args", "content"}],
    #          "status": "complete"|"incomplete"|"failed", "seconds": float}}
    # Reducer merge_outputs: every agent adds its own entry, so parallel agents never overwrite each other.
    agent_outputs: Annotated[dict, merge_outputs]

    # Set by the guardrail when the composer must fix its reply; None otherwise.
    guardrail_feedback: str | None

    # 0 or 1 within a turn.
    guardrail_retries: int

    # Summary of older messages (replaced by summarize).
    summary: str


class WorkerState(TypedDict, total=False):
    """Private state of one parallel agent. Its keys differ from State on purpose: only agent_outputs is
    returned to the main graph (two workers writing e.g. `slots` in the same step would be an error)."""

    agent: str
    dialogue: list                         # recent customer/assistant messages, copied in by Send
    known_slots: dict
    conversation_summary: str
    started: float                         # time.perf_counter() when the router started this worker
    scratch: Annotated[list, add_messages]  # this worker's tool-call AIMessages + ToolMessages
    error: str | None                      # set when Gemini stayed unavailable after all retries
    agent_outputs: Annotated[dict, merge_outputs]


class WorkerOutput(TypedDict):
    agent_outputs: Annotated[dict, merge_outputs]
