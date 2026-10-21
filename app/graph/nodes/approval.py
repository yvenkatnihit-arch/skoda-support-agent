"""approval + ask_user for the action agents (lead, support...: the agents with action tools, from the catalog).
They run one at a time in the main graph after the parallel agents; active_action says which one is running.
approval: read-only tools go straight to the tools node; action tools go to ask_user.
ask_user: pauses the graph with interrupt() until the customer confirms or cancels.
rejected_calls is shared with the parallel workers' own approval step (app/graph/nodes/worker.py)."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command, interrupt

from app.agents import Catalog
from app.agents.base import AgentSpec
from app.tools.registry import TOOLS_REQUIRING_APPROVAL, precheck_problem

NOT_ALLOWED = "NOT DONE: '{tool}' is not one of your tools. Use only your own tools."
CANCELLED_BY_CUSTOMER = "NOT DONE: the customer cancelled this action."
YES_WORDS = {"yes", "y", "confirm", "confirmed", "ok", "okay", "approve", "approved", "true"}
SCRATCH = "action_messages"


def pending_calls(scratch: list) -> list[dict]:
    last = scratch[-1] if scratch else None
    return list(last.tool_calls) if isinstance(last, AIMessage) else []


def rejected_calls(spec: AgentSpec, calls: list[dict], *, allow_actions: bool = True) -> list[ToolMessage] | None:
    """NOT DONE ToolMessages for every call if any requested tool isn't the agent's own (or is an action tool
    where actions aren't allowed); None if all calls may run. A model naming a tool must never be enough."""
    allowed = spec.tool_names
    bad = [c for c in calls if c["name"] not in allowed or (not allow_actions and c["name"] in TOOLS_REQUIRING_APPROVAL)]
    if not bad:
        return None
    return [ToolMessage(content=NOT_ALLOWED.format(tool=c["name"]) if c in bad else
                        "NOT DONE: skipped because another requested tool was not allowed.",
                        tool_call_id=c["id"], name=c["name"]) for c in calls]


def make_approval(catalog: Catalog):
    def approval(state: dict) -> Command:
        lead = catalog.agents[state["active_action"]]
        calls = pending_calls(state.get(SCRATCH) or [])
        rejected = rejected_calls(lead, calls)
        if rejected:
            return Command(goto=lead.name, update={SCRATCH: rejected})
        actions = [c for c in calls if c["name"] in TOOLS_REQUIRING_APPROVAL]
        problems = {c["id"]: precheck_problem(c["name"], c["args"]) for c in actions}
        if any(problems.values()):  # don't ask the customer to confirm something incomplete or that would be rejected
            return Command(goto=lead.name, update={SCRATCH: [
                ToolMessage(content=f"NOT DONE: {problems.get(c['id']) or 'skipped until the details are fixed.'}",
                            tool_call_id=c["id"], name=c["name"]) for c in calls]})
        if actions:
            return Command(goto="ask_user")
        return Command(goto="tools")

    return approval


def parse_decision(decision) -> tuple[bool, str | None]:
    """Resume value from the UI -> (approved, feedback). Anything that isn't a clear yes counts as no:
    an action that saves customer data must never run on an ambiguous answer."""
    if isinstance(decision, bool):
        return decision, None
    if isinstance(decision, str):
        return decision.strip().lower() in YES_WORDS, None
    if isinstance(decision, dict):
        approved = decision.get("approved")
        feedback = decision.get("feedback")
        feedback = feedback.strip() if isinstance(feedback, str) and feedback.strip() else None
        return approved is True, feedback
    return False, None


def make_ask_user(catalog: Catalog):
    def ask_user(state: dict) -> Command:
        lead = catalog.agents[state["active_action"]]
        calls = pending_calls(state.get(SCRATCH) or [])
        decision = interrupt({
            "type": "approval",
            "agent": lead.name,
            "actions": [{"tool": c["name"], "details": c["args"]} for c in calls
                        if c["name"] in TOOLS_REQUIRING_APPROVAL],
            "question": "Shall I go ahead with this?",
        })
        approved, feedback = parse_decision(decision)
        if approved:
            return Command(goto="tools")
        update = [ToolMessage(content=CANCELLED_BY_CUSTOMER, tool_call_id=c["id"], name=c["name"]) for c in calls]
        if feedback:
            update.append(HumanMessage(content=f"Customer's feedback on the cancelled action: {feedback}"))
        return Command(goto=lead.name, update={SCRATCH: update})

    return ask_user
