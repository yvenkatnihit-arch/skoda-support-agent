"""Hand-off to a human (ROADMAP section 13): the supervisor's escalation flag or repeated failures send the turn to
escalate, which gives the composer the hand-off notes; a later ticket is high priority. Offline, scripted LLMs."""

import json

from langchain_core.messages import AIMessage, HumanMessage

from app.graph.nodes.escalate import handoff_notes
from app.graph.nodes.router import needs_person, route_after_supervisor
from app.graph.nodes.supervisor import RouterOutput
from app.graph.runner import FRIENDLY_ERROR, run_turn
from app.graph.slots import Slots
from app.llm import LLMUnavailableError
from app.tools.ticket_tools import create_ticket
from tests.test_graph import CONFIG, ai_texts, ask, call, make, route
from tests.test_reliability import DEADLINE
from tests.test_support import TICKET, _rows, tickets_file  # noqa: F401 (fixture)


def flagged(reason, intents=("support_request",), **slots):
    return RouterOutput(intents=list(intents), slots=Slots(**slots), escalation=reason)


def test_needs_person_on_a_flag_or_after_repeated_failures():
    assert needs_person({"escalation": "frustration"})
    assert not needs_person({"strikes": 1}) and needs_person({"strikes": 2})  # ESCALATE_AFTER_STRIKES = 2
    # escalation is checked before clarify: no more questions for an upset customer
    assert route_after_supervisor({"escalation": "safety", "intents": ["quotation"], "slots": {}}) == "escalate"


def test_handoff_notes_per_reason():
    assert "1800 2700 260" in handoff_notes("frustration") and "callback" in handoff_notes("frustration")
    safety = handoff_notes("safety")
    assert "112" in safety and "roadside" in safety and safety.index("112") < safety.index("toll-free")
    assert "don't argue" in handoff_notes("dispute")


def test_frustration_hands_off_instead_of_answering():
    graph, llms = make(supervisor=[flagged("frustration")],
                       composer=[AIMessage(content="I'm really sorry. Call 1800 2700 260, or I can arrange a callback.")])
    state = ask(graph, "This is useless, I've asked about my refund three times!")
    assert llms.agents["support"].calls == [] and llms.clarify.calls == []  # no agent, no more questions
    assert "HAND-OFF TO A PERSON" in llms.composer.calls[0][0].content
    assert state["escalated"] is True and state["strikes"] == 0
    assert ai_texts(state)[-1].startswith("I'm really sorry")


def test_two_stuck_turns_in_a_row_hand_off_on_the_third():
    stuck = route(["quotation"])  # no model: clarify each time
    graph, llms = make(supervisor=[stuck, stuck, stuck],
                       clarify=[AIMessage(content="Which variant and city?"), AIMessage(content="Which variant?")],
                       composer=[AIMessage(content="Sorry I'm not getting this right. Call 1800 2700 260.")])
    assert ask(graph, "Slavia price")["strikes"] == 1
    assert ask(graph, "the price")["strikes"] == 2
    state = ask(graph, "just tell me the price")
    assert len(llms.clarify.calls) == 2  # not asked a third time
    assert "hasn't been able to help" in llms.composer.calls[0][0].content and state["strikes"] == 0


def test_a_good_reply_resets_the_counter_and_a_safe_reply_counts():
    graph, _ = make(supervisor=[route(["quotation"]), route(["small_talk"]), route(["small_talk"])],
                    clarify=[AIMessage(content="Which variant and city?")],
                    composer=[AIMessage(content="Hello!"),
                              AIMessage(content="From ₹9 lakh"), AIMessage(content="From ₹9 lakh")])
    assert ask(graph, "Slavia price")["strikes"] == 1
    assert ask(graph, "hi")["strikes"] == 0                     # a good, checked reply
    assert ask(graph, "price?")["strikes"] == 1                 # invented figure twice -> safe reply


def test_a_failed_turn_counts_toward_a_hand_off():
    class Down:
        def invoke(self, _messages):
            raise LLMUnavailableError("composer", 4, DEADLINE)

    graph, llms = make(supervisor=[route(["small_talk"])] * 2)
    llms.composer.invoke = Down().invoke
    first = run_turn(graph, CONFIG, {"messages": [HumanMessage("Hi")]})
    assert first.reply == FRIENDLY_ERROR and graph.get_state(CONFIG).values["strikes"] == 1
    run_turn(graph, CONFIG, {"messages": [HumanMessage("Hello?")]})
    assert graph.get_state(CONFIG).values["strikes"] == 2       # the next turn will hand off


def test_after_a_hand_off_tickets_are_high_priority(tickets_file):  # noqa: F811
    asha = {k: TICKET[k] for k in ("name", "phone", "city", "model")}
    graph, llms = make(
        supervisor=[flagged("dispute", **asha), route(["support_request"], **asha)],
        composer=[AIMessage(content="Sorry. Call 1800 2700 260, or I can raise an escalated ticket."),
                  AIMessage(content="Ticket raised.")],
        support=[call("create_ticket", {**TICKET, "priority": "high"}, "t1"), AIMessage(content="Ticket raised")],
    )
    ask(graph, "I'll take you to consumer court over this refund")
    ask(graph, "Yes, raise the ticket")
    assert "priority \"high\"" in llms.agents["support"].calls[0][0].content  # the agent was told
    from langgraph.types import Command
    graph.invoke(Command(resume={"approved": True}), CONFIG)
    [row] = _rows(tickets_file)
    assert row["priority"] == "high"


def test_ticket_priority_defaults_to_normal(tickets_file):  # noqa: F811
    out = json.loads(create_ticket.invoke(TICKET))
    assert out["priority"] == "normal" and _rows(tickets_file)[0]["priority"] == "normal"


def test_the_internal_label_never_reaches_the_customer():
    graph, _ = make(supervisor=[flagged("frustration")],
                    composer=[AIMessage(content="HAND-OFF TO A PERSON: I'm sorry. Call 1800 2700 260.")])
    assert ai_texts(ask(graph, "useless bot"))[-1] == "I'm sorry. Call 1800 2700 260."
