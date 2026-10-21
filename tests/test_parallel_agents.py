"""Parallel agents (ROADMAP section 7, option A): the non-lead agents run at the same time, their outputs are
merged, lead runs after them, and one failing agent doesn't sink the others. Scripted fake LLMs, no API calls."""

import threading

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from app.graph.nodes.composer import in_reply_order
from app.graph.runner import run_turn
from app.graph.state import merge_outputs
from app.llm import LLMUnavailableError
from app.memory.file_store import deserialize_state
from tests.test_graph import CONFIG, KYLAQ_AT_HYD, ScriptedLLM, ai_texts, ask, call, leads_file, make, route  # noqa: F401
from tests.test_reliability import DEADLINE

SHOWROOM = {"lat": 17.4483, "long": 78.3915, "dealer_type": "showroom"}
PRICE_AND_SHOWROOM = dict(KYLAQ_AT_HYD, lat=17.4483, long=78.3915)


class WaitsForTheOther(ScriptedLLM):
    """Each agent's first call waits until the other agent has also started: passes only if they run in parallel."""

    def __init__(self, barrier: threading.Barrier, *responses):
        super().__init__(*responses)
        self.barrier = barrier

    def invoke(self, messages):
        if not self.calls:
            self.barrier.wait()  # raises BrokenBarrierError after the timeout if the agents ran one by one
        return super().invoke(messages)


def test_merge_reducer_adds_entries_and_none_resets():
    assert merge_outputs({"quotation": 1}, {"locator": 2}) == {"quotation": 1, "locator": 2}
    assert merge_outputs({"quotation": 1}, None) == {}
    assert merge_outputs(None, {"lead": 3}) == {"lead": 3}


def test_two_agents_run_at_the_same_time_and_both_outputs_are_merged():
    barrier = threading.Barrier(2, timeout=5)
    graph, llms = make(
        supervisor=[route(["quotation", "locator"], **PRICE_AND_SHOWROOM)],
        composer=[AIMessage(content="About ₹12,92,125. Nearest showroom listed.")],
    )
    llms.agents["quotation"].invoke = WaitsForTheOther(
        barrier, call("calculate_on_road_price", KYLAQ_AT_HYD, "q1"), AIMessage(content="1292125")).invoke
    llms.agents["locator"].invoke = WaitsForTheOther(
        barrier, call("find_nearest_dealers", SHOWROOM, "l1"), AIMessage(content="Nearest showroom")).invoke
    state = ask(graph, "Kylaq Signature AT price in Hyderabad and nearest showroom")
    outputs = state["agent_outputs"]
    assert set(outputs) == {"quotation", "locator"}
    assert outputs["quotation"]["tool_results"][0]["tool"] == "calculate_on_road_price"
    assert outputs["locator"]["tool_results"][0]["tool"] == "find_nearest_dealers"
    assert all(o["status"] == "complete" for o in outputs.values())
    assert ai_texts(state) == ["About ₹12,92,125. Nearest showroom listed."]  # one reply from both


def test_one_failing_agent_keeps_the_others_answer():
    class Down:
        def invoke(self, _messages):
            raise LLMUnavailableError("locator", 4, DEADLINE)

    graph, llms = make(
        supervisor=[route(["quotation", "locator"], **PRICE_AND_SHOWROOM)],
        quotation=[call("calculate_on_road_price", KYLAQ_AT_HYD, "q1"), AIMessage(content="1292125")],
        composer=[AIMessage(content="About ₹12,92,125. Sorry, I couldn't look up showrooms right now.")],
    )
    llms.agents["locator"].invoke = Down().invoke
    result = run_turn(graph, CONFIG, {"messages": [HumanMessage("price and showroom")]})
    assert result.status == "reply"
    outputs = graph.get_state(CONFIG).values["agent_outputs"]
    assert outputs["quotation"]["status"] == "complete" and outputs["locator"]["status"] == "failed"
    findings = llms.composer.calls[0][0].content
    assert '"failed"' in findings and "1292125" in findings


def test_lead_runs_after_the_group_sees_their_results_and_resume_reuses_them(leads_file):  # noqa: F811
    args = {"name": "Asha", "phone": "9123456789", "city": "Hyderabad", "model": "Kylaq"}
    graph, llms = make(
        supervisor=[route(["locator", "lead_capture"], name="Asha", phone="9123456789", city="Hyderabad",
                          model="Kylaq", lat=17.4483, long=78.3915)],
        locator=[call("find_nearest_dealers", SHOWROOM, "l1"), AIMessage(content="Nearest: Skoda Madhapur")],
        lead=[call("book_test_drive", args, "c1"), AIMessage(content="Test drive requested")],
        composer=[AIMessage(content="Booked near you.")],
    )
    paused = ask(graph, "Nearest showroom and book a Kylaq test drive")
    assert "__interrupt__" in paused and graph.get_state(CONFIG).next == ("ask_user",)
    lead_system = llms.agents["lead"].calls[0][0]
    assert isinstance(lead_system, SystemMessage) and "Nearest: Skoda Madhapur" in lead_system.content
    assert "locator" in graph.get_state(CONFIG).values["agent_outputs"]  # already merged while paused

    state = graph.invoke(Command(resume={"approved": True}), CONFIG)
    assert len(llms.agents["locator"].calls) == 2  # not run again after the Confirm
    assert list(state["agent_outputs"]) == ["locator", "lead"]
    assert ai_texts(state)[-1] == "Booked near you."


def test_reply_order_is_fixed_whatever_order_agents_finish():
    assert list(in_reply_order({"locator": 1, "lead": 2, "car_info": 3, "quotation": 4})) == \
        ["car_info", "quotation", "locator", "lead"]


def test_failed_turn_restores_the_previous_agent_outputs():
    graph, llms = make(supervisor=[route(["quotation"], **KYLAQ_AT_HYD), route(["small_talk"])],
                       quotation=[AIMessage(content="quote")], composer=[AIMessage(content="quote reply")])
    ask(graph, "Kylaq Signature AT price in Hyderabad")
    before = graph.get_state(CONFIG).values["agent_outputs"]

    class Down:
        def invoke(self, _messages):
            raise LLMUnavailableError("composer", 4, DEADLINE)

    llms.composer.invoke = Down().invoke
    assert run_turn(graph, CONFIG, {"messages": [HumanMessage("Thanks")]}).status == "error"
    assert graph.get_state(CONFIG).values["agent_outputs"] == before


def test_conversations_saved_before_the_change_still_load():
    old = {"messages": [], "queue": ["quotation"], "active_agent": "quotation", "agent_messages": [],
           "slots": {"city": "Pune"}}
    assert deserialize_state(old) == {"messages": [], "slots": {"city": "Pune"}}
