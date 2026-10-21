"""Graph tests with scripted fake LLMs (no API calls). The real tools run on the real seed data."""

import csv

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph import END
from langgraph.types import Command

from app.graph.build_graph import AGENT_NAMES, GraphLLMs, build_graph, recursion_limit
from app.graph.nodes.agent_factory import CANCELLED, patch_unanswered_tool_calls
from app.graph.nodes.approval import CANCELLED_BY_CUSTOMER, parse_decision
from app.graph.nodes.guardrail import SAFE_REPLY, extract_figures, unsupported_figures
from app.graph.nodes.router import after_agents, after_guardrail, agent_router, route_after_supervisor
from app.graph.nodes.save_output import save_output
from app.graph.nodes.summarize import make_summarize, split_for_summary
from app.graph.nodes.supervisor import RouterOutput, Slots, agents_for, merge_slots
from app.memory.file_store import InvalidThreadId, load_state, restore_if_missing, save_state
from app.tools import lead_tools


class ScriptedLLM:
    """Returns the scripted responses in order and records every request. Fails loudly if called too often."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("LLM called more times than scripted")
        return self.responses.pop(0)


def call(name, args, id_):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id_, "type": "tool_call"}])


def route(intents, **slots):
    return RouterOutput(intents=intents, slots=Slots(**slots))


def make(supervisor=(), clarify=(), composer=(), summarize=(), **agents):
    llms = GraphLLMs(
        supervisor=ScriptedLLM(*supervisor), clarify=ScriptedLLM(*clarify),
        composer=ScriptedLLM(*composer), summarize=ScriptedLLM(*summarize),
        agents={a: ScriptedLLM(*agents.get(a, ())) for a in AGENT_NAMES},
    )
    return build_graph(llms), llms


CONFIG = {"configurable": {"thread_id": "t1"}, "recursion_limit": recursion_limit(5)}
KYLAQ_AT_HYD = {"model": "Kylaq", "variant": "Signature", "transmission": "AT", "city": "Hyderabad"}


def ask(graph, text, config=CONFIG):
    return graph.invoke({"messages": [HumanMessage(content=text)]}, config)


def ai_texts(state):
    return [m.text for m in state["messages"] if isinstance(m, AIMessage)]


@pytest.fixture
def leads_file(monkeypatch, tmp_path):
    path = tmp_path / "leads.csv"
    monkeypatch.setattr(lead_tools, "_leads_path", lambda: path)
    return path


# ---- supervisor helpers ----------------------------------------------------------

def test_agents_for_dedupes_and_puts_lead_last():
    assert agents_for(["lead_capture", "service_info", "quotation", "service_info"]) == ["service", "quotation", "lead"]
    assert agents_for(["small_talk"]) == [] and agents_for(["out_of_scope", "car_info"]) == ["car_info"]


def test_merge_slots_overwrites_keeps_and_normalises():
    old = {"model": "Kylaq", "variant": "Signature", "city": "Hyderabad", "transmission": "MT"}
    # same car: only what changed is replaced
    same = merge_slots(old, Slots(transmission="AT", model="kylaq"))
    assert same["transmission"] == "AT" and same["variant"] == "Signature" and same["city"] == "Hyderabad"
    # another car: the old car's trim and gearbox don't carry over (Octavia RS has no "Signature"); city stays
    other = merge_slots(old, Slots(model="octavia", phone="+91 98765 43210"))
    assert other["model"] == "Octavia RS" and other["variant"] is None and other["transmission"] is None
    assert other["city"] == "Hyderabad" and other["phone"] == "9876543210"
    # another car with its own details in the same message keeps those
    both = merge_slots(old, Slots(model="Slavia", variant="Prestige", transmission="DSG"))
    assert (both["variant"], both["transmission"]) == ("Prestige", "DSG")
    assert merge_slots({"phone": "9876543210"}, Slots(phone="12345"))["phone"] == "9876543210"  # invalid dropped


def test_merge_slots_ignores_placeholders_and_uses_official_variant_names():
    old = {"model": "Kylaq", "variant": "Signature", "city": "Hyderabad"}
    assert merge_slots(old, Slots(variant="null", city="None"))["variant"] == "Signature"   # Gemini's "null" text
    assert merge_slots(old, Slots(variant="null", city="None"))["city"] == "Hyderabad"
    assert merge_slots({}, Slots(model="Kylaq", variant="Signature Plus"))["variant"] == "Signature+"
    assert merge_slots({}, Slots(model="Kylaq", variant="signature +"))["variant"] == "Signature+"
    assert merge_slots({}, Slots(model="Kylaq", variant="Turbo"))["variant"] == "Turbo"  # unknown: kept, tool lists options


def test_switching_car_drops_details_gemini_copied_from_the_old_car():
    old = {"model": "Kylaq", "variant": "Signature", "transmission": "AT", "city": "Hyderabad"}
    copied = Slots(model="Kodiaq", transmission="AT")  # Gemini repeated the old gearbox
    merged = merge_slots(old, copied, "what about the Kodiaq?")  # the "at" in "what" isn't a gearbox
    assert merged["model"] == "Kodiaq" and merged["transmission"] is None and merged["city"] == "Hyderabad"
    said = merge_slots(old, copied, "what about the Kodiaq automatic?")
    assert said["transmission"] == "AT"
    same_car = merge_slots(old, Slots(transmission="MT"), "and the manual?")
    assert same_car["transmission"] == "MT" and same_car["variant"] == "Signature"
    # the first car in a conversation isn't a switch: Gemini's extracted details are kept as they are
    first = merge_slots({}, Slots(model="Kylaq", variant="Signature", transmission="AT"), "price and showroom")
    assert (first["variant"], first["transmission"]) == ("Signature", "AT")


# ---- routing functions -----------------------------------------------------------

def _targets(route):
    return [s.node for s in route] if isinstance(route, list) else route


def test_router_sends_to_all_parallel_agents_or_clarify_lead_composer():
    assert route_after_supervisor({"intents": ["quotation"], "slots": {"model": "Slavia"}}) == "clarify"
    assert _targets(route_after_supervisor({"intents": ["quotation"], "slots": KYLAQ_AT_HYD})) == ["quotation"]
    loc = {"intents": ["locator"]}
    assert route_after_supervisor({**loc, "slots": {}}) == "clarify"
    assert route_after_supervisor({**loc, "slots": {"lat": 17.4}}) == "clarify"   # lat without long isn't enough
    assert _targets(route_after_supervisor({**loc, "slots": {"lat": 17.4, "long": 78.4}})) == ["locator"]
    assert _targets(route_after_supervisor({**loc, "slots": {"city": "Mumbai"}})) == ["locator"]
    assert route_after_supervisor({"intents": ["small_talk"], "slots": {}}) == "composer"
    assert route_after_supervisor({"intents": ["lead_capture"], "slots": {}}) == "lead"
    # several intents: every non-lead agent at once, each with its own input; lead waits for after_agents
    both = {"intents": ["lead_capture", "locator", "car_info"], "slots": {"city": "Pune"},
            "messages": [HumanMessage("q")], "summary": ""}
    sends = route_after_supervisor(both)
    assert _targets(sends) == ["locator", "car_info"]
    assert sends[0].arg["agent"] == "locator" and sends[0].arg["known_slots"] == {"city": "Pune"}
    assert after_agents(both) == "lead" and after_agents({"intents": ["locator"]}) == "composer"
    # any missing slot -> clarify first and no agent runs (locator is fine, quotation lacks the variant)
    assert route_after_supervisor({"intents": ["quotation", "locator"],
                                   "slots": {"model": "Kylaq", "city": "Pune"}}) == "clarify"


def test_agent_router_and_tool_round_cap():
    tool = call("get_variant_details", {"model": "Kylaq"}, "c1")
    assert agent_router({"action_messages": [tool]}) == "approval"
    assert agent_router({"action_messages": [AIMessage(content="done")]}) == "save_output"
    five = [call("get_variant_details", {"model": "Kylaq"}, f"c{i}") for i in range(5)]
    six = [*five, call("get_variant_details", {"model": "Kylaq"}, "c6")]
    assert agent_router({"action_messages": five}) == "approval"
    assert agent_router({"action_messages": six}) == "save_output"


def test_after_guardrail():
    assert after_guardrail({"guardrail_feedback": "bad", "messages": []}) == "composer"
    assert after_guardrail({"guardrail_feedback": None, "messages": [HumanMessage("x")] * 21}) == "summarize"
    assert after_guardrail({"guardrail_feedback": None, "messages": [HumanMessage("x")] * 20}) == END


def test_recursion_limit_covers_every_action_agent_loop_and_fixed_steps():
    assert recursion_limit(5) == 9 + 2 * 22 + 10  # two action agents (lead, support)


# ---- node helpers ----------------------------------------------------------------

def test_save_output_stores_answer_and_tool_results_and_clears():
    ai = call("calculate_range", {"model": "Kylaq"}, "c1")
    ai.id = "a1"
    tm = ToolMessage(content='{"range_km": 857}', tool_call_id="c1", name="calculate_range", id="t1")
    final = AIMessage(content="Range is 857 km", id="a2")
    out = save_output({"agent_outputs": {"locator": {"answer": "x"}}, "action_messages": [ai, tm, final],
                       "active_action": "lead"})
    assert list(out["agent_outputs"]) == ["lead"]  # only its own entry: the reducer keeps the others
    saved = out["agent_outputs"]["lead"]
    assert saved["answer"] == "Range is 857 km" and saved["status"] == "complete"
    assert saved["tool_results"] == [{"tool": "calculate_range", "args": {"model": "Kylaq"},
                                      "content": '{"range_km": 857}'}]
    assert [type(m) for m in out["action_messages"]] == [RemoveMessage] * 3


def test_unanswered_tool_calls_are_patched_in_the_request_only():
    stale = call("book_test_drive", {"model": "Kylaq"}, "c9")
    patched = patch_unanswered_tool_calls([stale])
    assert isinstance(patched[1], ToolMessage) and patched[1].content == CANCELLED


@pytest.mark.parametrize("decision,expected", [
    (True, (True, None)), ("yes", (True, None)), ("no", (False, None)), ("maybe", (False, None)),
    ({"approved": True}, (True, None)), ({"approved": False, "feedback": " later "}, (False, "later")),
    ({"approved": "yes"}, (False, None)), (None, (False, None)),
])
def test_parse_decision_only_clear_yes_approves(decision, expected):
    assert parse_decision(decision) == expected


def test_guardrail_figure_extraction_and_matching():
    outputs = {"quotation": {"tool_results": [{"content": '{"estimated_on_road_price": 1292125, "distance_km": 1.8,'
                                                          ' "range_km": 857}'}]}}
    reply = ("On-road ₹12,92,125 (about ₹12.92 lakh, Rs 12.92 lakh). Range 857 km, 19.05 kmpl. "
             "Showroom 1.8 km away. 6 airbags, 4 years.")
    kinds = [(f.kind, f.value) for f in extract_figures(reply)]
    assert ("money", 1292125) in kinds and ("km", 857) in kinds and ("km", 1.8) in kinds
    assert not any(v == 19.05 for _, v in kinds)  # kmpl is not a km figure
    assert unsupported_figures(reply, outputs) == []
    assert unsupported_figures("Only ₹13,50,000 and 5 km away", outputs) == ["₹13,50,000", "5 km"]
    assert unsupported_figures("Prices from ₹10 lakh", {}) == ["₹10 lakh"]


def test_summarize_cuts_at_the_second_last_question():
    msgs = []
    for i in range(11):
        msgs += [HumanMessage(content=f"q{i}", id=f"h{i}"), AIMessage(content=f"a{i}", id=f"a{i}")]
    older, kept = split_for_summary(msgs)
    assert kept[0].id == "h9" and len(older) == 18
    llm = ScriptedLLM(AIMessage(content="Customer asked about Kylaq prices."))
    out = make_summarize(llm)({"messages": msgs, "summary": ""})
    assert out["summary"] == "Customer asked about Kylaq prices."
    assert {m.id for m in out["messages"]} == {m.id for m in older}
    assert split_for_summary([HumanMessage("only one", id="x")])[0] == []
    # Gemini rejects a request with only a system message ("contents are required"): the transcript must be
    # sent as a normal message
    [system, user] = llm.calls[0]
    assert isinstance(system, SystemMessage) and isinstance(user, HumanMessage) and "q0" in user.content


def test_summary_failure_keeps_the_messages_and_does_not_fail_the_turn():
    from app.llm import LLMUnavailableError

    class Down:
        def invoke(self, _messages):
            raise LLMUnavailableError("summarize", 4, ValueError("contents are required."))

    msgs = [m for i in range(11) for m in (HumanMessage(f"q{i}", id=f"h{i}"), AIMessage(f"a{i}", id=f"a{i}"))]
    assert make_summarize(Down())({"messages": msgs, "summary": ""}) == {}
    assert make_summarize(ScriptedLLM(AIMessage(content="  ")))({"messages": msgs, "summary": ""}) == {}


def test_a_long_conversation_gets_summarized_and_keeps_working():
    graph, llms = make(supervisor=[route(["small_talk"])] * 11,
                       composer=[AIMessage(content=f"reply {i}") for i in range(11)],
                       summarize=[AIMessage(content="Customer chatted.")])
    for i in range(11):
        state = ask(graph, f"message {i}")
    assert state["summary"] == "Customer chatted."  # summarize ran once, after message 11 (22 messages)
    assert [m.text for m in state["messages"]] == ["message 9", "reply 9", "message 10", "reply 10"]


# ---- full graph runs -------------------------------------------------------------

def test_quotation_flow_end_to_end():
    graph, llms = make(
        supervisor=[route(["quotation"], **KYLAQ_AT_HYD)],
        quotation=[call("calculate_on_road_price", KYLAQ_AT_HYD, "c1"),
                   AIMessage(content="Estimated on-road price 1292125")],
        composer=[AIMessage(content="The Kylaq Signature AT is about ₹12,92,125 on-road in Hyderabad.")],
    )
    state = ask(graph, "On-road price of Kylaq Signature AT in Hyderabad")
    assert ai_texts(state) == ["The Kylaq Signature AT is about ₹12,92,125 on-road in Hyderabad."]
    saved = state["agent_outputs"]["quotation"]
    assert saved["status"] == "complete" and saved["tool_results"][0]["tool"] == "calculate_on_road_price"
    assert '"estimated_on_road_price": 1292125' in saved["tool_results"][0]["content"]
    assert state.get("action_messages", []) == [] and state["guardrail_feedback"] is None
    assert saved["seconds"] >= 0  # each worker times itself
    # system prompt + tools are only in the request, never in State
    assert isinstance(llms.agents["quotation"].calls[0][0], SystemMessage)
    assert not any(isinstance(m, SystemMessage) for m in state["messages"])


def test_missing_slot_goes_to_clarify_and_ends_the_turn():
    graph, llms = make(supervisor=[route(["quotation"], model="Slavia")],
                       clarify=[AIMessage(content="Which variant, and which city?")])
    state = ask(graph, "Price of Slavia")
    assert ai_texts(state) == ["Which variant, and which city?"]
    assert llms.composer.calls == [] and llms.agents["quotation"].calls == []


def test_follow_up_keeps_saved_slots_without_clarify():
    graph, llms = make(
        supervisor=[route(["quotation"], model="Kylaq", variant="Signature", transmission="MT", city="Hyderabad"),
                    route(["quotation"], transmission="AT")],
        quotation=[AIMessage(content="MT quote"), AIMessage(content="AT quote")],
        composer=[AIMessage(content="MT reply"), AIMessage(content="AT reply")],
    )
    ask(graph, "Kylaq Signature MT price in Hyderabad")
    state = ask(graph, "what about the AT version?")
    assert state["slots"]["transmission"] == "AT" and state["slots"]["city"] == "Hyderabad"
    assert llms.clarify.calls == [] and ai_texts(state)[-1] == "AT reply"


def test_lead_runs_after_the_parallel_agents():
    graph, llms = make(
        supervisor=[route(["lead_capture", "locator"], lat=17.4483, long=78.3915)],
        locator=[call("find_nearest_dealers", {"lat": 17.4483, "long": 78.3915, "dealer_type": "showroom"}, "c1"),
                 AIMessage(content="Nearest showroom 2.4 km")],
        lead=[AIMessage(content="Need name and phone to book")],
        composer=[AIMessage(content="The nearest showroom is 2.4 km away. Please share your name and number.")],
    )
    state = ask(graph, "Nearest showroom to me and book a test drive")
    assert list(state["agent_outputs"]) == ["locator", "lead"]
    assert ai_texts(state)[-1].startswith("The nearest showroom is 2.4 km away")


def test_small_talk_goes_straight_to_composer():
    graph, llms = make(supervisor=[route(["small_talk"])], composer=[AIMessage(content="Hi! How can I help?")])
    state = ask(graph, "Hi!")
    assert ai_texts(state) == ["Hi! How can I help?"] and state["agent_outputs"] == {}


def test_action_tool_pauses_and_confirm_runs_it(leads_file):
    args = {"name": "Asha", "phone": "9123456789", "city": "Hyderabad", "model": "Kylaq"}
    graph, llms = make(
        supervisor=[route(["lead_capture"], name="Asha", phone="9123456789", city="Hyderabad", model="Kylaq")],
        lead=[call("book_test_drive", args, "c1"), AIMessage(content="Test drive requested")],
        composer=[AIMessage(content="Done! The showroom will call you.")],
    )
    paused = ask(graph, "Book a Kylaq test drive")
    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value["actions"][0]["tool"] == "book_test_drive"
    assert graph.get_state(CONFIG).next == ("ask_user",)
    assert not leads_file.exists()  # nothing saved before the customer confirms

    state = graph.invoke(Command(resume={"approved": True}), CONFIG)
    [row] = list(csv.DictReader(leads_file.open(encoding="utf-8")))
    assert row["request_type"] == "test_drive" and row["phone"] == "9123456789"
    assert ai_texts(state)[-1] == "Done! The showroom will call you."
    assert state["agent_outputs"]["lead"]["tool_results"][0]["tool"] == "book_test_drive"


def test_cancel_adds_not_done_and_returns_to_the_agent(leads_file):
    args = {"name": "Asha", "phone": "9123456789", "city": "Hyderabad", "model": "Kylaq"}
    graph, llms = make(
        supervisor=[route(["lead_capture"], name="Asha", phone="9123456789", city="Hyderabad", model="Kylaq")],
        lead=[call("book_test_drive", args, "c1"), AIMessage(content="Customer cancelled")],
        composer=[AIMessage(content="No problem, I've not booked anything.")],
    )
    ask(graph, "Book a Kylaq test drive")
    state = graph.invoke(Command(resume={"approved": False, "feedback": "maybe next week"}), CONFIG)
    second_request = llms.agents["lead"].calls[1]
    assert any(isinstance(m, ToolMessage) and m.content == CANCELLED_BY_CUSTOMER for m in second_request)
    assert any(isinstance(m, HumanMessage) and "maybe next week" in m.text for m in second_request)
    assert not leads_file.exists()
    assert ai_texts(state)[-1] == "No problem, I've not booked anything."


def test_agent_cannot_use_another_agents_tool(leads_file):
    graph, llms = make(
        supervisor=[route(["car_info"], model="Kylaq")],
        car_info=[call("save_lead", {"name": "X", "phone": "9123456789", "city": "Pune"}, "c1"),
                  AIMessage(content="Kylaq has 6 airbags")],
        composer=[AIMessage(content="The Kylaq has 6 airbags.")],
    )
    state = ask(graph, "Is the Kylaq safe?")
    assert "not one of your tools" in llms.agents["car_info"].calls[1][-1].content
    assert not leads_file.exists() and graph.get_state(CONFIG).next == ()
    assert ai_texts(state)[-1] == "The Kylaq has 6 airbags."


def test_guardrail_fix_replaces_the_reply_with_the_same_id():
    graph, llms = make(
        supervisor=[route(["quotation"], **KYLAQ_AT_HYD)],
        quotation=[call("calculate_on_road_price", KYLAQ_AT_HYD, "c1"), AIMessage(content="1292125")],
        composer=[AIMessage(content="It costs ₹99,99,999."), AIMessage(content="It costs ₹12,92,125.")],
    )
    state = ask(graph, "Kylaq Signature AT price in Hyderabad")
    assert ai_texts(state) == ["It costs ₹12,92,125."]  # one reply, not two
    assert "rejected" in llms.composer.calls[1][0].content and "₹99,99,999" in llms.composer.calls[1][0].content
    assert state["guardrail_retries"] == 1 and state["guardrail_feedback"] is None


def test_guardrail_second_failure_gives_safe_reply():
    graph, _ = make(supervisor=[route(["small_talk"])],
                    composer=[AIMessage(content="Cars start at ₹5 lakh"), AIMessage(content="From ₹6 lakh")])
    state = ask(graph, "Hi")
    assert ai_texts(state) == [SAFE_REPLY]


def test_tool_round_cap_stops_the_agent_as_incomplete():
    lookups = [call("get_variant_details", {"model": "Kylaq"}, f"c{i}") for i in range(6)]
    graph, llms = make(supervisor=[route(["quotation"], **KYLAQ_AT_HYD)], quotation=lookups,
                       composer=[AIMessage(content="Sorry, I couldn't finish that. Want a callback?")])
    state = ask(graph, "Kylaq Signature AT price in Hyderabad")
    saved = state["agent_outputs"]["quotation"]
    assert saved["status"] == "incomplete" and len(saved["tool_results"]) == 5
    assert len(llms.agents["quotation"].calls) == 6


def test_new_turn_resets_per_turn_fields():
    graph, _ = make(supervisor=[route(["quotation"], **KYLAQ_AT_HYD), route(["small_talk"])],
                    quotation=[AIMessage(content="quote")],
                    composer=[AIMessage(content="quote reply"), AIMessage(content="You're welcome!")])
    ask(graph, "Kylaq Signature AT price in Hyderabad")
    state = ask(graph, "Thanks")
    assert state["agent_outputs"] == {} and state["intents"] == ["small_talk"]


# ---- conversation file store -----------------------------------------------------

def test_save_load_and_restore_after_restart(tmp_path):
    graph, _ = make(supervisor=[route(["small_talk"])], composer=[AIMessage(content="Hello!")])
    ask(graph, "Hi")
    values = graph.get_state(CONFIG).values
    save_state("t1", values, base_dir=tmp_path)
    loaded = load_state("t1", base_dir=tmp_path)
    assert [m.text for m in loaded["messages"]] == ["Hi", "Hello!"]

    # "restart": a new graph with an empty MemorySaver picks the conversation up from the JSON file
    graph2, llms2 = make(supervisor=[route(["small_talk"])], composer=[AIMessage(content="Hi again!")])
    assert restore_if_missing(graph2, CONFIG, base_dir=tmp_path) is True
    assert restore_if_missing(graph2, CONFIG, base_dir=tmp_path) is False  # already there
    state = ask(graph2, "Hey")
    assert [m.text for m in state["messages"]] == ["Hi", "Hello!", "Hey", "Hi again!"]
    assert any(isinstance(m, HumanMessage) and m.text == "Hi" for m in llms2.supervisor.calls[0])


def test_thread_id_cannot_escape_the_folder(tmp_path):
    for bad in ("../secrets", "a/b", "", "x" * 65):
        with pytest.raises(InvalidThreadId):
            save_state(bad, {}, base_dir=tmp_path)
