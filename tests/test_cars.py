"""A list of cars in the slots (ROADMAP section 12): two cars in one question, comparisons, follow-ups, what each
agent sees, and clarify per car. Offline, scripted fake LLMs."""

import json

from langchain_core.messages import AIMessage, HumanMessage

from app.graph.car_slots import agent_view, cars_for, merge_slots, normalize
from app.graph.nodes.router import route_after_supervisor
from app.graph.nodes.supervisor import missing_slots
from app.graph.slots import CarMention, Slots
from app.tools.data_tools import get_variant_details
from tests.test_graph import CONFIG, ai_texts, ask, call, make, route


def two_questions():
    """Case A: "Kushaq Prestige DSG price in Bengaluru, and does the Slavia have a sunroof?" """
    return merge_slots({}, Slots(city="Bengaluru", cars=[
        CarMention(model="Kushaq", variant="Prestige", transmission="DSG", intents=["quotation"]),
        CarMention(model="Slavia", intents=["car_info"])]),
        "Kushaq Prestige DSG price in Bengaluru, and does the Slavia have a sunroof?")


def comparison():
    """Case B: "Is the Kylaq cheaper than the Kushaq in Hyderabad?" """
    return merge_slots({}, Slots(city="Hyderabad", cars=[CarMention(model="Kylaq"), CarMention(model="kushaq")]),
                       "Is the Kylaq cheaper than the Kushaq in Hyderabad?")


# ---- merging ---------------------------------------------------------------------

def test_two_cars_for_two_different_questions():
    s = two_questions()
    assert [c["model"] for c in s["cars"]] == ["Kushaq", "Slavia"] and s["active_cars"] == ["Kushaq", "Slavia"]
    assert [c["model"] for c in cars_for(s, "quotation")] == ["Kushaq"]
    assert [c["model"] for c in cars_for(s, "car_info")] == ["Slavia"]
    assert s["city"] == "Bengaluru"  # shared: one copy


def test_two_cars_in_one_comparison():
    s = comparison()
    assert s["active_cars"] == ["Kylaq", "Kushaq"]  # canonical names
    assert [c["model"] for c in cars_for(s, "quotation")] == ["Kylaq", "Kushaq"]


def test_follow_ups_keep_the_right_cars():
    s = comparison()
    mumbai = merge_slots(s, Slots(city="Mumbai"), "and in Mumbai?")
    assert mumbai["active_cars"] == ["Kylaq", "Kushaq"] and mumbai["city"] == "Mumbai"  # both re-priced

    one = merge_slots({}, Slots(cars=[CarMention(model="Kylaq", variant="Signature", transmission="MT")],
                                city="Hyderabad"), "Kylaq Signature MT price in Hyderabad")
    auto = merge_slots(one, Slots(cars=[CarMention(transmission="AT")]), "and the automatic?")
    assert auto["model"] == "Kylaq" and auto["transmission"] == "AT" and auto["variant"] == "Signature"

    kodiaq = merge_slots(auto, Slots(cars=[CarMention(model="Kodiaq", transmission="AT")]), "what about the Kodiaq?")
    assert kodiaq["active_cars"] == ["Kodiaq"] and kodiaq["transmission"] is None  # copied gearbox dropped
    assert {c["model"] for c in kodiaq["cars"]} == {"Kylaq", "Kodiaq"}  # the Kylaq is still remembered


def test_only_three_cars_are_remembered_and_old_flat_slots_still_load():
    s = comparison()
    for model in ("Slavia", "Kodiaq"):
        s = merge_slots(s, Slots(cars=[CarMention(model=model)]), f"what about the {model}?")
    assert [c["model"] for c in s["cars"]] == ["Kushaq", "Slavia", "Kodiaq"]
    old = normalize({"model": "Kylaq", "variant": "Signature", "city": "Pune"})  # saved before this change
    assert old["cars"] == [{"model": "Kylaq", "variant": "Signature", "transmission": None, "fuel": None}]
    assert old["active_cars"] == ["Kylaq"]


# ---- what agents see; clarify ------------------------------------------------------

def test_each_agent_sees_only_its_cars():
    s = two_questions()
    assert agent_view(s, "quotation") == {"city": "Bengaluru", "model": "Kushaq", "variant": "Prestige",
                                          "transmission": "DSG"}
    assert agent_view(s, "car_info") == {"city": "Bengaluru", "model": "Slavia"}
    view = agent_view(comparison(), "quotation")
    assert "model" not in view and [c["model"] for c in view["cars"]] == ["Kylaq", "Kushaq"]
    sends = route_after_supervisor({"intents": ["quotation", "car_info"], "slots": s, "messages": [], "summary": ""})
    known = {send.node: send.arg["known_slots"] for send in sends}
    assert known["quotation"]["model"] == "Kushaq" and known["car_info"]["model"] == "Slavia"


def test_clarify_is_per_car_and_comparisons_need_no_trim():
    one = merge_slots({}, Slots(cars=[CarMention(model="Kylaq")]), "Kylaq price")
    assert missing_slots(["quotation"], one) == {}  # no trim or city: the agent shows the price list instead
    assert missing_slots(["quotation"], merge_slots({}, Slots(city="Pune"), "price?")) == {"quotation": ["model"]}
    assert missing_slots(["quotation"], comparison()) == {}  # starting prices, no clarify
    a = two_questions()
    assert missing_slots(["quotation", "car_info"], a) == {}  # the Kushaq has its trim and city
    no_trim = merge_slots({}, Slots(city="Pune", cars=[CarMention(model="Kushaq", intents=["quotation"]),
                                                       CarMention(model="Slavia", intents=["car_info"])]), "x")
    assert missing_slots(["quotation", "car_info"], no_trim) == {}


def test_price_list_gives_the_starting_price():
    out = json.loads(get_variant_details.invoke({"model": "Kylaq"}))
    assert out["starting_ex_showroom_price"] == min(v["ex_showroom_price"] for v in out["variants"])
    assert out["starting_variant"]


# ---- end to end ----------------------------------------------------------------------

def test_a_comparison_without_trims_goes_straight_to_starting_prices():
    graph, llms = make(
        supervisor=[route(["quotation"], city="Hyderabad", cars=[CarMention(model="Kylaq"), CarMention(model="Kushaq")])],
        quotation=[AIMessage(content="", tool_calls=[
                       {"name": "get_variant_details", "args": {"model": "Kylaq"}, "id": "v1", "type": "tool_call"},
                       {"name": "get_variant_details", "args": {"model": "Kushaq"}, "id": "v2", "type": "tool_call"}]),
                   AIMessage(content="Kylaq starts at 759000, Kushaq at 1069000")],
        composer=[AIMessage(content="The Kylaq starts lower: ₹7,59,000 vs ₹10,69,000 ex-showroom.")],
    )
    state = ask(graph, "Is the Kylaq cheaper than the Kushaq in Hyderabad?")
    assert llms.clarify.calls == []  # no "which variant?"
    assert [r["tool"] for r in state["agent_outputs"]["quotation"]["tool_results"]] == ["get_variant_details"] * 2
    assert ai_texts(state)[-1] == "The Kylaq starts lower: ₹7,59,000 vs ₹10,69,000 ex-showroom."
    request = llms.agents["quotation"].calls[0][0].content
    assert '"cars"' in request and "Kushaq" in request  # the agent was told about both cars
