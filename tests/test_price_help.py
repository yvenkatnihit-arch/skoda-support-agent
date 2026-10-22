"""Price questions never dead-end (from a live chat where "what is the price" for the Slavia looped on "which
variant?"): a model is enough, everyday words become real trims, and clarify shows the choices. Offline."""

import json

from langchain_core.messages import AIMessage

from app.graph.car_slots import merge_slots
from app.graph.nodes.clarify import choices_for
from app.graph.slots import CarMention, Slots
from app.tools.data_tools import get_variant_details
from tests.test_graph import ai_texts, ask, call, make, route


def slavia(variant=None, transmission=None, prior=None):
    return merge_slots(prior or {}, Slots(cars=[CarMention(model="Slavia", variant=variant, transmission=transmission)]))


def test_everyday_words_become_real_trims():
    variants = json.loads(get_variant_details.invoke({"model": "Slavia"}))["variants"]
    cheapest = min(variants, key=lambda v: v["ex_showroom_price"])
    dearest = max(variants, key=lambda v: v["ex_showroom_price"])
    base = slavia("base")
    assert (base["variant"], base["transmission"]) == (cheapest["name"], cheapest["transmission"])
    top = slavia("top")
    assert (top["variant"], top["transmission"]) == (dearest["name"], dearest["transmission"])
    assert slavia("fully loaded")["variant"] == dearest["name"]
    assert slavia("top", "AT")["transmission"] == "AT"          # top within the gearbox asked for
    assert slavia("latest")["variant"] is None                  # the whole current lineup
    assert slavia("prestige")["variant"] == "Prestige"          # a real trim always wins


def test_screenshot_conversation_gets_the_price_list_not_another_question():
    """"what is the price" right after talking about the Slavia: no clarify, the variants and starting price."""
    graph, llms = make(
        supervisor=[route(["car_info"], model="Slavia"), route(["quotation"])],
        car_info=[AIMessage(content="The Slavia is a mid-size sedan")],
        quotation=[call("get_variant_details", {"model": "Slavia"}, "v1"),
                   AIMessage(content="Slavia starts at 1099000 (Classic MT); variants listed")],
        composer=[AIMessage(content="The Škoda Slavia is a mid-size sedan."),
                  AIMessage(content="The Slavia starts at ₹10,99,000. Which variant, and which city?")],
    )
    ask(graph, "tell me about the Slavia")
    state = ask(graph, "what is the price")
    assert llms.clarify.calls == []                              # no "which variant?" dead end
    results = state["agent_outputs"]["quotation"]["tool_results"]
    assert [r["tool"] for r in results] == ["get_variant_details"]
    assert "starting_ex_showroom_price" in results[0]["content"]
    assert "Model, no variant" in llms.agents["quotation"].calls[0][0].content  # the agent knows what to do
    assert ai_texts(state)[-1].startswith("The Slavia starts at")


def test_clarify_shows_the_choices():
    assert "Models on sale:" in choices_for({"quotation": ["model"]}, {})[0]
    assert "Cities with on-road price data:" in choices_for({"x": ["city"]}, {})[0]
    assert "pincode" in choices_for({"locator": ["area, pincode or city"]}, {})[0]
    variants = choices_for({"x": ["variant"]}, slavia())
    assert variants and variants[0].startswith("Slavia variants:") and '"base"' in variants[0]


def test_car_info_can_list_variants():
    from app.agents import CATALOG

    assert "get_variant_details" in CATALOG.agents["car_info"].tool_names
    assert "what is a variant" in CATALOG.agents["car_info"].prompt.lower()
