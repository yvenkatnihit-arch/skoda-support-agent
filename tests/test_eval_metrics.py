import json

from app.agents import CATALOG
from evals import metrics
from evals.judge import Claim, Verdict, faithfulness
from evals.run_evals import load_cases, percentile

QUOTE_OUTPUTS = {"quotation": {"answer": "x", "status": "complete", "tool_results": [
    {"tool": "calculate_on_road_price", "args": {}, "content": '{"estimated_on_road_price": 1292125}'}]}}
SEARCH_OUTPUTS = {"car_info": {"answer": "x", "status": "complete", "tool_results": [
    {"tool": "search_brochures", "args": {}, "content": json.dumps({"results": [
        {"source": "brochure/kushaq-brochure-sample.md", "section": "Škoda Kushaq > Safety", "text": "6 airbags"}]})}]}}


def test_eval_set_is_valid_and_covers_every_intent():
    cases = load_cases()
    assert len(cases) >= 40
    intents = {i for c in cases for opt in (c.get("expected_intents_any") or [c["expected_intents"]]) for i in opt}
    assert intents == set(CATALOG.intents)  # every agent's intent (incl. new agents) must have eval cases
    categories = {c["category"] for c in cases}
    assert {"multi_intent", "follow_up"} <= categories
    assert any(c.get("expect_clarify") for c in cases) and any(c.get("expect_pause") for c in cases)
    assert any(c.get("should_refuse") for c in cases)


def test_routing_needs_same_intents_and_same_agents_in_any_order():
    case = {"expected_intents": ["quotation", "locator"], "expected_agents": ["quotation", "locator"]}
    assert metrics.routing(case, ["locator", "quotation"], ["locator", "quotation"])["pass"]  # parallel: any order
    assert not metrics.routing(case, ["quotation", "locator"], ["quotation"])["pass"]
    assert not metrics.routing(case, ["quotation"], ["quotation", "locator"])["pass"]
    any_case = {"expected_intents_any": [["car_info"], ["car_info", "quotation"]]}
    assert metrics.routing(any_case, ["quotation", "car_info"], [])["pass"]


def test_slots_compare_case_insensitively_and_normalise_phone():
    case = {"expected_slots": {"city": "Hyderabad", "phone": "9123456780", "variant": "Signature"}}
    r = metrics.slots(case, {"city": "hyderabad", "phone": "+91 91234 56780", "variant": "Signature+"})
    assert r["correct"] == 2 and list(r["wrong"]) == ["variant"]
    assert metrics.slots({}, {}) is None


def test_tool_use_alternatives_and_empty_expectation():
    assert metrics.tool_use({"expected_tools": ["calculate_on_road_price"]}, QUOTE_OUTPUTS)["pass"]
    assert metrics.tool_use({"expected_tools": ["get_variant_details|calculate_on_road_price"]}, QUOTE_OUTPUTS)["pass"]
    assert not metrics.tool_use({"expected_tools": ["calculate_range"]}, QUOTE_OUTPUTS)["pass"]
    assert not metrics.tool_use({"expected_tools": []}, QUOTE_OUTPUTS)["pass"]  # no tool should have run
    assert metrics.tool_use({}, QUOTE_OUTPUTS) is None


def test_flow_checks_clarify_and_pause_both_ways():
    assert metrics.flow({"expect_clarify": True}, ["supervisor", "clarify"], False)["pass"]
    assert not metrics.flow({}, ["supervisor", "clarify"], False)["pass"]       # clarify not expected
    assert not metrics.flow({"expect_pause": True}, ["supervisor", "lead"], False)["pass"]


def test_retrieval_hit_matches_source_and_section():
    case = {"expected_sources": [{"source": "brochure/kushaq-brochure-sample.md", "section": "safety"}]}
    assert metrics.retrieval_hit(case, SEARCH_OUTPUTS)["hit"]
    miss = {"expected_sources": [{"source": "brochure/kushaq-brochure-sample.md", "section": "Comfort"}]}
    assert not metrics.retrieval_hit(miss, SEARCH_OUTPUTS)["hit"]
    assert not metrics.retrieval_hit(case, QUOTE_OUTPUTS)["hit"]  # search never ran


def test_key_facts_normalise_numbers_and_respect_boundaries():
    reply = "On-road about ₹12,92,125 (estimate). Top speed 250 km/h, 6 airbags."
    assert metrics.key_facts({"key_facts": ["12,92,125", "estimate"]}, reply)["pass"]
    assert metrics.key_facts({"key_facts": ["1,292,125"]}, reply)["pass"]           # western commas too
    assert metrics.key_facts({"key_facts": ["six airbags|6 airbags"]}, reply)["pass"]
    assert not metrics.key_facts({"key_facts": ["25"]}, reply)["pass"]               # 25 is not 250
    assert metrics.key_facts({"key_facts": ["44,414"]}, reply)["missing"] == ["44,414"]


def test_invented_figures_reuse_the_guardrail_check():
    assert metrics.invented_figures("It costs ₹12,92,125", QUOTE_OUTPUTS) == []
    assert metrics.invented_figures("It costs ₹9,99,999", QUOTE_OUTPUTS) == ["₹9,99,999"]


def test_faithfulness_score():
    assert faithfulness(Verdict(claims=[], declined=False)) == 1.0
    v = Verdict(claims=[Claim(claim="a", supported=True), Claim(claim="b", supported=False)], declined=False)
    assert faithfulness(v) == 0.5


def test_p90_uses_nearest_rank_and_is_never_below_the_median():
    assert percentile([11.8, 22.0], 90) == 22.0                      # was reported as 11.8 (the fastest)
    assert percentile([13.2, 14.1, 14.9, 16.5], 90) == 16.5          # was reported as 14.9
    assert percentile(list(range(1, 51)), 90) == 45                  # 45th of 50
    assert percentile([7.0], 90) == 7.0
