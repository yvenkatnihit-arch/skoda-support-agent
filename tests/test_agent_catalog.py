"""The agent catalog and tool registry: agents come from their own files, tools say where they belong, and a
new agent works end to end from one AgentSpec plus one register() line. Offline, scripted fake LLMs."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from app.agents import CATALOG, RESERVED_INTENTS, CatalogError, build_catalog
from app.agents.base import AgentSpec
from app.graph.build_graph import GraphLLMs, build_graph
from app.graph.nodes.supervisor import router_output_model
from app.prompts.supervisor import SUPERVISOR_PROMPT
from app.tools import registry
from app.tools.registry import TOOL_REGISTRY, RegistryError, ToolEntry, register, tools_for
from tests.test_graph import CONFIG, KYLAQ_AT_HYD, ScriptedLLM, call

FINANCE = AgentSpec(name="finance", intent="finance", role="car finance specialist",
                    intent_help='EMI, car loans, down payment. Examples: "EMI for a Kylaq?"',
                    prompt="Your job: estimate EMIs with emi_estimate.", reply_order=25,
                    status_text="Working out the EMI...", required_slots=("model",))


@tool
def emi_estimate(model: str) -> str:
    """Estimate a monthly EMI for a Skoda model over 5 years with 20% down payment (test tool)."""
    return json.dumps({"model": model, "emi_per_month": 18500})


@pytest.fixture
def with_finance_tool(monkeypatch):
    monkeypatch.setitem(TOOL_REGISTRY, "emi_estimate", ToolEntry(emi_estimate, ("finance",), False, False))


# ---- the real catalog --------------------------------------------------------------

def test_catalog_finds_every_agent_file_in_reply_order():
    assert CATALOG.names == ("car_info", "quotation", "service", "locator", "lead", "support")
    assert CATALOG.action_agents == ("lead", "support")
    assert CATALOG.workers == ("car_info", "quotation", "service", "locator")
    assert CATALOG.intent_to_agent["service_info"] == "service"
    assert set(CATALOG.intents) == {"car_info", "quotation", "service_info", "locator", "lead_capture",
                                    "support_request", *RESERVED_INTENTS}


def test_action_agents_run_in_reply_order_one_after_another():
    intents = ["support_request", "locator", "lead_capture"]
    assert CATALOG.agents_for(intents) == ["locator", "lead", "support"]
    assert CATALOG.next_action_agent(intents) == "lead"
    assert CATALOG.next_action_agent(intents, after="lead") == "support"
    assert CATALOG.next_action_agent(intents, after="support") is None
    assert CATALOG.next_action_agent(["support_request"]) == "support"


def test_tools_come_from_the_registry_unchanged():
    assert CATALOG.agents["car_info"].tool_names == {"search_brochures", "get_variant_details"}
    assert CATALOG.agents["locator"].tool_names == {"geocode_location", "find_nearest_dealers"}
    assert CATALOG.agents["lead"].tool_names == {"save_lead", "book_test_drive"}
    assert CATALOG.agents["support"].tool_names == {"create_ticket", "get_ticket_status", "search_service_docs",
                                                    "find_nearest_dealers", "geocode_location"}
    assert registry.TOOLS_REQUIRING_APPROVAL == {"save_lead", "book_test_drive", "create_ticket"}
    assert not any(CATALOG.agents[w].has_action_tools for w in CATALOG.workers)


def test_supervisor_prompt_and_output_use_the_catalog_intents():
    prompt = SUPERVISOR_PROMPT.format(intents=CATALOG.intents_help(), models="Kylaq", slots="{}",
                                      last_assistant="", summary="")
    for spec in CATALOG.agents.values():
        assert f"- {spec.intent}: " in prompt
    output = router_output_model(CATALOG.intents)
    assert output(intents=["quotation"]).intents == ["quotation"]
    with pytest.raises(ValueError):
        output(intents=["finance"])  # not an intent until a finance agent exists


# ---- registry rules ----------------------------------------------------------------

def test_register_rejects_duplicates_shared_actions_and_homeless_tools(monkeypatch):
    monkeypatch.setattr(registry, "TOOL_REGISTRY", dict(TOOL_REGISTRY))
    with pytest.raises(RegistryError, match="twice"):
        register(registry.search_brochures, agents=["car_info"])
    with pytest.raises(RegistryError, match="can't be shared"):
        register(emi_estimate, shared=True, action=True)
    with pytest.raises(RegistryError, match="agents="):
        register(emi_estimate)


def test_a_shared_tool_goes_to_every_agent(monkeypatch):
    monkeypatch.setitem(TOOL_REGISTRY, "emi_estimate", ToolEntry(emi_estimate, (), True, False))
    assert all("emi_estimate" in {t.name for t in tools_for(a)} for a in CATALOG.names)


# ---- startup checks ----------------------------------------------------------------

def _spec(**kw):
    base = dict(name="finance", intent="finance", role="r", intent_help="h", prompt="p", reply_order=25)
    return AgentSpec(**{**base, **kw})


def test_catalog_refuses_broken_setups(with_finance_tool, monkeypatch):
    existing = list(CATALOG.agents.values())
    problems = {
        "is used twice": [*existing, _spec(name="quotation")],
        "belongs to two agents": [*existing, _spec(intent="quotation")],
        "is reserved": [*existing, _spec(intent="small_talk")],
        "unknown required slots": [*existing, _spec(required_slots=("salary",))],
        "is a built-in graph node": [*existing, _spec(name="composer")],
    }
    for message, specs in problems.items():
        with pytest.raises(CatalogError, match=message):
            build_catalog(specs)
    with pytest.raises(CatalogError, match="has no tools"):
        build_catalog([*existing, _spec(name="no_tools", intent="no_tools")])
    with pytest.raises(CatalogError, match="unknown agent 'finance'"):
        build_catalog(existing)  # the registry gives emi_estimate to an agent that doesn't exist
    monkeypatch.setitem(TOOL_REGISTRY, "emi_estimate", ToolEntry(emi_estimate, ("finance",), False, True))
    assert build_catalog([*existing, FINANCE]).action_agents == ("finance", "lead", "support")  # reply order;
    # an action tool makes it an action agent: it runs after the parallel group, one at a time


# ---- a brand-new agent, end to end ------------------------------------------------

def test_a_new_agent_from_one_spec_runs_in_parallel_and_takes_its_reply_slot(with_finance_tool):
    catalog = build_catalog([*CATALOG.agents.values(), FINANCE])
    assert catalog.names == ("car_info", "quotation", "finance", "service", "locator", "lead", "support")
    assert "finance" in catalog.workers and catalog.action_agents == ("lead", "support")
    assert [t.name for t in catalog.agents["finance"].tools] == ["emi_estimate"]

    routed = router_output_model(catalog.intents)(intents=["finance", "quotation"], slots=KYLAQ_AT_HYD)
    llms = GraphLLMs(
        supervisor=ScriptedLLM(routed), clarify=ScriptedLLM(), summarize=ScriptedLLM(),
        composer=ScriptedLLM(AIMessage(content="About ₹12,92,125; EMI about ₹18,500 a month.")),
        agents={a: ScriptedLLM() for a in catalog.names},
    )
    llms.agents["finance"] = ScriptedLLM(call("emi_estimate", {"model": "Kylaq"}, "f1"), AIMessage(content="EMI 18500"))
    llms.agents["quotation"] = ScriptedLLM(call("calculate_on_road_price", KYLAQ_AT_HYD, "q1"),
                                           AIMessage(content="1292125"))
    graph = build_graph(llms, catalog=catalog)
    state = graph.invoke({"messages": [HumanMessage("Kylaq Signature AT price in Hyderabad and the EMI?")]}, CONFIG)

    assert state["agent_outputs"]["finance"]["tool_results"][0]["tool"] == "emi_estimate"
    assert state["agent_outputs"]["finance"]["status"] == "complete"
    assert "EMI 18500" in llms.composer.calls[0][0].content
    findings = llms.composer.calls[0][0].content
    assert findings.index('"quotation"') < findings.index('"finance"')  # reply_order 20 before 25
    supervisor_prompt = llms.supervisor.calls[0][0]
    assert isinstance(supervisor_prompt, SystemMessage) and "- finance: EMI, car loans" in supervisor_prompt.content
    assert state["messages"][-1].text == "About ₹12,92,125; EMI about ₹18,500 a month."


def test_a_new_agents_required_slot_triggers_clarify(with_finance_tool):
    catalog = build_catalog([*CATALOG.agents.values(), FINANCE])
    routed = router_output_model(catalog.intents)(intents=["finance"])
    llms = GraphLLMs(supervisor=ScriptedLLM(routed), clarify=ScriptedLLM(AIMessage(content="Which model?")),
                     composer=ScriptedLLM(), summarize=ScriptedLLM(), agents={a: ScriptedLLM() for a in catalog.names})
    state = build_graph(llms, catalog=catalog).invoke({"messages": [HumanMessage("What's the EMI?")]}, CONFIG)
    assert state["messages"][-1].text == "Which model?" and llms.agents["finance"].calls == []
