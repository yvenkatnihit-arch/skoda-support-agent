"""Observability: turn ids and turn summaries in the logs, and phone-masked LangSmith tracing (all offline)."""

import os

import pytest
import structlog
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tracers.langchain import LangChainTracer
from structlog.testing import capture_logs

from app.config import Settings
from app.graph.runner import run_turn
from app.llm import LoggedLLM
from app.observability import (current_stats, disable_auto_tracing, mask_trace, tracing_callbacks,
                               turn_context)
from tests.test_graph import CONFIG, KYLAQ_AT_HYD, call, leads_file, make, route  # noqa: F401 (fixture)

MERGE = [structlog.contextvars.merge_contextvars]


def settings(**kw) -> Settings:
    return Settings(_env_file=None, google_api_key="test", **kw)


# ---- part 1: turn logs -----------------------------------------------------------

def test_turn_context_tags_every_log_line_and_resets_afterwards():
    with capture_logs(MERGE) as logs, turn_context("thread-1") as (turn_id, stats):
        structlog.get_logger().info("something")
        assert current_stats() is stats
    assert logs[0]["turn_id"] == turn_id and logs[0]["thread_id"] == "thread-1"
    assert current_stats() is None


def test_logged_llm_adds_its_calls_and_tokens_to_the_turn():
    raw = AIMessage(content="ok", usage_metadata={
        "input_tokens": 100, "output_tokens": 40, "total_tokens": 140,
        "output_token_details": {"reasoning": 25}, "input_token_details": {"cache_read": 60}})

    class Once:
        def invoke(self, _messages):
            return raw

    llm = LoggedLLM(Once(), "composer", "gemini-2.5-flash", settings())
    with turn_context("t") as (_, stats):
        llm.invoke([HumanMessage("hi")])
        llm.invoke([HumanMessage("hi")])
    d = stats.as_dict()
    assert (d["gemini_calls"], d["input_tokens"], d["output_tokens"]) == (2, 200, 80)
    assert (d["thinking_tokens"], d["cached_input_tokens"]) == (50, 120)
    llm.invoke([HumanMessage("outside a turn")])  # no turn running: nothing to count, no error


def _turn_line(logs):
    lines = [e for e in logs if e["event"] == "turn"]
    assert len(lines) == 1
    return lines[0]


def test_turn_summary_for_a_price_question_has_names_and_times_but_no_text():
    graph, _ = make(
        supervisor=[route(["quotation"], **KYLAQ_AT_HYD)],
        quotation=[call("calculate_on_road_price", KYLAQ_AT_HYD, "c1"), AIMessage(content="1292125")],
        composer=[AIMessage(content="About ₹12,92,125 on-road in Hyderabad.")],
    )
    with capture_logs(MERGE) as logs:
        result = run_turn(graph, CONFIG, {"messages": [HumanMessage("Kylaq Signature AT price in Hyderabad")]})
    assert result.status == "reply"
    line = _turn_line(logs)
    assert line["status"] == "reply" and line["resumed"] is False and line["guardrail"] == "ok"
    assert line["nodes"][0] == "supervisor" and line["nodes"][-1] == "guardrail"
    assert line["intents"] == ["quotation"] and line["agents"] == ["quotation"]
    assert line["tools"] == ["calculate_on_road_price"]
    assert line["slots_filled"] == ["city", "model", "transmission", "variant"]
    assert set(line["node_seconds"]) == set(line["nodes"]) and line["total_s"] >= 0
    assert line["turn_id"] and line["thread_id"] == "t1"
    flat = str(line)
    assert "Hyderabad" not in flat and "12,92,125" not in flat  # no question/reply text or slot values


def test_turn_summary_for_clarify_and_for_a_pause(leads_file):  # noqa: F811
    graph, _ = make(supervisor=[route(["quotation"], model="Slavia")], clarify=[AIMessage(content="Which city?")])
    with capture_logs(MERGE) as logs:
        run_turn(graph, CONFIG, {"messages": [HumanMessage("Price of Slavia")]})
    line = _turn_line(logs)
    assert line["clarify"] is True and line["guardrail"] is None and line["agents"] == []

    args = {"name": "Asha", "phone": "9123456789", "city": "Hyderabad", "model": "Kylaq"}
    graph, _ = make(supervisor=[route(["lead_capture"], **args)], lead=[call("book_test_drive", args, "c1")])
    config = {**CONFIG, "configurable": {"thread_id": "t2"}}
    with capture_logs(MERGE) as logs:
        result = run_turn(graph, config, {"messages": [HumanMessage("Book a Kylaq test drive")]})
    line = _turn_line(logs)
    assert result.status == "paused" and line["status"] == "paused"
    assert line["awaiting"] == ["book_test_drive"]


# ---- part 2: LangSmith -----------------------------------------------------------

def test_tracing_is_off_by_default_and_needs_a_key_when_on():
    assert tracing_callbacks(settings()) == []
    with pytest.raises(RuntimeError, match="LANGSMITH_API_KEY"):
        tracing_callbacks(settings(langsmith_tracing=True))


def test_tracer_client_masks_phones_in_inputs_outputs_and_errors():
    [tracer] = tracing_callbacks(settings(langsmith_tracing=True, langsmith_api_key="test-key",
                                          langsmith_project="p"))
    assert isinstance(tracer, LangChainTracer) and tracer.project_name == "p"
    client = tracer.client
    inputs = {"messages": [{"type": "human", "content": "I'm Asha, call me on 91234 56780"}],
              "slots": {"phone": "+91 9123456780", "city": "Hyderabad"}}
    hidden = client._hide_run_inputs(inputs)
    assert "56780" not in str(hidden) and "XXXXXXXX80" in str(hidden) and "Hyderabad" in str(hidden)
    assert "9123456780" not in str(client._hide_run_outputs({"output": "Booked for 9123456780"}))
    assert mask_trace("error near 9123456780") == "error near XXXXXXXX80"


def test_shell_auto_tracing_is_switched_off(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    disable_auto_tracing()
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false" and os.environ["LANGSMITH_TRACING"] == "false"
