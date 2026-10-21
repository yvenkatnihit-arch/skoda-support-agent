"""Gemini call logging/retries, thinking budgets, context trimming, safe failure with rollback."""

import httpx
import pytest
from google.genai import errors as genai_errors
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from structlog.testing import capture_logs

from app.config import THINKING_BUDGET_DEFAULTS, Settings
from app.graph.nodes.context import compact_findings, recent_dialogue
from app.graph.runner import FRIENDLY_ERROR, run_turn
from app.llm import LLMUnavailableError, LoggedLLM, is_transient, retry_delay, thinking_kwargs
from tests.test_graph import CONFIG, ai_texts, ask, call, make, route

DEADLINE = genai_errors.ServerError(504, {"error": {"code": 504, "message": "Deadline expired",
                                                    "status": "DEADLINE_EXCEEDED"}})


def settings(**kw) -> Settings:
    return Settings(_env_file=None, google_api_key="test", **kw)


class Flaky:
    """Raises the given exceptions in order, then returns the value."""

    def __init__(self, errors, value):
        self.errors, self.value, self.calls = list(errors), value, 0

    def invoke(self, _messages):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.value


# ---- thinking budgets ------------------------------------------------------------

def test_thinking_defaults_off_for_classifiers_largest_for_composer():
    s = settings()
    assert s.thinking_budget_for("supervisor") == 0 == s.thinking_budget_for("clarify") == s.thinking_budget_for("summarize")
    assert s.thinking_budget_for("composer") == max(THINKING_BUDGET_DEFAULTS.values())
    assert settings(supervisor_thinking_budget=256).thinking_budget_for("supervisor") == 256


def test_thinking_kwargs_per_model_family():
    assert thinking_kwargs("gemini-2.5-flash", 0) == {"thinking_budget": 0}
    assert thinking_kwargs("gemini-2.5-pro", 0) == {"thinking_budget": 128}  # pro can't turn thinking off
    assert thinking_kwargs("gemini-2.5-flash", -1) == {"thinking_budget": -1}
    assert thinking_kwargs("gemini-3-flash-preview", 0) == {"thinking_level": "minimal"}
    assert thinking_kwargs("gemini-3-pro-preview", 2048) == {"thinking_level": "medium"}
    assert thinking_kwargs("gemini-3-pro-preview", -1) == {}


def test_every_node_has_an_output_cap_and_supervisor_uses_function_calling():
    from app.config import LLM_NODES
    from app.llm import NODE_MAX_OUTPUT_TOKENS, STRUCTURED_OUTPUT_METHOD, get_chat_model

    assert set(NODE_MAX_OUTPUT_TOKENS) == set(LLM_NODES)
    assert STRUCTURED_OUTPUT_METHOD == "function_calling"
    s = settings()
    assert get_chat_model("supervisor", s).max_output_tokens == 512           # thinking off: just the cap
    assert get_chat_model("composer", s).max_output_tokens == 2048 + 512      # cap + thinking budget
    assert get_chat_model("supervisor", s).max_retries == 1                   # SDK retries off; LoggedLLM retries


def test_output_cut_at_token_limit_is_logged():
    raw = AIMessage(content="x", response_metadata={"finish_reason": "MAX_TOKENS"})
    llm = LoggedLLM(Flaky([], raw), "composer", "m", settings())
    with capture_logs() as logs:
        llm.invoke([HumanMessage("hi")])
    assert any(e["event"] == "gemini output cut at the token limit" for e in logs)


# ---- retries ---------------------------------------------------------------------

def test_transient_errors_are_recognised():
    assert is_transient(DEADLINE)
    assert is_transient(httpx.ReadTimeout("slow"))
    assert is_transient(genai_errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}))
    wrapped = RuntimeError("wrapper")
    wrapped.__cause__ = DEADLINE
    assert is_transient(wrapped)
    assert not is_transient(genai_errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT"}}))
    assert not is_transient(ValueError("bad schema"))


def test_backoff_grows_and_is_capped():
    s = settings(llm_retry_base_delay_s=1, llm_retry_max_delay_s=4)
    delays = [retry_delay(n, s) for n in (1, 2, 3, 4)]
    assert 1 <= delays[0] <= 1.25 and 2 <= delays[1] <= 2.5 and 4 <= delays[2] <= 5 and 4 <= delays[3] <= 5


def test_logged_llm_retries_transient_errors_with_growing_waits():
    waits = []
    inner = Flaky([DEADLINE, httpx.ConnectTimeout("x")], AIMessage(content="ok"))
    llm = LoggedLLM(inner, "supervisor", "gemini-2.5-flash", settings(llm_max_retries=3), sleep=waits.append)
    with capture_logs() as logs:
        assert llm.invoke([HumanMessage("hi")]).text == "ok"
    assert inner.calls == 3 and len(waits) == 2 and waits[1] > waits[0]
    ok = [e for e in logs if e["event"] == "gemini call"][0]
    assert ok["node"] == "supervisor" and ok["attempt"] == 3 and ok["request_chars"] == 2 and "elapsed_s" in ok
    assert [e["event"] for e in logs].count("gemini call failed") == 2


def test_logged_llm_gives_up_after_all_retries():
    inner = Flaky([DEADLINE] * 10, None)
    llm = LoggedLLM(inner, "lead", "gemini-2.5-flash", settings(llm_max_retries=2), sleep=lambda _: None)
    with pytest.raises(LLMUnavailableError, match="'lead' failed after 3 attempt"):
        llm.invoke([HumanMessage("hi")])
    assert inner.calls == 3


def test_logged_llm_does_not_retry_bad_requests():
    inner = Flaky([genai_errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT"}})], None)
    llm = LoggedLLM(inner, "composer", "gemini-2.5-flash", settings(), sleep=lambda _: None)
    with pytest.raises(LLMUnavailableError):
        llm.invoke([HumanMessage("hi")])
    assert inner.calls == 1


def test_structured_output_raw_is_unwrapped():
    raw = AIMessage(content="", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    parsed = route(["small_talk"])
    llm = LoggedLLM(Flaky([], {"raw": raw, "parsed": parsed, "parsing_error": None}), "supervisor", "m", settings())
    with capture_logs() as logs:
        assert llm.invoke([HumanMessage("hi")]) is parsed
    assert logs[0]["input_tokens"] == 10
    bad = LoggedLLM(Flaky([], {"raw": raw, "parsed": None, "parsing_error": ValueError("x")}), "supervisor", "m", settings())
    with pytest.raises(LLMUnavailableError, match="unparseable"):
        bad.invoke([HumanMessage("hi")])


# ---- context trimming ------------------------------------------------------------

def test_recent_dialogue_keeps_questions_whole_and_clips_long_replies():
    long_reply = AIMessage(content="x" * 5000, id="a1")
    msgs = [HumanMessage("old", id="h0"), long_reply, HumanMessage("q" * 900, id="h1")]
    out = recent_dialogue(msgs, 2, 600)
    assert [m.id for m in out] == ["a1", "h1"]
    assert len(out[0].text) < 650 and out[1].text == "q" * 900
    assert len(long_reply.text) == 5000  # State's message isn't modified


def test_compact_findings_caps_tool_results():
    outputs = {"car_info": {"answer": "a", "status": "complete",
                            "tool_results": [{"tool": "search_brochures", "args": {}, "content": "y" * 20000}]}}
    compact = compact_findings(outputs)
    assert len(compact["car_info"]["tool_results"][0]["content"]) < 6100
    assert len(outputs["car_info"]["tool_results"][0]["content"]) == 20000


# ---- invalid phone never reaches the customer's Confirm ---------------------------

def test_invalid_phone_is_sent_back_to_the_agent_without_pausing():
    bad = {"name": "Venkat", "phone": "1234567890", "city": "Hyderabad", "model": "Kodiaq"}
    graph, llms = make(
        supervisor=[route(["lead_capture"], name="Venkat", city="Hyderabad", model="Kodiaq")],
        lead=[call("book_test_drive", bad, "c1"), AIMessage(content="Need a valid mobile number")],
        composer=[AIMessage(content="Please share a valid 10-digit mobile number.")],
    )
    state = ask(graph, "hyderabad,1234567890,venkat")
    assert "__interrupt__" not in state and graph.get_state(CONFIG).next == ()
    retry_request = llms.agents["lead"].calls[1]
    assert any(isinstance(m, ToolMessage) and "valid 10-digit" in m.content for m in retry_request)
    assert state["slots"]["phone"] is None  # the supervisor dropped the invalid number too


# ---- a failed turn: friendly message, State rolled back, next turn works ----------

class SometimesDown:
    """Composer stand-in: fails (like Gemini after all retries) while `down` is True."""

    def __init__(self, *replies):
        self.down, self.replies = False, list(replies)

    def invoke(self, _messages):
        if self.down:
            raise LLMUnavailableError("composer", 4, DEADLINE)
        return self.replies.pop(0)


def test_failed_turn_rolls_back_and_next_message_works():
    graph, llms = make(supervisor=[route(["small_talk"])] * 3)
    composer = SometimesDown(AIMessage(content="Hello!"), AIMessage(content="Hi again!"))
    llms.composer.invoke = composer.invoke

    assert run_turn(graph, CONFIG, {"messages": [HumanMessage("Hi")]}).reply == "Hello!"

    composer.down = True  # fails late in the turn, after supervisor already wrote to State
    with capture_logs() as logs:
        result = run_turn(graph, CONFIG, {"messages": [HumanMessage("Tell me about the Kylaq")]})
    assert result.status == "error" and result.reply == FRIENDLY_ERROR
    assert "DEADLINE" not in result.reply  # the raw error is in the logs only
    assert any(e["event"] == "turn failed; state rolled back" for e in logs)
    after = graph.get_state(CONFIG)
    assert [m.text for m in after.values["messages"]] == ["Hi", "Hello!"]  # failed question not left behind
    assert after.next == ()

    composer.down = False
    ok = run_turn(graph, CONFIG, {"messages": [HumanMessage("Hello?")]})
    assert ok.status == "reply" and ok.reply == "Hi again!"
    assert ai_texts(graph.get_state(CONFIG).values) == ["Hello!", "Hi again!"]


class Unavailable:
    def invoke(self, _messages):
        raise LLMUnavailableError("supervisor", 4, DEADLINE)


def test_failure_on_the_very_first_turn_leaves_an_empty_thread():
    graph, llms = make()
    llms.supervisor.invoke = Unavailable().invoke
    config = {"configurable": {"thread_id": "fresh"}, "recursion_limit": 50}
    result = run_turn(graph, config, {"messages": [HumanMessage("Hi")]})
    assert result.status == "error"
    assert graph.get_state(config).values.get("messages", []) == []
