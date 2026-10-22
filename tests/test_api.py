"""API (Phase 5): streaming chat, Confirm/Cancel over the API, history, validation and safe errors.
The graph uses scripted fake LLMs, so no Gemini calls; conversations are saved to a temp folder."""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.api.main import create_app
from app.graph.runner import FRIENDLY_ERROR
from app.llm import LLMUnavailableError
from app.memory import file_store
from tests.test_graph import KYLAQ_AT_HYD, call, leads_file, make, route  # noqa: F401 (fixture)
from tests.test_reliability import DEADLINE


@pytest.fixture(autouse=True)
def conversations(monkeypatch, tmp_path):
    folder = tmp_path / "conversations"
    real = file_store._path
    monkeypatch.setattr(file_store, "_path", lambda thread_id, base_dir=None: real(thread_id, base_dir or folder))
    return folder


def events(response) -> list[tuple[str, dict]]:
    out = []
    for chunk in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in chunk.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def client(**llms):
    graph, fakes = make(**llms)
    return TestClient(create_app(graph)), fakes


def test_chat_streams_progress_then_the_checked_reply(conversations):
    c, _ = client(supervisor=[route(["quotation"], **KYLAQ_AT_HYD)],
                  quotation=[call("calculate_on_road_price", KYLAQ_AT_HYD, "q1"), AIMessage(content="1292125")],
                  composer=[AIMessage(content="About ₹12,92,125 on-road in Hyderabad.")])
    with c:
        r = c.post("/api/chat", json={"thread_id": "web1", "message": "Kylaq Signature AT price in Hyderabad"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = events(r)
    names = [e for e, _ in ev]
    assert names[0] == "thread" and names[-1] == "done" and names[-2] == "reply"
    assert "status" in names and names.index("status") < names.index("reply")  # progress before the reply
    assert ev[-2][1]["text"] == "About ₹12,92,125 on-road in Hyderabad."
    assert (conversations / "web1.json").exists()  # saved at the turn end


def test_a_new_chat_gets_a_thread_id_and_history_redraws_it():
    c, _ = client(supervisor=[route(["small_talk"])], composer=[AIMessage(content="Hi! How can I help?")])
    with c:
        thread_id = events(c.post("/api/chat", json={"message": "Hello"}))[0][1]["thread_id"]
        view = c.get(f"/api/threads/{thread_id}").json()
    assert view["messages"] == [{"role": "customer", "text": "Hello"},
                                {"role": "assistant", "text": "Hi! How can I help?"}]
    assert view["pending"] is None


def test_confirm_cancel_over_the_api(leads_file):  # noqa: F811
    args = {"name": "Asha", "phone": "9123456789", "city": "Hyderabad", "model": "Kylaq"}
    c, _ = client(supervisor=[route(["lead_capture"], **args)],
                  lead=[call("book_test_drive", args, "c1"), AIMessage(content="Test drive requested")],
                  composer=[AIMessage(content="Done! The showroom will call you.")])
    with c:
        ev = events(c.post("/api/chat", json={"thread_id": "web2", "message": "Book a Kylaq test drive"}))
        confirm = dict(ev)["confirm"]
        assert confirm["actions"][0]["tool"] == "book_test_drive" and not leads_file.exists()
        assert c.get("/api/threads/web2").json()["pending"]["actions"][0]["tool"] == "book_test_drive"
        done = events(c.post("/api/resume", json={"thread_id": "web2", "approved": True}))
    assert dict(done)["reply"]["text"] == "Done! The showroom will call you." and leads_file.exists()


def test_resume_without_a_pause_is_refused():
    c, _ = client(supervisor=[route(["small_talk"])], composer=[AIMessage(content="Hi")])
    with c:
        c.post("/api/chat", json={"thread_id": "web3", "message": "Hi"})
        assert c.post("/api/resume", json={"thread_id": "web3", "approved": True}).status_code == 409


def test_bad_input_is_rejected():
    c, _ = client()
    with c:
        assert c.post("/api/chat", json={"thread_id": "../etc", "message": "hi"}).status_code == 400
        assert c.post("/api/chat", json={"message": ""}).status_code == 422
        assert c.post("/api/chat", json={"message": "x" * 2001}).status_code == 422
        assert c.post("/api/chat", json={"message": "hi", "lat": 51.5, "long": -0.1}).status_code == 422  # not India
        assert c.get("/api/threads/a%2Fb").status_code in (400, 404)


def test_a_failure_shows_only_the_friendly_message():
    class Down:
        def invoke(self, _messages):
            raise LLMUnavailableError("composer", 4, DEADLINE)

    c, fakes = client(supervisor=[route(["small_talk"])])
    fakes.composer.invoke = Down().invoke
    with c:
        ev = dict(events(c.post("/api/chat", json={"thread_id": "web4", "message": "Hi"})))
    assert ev["error"]["text"] == FRIENDLY_ERROR and "DEADLINE" not in json.dumps(ev)


def test_shared_location_becomes_the_lat_long_slots():
    c, fakes = client(supervisor=[route(["locator"])],
                      locator=[AIMessage(content="Nearest showroom 2.4 km")],
                      composer=[AIMessage(content="The nearest showroom is 2.4 km away.")])
    with c:
        events(c.post("/api/chat", json={"thread_id": "web5", "message": "Nearest showroom to me",
                                         "lat": 17.4483, "long": 78.3915}))
    request = fakes.agents["locator"].calls[0][0].content
    assert "17.4483" in request and "78.3915" in request  # the locator was given the shared position


def test_the_chat_page_and_health():
    c, _ = client()
    with c:
        assert c.get("/api/health").json() == {"status": "ok"}
        page = c.get("/")
    assert page.status_code == 200 and "Car support assistant" in page.text
