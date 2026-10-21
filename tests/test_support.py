"""Support agent and ticket tools (ROADMAP section 9): tickets are validated and saved safely, status needs the
matching phone, and several action agents run one at a time, each with its own Confirm. Offline."""

import csv
import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.data.repository import get_repository
from app.tools import ticket_tools
from app.tools.ticket_tools import NOT_FOUND, create_ticket, get_ticket_status, normalize_registration
from tests.test_graph import CONFIG, ai_texts, ask, call, leads_file, make, route  # noqa: F401 (fixture)

ASHA = {"name": "Asha Rao", "phone": "9123456780", "city": "Hyderabad", "model": "Kushaq"}
TICKET = {**ASHA, "category": "service_experience", "issue_type": "not_fixed_or_not_done",
          "dealer_name": "Skoda Madhapur", "incident_date": "last Friday",
          "description": "AC stopped working two days after the service; it was supposed to be fixed."}


@pytest.fixture
def tickets_file(monkeypatch, tmp_path):
    path = tmp_path / "tickets.csv"
    monkeypatch.setattr(ticket_tools, "_tickets_path", lambda: path)
    return path


def _rows(path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


# ---- ticket tools ------------------------------------------------------------------

def test_create_ticket_saves_a_numbered_ticket_and_masks_the_phone(tickets_file):
    out = json.loads(create_ticket.invoke(TICKET))
    assert out["status"] == "saved" and out["ticket_status"] == "open"
    assert out["ticket_id"].startswith("SKD-") and out["ticket_id"].endswith("-0001")
    assert out["phone"] == "XXXXXXXX80" and out["response_promise"] == "within 24 hours"
    second = json.loads(create_ticket.invoke({**TICKET, "category": "billing", "issue_type": "double_charge"}))
    assert second["ticket_id"].endswith("-0002")
    [row, _] = _rows(tickets_file)
    assert row["phone"] == "9123456780" and row["model"] == "Kushaq" and row["status"] == "open"


def test_create_ticket_validates_before_saving(tickets_file):
    assert "mobile number" in create_ticket.invoke({**TICKET, "phone": "1234567890"})
    assert "which city" in create_ticket.invoke({**TICKET, "city": ""})
    assert "what happened" in create_ticket.invoke({**TICKET, "description": "bad"})
    assert "registration" in create_ticket.invoke({**TICKET, "registration_number": "HELLO"})
    assert "Unknown dealer" in create_ticket.invoke({**TICKET, "dealer_id": "nope"})
    assert "What went wrong with the service" in create_ticket.invoke({**TICKET, "issue_type": "billing_stuff"})
    assert not tickets_file.exists()  # nothing saved


def test_ticket_fields_are_cleaned(tickets_file):
    dealer = get_repository().list_dealers()[0]
    create_ticket.invoke({**TICKET, "description": "=HYPERLINK(\"x\") the car won't start after the service",
                          "registration_number": "ts 09 ab 1234", "dealer_id": dealer.id})
    [row] = _rows(tickets_file)
    assert row["description"].startswith("'=")  # spreadsheet formula injection neutralised
    assert row["registration_number"] == "TS09AB1234" and row["dealer_id"] == dealer.id


def test_registration_formats():
    assert normalize_registration("MH-12-AB-1234") == "MH12AB1234"
    assert normalize_registration("22 BH 1234 AA") == "22BH1234AA"
    assert normalize_registration("") is None


def test_ticket_status_needs_the_matching_phone(tickets_file):
    tid = json.loads(create_ticket.invoke(TICKET))["ticket_id"]
    found = json.loads(get_ticket_status.invoke({"ticket_id": tid.lower(), "phone": "+91 91234 56780"}))
    assert found["ticket_status"] == "open" and found["phone"] == "XXXXXXXX80"
    # wrong phone, unknown number and junk all get the same answer: nothing about other people's tickets leaks
    assert get_ticket_status.invoke({"ticket_id": tid, "phone": "9876543210"}) == NOT_FOUND
    assert get_ticket_status.invoke({"ticket_id": "SKD-20990101-0001", "phone": "9123456780"}) == NOT_FOUND
    assert get_ticket_status.invoke({"ticket_id": "x", "phone": "y"}) == NOT_FOUND


# ---- the support agent in the graph -------------------------------------------------

def test_complaint_pauses_for_confirm_then_saves_the_ticket(tickets_file):
    graph, llms = make(
        supervisor=[route(["support_request"], **ASHA)],
        support=[call("create_ticket", TICKET, "t1"), AIMessage(content="Ticket raised")],
        composer=[AIMessage(content="Sorry about that. Your ticket is raised.")],
    )
    paused = ask(graph, "My Kushaq's AC stopped working after the service")
    action = paused["__interrupt__"][0].value
    assert action["agent"] == "support" and action["actions"][0]["tool"] == "create_ticket"
    assert not tickets_file.exists()  # nothing saved before Confirm

    state = graph.invoke(Command(resume={"approved": True}), CONFIG)
    assert len(_rows(tickets_file)) == 1
    assert state["agent_outputs"]["support"]["tool_results"][0]["tool"] == "create_ticket"
    assert ai_texts(state)[-1] == "Sorry about that. Your ticket is raised."


def test_cancelled_complaint_saves_nothing(tickets_file):
    graph, llms = make(
        supervisor=[route(["support_request"], **ASHA)],
        support=[call("create_ticket", TICKET, "t1"), AIMessage(content="Customer cancelled")],
        composer=[AIMessage(content="No problem, I haven't raised anything.")],
    )
    ask(graph, "Complaint about my service")
    graph.invoke(Command(resume={"approved": False}), CONFIG)
    assert not tickets_file.exists()


def test_two_action_agents_run_one_after_another_each_with_its_own_confirm(tickets_file, leads_file):  # noqa: F811
    drive = {"name": "Asha Rao", "phone": "9123456780", "city": "Hyderabad", "model": "Kylaq"}
    graph, llms = make(
        supervisor=[route(["support_request", "lead_capture"], **ASHA)],
        lead=[call("book_test_drive", drive, "d1"), AIMessage(content="Test drive requested")],
        support=[call("create_ticket", TICKET, "t1"), AIMessage(content="Ticket raised")],
        composer=[AIMessage(content="Test drive requested and ticket raised.")],
    )
    first = ask(graph, "Book a Kylaq test drive, and I want to complain about my Kushaq's AC")
    assert first["__interrupt__"][0].value["agent"] == "lead"  # reply order: lead (50) before support (60)
    assert llms.agents["support"].calls == []  # support waits its turn

    second = graph.invoke(Command(resume={"approved": True}), CONFIG)
    assert second["__interrupt__"][0].value["agent"] == "support"
    assert leads_file.exists() and not tickets_file.exists()
    support_prompt = llms.agents["support"].calls[0][0].content
    assert "Test drive requested" in support_prompt  # sees the earlier action agent's result

    state = graph.invoke(Command(resume={"approved": True}), CONFIG)
    assert list(state["agent_outputs"]) == ["lead", "support"] and len(_rows(tickets_file)) == 1
    assert ai_texts(state)[-1] == "Test drive requested and ticket raised."
    assert state["active_action"] == "support" and state.get("action_messages", []) == []


def test_support_cannot_use_the_lead_agents_tools(tickets_file, leads_file):  # noqa: F811
    graph, llms = make(
        supervisor=[route(["support_request"], **ASHA)],
        support=[call("book_test_drive", {**ASHA, "model": "Kylaq"}, "x1"), AIMessage(content="Can't do that")],
        composer=[AIMessage(content="I can raise a complaint for you.")],
    )
    state = ask(graph, "complaint")
    assert "not one of your tools" in llms.agents["support"].calls[1][-1].content
    assert "__interrupt__" not in state and not leads_file.exists()


# ---- understand the issue before raising a ticket ----------------------------------

def test_billing_needs_the_kind_of_problem_then_dealer_and_date():
    from app.tools.ticket_tools import ticket_problem

    base = {**ASHA, "category": "billing", "description": "There is a problem with my service bill."}
    ask = ticket_problem(base)
    assert "What's wrong with the bill?" in ask and "charged twice" in ask and "printing" in ask
    ask = ticket_problem({**base, "issue_type": "charged_for_work_not_done"})
    assert "which dealer or service centre" in ask and "when it happened" in ask
    done = {**base, "issue_type": "charged_for_work_not_done", "dealer_name": "Skoda Madhapur",
            "incident_date": "last Friday"}
    assert ticket_problem(done) is None


def test_each_category_asks_its_own_questions():
    from app.tools.ticket_tools import INTAKE, ticket_problem

    for category, intake in INTAKE.items():
        assert intake.question in ticket_problem({**ASHA, "category": category, "description": "x" * 30})
    vehicle = {**ASHA, "category": "vehicle_problem", "issue_type": "wont_start", "incident_date": "today",
               "description": "The car won't start since this morning."}
    assert "safe to drive" in ticket_problem(vehicle)
    assert ticket_problem({**vehicle, "drivable": False}) is None
    assert ticket_problem({**ASHA, "category": "other", "issue_type": "other",
                           "description": "I want to give feedback about the showroom music."}) is None


def test_a_vague_ticket_never_reaches_confirm(tickets_file):
    vague = {**ASHA, "category": "billing", "issue_type": "other_stuff", "description": "Problem with the bill."}
    graph, llms = make(
        supervisor=[route(["support_request"], **ASHA)],
        support=[call("create_ticket", vague, "t1"),
                 AIMessage(content="Ask: what's wrong with the bill (wrong amount, charged twice...)?")],
        composer=[AIMessage(content="Sorry to hear that. What's wrong with the bill?")],
    )
    state = ask(graph, "I have a problem with my bill")
    assert "__interrupt__" not in state and graph.get_state(CONFIG).next == ()  # no Confirm shown
    sent_back = llms.agents["support"].calls[1][-1].content
    assert "What's wrong with the bill?" in sent_back and "Don't raise the ticket yet" in sent_back
    assert not tickets_file.exists()
    assert ai_texts(state)[-1] == "Sorry to hear that. What's wrong with the bill?"


def test_support_prompt_lists_every_category_and_sees_more_history():
    from app.agents import CATALOG
    from app.tools.ticket_tools import INTAKE

    spec = CATALOG.agents["support"]
    assert spec.history_messages == 10 and CATALOG.agents["quotation"].history_messages == 4
    for intake in INTAKE.values():
        assert intake.question in spec.prompt


def test_old_ticket_files_are_upgraded_to_the_new_columns(tickets_file):
    tickets_file.write_text("ticket_id,created_at,status,category,name,phone,city,model,registration_number,"
                            "dealer_id,description\nSKD-20261021-0001,2026-10-21T10:00:00+00:00,open,other,A,"
                            "9123456780,Hyderabad,Kushaq,,,old ticket\n", encoding="utf-8")
    out = json.loads(create_ticket.invoke(TICKET))
    assert out["ticket_id"].endswith("-0002") or not out["ticket_id"].startswith("SKD-20261021")
    rows = _rows(tickets_file)
    assert len(rows) == 2 and rows[0]["description"] == "old ticket" and rows[1]["issue_type"] == "not_fixed_or_not_done"
    found = json.loads(get_ticket_status.invoke({"ticket_id": "SKD-20261021-0001", "phone": "9123456780"}))
    assert found["ticket_status"] == "open"
