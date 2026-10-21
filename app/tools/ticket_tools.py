"""Support tickets in data/tickets.csv: create_ticket (an action tool: the customer confirms first) and
get_ticket_status (read-only; needs the ticket number AND the mobile it was raised with).

A ticket must describe the issue properly for its category before it can be raised: INTAKE says, per category,
which issue types exist and which details are needed. ticket_problem() checks that, and the approval step runs
it BEFORE showing Confirm, so the customer is asked the missing questions instead of confirming a vague ticket.

The file holds personal data and is gitignored; tool results mask the phone number. Same safety as the lead
tools: contact details checked, spreadsheet formula injection neutralised, writes under a lock.
"""

import csv
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import structlog
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import get_settings
from app.data.repository import get_repository
from app.log import mask_phone
from app.tools.common import ToolInputError, resolve_model, returns_errors_as_text, to_json
from app.tools.lead_tools import _clean_text, contact_problem, normalize_indian_mobile

log = structlog.get_logger()


@dataclass(frozen=True)
class Intake:
    question: str                 # how to ask for the issue type
    issue_types: dict[str, str]   # issue_type -> what it means (shown to the agent and the customer)
    needs: tuple[str, ...]        # details required before the ticket can be raised
    ask_for: str                  # how to ask for those details


# What the bot must find out for each category before raising a ticket. Edit here to change the questions.
INTAKE: dict[str, Intake] = {
    "billing": Intake(
        "What's wrong with the bill?",
        {"wrong_amount": "the amount or calculation looks wrong",
         "charged_for_work_not_done": "charged for work that wasn't done",
         "double_charge": "charged twice",
         "invoice_error": "invoice details or printing are wrong (name, car number, GST...)",
         "payment_or_refund_pending": "a payment or refund hasn't come through",
         "other": "something else about the bill"},
        ("dealer", "incident_date"),
        "which service centre or dealer, and roughly when; the invoice number and amount if they have them"),
    "vehicle_problem": Intake(
        "What's happening with the car?",
        {"noise_or_vibration": "a noise or vibration", "warning_light": "a warning light is on",
         "ac_or_climate": "the AC or climate control", "electrical": "lights, battery, screen or other electrics",
         "wont_start": "the car won't start", "other": "something else"},
        ("incident_date", "drivable"),
        "since when, whether it started after a service, and whether the car is safe to drive"),
    "service_experience": Intake(
        "What went wrong with the service?",
        {"delay": "the car wasn't ready on time", "not_fixed_or_not_done": "the work wasn't done or the issue isn't fixed",
         "damage_during_service": "the car was damaged during the service", "cost_above_estimate": "it cost more than the estimate",
         "other": "something else"},
        ("dealer", "incident_date"),
        "which service centre, and the date of the visit"),
    "dealer_staff": Intake(
        "What happened with the dealer or staff?",
        {"behaviour": "staff behaviour", "wrong_information": "they gave wrong information",
         "delivery_delay": "delay in delivery of a new car", "documents": "documents (RC, insurance, invoice) not given",
         "other": "something else"},
        ("dealer", "incident_date"),
        "which dealer, and when it happened"),
    "other": Intake("What's the issue?", {"other": "anything else"}, (), "what happened, in a few sentences"),
}
Category = Literal["vehicle_problem", "service_experience", "dealer_staff", "billing", "other"]

TICKET_COLUMNS = ["ticket_id", "created_at", "status", "category", "issue_type", "name", "phone", "city", "model",
                  "registration_number", "dealer_id", "dealer_name", "incident_date", "invoice_number", "amount",
                  "drivable", "description"]
DESCRIPTION_MIN, DESCRIPTION_MAX = 20, 1000
# Indian registration numbers: state + district + series + 4 digits (TS09AB1234), or Bharat series (22BH1234AA).
REGISTRATION_RES = (re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$"), re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$"))
TICKET_ID_RE = re.compile(r"^SKD-\d{8}-\d{4}$")
NOT_FOUND = ("NOT_FOUND: no ticket was found with that ticket number and mobile number. Ask the customer to "
             "check both (the ticket number looks like SKD-20261021-0001).")

_write_lock = threading.Lock()


def intake_guide() -> str:
    """INTAKE as text for the support agent's prompt."""
    lines = []
    for category, intake in INTAKE.items():
        options = "; ".join(f"{k} = {v}" for k, v in intake.issue_types.items())
        lines.append(f"- {category}: ask \"{intake.question}\" ({options}). Then ask {intake.ask_for}.")
    return "\n".join(lines)


def _blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def ticket_problem(args: dict) -> str | None:
    """What the agent must still ask before this ticket can be raised, or None if it is complete. Run by the
    approval step before Confirm (so the customer never confirms a vague ticket) and by create_ticket itself."""
    contact = contact_problem(args.get("name"), args.get("phone"), args.get("city"))
    if contact:
        return contact
    category = args.get("category")
    intake = INTAKE.get(category)
    if intake is None:
        return f"Pick a category: {', '.join(INTAKE)}."
    issue_type = args.get("issue_type")
    if issue_type not in intake.issue_types:
        options = "; ".join(intake.issue_types.values())
        return f"Ask the customer first: \"{intake.question}\" ({options}). Don't raise the ticket yet."
    missing = []
    if "dealer" in intake.needs and _blank(args.get("dealer_id")) and _blank(args.get("dealer_name")):
        missing.append("which dealer or service centre")
    if "incident_date" in intake.needs and _blank(args.get("incident_date")):
        missing.append("when it happened")
    if "drivable" in intake.needs and args.get("drivable") is None:
        missing.append("whether the car is safe to drive")
    if len(re.sub(r"\s+", " ", args.get("description") or "").strip()) < DESCRIPTION_MIN:
        missing.append("what happened, in a sentence or two")
    if missing:
        return f"Ask the customer {', and '.join(missing)} before raising the ticket ({intake.ask_for})."
    return None


def _tickets_path() -> Path:
    return get_settings().tickets_file


def _read_tickets() -> list[dict]:
    path = _tickets_path()
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_row(row: dict) -> None:
    """Append one ticket. A file written with older columns is rewritten with the current ones first."""
    path = _tickets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_header = None
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), None)
    if existing_header and existing_header != TICKET_COLUMNS:
        old = _read_tickets()
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TICKET_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(old)
        existing_header = TICKET_COLUMNS
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TICKET_COLUMNS)
        if not existing_header:
            writer.writeheader()
        writer.writerow(row)


def normalize_registration(raw: str | None) -> str | None:
    """'TS 09 AB 1234' -> 'TS09AB1234'; None if empty. Raises if it doesn't look like an Indian registration."""
    if not raw or not raw.strip():
        return None
    reg = re.sub(r"[\s.-]", "", raw).upper()
    if not any(r.match(reg) for r in REGISTRATION_RES):
        raise ToolInputError(f"'{raw}' doesn't look like an Indian registration number (e.g. TS09AB1234). "
                             "Ask the customer to check it, or leave it out.")
    return reg


def _next_ticket_id(existing: list[dict], today: str) -> str:
    prefix = f"SKD-{today}-"
    used = [int(r["ticket_id"][-4:]) for r in existing if r.get("ticket_id", "").startswith(prefix)]
    return f"{prefix}{max(used, default=0) + 1:04d}"


class CreateTicketArgs(BaseModel):
    name: str = Field(description="Customer's name")
    phone: str = Field(description="Customer's 10-digit Indian mobile number")
    city: str = Field(description="Customer's city")
    model: str = Field(description="The customer's Skoda model, e.g. 'Kushaq'")
    category: Category = Field(description="vehicle_problem, service_experience, dealer_staff, billing or other")
    issue_type: str = Field(description="The kind of issue within the category, as the customer described it "
                                        "(see the issue types in your instructions)")
    description: str = Field(description="What happened, in the customer's words, with the details they gave "
                                         "(20-1000 characters)")
    incident_date: str | None = Field(default=None, description="When it happened or started, as the customer "
                                                                "said it (e.g. '2026-10-17', 'last Friday')")
    dealer_id: str | None = Field(default=None, description="dealer_id from find_nearest_dealers, if known")
    dealer_name: str | None = Field(default=None, description="Dealer or service centre name as the customer said "
                                                              "it, if there's no dealer_id")
    invoice_number: str | None = Field(default=None, description="Invoice number, for billing issues, if given")
    amount: int | None = Field(default=None, ge=0, description="Amount in rupees the issue is about, if given")
    drivable: bool | None = Field(default=None, description="Vehicle problems: is the car safe to drive?")
    registration_number: str | None = Field(default=None, description="Car registration, e.g. TS09AB1234, if given")


@tool("create_ticket", args_schema=CreateTicketArgs)
@returns_errors_as_text
def create_ticket(name: str, phone: str, city: str, model: str, category: str, issue_type: str, description: str,
                  incident_date: str | None = None, dealer_id: str | None = None, dealer_name: str | None = None,
                  invoice_number: str | None = None, amount: int | None = None, drivable: bool | None = None,
                  registration_number: str | None = None) -> str:
    """ACTION: raise a support ticket for a complaint or problem. Needs the customer's confirmation.
    Only call it once you understand the issue: its category, issue type and the details your instructions list."""
    args = dict(name=name, phone=phone, city=city, category=category, issue_type=issue_type,
                description=description, incident_date=incident_date, dealer_id=dealer_id,
                dealer_name=dealer_name, drivable=drivable)
    problem = ticket_problem(args)
    if problem:
        raise ToolInputError(problem)
    text = _clean_text(description, "description", DESCRIPTION_MAX)
    car = resolve_model(model)
    reg = normalize_registration(registration_number)
    dealer = None
    if dealer_id:
        dealer = next((d for d in get_repository().list_dealers() if d.id == dealer_id), None)
        if dealer is None:
            raise ToolInputError(f"Unknown dealer '{dealer_id}'. Use a dealer_id from find_nearest_dealers, "
                                 "or give the dealer_name the customer said.")
    now = datetime.now(timezone.utc)
    with _write_lock:
        row = {
            "ticket_id": _next_ticket_id(_read_tickets(), now.strftime("%Y%m%d")),
            "created_at": now.isoformat(timespec="seconds"), "status": "open", "category": category,
            "issue_type": issue_type, "name": _clean_text(name, "name", 80), "phone": normalize_indian_mobile(phone),
            "city": _clean_text(city, "city", 60), "model": car.name, "registration_number": reg or "",
            "dealer_id": dealer.id if dealer else "",
            "dealer_name": dealer.name if dealer else _clean_text(dealer_name, "dealer_name", 120),
            "incident_date": _clean_text(incident_date, "incident_date", 40),
            "invoice_number": _clean_text(invoice_number, "invoice_number", 40),
            "amount": "" if amount is None else str(amount),
            "drivable": "" if drivable is None else ("yes" if drivable else "no"),
            "description": text,
        }
        _write_row(row)
    log.info("ticket created", ticket_id=row["ticket_id"], category=category, issue_type=issue_type)
    promise = get_settings().support_response_promise
    return to_json({
        "status": "saved", "ticket_id": row["ticket_id"], "ticket_status": "open", "category": category,
        "issue_type": issue_type, "model": car.name, "phone": mask_phone(row["phone"]),
        "dealer": row["dealer_name"] or None, "response_promise": promise,
        "message": f"Ticket {row['ticket_id']} raised. The Skoda customer care team will contact the customer "
                   f"{promise}.",
    })


class TicketStatusArgs(BaseModel):
    ticket_id: str = Field(description="Ticket number, e.g. SKD-20261021-0001")
    phone: str = Field(description="The mobile number the ticket was raised with")


@tool("get_ticket_status", args_schema=TicketStatusArgs)
@returns_errors_as_text
def get_ticket_status(ticket_id: str, phone: str) -> str:
    """Look up a support ticket's status. Needs BOTH the ticket number and the mobile it was raised with."""
    tid = (ticket_id or "").strip().upper()
    mobile = normalize_indian_mobile(phone or "")
    if not TICKET_ID_RE.match(tid) or mobile is None:
        return NOT_FOUND
    row = next((r for r in _read_tickets() if r.get("ticket_id") == tid and r.get("phone") == mobile), None)
    if row is None:  # same answer whether the number exists or not: don't reveal other customers' tickets
        return NOT_FOUND
    return to_json({"ticket_id": tid, "ticket_status": row["status"], "category": row["category"],
                    "issue_type": row.get("issue_type") or None, "model": row["model"],
                    "raised_on": row["created_at"][:10], "phone": mask_phone(mobile)})
