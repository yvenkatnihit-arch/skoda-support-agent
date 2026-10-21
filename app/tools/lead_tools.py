"""Action tools that write customer details to data/leads.csv.

These are in TOOLS_REQUIRING_APPROVAL (Phase 4): the graph pauses and the customer must confirm
before they run. The file holds personal data and is gitignored; tool results mask the phone number.
"""

import csv
import re
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import structlog
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import get_settings
from app.data.repository import get_repository
from app.log import mask_phone
from app.tools.common import ToolInputError, resolve_model, returns_errors_as_text, to_json

log = structlog.get_logger()

LEAD_COLUMNS = ["lead_id", "created_at", "request_type", "name", "phone", "city", "model",
                "preferred_date", "dealer_id", "notes"]
MOBILE_RE = re.compile(r"^[6-9]\d{9}$")
MAX_TEST_DRIVE_DAYS_AHEAD = 60

_write_lock = threading.Lock()


def normalize_indian_mobile(raw: str) -> str | None:
    """10-digit Indian mobile number, or None if it isn't one. Strips spaces, dashes, +91, 91 and a leading 0."""
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    return digits if MOBILE_RE.match(digits) else None


def _leads_path() -> Path:
    return get_settings().leads_file


def _clean_text(value: str | None, field: str, max_len: int) -> str:
    text = re.sub(r"\s+", " ", (value or "")).strip()
    if len(text) > max_len:
        raise ToolInputError(f"{field} is too long (max {max_len} characters).")
    # neutralise spreadsheet formula injection when the CSV is opened in Excel
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


def _append_lead(row: dict) -> None:
    path = _leads_path()
    with _write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LEAD_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)


def contact_problem(name: str | None, phone: str | None, city: str | None) -> str | None:
    """Why these contact details can't be saved, or None if they're fine. Also used by the approval node,
    so the customer is never asked to confirm details that would be rejected."""
    if len(re.sub(r"\s+", " ", name or "").strip()) < 2:
        return "Ask the customer for their name."
    if normalize_indian_mobile(phone or "") is None:
        return ("That doesn't look like a valid 10-digit Indian mobile number (it should start with 6, 7, 8 or 9). "
                "Ask the customer for their mobile number again.")
    if len(re.sub(r"\s+", " ", city or "").strip()) < 2:
        return "Ask the customer which city they are in."
    return None


def _base_row(request_type: str, name: str, phone: str, city: str, model: str | None) -> dict:
    problem = contact_problem(name, phone, city)
    if problem:
        raise ToolInputError(problem)
    clean_name = _clean_text(name, "name", 80)
    mobile = normalize_indian_mobile(phone)
    clean_city = _clean_text(city, "city", 60)
    model_name = resolve_model(model).name if model else ""
    return {
        "lead_id": uuid.uuid4().hex[:10],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request_type": request_type,
        "name": clean_name, "phone": mobile, "city": clean_city, "model": model_name,
        "preferred_date": "", "dealer_id": "", "notes": "",
    }


def _confirmation(row: dict, message: str) -> str:
    log.info("lead saved", lead_id=row["lead_id"], request_type=row["request_type"], city=row["city"])
    return to_json({
        "status": "saved", "lead_id": row["lead_id"], "request_type": row["request_type"],
        "name": row["name"], "phone": mask_phone(row["phone"]), "city": row["city"], "model": row["model"],
        "preferred_date": row["preferred_date"] or None, "message": message,
    })


class SaveLeadArgs(BaseModel):
    name: str = Field(description="Customer's name")
    phone: str = Field(description="Customer's 10-digit Indian mobile number")
    city: str = Field(description="Customer's city")
    model_of_interest: str | None = Field(default=None, description="Skoda model they are interested in, if any")
    request_type: Literal["callback", "brochure", "general"] = Field(
        default="callback", description="'callback' for a call from Skoda, 'brochure' to receive a brochure")
    notes: str | None = Field(default=None, description="Short note about what the customer wants")


@tool("save_lead", args_schema=SaveLeadArgs)
@returns_errors_as_text
def save_lead(name: str, phone: str, city: str, model_of_interest: str | None = None,
              request_type: str = "callback", notes: str | None = None) -> str:
    """ACTION: save the customer's contact details so Skoda can call back or send a brochure.
    Needs the customer's confirmation. Only call it once you have name, phone and city."""
    row = _base_row(request_type, name, phone, city, model_of_interest)
    row["notes"] = _clean_text(notes, "notes", 300)
    _append_lead(row)
    what = "send the brochure" if request_type == "brochure" else "call you back"
    return _confirmation(row, f"Thanks {row['name']}, a Skoda team member will {what} soon.")


class TestDriveArgs(BaseModel):
    name: str = Field(description="Customer's name")
    phone: str = Field(description="Customer's 10-digit Indian mobile number")
    city: str = Field(description="Customer's city")
    model: str = Field(description="Skoda model to test drive, e.g. 'Kylaq'")
    preferred_date: date | None = Field(default=None, description="Preferred date, YYYY-MM-DD, today or later")
    dealer_id: str | None = Field(default=None, description="dealer_id from find_nearest_dealers, if chosen")


@tool("book_test_drive", args_schema=TestDriveArgs)
@returns_errors_as_text
def book_test_drive(name: str, phone: str, city: str, model: str, preferred_date: date | None = None,
                    dealer_id: str | None = None) -> str:
    """ACTION: request a test drive. Needs the customer's confirmation. The showroom calls the customer
    to fix the exact time, so this is a request, not a confirmed slot."""
    car = resolve_model(model)
    if car.status != "on_sale":
        raise ToolInputError(f"The {car.name} is not on sale yet, so test drives can't be booked. "
                             "Offer to save a callback request instead.")
    row = _base_row("test_drive", name, phone, city, car.name)
    if preferred_date is not None:
        today = date.today()
        if preferred_date < today:
            raise ToolInputError("The preferred date is in the past. Ask the customer for a date from today on.")
        if preferred_date > today + timedelta(days=MAX_TEST_DRIVE_DAYS_AHEAD):
            raise ToolInputError(f"Please pick a date within the next {MAX_TEST_DRIVE_DAYS_AHEAD} days.")
        row["preferred_date"] = preferred_date.isoformat()
    if dealer_id:
        dealer = next((d for d in get_repository().list_dealers(dealer_type="showroom") if d.id == dealer_id), None)
        if dealer is None:
            raise ToolInputError(f"Unknown showroom '{dealer_id}'. Use a dealer_id from find_nearest_dealers.")
        row["dealer_id"] = dealer.id
    _append_lead(row)
    when = f" for {row['preferred_date']}" if row["preferred_date"] else ""
    return _confirmation(row, f"Test drive request for the {car.name}{when} saved. The showroom will call "
                              "to confirm the time.")


LEAD_TOOLS = [save_lead, book_test_drive]
