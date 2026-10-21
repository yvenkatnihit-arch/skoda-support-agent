"""Slot merging with a list of cars (ROADMAP section 12).

Stored slots look like:
    {city, state, area_or_pincode, lat, long, name, phone,          # shared: the customer's details
     cars: [{model, variant, transmission, fuel}, ...],           # remembered cars, most recent last
     active_cars: ["Kylaq", "Kushaq"],                            # the cars this turn is about
     car_intents: {"Kylaq": ["quotation"]},                       # which intents a car is for (none = all)
     model, variant, transmission, fuel}                          # mirror of the current car (last active)

The flat mirror keeps single-car code (prompts, lead/support tools, tests) working as before.
"""

import re

from app.data.repository import get_repository
from app.graph.slots import CAR_KEYS, MAX_REMEMBERED_CARS, SHARED_KEYS, CarMention, Slots
from app.tools.lead_tools import normalize_indian_mobile

# Text Gemini sometimes writes for "no value" instead of null: never stored as a slot.
PLACEHOLDERS = {"null", "none", "n/a", "na", "unknown", "not specified", "not given", "-", "?"}
# Car details that belong to one particular car (a new car never inherits another car's).
CAR_SPECIFIC_SLOTS = ("variant", "transmission", "fuel")
# Words that show the customer actually said a gearbox / fuel (Gemini sometimes copies the old car's values).
MENTIONS = {
    "transmission": {"MT": r"\bmt\b|\bmanual", "AT": r"\bat\b|\bautomatic|\bauto\b|\btorque",
                     "DSG": r"\bdsg\b|dual[- ]clutch"},
    "fuel": {"petrol": r"\bpetrol|\btsi\b", "diesel": r"\bdiesel", "electric": r"\belectric|\bev\b"},
}
INTERNAL_KEYS = ("active_cars", "car_intents")


def _squash(text: str) -> str:
    return re.sub(r"\s*plus\b", "+", text.lower()).replace(" ", "")


def _mentioned(key: str, value, message: str) -> bool:
    """Did the customer's own message say this variant / transmission / fuel? Whole words only: the "at" in
    "what" is not an automatic gearbox."""
    text = (message or "").lower()
    if key == "variant":
        return _squash(str(value)) in _squash(text)
    pattern = MENTIONS.get(key, {}).get(str(value))
    return bool(pattern and re.search(pattern, text))


def _given(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().lower() not in PLACEHOLDERS
    return True


def _clean(value):
    return value.strip() if isinstance(value, str) else value


# ---- reading stored slots ------------------------------------------------------------

def normalize(slots: dict | None) -> dict:
    """Stored slots in the current shape. Older conversations had only flat model/variant/...: one car."""
    s = dict(slots or {})
    cars = [dict(c) for c in (s.get("cars") or [])]
    if not cars and any(s.get(k) for k in CAR_KEYS):
        cars = [{k: s.get(k) for k in CAR_KEYS}]
    s["cars"] = cars
    s["active_cars"] = list(s.get("active_cars") or ([cars[-1].get("model")] if cars else []))
    s["car_intents"] = dict(s.get("car_intents") or {})
    for key in SHARED_KEYS:
        s.setdefault(key, None)
    return s


def _car_by_model(cars: list[dict], name: str | None) -> dict | None:
    return next((c for c in cars if c.get("model") == name), None) if name else None


def _active(s: dict) -> list[dict]:
    """Active cars of already-normalized slots (the same dicts, so updates stick)."""
    active = [c for name in s["active_cars"] if (c := _car_by_model(s["cars"], name))]
    return active or ([s["cars"][-1]] if s["cars"] else [])


def active_car_list(slots: dict) -> list[dict]:
    return _active(normalize(slots))


def cars_for(slots: dict, intent: str | None) -> list[dict]:
    """The active cars an agent works on: those tagged for its intent, or all active cars if none is tagged."""
    s = normalize(slots)
    active = _active(s)
    if intent is None:
        return active
    tagged = [c for c in active if intent in (s["car_intents"].get(c.get("model")) or [])]
    untagged = [c for c in active if not s["car_intents"].get(c.get("model"))]
    return tagged + untagged if tagged else active


def agent_view(slots: dict, intent: str | None) -> dict:
    """What an agent sees: the shared details plus only its cars. With exactly one car, its details are also
    given flat (model, variant, ...), so single-car prompts and tools work as before."""
    s = normalize(slots)
    view = {k: s.get(k) for k in SHARED_KEYS}
    cars = cars_for(s, intent)
    if len(cars) == 1:
        view.update({k: cars[0].get(k) for k in CAR_KEYS})
    elif cars:
        view["cars"] = [{k: v for k, v in c.items() if v is not None} for c in cars]
    return {k: v for k, v in view.items() if v not in (None, "")}


def public_view(slots: dict) -> dict:
    """Slots for the composer and logs: shared details, the current car flat, and every active car."""
    s = normalize(slots)
    view = {k: v for k, v in s.items() if k not in INTERNAL_KEYS and k != "cars"}
    active = _active(s)
    if len(active) > 1:
        view["cars"] = [{k: v for k, v in c.items() if v is not None} for c in active]
    return view


# ---- merging -------------------------------------------------------------------------

def _mentions_from(new: Slots) -> list[CarMention]:
    if new.cars:
        return list(new.cars)
    flat = {k: getattr(new, k) for k in CAR_KEYS}
    return [CarMention(**flat)] if any(_given(v) for v in flat.values()) else []


def merge_slots(old: dict | None, new: Slots, message: str | None = None) -> dict:
    """Merge the supervisor's new slots into the stored ones.

    Shared details: new values overwrite, missing ones are kept; phone normalised (an invalid one is dropped so
    the lead / support agent asks again). Cars: a car with a model updates that model's remembered entry (or is
    added); a car without a model ("and the automatic?") updates the current car. A newly added car never inherits
    another car's trim / gearbox / fuel, and, when other cars were already known, keeps one only if the customer's
    message (`message`) actually says it (Gemini sometimes copies the old car's details). Placeholder text
    ("null", "none"...) counts as missing; model and trim names are canonicalised ("Signature Plus" ->
    "Signature+"). The cars named in the message become the active cars; if none is named they stay as before.
    """
    repo = get_repository()
    s = normalize(old)
    cars: list[dict] = s["cars"]
    had_cars = bool(cars)

    for key in SHARED_KEYS:
        value = getattr(new, key)
        if not _given(value):
            continue
        value = _clean(value)
        if key == "phone":
            value = normalize_indian_mobile(value)
            if value is None:
                continue
        s[key] = value

    named: list[str] = []
    intents_by_car: dict[str, list[str]] = {}
    for mention in _mentions_from(new):
        given = {k: _clean(getattr(mention, k)) for k in CAR_KEYS if _given(getattr(mention, k))}
        if "model" in given:
            model = repo.get_model(given["model"])
            given["model"] = model.name if model else given["model"]
            car = _car_by_model(cars, given["model"])
            if car is None:
                unnamed = next((c for c in cars if not c.get("model")), None)  # "price of Signature AT" earlier
                car = unnamed if unnamed is not None else {k: None for k in CAR_KEYS}
                if unnamed is None:
                    if had_cars and message is not None:  # a switch: drop details Gemini copied from another car
                        for key in CAR_SPECIFIC_SLOTS:
                            if key in given and not _mentioned(key, given[key], message):
                                del given[key]
                    cars.append(car)
        else:
            current = _active(s)
            car = current[-1] if current else None
            if car is None:
                car = {k: None for k in CAR_KEYS}
                cars.append(car)
        car.update(given)
        if car.get("model") and car.get("variant"):
            model = repo.get_model(car["model"])
            if model:
                car["variant"] = repo.canonical_variant_name(model.id, car["variant"]) or car["variant"]
        if car.get("model") and car["model"] not in named:
            named.append(car["model"])
            if mention.intents:
                intents_by_car[car["model"]] = list(dict.fromkeys(mention.intents))

    if named:
        s["active_cars"] = named
        s["car_intents"] = intents_by_car
        cars.sort(key=lambda c: c.get("model") in named)  # most recent last (stable for the rest)
    s["cars"] = cars[-MAX_REMEMBERED_CARS:]
    s["active_cars"] = [m for m in s["active_cars"] if _car_by_model(s["cars"], m)] or \
                       ([s["cars"][-1].get("model")] if s["cars"] else [])
    current = _active(s)
    for key in CAR_KEYS:  # flat mirror of the current car
        s[key] = current[-1].get(key) if current else None
    return s
