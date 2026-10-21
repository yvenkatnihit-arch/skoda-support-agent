"""The details (slots) the supervisor extracts from the conversation.

Two kinds (ROADMAP section 12):
- shared details, one copy because they belong to the customer: city, state, area_or_pincode, lat, long, name, phone;
- car details, one entry per car in `cars`: model, variant, transmission, fuel (+ which intents the car is for),
  so a question can be about two cars ("Is the Kylaq cheaper than the Kushaq?").

Stored slots also keep the flat model / variant / transmission / fuel of the car being discussed (the last active
car), so single-car code reads them as before. Agents name these fields in required_slots / required_car_slots.
"""

from typing import Literal

from pydantic import BaseModel, Field

CAR_KEYS: tuple[str, ...] = ("model", "variant", "transmission", "fuel")
SHARED_KEYS: tuple[str, ...] = ("city", "state", "area_or_pincode", "lat", "long", "name", "phone")
MAX_REMEMBERED_CARS = 3


class CarMention(BaseModel):
    """One car the customer's latest message is about."""
    model: str | None = Field(default=None, description="Skoda model; null for a follow-up about the same car "
                                                        "(e.g. 'and the automatic?')")
    variant: str | None = None
    transmission: Literal["MT", "AT", "DSG"] | None = None
    fuel: Literal["petrol", "diesel", "electric"] | None = None
    intents: list[str] = Field(default_factory=list,
                               description="Which of the message's intents this car is for; empty = all of them")


class Slots(BaseModel):
    """The supervisor's output. Car details go in `cars`; the flat model/variant/transmission/fuel are still
    accepted (treated as one car) so older callers and single-car answers keep working."""
    model: str | None = None
    variant: str | None = None
    fuel: Literal["petrol", "diesel", "electric"] | None = None
    transmission: Literal["MT", "AT", "DSG"] | None = None
    city: str | None = None
    state: str | None = None
    area_or_pincode: str | None = None
    lat: float | None = None
    long: float | None = None
    name: str | None = None
    phone: str | None = None
    cars: list[CarMention] = Field(default_factory=list,
                                   description="Every car the latest message is about, one entry each")


SLOT_KEYS: tuple[str, ...] = (*CAR_KEYS, *SHARED_KEYS)
