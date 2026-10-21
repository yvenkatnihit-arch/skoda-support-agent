"""Pydantic models for every seed data file. Each row is validated on load."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Fuel = Literal["petrol", "diesel", "electric"]
Transmission = Literal["MT", "AT", "DSG"]
DealerType = Literal["showroom", "service", "both"]
ModelStatus = Literal["on_sale", "upcoming", "discontinued"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CarModel(Record):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    name: str = Field(min_length=1)
    body_type: str = Field(min_length=1)
    segment: str = Field(min_length=1)
    launch_year: int = Field(ge=1990, le=2100)
    is_ev: bool
    status: ModelStatus
    notes: str = ""


class Variant(Record):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    model_id: str
    name: str = Field(min_length=1)
    fuel: Fuel
    transmission: Transmission
    engine_cc: int | None = Field(default=None, gt=0)
    power_ps: float | None = Field(default=None, gt=0)
    torque_nm: float | None = Field(default=None, gt=0)
    # Optional: some figures are not published (e.g. Octavia RS has no ARAI mileage).
    mileage_kmpl: float | None = Field(default=None, gt=0)
    battery_kwh: float | None = Field(default=None, gt=0)
    claimed_range_km: float | None = Field(default=None, gt=0)
    fuel_tank_l: float | None = Field(default=None, gt=0)
    ex_showroom_price: int = Field(gt=0)
    price_city: str = Field(min_length=1)
    price_as_of: date
    source_url: str = Field(min_length=1)
    spec_source_url: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_powertrain_fields(self) -> "Variant":
        if self.fuel == "electric":
            missing = [f for f in ("battery_kwh", "claimed_range_km") if getattr(self, f) is None]
            if missing:
                raise ValueError(f"electric variant needs {', '.join(missing)}")
        elif self.engine_cc is None:
            raise ValueError(f"{self.fuel} variant needs engine_cc")
        return self


class VariantFeature(Record):
    variant_id: str
    category: str = Field(min_length=1)
    feature: str = Field(min_length=1)
    source_url: str = Field(min_length=1)


class RtoRate(Record):
    """Road tax for a state and fuel. The band is [price_band_min, price_band_max) on ex-showroom price."""

    state: str = Field(min_length=1)
    fuel: Fuel
    price_band_min: int = Field(ge=0)
    price_band_max: int = Field(gt=0)
    rate_percent: float = Field(ge=0, le=100)
    cess_percent_of_tax: float = Field(ge=0, le=100)
    flat_cess: int = Field(ge=0)
    source_url: str = Field(min_length=1)
    as_of: date

    @model_validator(mode="after")
    def check_band(self) -> "RtoRate":
        if self.price_band_min >= self.price_band_max:
            raise ValueError("price_band_min must be less than price_band_max")
        return self


class ServiceScheduleItem(Record):
    model_id: str
    fuel: Fuel
    transmission: Transmission
    interval_km: int = Field(gt=0)
    interval_months: int = Field(gt=0)
    service_type: str = Field(min_length=1)
    est_cost_min: int = Field(ge=0)
    est_cost_max: int = Field(ge=0)
    as_of: date
    source_url: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_costs(self) -> "ServiceScheduleItem":
        if self.est_cost_min > self.est_cost_max:
            raise ValueError("est_cost_min must not exceed est_cost_max")
        return self


class Dealer(Record):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    type: DealerType
    address: str = Field(min_length=1)
    city: str = Field(min_length=1)
    state: str = Field(min_length=1)
    pincode: str = Field(pattern=r"^\d{6}$")
    phone: str = Field(min_length=1)
    timings: str = Field(min_length=1)
    services_offered: list[str] = Field(min_length=1)
    latitude: float = Field(ge=6, le=38)  # India bounding box
    longitude: float = Field(ge=68, le=98)
    source_url: str = Field(min_length=1)
    as_of: date

    @field_validator("services_offered", mode="before")
    @classmethod
    def split_services(cls, value: object) -> object:
        if isinstance(value, str):
            return [s.strip() for s in value.split(";") if s.strip()]
        return value


class Pincode(Record):
    pincode: str = Field(pattern=r"^\d{6}$")
    area: str = Field(min_length=1)
    city: str = Field(min_length=1)
    state: str = Field(min_length=1)
    latitude: float = Field(ge=6, le=38)
    longitude: float = Field(ge=68, le=98)


class InsuranceRule(Record):
    percent_of_ex_showroom: dict[Fuel, float]
    source_url: str = Field(min_length=1)
    is_assumption: bool = False


class TcsRule(Record):
    threshold: int = Field(gt=0)
    rate_percent: float = Field(ge=0, le=100)
    note: str = ""
    source_url: str = Field(min_length=1)


class RegistrationRule(Record):
    registration_fee: int = Field(ge=0)
    hsrp_fee: int = Field(ge=0)
    note: str = ""
    source_url: str = Field(min_length=1)


class FeeRule(Record):
    fee: int = Field(ge=0)
    note: str = ""
    source_url: str = Field(min_length=1)
    is_assumption: bool = False


class Accessory(Record):
    id: str = Field(min_length=1)
    model_id: str
    name: str = Field(min_length=1)
    price: int = Field(ge=0)
    source_url: str = Field(min_length=1)


class PricingRules(Record):
    """Constants for the on-road price estimate that are not tied to one state or variant."""

    as_of: date
    insurance: InsuranceRule
    tcs: TcsRule
    registration: RegistrationRule
    fastag: FeeRule
    handling_charges: FeeRule
    accessories: list[Accessory]
