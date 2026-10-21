"""Pure calculation functions. No I/O and no LLM: every number shown to a customer is computed here.

Money is whole rupees, rounded half-up. Functions raise ValueError with a customer-readable
message when the input can't be calculated; the tool layer turns that into an error string.
"""

import math
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

from app.data.schemas import Accessory, PricingRules, RtoRate, ServiceScheduleItem, Variant

EARTH_RADIUS_KM = 6371.0088


def rupees(amount: float | Decimal) -> int:
    return int(Decimal(str(amount)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def percent_of(amount: int, percent: float) -> int:
    return rupees(Decimal(amount) * Decimal(str(percent)) / Decimal(100))


# ---- on-road price -------------------------------------------------------------

@dataclass
class PriceLine:
    item: str
    amount: int
    detail: str
    source_url: str
    is_assumption: bool = False


@dataclass
class OnRoadQuote:
    variant_id: str
    state: str
    ex_showroom_price: int
    lines: list[PriceLine] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(line.amount for line in self.lines)


def on_road_price(
    variant: Variant,
    state: str,
    rto: RtoRate,
    rules: PricingRules,
    accessories: list[Accessory] = (),
) -> OnRoadQuote:
    """Estimated on-road price: ex-showroom + road tax + cess + registration + insurance + TCS + FASTag
    + handling + chosen accessories."""
    if rto.fuel != variant.fuel or not (rto.price_band_min <= variant.ex_showroom_price < rto.price_band_max):
        raise ValueError("road tax band does not match this variant's fuel and price")
    ex = variant.ex_showroom_price
    q = OnRoadQuote(variant_id=variant.id, state=state, ex_showroom_price=ex)

    q.lines.append(PriceLine("Ex-showroom price", ex, f"{variant.price_city} ex-showroom, as of {variant.price_as_of}",
                             variant.source_url))

    road_tax = percent_of(ex, rto.rate_percent)
    q.lines.append(PriceLine("Road tax (RTO)", road_tax,
                             f"{rto.rate_percent:g}% of ex-showroom ({state}, {variant.fuel})", rto.source_url))
    cess = percent_of(road_tax, rto.cess_percent_of_tax) + rto.flat_cess
    if cess:
        parts = []
        if rto.cess_percent_of_tax:
            parts.append(f"{rto.cess_percent_of_tax:g}% of road tax")
        if rto.flat_cess:
            parts.append(f"flat Rs {rto.flat_cess}")
        q.lines.append(PriceLine("Road tax cess", cess, " + ".join(parts), rto.source_url))

    reg = rules.registration
    q.lines.append(PriceLine("Registration + number plates (HSRP)", reg.registration_fee + reg.hsrp_fee,
                             f"registration Rs {reg.registration_fee} + HSRP Rs {reg.hsrp_fee}", reg.source_url))

    ins = rules.insurance
    ins_pct = ins.percent_of_ex_showroom.get(variant.fuel)
    if ins_pct is None:
        raise ValueError(f"no insurance estimate configured for {variant.fuel} cars")
    q.lines.append(PriceLine("Insurance (1st year, estimate)", percent_of(ex, ins_pct),
                             f"about {ins_pct:g}% of ex-showroom; actual premium depends on insurer and add-ons",
                             ins.source_url, ins.is_assumption))

    tcs = rules.tcs
    if ex > tcs.threshold:
        q.lines.append(PriceLine("TCS (income tax collected at source)", percent_of(ex, tcs.rate_percent),
                                 f"{tcs.rate_percent:g}% because ex-showroom exceeds Rs {tcs.threshold}; "
                                 "claimable against your income tax", tcs.source_url))

    if rules.fastag.fee:
        q.lines.append(PriceLine("FASTag", rules.fastag.fee, rules.fastag.note or "issuance + deposit",
                                 rules.fastag.source_url, rules.fastag.is_assumption))
    if rules.handling_charges.fee:
        q.lines.append(PriceLine("Handling charges", rules.handling_charges.fee, rules.handling_charges.note,
                                 rules.handling_charges.source_url, rules.handling_charges.is_assumption))

    for acc in accessories:
        if acc.model_id != variant.model_id:
            raise ValueError(f"accessory '{acc.name}' is not for this model")
        q.lines.append(PriceLine(f"Accessory: {acc.name}", acc.price, "optional", acc.source_url))
    return q


# ---- range ---------------------------------------------------------------------

@dataclass
class RangeEstimate:
    variant_id: str
    range_km: int
    method: str  # "mileage x tank" or "claimed EV range"
    detail: str


def driving_range(variant: Variant) -> RangeEstimate:
    if variant.fuel == "electric":
        if variant.claimed_range_km is None:
            raise ValueError("claimed range is not published for this EV")
        return RangeEstimate(variant.id, rupees(variant.claimed_range_km), "claimed EV range",
                             f"claimed range on a full {variant.battery_kwh:g} kWh battery")
    missing = [label for label, value in (("claimed mileage", variant.mileage_kmpl),
                                          ("fuel tank capacity", variant.fuel_tank_l)) if value is None]
    if missing:
        raise ValueError(f"cannot calculate range: {' and '.join(missing)} not published for this variant")
    km = rupees(Decimal(str(variant.mileage_kmpl)) * Decimal(str(variant.fuel_tank_l)))
    return RangeEstimate(variant.id, km, "mileage x tank",
                         f"{variant.mileage_kmpl:g} kmpl x {variant.fuel_tank_l:g} L tank (claimed figures; "
                         "real-world range is usually lower)")


# ---- service cost --------------------------------------------------------------

@dataclass
class ServiceCostEstimate:
    years: int
    visits: list[ServiceScheduleItem]

    @property
    def total_min(self) -> int:
        return sum(v.est_cost_min for v in self.visits)

    @property
    def total_max(self) -> int:
        return sum(v.est_cost_max for v in self.visits)


def cumulative_cost_by_year(schedule: list[ServiceScheduleItem]) -> dict[int, dict[str, int]]:
    """{year: {"min": total, "max": total}} for every whole year the schedule covers."""
    years = max((s.interval_months for s in schedule), default=0) // 12
    return {y: {"min": service_cost(schedule, y).total_min, "max": service_cost(schedule, y).total_max}
            for y in range(1, years + 1)}


def service_cost(schedule: list[ServiceScheduleItem], years: int) -> ServiceCostEstimate:
    """Sum of scheduled visits due within `years` (by the months interval)."""
    if not schedule:
        raise ValueError("no service schedule available")
    if not 1 <= years <= 10:
        raise ValueError("years must be between 1 and 10")
    covered_months = max(s.interval_months for s in schedule)
    if years * 12 > covered_months:
        raise ValueError(f"the service schedule only covers {covered_months // 12} years")
    visits = sorted((s for s in schedule if s.interval_months <= years * 12), key=lambda s: s.interval_km)
    return ServiceCostEstimate(years, visits)


# ---- distance ------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km (straight line, not road distance)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))
