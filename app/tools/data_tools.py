"""Read-only tools over the structured seed data: variants, on-road price, range, service."""

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.data.repository import get_repository
from app.tools import calculators
from app.tools.common import (
    ToolInputError,
    resolve_model,
    resolve_one_variant,
    resolve_variants,
    returns_errors_as_text,
    to_json,
    variant_summary,
)

FuelArg = Literal["petrol", "diesel", "electric"]
TransmissionArg = Literal["MT", "AT", "DSG"]


class VariantArgs(BaseModel):
    model: str = Field(description="Skoda model name, e.g. 'Kylaq', 'Kushaq', 'Slavia', 'Kodiaq', 'Octavia RS'")
    variant: str | None = Field(default=None, description="Trim name, e.g. 'Signature+', 'Prestige', 'Monte Carlo'. "
                                                          "Leave empty to list all variants of the model.")
    fuel: FuelArg | None = Field(default=None, description="Fuel type, if the customer said one")
    transmission: TransmissionArg | None = Field(
        default=None, description="MT = manual, AT = torque-converter automatic, DSG = dual-clutch automatic")


@tool("get_variant_details", args_schema=VariantArgs)
@returns_errors_as_text
def get_variant_details(model: str, variant: str | None = None, fuel: str | None = None,
                        transmission: str | None = None) -> str:
    """Look up Skoda variants: ex-showroom price, engine, power, torque, mileage, tank and key features.
    Without a variant it lists every variant of the model. Use it to confirm the exact variant before
    quoting, and to answer spec questions about a specific variant."""
    car = resolve_model(model)
    rows = resolve_variants(car, variant, fuel, transmission)
    repo = get_repository()
    out = {"model": car.name, "model_status": car.status, "model_notes": car.notes, "is_estimate": False}
    if car.status == "upcoming":
        out["warning"] = "This model is not launched in India yet. Figures are media estimates, not official prices."
    if variant is None:
        out["variants"] = [{"variant_id": v.id, "name": v.name, "fuel": v.fuel, "transmission": v.transmission,
                            "ex_showroom_price": v.ex_showroom_price, "price_as_of": v.price_as_of} for v in rows]
        out["price_source_url"] = rows[0].source_url
        cheapest = min(rows, key=lambda v: v.ex_showroom_price)  # computed here so the model never picks it
        out["starting_ex_showroom_price"] = cheapest.ex_showroom_price
        out["starting_variant"] = f"{cheapest.name} {cheapest.transmission}"
        out["starting_price_city"] = cheapest.price_city
        return to_json(out)
    details = []
    for v in rows:
        features: dict[str, list[str]] = {}
        for f in repo.get_features(v.id):
            features.setdefault(f.category, []).append(f.feature)
        details.append({**variant_summary(v), "features": features})
    out["variants"] = details
    return to_json(out)


class OnRoadArgs(BaseModel):
    model: str = Field(description="Skoda model name, e.g. 'Kylaq'")
    variant: str = Field(description="Trim name, e.g. 'Signature', 'Prestige+'")
    city: str = Field(description="Customer's city, e.g. 'Hyderabad'")
    state: str | None = Field(default=None, description="State, only if the customer gave it or the city is ambiguous")
    fuel: FuelArg | None = None
    transmission: TransmissionArg | None = Field(
        default=None, description="MT, AT or DSG. Needed when the trim is sold with more than one gearbox.")
    accessories: list[str] = Field(default_factory=list,
                                   description="Optional accessory names or ids the customer asked to include")


@tool("calculate_on_road_price", args_schema=OnRoadArgs)
@returns_errors_as_text
def calculate_on_road_price(model: str, variant: str, city: str, state: str | None = None,
                            fuel: str | None = None, transmission: str | None = None,
                            accessories: list[str] | None = None) -> str:
    """Estimate the on-road price of one exact variant in a city: ex-showroom + road tax (by state) + cess +
    registration + insurance + TCS + FASTag + optional accessories. Returns every line with its source.
    The result is an ESTIMATE; always tell the customer to confirm the final price with the dealer."""
    repo = get_repository()
    car = resolve_model(model)
    if car.status != "on_sale":
        raise ToolInputError(f"The {car.name} is not on sale in India yet, so there is no on-road price. "
                             f"{car.notes}")
    v = resolve_one_variant(car, variant, fuel, transmission)

    state_name = state or repo.state_for_city(city)
    if state_name is None:
        raise ToolInputError(f"I don't know which state '{city}' is in. Ask the customer for their state.")
    rto = repo.get_rto_rate(state_name, v.fuel, v.ex_showroom_price)
    if rto is None:
        raise ToolInputError(f"No road tax data for {state_name}. Supported states: "
                             f"{', '.join(repo.list_rto_states())}.")

    chosen = []
    available = repo.accessories_for(car.id)
    for wanted in accessories or []:
        key = wanted.strip().lower()
        match = next((a for a in available if key in (a.id.lower(), a.name.lower())), None)
        if match is None:
            names = ", ".join(a.name for a in available) or "none in our data"
            raise ToolInputError(f"Unknown accessory '{wanted}' for {car.name}. Available: {names}.")
        chosen.append(match)

    quote = calculators.on_road_price(v, rto.state, rto, repo.pricing_rules, chosen)
    return to_json({
        "model": car.name,
        "variant": f"{v.name} {v.transmission}",
        "variant_id": v.id,
        "fuel": v.fuel,
        "city": city,
        "state": rto.state,
        "is_estimate": True,
        "data_as_of": str(repo.pricing_rules.as_of),
        "ex_showroom_note": f"Ex-showroom price is the {v.price_city} figure; it can differ slightly in {city}.",
        "breakdown": [{"item": l.item, "amount": l.amount, "detail": l.detail, "source_url": l.source_url,
                       "is_assumption": l.is_assumption} for l in quote.lines],
        "estimated_on_road_price": quote.total,
        "accessories_available": [{"name": a.name, "price": a.price} for a in available if a not in chosen],
    })


@tool("calculate_range", args_schema=VariantArgs)
@returns_errors_as_text
def calculate_range(model: str, variant: str | None = None, fuel: str | None = None,
                    transmission: str | None = None) -> str:
    """Estimated driving range on a full tank (claimed mileage x fuel tank) or full charge (claimed EV range)
    for the matching variants. Without a variant it covers every powertrain of the model."""
    car = resolve_model(model)
    rows = resolve_variants(car, variant, fuel, transmission)
    results, seen = [], set()
    for v in rows:
        key = (v.fuel, v.transmission, v.mileage_kmpl, v.fuel_tank_l, v.claimed_range_km)
        if key in seen:
            continue  # same powertrain figures as a variant already listed
        seen.add(key)
        if variant:
            label = f"{v.name} {v.transmission}"
        else:  # whole model: label by powertrain, not by whichever trim came first
            engine = f"{v.engine_cc} cc" if v.engine_cc else f"{v.battery_kwh:g} kWh"
            label = f"{v.fuel} {v.transmission}, {engine}"
        try:
            r = calculators.driving_range(v)
            results.append({"variant": label, "range_km": r.range_km,
                            "method": r.method, "detail": r.detail, "source_url": v.spec_source_url})
        except ValueError as exc:
            results.append({"variant": label, "range_km": None, "error": str(exc)})
    return to_json({"model": car.name, "is_estimate": True, "ranges": results})


class ServiceArgs(BaseModel):
    model: str = Field(description="Skoda model name, e.g. 'Kushaq'")
    fuel: FuelArg | None = None
    transmission: TransmissionArg | None = Field(
        default=None, description="MT, AT or DSG. Leave empty to get every gearbox of the model.")


class ServiceCostArgs(ServiceArgs):
    years: int = Field(ge=1, le=5, description="Number of years of ownership to estimate, 1 to 5")


def _schedules(model: str, fuel: str | None, transmission: str | None):
    repo = get_repository()
    car = resolve_model(model)
    combos = sorted({(v.fuel, v.transmission) for v in repo.find_variants(car.id, fuel=fuel,
                                                                            transmission=transmission)})
    if not combos:
        raise ToolInputError(f"No {car.name} variant with {fuel or ''} {transmission or ''}".strip() + ".")
    out = []
    for f, t in combos:
        rows = repo.get_service_schedule(car.id, f, t)
        out.append((f, t, rows))
    if not any(rows for _, _, rows in out):
        raise ToolInputError(f"No service schedule is available for the {car.name}. "
                             "Suggest the customer asks a service centre.")
    return car, out


@tool("get_service_cost", args_schema=ServiceCostArgs)
@returns_errors_as_text
def get_service_cost(model: str, years: int, fuel: str | None = None, transmission: str | None = None) -> str:
    """Estimated total scheduled-service cost for the first 1-5 years (assumes up to 15,000 km a year),
    per gearbox, with every visit listed. Repairs, tyres and wear items are not included."""
    car, schedules = _schedules(model, fuel, transmission)
    results = []
    for f, t, rows in schedules:
        if not rows:
            results.append({"fuel": f, "transmission": t, "error": "no schedule available"})
            continue
        est = calculators.service_cost(rows, years)
        results.append({
            "fuel": f, "transmission": t,
            "total_min": est.total_min, "total_max": est.total_max,
            "cumulative_cost_by_year": calculators.cumulative_cost_by_year(rows),
            "visits": [{"km": s.interval_km, "months": s.interval_months, "type": s.service_type,
                        "cost_min": s.est_cost_min, "cost_max": s.est_cost_max} for s in est.visits],
            "as_of": str(rows[0].as_of), "source_url": rows[0].source_url,
        })
    return to_json({"model": car.name, "years": years, "is_estimate": True, "costs": results})


@tool("get_service_schedule", args_schema=ServiceArgs)
@returns_errors_as_text
def get_service_schedule(model: str, fuel: str | None = None, transmission: str | None = None) -> str:
    """Scheduled service visits for a model (km, months, type and estimated cost of each visit), per gearbox,
    plus the running total cost after each year (use cumulative_cost_by_year for 'cost for N years')."""
    car, schedules = _schedules(model, fuel, transmission)
    results = [{
        "fuel": f, "transmission": t,
        "visits": [{"km": s.interval_km, "months": s.interval_months, "type": s.service_type,
                    "cost_min": s.est_cost_min, "cost_max": s.est_cost_max} for s in rows],
        "cumulative_cost_by_year": calculators.cumulative_cost_by_year(rows) if rows else {},
        "as_of": str(rows[0].as_of) if rows else None,
        "source_url": rows[0].source_url if rows else None,
    } for f, t, rows in schedules]
    return to_json({"model": car.name, "is_estimate": True, "schedules": results})


QUOTATION_TOOLS = [get_variant_details, calculate_on_road_price, calculate_range, get_service_cost]
