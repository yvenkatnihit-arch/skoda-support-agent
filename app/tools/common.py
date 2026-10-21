"""Helpers shared by all tools: error-as-text wrapper, JSON output, model/variant resolution."""

import functools
import json
from collections.abc import Callable

import structlog

from app.data.repository import get_repository
from app.data.schemas import CarModel, Variant

log = structlog.get_logger()


class ToolInputError(ValueError):
    """A problem the customer or the agent can fix (unknown variant, missing detail, ...)."""


def returns_errors_as_text(fn: Callable[..., str]) -> Callable[..., str]:
    """Never let a tool crash the agent: known problems and unexpected failures come back as text."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> str:
        try:
            return fn(*args, **kwargs)
        except (ToolInputError, ValueError) as exc:
            return f"ERROR: {exc}"
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all, logged with traceback
            log.exception("tool failed", tool=fn.__name__)
            return f"ERROR: {fn.__name__} failed unexpectedly ({type(exc).__name__}). Offer the customer a callback."

    return wrapper


def to_json(data) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def resolve_model(name: str) -> CarModel:
    repo = get_repository()
    model = repo.get_model(name)
    if model is None:
        names = ", ".join(m.name for m in repo.list_models())
        raise ToolInputError(f"'{name}' is not a Skoda model we have data for. Models: {names}.")
    return model


def resolve_variants(model: CarModel, variant: str | None, fuel: str | None,
                     transmission: str | None) -> list[Variant]:
    """Variants matching the filters. Raises with the available options when nothing matches."""
    repo = get_repository()
    rows = repo.find_variants(model.id, name=variant, fuel=fuel, transmission=transmission)
    if rows:
        return rows
    options = sorted({f"{v.name} {v.transmission}" for v in repo.find_variants(model.id)})
    wanted = " ".join(x for x in (variant, fuel, transmission) if x)
    raise ToolInputError(f"No {model.name} variant matches '{wanted}'. Available: {', '.join(options)}.")


def resolve_one_variant(model: CarModel, variant: str, fuel: str | None, transmission: str | None) -> Variant:
    rows = resolve_variants(model, variant, fuel, transmission)
    if len(rows) > 1:
        options = ", ".join(f"{v.name} {v.transmission}" for v in rows)
        raise ToolInputError(f"'{variant}' matches several {model.name} variants ({options}). "
                             "Ask the customer which transmission they want.")
    return rows[0]


def variant_summary(v: Variant) -> dict:
    return {
        "variant_id": v.id,
        "name": v.name,
        "fuel": v.fuel,
        "transmission": v.transmission,
        "engine_cc": v.engine_cc,
        "power_ps": v.power_ps,
        "torque_nm": v.torque_nm,
        "claimed_mileage_kmpl": v.mileage_kmpl,
        "battery_kwh": v.battery_kwh,
        "claimed_range_km": v.claimed_range_km,
        "fuel_tank_l": v.fuel_tank_l,
        "ex_showroom_price": v.ex_showroom_price,
        "price_city": v.price_city,
        "price_as_of": v.price_as_of,
        "price_source_url": v.source_url,
        "spec_source_url": v.spec_source_url,
    }
