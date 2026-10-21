"""Data-access layer over the seed files in data/seed/.

Files are validated with the pydantic models in schemas.py and held as pandas DataFrames.
Agents and tools only use the query methods here, so a real database can replace the files later.
"""

import json
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

import pandas as pd
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.data.schemas import (
    CarModel,
    Dealer,
    Pincode,
    PricingRules,
    RtoRate,
    ServiceScheduleItem,
    Variant,
    VariantFeature,
)

R = TypeVar("R", bound=BaseModel)

MAX_REPORTED_ERRORS = 20


class DataValidationError(Exception):
    """A seed file is missing or has malformed rows. Lists every problem found in that file."""

    def __init__(self, file: Path | str, errors: list[str]):
        self.file = Path(file).name
        self.errors = errors
        shown = errors[:MAX_REPORTED_ERRORS]
        more = f"\n  ... and {len(errors) - len(shown)} more" if len(errors) > len(shown) else ""
        super().__init__(f"Invalid data in {self.file}:\n  " + "\n  ".join(shown) + more)


def _format_errors(prefix: str, exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"]) or "row"
        out.append(f"{prefix}: {field}: {err['msg']}")
    return out


def _read_csv(path: Path, schema: type[R]) -> list[R]:
    if not path.exists():
        raise DataValidationError(path, ["file not found"])
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    columns = set(df.columns)
    fields = schema.model_fields
    required = {name for name, f in fields.items() if f.is_required()}
    errors = [f"missing column '{c}'" for c in sorted(required - columns)]
    errors += [f"unknown column '{c}'" for c in sorted(columns - set(fields))]
    if errors:
        raise DataValidationError(path, errors)

    records: list[R] = []
    for i, row in enumerate(df.to_dict("records")):
        clean = {k: v for k, v in row.items() if v.strip() != ""}  # blank cell -> field default
        try:
            records.append(schema.model_validate(clean))
        except ValidationError as exc:
            errors += _format_errors(f"line {i + 2}", exc)  # +2: header row and 1-based lines
    if errors:
        raise DataValidationError(path, errors)
    return records


def _read_json_list(path: Path, schema: type[R]) -> list[R]:
    if not path.exists():
        raise DataValidationError(path, ["file not found"])
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataValidationError(path, [f"invalid JSON: {exc}"]) from exc
    if not isinstance(raw, list):
        raise DataValidationError(path, ["expected a JSON list of objects"])
    records, errors = [], []
    for i, item in enumerate(raw):
        try:
            records.append(schema.model_validate(item))
        except ValidationError as exc:
            errors += _format_errors(f"item {i}", exc)
    if errors:
        raise DataValidationError(path, errors)
    return records


def _read_json_object(path: Path, schema: type[R]) -> R:
    if not path.exists():
        raise DataValidationError(path, ["file not found"])
    try:
        return schema.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise DataValidationError(path, _format_errors("object", exc)) from exc


def _check_unique(file: str, records: Iterable[BaseModel], key: str) -> None:
    seen, dupes = set(), []
    for r in records:
        value = getattr(r, key)
        if value in seen:
            dupes.append(f"duplicate {key} '{value}'")
        seen.add(value)
    if dupes:
        raise DataValidationError(file, dupes)


def _check_refs(file: str, records: Iterable[BaseModel], key: str, valid: set[str], target: str) -> None:
    bad = sorted({getattr(r, key) for r in records} - valid)
    if bad:
        raise DataValidationError(file, [f"{key} '{b}' not found in {target}" for b in bad])


def _norm(text: str) -> str:
    """Normalise a name for matching: lower case, 'plus' -> '+', no spaces/hyphens/dots."""
    text = text.lower().replace("škoda", "").replace("skoda", "")
    text = re.sub(r"\s*plus\b", "+", text)
    return re.sub(r"[\s\-_.]", "", text)


def _to_frame(records: list[BaseModel]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in records])


class Repository:
    """Loads and validates all seed files once, then answers queries from memory."""

    def __init__(self, seed_dir: Path):
        self.seed_dir = Path(seed_dir)
        d = self.seed_dir

        models = _read_json_list(d / "models.json", CarModel)
        variants = _read_csv(d / "variants.csv", Variant)
        features = _read_csv(d / "variant_features.csv", VariantFeature)
        rto_rates = _read_csv(d / "rto_rates.csv", RtoRate)
        schedule = _read_csv(d / "service_schedule.csv", ServiceScheduleItem)
        dealers = _read_csv(d / "dealers.csv", Dealer)
        pincodes = _read_csv(d / "pincodes.csv", Pincode)
        self.pricing_rules = _read_json_object(d / "pricing_rules.json", PricingRules)

        self._validate(models, variants, features, rto_rates, schedule, dealers, pincodes)

        self._models = {m.id: m for m in models}
        self._variants = {v.id: v for v in variants}
        self._features = features
        self._rto_rates = rto_rates
        self._schedule = schedule
        self._dealers = dealers
        self._pincodes = pincodes

        # DataFrames for analysis and bulk queries.
        self.models = _to_frame(models)
        self.variants = _to_frame(variants)
        self.features = _to_frame(features)
        self.rto_rates = _to_frame(rto_rates)
        self.service_schedule = _to_frame(schedule)
        self.dealers = _to_frame(dealers)
        self.pincodes = _to_frame(pincodes)

    def _validate(self, models, variants, features, rto_rates, schedule, dealers, pincodes) -> None:
        _check_unique("models.json", models, "id")
        _check_unique("variants.csv", variants, "id")
        _check_unique("dealers.csv", dealers, "id")
        _check_unique("pincodes.csv", pincodes, "pincode")
        _check_unique("pricing_rules.json", self.pricing_rules.accessories, "id")

        model_ids = {m.id for m in models}
        _check_refs("variants.csv", variants, "model_id", model_ids, "models.json")
        _check_refs("variant_features.csv", features, "variant_id", {v.id for v in variants}, "variants.csv")
        _check_refs("service_schedule.csv", schedule, "model_id", model_ids, "models.json")
        _check_refs("pricing_rules.json", self.pricing_rules.accessories, "model_id", model_ids, "models.json")

        combos: dict[tuple, str] = {}
        errors = []
        for v in variants:
            key = (v.model_id, _norm(v.name), v.transmission)
            if key in combos:
                errors.append(f"variants '{combos[key]}' and '{v.id}' have the same model, name and transmission")
            combos[key] = v.id
        if errors:
            raise DataValidationError("variants.csv", errors)

        bands: dict[tuple, list[RtoRate]] = {}
        for r in rto_rates:
            bands.setdefault((r.state.lower(), r.fuel), []).append(r)
        for (state, fuel), rows in bands.items():
            rows.sort(key=lambda r: r.price_band_min)
            for a, b in zip(rows, rows[1:]):
                if b.price_band_min < a.price_band_max:
                    errors.append(f"{a.state}/{fuel}: band {b.price_band_min}-{b.price_band_max} overlaps "
                                  f"{a.price_band_min}-{a.price_band_max}")
        if errors:
            raise DataValidationError("rto_rates.csv", errors)

    # ---- models and variants -------------------------------------------------

    def list_models(self) -> list[CarModel]:
        return list(self._models.values())

    def get_model(self, name_or_id: str) -> CarModel | None:
        """Match a model by id or name, ignoring case and spacing; a unique prefix also matches."""
        key = _norm(name_or_id)
        if not key:
            return None
        exact = [m for m in self._models.values() if key in (_norm(m.id), _norm(m.name))]
        if exact:
            return exact[0]
        prefix = [m for m in self._models.values() if _norm(m.name).startswith(key)]
        return prefix[0] if len(prefix) == 1 else None

    def get_variant(self, variant_id: str) -> Variant | None:
        return self._variants.get(variant_id)

    def find_variants(
        self,
        model_id: str,
        name: str | None = None,
        fuel: str | None = None,
        transmission: str | None = None,
    ) -> list[Variant]:
        """Variants of a model, optionally filtered by trim name, fuel and transmission.

        The trim name matches exactly after normalisation ("Signature Plus" == "Signature+").
        If nothing matches exactly, trims containing the given name are returned instead.
        """
        rows = [v for v in self._variants.values() if v.model_id == model_id]
        if fuel:
            rows = [v for v in rows if v.fuel == fuel.lower()]
        if transmission:
            rows = [v for v in rows if v.transmission == transmission.upper()]
        if name:
            key = _norm(name)
            exact = [v for v in rows if _norm(v.name) == key]
            rows = exact or [v for v in rows if key in _norm(v.name)]
        return sorted(rows, key=lambda v: v.ex_showroom_price)

    def canonical_variant_name(self, model_id: str, name: str) -> str | None:
        """The official trim name ("Signature Plus" -> "Signature+") if it matches one of the model's trims
        exactly after normalisation; None otherwise."""
        key = _norm(name)
        return next((v.name for v in self._variants.values() if v.model_id == model_id and _norm(v.name) == key),
                    None)

    def get_features(self, variant_id: str) -> list[VariantFeature]:
        return [f for f in self._features if f.variant_id == variant_id]

    # ---- pricing ---------------------------------------------------------------

    def get_rto_rate(self, state: str, fuel: str, ex_showroom_price: int) -> RtoRate | None:
        state_key, fuel_key = state.strip().lower(), fuel.strip().lower()
        for r in self._rto_rates:
            if (r.state.lower() == state_key and r.fuel == fuel_key
                    and r.price_band_min <= ex_showroom_price < r.price_band_max):
                return r
        return None

    def list_rto_states(self) -> list[str]:
        return sorted({r.state for r in self._rto_rates})

    def accessories_for(self, model_id: str):
        return [a for a in self.pricing_rules.accessories if a.model_id == model_id]

    # ---- service ---------------------------------------------------------------

    def get_service_schedule(self, model_id: str, fuel: str, transmission: str) -> list[ServiceScheduleItem]:
        rows = [s for s in self._schedule
                if s.model_id == model_id and s.fuel == fuel.lower() and s.transmission == transmission.upper()]
        return sorted(rows, key=lambda s: s.interval_km)

    # ---- locations -------------------------------------------------------------

    def list_dealers(self, dealer_type: str | None = None, city: str | None = None) -> list[Dealer]:
        """Dealers, optionally filtered. dealer_type 'showroom' or 'service' also includes 'both'."""
        rows = self._dealers
        if dealer_type:
            wanted = dealer_type.lower()
            rows = [d for d in rows if d.type == wanted or (d.type == "both" and wanted != "both")]
        if city:
            rows = [d for d in rows if d.city.lower() == city.strip().lower()]
        return list(rows)

    def find_pincode(self, pincode: str) -> Pincode | None:
        pincode = pincode.strip()
        return next((p for p in self._pincodes if p.pincode == pincode), None)

    def list_pincodes(self, city: str | None = None) -> list[Pincode]:
        if city is None:
            return list(self._pincodes)
        return [p for p in self._pincodes if p.city.lower() == city.strip().lower()]

    def find_area(self, area: str, city: str | None = None) -> list[Pincode]:
        key = _norm(area)
        rows = [p for p in self._pincodes if _norm(p.area) == key]
        if city:
            rows = [p for p in rows if p.city.lower() == city.strip().lower()]
        return rows

    def state_for_city(self, city: str) -> str | None:
        key = city.strip().lower()
        for row in (*self._pincodes, *self._dealers):
            if row.city.lower() == key:
                return row.state
        return None


@lru_cache
def get_repository() -> Repository:
    return Repository(get_settings().seed_dir)


if __name__ == "__main__":
    repo = get_repository()
    for name in ("models", "variants", "features", "rto_rates", "service_schedule", "dealers", "pincodes"):
        print(f"{name:17} {len(getattr(repo, name)):5} rows")
    print(f"{'accessories':17} {len(repo.pricing_rules.accessories):5} rows")
