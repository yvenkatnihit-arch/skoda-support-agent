import csv
import json
from datetime import date, timedelta

import httpx
import pytest

from app.tools import geo_tools, lead_tools, retrieval
from app.tools.data_tools import (
    calculate_on_road_price,
    calculate_range,
    get_service_cost,
    get_service_schedule,
    get_variant_details,
)
from app.tools.geo_tools import find_nearest_dealers, geocode_location
from app.tools.lead_tools import book_test_drive, normalize_indian_mobile, save_lead
from app.tools.registry import ALL_TOOLS, TOOLS_REQUIRING_APPROVAL, tools_for
from app.tools.retrieval import Chunk, IndexNotBuiltError, SearchHit, search_brochures, search_service_docs


def _json(result: str) -> dict:
    assert not result.startswith("ERROR"), result
    return json.loads(result)


# ---- registry ------------------------------------------------------------------

def test_registry_has_11_tools_with_docs():
    assert len(ALL_TOOLS) == 13 and len({t.name for t in ALL_TOOLS}) == 13
    assert all(t.description and len(t.description) > 40 for t in ALL_TOOLS)
    assert {t.name for t in tools_for("lead")} | {"create_ticket"} == TOOLS_REQUIRING_APPROVAL


# ---- data tools ----------------------------------------------------------------

def test_variant_details_lists_all_variants_without_a_trim():
    out = _json(get_variant_details.invoke({"model": "kylaq"}))
    assert len(out["variants"]) == 13 and out["model_status"] == "on_sale"


def test_variant_details_includes_features():
    out = _json(get_variant_details.invoke({"model": "Kylaq", "variant": "Signature", "transmission": "AT"}))
    [v] = out["variants"]
    assert v["ex_showroom_price"] == 1_049_000 and "6 airbags" in v["features"]["safety"]


def test_unknown_model_and_variant_come_back_as_errors():
    assert get_variant_details.invoke({"model": "Fabia"}).startswith("ERROR: 'Fabia' is not a Skoda model")
    err = get_variant_details.invoke({"model": "Kylaq", "variant": "Monte Carlo"})
    assert err.startswith("ERROR") and "Signature+ AT" in err  # lists the options


def test_on_road_price_returns_breakdown_with_sources():
    out = _json(calculate_on_road_price.invoke(
        {"model": "Kylaq", "variant": "Signature", "transmission": "AT", "city": "Hyderabad"}))
    assert out["state"] == "Telangana" and out["is_estimate"] is True
    assert out["estimated_on_road_price"] == sum(l["amount"] for l in out["breakdown"]) == 1_292_125
    assert all(l["source_url"] for l in out["breakdown"])


def test_on_road_price_errors():
    ambiguous = calculate_on_road_price.invoke({"model": "Kylaq", "variant": "Signature", "city": "Hyderabad"})
    assert "Signature MT, Signature AT" in ambiguous
    assert "not on sale" in calculate_on_road_price.invoke({"model": "Enyaq", "variant": "60", "city": "Delhi"})
    assert "which state" in calculate_on_road_price.invoke(
        {"model": "Kylaq", "variant": "Prestige+", "transmission": "AT", "city": "Panaji"})
    assert "Supported states" in calculate_on_road_price.invoke(
        {"model": "Kylaq", "variant": "Prestige+", "transmission": "AT", "city": "Panaji", "state": "Goa"})


def test_on_road_price_with_state_override_and_accessory():
    out = _json(calculate_on_road_price.invoke(
        {"model": "Kushaq", "variant": "Prestige", "transmission": "DSG", "city": "Mysuru", "state": "Karnataka",
         "accessories": ["Dashcam"]}))
    assert any(l["item"] == "Accessory: Dashcam" and l["amount"] == 10_999 for l in out["breakdown"])
    assert "Unknown accessory" in calculate_on_road_price.invoke(
        {"model": "Kushaq", "variant": "Prestige", "transmission": "DSG", "city": "Mumbai", "accessories": ["spoiler"]})


def test_range_per_powertrain_and_missing_figures():
    kylaq = _json(calculate_range.invoke({"model": "Kylaq"}))["ranges"]
    assert sorted(r["range_km"] for r in kylaq) == [857, 886]
    [rs] = _json(calculate_range.invoke({"model": "Octavia RS"}))["ranges"]
    assert rs["range_km"] is None and "not published" in rs["error"]


def test_service_cost_per_gearbox():
    out = _json(get_service_cost.invoke({"model": "Kushaq", "years": 3}))
    assert {c["transmission"] for c in out["costs"]} == {"MT", "AT", "DSG"}
    at = _json(get_service_cost.invoke({"model": "Kylaq", "years": 3, "transmission": "AT"}))["costs"][0]
    assert at["total_min"] == 18_996 and len(at["visits"]) == 5
    assert get_service_cost.invoke({"model": "Octavia RS", "years": 3}).startswith("ERROR: No service schedule")


def test_service_schedule_lists_visits():
    [sched] = _json(get_service_schedule.invoke({"model": "Kodiaq"}))["schedules"]
    assert sched["transmission"] == "DSG" and sched["visits"][0]["km"] == 1000


def test_unexpected_failure_is_returned_as_text(monkeypatch):
    def boom(*_):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("app.tools.data_tools.resolve_model", boom)
    out = get_variant_details.invoke({"model": "Kylaq"})
    assert out.startswith("ERROR: get_variant_details failed unexpectedly (RuntimeError)")


# ---- geo tools -----------------------------------------------------------------

@pytest.fixture
def geo(monkeypatch, tmp_path):
    """Offline geocoding: temp cache file and a scripted Nominatim."""
    calls = []
    script = {"result": None, "raise": None}

    def fake_search(query):
        calls.append(query)
        if script["raise"]:
            raise script["raise"]
        return script["result"]

    monkeypatch.setattr(geo_tools, "_cache_path", lambda: tmp_path / "geocode.json")
    monkeypatch.setattr(geo_tools, "_nominatim_search", fake_search)
    return calls, script


def test_gps_is_used_directly(geo):
    calls, _ = geo
    out = _json(geocode_location.invoke({"lat": 17.44, "long": 78.39}))
    assert out["source"] == "gps" and calls == []
    assert "outside India" in geocode_location.invoke({"lat": 51.5, "long": -0.1})
    assert "both lat and long" in geocode_location.invoke({"lat": 17.4})


def test_nominatim_result_is_cached(geo):
    calls, script = geo
    script["result"] = {"lat": 17.49, "long": 78.40, "display_name": "Kukatpally, Hyderabad"}
    first = _json(geocode_location.invoke({"location": "Kukatpally", "city": "Hyderabad"}))
    second = _json(geocode_location.invoke({"location": "kukatpally ", "city": "Hyderabad"}))
    assert first["source"] == "nominatim" and second["source"] == "cache"
    assert calls == ["Kukatpally, Hyderabad, India"]


def test_nominatim_down_falls_back_to_pincodes(geo):
    _, script = geo
    script["raise"] = httpx.ConnectError("offline")
    area = _json(geocode_location.invoke({"location": "Kukatpally", "city": "Hyderabad"}))
    assert area["source"] == "pincodes.csv" and "500072" in area["display_name"]
    pin = _json(geocode_location.invoke({"location": "560034"}))
    assert pin["source"] == "pincodes.csv" and "Koramangala" in pin["display_name"]
    city = _json(geocode_location.invoke({"city": "Mumbai"}))
    assert city["approximate"] is True


def test_unknown_place_asks_for_pincode(geo):
    _, script = geo
    script["result"] = None
    out = geocode_location.invoke({"location": "Nowhereville"})
    assert out.startswith("ERROR") and "pincode" in out
    assert "No location given" in geocode_location.invoke({})


def test_nearest_dealers_sorted_and_filtered():
    madhapur = (17.4483, 78.3915)
    out = _json(find_nearest_dealers.invoke({"lat": madhapur[0], "long": madhapur[1], "dealer_type": "showroom"}))
    dists = [d["distance_km"] for d in out["dealers"]]
    assert dists == sorted(dists) and len(dists) == 3
    assert all(d["type"] in ("showroom", "both") for d in out["dealers"])
    assert out["dealers"][0]["maps_link"].startswith("https://www.google.com/maps/search/?api=1&query=")
    assert "coverage_note" not in out


def test_far_away_position_gets_coverage_note():
    out = _json(find_nearest_dealers.invoke({"lat": 13.08, "long": 80.27, "limit": 1}))  # Chennai
    assert "coverage_note" in out


# ---- retrieval tools -----------------------------------------------------------

class FakeStore:
    def __init__(self, hits=None, exc=None):
        self.hits, self.exc, self.calls = hits or [], exc, []

    def search(self, query, k, *, models=None, doc_types=None, min_score=0.0, mode="hybrid", candidates=10,
               rrf_k=60):
        self.calls.append({"query": query, "models": models, "doc_types": doc_types, "min_score": min_score,
                           "mode": mode})
        if self.exc:
            raise self.exc
        return self.hits


def _hit(model="kushaq"):
    chunk = Chunk(id=0, text="Six airbags", model=model, doc_type="brochure", source="brochure/x.md",
                  section="Kushaq > Safety")
    return SearchHit(0.8, chunk)


def test_search_brochures_filters_by_model_and_type(monkeypatch):
    store = FakeStore(hits=[_hit()])
    monkeypatch.setattr(retrieval, "get_vector_store", lambda: store)
    out = _json(search_brochures.invoke({"query": "airbags", "model": "Kushaq"}))
    assert out["results"][0]["source"] == "brochure/x.md"
    assert store.calls[0]["models"] == ["kushaq", "all"]
    assert store.calls[0]["doc_types"] == ["brochure", "specs", "faq"]
    assert store.calls[0]["min_score"] > 0


def test_search_service_docs_uses_service_types_and_no_results(monkeypatch):
    store = FakeStore(hits=[])
    monkeypatch.setattr(retrieval, "get_vector_store", lambda: store)
    out = search_service_docs.invoke({"query": "warranty"})
    assert out.startswith("NO_RESULTS")
    assert store.calls[0]["doc_types"] == ["service", "warranty", "faq"] and store.calls[0]["models"] is None


def test_search_errors_are_text(monkeypatch):
    monkeypatch.setattr(retrieval, "get_vector_store", lambda: FakeStore(exc=IndexNotBuiltError("no index")))
    assert search_brochures.invoke({"query": "sunroof"}).startswith("ERROR: document search is unavailable")
    assert search_brochures.invoke({"query": "sunroof", "model": "Fabia"}).startswith("ERROR")


# ---- lead tools ----------------------------------------------------------------

@pytest.fixture
def leads_file(monkeypatch, tmp_path):
    path = tmp_path / "leads.csv"
    monkeypatch.setattr(lead_tools, "_leads_path", lambda: path)
    return path


def _rows(path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


@pytest.mark.parametrize("raw,expected", [
    ("9876543210", "9876543210"), ("+91 98765 43210", "9876543210"), ("09876543210", "9876543210"),
    ("919876543210", "9876543210"), ("1234567890", None), ("98765", None), ("", None),
])
def test_normalize_indian_mobile(raw, expected):
    assert normalize_indian_mobile(raw) == expected


def test_save_lead_writes_csv_and_masks_phone_in_result(leads_file):
    out = _json(save_lead.invoke({"name": "Ravi Kumar", "phone": "+91 98765 43210", "city": "Hyderabad",
                                  "model_of_interest": "kylaq", "request_type": "brochure"}))
    assert out["status"] == "saved" and out["phone"] == "XXXXXXXX10"
    [row] = _rows(leads_file)
    assert row["phone"] == "9876543210" and row["model"] == "Kylaq" and row["request_type"] == "brochure"


def test_save_lead_rejects_bad_phone_without_writing(leads_file):
    assert "valid 10-digit" in save_lead.invoke({"name": "Ravi", "phone": "12345", "city": "Pune"})
    assert not leads_file.exists()


def test_formula_injection_is_neutralised(leads_file):
    save_lead.invoke({"name": "=HYPERLINK(1)", "phone": "9876543210", "city": "Pune"})
    assert _rows(leads_file)[0]["name"].startswith("'=")


def test_book_test_drive_validates_date_model_and_dealer(leads_file):
    base = {"name": "Asha", "phone": "9123456789", "city": "Hyderabad", "model": "Kylaq"}
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    assert "in the past" in book_test_drive.invoke({**base, "preferred_date": yesterday})
    assert "not on sale" in book_test_drive.invoke({**base, "model": "Enyaq"})
    assert "Unknown showroom" in book_test_drive.invoke({**base, "dealer_id": "dlr-68888"})  # a service centre
    assert not leads_file.exists()

    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    out = _json(book_test_drive.invoke({**base, "preferred_date": tomorrow, "dealer_id": "dlr-538646"}))
    assert out["preferred_date"] == tomorrow
    [row] = _rows(leads_file)
    assert row["request_type"] == "test_drive" and row["dealer_id"] == "dlr-538646"
