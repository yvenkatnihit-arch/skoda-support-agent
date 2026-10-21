from pathlib import Path

import pytest

from app.data.repository import Repository
from app.tools import calculators as calc

SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "seed"


@pytest.fixture(scope="module")
def repo() -> Repository:
    return Repository(SEED_DIR)


def _quote(repo, variant_id: str, state: str, price: int | None = None, accessories=()):
    v = repo.get_variant(variant_id)
    if price is not None:
        v = v.model_copy(update={"ex_showroom_price": price})
    rto = repo.get_rto_rate(state, v.fuel, v.ex_showroom_price)
    return calc.on_road_price(v, state, rto, repo.pricing_rules, list(accessories))


def _line(quote, prefix: str) -> int | None:
    return next((l.amount for l in quote.lines if l.item.startswith(prefix)), None)


def test_rupees_round_half_up():
    assert calc.rupees(0.5) == 1 and calc.rupees(2.5) == 3 and calc.rupees(2.4999) == 2


def test_on_road_kylaq_signature_at_hyderabad(repo):
    q = _quote(repo, "kylaq-signature-at", "Telangana")
    assert _line(q, "Ex-showroom") == 1_049_000
    assert _line(q, "Road tax (RTO)") == 188_820        # 18% band (10-20 lakh)
    assert _line(q, "Road tax cess") == 5_000           # Telangana flat road safety cess
    assert _line(q, "Registration") == 1_600
    assert _line(q, "Insurance") == 36_715              # 3.5% assumption
    assert _line(q, "TCS") == 10_490                    # 1% above 10 lakh
    assert _line(q, "FASTag") == 500
    assert q.total == 1_292_125


def test_tcs_only_when_price_exceeds_threshold(repo):
    at_threshold = _quote(repo, "kylaq-signature-at", "Telangana", price=1_000_000)
    above = _quote(repo, "kylaq-signature-at", "Telangana", price=1_000_001)
    assert _line(at_threshold, "TCS") is None
    assert _line(above, "TCS") == 10_000


def test_rto_band_edges_pick_the_right_rate(repo):
    below = _quote(repo, "kylaq-signature-at", "Telangana", price=999_999)
    at = _quote(repo, "kylaq-signature-at", "Telangana", price=1_000_000)
    assert _line(below, "Road tax (RTO)") == calc.percent_of(999_999, 14)
    assert _line(at, "Road tax (RTO)") == 180_000


def test_karnataka_cess_is_percent_of_tax_plus_flat(repo):
    q = _quote(repo, "kylaq-signature-at", "Karnataka")
    tax = _line(q, "Road tax (RTO)")
    assert tax == calc.percent_of(1_049_000, 17)
    assert _line(q, "Road tax cess") == calc.percent_of(tax, 11) + 1_000


def test_delhi_has_no_cess_line(repo):
    q = _quote(repo, "kylaq-signature-at", "Delhi")
    assert _line(q, "Road tax cess") is None
    assert _line(q, "Road tax (RTO)") == calc.percent_of(1_049_000, 10)


def test_accessories_are_added_and_must_match_model(repo):
    dashcam = next(a for a in repo.accessories_for("kushaq") if a.id == "kushaq-dashcam")
    q = _quote(repo, "kushaq-signature-at", "Maharashtra", accessories=[dashcam])
    assert _line(q, "Accessory: Dashcam") == 10_999
    with pytest.raises(ValueError, match="not for this model"):
        _quote(repo, "kylaq-signature-at", "Telangana", accessories=[dashcam])


def test_mismatched_rto_band_is_rejected(repo):
    v = repo.get_variant("kylaq-signature-at")
    wrong = repo.get_rto_rate("Telangana", "petrol", 400_000)
    with pytest.raises(ValueError, match="band"):
        calc.on_road_price(v, "Telangana", wrong, repo.pricing_rules)


def test_range_petrol_is_mileage_times_tank(repo):
    r = calc.driving_range(repo.get_variant("kylaq-signature-at"))
    assert r.range_km == 857  # 19.05 x 45 = 857.25
    assert r.method == "mileage x tank"


def test_range_ev_uses_claimed_range(repo):
    assert calc.driving_range(repo.get_variant("enyaq-60-at")).range_km == 500


def test_range_needs_published_figures(repo):
    with pytest.raises(ValueError, match="claimed mileage"):
        calc.driving_range(repo.get_variant("octavia-rs-dsg"))
    with pytest.raises(ValueError, match="fuel tank"):
        calc.driving_range(repo.get_variant("kodiaq-lounge-dsg"))


def test_service_cost_matches_published_totals(repo):
    at = repo.get_service_schedule("kylaq", "petrol", "AT")
    mt = repo.get_service_schedule("kylaq", "petrol", "MT")
    assert calc.service_cost(at, 3).total_min == 18_996   # V3Cars 3-year AT total
    assert calc.service_cost(mt, 5).total_min == 37_662   # V3Cars 5-year MT total
    assert [v.interval_months for v in calc.service_cost(mt, 1).visits] == [1, 6, 12]


def test_service_cost_rejects_years_beyond_schedule(repo):
    with pytest.raises(ValueError, match="only covers 5 years"):
        calc.service_cost(repo.get_service_schedule("kylaq", "petrol", "MT"), 6)
    with pytest.raises(ValueError, match="no service schedule"):
        calc.service_cost([], 3)


def test_haversine_known_distances():
    assert calc.haversine_km(17.385, 78.4867, 17.385, 78.4867) == 0
    hyd_blr = calc.haversine_km(17.3850, 78.4867, 12.9716, 77.5946)
    assert 495 < hyd_blr < 505  # Hyderabad to Bengaluru is ~500 km in a straight line
    assert calc.haversine_km(28.6315, 77.2167, 28.6315, 77.2167 + 1) == pytest.approx(97.7, abs=0.5)
