import shutil
from pathlib import Path

import pytest

from app.data.repository import DataValidationError, Repository

SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "seed"


@pytest.fixture(scope="module")
def repo() -> Repository:
    return Repository(SEED_DIR)


@pytest.fixture
def seed_copy(tmp_path: Path) -> Path:
    dest = tmp_path / "seed"
    shutil.copytree(SEED_DIR, dest)
    return dest


def _replace_in(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"test setup: '{old}' not found in {path.name}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# ---- loading the real seed data ------------------------------------------------

def test_seed_data_loads(repo):
    assert len(repo.models) == 6
    assert len(repo.variants) > 0
    assert len(repo.dealers) > 0
    for city in ("Hyderabad", "Bengaluru", "Mumbai", "Delhi"):
        assert repo.list_dealers(city=city), f"no dealers for {city}"


def test_dealer_text_has_no_html_entities(repo):
    for d in repo.list_dealers():
        assert "&amp;" not in d.address and "&#" not in d.address, d.id


def test_get_model_matches_name_id_and_unique_prefix(repo):
    assert repo.get_model("KYLAQ").id == "kylaq"
    assert repo.get_model("Škoda Kushaq").id == "kushaq"
    assert repo.get_model("Octavia").id == "octavia-rs"
    assert repo.get_model("Fabia") is None
    assert repo.get_model("") is None


def test_find_variants_normalises_trim_names(repo):
    [v] = repo.find_variants("kylaq", name="Signature", transmission="AT")
    assert v.id == "kylaq-signature-at"
    plus = repo.find_variants("kylaq", name="Signature Plus", transmission="mt")
    assert [v.id for v in plus] == ["kylaq-signature-plus-mt"]
    assert {v.transmission for v in repo.find_variants("kushaq", name="Prestige")} == {"MT", "AT", "DSG"}


def test_variant_features_respect_at_only(repo):
    at = {f.feature for f in repo.get_features("kylaq-signature-at")}
    mt = {f.feature for f in repo.get_features("kylaq-signature-mt")}
    assert "Paddle shifters" in at
    assert "Paddle shifters" not in mt
    assert "6 airbags" in mt


def test_rto_band_lower_bound_is_inclusive(repo):
    assert repo.get_rto_rate("telangana", "petrol", 999_999).rate_percent == 14
    assert repo.get_rto_rate("Telangana", "petrol", 1_000_000).rate_percent == 18
    assert repo.get_rto_rate("Goa", "petrol", 1_000_000) is None


def test_service_schedule_is_sorted_by_km(repo):
    rows = repo.get_service_schedule("kylaq", "petrol", "AT")
    kms = [r.interval_km for r in rows]
    assert kms == sorted(kms) and kms[0] == 1000


def test_dealer_type_filter_includes_both(repo):
    showrooms = repo.list_dealers(dealer_type="showroom", city="Hyderabad")
    assert {d.type for d in showrooms} <= {"showroom", "both"}
    assert any(d.type == "both" for d in showrooms)


def test_pincode_and_area_lookup(repo):
    assert repo.find_pincode("500072").area == "Kukatpally"
    assert repo.find_area("kukatpally")[0].pincode == "500072"
    assert repo.state_for_city("Bengaluru") == "Karnataka"


# ---- malformed data fails with a clear error ------------------------------------

def test_bad_value_reports_file_line_and_field(seed_copy):
    _replace_in(seed_copy / "variants.csv", ",759000,", ",abc,")
    with pytest.raises(DataValidationError) as exc:
        Repository(seed_copy)
    assert exc.value.file == "variants.csv"
    assert "line 2: ex_showroom_price" in str(exc.value)


def test_missing_column_is_reported(seed_copy):
    _replace_in(seed_copy / "dealers.csv", "phone,", "telephone,")
    with pytest.raises(DataValidationError, match="missing column 'phone'"):
        Repository(seed_copy)


def test_unknown_model_reference_is_reported(seed_copy):
    _replace_in(seed_copy / "variants.csv", "kylaq-classic-mt,kylaq,", "kylaq-classic-mt,fabia,")
    with pytest.raises(DataValidationError, match="model_id 'fabia' not found"):
        Repository(seed_copy)


def test_overlapping_rto_bands_are_reported(seed_copy):
    _replace_in(seed_copy / "rto_rates.csv", "Delhi,petrol,600000,", "Delhi,petrol,500000,")
    with pytest.raises(DataValidationError, match="overlaps"):
        Repository(seed_copy)


def test_electric_variant_requires_battery_and_range(seed_copy):
    _replace_in(seed_copy / "variants.csv", ",63,500,", ",,500,")
    with pytest.raises(DataValidationError, match="electric variant needs battery_kwh"):
        Repository(seed_copy)


def test_missing_file_is_reported(seed_copy):
    (seed_copy / "pincodes.csv").unlink()
    with pytest.raises(DataValidationError, match="pincodes.csv"):
        Repository(seed_copy)
