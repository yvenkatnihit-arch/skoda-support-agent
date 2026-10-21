"""Location tools: geocode a place and find the nearest Skoda showrooms / service centres.

Geocoding order: GPS lat/long from the UI -> local cache (data/cache/geocode.json) -> OpenStreetMap
Nominatim -> local pincodes.csv. A Nominatim failure (timeout, HTTP error, no result) falls back to
the local file instead of failing.
"""

import json
import re
import threading
import time
from pathlib import Path
from typing import Literal

import httpx
import structlog
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import get_settings
from app.data.repository import get_repository
from app.tools.calculators import haversine_km
from app.tools.common import ToolInputError, returns_errors_as_text, to_json

log = structlog.get_logger()

INDIA_LAT = (6.0, 38.0)
INDIA_LON = (68.0, 98.0)
PINCODE_RE = re.compile(r"^\d{6}$")
NOMINATIM_MIN_INTERVAL_S = 1.0  # usage policy: at most 1 request per second
FAR_AWAY_KM = 50.0

_cache_lock = threading.Lock()
_rate_lock = threading.Lock()
_last_request = 0.0


def in_india(lat: float, lon: float) -> bool:
    return INDIA_LAT[0] <= lat <= INDIA_LAT[1] and INDIA_LON[0] <= lon <= INDIA_LON[1]


def maps_link(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"


# ---- cache ---------------------------------------------------------------------

def _cache_path() -> Path:
    return get_settings().cache_dir / "geocode.json"


def _cache_key(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _cache_get(query: str) -> dict | None:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(_cache_key(query))
    except (json.JSONDecodeError, OSError):
        log.warning("geocode cache unreadable; ignoring it", path=str(path))
        return None


def _cache_put(query: str, value: dict) -> None:
    path = _cache_path()
    with _cache_lock:
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (json.JSONDecodeError, OSError):
            data = {}
        data[_cache_key(query)] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)


# ---- Nominatim -----------------------------------------------------------------

def _nominatim_search(query: str) -> dict | None:
    """One Nominatim lookup restricted to India. Returns None when there is no usable result."""
    global _last_request
    s = get_settings()
    with _rate_lock:
        wait = NOMINATIM_MIN_INTERVAL_S - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        resp = httpx.get(
            s.nominatim_url,
            params={"q": query, "format": "jsonv2", "limit": 1, "countrycodes": "in"},
            headers={"User-Agent": s.nominatim_user_agent},
            timeout=s.nominatim_timeout_s,
        )
    resp.raise_for_status()
    hits = resp.json()
    if not hits:
        return None
    lat, lon = float(hits[0]["lat"]), float(hits[0]["lon"])
    if not in_india(lat, lon):
        return None
    return {"lat": round(lat, 6), "long": round(lon, 6), "display_name": hits[0].get("display_name", query)}


# ---- local fallback ------------------------------------------------------------

def _local_lookup(location: str, city: str | None) -> dict | None:
    repo = get_repository()
    loc = location.strip()
    if PINCODE_RE.match(loc):
        p = repo.find_pincode(loc)
        if p:
            return {"lat": p.latitude, "long": p.longitude, "display_name": f"{p.area}, {p.city} {p.pincode}",
                    "approximate": False}
        return None
    areas = repo.find_area(loc, city)
    if areas:
        p = areas[0]
        return {"lat": p.latitude, "long": p.longitude, "display_name": f"{p.area}, {p.city} {p.pincode}",
                "approximate": False}
    # a bare city name: centre of that city's known areas
    pins = repo.list_pincodes(city=city or loc)
    if pins:
        lat = sum(p.latitude for p in pins) / len(pins)
        lon = sum(p.longitude for p in pins) / len(pins)
        return {"lat": round(lat, 6), "long": round(lon, 6), "display_name": f"{pins[0].city} (approximate centre)",
                "approximate": True}
    return None


# ---- tools ---------------------------------------------------------------------

class GeocodeArgs(BaseModel):
    location: str | None = Field(default=None, description="Area name, pincode or address, e.g. 'Kukatpally', '500072'")
    city: str | None = Field(default=None, description="City, if known, to disambiguate the area, e.g. 'Hyderabad'")
    lat: float | None = Field(default=None, description="Latitude shared from the customer's device (GPS)")
    long: float | None = Field(default=None, description="Longitude shared from the customer's device (GPS)")


@tool("geocode_location", args_schema=GeocodeArgs)
@returns_errors_as_text
def geocode_location(location: str | None = None, city: str | None = None,
                     lat: float | None = None, long: float | None = None) -> str:
    """Turn an area, pincode, city or GPS position into latitude/longitude. Call this before
    find_nearest_dealers unless you already have lat/long."""
    if (lat is None) != (long is None):
        raise ToolInputError("Give both lat and long, or neither.")
    if lat is not None and long is not None:
        if not in_india(lat, long):
            raise ToolInputError("That position is outside India. Skoda India dealers are only in India.")
        return to_json({"lat": lat, "long": long, "display_name": "your shared location", "source": "gps",
                        "approximate": False})

    place = (location or city or "").strip()
    if not place:
        raise ToolInputError("No location given. Ask the customer for their area, pincode or city.")
    query = ", ".join(x for x in (place, city if city and location else None, "India") if x)

    cached = _cache_get(query)
    if cached:
        return to_json({**cached, "source": "cache"})

    try:
        hit = _nominatim_search(query)
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        log.warning("nominatim unavailable; using local pincodes", error=type(exc).__name__)
        hit = None
    if hit:
        value = {**hit, "approximate": False}
        _cache_put(query, value)
        return to_json({**value, "source": "nominatim"})

    local = _local_lookup(place, city)
    if local:
        return to_json({**local, "source": "pincodes.csv"})
    raise ToolInputError(f"I couldn't find '{place}'. Ask the customer for a nearby area name or a 6-digit pincode.")


class NearestDealersArgs(BaseModel):
    lat: float = Field(description="Latitude from geocode_location")
    long: float = Field(description="Longitude from geocode_location")
    dealer_type: Literal["showroom", "service", "any"] = Field(
        default="any", description="'showroom' for sales/test drives, 'service' for servicing, or 'any'")
    limit: int = Field(default=3, ge=1, le=10, description="How many outlets to return")


@tool("find_nearest_dealers", args_schema=NearestDealersArgs)
@returns_errors_as_text
def find_nearest_dealers(lat: float, long: float, dealer_type: str = "any", limit: int = 3) -> str:
    """Nearest official Skoda showrooms and/or service centres to a position, sorted by straight-line
    distance, with address, phone, opening hours, services and a Google Maps link."""
    if not in_india(lat, long):
        raise ToolInputError("That position is outside India.")
    repo = get_repository()
    dealers = repo.list_dealers(dealer_type=None if dealer_type == "any" else dealer_type)
    if not dealers:
        raise ToolInputError(f"No {dealer_type} outlets in our data.")
    ranked = sorted(((haversine_km(lat, long, d.latitude, d.longitude), d) for d in dealers), key=lambda x: x[0])
    results = [{
        "dealer_id": d.id, "name": d.name, "type": d.type, "address": d.address, "city": d.city,
        "phone": d.phone, "timings": d.timings, "services_offered": d.services_offered,
        "distance_km": round(dist, 1), "maps_link": maps_link(d.latitude, d.longitude),
        "source_url": d.source_url, "as_of": str(d.as_of),
    } for dist, d in ranked[:limit]]
    out = {"dealer_type": dealer_type, "distance_note": "straight-line distance; road distance is longer",
           "dealers": results}
    if results[0]["distance_km"] > FAR_AWAY_KM:
        cities = sorted({d.city for d in repo.list_dealers()})
        out["coverage_note"] = (f"The nearest outlet is {results[0]['distance_km']} km away. Our dealer data covers "
                                f"{', '.join(cities)}; for other cities see https://dealers.skoda-auto.co.in/")
    return to_json(out)


LOCATOR_TOOLS = [geocode_location, find_nearest_dealers]
