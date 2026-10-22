"""Clarify: ask ONE short question for the missing detail(s), with the choices to pick from; the turn then ends.

The choices (models, cities with price data, a model's variants, how to give a location) come from the data
files in Python, so the customer never has to guess what to answer and no extra Gemini call is needed.
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents import CATALOG, Catalog
from app.data.repository import get_repository
from app.graph.car_slots import cars_for, public_view
from app.graph.nodes.supervisor import missing_slots
from app.prompts.clarify import CLARIFY_PROMPT

LOCATION_HINT = "Location: an area or pincode (e.g. Kukatpally, 560034), a city, or share your location."


def model_choices() -> str:
    models = get_repository().list_models()
    on_sale = [m.name for m in models if m.status == "on_sale"]
    upcoming = [m.name for m in models if m.status == "upcoming"]
    return ("Models on sale: " + ", ".join(on_sale)
            + (f". Coming soon: {', '.join(upcoming)}." if upcoming else "."))


def city_choices() -> str:
    """Cities we can price (their state has road-tax data)."""
    repo = get_repository()
    states = {s.lower() for s in repo.list_rto_states()}
    cities = sorted({p.city for p in repo.list_pincodes() if (p.state or "").lower() in states}
                    | {d.city for d in repo.list_dealers() if (d.state or "").lower() in states})
    return "Cities with on-road price data: " + ", ".join(cities) + "."


def variant_choices(model_name: str) -> str | None:
    repo = get_repository()
    model = repo.get_model(model_name)
    if model is None:
        return None
    names = list(dict.fromkeys(f"{v.name} {v.transmission}" for v in repo.find_variants(model.id)))
    return f"{model.name} variants: " + ", ".join(names) + ' (or say "base" for the cheapest, "top" for the most expensive).'


def choices_for(missing: dict[str, list[str]], slots: dict) -> list[str]:
    gaps = [gap for g in missing.values() for gap in g]
    out = []
    if any(g == "model" or g.startswith("model ") for g in gaps):
        out.append(model_choices())
    if any(g.startswith("variant") for g in gaps):
        for intent in missing:
            for car in cars_for(slots, intent):
                if car.get("model") and not car.get("variant") and (line := variant_choices(car["model"])):
                    out.append(line)
    if any(g == "city" for g in gaps):
        out.append(city_choices())
    if any("pincode" in g for g in gaps):
        out.append(LOCATION_HINT)
    return list(dict.fromkeys(out))


def make_clarify(llm, catalog: Catalog = CATALOG):
    def clarify(state: dict) -> dict:
        slots = state.get("slots") or {}
        missing = missing_slots(state.get("intents") or [], slots, catalog)
        wanted = sorted({gap for gaps in missing.values() for gap in gaps})
        options = choices_for(missing, slots)
        system = SystemMessage(CLARIFY_PROMPT.format(
            missing=", ".join(wanted),
            intents=", ".join(missing),
            slots=json.dumps({k: v for k, v in public_view(slots).items() if v and k != "phone"}, ensure_ascii=False),
            options="Choices:\n" + "\n".join(f"- {o}" for o in options) if options else "",
        ))
        last_question = next((m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)), None)
        reply = llm.invoke([system, *([last_question] if last_question else [])])
        # asking again counts toward a hand-off: after ESCALATE_AFTER_STRIKES the router offers a person instead
        return {"messages": [AIMessage(content=reply.text)], "strikes": (state.get("strikes") or 0) + 1}

    return clarify
