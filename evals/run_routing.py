"""Routing check: run ONLY the supervisor (1 Gemini call per case) on hard routing cases and compare intents,
agents and slots with what's expected. Cheap way to measure and improve intent detection.

    python -m evals.run_routing                 # all cases in evals/routing_cases.jsonl
    python -m evals.run_routing --only r-01,r-05

Results are saved to evals/results/routing-<timestamp>.json (gitignored).
"""

import argparse
import json
import sys
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage

from app.agents import CATALOG
from app.graph.build_graph import GraphLLMs
from app.graph.car_slots import active_car_list, cars_for, public_view
from app.graph.nodes.supervisor import make_supervisor, missing_slots
from app.log import configure_logging, mask_phone
from app.observability import disable_auto_tracing
from evals.metrics import _slot_equal
from evals.run_evals import EVAL_DIR, RESULTS_DIR


def load(path=EVAL_DIR / "routing_cases.jsonl") -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_cars(case: dict, slots: dict, intents: list[str]) -> list[str]:
    """Problems with the list of cars: each expected car is active, with its details, for its intents."""
    expected = case.get("expected_cars")
    if not expected:
        return []
    problems = []
    active = [c.get("model") for c in active_car_list(slots)]
    if set(active) != {c["model"] for c in expected}:
        problems.append(f"active cars {active}, expected {[c['model'] for c in expected]}")
    for exp in expected:
        car = next((c for c in active_car_list(slots) if c.get("model") == exp["model"]), None)
        if car is None:
            continue
        for key in ("variant", "transmission", "fuel"):
            if key in exp and not _slot_equal(key, exp[key], car.get(key)):
                problems.append(f"{exp['model']} {key}: expected {exp[key]}, got {car.get(key)}")
        for intent in exp.get("intents", []):
            if exp["model"] not in [c.get("model") for c in cars_for(slots, intent)]:
                problems.append(f"{exp['model']} not given to {intent}")
    if case.get("expected_no_clarify") and missing_slots(intents, slots):
        problems.append(f"would clarify: {missing_slots(intents, slots)}")
    return problems


def check(case: dict, intents: list[str], slots: dict) -> dict:
    options = case.get("expected_intents_any") or [case["expected_intents"]]
    intents_ok = any(set(o) == set(intents) for o in options)
    wrong = {k: {"expected": v, "got": slots.get(k)} for k, v in (case.get("expected_slots") or {}).items()
             if not _slot_equal(k, v, slots.get(k))}
    car_problems = check_cars(case, slots, intents)
    return {"intents_ok": intents_ok, "slots_ok": not wrong, "wrong_slots": wrong, "car_problems": car_problems,
            "pass": intents_ok and not wrong and not car_problems}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated case ids")
    args = parser.parse_args()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    configure_logging("INFO", RESULTS_DIR / f"routing-{run_id}.log")
    disable_auto_tracing()

    cases = load()
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only.split(","))]
    supervisor = make_supervisor(GraphLLMs.from_settings().supervisor)
    results = []
    for case in cases:
        history = [HumanMessage(m["content"]) if m["role"] == "human" else AIMessage(m["content"])
                   for m in case.get("history", [])]
        update = supervisor({"messages": [*history, HumanMessage(case["message"])], "slots": case.get("slots", {})})
        intents, slots = update["intents"], update["slots"]
        result = check(case, intents, slots)
        results.append({"id": case["id"], "what": case["what"], "intents": intents,
                        "agents": CATALOG.agents_for(intents),
                        "slots": {k: v for k, v in public_view(slots).items() if v not in (None, "", [], {})},
                        **result})

    print(f"\nRouting check: {sum(r['pass'] for r in results)}/{len(results)} passed\n")
    for r in results:
        mark = "PASS" if r["pass"] else "FAIL"
        print(f"{mark}  {r['id']}  {r['what']}")
        print(f"      intents {r['intents']}  -> agents {r['agents']}")
        print(f"      slots   {mask_phone(json.dumps(r['slots'], ensure_ascii=False))}")
        if not r["intents_ok"]:
            case = next(c for c in cases if c["id"] == r["id"])
            print(f"      ! expected intents {case.get('expected_intents') or case.get('expected_intents_any')}")
        for problem in r["car_problems"]:
            print(f"      ! {problem}")
        for k, v in r["wrong_slots"].items():
            print(f"      ! slot {k}: expected {mask_phone(str(v['expected']))}, got {mask_phone(str(v['got']))}")
    out = RESULTS_DIR / f"routing-{run_id}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
