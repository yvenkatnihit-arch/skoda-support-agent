"""Run the eval set through the real graph (real Gemini) and print a score report.

    python -m evals.run_evals                     # all cases, with the LLM judge
    python -m evals.run_evals --no-judge          # deterministic metrics only (cheaper, faster)
    python -m evals.run_evals --only q-01,ld-02   # selected cases
    python -m evals.run_evals --category lead

Results are saved to evals/results/<timestamp>.json (gitignored) so runs can be compared.
Test leads and tickets go to evals/results/, never to data/.
"""

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from app.config import PROJECT_ROOT
from app.graph.build_graph import AGENT_NAMES, build_graph, recursion_limit
from app.log import configure_logging, mask_phone
from app.observability import disable_auto_tracing
from app.graph.nodes.supervisor import agents_for
from app.tools import lead_tools, ticket_tools
from evals import metrics
from evals.judge import faithfulness, judge, make_judge

EVAL_DIR = PROJECT_ROOT / "evals"
RESULTS_DIR = EVAL_DIR / "results"


def load_cases(path: Path = EVAL_DIR / "test_questions.jsonl") -> list[dict]:
    cases = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path.name} line {n}: invalid JSON ({exc})") from exc
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate case ids in test_questions.jsonl")
    return cases


def play(graph, case: dict, run_id: str) -> dict:
    """Play all turns. Returns what happened on the LAST turn (that's what the expectations describe)."""
    config = {"configurable": {"thread_id": f"eval-{run_id}-{case['id']}"}, "recursion_limit": recursion_limit()}
    turn_times = []
    for text in case["turns"]:
        nodes, supervisor, paused, start = [], {}, False, time.perf_counter()
        payload = {"messages": [HumanMessage(content=text)]}
        while payload is not None:
            pending = None
            for chunk in graph.stream(payload, config, stream_mode="updates"):
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        pending = update[0].value
                    else:
                        nodes.append(node)
                        if node == "supervisor":
                            supervisor = update
            payload = None
            if pending is not None:
                paused = True
                payload = Command(resume={"approved": bool(case.get("approve", False))})
        turn_times.append(round(time.perf_counter() - start, 2))
    values = graph.get_state(config).values
    reply = next((m.text for m in reversed(values.get("messages", [])) if isinstance(m, AIMessage)), "")
    return {"nodes": nodes, "intents": supervisor.get("intents", []),
            "agents": agents_for(supervisor.get("intents", [])),
            "slots": values.get("slots", {}), "agent_outputs": values.get("agent_outputs", {}),
            "reply": reply, "paused": paused, "turn_seconds": turn_times}


def score(case: dict, run: dict, judge_llm) -> dict:
    out = {
        "routing": metrics.routing(case, run["intents"], run["agents"]),
        "slots": metrics.slots(case, run["slots"]),
        "tools": metrics.tool_use(case, run["agent_outputs"]),
        "flow": metrics.flow(case, run["nodes"], run["paused"]),
        "retrieval": metrics.retrieval_hit(case, run["agent_outputs"]),
        "key_facts": metrics.key_facts(case, run["reply"]),
        "invented_figures": metrics.invented_figures(run["reply"], run["agent_outputs"]),
    }
    if judge_llm is not None:
        verdict = judge(judge_llm, case["turns"][-1], run["reply"], run["agent_outputs"])
        out["faithfulness"] = {"score": round(faithfulness(verdict), 3),
                               "unsupported": [c.claim for c in verdict.claims if not c.supported],
                               "reasons": [c.reason for c in verdict.claims if not c.supported]}
        out["declined"] = verdict.declined
    if case.get("should_refuse"):
        declined = out.get("declined")
        out["refusal"] = {"pass": (declined is not False) and not out["invented_figures"],
                          "declined": declined, "invented_figures": out["invented_figures"]}
    return out


def _pct(hits: int, total: int) -> str:
    return f"{hits}/{total} {100 * hits / total:5.1f}%" if total else "n/a"


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile: the smallest value with at least pct% of the values at or below it.
    (Rounds the rank up, so with few cases p90 is the slowest one, never below the median.)"""
    ordered = sorted(values)
    return ordered[max(math.ceil(pct / 100 * len(ordered)), 1) - 1]


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    s = lambda key: [r["scores"][key] for r in ok if r["scores"].get(key) is not None]  # noqa: E731
    slot_rows = s("slots")
    faith = [x["score"] for x in s("faithfulness")]
    times = [t for r in ok for t in r["run"]["turn_seconds"][-1:]]
    return {
        "cases": len(results), "errors": len(results) - len(ok),
        "routing_accuracy": (sum(x["pass"] for x in s("routing")), len(s("routing"))),
        "slot_accuracy": (sum(x["correct"] for x in slot_rows), sum(x["total"] for x in slot_rows)),
        "tool_use_accuracy": (sum(x["pass"] for x in s("tools")), len(s("tools"))),
        "flow_accuracy": (sum(x["pass"] for x in s("flow")), len(s("flow"))),
        "retrieval_hit_at_k": (sum(x["hit"] for x in s("retrieval")), len(s("retrieval"))),
        "key_fact_correctness": (sum(x["pass"] for x in s("key_facts")), len(s("key_facts"))),
        "faithfulness_avg": round(statistics.mean(faith), 3) if faith else None,
        "faithful_cases": (sum(f == 1.0 for f in faith), len(faith)),
        "refused_correctly": (sum(x["pass"] for x in s("refusal")), len(s("refusal"))),
        "latency_median_s": round(statistics.median(times), 1) if times else None,
        "latency_p90_s": round(percentile(times, 90), 1) if times else None,
    }


def failures(r: dict) -> list[str]:
    if "error" in r:
        return [f"ERROR {r['error']}"]
    sc, out = r["scores"], []
    if not sc["routing"]["pass"]:
        out.append(f"routing: got intents={sc['routing']['actual_intents']} agents={sc['routing']['actual_agents']}")
    if sc["slots"] and sc["slots"]["wrong"]:
        out.append(f"slots: {sc['slots']['wrong']}")
    if sc["tools"] and not sc["tools"]["pass"]:
        out.append(f"tools: missing={sc['tools']['missing']} called={sc['tools']['called']}")
    if not sc["flow"]["pass"]:
        out.append(f"flow: clarify={sc['flow']['clarify']} pause={sc['flow']['pause']}")
    if sc["retrieval"] and not sc["retrieval"]["hit"]:
        out.append(f"retrieval miss: got {sc['retrieval']['retrieved'][:4]}")
    if sc["key_facts"] and not sc["key_facts"]["pass"]:
        out.append(f"key facts missing: {sc['key_facts']['missing']}")
    if sc.get("faithfulness") and sc["faithfulness"]["score"] < 1:
        out.append(f"faithfulness {sc['faithfulness']['score']}: unsupported {sc['faithfulness']['unsupported'][:3]}")
    if sc.get("refusal") and not sc["refusal"]["pass"]:
        out.append(f"refusal failed: {sc['refusal']}")
    return out


def print_report(summary: dict, results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"Eval report: {summary['cases']} cases, {summary['errors']} errors")
    print("=" * 72)
    rows = [("Routing accuracy (intents + agents)", "routing_accuracy"), ("Slot accuracy (per slot)", "slot_accuracy"),
            ("Tool-use accuracy", "tool_use_accuracy"), ("Flow (clarify / Confirm pause)", "flow_accuracy"),
            ("Retrieval hit@k", "retrieval_hit_at_k"), ("Key-fact correctness", "key_fact_correctness"),
            ("Refused correctly", "refused_correctly")]
    for label, key in rows:
        print(f"  {label:38} {_pct(*summary[key])}")
    if summary["faithfulness_avg"] is not None:
        print(f"  {'Faithfulness (LLM judge, average)':38} {summary['faithfulness_avg']:.3f}"
              f"   fully faithful: {_pct(*summary['faithful_cases'])}")
    print(f"  {'Latency (last turn)':38} median {summary['latency_median_s']} s, p90 {summary['latency_p90_s']} s")
    failed = [(r["id"], failures(r)) for r in results if failures(r)]
    print(f"\nCases with a failure: {len(failed)}")
    for case_id, problems in failed:
        print(f"  {case_id}")
        for p in problems:
            print(f"     - {mask_phone(p)[:240]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--category")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge (faithfulness, refusals)")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    configure_logging("INFO", RESULTS_DIR / f"{run_id}.log")
    disable_auto_tracing()  # evals are never traced to LangSmith (a shell variable must not send them unmasked)
    lead_tools._leads_path = lambda: RESULTS_DIR / f"{run_id}-leads.csv"  # keep eval leads out of data/
    ticket_tools._tickets_path = lambda: RESULTS_DIR / f"{run_id}-tickets.csv"  # and eval tickets

    cases = load_cases()
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if not cases:
        print("No cases selected.")
        return 1

    graph = build_graph()
    judge_llm = None if args.no_judge else make_judge()
    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i:2}/{len(cases)}] {case['id']:7} {case['turns'][-1][:60]}", flush=True)
        try:
            run = play(graph, case, run_id)
            results.append({"id": case["id"], "category": case["category"], "run": run,
                            "scores": score(case, run, judge_llm)})
        except Exception as exc:  # noqa: BLE001 - one broken case shouldn't stop the run
            results.append({"id": case["id"], "category": case["category"], "error": f"{type(exc).__name__}: {exc}"[:300]})

    summary = summarize(results)
    print_report(summary, results)
    out = RESULTS_DIR / f"{run_id}.json"
    out.write_text(json.dumps({"run_id": run_id, "summary": summary, "results": results},
                              ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
