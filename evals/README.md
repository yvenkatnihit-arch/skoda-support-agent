# Evals

`test_questions.jsonl` holds the eval set (one case per line). `run_evals.py` plays every case through the
real graph with real Gemini, scores it, prints a report and saves `results/<timestamp>.json` (gitignored).

```
python -m evals.run_evals                    # all cases + LLM judge (~10 min)
python -m evals.run_evals --no-judge         # deterministic metrics only
python -m evals.run_evals --only q-01,ld-02  # selected cases
python -m evals.run_evals --category lead    # one category
```

Re-run it after every change and compare the summary with the previous run.
Test leads are written to `evals/results/`, never to `data/leads.csv`.

## Metrics

| Metric | Question | Technique |
|---|---|---|
| Routing accuracy | right intents and right agents (both as sets; agents run in parallel) | exact match, no LLM |
| Slot accuracy | model, variant, city, phone… extracted correctly? | per-slot match (case-insensitive; phone normalised) |
| Tool-use accuracy | did the agents call the expected tools? `[]` = no tool may run | set match on `agent_outputs` |
| Flow | clarify asked / Confirm-Cancel pause happened exactly when expected? | node trace |
| Retrieval hit@k | was an expected source file + section among the passages returned (top `RETRIEVAL_TOP_K`)? | match on search tool results |
| Key-fact correctness | does the reply contain every key fact? | substring match; numbers compared without commas and not inside longer numbers |
| Faithfulness | share of the reply's factual claims supported by the tool results | LLM-as-judge (Gemini, temperature 0, fixed rubric) |
| Refused correctly | for `should_refuse` cases: did it decline / say "not available" without inventing ₹ or km figures? | judge's `declined` + the guardrail's figure check |
| Latency | seconds for the last turn of each case | timer |

Deterministic metrics are exact and repeatable; the judge is used only where meaning matters, and its reason is
printed for every unsupported claim so you can check it.

## Case format

```json
{"id": "q-06", "category": "follow_up",
 "turns": ["On-road price of Kylaq Signature MT in Hyderabad", "what about the AT version?"],
 "expected_intents": ["quotation"],            "expected_agents": ["quotation"],
 "expected_slots": {"transmission": "AT", "city": "Hyderabad"},
 "expected_tools": ["calculate_on_road_price"],
 "key_facts": ["12,92,125"],
 "expected_sources": [{"source": "brochure/x.md", "section": "Safety"}],
 "expect_clarify": false, "expect_pause": false, "approve": true, "should_refuse": false}
```

- Expectations describe the **last** turn; earlier turns set up context.
- When you add an agent (ROADMAP section 8), add cases for its intent here.
- `expected_intents_any`: list of acceptable intent sets (instead of `expected_intents`).
- `"a|b"` in `expected_tools` or `key_facts` = either is fine.
- `approve`: what the simulated customer answers at Confirm/Cancel.
- Compute expected figures with the tools (never by hand) and update them when `data/seed/` changes.
