# Škoda India customer-support agent

A multi-agent chat assistant for Škoda India customers, built with **LangGraph** and **Gemini 2.5**. It answers questions about cars, gives on-road price estimates, finds the nearest showrooms and service centres, explains servicing and warranty, books test drives and callbacks, and raises and tracks complaint tickets.

It uses no database: car data lives in CSV/JSON files, documents are searched with a local **FAISS** index, and conversations, leads and tickets are saved as local files.

> **Independent project**, not affiliated with or endorsed by Škoda Auto. Car data comes from public sources listed in [data/seed/README.md](data/seed/README.md); prices are estimates.
>
> **Status:** the agent works end to end in the terminal (CLI) and in LangGraph Studio. The API and customer UI (Phase 5) are not built yet. [ROADMAP.md](ROADMAP.md) is the source of truth for the design and every decision.

---

## What it can do

| Area | Examples |
|---|---|
| **Car information** (brochures, specs, FAQ) | "Does the Kylaq have a sunroof?" · "Compare the safety of the Slavia and the Kushaq" |
| **Price quotes** | "On-road price of Kylaq Signature AT in Hyderabad" → full breakdown (ex-showroom, road tax, insurance, TCS…) · "Is the Kylaq cheaper than the Kushaq?" → starting prices |
| **Showroom / service-centre locator** | "Nearest showroom to Kukatpally" · "Service centre near 560034" |
| **Servicing and warranty** | "Service cost for the Kodiaq for 3 years" · "What does the extended warranty cover?" |
| **Test drives, callbacks, brochures** | "Book a Kylaq test drive" → asks you to **Confirm** before saving |
| **Complaints and tickets** | "There's a problem with my service bill" → asks what's wrong, then raises a ticket (`SKD-20261021-0001`) after you **Confirm** · "Status of ticket SKD-…?" |
| **Several things at once** | "Kylaq price in Hyderabad and nearest showroom to Madhapur" → both answered together |
| **Hand-off to a person** | an upset customer, a safety issue ("my brakes failed") or a dispute, or the bot failing twice in a row → apologises, safety advice first if needed, and offers customer care, a callback or a high-priority ticket |

Every price and distance comes from Python tools, never from the model, and a **guardrail** checks every ₹ and km figure in a reply against the tool results before the customer sees it.

---

## How it works

```
customer message
   ▼
supervisor  ── Gemini: intents + details (car, city, contact…)
   ▼
router ──(needs a person)──▶ escalate: apology + ways to reach a person → composer
   ├──(detail missing)──▶ clarify: asks one question
   │
   ├─▶ car_info · quotation · service · locator      ← run IN PARALLEL, each with its own tools
   │          (merge results)
   ├─▶ lead → support                                 ← "action" agents, one at a time, each pauses for Confirm/Cancel
   ▼
composer  ── Gemini: one reply from all results
   ▼
guardrail ── every ₹ / km figure must come from a tool
   ▼
reply (+ conversation summarised after 20 messages)
```

- **6 agents, 13 tools.** Each agent only sees its own tools.
- **Hybrid document search:** FAISS (meaning) + BM25 (exact words), merged with Reciprocal Rank Fusion.
- **A list of cars in the slots:** a question can be about two cars ("Kushaq price and Slavia sunroof", "Kylaq or Kushaq?").
- Details: [ROADMAP.md](ROADMAP.md) sections 3–5 (State, graph, run-time flow), 7 (parallel agents), 9 (support), 12 (cars). Generated diagram: [docs/graph.md](docs/graph.md).

---

## Setup

Requires **Python 3.14** and a **Gemini API key** ([Google AI Studio](https://aistudio.google.com/apikey)).

```powershell
# from the project folder
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

copy .env.example .env          # then open .env and set GOOGLE_API_KEY
.venv\Scripts\python.exe -m ingestion.build_index      # builds the document index (a few embedding calls)
```

If `import pandas` fails with *"An Application Control policy has blocked this file"* (pyarrow), run `.venv\Scripts\python.exe -m pip uninstall -y pyarrow`: the agent doesn't need it (only Streamlit does).

Never commit `.env`. All settings are documented in [.env.example](.env.example); the main ones:

| Setting | Default | What it does |
|---|---|---|
| `GOOGLE_API_KEY` | — | your Gemini key (required) |
| `DEFAULT_MODEL` | `gemini-2.5-flash` | model for every node; per-node overrides like `SUPERVISOR_MODEL` |
| `<NODE>_THINKING_BUDGET` | 0–512 | thinking tokens per node (0 = off) |
| `MAX_TOOL_ROUNDS` | 5 | tool rounds per agent per question |
| `RETRIEVAL_MODE` | `hybrid` | `hybrid` (FAISS + BM25) or `vector` |
| `SUPPORT_RESPONSE_PROMISE` | `within 24 hours` | what the ticket confirmation promises the customer |
| `CUSTOMER_CARE_NUMBER` / `ESCALATE_AFTER_STRIKES` | `1800 2700 260` / 2 | the number offered on a hand-off, and how many failed turns in a row trigger one |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` | off | optional tracing to LangSmith (phone numbers masked) |

---

## Running it

**Chat in the terminal:**
```powershell
.venv\Scripts\python.exe -m app.cli                 # new conversation
.venv\Scripts\python.exe -m app.cli --thread abc    # continue a saved conversation
.venv\Scripts\python.exe -m app.cli --verbose       # show which step is running
```
Type `exit` to quit. When it wants to save a test drive, callback or ticket it asks `Confirm? (y/n)`. Each question uses about 4–8 Gemini calls.

**LangGraph Studio (browser, for developers):** chat with the graph, watch each step run live, inspect and edit the State. It runs in its own venv because its dev server needs an older `structlog`:
```powershell
py -3.14 -m venv .venv-studio
Get-Content requirements.txt | Where-Object { $_ -notmatch '^structlog' } | Set-Content $env:TEMP\req-studio.txt
.venv-studio\Scripts\python.exe -m pip install -r $env:TEMP\req-studio.txt -r requirements-studio.txt

.venv-studio\Scripts\langgraph dev                 # opens https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```
At a Confirm/Cancel pause, resume with `{"approved": true}`. See [ROADMAP.md](ROADMAP.md) section 10.

---

## Testing and evaluation

```powershell
.venv\Scripts\python.exe -m pytest -q                                  # 197 offline tests, no API calls
.venv\Scripts\python.exe -m evals.run_routing                          # intents + slots, 1 Gemini call per case (17 cases)
.venv\Scripts\python.exe -m evals.run_evals --only q-01,m-01 --no-judge  # end-to-end cases, ~5 calls each
.venv\Scripts\python.exe -m ingestion.test_retrieval --mode hybrid     # see which document chunks a search returns
```

- `evals/test_questions.jsonl` has 56 end-to-end cases scored on routing, slots, tool use, flow, retrieval hit@k, key facts, faithfulness (LLM judge), refusals and latency. Run a few with `--only` rather than all of them (the full set is ~250+ Gemini calls). See [evals/README.md](evals/README.md).
- `evals/routing_cases.jsonl` has the hard routing and slot cases (two cars, comparisons, Hinglish, follow-ups, contact details, injection).

---

## Project layout

```
app/
  agents/        one file per agent (AgentSpec: intent, prompt, required details, reply order) + the catalog
  tools/         the 13 tools and registry.py (one register() line per tool)
  graph/         State, slots, graph builder, nodes (supervisor, router, worker, composer, guardrail…), runner
  prompts/       one prompt per node / agent
  data/          pydantic schemas + repository (loads and checks data/seed/)
  memory/        conversation save / restore (data/conversations/)
  cli.py  llm.py  config.py  log.py  observability.py
ingestion/       document loaders, chunker, index builder
data/
  seed/          car data: models, variants, features, road tax, service schedule, dealers, pincodes, pricing rules
  raw/           documents for search (brochures, specs, FAQ, service, warranty)
  index/         FAISS index (built by ingestion.build_index)
  leads.csv  tickets.csv  conversations/  logs/  cache/     ← created at run time, gitignored
evals/           eval sets, metrics, LLM judge, runners
tests/           offline tests (scripted fake LLMs)
docs/graph.md    diagram generated from the compiled graph (python -m app.graph.build_graph)
```

---

## Extending it

- **A new tool:** write it in `app/tools/`, then add one line to `app/tools/registry.py`, e.g. `register(my_tool, agents=["quotation"])`. A tool that saves something adds `action=True` (the customer must confirm) and optionally a `precheck`.
- **A new agent:** one file in `app/agents/` defining `AGENT = AgentSpec(...)`, its tools registered for it, and eval cases for its intent. The supervisor, router, clarify, graph, composer order and CLI pick it up automatically; startup checks refuse a broken setup. See [ROADMAP.md](ROADMAP.md) section 8.
- **New car data:** edit the files in `data/seed/` (sources and caveats in [data/seed/README.md](data/seed/README.md)); the app checks them when it starts.
- **New documents:** add them to `data/raw/<brochure|specs|faq|service|warranty>/` and rerun `python -m ingestion.build_index`.

---

## Logs, tracing and privacy

- Every turn writes one summary line to `data/logs/app.log`: steps and seconds, intents, agents, tools, Gemini calls and tokens, guardrail result. No message text.
- With `LANGSMITH_TRACING=true`, each turn is also sent to LangSmith as a clickable trace. Phone numbers are masked before anything leaves the machine; names and other text are not.
- Phone numbers are masked in logs, traces and replies. Leads and tickets (with full phone numbers) are stored only in `data/leads.csv` and `data/tickets.csv`, which are gitignored.

---

## Known limits

- Road tax is available for 4 states (Telangana, Karnataka, Maharashtra, Delhi) and dealers for 4 cities; ex-showroom prices are Delhi figures. Insurance % and the FASTag fee are assumptions (flagged in every quote).
- The documents in `data/raw/` are samples written from public Škoda India information; replace them with real brochures and rebuild the index.
- A Confirm/Cancel pause can't be resumed after the app restarts.
- Not built yet: the API and customer UI (Phase 5).
