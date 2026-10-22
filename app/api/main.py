"""The API: the doorway between the chat page (ui/index.html) and the agent (ROADMAP Phase 5).

    .venv\\Scripts\\python.exe -m uvicorn app.api.main:app --port 8000      # then open http://127.0.0.1:8000

POST /api/chat and POST /api/resume stream Server-Sent Events while the turn runs:
    event: thread    {"thread_id"}                      first, so the page can remember the conversation
    event: status    {"node", "text"}                   progress ("Working out the price...")
    event: reply     {"text"}                           the checked reply (only after the guardrail)
    event: confirm   {"actions", "question", "agent"}   the agent wants to save something: ask Confirm/Cancel
    event: error     {"text"}                           friendly message; details are in the logs only
    event: done      {"thread_id"}
GET /api/threads/{id} returns the conversation (to redraw the page); GET /api/health returns ok.

Every turn goes through run_turn (rollback on errors, hand-off counter, turn logs, tracing), and the State is
saved to data/conversations/<thread_id>.json at every turn end. Local demo: no login or rate limiting yet; add
them before putting this on the internet.
"""

import asyncio
import json
import threading
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from app.api.schemas import ChatMessage, ChatRequest, ResumeRequest, ThreadView
from app.config import PROJECT_ROOT, get_settings
from app.graph.build_graph import recursion_limit
from app.graph.car_slots import normalize
from app.graph.progress import STATUS
from app.graph.runner import FRIENDLY_ERROR, run_turn
from app.log import configure_logging
from app.memory.file_store import (RESTORE_AS_NODE, InvalidThreadId, restore_if_missing, save_state,
                                   validate_thread_id)
from app.observability import disable_auto_tracing, flush_tracing, tracing_callbacks

log = structlog.get_logger()

UI_FILE = PROJECT_ROOT / "ui" / "index.html"


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(graph=None) -> FastAPI:
    """graph: tests pass a graph built with scripted fake LLMs; otherwise the real one is built at startup."""
    state = {"graph": graph, "callbacks": []}
    locks: dict[str, threading.Lock] = {}
    locks_guard = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        s = get_settings()
        if state["graph"] is None:  # the real app: logs to a file, optional masked tracing, real Gemini
            from app.graph.build_graph import build_graph

            configure_logging(s.log_level, s.data_dir / "logs" / "api.log")
            disable_auto_tracing()
            state["callbacks"] = tracing_callbacks(s)
            state["graph"] = build_graph()
        yield
        flush_tracing(s)

    app = FastAPI(title="Škoda support agent", lifespan=lifespan)

    def config_for(thread_id: str) -> dict:
        try:
            validate_thread_id(thread_id)  # letters, digits, - and _ only: ids come from the browser
        except InvalidThreadId as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit(),
                "callbacks": state["callbacks"]}

    def lock_for(thread_id: str) -> threading.Lock:
        with locks_guard:  # one turn at a time per conversation (two tabs, double clicks)
            return locks.setdefault(thread_id, threading.Lock())

    def run_and_stream(thread_id: str, config: dict, payload, location: tuple | None = None) -> StreamingResponse:
        graph = state["graph"]

        async def events():
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def emit(event, data):
                loop.call_soon_threadsafe(queue.put_nowait, (event, data))

            def work():
                last = None

                def on_node(node: str):
                    nonlocal last
                    text = STATUS.get(node)
                    if text and text != last:
                        last = text
                        emit("status", {"node": node, "text": text})

                with lock_for(thread_id):
                    try:
                        restore_if_missing(graph, config)
                        if location is not None:  # shared location: becomes the lat/long slots for this turn
                            slots = normalize(graph.get_state(config).values.get("slots"))
                            slots["lat"], slots["long"] = location
                            graph.update_state(config, {"slots": slots}, as_node=RESTORE_AS_NODE)
                        result = run_turn(graph, config, payload, on_node)
                        values = graph.get_state(config).values
                        save_state(thread_id, values, paused=result.status == "paused")
                        if result.status == "paused":
                            pending = result.pending or {}
                            emit("confirm", {"agent": pending.get("agent"), "actions": pending.get("actions", []),
                                             "question": pending.get("question", "Shall I go ahead?")})
                        elif result.status == "error":
                            emit("error", {"text": result.reply or FRIENDLY_ERROR})
                        else:
                            emit("reply", {"text": result.reply or ""})
                    except Exception:  # run_turn already handles agent failures; this is a last safety net
                        log.exception("api turn failed")
                        emit("error", {"text": FRIENDLY_ERROR})
                    finally:
                        emit("done", {"thread_id": thread_id})
                        emit(None, None)

            yield _sse("thread", {"thread_id": thread_id})
            worker = loop.run_in_executor(None, work)
            while True:
                event, data = await queue.get()
                if event is None:
                    break
                yield _sse(event, data)
            await worker

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/chat")
    def chat(req: ChatRequest) -> StreamingResponse:
        thread_id = req.thread_id or uuid.uuid4().hex[:12]
        config = config_for(thread_id)
        location = (req.lat, req.long) if req.lat is not None and req.long is not None else None
        return run_and_stream(thread_id, config, {"messages": [HumanMessage(content=req.message.strip())]},
                              location)

    @app.post("/api/resume")
    def resume(req: ResumeRequest) -> StreamingResponse:
        config = config_for(req.thread_id)
        graph = state["graph"]
        restore_if_missing(graph, config)
        if not graph.get_state(config).next:
            raise HTTPException(status_code=409, detail="Nothing is waiting for confirmation in this conversation.")
        return run_and_stream(req.thread_id, config,
                              Command(resume={"approved": req.approved, "feedback": req.feedback}))

    @app.get("/api/threads/{thread_id}")
    def thread(thread_id: str) -> ThreadView:
        config = config_for(thread_id)
        graph = state["graph"]
        restore_if_missing(graph, config)
        snapshot = graph.get_state(config)
        messages = [ChatMessage(role="customer" if isinstance(m, HumanMessage) else "assistant", text=m.text)
                    for m in snapshot.values.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))]
        pending = next((i.value for t in snapshot.tasks for i in t.interrupts), None)
        return ThreadView(thread_id=thread_id, messages=messages, pending=pending)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(UI_FILE, media_type="text/html")

    return app


app = create_app()
