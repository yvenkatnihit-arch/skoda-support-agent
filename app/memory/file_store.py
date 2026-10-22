"""Save the full graph State to data/conversations/{thread_id}.json at every turn end, and restore it into
the checkpointer when a thread is unknown (e.g. after an app restart). Swap this module for a database
checkpointer later without touching the graph."""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import messages_from_dict, messages_to_dict

from app.config import get_settings
from app.graph.state import State

THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MESSAGE_FIELDS = ("messages", "action_messages")
STATE_FIELDS = set(State.__annotations__)  # older files also had queue, active_agent, agent_messages or
                                           # lead_messages: those are ignored on load
RESTORE_AS_NODE = "summarize"  # its only edge is END, so the next input starts a fresh run at START


class InvalidThreadId(ValueError):
    pass


def _path(thread_id: str, base_dir: Path | None = None) -> Path:
    if not THREAD_ID_RE.match(thread_id or ""):  # thread ids come from clients: never let them escape the folder
        raise InvalidThreadId("thread_id must be 1-64 letters, digits, '-' or '_'")
    return (base_dir or get_settings().conversations_dir) / f"{thread_id}.json"


def validate_thread_id(thread_id: str) -> None:
    """Raise InvalidThreadId unless the id is 1-64 letters, digits, '-' or '_' (ids come from clients)."""
    _path(thread_id)


def serialize_state(values: dict) -> dict:
    out = {}
    for key, value in values.items():
        out[key] = messages_to_dict(value) if key in MESSAGE_FIELDS else value
    return out


def deserialize_state(data: dict) -> dict:
    return {k: (messages_from_dict(v) if k in MESSAGE_FIELDS else v) for k, v in data.items() if k in STATE_FIELDS}


def save_state(thread_id: str, values: dict, *, paused: bool = False, base_dir: Path | None = None) -> Path:
    path = _path(thread_id, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"thread_id": thread_id, "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "paused_for_approval": paused, "state": serialize_state(values)}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_state(thread_id: str, base_dir: Path | None = None) -> dict | None:
    path = _path(thread_id, base_dir)
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    return deserialize_state(doc["state"])


def restore_if_missing(graph, config: dict, base_dir: Path | None = None) -> bool:
    """If the checkpointer has nothing for this thread but a JSON file exists, load it with update_state().
    A pause at ask_user can't be resumed after a restart; the next question starts a normal new turn."""
    if graph.get_state(config).values:
        return False
    values = load_state(config["configurable"]["thread_id"], base_dir)
    if not values:
        return False
    graph.update_state(config, values, as_node=RESTORE_AS_NODE)
    return True
