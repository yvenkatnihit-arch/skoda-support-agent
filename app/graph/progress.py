"""Progress messages shown while a turn runs (CLI and API). Agents' texts come from their own files."""

from app.agents import CATALOG

STATUS = {
    "supervisor": "Understanding your question...",
    **{name: spec.status_text for name, spec in CATALOG.agents.items()},
    "escalate": "Getting you to the right person...",
    "tools": "Looking that up...",
    "composer": "Writing the reply...",
}
