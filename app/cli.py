"""Chat with the assistant in the terminal (real Gemini).

    python -m app.cli                     # new conversation
    python -m app.cli --thread my-chat-1  # continue a saved conversation
    python -m app.cli --verbose           # also show which node is running

Logs (one "turn" summary line per question) go to data/logs/app.log. With LANGSMITH_TRACING=true in .env,
each turn is also sent to LangSmith, phone-masked.

Type 'exit' to quit. When the assistant wants to save your details it asks you to confirm (y/n).
"""

import argparse
import json
import sys
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.config import get_settings
from app.graph.build_graph import build_graph, recursion_limit
from app.graph.progress import STATUS
from app.graph.runner import run_turn
from app.log import configure_logging, mask_phone
from app.memory.file_store import restore_if_missing, save_state
from app.observability import disable_auto_tracing, flush_tracing, tracing_callbacks

def show_status(node: str) -> None:
    if node in STATUS:
        print(f"  · {STATUS[node]}")


def ask_confirmation(pending: dict) -> dict:
    print("\nAssistant: I'd like to do this for you:")
    for action in pending.get("actions", []):
        details = ", ".join(f"{k}={v}" for k, v in action["details"].items() if v not in (None, ""))
        print(f"  - {action['tool']}: {details}")
    answer = input("Confirm? (y/n, or type feedback to cancel with a note): ").strip()
    if answer.lower() in ("y", "yes"):
        return {"approved": True}
    return {"approved": False, "feedback": None if answer.lower() in ("n", "no", "") else answer}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thread", default=None, help="thread id to continue (letters, digits, - and _)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    s = get_settings()
    # Logs (incl. each Gemini call's timing and any error) go to data/logs/app.log; --verbose shows them here too.
    configure_logging(s.log_level, None if args.verbose else s.data_dir / "logs" / "app.log")
    disable_auto_tracing()
    try:
        callbacks = tracing_callbacks(s)
    except RuntimeError as exc:
        print(f"Setup problem: {exc}", file=sys.stderr)
        return 1
    thread_id = args.thread or uuid.uuid4().hex[:12]
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit(s.max_tool_rounds),
              "callbacks": callbacks}
    graph = build_graph()
    if s.langsmith_tracing:
        print(f"(LangSmith tracing on, project '{s.langsmith_project}', phone numbers masked)")
    try:
        return chat(graph, config, thread_id, args.verbose)
    finally:
        flush_tracing(s)


def chat(graph, config: dict, thread_id: str, verbose: bool) -> int:
    if restore_if_missing(graph, config):
        print(f"(restored conversation {thread_id})")
    print(f"Škoda India assistant. Conversation id: {thread_id}. Type 'exit' to quit.\n")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if text.lower() in ("exit", "quit"):
            return 0
        if not text:
            continue
        on_node = show_status if verbose else None
        result = run_turn(graph, config, {"messages": [HumanMessage(content=text)]}, on_node)
        while result.status == "paused":
            save_state(thread_id, graph.get_state(config).values, paused=True)
            result = run_turn(graph, config, Command(resume=ask_confirmation(result.pending)), on_node)
        if result.status == "error":  # State was rolled back; details are in the log only
            print(f"\nAssistant: {result.reply}\n")
            continue
        values = graph.get_state(config).values
        save_state(thread_id, values)
        print(f"\nAssistant: {result.reply or '(no reply)'}\n")
        if verbose:
            print(f"  [intents={values.get('intents')} slots="
                  f"{mask_phone(json.dumps({k: v for k, v in (values.get('slots') or {}).items() if v}))}]\n")


if __name__ == "__main__":
    sys.exit(main())
