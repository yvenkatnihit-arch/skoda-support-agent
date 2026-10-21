r"""Entry point for LangGraph Studio (a development tool, not for customers).

    .venv-studio\Scripts\langgraph dev        # from the project folder; opens Studio in the browser

The dev server brings its own conversation memory, so the graph is compiled without one. Logs go to
data/logs/studio.log. Tracing follows LANGSMITH_TRACING in .env and uses the same phone-masked tracer as the
CLI; the SDK's own environment-variable tracing is switched off so nothing is sent unmasked.
"""

from app.config import get_settings
from app.graph.build_graph import build_graph, recursion_limit
from app.log import configure_logging
from app.observability import disable_auto_tracing, tracing_callbacks

_settings = get_settings()
configure_logging(_settings.log_level, _settings.data_dir / "logs" / "studio.log")
disable_auto_tracing()

graph = build_graph(memory=False).with_config(
    recursion_limit=recursion_limit(_settings.max_tool_rounds),
    callbacks=tracing_callbacks(_settings),
    run_name="skoda_turn",
)
