from app.agents.base import AgentSpec
from app.prompts.car_info import CAR_INFO_PROMPT

AGENT = AgentSpec(
    name="car_info",
    intent="car_info",
    intent_help=(
        "features, specs, variants, engines, safety, dimensions, colours, and comparisons of those between\n"
        "  models (not prices: a price comparison is quotation).\n"
        "  Also which variants a model has, or what a variant is.\n"
        '  Examples: "What safety features does the Kushaq have?" / "Kushaq vs Slavia" / '
        '"Does the Kylaq have a sunroof?"'),
    role="car information specialist",
    prompt=CAR_INFO_PROMPT,
    reply_order=10,
    status_text="Checking the brochures...",
)
