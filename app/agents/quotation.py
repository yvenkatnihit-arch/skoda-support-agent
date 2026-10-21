from app.agents.base import AgentSpec
from app.prompts.quotation import QUOTATION_PROMPT

AGENT = AgentSpec(
    name="quotation",
    intent="quotation",
    intent_help=(
        "price, on-road price, quote, cost to buy, range on a full tank, ownership cost for buying, and price\n"
        '  comparisons between models ("which is cheaper?", "Kylaq or Kushaq, which costs less?").\n'
        '  Examples: "On-road price of Kylaq Signature AT in Hyderabad" / "Price of Slavia" / '
        '"what about the AT version?" (right after a quote)'),
    role="price quotation specialist",
    prompt=QUOTATION_PROMPT,
    reply_order=20,
    status_text="Working out the price...",
    required_slots=("city",),
    required_car_slots=("model", "variant"),
    comparison_relaxes=True,  # "Is the Kylaq cheaper than the Kushaq?" -> compare starting prices, no clarify
)
