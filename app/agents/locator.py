from app.agents.base import AgentSpec
from app.prompts.locator import LOCATOR_PROMPT

AGENT = AgentSpec(
    name="locator",
    intent="locator",
    intent_help=(
        "nearest showroom, dealer or service centre, their address, phone or location.\n"
        '  Examples: "Nearest showroom to Kukatpally" / "Service centre near 560034" / '
        '"Where is the Skoda dealer in Andheri?"'),
    role="showroom and service centre locator",
    prompt=LOCATOR_PROMPT,
    reply_order=40,
    status_text="Finding showrooms...",
    required_slots=(("area_or_pincode", "city", ("lat", "long")),),  # any one of these
)
