from app.agents.base import AgentSpec
from app.prompts.lead import LEAD_PROMPT

AGENT = AgentSpec(
    name="lead",
    intent="lead_capture",
    intent_help=(
        "test drive, brochure request, callback, talk to a person, or the customer giving their contact details.\n"
        '  Examples: "Book a test drive" / "Send me the Kylaq brochure" / "Can someone call me?" / '
        '"Ravi, 9876543210"'),
    role="test drive, brochure and callback specialist",
    prompt=LEAD_PROMPT,
    reply_order=50,
    status_text="Preparing your request...",
)
