from app.agents.base import AgentSpec
from app.prompts.support import SUPPORT_PROMPT

AGENT = AgentSpec(
    name="support",
    intent="support_request",
    intent_help=(
        "complaints and problems from existing owners: something wrong with the car, a bad service visit, a\n"
        "  dealer or staff issue, a billing problem, raising a ticket, or the status of a ticket. Also the customer\n"
        "  giving contact details after the assistant asked for them to raise a ticket.\n"
        '  Examples: "My Kushaq\'s AC stopped working after the service" / "I want to complain about the dealer" /\n'
        '  "Status of ticket SKD-20261021-0001?"'),
    role="customer support and complaints specialist",
    prompt=SUPPORT_PROMPT,
    reply_order=60,
    status_text="Looking into your issue...",
    history_messages=10,  # details are gathered over several turns; early ones must not drop out of view
)
