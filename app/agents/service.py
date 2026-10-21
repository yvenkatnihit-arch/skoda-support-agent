from app.agents.base import AgentSpec
from app.prompts.service import SERVICE_PROMPT

AGENT = AgentSpec(
    name="service",
    intent="service_info",
    intent_help=(
        "servicing, service schedule or service cost, booking a service, warranty, extended warranty,\n"
        "  roadside assistance, service packages, service centre working hours.\n"
        '  Examples: "Service cost for Kodiaq for 3 years" / "What does the warranty cover?" / '
        '"How do I book a service?"'),
    role="service and warranty specialist",
    prompt=SERVICE_PROMPT,
    reply_order=30,
    status_text="Checking service details...",
)
