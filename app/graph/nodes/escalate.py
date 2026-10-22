"""Escalate: hand the customer to a person (ROADMAP section 13). Plain Python, no Gemini call.

The router sends a turn here when the supervisor flags frustration, a safety issue or a dispute, or when the bot
has failed or got stuck ESCALATE_AFTER_STRIKES turns in a row. It writes the hand-off details as the only
finding, and the composer turns them into an empathetic reply. A callback (lead) or an escalated ticket
(support) follows on the next turn, through the normal Confirm/Cancel flow.
"""

import structlog

from app.config import get_settings

log = structlog.get_logger()

ESCALATION_AGENT = "escalation"  # its key in agent_outputs
MARKER = "HAND-OFF TO A PERSON"  # tells the composer these are instructions; stripped if it leaks into a reply

REASONS = {
    "frustration": "The customer is frustrated with the assistant: apologise sincerely in one line, without "
                   "defending the assistant or repeating earlier answers.",
    "safety": "This may be a safety issue: start by checking they're safe. Tell them that if anyone is hurt or in "
              "danger they should call 112, and that if the car may be unsafe they shouldn't drive it but call "
              "Škoda Assist, the 24x7 roadside assistance, through customer care.",
    "dispute": "This is a dispute (legal, refund or compensation): don't argue, promise or admit anything; say a "
               "person from customer care will review it.",
    "strikes": "The assistant hasn't been able to help in the last few messages: apologise in one line and offer "
               "a person instead of trying again.",
}


def handoff_notes(reason: str) -> str:
    care = get_settings().customer_care_number
    return (f"{MARKER}: instructions for your reply, not text to copy. {REASONS.get(reason, REASONS['strikes'])} "
            f"Then, in your own warm, natural words, say a person can help and offer three ways: calling Škoda "
            f"India customer care on the toll-free number {care}; a callback from the Škoda team (they just share "
            "their name, mobile number and city); or, for a complaint, raising a high-priority ticket. Ask which "
            "they'd like. Don't answer the rest of the message this time.")


def escalate(state: dict) -> dict:
    reason = state.get("escalation") or "strikes"
    log.info("escalated to a person", reason=reason, strikes=state.get("strikes") or 0)
    return {
        "agent_outputs": {ESCALATION_AGENT: {"answer": handoff_notes(reason), "tool_results": [],
                                             "status": "complete"}},
        "strikes": 0,          # handed off: start counting again
        "escalated": True,     # support raises its ticket as high priority from now on
    }
