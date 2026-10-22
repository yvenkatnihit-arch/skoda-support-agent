"""Request and response shapes for the API. Inputs are validated here: the customer's text is untrusted."""

from pydantic import BaseModel, Field

MAX_MESSAGE_CHARS = 2000
MAX_FEEDBACK_CHARS = 500


class ChatRequest(BaseModel):
    thread_id: str | None = Field(default=None, description="Conversation id; omit to start a new conversation")
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    lat: float | None = Field(default=None, ge=6.0, le=38.0, description="Shared location (India only)")
    long: float | None = Field(default=None, ge=68.0, le=98.0)


class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool
    feedback: str | None = Field(default=None, max_length=MAX_FEEDBACK_CHARS)


class ChatMessage(BaseModel):
    role: str  # "customer" | "assistant"
    text: str


class ThreadView(BaseModel):
    thread_id: str
    messages: list[ChatMessage]
    pending: dict | None = None  # the Confirm/Cancel request, if the conversation is paused
