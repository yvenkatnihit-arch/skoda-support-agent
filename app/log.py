"""structlog setup. Every log event passes through mask_phone_numbers before it is written."""

import logging
import re
from pathlib import Path

import structlog

# Indian mobile/landline-style numbers: optional +91/91/0 prefix, then 10 digits (spaces/hyphens allowed).
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?|0)?[1-9](?:[\s-]?\d){9}(?!\d)")


def mask_phone(text: str) -> str:
    """Replace each phone number with X's, keeping only its last 2 digits."""

    def _mask(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group())
        return "X" * 8 + digits[-2:]

    return PHONE_RE.sub(_mask, text)


def mask_value(value):
    """Mask phone numbers in every string inside a str / dict / list / tuple."""
    if isinstance(value, str):
        return mask_phone(value)
    if isinstance(value, dict):
        return {k: mask_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(mask_value(v) for v in value)
    return value


def mask_phone_numbers(_logger, _method, event_dict: dict) -> dict:
    """structlog processor: mask phone numbers in every string field of the event."""
    return {k: mask_value(v) for k, v in event_dict.items()}


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Console logging, or JSON lines appended to log_file (so errors stay out of the customer's view)."""
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,  # before masking, so tracebacks are masked too
        mask_phone_numbers,
    ]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        processors.append(structlog.processors.JSONRenderer(ensure_ascii=False))
        factory = structlog.PrintLoggerFactory(file=log_file.open("a", encoding="utf-8"))
    else:
        processors.append(structlog.dev.ConsoleRenderer())
        factory = structlog.PrintLoggerFactory()
    structlog.configure(
        processors=processors,
        logger_factory=factory,
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelNamesMapping()[level.upper()]),
        cache_logger_on_first_use=True,
    )
