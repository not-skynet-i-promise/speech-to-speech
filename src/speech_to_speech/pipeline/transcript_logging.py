"""Opt-in gate for writing conversation content to application loggers.

Operational logs are retained by service managers, containers and hosted logging systems, so
transcript text written through ``logger.*`` outlives the conversation in places the user
never agreed to. Content is therefore omitted by default: log sites pass transcripts through
:func:`transcript_for_log`, which yields a character count unless transcript logging has been
explicitly enabled.

Full transcripts are still valuable when debugging STT, LLM, TTS and Realtime behaviour, so
``--log_transcripts`` turns them back on for a run. The gate is a module-level flag set once
at startup rather than a parameter threaded through every handler: the default has to be
"off" everywhere, including in code paths that never think about it, and a single default-off
switch cannot be missed by a new call site the way a constructor argument can.

Rich/terminal conversation display is unaffected -- that is an operator watching a live
session, not a retained log.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_log_transcripts = False

TRANSCRIPT_LOGGING_WARNING = (
    "Transcript logging is ENABLED (--log_transcripts): full user and assistant "
    "conversation content will be written to the application log. Anywhere those logs are "
    "collected or retained -- journald, container logs, hosted log aggregators -- will hold "
    "sensitive conversation data. Do not enable this in production."
)


def set_log_transcripts(enabled: bool) -> None:
    """Enable or disable transcript content in log records for this process."""
    global _log_transcripts
    _log_transcripts = bool(enabled)


def log_transcripts_enabled() -> bool:
    """Whether log sites may include conversation content."""
    return _log_transcripts


def warn_if_log_transcripts_enabled() -> None:
    """Emit the prominent opt-in warning, before any conversation is processed."""
    if _log_transcripts:
        logger.warning(TRANSCRIPT_LOGGING_WARNING)


def transcript_for_log(text: object) -> str:
    """Render *text* for a log record: the content when opted in, otherwise its length.

    Always returns a string, so call sites keep a single ``%s`` placeholder and read the
    same in both modes:

        Transcription completed (language=en): chars=42
        Transcription completed (language=en): see you at nine
    """
    value = "" if text is None else str(text)
    if _log_transcripts:
        return value
    return f"chars={len(value)}"


def transcript_for_response_log(text: object, response: object | None) -> str:
    """Render response content while always suppressing out-of-band prose.

    ``conversation: none`` responses intentionally bypass durable conversation
    and client text sinks. That stronger boundary also applies when the operator
    has explicitly enabled ordinary transcript logging.
    """
    conversation = getattr(response, "conversation", None)
    if isinstance(response, dict):
        conversation = response.get("conversation")
    if conversation == "none":
        return "content=isolated-response-suppressed"
    return transcript_for_log(text)


def assistant_console_text(text: object, response: object | None) -> str | None:
    """Return live assistant text unless the response is explicitly isolated."""
    conversation = getattr(response, "conversation", None)
    if isinstance(response, dict):
        conversation = response.get("conversation")
    if conversation == "none":
        return None
    return "" if text is None else str(text)


def log_exception(
    target: logging.Logger,
    message: str,
    exc: BaseException,
    *,
    level: int = logging.ERROR,
) -> None:
    """Log an exception without exposing its message or traceback by default."""
    if _log_transcripts:
        target.log(
            level,
            "%s (%s): %s",
            message,
            type(exc).__name__,
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return
    target.log(level, "%s (%s)", message, type(exc).__name__)


def log_response_exception(
    target: logging.Logger,
    message: str,
    exc: BaseException,
    response: object | None,
    *,
    level: int = logging.ERROR,
) -> None:
    """Log failures without ever exposing isolated response content."""
    if assistant_console_text("isolated", response) is None:
        target.log(level, "%s (%s)", message, type(exc).__name__)
        return
    log_exception(target, message, exc, level=level)
