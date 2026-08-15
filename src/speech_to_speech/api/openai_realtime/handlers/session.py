from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from openai.types.realtime import (
    SessionCreatedEvent,
    SessionUpdateEvent,
)
from openai.types.realtime.realtime_transcription_session_create_request import (
    RealtimeTranscriptionSessionCreateRequest,
)

from speech_to_speech.api.openai_realtime.handlers.base import RealtimeBaseHandler
from speech_to_speech.api.openai_realtime.transcript_barrier import (
    TRANSCRIPT_BARRIER_FIELD,
    TRANSCRIPT_BARRIER_VERSION,
    TranscriptBarrierReadyEvent,
    parse_transcript_barrier_request,
)

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import ServerEvent

logger = logging.getLogger(__name__)


class SessionHandler(RealtimeBaseHandler):
    """Owns session lifecycle: config updates and session.created events."""

    def handle_session_update(self, conn_id: str, event: SessionUpdateEvent) -> Optional[ServerEvent]:
        """Apply session config changes.

        Only ``RealtimeSessionCreateRequest`` sessions are accepted;
        ``RealtimeTranscriptionSessionCreateRequest`` sessions not yet supported.
        Incoming fields are deep-merged into the existing session so that
        partial updates preserve previously-set values.
        """
        s = event.session
        if s is None:
            return None

        if isinstance(s, RealtimeTranscriptionSessionCreateRequest):
            return self.make_error(
                message="Only 'realtime' session type is supported; transcription sessions are not.",
                _type="invalid_session_type",
            )

        cfg = self._state(conn_id).runtime_config
        state = self._state(conn_id)
        extra = s.model_extra or {}
        barrier_requested = TRANSCRIPT_BARRIER_FIELD in extra
        barrier_nonce: str | None = None
        if barrier_requested:
            barrier_nonce = parse_transcript_barrier_request(extra.get(TRANSCRIPT_BARRIER_FIELD))
            extra.pop(TRANSCRIPT_BARRIER_FIELD, None)
            s.model_fields_set.discard(TRANSCRIPT_BARRIER_FIELD)
            if (
                barrier_nonce is None
                or cfg.transcript_barrier_version is not None
                or cfg.transcript_barrier_session_updates != 0
                or state.audio_buffer_has_data
                or bool(state.audio_remainder)
                or state.input_audio_duration_s != 0.0
                or state.current_item_id is not None
                or state.last_item_id is not None
                or state.in_response
                or state.response_pending
                or bool(cfg.chat.buffer)
            ):
                return self._service.poison_transcript_barrier(conn_id, "invalid_transcript_barrier")

        model = getattr(s, "model", None)
        if model is not None:
            if cfg.transcript_barrier_private or barrier_nonce is not None:
                logger.info("Private session model updated; content redacted")
            else:
                logger.info("Session model set to: %s", model)

        current = cfg.session
        if current is None:
            cfg.session = s
        else:
            cfg.apply_session_update(s)
        cfg.transcript_barrier_session_updates += 1
        logger.info("Session configuration updated")
        if barrier_nonce is not None:
            cfg.transcript_barrier_version = TRANSCRIPT_BARRIER_VERSION
            cfg.transcript_barrier_nonce = barrier_nonce
            cfg.chat.enable_private_content_logging()
            return TranscriptBarrierReadyEvent(
                event_id=self._next_event_id(),
                nonce=barrier_nonce,
            )
        return None

    def build_session_created(self, conn_id: str) -> SessionCreatedEvent:
        """Build a SessionCreatedEvent populated with the current config."""
        cfg = self._state(conn_id).runtime_config
        session = cfg.session
        return SessionCreatedEvent(
            type="session.created",
            event_id=self._next_event_id(),
            session=session,
        )
