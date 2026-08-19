from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openai.types.realtime import (
    SessionCreatedEvent,
    SessionUpdateEvent,
)
from openai.types.realtime.realtime_transcription_session_create_request import (
    RealtimeTranscriptionSessionCreateRequest,
)

from speech_to_speech.api.openai_realtime.handlers.base import RealtimeBaseHandler
from speech_to_speech.api.openai_realtime.home_assistant_guard import (
    HOME_ASSISTANT_GUARD_FIELD,
    HOME_ASSISTANT_GUARD_VERSION,
    HOME_ASSISTANT_TOOL_PREFIX,
    HomeAssistantGuardReadyEvent,
    parse_home_assistant_guard_request,
    session_contract,
    valid_guarded_tool_choice,
)
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

    def handle_session_update(
        self,
        conn_id: str,
        event: SessionUpdateEvent,
    ) -> ServerEvent | list[ServerEvent] | None:
        """Apply session config changes.

        Only ``RealtimeSessionCreateRequest`` sessions are accepted;
        ``RealtimeTranscriptionSessionCreateRequest`` sessions not yet supported.
        Incoming fields are deep-merged into the existing session so that
        partial updates preserve previously-set values.
        """
        s = event.session
        if s is None:
            return None

        state = self._state(conn_id)
        cfg = state.runtime_config
        cfg.transcript_barrier_session_updates += 1
        extra = s.model_extra or {}
        barrier_requested = TRANSCRIPT_BARRIER_FIELD in extra
        home_assistant_requested = HOME_ASSISTANT_GUARD_FIELD in extra
        home_assistant_required = self._service.home_assistant_guard_required
        private_request = barrier_requested or home_assistant_requested or home_assistant_required

        if isinstance(s, RealtimeTranscriptionSessionCreateRequest):
            barrier_error = None
            if barrier_requested:
                barrier_error = self._service.poison_transcript_barrier(
                    conn_id,
                    "invalid_transcript_barrier",
                )
            if home_assistant_requested or home_assistant_required:
                home_assistant_error = self._service.poison_home_assistant_guard(
                    conn_id,
                    "invalid_home_assistant_guard",
                )
                if barrier_error is None:
                    return home_assistant_error
            if barrier_error is not None:
                return barrier_error
            return self.make_error(
                message="Only 'realtime' session type is supported; transcription sessions are not.",
                _type="invalid_session_type",
            )

        barrier_nonce: str | None = None
        home_assistant_nonce: str | None = None
        home_assistant_digest: str | None = None
        home_assistant_tool_count = 0
        ordered_tool_names: tuple[str, ...] = ()

        if barrier_requested:
            barrier_nonce = parse_transcript_barrier_request(
                extra.get(TRANSCRIPT_BARRIER_FIELD),
            )
            extra.pop(TRANSCRIPT_BARRIER_FIELD, None)
            s.model_fields_set.discard(TRANSCRIPT_BARRIER_FIELD)

        try:
            home_assistant_digest, home_assistant_tool_count, ordered_tool_names = session_contract(
                getattr(s, "instructions", None),
                getattr(s, "tools", None),
            )
        except (TypeError, ValueError):
            home_assistant_digest = None
            ordered_tool_names = ()
        has_home_assistant_tools = any(name.startswith(HOME_ASSISTANT_TOOL_PREFIX) for name in ordered_tool_names)
        if home_assistant_requested:
            if home_assistant_digest is not None:
                home_assistant_nonce = parse_home_assistant_guard_request(
                    extra.get(HOME_ASSISTANT_GUARD_FIELD),
                    expected_digest=home_assistant_digest,
                    expected_tool_count=home_assistant_tool_count,
                )
            extra.pop(HOME_ASSISTANT_GUARD_FIELD, None)
            s.model_fields_set.discard(HOME_ASSISTANT_GUARD_FIELD)

        home_assistant_context = (
            home_assistant_requested
            or home_assistant_required
            or cfg.home_assistant_guard_enabled
            or cfg.home_assistant_guard_failed
        )
        invalid_activation = private_request and (
            cfg.transcript_barrier_session_updates != 1
            or state.audio_buffer_has_data
            or state.audio_append_seen
            or bool(state.audio_remainder)
            or state.input_audio_duration_s != 0.0
            or state.current_item_id is not None
            or state.last_item_id is not None
            or state.in_response
            or state.response_pending
            or bool(cfg.chat.buffer)
            or self._service.cancel_scope is None
            or not self._service.cancel_scope_wiring_verified
        )
        invalid_barrier = barrier_requested and (barrier_nonce is None or cfg.transcript_barrier_version is not None)
        invalid_home_assistant = home_assistant_context and (
            not self._service.home_assistant_guard_supported
            or not home_assistant_requested
            or home_assistant_nonce is None
            or home_assistant_digest is None
            or not has_home_assistant_tools
            or not valid_guarded_tool_choice(getattr(s, "tool_choice", None))
            or cfg.home_assistant_guard_version is not None
        )
        if invalid_activation or invalid_barrier or invalid_home_assistant:
            barrier_error = None
            if barrier_requested or cfg.transcript_barrier_enabled:
                barrier_error = self._service.poison_transcript_barrier(
                    conn_id,
                    "invalid_transcript_barrier",
                )
            if home_assistant_context:
                return self._service.poison_home_assistant_guard(
                    conn_id,
                    "invalid_home_assistant_guard",
                )
            assert barrier_error is not None
            return barrier_error

        if private_request:
            cancel_scope = self._service.cancel_scope
            assert cancel_scope is not None
            with cfg.transcript_barrier_state_guard():
                with cancel_scope.private_activation_guard() as quiescent:
                    if not quiescent:
                        if barrier_requested:
                            self._service.poison_transcript_barrier(
                                conn_id,
                                "invalid_transcript_barrier",
                            )
                        if home_assistant_requested:
                            return self._service.poison_home_assistant_guard(
                                conn_id,
                                "invalid_home_assistant_guard",
                            )
                        return self._service.poison_transcript_barrier(
                            conn_id,
                            "invalid_transcript_barrier",
                        )

                    model = getattr(s, "model", None)
                    if model is not None:
                        logger.info("Private session model updated; content redacted")
                    cfg.apply_session_update(s)
                    ready_events: list[ServerEvent] = []
                    if barrier_requested:
                        assert barrier_nonce is not None
                        cfg.transcript_barrier_version = TRANSCRIPT_BARRIER_VERSION
                        cfg.transcript_barrier_nonce = barrier_nonce
                        ready_events.append(
                            TranscriptBarrierReadyEvent(
                                event_id=self._next_event_id(),
                                nonce=barrier_nonce,
                            )
                        )
                    if home_assistant_requested:
                        assert home_assistant_nonce is not None
                        assert home_assistant_digest is not None
                        cfg.home_assistant_guard_version = HOME_ASSISTANT_GUARD_VERSION
                        cfg.home_assistant_guard_nonce = home_assistant_nonce
                        cfg.home_assistant_guard_contract_sha256 = home_assistant_digest
                        cfg.home_assistant_guard_tool_count = home_assistant_tool_count
                        cfg.home_assistant_guard_tool_names = ordered_tool_names
                        ready_events.append(
                            HomeAssistantGuardReadyEvent(
                                event_id=self._next_event_id(),
                                nonce=home_assistant_nonce,
                                session_contract_sha256=home_assistant_digest,
                                tool_count=home_assistant_tool_count,
                            )
                        )
                    cfg.chat.enable_private_content_logging()
                    logger.info("Session configuration updated")
                    return ready_events[0] if len(ready_events) == 1 else ready_events

        model = getattr(s, "model", None)
        if model is not None:
            if cfg.sensitive_content or private_request:
                logger.info("Private session model updated; content redacted")
            else:
                logger.info("Session model set to: %s", model)

        current = cfg.session
        if current is None:
            cfg.session = s
        else:
            cfg.apply_session_update(s)
        logger.info("Session configuration updated")
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
