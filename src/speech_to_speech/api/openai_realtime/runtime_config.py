from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock
from typing import Any

from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.realtime.realtime_audio_config import RealtimeAudioConfig
from openai.types.realtime.realtime_audio_config_input import RealtimeAudioConfigInput
from openai.types.realtime.realtime_audio_config_output import RealtimeAudioConfigOutput
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from speech_to_speech.LLM.chat import Chat


def _apply_update(current: BaseModel, update: BaseModel) -> None:
    """Apply explicitly-set fields from *update* onto *current* in-place,
    recursing into nested BaseModel children so partial nested updates
    don't overwrite unset fields.

    Only fields present in update.model_fields_set (i.e. actually
    sent by the client) are considered.
    """
    for field_name in update.model_fields_set:
        new_val = getattr(update, field_name)
        old_val = getattr(current, field_name, None)
        if isinstance(new_val, BaseModel) and isinstance(old_val, BaseModel):
            _apply_update(old_val, new_val)
        else:
            setattr(current, field_name, new_val)


class RuntimeConfig(BaseModel):
    """
    Shared mutable configuration written by the RealtimeService on
    session.update and read by pipeline handlers (VAD, LLM, TTS) during
    processing.  Python's GIL makes simple attribute reads/writes atomic,
    so no explicit locking is needed for primitive values.

    The canonical state lives in 'session' (a full
    'RealtimeSessionCreateRequest').
    """

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    chat: Chat = Field(default_factory=lambda: Chat(10))
    session: RealtimeSessionCreateRequest = Field(
        default_factory=lambda: RealtimeSessionCreateRequest(type="realtime"),
        validate_default=True,
    )
    transcript_barrier_version: int | None = Field(default=None, exclude=True)
    transcript_barrier_nonce: str | None = Field(default=None, exclude=True)
    transcript_barrier_failed: bool = Field(default=False, exclude=True)
    transcript_barrier_session_updates: int = Field(default=0, exclude=True)
    transcript_barrier_sequence: int = Field(default=0, exclude=True)
    transcript_barrier_pending_sequence: int | None = Field(default=None, exclude=True)
    transcript_barrier_pending_item_id: str | None = Field(default=None, exclude=True)
    transcript_barrier_pending_transcript: str | None = Field(default=None, exclude=True, repr=False)
    _transcript_barrier_state_lock: Any = PrivateAttr(default_factory=RLock)

    @field_validator("session", mode="after")
    @classmethod
    def _ensure_audio_structure(cls, v: RealtimeSessionCreateRequest) -> RealtimeSessionCreateRequest:
        """Guarantee 'audio.input' and 'audio.output' are never None."""
        if v.audio is None:
            v.audio = RealtimeAudioConfig()
        if v.audio.input is None:
            v.audio.input = RealtimeAudioConfigInput()
        if v.audio.output is None:
            v.audio.output = RealtimeAudioConfigOutput()
        return v

    @property
    def interrupt_response_enabled(self) -> bool:
        """Whether barge-in should cancel an active response.

        Reads 'turn_detection.interrupt_response' from the session config,
        handling both Pydantic models ('ServerVad') and plain dicts.
        Defaults to 'True' (OpenAI API default).
        """
        assert self.session.audio is not None and self.session.audio.input is not None
        td = self.session.audio.input.turn_detection
        if td is None:
            return True
        if hasattr(td, "interrupt_response"):
            val = td.interrupt_response
        elif isinstance(td, dict):
            val = td.get("interrupt_response", True)
        else:
            return True
        return val if val is not None else True

    @property
    def create_response_enabled(self) -> bool:
        """Whether a completed transcription should trigger a response.

        Reads 'turn_detection.create_response' from the session config,
        handling both Pydantic models ('ServerVad') and plain dicts.
        Defaults to 'True' (OpenAI API default).
        """
        assert self.session.audio is not None and self.session.audio.input is not None
        td = self.session.audio.input.turn_detection
        if td is None:
            return True
        if hasattr(td, "create_response"):
            val = td.create_response
        elif isinstance(td, dict):
            val = td.get("create_response", True)
        else:
            return True
        return val if val is not None else True

    def apply_session_update(self, update: RealtimeSessionCreateRequest) -> None:
        """Merge non-None, explicitly-set fields from 'update' into the
        current 'session', preserving any fields not present in the update."""
        _apply_update(self.session, update)

    @property
    def transcript_barrier_enabled(self) -> bool:
        """Whether this session ever completed the exact opt-in handshake.

        Protocol authority is granted only by a completed handshake.  Content
        privacy while rejected or poisoned work drains is represented
        separately by :attr:`transcript_barrier_private`.
        """
        return self.transcript_barrier_version == 1 and self.transcript_barrier_nonce is not None

    @property
    def transcript_barrier_private(self) -> bool:
        """Whether pipeline content must be treated as private while draining.

        A rejected activation never gains protocol authority, but it still
        makes the session private before the socket closes.  Audio may already
        be inside worker queues at that point, so every content/log/output gate
        must remain redacted and fail closed until the session is fully reset.
        """
        return self.transcript_barrier_enabled or self.transcript_barrier_failed

    @property
    def transcript_barrier_operational(self) -> bool:
        """Whether the negotiated barrier may still accept protocol work."""
        return self.transcript_barrier_enabled and not self.transcript_barrier_failed

    @contextmanager
    def transcript_barrier_state_guard(self) -> Iterator[None]:
        """Serialize barrier failure transitions with conversation write-back.

        A provider iterator may still be running on a pipeline thread while the
        websocket thread poisons the session.  Callers that either latch failure
        or persist model output must hold this guard so those operations have a
        single ordering: a history write completes before poison, or observes the
        poison and is discarded.
        """
        with self._transcript_barrier_state_lock:
            yield

    @contextmanager
    def transcript_barrier_content_guard(self) -> Iterator[bool]:
        """Atomically select whether a content-bearing side effect is private."""
        with self.transcript_barrier_state_guard():
            yield self.transcript_barrier_private

    def next_transcript_barrier_sequence(self) -> int:
        """Allocate one monotonic event sequence within the current session."""
        self.transcript_barrier_sequence += 1
        return self.transcript_barrier_sequence

    @property
    def transcript_barrier_pending(self) -> bool:
        return self.transcript_barrier_pending_sequence is not None

    def clear_transcript_barrier_pending(self) -> None:
        """Scrub the only short-lived raw transcript retained by the barrier."""
        self.transcript_barrier_pending_sequence = None
        self.transcript_barrier_pending_item_id = None
        self.transcript_barrier_pending_transcript = None
