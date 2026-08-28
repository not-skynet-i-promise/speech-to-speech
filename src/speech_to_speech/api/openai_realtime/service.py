import logging
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from queue import Queue
from threading import Event as ThreadingEvent
from typing import Any, Callable, Iterator, Literal, Optional, TypeVar, Union

from openai.types.realtime import (
    ConversationItem,
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeError,
    RealtimeErrorEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCancelEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    SessionCreatedEvent,
    SessionUpdateEvent,
)
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from speech_to_speech.api.openai_realtime.handlers import (
    AudioHandler,
    ConversationHandler,
    ResponseHandler,
    SessionHandler,
)
from speech_to_speech.api.openai_realtime.home_assistant_guard import (
    HomeAssistantGuardReadyEvent,
    session_contract,
    valid_guarded_tool_choice,
)
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.transcript_barrier import (
    TRANSCRIPT_BARRIER_MAX_CHARS,
    TranscriptBarrierCompletedServerEvent,
    TranscriptBarrierDiscardedServerEvent,
    TranscriptBarrierFailedServerEvent,
    TranscriptBarrierReadyEvent,
    TranscriptBarrierResolvedServerEvent,
    TranscriptBarrierResolveEvent,
)
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import (
    AssistantTextEvent,
    PartialTranscriptionEvent,
    PipelineEvent,
    ResponseFailedEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TokenUsageEvent,
    TranscriptBarrierCompletedEvent,
    TranscriptBarrierDiscardedEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.messages import GenerateResponseRequest
from speech_to_speech.pipeline.queue_types import TextPromptItem
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.utils.utils import _generate_id, is_out_of_band

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512
BYTES_PER_SAMPLE = 2
CHUNK_SIZE_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE
MAX_TRACKED_PROTOCOL_ITEMS = 2048
MAX_DEFERRED_RESPONSE_REQUESTS = 8

_ResponseStatus = Literal["completed", "cancelled", "failed", "incomplete", "in_progress"]
_StatusReason = Literal["turn_detected", "client_cancelled", "max_output_tokens", "content_filter"]

_EVENT_TYPE_TO_MODEL: dict[str, type[BaseModel]] = {
    "input_audio_buffer.append": InputAudioBufferAppendEvent,
    "session.update": SessionUpdateEvent,
    "conversation.item.create": ConversationItemCreateEvent,
    "conversation.item.delete": ConversationItemDeleteEvent,
    "response.create": ResponseCreateEvent,
    "response.cancel": ResponseCancelEvent,
    "reachy.transcript_barrier.resolve": TranscriptBarrierResolveEvent,
}

ClientEvent = Union[
    InputAudioBufferAppendEvent,
    SessionUpdateEvent,
    ConversationItemCreateEvent,
    ConversationItemDeleteEvent,
    ResponseCreateEvent,
    ResponseCancelEvent,
    TranscriptBarrierResolveEvent,
]

ServerEvent = Union[
    SessionCreatedEvent,
    RealtimeErrorEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    ConversationItemCreatedEvent,
    ConversationItemDeletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    HomeAssistantGuardReadyEvent,
    TranscriptBarrierReadyEvent,
    TranscriptBarrierCompletedServerEvent,
    TranscriptBarrierDiscardedServerEvent,
    TranscriptBarrierFailedServerEvent,
    TranscriptBarrierResolvedServerEvent,
]

RealtimeEvent = Union[ClientEvent, ServerEvent]


_UsageMetricsT = TypeVar("_UsageMetricsT", bound="UsageMetrics")


class UsageMetrics(BaseModel):
    """Per-response usage counters.

    Supports ``+=`` for rolling per-response metrics into a global total
    and ``reset()`` for clearing per-response state after rollup.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    audio_duration_s: float = 0.0
    responses_completed: int = 0
    responses_cancelled: int = 0
    tool_calls: int = 0
    turns: int = 0

    def __iadd__(self: _UsageMetricsT, other: "UsageMetrics") -> _UsageMetricsT:
        for field in UsageMetrics.model_fields:
            setattr(self, field, getattr(self, field) + getattr(other, field))
        return self

    def reset(self) -> None:
        for field, info in UsageMetrics.model_fields.items():
            setattr(self, field, info.default)


class GlobalUsageMetrics(UsageMetrics):
    """Server-wide metrics that extend per-response counters with
    connection and error tracking."""

    connections: int = 0
    # connection duration in seconds.
    # latency tts, llm, vad, stt (mean, max, p90)
    errors_by_type: dict[str, int] = Field(default_factory=dict)

    def record_error(self, error_type: str) -> None:
        self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1

    @property
    def total_errors(self) -> int:
        return sum(self.errors_by_type.values())


class ConnState(BaseModel):
    """Per-connection mutable state, including all protocol-level IDs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str = Field(default_factory=lambda: _generate_id("session"))
    conversation_id: str = Field(default_factory=lambda: _generate_id("conv"))
    runtime_config: RuntimeConfig = Field(default_factory=RuntimeConfig)
    in_response: bool = False
    response_pending: bool = False
    audio_buffer_has_data: bool = False
    audio_append_seen: bool = False
    audio_remainder: bytes = b""
    current_response_id: Optional[str] = None
    current_item_id: Optional[str] = None
    content_index: int = 0
    input_content_index: int = 0
    input_audio_duration_s: float = 0.0
    last_item_id: Optional[str] = None
    current_response_params: RealtimeResponseCreateParams | None = None
    next_output_index: int = 0
    current_output_index: int | None = None
    current_output_kind: Literal["text", "tool_call"] | None = None
    current_output_item_id: str | None = None
    last_text_item_id: str | None = None
    last_text_output_index: int | None = None
    audio_output_started: bool = False
    # Each entry contains item_id, output_index, and the contiguous text parts
    # for one assistant message. Kept as plain internal data to avoid coupling
    # connection state to protocol event models.
    pending_text_outputs: list[dict[str, Any]] = Field(default_factory=list)
    response_usage: UsageMetrics = Field(default_factory=UsageMetrics)
    speculative_turn_id: Optional[str] = None
    speculative_turn_revision: Optional[int] = None
    speculative_user_turn_id: Optional[str] = None
    speculative_user_turn_revision: Optional[int] = None
    speculative_user_speech_stopped_at_s: Optional[float] = None
    speculative_user_item_id: Optional[str] = None
    speculative_input_item_id: Optional[str] = None
    speculative_audio_duration_s: float = 0.0
    pending_response_turn_id: Optional[str] = None
    pending_response_turn_revision: Optional[int] = None
    pending_response_request: GenerateResponseRequest | None = None
    pending_response_enqueued: bool = False
    # Later distinct turns wait here in arrival order while the current/pending
    # response owns the model lane.  The bound prevents an unattended client
    # from growing per-connection memory indefinitely; overflow is reported to
    # the client instead of silently replacing an accepted turn.
    deferred_response_requests: list[GenerateResponseRequest] = Field(default_factory=list)
    active_response_turn_id: Optional[str] = None
    active_response_turn_revision: Optional[int] = None
    active_response_cancel_generation: Optional[int] = None
    active_response_input_item_id: Optional[str] = None
    active_response_input_item_ids: set[str] = Field(default_factory=set)
    last_closed_response_turn_id: Optional[str] = None
    last_closed_response_cancel_generation: Optional[int] = None
    # A provider failure is surfaced on the text side-channel before its normal
    # terminal sentinel reaches the audio queue. Keep the active slot owned until
    # that sentinel arrives, otherwise it can close a newly promoted successor.
    response_failure_pending: bool = False
    # Exact default-conversation tail that an explicit in-band
    # ``response.create`` may own. Client-created/input items clear the audio
    # turn fields so an old audio deletion cannot cancel an unrelated response.
    response_context_input_item_id: Optional[str] = None
    response_context_input_item_ids: set[str] = Field(default_factory=set)
    response_context_turn_id: Optional[str] = None
    response_context_turn_revision: Optional[int] = None
    response_context_speech_stopped_at_s: Optional[float] = None
    # Client conversation.item.create items that arrived while a response was
    # generating. Applying them mid-generation races the LLM handler's chat
    # write-back (cross-thread), so they are buffered here and flushed in order
    # once the response completes. See ConversationHandler.flush_deferred_items.
    deferred_items: list[ConversationItem] = Field(default_factory=list)
    transcript_barrier_replacement_item_ids: set[str] = Field(default_factory=set)
    protocol_item_ids: list[str] = Field(default_factory=list)
    protocol_item_sequences: dict[str, int] = Field(default_factory=dict)
    deferred_protocol_item_sequences: dict[str, int] = Field(default_factory=dict)
    next_protocol_item_sequence: int = 0
    audio_input_item_ids: set[str] = Field(default_factory=set)
    input_item_chat_ids: dict[str, str] = Field(default_factory=dict)
    input_item_turn_ids: dict[str, str] = Field(default_factory=dict)
    turn_input_item_ids: dict[str, str] = Field(default_factory=dict)
    deleted_input_item_ids: OrderedDict[str, None] = Field(default_factory=OrderedDict)
    speculative_turn_tracker: SpeculativeTurnTracker | None = Field(default=None, exclude=True)

    def retire_reused_audio_item_id(self, item_id: str) -> None:
        """Drop stale audio identity before a deleted protocol ID is reused.

        Deleted audio IDs intentionally remain tombstoned so late STT events
        cannot recreate them.  A later client-created item may legally reuse
        the protocol ID, however, and must then be treated as that new item on
        deletion rather than as an unmapped audio placeholder.
        """

        if item_id not in self.audio_input_item_ids and item_id not in self.deleted_input_item_ids:
            return
        self.audio_input_item_ids.discard(item_id)
        self.deleted_input_item_ids.pop(item_id, None)
        retired_chat_id = self.input_item_chat_ids.pop(item_id, None)
        if retired_chat_id is not None:
            self.runtime_config.chat.retire_user_message_deletable(retired_chat_id)
        retired_turn_id = self.input_item_turn_ids.pop(item_id, None)
        if retired_turn_id is not None and self.speculative_turn_tracker is not None:
            self.speculative_turn_tracker.discard(retired_turn_id)
        if retired_turn_id is not None and self.turn_input_item_ids.get(retired_turn_id) == item_id:
            self.turn_input_item_ids.pop(retired_turn_id, None)
        if self.speculative_input_item_id == item_id:
            self.speculative_input_item_id = None
        if retired_chat_id is not None and self.speculative_user_item_id == retired_chat_id:
            self.speculative_user_item_id = None

    def record_protocol_item(self, item_id: str) -> None:
        """Record one protocol-visible conversation item in creation order."""
        newly_created = item_id not in self.protocol_item_ids
        if newly_created:
            sequence = self.deferred_protocol_item_sequences.pop(item_id, None)
            if sequence is None:
                self.next_protocol_item_sequence += 1
                sequence = self.next_protocol_item_sequence
            self.protocol_item_sequences[item_id] = sequence
            self.protocol_item_ids.append(item_id)
        while len(self.protocol_item_ids) > MAX_TRACKED_PROTOCOL_ITEMS:
            retired_item_id = self.protocol_item_ids.pop(0)
            self.protocol_item_sequences.pop(retired_item_id, None)
            self.audio_input_item_ids.discard(retired_item_id)
            retired_chat_id = self.input_item_chat_ids.pop(retired_item_id, retired_item_id)
            self.runtime_config.chat.retire_user_message_deletable(retired_chat_id)
            retired_turn_id = self.input_item_turn_ids.pop(retired_item_id, None)
            if retired_turn_id is not None and self.speculative_turn_tracker is not None:
                self.speculative_turn_tracker.discard(retired_turn_id)
            if retired_turn_id is not None and self.turn_input_item_ids.get(retired_turn_id) == retired_item_id:
                self.turn_input_item_ids.pop(retired_turn_id, None)
            self.deleted_input_item_ids.pop(retired_item_id, None)
        if newly_created:
            self.last_item_id = item_id

    def reserve_deferred_protocol_item(self, item_id: str) -> None:
        """Assign an immutable admission sequence before deferred chat commit."""

        self.next_protocol_item_sequence += 1
        self.deferred_protocol_item_sequences[item_id] = self.next_protocol_item_sequence

    def drop_deferred_protocol_item(self, item_id: str) -> None:
        """Release the sequence reservation for a deferred item that was rejected."""

        self.deferred_protocol_item_sequences.pop(item_id, None)

    def record_deleted_input_item(self, item_id: str) -> None:
        """Remember a deleted audio item without allowing per-session state growth."""
        self.deleted_input_item_ids[item_id] = None
        self.deleted_input_item_ids.move_to_end(item_id)
        while len(self.deleted_input_item_ids) > MAX_TRACKED_PROTOCOL_ITEMS:
            retired_item_id, _ = self.deleted_input_item_ids.popitem(last=False)
            self.audio_input_item_ids.discard(retired_item_id)
            retired_chat_id = self.input_item_chat_ids.pop(retired_item_id, None)
            if retired_chat_id is not None:
                self.runtime_config.chat.retire_user_message_deletable(retired_chat_id)
            retired_turn_id = self.input_item_turn_ids.pop(retired_item_id, None)
            if retired_turn_id is not None and self.turn_input_item_ids.get(retired_turn_id) == retired_item_id:
                self.turn_input_item_ids.pop(retired_turn_id, None)

    def remove_protocol_item(self, item_id: str) -> None:
        """Remove one protocol item and expose only the surviving protocol tail."""
        self.protocol_item_ids = [existing for existing in self.protocol_item_ids if existing != item_id]
        self.protocol_item_sequences.pop(item_id, None)
        self.last_item_id = self.protocol_item_ids[-1] if self.protocol_item_ids else None


class RealtimeService:
    """Translates between OpenAI Realtime protocol events and internal pipeline messages.

    One instance is shared across all WebSocket connections.  Per-connection
    state (response lifecycle, audio buffer) is tracked internally by
    connection id.
    """

    def __init__(
        self,
        text_prompt_queue: Queue[TextPromptItem] | None = None,
        should_listen: ThreadingEvent | None = None,
        chat_size: int = 10,
        speculative_turns: SpeculativeTurnTracker | None = None,
        cancel_scope: CancelScope | None = None,
        home_assistant_guard_supported: bool = False,
        home_assistant_guard_required: bool = False,
    ) -> None:
        self.text_prompt_queue = text_prompt_queue
        self.should_listen = should_listen
        self._chat_size = chat_size
        self.speculative_turns = speculative_turns
        self.cancel_scope = cancel_scope
        self.home_assistant_guard_supported = home_assistant_guard_supported
        self.home_assistant_guard_required = home_assistant_guard_required
        if home_assistant_guard_required and not home_assistant_guard_supported:
            raise ValueError("required Home Assistant guard needs the Chat Completions backend")
        self._cancel_scope_wiring_verified = False
        self._conns: dict[str, ConnState] = {}
        self.total_usage = GlobalUsageMetrics()

        self.audio = AudioHandler(self)
        self.session = SessionHandler(self)
        self.response = ResponseHandler(self)
        self.conversation = ConversationHandler(self)

        self._pipeline_dispatch: dict[type[PipelineEvent], Callable[..., list[ServerEvent]]] = {
            SpeechStartedEvent: self.audio.on_speech_started,
            SpeechStoppedEvent: self.audio.on_speech_stopped,
            TokenUsageEvent: self._on_token_usage,
            PartialTranscriptionEvent: self.conversation.on_partial_transcription,
            TranscriptionCompletedEvent: self._on_transcription_completed,
            TranscriptBarrierCompletedEvent: self._on_transcript_barrier_completed,
            TranscriptBarrierDiscardedEvent: self._on_transcript_barrier_discarded,
            ResponseFailedEvent: self._on_response_failed,
        }

    def verify_cancel_scope_wiring(self, *consumer_scopes: CancelScope | None) -> bool:
        """Latch whether every cancellation consumer shares the service scope."""
        scope = self.cancel_scope
        self._cancel_scope_wiring_verified = bool(
            scope is not None and consumer_scopes and all(consumer_scope is scope for consumer_scope in consumer_scopes)
        )
        return self._cancel_scope_wiring_verified

    @property
    def cancel_scope_wiring_verified(self) -> bool:
        return self._cancel_scope_wiring_verified

    # ── Connection lifecycle ─────────────────────

    def register(self) -> str:
        """Register a new connection and return its session_id."""
        if self.speculative_turns:
            self.speculative_turns.reset()
        state = ConnState(
            runtime_config=RuntimeConfig(chat=Chat(self._chat_size)),
            speculative_turn_tracker=self.speculative_turns,
        )
        self._conns[state.session_id] = state
        self.total_usage.connections += 1
        return state.session_id

    def unregister(self, conn_id: str) -> None:
        self.scrub_private_protocols_for_disconnect(conn_id)
        st = self._conns.pop(conn_id, None)
        if st is not None:
            # Suppress any in-flight compaction splice so a daemon worker can't
            # mutate a Chat tied to a closed session, and don't make further
            # billable LLM calls on its behalf once the splice is suppressed.
            st.runtime_config.chat.close()
            self.total_usage += st.response_usage
            logger.info(
                "Session %s unregistered — cumulative: input_tokens=%d, output_tokens=%d, audio=%.2fs",
                conn_id,
                self.total_usage.input_tokens,
                self.total_usage.output_tokens,
                self.total_usage.audio_duration_s,
            )

    def _state(self, conn_id: str) -> ConnState:
        return self._conns[conn_id]

    @property
    def connection_ids(self) -> list[str]:
        return list(self._conns)

    # ── Client event parsing ─────────────────────

    @staticmethod
    def _next_event_id() -> str:
        return _generate_id("event")

    def parse_client_event(
        self,
        raw: object,
        *,
        redact_private_content: bool = False,
    ) -> Optional[ClientEvent]:
        if not isinstance(raw, Mapping):
            logger.warning(
                "Private client event must be an object; content redacted"
                if redact_private_content
                else "Client event must be a JSON object"
            )
            return None
        raw_type = raw.get("type")
        event_type: Optional[str] = raw_type if isinstance(raw_type, str) else None
        if event_type is None:
            if redact_private_content:
                logger.warning("Private client event missing type; content redacted")
            else:
                logger.warning("Client event missing 'type' field")
            return None
        model_cls = _EVENT_TYPE_TO_MODEL.get(event_type)
        if model_cls is None:
            if redact_private_content:
                logger.warning("Unknown private client event type; content redacted")
            else:
                logger.warning("Unknown client event type: %s", event_type)
            return None
        try:
            return model_cls.model_validate(raw)  # type: ignore[return-value]
        except ValidationError as e:
            if event_type == "reachy.transcript_barrier.resolve":
                logger.error("Invalid private transcript barrier resolution payload")
            elif redact_private_content:
                logger.error("Invalid private client event payload; content redacted")
            else:
                logger.error("Invalid %s payload: %s", event_type, e)
            return None

    # ── Client event handlers ────────────────────

    def build_session_created(self, conn_id: str) -> SessionCreatedEvent:
        return self.session.build_session_created(conn_id)

    def handle_session_update(
        self,
        conn_id: str,
        event: SessionUpdateEvent,
    ) -> ServerEvent | list[ServerEvent] | None:
        return self.session.handle_session_update(conn_id, event)

    def transcript_barrier_enabled(self) -> bool:
        """Return the barrier state for this single-session pipeline unit."""
        if len(self._conns) != 1:
            return False
        return next(iter(self._conns.values())).runtime_config.transcript_barrier_enabled

    def transcript_barrier_private(self) -> bool:
        """Return the sticky privacy state for this single-session unit."""
        if len(self._conns) != 1:
            return False
        return next(iter(self._conns.values())).runtime_config.transcript_barrier_private

    def sensitive_content(self) -> bool:
        if len(self._conns) != 1:
            return False
        return next(iter(self._conns.values())).runtime_config.sensitive_content

    def home_assistant_guard_enabled(self) -> bool:
        if len(self._conns) != 1:
            return False
        return next(iter(self._conns.values())).runtime_config.home_assistant_guard_enabled

    def home_assistant_guard_failed(self, conn_id: str) -> bool:
        return self._state(conn_id).runtime_config.home_assistant_guard_failed

    def home_assistant_guard_pending(self, conn_id: str) -> bool:
        cfg = self._state(conn_id).runtime_config
        return self.home_assistant_guard_required and not (
            cfg.home_assistant_guard_enabled or cfg.home_assistant_guard_failed
        )

    def private_protocol_failed(self, conn_id: str) -> bool:
        return self._state(conn_id).runtime_config.private_protocol_failed

    def private_protocol_poisoned(self) -> bool:
        if len(self._conns) != 1:
            return False
        return next(iter(self._conns.values())).runtime_config.private_protocol_failed

    def transcript_barrier_failed(self, conn_id: str) -> bool:
        return self._state(conn_id).runtime_config.transcript_barrier_failed

    def transcript_barrier_poisoned(self) -> bool:
        """Backward-compatible name for the complete private failure gate."""
        return self.private_protocol_poisoned()

    @contextmanager
    def transcript_barrier_pipeline_state_guard(self) -> Iterator[tuple[bool, bool]]:
        """Linearize notifier content side effects with barrier poison."""
        if len(self._conns) != 1:
            # A pipeline transcription without its one live connection is
            # stale work, never an ordinary turn for a future session.
            yield True, True
            return
        cfg = next(iter(self._conns.values())).runtime_config
        with cfg.transcript_barrier_state_guard():
            yield cfg.transcript_barrier_private, cfg.private_protocol_failed

    @contextmanager
    def sensitive_pipeline_state_guard(self) -> Iterator[tuple[bool, bool]]:
        if len(self._conns) != 1:
            yield True, True
            return
        cfg = next(iter(self._conns.values())).runtime_config
        with cfg.transcript_barrier_state_guard():
            yield cfg.sensitive_content, cfg.private_protocol_failed

    def transcript_barrier_audio_allowed(self, conn_id: str) -> bool:
        cfg = self._state(conn_id).runtime_config
        return not cfg.private_protocol_failed and not cfg.transcript_barrier_pending

    def poison_home_assistant_guard(self, conn_id: str, error_type: str) -> RealtimeErrorEvent:
        st = self._state(conn_id)
        cfg = st.runtime_config
        completed = cfg.home_assistant_guard_enabled
        cfg.fail_home_assistant_guard()
        with cfg.transcript_barrier_state_guard():
            if not completed:
                cfg.chat.reset(private_content_logging=True, suspend_compaction=True)
            st.deferred_items.clear()
        return self.make_error("Home Assistant guard protocol violation.", error_type)

    def poison_transcript_barrier(self, conn_id: str, error_type: str) -> RealtimeErrorEvent:
        st = self._state(conn_id)
        cfg = st.runtime_config
        with cfg.transcript_barrier_state_guard():
            handshake_completed = cfg.transcript_barrier_enabled
            cfg.clear_transcript_barrier_pending()
            cfg.transcript_barrier_failed = True
            if not handshake_completed:
                cfg.chat.reset(
                    private_content_logging=True,
                    suspend_compaction=True,
                )
            else:
                cfg.chat.enable_private_content_logging()
                cfg.chat.suspend_compaction()
            st.deferred_items.clear()
        return self.make_error("Private transcript barrier protocol violation.", error_type)

    def scrub_transcript_barrier_for_disconnect(self, conn_id: str) -> None:
        """Synchronously scrub private state before asynchronous pipeline drain."""
        st = self._conns.get(conn_id)
        if st is None:
            return
        cfg = st.runtime_config
        if not cfg.transcript_barrier_private and not cfg.transcript_barrier_pending:
            return
        with cfg.transcript_barrier_state_guard():
            handshake_completed = cfg.transcript_barrier_enabled
            cfg.clear_transcript_barrier_pending()
            cfg.transcript_barrier_failed = True
            if not handshake_completed:
                cfg.chat.reset(
                    private_content_logging=True,
                    suspend_compaction=True,
                )
            else:
                cfg.chat.enable_private_content_logging()
                cfg.chat.suspend_compaction()
            st.deferred_items.clear()

    def scrub_private_protocols_for_disconnect(self, conn_id: str) -> None:
        self.scrub_transcript_barrier_for_disconnect(conn_id)
        st = self._conns.get(conn_id)
        if st is None or not st.runtime_config.home_assistant_guard_enabled:
            return
        st.runtime_config.fail_home_assistant_guard()
        st.deferred_items.clear()

    def handle_audio_append(self, conn_id: str, event: InputAudioBufferAppendEvent) -> list[bytes]:
        cfg = self._state(conn_id).runtime_config
        with cfg.transcript_barrier_state_guard():
            if cfg.private_protocol_failed:
                return []
            if not self.transcript_barrier_audio_allowed(conn_id):
                self.poison_transcript_barrier(conn_id, "transcript_barrier_pending")
                return []
            self._state(conn_id).audio_append_seen = True
            return self.audio.handle_audio_append(conn_id, event)

    def handle_audio_commit(self, conn_id: str) -> RealtimeErrorEvent | None:
        cfg = self._state(conn_id).runtime_config
        with cfg.transcript_barrier_state_guard():
            if cfg.private_protocol_failed:
                return self.make_error("Private session failed.", "private_session_failed")
            return self.audio.handle_audio_commit(conn_id)

    def encode_audio_chunk(self, conn_id: str, audio: bytes) -> list[ServerEvent]:
        return self.audio.encode_audio_chunk(conn_id, audio)

    def handle_response_create(self, conn_id: str, event: ResponseCreateEvent) -> ServerEvent | None:
        cfg = self._state(conn_id).runtime_config
        with cfg.transcript_barrier_state_guard():
            if cfg.private_protocol_failed:
                return self.make_error("Private session failed.", "private_session_failed")
            if cfg.transcript_barrier_pending:
                return self.poison_transcript_barrier(conn_id, "transcript_barrier_pending")
            if self.home_assistant_guard_pending(conn_id):
                return self.poison_home_assistant_guard(conn_id, "invalid_home_assistant_guard")
            response = event.response
            if cfg.home_assistant_guard_enabled and response is not None:
                if not valid_guarded_tool_choice(response.tool_choice):
                    return self.poison_home_assistant_guard(conn_id, "invalid_home_assistant_guard")
                explicit_tools = "tools" in response.model_fields_set
                tools_disabled = response.tool_choice == "none" and (not explicit_tools or response.tools == [])
                if is_out_of_band(response) and not tools_disabled:
                    return self.poison_home_assistant_guard(conn_id, "invalid_home_assistant_guard")
                if not tools_disabled:
                    effective_instructions = response.instructions or cfg.session.instructions
                    effective_tools = response.tools if explicit_tools else cfg.session.tools
                    try:
                        digest, tool_count, tool_names = session_contract(
                            effective_instructions,
                            effective_tools,
                        )
                    except (TypeError, ValueError):
                        return self.poison_home_assistant_guard(conn_id, "invalid_home_assistant_guard")
                    if (
                        digest != cfg.home_assistant_guard_contract_sha256
                        or tool_count != cfg.home_assistant_guard_tool_count
                        or tool_names != cfg.home_assistant_guard_tool_names
                    ):
                        return self.poison_home_assistant_guard(conn_id, "invalid_home_assistant_guard")
            return self.response.handle_response_create(conn_id, event)

    def handle_response_cancel(self, conn_id: str) -> list[ServerEvent]:
        return self.response.handle_response_cancel(conn_id)

    def finish_response(
        self,
        conn_id: str,
        status: _ResponseStatus = "completed",
        reason: _StatusReason | None = None,
    ) -> list[ServerEvent]:
        return self.response.finish_response(conn_id, status, reason)

    def handle_conversation_item_create(self, conn_id: str, event: ConversationItemCreateEvent) -> list[ServerEvent]:
        cfg = self._state(conn_id).runtime_config
        with cfg.transcript_barrier_state_guard():
            if cfg.private_protocol_failed:
                return []
            if cfg.transcript_barrier_pending:
                return [self.poison_transcript_barrier(conn_id, "transcript_barrier_pending")]
            return self.conversation.handle_conversation_item_create(conn_id, event)

    def handle_conversation_item_delete(
        self,
        conn_id: str,
        event: ConversationItemDeleteEvent,
        *,
        defer_successor_enqueue: bool = False,
    ) -> list[ServerEvent]:
        """Delete one exact user item while preserving the guarded session fence."""
        cfg = self._state(conn_id).runtime_config
        with cfg.transcript_barrier_state_guard():
            if cfg.private_protocol_failed:
                return []
            if cfg.transcript_barrier_pending:
                error = self.poison_transcript_barrier(conn_id, "transcript_barrier_pending")
                error.error.event_id = event.event_id
                return [error]
            return self.conversation.handle_conversation_item_delete(
                conn_id,
                event,
                defer_successor_enqueue=defer_successor_enqueue,
            )

    def enqueue_pending_response(self, conn_id: str) -> None:
        """Release a response held across router-owned cancelled-output flushing."""
        self.response.enqueue_pending_request(conn_id)

    def handle_transcript_barrier_resolve(
        self,
        conn_id: str,
        event: TranscriptBarrierResolveEvent,
    ) -> list[ServerEvent]:
        """Resolve exactly one pending final without exposing it to ordinary sinks."""
        cfg = self._state(conn_id).runtime_config
        expected_transcript = cfg.transcript_barrier_pending_transcript
        valid_context = (
            cfg.transcript_barrier_operational
            and cfg.transcript_barrier_pending
            and event.nonce == cfg.transcript_barrier_nonce
            and event.sequence == cfg.transcript_barrier_pending_sequence
            and event.input_item_id == cfg.transcript_barrier_pending_item_id
        )
        if not valid_context or expected_transcript is None:
            return [self.poison_transcript_barrier(conn_id, "invalid_transcript_barrier_resolution")]

        item_id: str | None = None
        created: list[ServerEvent] = []
        if event.action == "accept":
            item = event.item
            expected_item = (
                {
                    "content": [{"text": expected_transcript, "type": "input_text"}],
                    "id": item.id,
                    "role": "user",
                    "type": "message",
                }
                if item is not None
                else None
            )
            valid_item = (
                item is not None
                and len(item.content) == 1
                and item.content[0].type == "input_text"
                and item.content[0].text == expected_transcript
                and item.model_dump(exclude_none=True) == expected_item
            )
            if not valid_item or item is None:
                return [self.poison_transcript_barrier(conn_id, "invalid_transcript_barrier_resolution")]
            item_id = item.id
            assert item_id is not None
            st = self._state(conn_id)
            existing_ids = {
                existing_id
                for existing_id in (
                    st.last_item_id,
                    st.current_item_id,
                    st.current_output_item_id,
                    *(getattr(entry, "id", None) for entry in cfg.chat.buffer),
                    *(getattr(entry, "id", None) for entry in st.deferred_items),
                )
                if existing_id is not None
            }
            if item_id in existing_ids or item_id in st.transcript_barrier_replacement_item_ids:
                return [self.poison_transcript_barrier(conn_id, "invalid_transcript_barrier_resolution")]
            cfg.clear_transcript_barrier_pending()
            cfg.chat.resume_compaction()
            if not st.in_response:
                created.extend(self.conversation.flush_deferred_items(conn_id))
            item_created = self.conversation._apply_item(conn_id, item)
            created.extend(item_created)
            if not item_created or item_created[0].type == "error":
                self.poison_transcript_barrier(conn_id, "invalid_transcript_barrier_resolution")
                return created
            st.transcript_barrier_replacement_item_ids.add(item_id)
            action: Literal["accepted", "discarded"] = "accepted"
        else:
            cfg.clear_transcript_barrier_pending()
            cfg.chat.resume_compaction()
            st = self._state(conn_id)
            if not st.in_response:
                created.extend(self.conversation.flush_deferred_items(conn_id))
            action = "discarded"

        return [
            *created,
            TranscriptBarrierResolvedServerEvent(
                event_id=self._next_event_id(),
                nonce=event.nonce,
                sequence=event.sequence,
                input_item_id=event.input_item_id,
                replacement_item_id=item_id,
                action=action,
            ),
        ]

    def dispatch_pipeline_event(self, conn_id: str, event: PipelineEvent) -> list[ServerEvent]:
        """Route a pipeline text_output_queue event to the appropriate handler."""
        events = self._dispatch_pipeline_event(conn_id, event, wait_for_pending_reopen=True)
        return [] if events is None else events

    def try_dispatch_pipeline_event(self, conn_id: str, event: PipelineEvent) -> list[ServerEvent] | None:
        """Non-blocking dispatch.

        Returns ``None`` when dispatch must be retried after a speculative
        reopen candidate resolves.
        """
        return self._dispatch_pipeline_event(conn_id, event, wait_for_pending_reopen=False)

    def should_defer_pipeline_event(self, event: PipelineEvent) -> bool:
        if self.speculative_turns is None or not isinstance(event, (AssistantTextEvent, TokenUsageEvent)):
            return False
        return self.speculative_turns.has_pending_reopen_or_grace(
            getattr(event, "turn_id", None),
            getattr(event, "turn_revision", None),
        )

    def _dispatch_pipeline_event(
        self,
        conn_id: str,
        event: PipelineEvent,
        *,
        wait_for_pending_reopen: bool,
    ) -> list[ServerEvent] | None:
        cfg = self._state(conn_id).runtime_config
        while True:
            # The speculative tracker may wait for its reopen grace. Never do
            # that while holding the private-failure lock: provider rejection
            # must be able to poison the session immediately.
            is_stale = self._is_stale_turn_event(event, wait_for_pending_reopen=wait_for_pending_reopen)
            if is_stale is None:
                return None
            if is_stale:
                logger.info(
                    "Ignoring stale %s for turn=%s rev=%s",
                    event.type,
                    getattr(event, "turn_id", None),
                    getattr(event, "turn_revision", None),
                )
                return []

            with cfg.transcript_barrier_state_guard():
                if cfg.private_protocol_failed:
                    logger.debug("Dropping pipeline event after private barrier failure")
                    return []
                if cfg.transcript_barrier_enabled and isinstance(
                    event,
                    (PartialTranscriptionEvent, TranscriptionCompletedEvent),
                ):
                    logger.info("Rejecting ordinary transcription event after private barrier activation")
                    return [self.poison_transcript_barrier(conn_id, "invalid_transcript_barrier_event")]
                if isinstance(event, AssistantTextEvent):
                    events = self.response.on_assistant_text(
                        conn_id,
                        event,
                        wait_for_pending_reopen=False,
                    )
                    if events is None:
                        if not wait_for_pending_reopen:
                            return None
                        # A reopen began after the outside-lock check. Retry so
                        # its bounded wait occurs only after releasing this lock.
                        continue
                    self._observe_turn_event(event)
                    return events
                self._observe_turn_event(event)
                handler = self._pipeline_dispatch.get(type(event))
                if handler is None:
                    logger.debug("Unhandled pipeline event type: %s", type(event).__name__)
                    return []
                return handler(conn_id, event)

    def _is_stale_turn_event(self, event: PipelineEvent, *, wait_for_pending_reopen: bool = True) -> bool | None:
        if self.speculative_turns is None:
            return False
        if not isinstance(
            event,
            (
                PartialTranscriptionEvent,
                TranscriptionCompletedEvent,
                TranscriptBarrierCompletedEvent,
                TranscriptBarrierDiscardedEvent,
                SpeechStartedEvent,
                SpeechStoppedEvent,
                AssistantTextEvent,
                TokenUsageEvent,
                ResponseFailedEvent,
            ),
        ):
            return False
        turn_id = getattr(event, "turn_id", None)
        turn_revision = getattr(event, "turn_revision", None)
        if isinstance(event, (AssistantTextEvent, TokenUsageEvent)):
            is_latest: bool | None
            if wait_for_pending_reopen:
                is_latest = self.speculative_turns.is_latest_after_reopen_grace(turn_id, turn_revision)
            else:
                is_latest = self.speculative_turns.try_is_latest_after_reopen_grace(turn_id, turn_revision)
            if is_latest is None:
                return None
            return not is_latest
        return not self.speculative_turns.is_latest(turn_id, turn_revision)

    def _observe_turn_event(self, event: PipelineEvent) -> None:
        if self.speculative_turns is None:
            return
        self.speculative_turns.observe(
            getattr(event, "turn_id", None),
            getattr(event, "turn_revision", None),
        )

    # ── STT → LM bridge ────────────────────────────

    def _on_transcription_completed(self, conn_id: str, event: TranscriptionCompletedEvent) -> list[ServerEvent]:
        """Emit and store a final transcription, then trigger LM when configured."""
        st = self._state(conn_id)
        input_item_id = (
            st.turn_input_item_ids.get(event.turn_id) if event.turn_id is not None else st.speculative_input_item_id
        )
        if input_item_id is None:
            input_item_id = self.response._current_item_id(conn_id)
            st.speculative_input_item_id = input_item_id
            st.audio_input_item_ids.add(input_item_id)
            st.record_protocol_item(input_item_id)
            if event.turn_id is not None:
                st.input_item_turn_ids[input_item_id] = event.turn_id
                st.turn_input_item_ids[event.turn_id] = input_item_id
        if input_item_id in st.deleted_input_item_ids:
            logger.debug("Ignoring transcription for a deleted input item")
            return []
        same_speculative_turn = event.turn_id is not None and event.turn_id == st.speculative_user_turn_id
        if same_speculative_turn:
            st.response_usage.audio_duration_s -= st.speculative_audio_duration_s
        else:
            st.speculative_audio_duration_s = 0.0

        events = self.conversation.on_transcription_completed(conn_id, event, item_id=input_item_id)
        if event.turn_id is not None:
            st.speculative_audio_duration_s = st.input_audio_duration_s

        cfg = st.runtime_config
        transcript = event.transcript
        if transcript:
            if same_speculative_turn and st.speculative_user_item_id:
                replaced = cfg.chat.replace_user_message_text(st.speculative_user_item_id, transcript)
                if not replaced:
                    item = cfg.chat.add_item(make_user_message(transcript))
                    st.speculative_user_item_id = item.id
            else:
                item = cfg.chat.add_item(make_user_message(transcript))
                st.speculative_user_item_id = item.id
            if st.speculative_user_item_id is not None:
                st.input_item_chat_ids[input_item_id] = st.speculative_user_item_id
                cfg.chat.mark_user_message_deletable(st.speculative_user_item_id)
        elif same_speculative_turn and st.speculative_user_item_id:
            cfg.chat.remove_user_message(st.speculative_user_item_id)
            st.input_item_chat_ids.pop(input_item_id, None)
            st.speculative_user_item_id = None
            if input_item_id in st.response_context_input_item_ids:
                st.response_context_input_item_id = None
                st.response_context_input_item_ids.clear()
                st.response_context_turn_id = None
                st.response_context_turn_revision = None
                st.response_context_speech_stopped_at_s = None
            if event.turn_id is not None:
                events.extend(self.response.discard_turn(conn_id, event.turn_id))
                if not st.in_response and not st.response_pending and self.should_listen is not None:
                    self.should_listen.set()
        elif event.turn_id is not None and event.turn_id != st.speculative_user_turn_id:
            st.speculative_user_item_id = None

        if event.turn_id is not None:
            st.speculative_user_turn_id = event.turn_id
            st.speculative_user_turn_revision = event.turn_revision
            st.speculative_user_speech_stopped_at_s = event.speech_stopped_at_s
        if transcript:
            st.response_context_input_item_id = input_item_id
            st.response_context_input_item_ids = {input_item_id}
            st.response_context_turn_id = event.turn_id
            st.response_context_turn_revision = event.turn_revision
            st.response_context_speech_stopped_at_s = event.speech_stopped_at_s

        queue = self.text_prompt_queue
        if queue and transcript and cfg.create_response_enabled:
            request = GenerateResponseRequest(
                runtime_config=cfg,
                chat_snapshot=cfg.chat.copy(),
                response_user_item_id=st.speculative_user_item_id,
                response_user_item_ids=(
                    {st.speculative_user_item_id} if st.speculative_user_item_id is not None else set()
                ),
                admitted_protocol_item_ids={
                    *st.protocol_item_ids,
                    *(item.id for item in st.deferred_items if item.id is not None),
                },
                admitted_protocol_sequence=st.next_protocol_item_sequence,
                language_code=event.language_code,
                turn_id=event.turn_id,
                turn_revision=event.turn_revision,
                speech_stopped_at_s=event.speech_stopped_at_s,
                cancel_generation=(self.cancel_scope.generation if self.cancel_scope else None),
            )
            # One protocol response is active at a time.  Hold a later turn at
            # this boundary until the active response closes so its model output
            # cannot be folded into the prior response or cancelled with it.
            if st.in_response and event.turn_id == st.active_response_turn_id:
                # A speculative revision continues the same protocol response;
                # the turn tracker makes the older queued revision stale.
                st.active_response_turn_revision = event.turn_revision
                queue.put(request)
            elif st.in_response and st.response_pending and event.turn_id == st.pending_response_turn_id:
                st.pending_response_turn_revision = event.turn_revision
                st.pending_response_request = request
                st.pending_response_enqueued = False
            elif st.response_pending and event.turn_id == st.pending_response_turn_id:
                st.pending_response_turn_revision = event.turn_revision
                st.pending_response_request = request
                queue.put(request)
                st.pending_response_enqueued = True
            else:
                deferred_index = next(
                    (
                        index
                        for index, candidate in enumerate(st.deferred_response_requests)
                        if candidate.turn_id == event.turn_id
                    ),
                    None,
                )
                if deferred_index is not None:
                    # Coalesce a speculative revision without changing the
                    # distinct turn's place in the FIFO.
                    st.deferred_response_requests[deferred_index] = request
                elif st.in_response and not st.response_pending:
                    st.response_pending = True
                    st.pending_response_turn_id = event.turn_id
                    st.pending_response_turn_revision = event.turn_revision
                    st.pending_response_request = request
                    st.pending_response_enqueued = False
                elif st.response_pending:
                    if len(st.deferred_response_requests) >= MAX_DEFERRED_RESPONSE_REQUESTS:
                        events.append(
                            self.make_error(
                                "Too many responses are waiting to run.",
                                "response_queue_full",
                            )
                        )
                    else:
                        st.deferred_response_requests.append(request)
                else:
                    st.response_pending = True
                    st.pending_response_turn_id = event.turn_id
                    st.pending_response_turn_revision = event.turn_revision
                    st.pending_response_request = request
                    cfg.chat.protect_response_turn(request.response_user_item_id)
                    queue.put(request)
                    st.pending_response_enqueued = True

        return events

    def _transcript_barrier_context(self, conn_id: str) -> tuple[RuntimeConfig, str, int, str]:
        st = self._state(conn_id)
        cfg = st.runtime_config
        nonce = cfg.transcript_barrier_nonce
        if not cfg.transcript_barrier_operational or nonce is None:
            raise RuntimeError("private transcript barrier event arrived without an active handshake")
        sequence = cfg.next_transcript_barrier_sequence()
        item_id = self.conversation._input_item_id(conn_id)
        st.response_usage.audio_duration_s += st.input_audio_duration_s
        return cfg, nonce, sequence, item_id

    def _on_transcript_barrier_completed(
        self,
        conn_id: str,
        event: TranscriptBarrierCompletedEvent,
    ) -> list[ServerEvent]:
        """Emit one private final without storing it or triggering generation."""
        st = self._state(conn_id)
        if st.runtime_config.transcript_barrier_failed:
            return []
        cfg, nonce, sequence, item_id = self._transcript_barrier_context(conn_id)
        if cfg.transcript_barrier_pending or st.in_response or st.response_pending or st.deferred_items:
            self.poison_transcript_barrier(conn_id, "transcript_barrier_pending")
            return [
                TranscriptBarrierFailedServerEvent(
                    event_id=self._next_event_id(),
                    nonce=nonce,
                    sequence=sequence,
                    item_id=item_id,
                    reason="overlapping_transcript",
                )
            ]
        if not event.transcript.strip() or len(event.transcript) > TRANSCRIPT_BARRIER_MAX_CHARS:
            self.poison_transcript_barrier(conn_id, "invalid_transcript_barrier_transcript")
            return [
                TranscriptBarrierFailedServerEvent(
                    event_id=self._next_event_id(),
                    nonce=nonce,
                    sequence=sequence,
                    item_id=item_id,
                    reason="transcript_too_large",
                )
            ]
        cfg.chat.suspend_compaction()
        cfg.transcript_barrier_pending_sequence = sequence
        cfg.transcript_barrier_pending_item_id = item_id
        cfg.transcript_barrier_pending_transcript = event.transcript
        return [
            TranscriptBarrierCompletedServerEvent(
                event_id=self._next_event_id(),
                nonce=nonce,
                sequence=sequence,
                item_id=item_id,
                transcript=event.transcript,
                language_code=event.language_code,
            )
        ]

    def _on_transcript_barrier_discarded(
        self,
        conn_id: str,
        event: TranscriptBarrierDiscardedEvent,
    ) -> list[ServerEvent]:
        """Emit a content-free empty-final acknowledgement."""
        st = self._state(conn_id)
        if st.runtime_config.transcript_barrier_failed:
            return []
        cfg, nonce, sequence, item_id = self._transcript_barrier_context(conn_id)
        if cfg.transcript_barrier_pending or st.in_response or st.response_pending or st.deferred_items:
            self.poison_transcript_barrier(conn_id, "transcript_barrier_pending")
            return [
                TranscriptBarrierFailedServerEvent(
                    event_id=self._next_event_id(),
                    nonce=nonce,
                    sequence=sequence,
                    item_id=item_id,
                    reason="overlapping_transcript",
                )
            ]
        return [
            TranscriptBarrierDiscardedServerEvent(
                event_id=self._next_event_id(),
                nonce=nonce,
                sequence=sequence,
                item_id=item_id,
            )
        ]

    # ── Metrics ────────────────────────────────────

    def _on_token_usage(self, conn_id: str, event: TokenUsageEvent) -> list[ServerEvent]:
        """Accumulate input/output token counts on the connection's usage metrics."""
        if self.speculative_turns and not self.speculative_turns.is_latest(
            event.turn_id,
            event.turn_revision,
        ):
            logger.debug("Dropping stale token usage for turn=%s rev=%s", event.turn_id, event.turn_revision)
            return []
        st = self._state(conn_id)
        if st.in_response:
            owner_turn_id = st.active_response_turn_id
            owner_turn_revision = st.active_response_turn_revision
            owner_generation = st.active_response_cancel_generation
        elif st.response_pending and st.pending_response_request is not None:
            owner_turn_id = st.pending_response_turn_id
            owner_turn_revision = st.pending_response_turn_revision
            owner_generation = st.pending_response_request.cancel_generation
        else:
            owner_turn_id = None
            owner_turn_revision = None
            owner_generation = None
        if event.cancel_generation is not None:
            if (st.in_response or st.response_pending) and owner_generation != event.cancel_generation:
                logger.debug(
                    "Dropping token usage for stale cancellation generation=%s (active=%s)",
                    event.cancel_generation,
                    owner_generation,
                )
                return []
        if event.turn_id is not None and (st.in_response or st.response_pending):
            if event.turn_id != owner_turn_id or (
                event.turn_revision is not None
                and owner_turn_revision is not None
                and event.turn_revision != owner_turn_revision
            ):
                logger.debug(
                    "Dropping token usage for non-active turn=%s rev=%s",
                    event.turn_id,
                    event.turn_revision,
                )
                return []
        st.response_usage.input_tokens += event.input_tokens
        st.response_usage.output_tokens += event.output_tokens
        logger.info(
            "Token usage (response): input=%d, output=%d",
            st.response_usage.input_tokens,
            st.response_usage.output_tokens,
        )
        return []

    def _on_response_failed(self, conn_id: str, event: ResponseFailedEvent) -> list[ServerEvent]:
        """Surface the failure to the client and close the response as ``failed``.

        Emitted when generation failed (e.g. invalid out-of-band input, or the
        provider rejecting an empty context). A top-level ``error`` event carries
        the human-readable reason — ``response.done.status_details.error`` only
        has code/type, no message. The active slot stays owned until the matching
        EndOfResponse reaches the audio queue; closing it here would promote a
        successor that the failed response's later terminal sentinel could erase.
        """
        state = self._state(conn_id)
        private_barrier = state.runtime_config.transcript_barrier_private
        if private_barrier:
            message = "Private response failed."
            logger.info("Private response failed; content redacted")
        else:
            message = event.message
            logger.info("Response failed: %s", message)
        if (not state.in_response and not state.response_pending) or state.response_failure_pending:
            return []
        owner_turn_id = state.active_response_turn_id if state.in_response else state.pending_response_turn_id
        owner_turn_revision = (
            state.active_response_turn_revision if state.in_response else state.pending_response_turn_revision
        )
        owner_generation = (
            state.active_response_cancel_generation
            if state.in_response
            else (
                state.pending_response_request.cancel_generation if state.pending_response_request is not None else None
            )
        )
        if event.cancel_generation is not None and event.cancel_generation != owner_generation:
            logger.debug(
                "Ignoring response failure for non-active generation=%s (owner=%s)",
                event.cancel_generation,
                owner_generation,
            )
            return []
        if event.cancel_generation is None and self.cancel_scope is not None:
            logger.debug("Ignoring uncorrelated response failure while generation tracking is active")
            return []
        if event.turn_id is not None:
            if event.turn_id != owner_turn_id or (
                event.turn_revision is not None
                and owner_turn_revision is not None
                and event.turn_revision != owner_turn_revision
            ):
                logger.debug(
                    "Ignoring response failure for non-active turn=%s rev=%s",
                    event.turn_id,
                    event.turn_revision,
                )
                return []
        created = not state.in_response
        if created:
            self.response._ensure_response(conn_id)
        state.response_failure_pending = True
        events: list[ServerEvent] = []
        if created:
            events.append(
                ResponseCreatedEvent(
                    type="response.created",
                    event_id=self.response._next_event_id(),
                    response=self.response._build_response(conn_id, "in_progress"),
                )
            )
        events.append(self.make_error(message, "response_failed"))
        return events

    def get_usage(self) -> dict[str, Any]:
        """Return cumulative usage metrics across all completed responses."""
        data = self.total_usage.model_dump()
        data["total_tokens"] = data["input_tokens"] + data["output_tokens"]
        data["total_errors"] = self.total_usage.total_errors
        return data

    # ── Error ───────────────────────────────────

    def make_error(self, message: str, _type: str) -> RealtimeErrorEvent:
        self.total_usage.record_error(_type)
        return build_error_event(message, _type)


def build_error_event(message: str, error_type: str) -> RealtimeErrorEvent:
    """Construct a RealtimeErrorEvent without touching any service-instance state.

    Used by the websocket route handler on pool rejection, where no unit's
    service should be charged with the error in its usage metrics.
    """
    return RealtimeErrorEvent(
        type="error",
        error=RealtimeError(message=message, type=error_type),
        event_id=_generate_id("event"),
    )
