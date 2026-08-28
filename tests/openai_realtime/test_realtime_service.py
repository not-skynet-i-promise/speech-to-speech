"""Unit tests for api.openai_realtime.service.RealtimeService.

Every public method is exercised and the emitted OpenAI Realtime events are
validated for correct type, attributes, and state transitions.
"""

import base64
import json
import logging
from queue import Queue
from threading import Event, Thread
from time import sleep

import pytest
from openai.types.realtime import (
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
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
from openai.types.realtime.realtime_session_create_request import RealtimeSessionCreateRequest

from speech_to_speech.api.openai_realtime.home_assistant_guard import (
    HomeAssistantGuardReadyEvent,
    session_contract,
)
from speech_to_speech.api.openai_realtime.service import (
    CHUNK_SIZE_BYTES,
    RealtimeService,
)
from speech_to_speech.api.openai_realtime.transcript_barrier import (
    TRANSCRIPT_BARRIER_MAX_CHARS,
    TranscriptBarrierCompletedServerEvent,
    TranscriptBarrierDiscardedServerEvent,
    TranscriptBarrierFailedServerEvent,
    TranscriptBarrierReadyEvent,
    TranscriptBarrierResolvedServerEvent,
    TranscriptBarrierResolveEvent,
)
from speech_to_speech.LLM.chat import make_assistant_message
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import (
    AssistantTextEvent,
    PartialTranscriptionEvent,
    ResponseFailedEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TokenUsageEvent,
    TranscriptBarrierCompletedEvent,
    TranscriptBarrierDiscardedEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.messages import AssistantTextPart, AssistantToolCallPart, GenerateResponseRequest
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pcm_bytes(n_samples: int) -> bytes:
    """Return n_samples * 2 zero bytes (valid PCM16 silence)."""
    return b"\x00" * (n_samples * 2)


def _home_assistant_tools() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "name": "home_assistant__GetLiveContext",
            "description": "Read exposed state.",
            "parameters": {"type": "object", "properties": {"area": {"type": "string"}}},
        },
        {
            "type": "function",
            "name": "get_local_time",
            "parameters": {"type": "object", "properties": {}},
        },
    ]


def _activate_home_assistant_guard(service: RealtimeService, conn_id: str) -> HomeAssistantGuardReadyEvent:
    service.home_assistant_guard_supported = True
    tools = _home_assistant_tools()
    contract = RealtimeSessionCreateRequest(type="realtime", instructions="Use exposed tools.", tools=tools)
    digest, tool_count, _names = session_contract(contract.instructions, contract.tools)
    result = service.handle_session_update(
        conn_id,
        SessionUpdateEvent(
            type="session.update",
            session={
                "type": "realtime",
                "instructions": contract.instructions,
                "tools": tools,
                "reachy_home_assistant_guard": {
                    "version": 1,
                    "nonce": "19" * 32,
                    "session_contract_sha256": digest,
                    "tool_count": tool_count,
                },
            },
        ),
    )
    assert isinstance(result, HomeAssistantGuardReadyEvent)
    return result


def _b64_pcm(n_samples: int) -> str:
    return base64.b64encode(_pcm_bytes(n_samples)).decode("ascii")


def _make_audio_append(audio_b64: str) -> InputAudioBufferAppendEvent:
    return InputAudioBufferAppendEvent(type="input_audio_buffer.append", audio=audio_b64)


# ===================================================================
# Connection lifecycle
# ===================================================================


class TestConnectionLifecycle:
    def test_register_creates_session_id(self, service):
        sid = service.register()
        assert sid.startswith("session_")
        st = service._state(sid)
        assert st.conversation_id.startswith("conv_")
        assert st.in_response is False
        assert st.last_item_id is None
        service.unregister(sid)

    def test_unregister_removes_state(self, service):
        sid = service.register()
        service.unregister(sid)
        with pytest.raises(KeyError):
            service._state(sid)

    def test_build_session_created(self, service, conn_id, runtime_config):
        service.handle_session_update(
            conn_id,
            SessionUpdateEvent(
                type="session.update",
                session={
                    "type": "realtime",
                    "instructions": "Be helpful",
                    "tools": [{"type": "function", "name": "get_weather"}],
                    "tool_choice": "auto",
                    "audio": {
                        "input": {"turn_detection": {"type": "server_vad"}},
                        "output": {"voice": "echo"},
                    },
                },
            ),
        )

        evt = service.build_session_created(conn_id)
        assert isinstance(evt, SessionCreatedEvent)
        assert evt.event_id.startswith("event_")
        assert evt.session is not None
        assert evt.session.instructions == "Be helpful"
        assert evt.session.tools is not None
        assert evt.session.tool_choice == "auto"
        assert evt.session.audio.output.voice == "echo"
        assert evt.session.audio.input.turn_detection.type == "server_vad"


# ===================================================================
# Client event parsing
# ===================================================================


class TestParseClientEvent:
    def test_parse_valid_audio_append(self, service):
        raw = {"type": "input_audio_buffer.append", "audio": "AAAA"}
        evt = service.parse_client_event(raw)
        assert isinstance(evt, InputAudioBufferAppendEvent)

    def test_parse_valid_session_update(self, service):
        raw = {"type": "session.update", "session": {"type": "realtime"}, "voice": "alloy"}
        evt = service.parse_client_event(raw)
        assert isinstance(evt, SessionUpdateEvent)
        assert evt.voice == "alloy"

    def test_parse_valid_conversation_item_create(self, service):
        raw = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            },
        }
        evt = service.parse_client_event(raw)
        assert isinstance(evt, ConversationItemCreateEvent)

    def test_parse_valid_conversation_item_delete(self, service):
        evt = service.parse_client_event(
            {"type": "conversation.item.delete", "event_id": "event_delete", "item_id": "item_audio"}
        )
        assert isinstance(evt, ConversationItemDeleteEvent)

    def test_parse_valid_response_create(self, service):
        raw = {"type": "response.create"}
        evt = service.parse_client_event(raw)
        assert isinstance(evt, ResponseCreateEvent)

    def test_parse_valid_response_cancel(self, service):
        raw = {"type": "response.cancel"}
        evt = service.parse_client_event(raw)
        assert isinstance(evt, ResponseCancelEvent)

    def test_parse_unknown_event_type(self, service):
        assert service.parse_client_event({"type": "bogus.event"}) is None

    def test_parse_invalid_payload(self, service):
        raw = {"type": "input_audio_buffer.append"}  # missing required 'audio'
        assert service.parse_client_event(raw) is None

    def test_parse_invalid_private_resolution_never_logs_payload(self, service, caplog):
        canary = "PRIVATE_RESOLUTION_CANARY"
        raw = {
            "type": "reachy.transcript_barrier.resolve",
            "version": 1,
            "nonce": "ab" * 32,
            "sequence": 1,
            "input_item_id": "item_1",
            "action": "accept",
            "item": {
                "id": "invalid-id",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": canary}],
            },
        }

        with caplog.at_level(logging.ERROR):
            assert service.parse_client_event(raw) is None

        assert canary not in caplog.text
        assert "Invalid private transcript barrier resolution payload" in caplog.text

    def test_private_mode_redacts_all_invalid_client_event_content(self, service, caplog):
        canary = "PII_JOSH_PRIVATE_CLIENT_EVENT"
        raw = {
            "type": "conversation.item.create",
            "item": {
                "id": canary,
                "type": "message",
                "role": "invalid",
                "content": [{"type": "input_text", "text": canary}],
            },
        }

        with caplog.at_level(logging.ERROR):
            assert service.parse_client_event(raw, redact_private_content=True) is None

        assert canary not in caplog.text
        assert "Invalid private client event payload; content redacted" in caplog.text

    @pytest.mark.parametrize(
        "mutation",
        [
            lambda raw: raw.pop("version"),
            lambda raw: raw.update(version=True),
            lambda raw: raw.update(sequence=True),
            lambda raw: raw.update(item=None),
        ],
    )
    def test_private_discard_resolution_rejects_coercions_and_extra_nulls(self, service, mutation):
        raw = {
            "type": "reachy.transcript_barrier.resolve",
            "version": 1,
            "nonce": "ab" * 32,
            "sequence": 1,
            "input_item_id": "item_1",
            "action": "discard",
        }
        mutation(raw)

        assert service.parse_client_event(raw) is None

    @pytest.mark.parametrize(
        "path",
        ["status", "object", "content_audio", "content_transcript"],
    )
    def test_private_accept_resolution_rejects_optional_null_fields(self, service, path):
        raw = {
            "type": "reachy.transcript_barrier.resolve",
            "version": 1,
            "nonce": "ab" * 32,
            "sequence": 1,
            "input_item_id": "item_1",
            "action": "accept",
            "item": {
                "id": "msg_private_1",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "private"}],
            },
        }
        if path in {"status", "object"}:
            raw["item"][path] = None
        else:
            raw["item"]["content"][0][path.removeprefix("content_")] = None

        assert service.parse_client_event(raw) is None


# ===================================================================
# Audio append
# ===================================================================


class TestHandleAudioAppend:
    def test_audio_append_decodes_and_chunks(self, service, conn_id):
        audio_b64 = _b64_pcm(512 * 3)  # exactly 3 chunks
        evt = _make_audio_append(audio_b64)
        chunks = service.handle_audio_append(conn_id, evt)
        assert len(chunks) == 3
        assert all(len(c) == CHUNK_SIZE_BYTES for c in chunks)
        assert service._state(conn_id).audio_buffer_has_data is True

    def test_audio_append_invalid_base64(self, service, conn_id):
        evt = InputAudioBufferAppendEvent(type="input_audio_buffer.append", audio="!!!invalid!!!")
        chunks = service.handle_audio_append(conn_id, evt)
        assert chunks == []

    def test_audio_append_undersized_tail(self, service, conn_id):
        audio_b64 = _b64_pcm(512 + 100)  # 1 full chunk + 100 samples remainder
        evt = _make_audio_append(audio_b64)
        chunks = service.handle_audio_append(conn_id, evt)
        assert len(chunks) == 1


# ===================================================================
# Session update
# ===================================================================


class TestHandleSessionUpdate:
    def _make_update(self, **session_fields) -> SessionUpdateEvent:
        session_fields.setdefault("type", "realtime")
        return SessionUpdateEvent(type="session.update", session=session_fields)  # type: ignore[arg-type]

    def test_session_update_voice(self, service, conn_id, runtime_config):
        evt = self._make_update(
            audio={"output": {"voice": "shimmer"}},
        )
        service.handle_session_update(conn_id, evt)
        assert runtime_config.session.audio.output.voice == "shimmer"

    def test_session_update_instructions(self, service, conn_id, runtime_config):
        service.handle_session_update(conn_id, self._make_update(instructions="Be concise"))
        assert runtime_config.session.instructions == "Be concise"

    def test_private_transcript_barrier_requires_one_exact_handshake(
        self,
        service,
        conn_id,
        runtime_config,
    ):
        nonce = "ab" * 32
        result = service.handle_session_update(
            conn_id,
            self._make_update(
                reachy_private_transcript_barrier={"version": 1, "nonce": nonce},
            ),
        )

        assert isinstance(result, TranscriptBarrierReadyEvent)
        assert result.nonce == nonce
        assert runtime_config.transcript_barrier_enabled is True
        assert "reachy_private_transcript_barrier" not in (runtime_config.session.model_extra or {})
        assert runtime_config.chat._private_content_logging is True

        duplicate = service.handle_session_update(
            conn_id,
            self._make_update(
                reachy_private_transcript_barrier={"version": 1, "nonce": nonce},
            ),
        )
        assert isinstance(duplicate, RealtimeErrorEvent)
        assert duplicate.error.type == "invalid_transcript_barrier"
        assert runtime_config.transcript_barrier_failed is True
        assert runtime_config.transcript_barrier_enabled is True
        assert runtime_config.transcript_barrier_operational is False

    def test_private_transcript_barrier_requires_shared_cancel_scope(self, runtime_config):
        service = RealtimeService(text_prompt_queue=Queue())
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config
        try:
            result = service.handle_session_update(
                conn_id,
                self._make_update(
                    reachy_private_transcript_barrier={"version": 1, "nonce": "ef" * 32},
                ),
            )

            assert isinstance(result, RealtimeErrorEvent)
            assert result.error.type == "invalid_transcript_barrier"
            assert runtime_config.transcript_barrier_failed is True
            assert runtime_config.transcript_barrier_enabled is False
        finally:
            service.unregister(conn_id)

    def test_private_transcript_barrier_rejects_miswired_cancel_scopes(self, runtime_config):
        service_scope = CancelScope()
        service = RealtimeService(text_prompt_queue=Queue(), cancel_scope=service_scope)
        assert not service.verify_cancel_scope_wiring(service_scope, CancelScope())
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config
        try:
            result = service.handle_session_update(
                conn_id,
                self._make_update(
                    reachy_private_transcript_barrier={"version": 1, "nonce": "fe" * 32},
                ),
            )

            assert isinstance(result, RealtimeErrorEvent)
            assert result.error.type == "invalid_transcript_barrier"
            assert runtime_config.transcript_barrier_failed is True
            assert runtime_config.transcript_barrier_enabled is False
        finally:
            service.unregister(conn_id)

    def test_private_transcript_barrier_rejects_active_response_admission(
        self,
        service,
        conn_id,
        runtime_config,
        cancel_scope,
    ):
        with cancel_scope.response_admission(cancel_scope.generation) as (admitted, generation):
            assert admitted is True
            assert generation == cancel_scope.generation
            result = service.handle_session_update(
                conn_id,
                self._make_update(
                    reachy_private_transcript_barrier={"version": 1, "nonce": "fa" * 32},
                ),
            )

        assert isinstance(result, RealtimeErrorEvent)
        assert result.error.type == "invalid_transcript_barrier"
        assert runtime_config.transcript_barrier_failed is True
        assert runtime_config.transcript_barrier_enabled is False

    def test_barrier_ready_waits_for_cancelled_provider_content_guard(
        self,
        service,
        conn_id,
        runtime_config,
        text_prompt_queue,
    ):
        created = service.handle_response_create(
            conn_id,
            ResponseCreateEvent(type="response.create", response={"conversation": "none"}),
        )
        assert isinstance(created, ResponseCreatedEvent)
        request = text_prompt_queue.get(timeout=1.0)
        assert isinstance(request, GenerateResponseRequest)

        provider_entered = Event()
        release_provider = Event()
        provider_finished = Event()
        ordering: list[str] = []

        def cancelled_provider_log() -> None:
            with request.runtime_config.transcript_barrier_content_guard() as private_content:
                assert private_content is False
                provider_entered.set()
                assert release_provider.wait(timeout=1.0)
                ordering.append("ordinary_provider_log")
            provider_finished.set()

        provider_thread = Thread(target=cancelled_provider_log)
        provider_thread.start()
        assert provider_entered.wait(timeout=1.0)

        cancelled = service.handle_response_cancel(conn_id)
        assert any(isinstance(event, ResponseDoneEvent) for event in cancelled)
        state = service._state(conn_id)
        assert state.in_response is False
        assert state.response_pending is False
        assert state.current_item_id is None
        assert state.last_item_id is None
        assert runtime_config.chat.buffer == []

        activation_returned = Event()
        activation_result: list[object] = []

        def activate_barrier() -> None:
            activation_result.append(
                service.handle_session_update(
                    conn_id,
                    self._make_update(
                        reachy_private_transcript_barrier={"version": 1, "nonce": "cd" * 32},
                    ),
                )
            )
            ordering.append("ready")
            activation_returned.set()

        activation_thread = Thread(target=activate_barrier)
        activation_thread.start()
        assert not activation_returned.wait(timeout=0.05)
        assert runtime_config.transcript_barrier_enabled is False

        release_provider.set()
        provider_thread.join(timeout=1.0)
        activation_thread.join(timeout=1.0)

        assert not provider_thread.is_alive()
        assert not activation_thread.is_alive()
        assert provider_finished.is_set()
        assert activation_returned.is_set()
        assert ordering == ["ordinary_provider_log", "ready"]
        assert len(activation_result) == 1
        assert isinstance(activation_result[0], TranscriptBarrierReadyEvent)
        assert runtime_config.transcript_barrier_private is True

    def test_private_transcript_barrier_must_be_in_first_session_update(
        self,
        service,
        conn_id,
        runtime_config,
    ):
        service.handle_session_update(conn_id, self._make_update(instructions="ordinary session"))

        result = service.handle_session_update(
            conn_id,
            self._make_update(
                reachy_private_transcript_barrier={"version": 1, "nonce": "bc" * 32},
            ),
        )

        assert isinstance(result, RealtimeErrorEvent)
        assert result.error.type == "invalid_transcript_barrier"
        assert runtime_config.transcript_barrier_enabled is False
        assert runtime_config.transcript_barrier_failed is True

    def test_private_transcript_barrier_rejects_audio_before_handshake(
        self,
        service,
        conn_id,
        runtime_config,
        text_prompt_queue,
        caplog,
    ):
        assert service.handle_audio_append(conn_id, _make_audio_append(_b64_pcm(512)))
        assert service.handle_audio_commit(conn_id) is None
        assert service._state(conn_id).audio_buffer_has_data is False
        assert service._state(conn_id).audio_append_seen is True

        result = service.handle_session_update(
            conn_id,
            self._make_update(
                reachy_private_transcript_barrier={"version": 1, "nonce": "de" * 32},
            ),
        )

        assert isinstance(result, RealtimeErrorEvent)
        assert result.error.type == "invalid_transcript_barrier"
        assert runtime_config.transcript_barrier_enabled is False
        assert runtime_config.transcript_barrier_failed is True
        assert runtime_config.transcript_barrier_private is True
        assert runtime_config.chat._private_content_logging is True

        transcript = "REJECTED_ACTIVATION_DRAIN_CONTENT"
        with caplog.at_level(logging.DEBUG):
            events = service.dispatch_pipeline_event(
                conn_id,
                TranscriptBarrierCompletedEvent(transcript=transcript),
            )
            ordinary_events = service.dispatch_pipeline_event(
                conn_id,
                TranscriptionCompletedEvent(transcript=transcript),
            )

        assert events == []
        assert ordinary_events == []
        assert text_prompt_queue.empty()
        assert runtime_config.chat.buffer == []
        assert transcript not in caplog.text

    def test_private_transcript_barrier_poison_drops_ordinary_final_after_ready(
        self,
        service,
        conn_id,
        runtime_config,
        text_prompt_queue,
        caplog,
    ):
        ready = service.handle_session_update(
            conn_id,
            self._make_update(
                reachy_private_transcript_barrier={"version": 1, "nonce": "db" * 32},
            ),
        )
        assert isinstance(ready, TranscriptBarrierReadyEvent)

        transcript = "LATE_ORDINARY_FINAL_AFTER_READY"
        with caplog.at_level(logging.DEBUG):
            events = service.dispatch_pipeline_event(
                conn_id,
                TranscriptionCompletedEvent(transcript=transcript),
            )

        assert events is not None
        assert len(events) == 1
        assert isinstance(events[0], RealtimeErrorEvent)
        assert events[0].error.type == "invalid_transcript_barrier_event"
        assert runtime_config.transcript_barrier_failed is True
        assert runtime_config.transcript_barrier_pending_transcript is None
        assert runtime_config.chat.buffer == []
        assert text_prompt_queue.empty()
        assert transcript not in caplog.text

    @pytest.mark.parametrize(
        "barrier_request",
        [
            None,
            {},
            {"version": True, "nonce": "ab" * 32},
            {"version": 2, "nonce": "ab" * 32},
            {"version": 1, "nonce": "AB" * 32},
            {"version": 1, "nonce": "ab" * 31},
            {"version": 1, "nonce": "ab" * 32, "extra": "forbidden"},
        ],
    )
    def test_private_transcript_barrier_rejects_malformed_activation(
        self,
        service,
        conn_id,
        runtime_config,
        barrier_request,
    ):
        result = service.handle_session_update(
            conn_id,
            self._make_update(reachy_private_transcript_barrier=barrier_request),
        )

        assert isinstance(result, RealtimeErrorEvent)
        assert result.error.type == "invalid_transcript_barrier"
        assert runtime_config.transcript_barrier_enabled is False
        assert runtime_config.transcript_barrier_failed is True

    def test_session_update_tools_and_tool_choice(self, service, conn_id, runtime_config):
        tools = [{"type": "function", "name": "f1"}]
        service.handle_session_update(conn_id, self._make_update(tools=tools, tool_choice="required"))
        assert runtime_config.session.tools is not None
        assert runtime_config.session.tool_choice == "required"

    def test_session_update_rejects_transcription_session(self, service, conn_id, runtime_config):
        raw = {
            "type": "session.update",
            "session": {"type": "transcription"},
        }
        evt = SessionUpdateEvent.model_validate(raw)
        err = service.handle_session_update(conn_id, evt)
        assert isinstance(err, RealtimeErrorEvent)
        assert err.error.type == "invalid_session_type"
        assert runtime_config.transcript_barrier_session_updates == 1

        barrier = service.handle_session_update(
            conn_id,
            self._make_update(
                reachy_private_transcript_barrier={"version": 1, "nonce": "aa" * 32},
            ),
        )
        assert isinstance(barrier, RealtimeErrorEvent)
        assert barrier.error.type == "invalid_transcript_barrier"
        assert runtime_config.transcript_barrier_failed is True

    def test_session_update_nested_audio_format(self, service, conn_id, runtime_config):
        raw = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "audio": {
                    "input": {"turn_detection": {"type": "server_vad", "threshold": 0.5}},
                    "output": {"voice": "nova"},
                },
            },
        }
        evt = SessionUpdateEvent.model_validate(raw)
        service.handle_session_update(conn_id, evt)
        assert runtime_config.session.audio.output.voice == "nova"
        assert runtime_config.session.audio.input.turn_detection.type == "server_vad"

    def test_session_update_merges_partial_updates(self, service, conn_id, runtime_config):
        """Partial updates preserve previously-set fields."""
        service.handle_session_update(
            conn_id,
            self._make_update(
                audio={"output": {"voice": "echo"}},
                instructions="Be helpful",
            ),
        )
        assert runtime_config.session.audio.output.voice == "echo"
        assert runtime_config.session.instructions == "Be helpful"

        service.handle_session_update(conn_id, self._make_update(instructions="Be concise"))
        assert runtime_config.session.instructions == "Be concise"
        assert runtime_config.session.audio.output.voice == "echo"  # preserved from first update


# ===================================================================
# Conversation item create
# ===================================================================


class TestHandleConversationItemCreate:
    def _text_event(self, text: str = "hello", item_id: str = "msg_abc") -> ConversationItemCreateEvent:
        return ConversationItemCreateEvent(
            type="conversation.item.create",
            item={  # type: ignore[arg-type]
                "id": item_id,
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        )

    def test_text_input_emits_conversation_item_created(
        self,
        service,
        conn_id,
        text_prompt_queue,
    ):
        events = service.handle_conversation_item_create(conn_id, self._text_event("hi"))
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, ConversationItemCreatedEvent)
        assert evt.previous_item_id is None  # first item
        assert evt.item.role == "user"
        assert evt.item.content[0].type == "input_text"
        assert evt.item.content[0].text == "hi"
        last = service._state(conn_id).runtime_config.chat.buffer[-1]
        assert last.role == "user"
        assert last.content[0].type == "input_text"
        assert last.content[0].text == "hi"

    def test_text_input_previous_item_id_chain(self, service, conn_id):
        e1 = service.handle_conversation_item_create(conn_id, self._text_event("a", "msg_1"))
        e2 = service.handle_conversation_item_create(conn_id, self._text_event("b", "msg_2"))
        assert e1[0].previous_item_id is None
        assert e2[0].previous_item_id == e1[0].item.id

    def test_function_call_output_forwarded(self, service, conn_id, text_prompt_queue):
        from openai.types.realtime.realtime_conversation_item_function_call import (
            RealtimeConversationItemFunctionCall,
        )

        service._state(conn_id).runtime_config.chat.add_item(
            RealtimeConversationItemFunctionCall(
                type="function_call", call_id="call_1", name="get_weather", arguments="{}"
            )
        )
        evt = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "function_call_output", "output": '{"result": 42}', "call_id": "call_1"},
        )
        events = service.handle_conversation_item_create(conn_id, evt)
        assert len(events) == 1
        assert isinstance(events[0], ConversationItemCreatedEvent)
        last = service._state(conn_id).runtime_config.chat.buffer[-1]
        assert last.type == "function_call_output"
        assert last.call_id == "call_1"
        assert last.output == '{"result": 42}'

    def test_function_call_output_rejected_for_unknown_call_id(self, service, conn_id, text_prompt_queue):
        evt = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "function_call_output", "output": '{"result": 42}', "call_id": "call_unknown"},
        )
        events = service.handle_conversation_item_create(conn_id, evt)
        assert len(events) == 1
        assert isinstance(events[0], RealtimeErrorEvent)
        assert "call_unknown" in events[0].error.message
        assert not any(
            getattr(e, "type", None) == "function_call_output"
            for e in service._state(conn_id).runtime_config.chat.buffer
        )

    def test_input_image_forwarded(self, service, conn_id, text_prompt_queue):
        evt = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": "https://example.com/img.png"}],
            },
        )
        events = service.handle_conversation_item_create(conn_id, evt)
        assert len(events) == 1
        assert isinstance(events[0], ConversationItemCreatedEvent)
        last = service._state(conn_id).runtime_config.chat.buffer[-1]
        assert last.role == "user"
        assert last.content[0].type == "input_image"
        assert last.content[0].image_url == "https://example.com/img.png"

    def test_mixed_text_and_image_forwarded(self, service, conn_id, text_prompt_queue):
        evt = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is this?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,abc123"},
                ],
            },
        )
        events = service.handle_conversation_item_create(conn_id, evt)
        assert len(events) == 1
        last = service._state(conn_id).runtime_config.chat.buffer[-1]
        assert last.role == "user"
        assert len(last.content) == 2
        assert last.content[0].type == "input_text"
        assert last.content[0].text == "What is this?"
        assert last.content[1].type == "input_image"
        assert last.content[1].image_url == "data:image/png;base64,abc123"


class TestHandleConversationItemDelete:
    def test_duplicate_live_item_id_is_rejected_without_mutating_chat(self, service, conn_id):
        event = ConversationItemCreateEvent(
            type="conversation.item.create",
            event_id="create_duplicate",
            item={
                "id": "msg_duplicate",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "one"}],
            },
        )
        service.handle_conversation_item_create(conn_id, event)

        duplicate = service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                event_id="create_duplicate_again",
                item={
                    "id": "msg_duplicate",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "two"}],
                },
            ),
        )

        assert isinstance(duplicate[0], RealtimeErrorEvent)
        assert duplicate[0].error.type == "duplicate_item_id"
        assert duplicate[0].error.event_id == "create_duplicate_again"
        assert [item.content[0].text for item in service._state(conn_id).runtime_config.chat.buffer] == ["one"]

    def test_duplicate_deferred_item_id_is_rejected_before_flush(self, service, conn_id):
        st = service._state(conn_id)
        st.in_response = True
        first = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "id": "msg_deferred_duplicate",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "one"}],
            },
        )
        assert service.handle_conversation_item_create(conn_id, first) == []

        duplicate = service.handle_conversation_item_create(conn_id, first)

        assert isinstance(duplicate[0], RealtimeErrorEvent)
        assert len(st.deferred_items) == 1
        assert st.runtime_config.chat.buffer == []

    def test_deletes_exact_created_user_item(self, service, conn_id):
        created = service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={
                    "id": "msg_echo",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "echo"}],
                },
            ),
        )
        assert isinstance(created[0], ConversationItemCreatedEvent)

        events = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(
                type="conversation.item.delete",
                event_id="event_delete",
                item_id="msg_echo",
            ),
        )

        assert len(events) == 1
        assert isinstance(events[0], ConversationItemDeletedEvent)
        assert events[0].item_id == "msg_echo"
        assert service._state(conn_id).runtime_config.chat.buffer == []

    def test_audio_item_id_deletes_its_exact_stored_transcript(self, service, conn_id):
        st = service._state(conn_id)
        stored = st.runtime_config.chat.add_item(
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "assistant echo"}],
                },
            ).item
        )
        assert stored.id is not None
        st.speculative_input_item_id = "item_audio"
        st.speculative_user_item_id = stored.id
        st.audio_input_item_ids.add("item_audio")
        st.input_item_chat_ids["item_audio"] = stored.id
        st.record_protocol_item("item_audio")

        events = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(
                type="conversation.item.delete",
                event_id="event_delete",
                item_id="item_audio",
            ),
        )

        assert isinstance(events[0], ConversationItemDeletedEvent)
        assert events[0].item_id == "item_audio"
        assert st.runtime_config.chat.buffer == []
        assert st.speculative_user_item_id is None
        assert st.last_item_id is None

    def test_delete_cancels_the_exact_queued_response_owner(self, service, conn_id, cancel_scope):
        started = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_pending", turn_revision=0),
        )[0]
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="assistant echo", turn_id="turn_pending", turn_revision=0),
        )
        st = service._state(conn_id)
        assert st.response_pending
        assert st.pending_response_turn_id == "turn_pending"

        events = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id=started.item_id),
        )

        assert isinstance(events[0], ConversationItemDeletedEvent)
        assert not st.response_pending
        assert st.pending_response_turn_id is None
        assert cancel_scope.generation == 1

    def test_delete_cancels_and_terminalizes_the_exact_active_response(self):
        tracker = SpeculativeTurnTracker()
        cancel_scope = CancelScope()
        service = RealtimeService(
            text_prompt_queue=Queue(),
            should_listen=Event(),
            speculative_turns=tracker,
            cancel_scope=cancel_scope,
        )
        conn_id = service.register()
        started = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_active", turn_revision=0),
        )[0]
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="assistant echo", turn_id="turn_active", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(text="stale reply", turn_id="turn_active", turn_revision=0),
        )
        st = service._state(conn_id)
        assert st.in_response
        assert st.active_response_turn_id == "turn_active"

        events = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id=started.item_id),
        )

        assert isinstance(events[0], ConversationItemDeletedEvent)
        assert any(isinstance(event, ResponseDoneEvent) and event.response.status == "cancelled" for event in events)
        assert not st.in_response
        assert not st.response_pending
        assert st.active_response_turn_id is None
        assert cancel_scope.generation == 1
        service.unregister(conn_id)

    def test_active_delete_flushes_deferred_item_after_deleted_protocol_tail(self):
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(
            text_prompt_queue=Queue(),
            should_listen=Event(),
            speculative_turns=tracker,
            cancel_scope=CancelScope(),
        )
        conn_id = service.register()
        started = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_active", turn_revision=0),
        )[0]
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="assistant echo", turn_id="turn_active", turn_revision=0),
        )
        service.encode_audio_chunk(conn_id, _pcm_bytes(256))
        assert (
            service.handle_conversation_item_create(
                conn_id,
                ConversationItemCreateEvent(
                    type="conversation.item.create",
                    item={
                        "id": "msg_deferred",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "next turn"}],
                    },
                ),
            )
            == []
        )

        events = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id=started.item_id),
        )

        created = next(event for event in events if isinstance(event, ConversationItemCreatedEvent))
        assert created.item.id == "msg_deferred"
        assert created.previous_item_id is None
        assert service._state(conn_id).protocol_item_ids == ["msg_deferred"]
        assert service._state(conn_id).last_item_id == "msg_deferred"
        service.unregister(conn_id)

    def test_active_delete_requeues_an_unrelated_pending_turn(self):
        tracker = SpeculativeTurnTracker()
        cancel_scope = CancelScope()
        prompt_queue = Queue()
        service = RealtimeService(
            text_prompt_queue=prompt_queue,
            should_listen=Event(),
            speculative_turns=tracker,
            cancel_scope=cancel_scope,
        )
        conn_id = service.register()
        active = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_active", turn_revision=0),
        )[0]
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="first", turn_id="turn_active", turn_revision=0),
        )
        active_request = prompt_queue.get_nowait()
        assert active_request.turn_id == "turn_active"
        service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(text="first reply", turn_id="turn_active", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_next", turn_revision=0, interrupt_response=False),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="second", turn_id="turn_next", turn_revision=0),
        )
        st = service._state(conn_id)
        assert prompt_queue.empty()
        assert st.pending_response_turn_id == "turn_next"

        service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id=active.item_id),
        )

        resumed = prompt_queue.get_nowait()
        assert resumed.turn_id == "turn_next"
        assert resumed.cancel_generation == cancel_scope.generation == 1
        assert st.response_pending
        assert st.pending_response_turn_id == "turn_next"
        assert st.pending_response_request is resumed
        service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(text="second reply", turn_id="turn_next", turn_revision=0),
        )
        assert st.in_response
        assert st.active_response_turn_id == "turn_next"
        assert not st.response_pending
        assert st.pending_response_request is None
        service.unregister(conn_id)

    def test_three_distinct_turns_are_preserved_in_fifo_order(self):
        tracker = SpeculativeTurnTracker()
        prompt_queue = Queue()
        service = RealtimeService(
            text_prompt_queue=prompt_queue,
            speculative_turns=tracker,
            cancel_scope=CancelScope(),
        )
        conn_id = service.register()

        for index in range(3):
            turn_id = f"turn_{index}"
            service.dispatch_pipeline_event(
                conn_id,
                SpeechStartedEvent(turn_id=turn_id, turn_revision=0, interrupt_response=False),
            )
            service.dispatch_pipeline_event(
                conn_id,
                TranscriptionCompletedEvent(
                    transcript=f"question {index}",
                    turn_id=turn_id,
                    turn_revision=0,
                ),
            )
            if index == 0:
                active_request = prompt_queue.get_nowait()
                assert active_request.turn_id == "turn_0"
                assert active_request.response_user_item_id is not None
                assert [item.content[0].text for item in active_request.chat_snapshot.buffer] == ["question 0"]
                service.dispatch_pipeline_event(
                    conn_id,
                    AssistantTextEvent(text="reply 0", turn_id="turn_0", turn_revision=0),
                )
                service._state(conn_id).runtime_config.chat.add_response_item(
                    make_assistant_message("reply 0"),
                    after_user_id=active_request.response_user_item_id,
                )

        st = service._state(conn_id)
        assert st.active_response_turn_id == "turn_0"
        assert st.pending_response_turn_id == "turn_1"
        assert [request.turn_id for request in st.deferred_response_requests] == ["turn_2"]
        assert [(item.role, item.content[0].text) for item in st.pending_response_request.chat_snapshot.buffer] == [
            ("user", "question 0"),
            ("assistant", "reply 0"),
            ("user", "question 1"),
        ]
        assert [
            (item.role, item.content[0].text) for item in st.deferred_response_requests[0].chat_snapshot.buffer
        ] == [
            ("user", "question 0"),
            ("assistant", "reply 0"),
            ("user", "question 1"),
            ("user", "question 2"),
        ]
        assert prompt_queue.empty()

        service.finish_response(conn_id)
        second_request = prompt_queue.get_nowait()
        assert second_request.turn_id == "turn_1"
        assert [(item.role, item.content[0].text) for item in second_request.chat_snapshot.buffer] == [
            ("user", "question 0"),
            ("assistant", "reply 0"),
            ("user", "question 1"),
        ]
        service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(text="reply 1", turn_id="turn_1", turn_revision=0),
        )
        st.runtime_config.chat.add_response_item(
            make_assistant_message("reply 1"),
            after_user_id=second_request.response_user_item_id,
        )
        assert st.active_response_turn_id == "turn_1"
        assert st.pending_response_turn_id == "turn_2"
        assert st.deferred_response_requests == []

        service.finish_response(conn_id)
        third_request = prompt_queue.get_nowait()
        assert third_request.turn_id == "turn_2"
        assert [(item.role, item.content[0].text) for item in third_request.chat_snapshot.buffer] == [
            ("user", "question 0"),
            ("assistant", "reply 0"),
            ("user", "question 1"),
            ("assistant", "reply 1"),
            ("user", "question 2"),
        ]
        service.unregister(conn_id)

    def test_deferred_item_flush_cannot_retire_accepted_fifo_turns(self, monkeypatch):
        import speech_to_speech.api.openai_realtime.service as service_module

        monkeypatch.setattr(service_module, "MAX_TRACKED_PROTOCOL_ITEMS", 3)
        tracker = SpeculativeTurnTracker()
        prompt_queue = Queue()
        service = RealtimeService(
            text_prompt_queue=prompt_queue,
            speculative_turns=tracker,
            cancel_scope=CancelScope(),
        )
        conn_id = service.register()
        created_items = {}
        for turn_id in ("turn_active", "turn_pending", "turn_deferred"):
            created_items[turn_id] = service.dispatch_pipeline_event(
                conn_id,
                SpeechStartedEvent(turn_id=turn_id, turn_revision=0, interrupt_response=False),
            )[0]
            service.dispatch_pipeline_event(
                conn_id,
                TranscriptionCompletedEvent(transcript=turn_id, turn_id=turn_id, turn_revision=0),
            )
            if turn_id == "turn_active":
                prompt_queue.get_nowait()
                service.dispatch_pipeline_event(
                    conn_id,
                    AssistantTextEvent(text="active reply", turn_id=turn_id, turn_revision=0),
                )

        for index in range(3):
            assert (
                service.handle_conversation_item_create(
                    conn_id,
                    ConversationItemCreateEvent(
                        type="conversation.item.create",
                        item={
                            "id": f"msg_deferred_{index}",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": str(index)}],
                        },
                    ),
                )
                == []
            )

        service.finish_response(conn_id)

        state = service._state(conn_id)
        promoted = prompt_queue.get_nowait()
        assert promoted.turn_id == "turn_pending"
        assert state.pending_response_turn_id == "turn_pending"
        assert [request.turn_id for request in state.deferred_response_requests] == ["turn_deferred"]
        for turn_id in ("turn_pending", "turn_deferred"):
            input_item_id = created_items[turn_id].item_id
            assert input_item_id in state.protocol_item_ids
            assert state.turn_input_item_ids[turn_id] == input_item_id
            assert tracker.is_latest(turn_id, 0)
        assert len(state.protocol_item_ids) == 3
        service.unregister(conn_id)

    def test_deleting_held_turn_preserves_fifo_before_a_new_arrival(self):
        tracker = SpeculativeTurnTracker()
        prompt_queue = Queue()
        service = RealtimeService(
            text_prompt_queue=prompt_queue,
            speculative_turns=tracker,
            cancel_scope=CancelScope(),
        )
        conn_id = service.register()
        created_items = {}
        for turn_id in ("turn_a", "turn_b", "turn_c"):
            created_items[turn_id] = service.dispatch_pipeline_event(
                conn_id,
                SpeechStartedEvent(turn_id=turn_id, turn_revision=0, interrupt_response=False),
            )[0]
            service.dispatch_pipeline_event(
                conn_id,
                TranscriptionCompletedEvent(transcript=turn_id, turn_id=turn_id, turn_revision=0),
            )
            if turn_id == "turn_a":
                prompt_queue.get_nowait()
                service.dispatch_pipeline_event(
                    conn_id,
                    AssistantTextEvent(text="reply a", turn_id=turn_id, turn_revision=0),
                )

        service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(
                type="conversation.item.delete",
                item_id=created_items["turn_b"].item_id,
            ),
        )
        state = service._state(conn_id)
        assert state.pending_response_turn_id == "turn_c"
        assert state.deferred_response_requests == []

        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_d", turn_revision=0, interrupt_response=False),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="turn_d", turn_id="turn_d", turn_revision=0),
        )
        assert state.pending_response_turn_id == "turn_c"
        assert [request.turn_id for request in state.deferred_response_requests] == ["turn_d"]

        service.finish_response(conn_id)
        assert prompt_queue.get_nowait().turn_id == "turn_c"
        service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(text="reply c", turn_id="turn_c", turn_revision=0),
        )
        service.finish_response(conn_id)
        assert prompt_queue.get_nowait().turn_id == "turn_d"
        service.unregister(conn_id)

    def test_deleting_a_deep_queued_turn_releases_capacity_immediately(self):
        tracker = SpeculativeTurnTracker()
        prompt_queue = Queue()
        service = RealtimeService(
            text_prompt_queue=prompt_queue,
            speculative_turns=tracker,
            cancel_scope=CancelScope(),
        )
        conn_id = service.register()
        created_items = {}
        for turn_id in ("turn_a", "turn_b", "turn_c"):
            created_items[turn_id] = service.dispatch_pipeline_event(
                conn_id,
                SpeechStartedEvent(turn_id=turn_id, turn_revision=0, interrupt_response=False),
            )[0]
            service.dispatch_pipeline_event(
                conn_id,
                TranscriptionCompletedEvent(transcript=turn_id, turn_id=turn_id, turn_revision=0),
            )
            if turn_id == "turn_a":
                prompt_queue.get_nowait()
                service.dispatch_pipeline_event(
                    conn_id,
                    AssistantTextEvent(text="reply a", turn_id=turn_id, turn_revision=0),
                )

        service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(
                type="conversation.item.delete",
                item_id=created_items["turn_c"].item_id,
            ),
        )

        state = service._state(conn_id)
        assert state.pending_response_turn_id == "turn_b"
        assert state.deferred_response_requests == []
        service.unregister(conn_id)

    def test_promoted_turn_excludes_client_context_that_arrived_after_admission(self):
        prompt_queue = Queue()
        service = RealtimeService(
            text_prompt_queue=prompt_queue,
            speculative_turns=SpeculativeTurnTracker(),
            cancel_scope=CancelScope(),
        )
        conn_id = service.register()
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_active", turn_revision=0, interrupt_response=False),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="first", turn_id="turn_active", turn_revision=0),
        )
        prompt_queue.get_nowait()
        service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(text="first reply", turn_id="turn_active", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_next", turn_revision=0, interrupt_response=False),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="second", turn_id="turn_next", turn_revision=0),
        )
        service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={
                    "id": "msg_future_context",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "future client context"}],
                },
            ),
        )
        service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={
                    "id": "sys_future_context",
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "future system context"}],
                },
            ),
        )
        state = service._state(conn_id)
        state.runtime_config.chat.add_response_item(
            make_assistant_message("committed first reply"),
            after_user_id=state.input_item_chat_ids[state.turn_input_item_ids["turn_active"]],
        )

        service.finish_response(conn_id)

        promoted = prompt_queue.get_nowait()
        texts = [
            part.text
            for item in promoted.chat_snapshot.buffer
            for part in getattr(item, "content", [])
            if getattr(part, "text", None)
        ]
        assert texts == ["first", "committed first reply", "second"]
        assert promoted.chat_snapshot.init_chat_message is None
        assert state.runtime_config.chat.init_chat_message is not None
        assert state.runtime_config.chat.init_chat_message.content[0].text == "future system context"
        assert any(
            getattr(part, "text", None) == "future client context"
            for item in state.runtime_config.chat.buffer
            for part in getattr(item, "content", [])
        )
        service.unregister(conn_id)

    def test_promoted_turn_refreshes_prior_context_without_later_queued_user(self):
        prompt_queue = Queue()
        service = RealtimeService(
            text_prompt_queue=prompt_queue,
            speculative_turns=SpeculativeTurnTracker(),
            cancel_scope=CancelScope(),
        )
        conn_id = service.register()
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_active", turn_revision=0, interrupt_response=False),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="first", turn_id="turn_active", turn_revision=0),
        )
        prompt_queue.get_nowait()
        service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(text="first reply", turn_id="turn_active", turn_revision=0),
        )
        service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={
                    "id": "msg_deferred_context",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "accepted context"}],
                },
            ),
        )
        for turn_id, transcript in (("turn_next", "second"), ("turn_later", "third")):
            service.dispatch_pipeline_event(
                conn_id,
                SpeechStartedEvent(turn_id=turn_id, turn_revision=0, interrupt_response=False),
            )
            service.dispatch_pipeline_event(
                conn_id,
                TranscriptionCompletedEvent(transcript=transcript, turn_id=turn_id, turn_revision=0),
            )
        state = service._state(conn_id)
        state.runtime_config.chat.add_item(make_assistant_message("committed first reply"))

        service.finish_response(conn_id)

        promoted = prompt_queue.get_nowait()
        assert promoted.turn_id == "turn_next"
        assert [(item.role, item.content[0].text) for item in promoted.chat_snapshot.buffer] == [
            ("user", "first"),
            ("assistant", "committed first reply"),
            ("user", "accepted context"),
            ("user", "second"),
        ]
        service.unregister(conn_id)

    def test_promoted_turn_restores_exact_user_after_completed_compaction(self):
        from speech_to_speech.LLM.chat import Chat, CompactionResult

        prompt_queue = Queue()
        service = RealtimeService(
            text_prompt_queue=prompt_queue,
            speculative_turns=SpeculativeTurnTracker(),
            cancel_scope=CancelScope(),
        )
        conn_id = service.register()
        state = service._state(conn_id)
        state.runtime_config.chat = Chat(size=2)

        for turn_id, transcript in (
            ("turn_active", "first exact"),
            ("turn_target", "second exact"),
            ("turn_later", "third exact"),
        ):
            service.dispatch_pipeline_event(
                conn_id,
                SpeechStartedEvent(turn_id=turn_id, turn_revision=0, interrupt_response=False),
            )
            service.dispatch_pipeline_event(
                conn_id,
                TranscriptionCompletedEvent(transcript=transcript, turn_id=turn_id, turn_revision=0),
            )
            if turn_id == "turn_active":
                active_request = prompt_queue.get_nowait()
                service.dispatch_pipeline_event(
                    conn_id,
                    AssistantTextEvent(text="first reply", turn_id=turn_id, turn_revision=0),
                )
                state.runtime_config.chat.add_response_item(
                    make_assistant_message("first reply"),
                    after_user_id=active_request.response_user_item_id,
                )

        target_chat_id = state.input_item_chat_ids[state.turn_input_item_ids["turn_target"]]
        # Isolate the completed-compaction restoration branch. Production keeps
        # this owner leased while active; that hard-eviction invariant has its
        # own regression test.
        state.runtime_config.chat.release_response_turn(active_request.response_user_item_id, force=True)
        state.runtime_config.chat.trim_if_needed(
            lambda _snapshot: CompactionResult(user_summary="lossy summary", assistant_summary="summary reply")
        )
        assert state.runtime_config.chat._compact_thread is not None
        state.runtime_config.chat._compact_thread.join(timeout=2.0)
        assert state.runtime_config.chat.user_message(target_chat_id) is None

        service.finish_response(conn_id)

        promoted = prompt_queue.get_nowait()
        assert promoted.turn_id == "turn_target"
        assert promoted.response_user_item_id == target_chat_id
        snapshot_texts = [
            part.text
            for item in promoted.chat_snapshot.buffer
            for part in getattr(item, "content", [])
            if getattr(part, "text", None)
        ]
        assert "second exact" in snapshot_texts
        assert "third exact" not in snapshot_texts
        assert state.runtime_config.chat.user_message(target_chat_id) is not None
        reply = state.runtime_config.chat.add_response_item(
            make_assistant_message("second exact reply"),
            after_user_id=promoted.response_user_item_id,
        )
        assert reply is not None
        assert state.runtime_config.chat.response_owner_for_item(reply.id) == target_chat_id
        service.unregister(conn_id)

    def test_response_fifo_overflow_is_reported_without_replacing_accepted_turns(self):
        prompt_queue = Queue()
        service = RealtimeService(text_prompt_queue=prompt_queue)
        conn_id = service.register()
        overflow_events = []

        for index in range(10):
            turn_id = f"turn_{index}"
            service.dispatch_pipeline_event(
                conn_id,
                SpeechStartedEvent(turn_id=turn_id, turn_revision=0, interrupt_response=False),
            )
            overflow_events = service.dispatch_pipeline_event(
                conn_id,
                TranscriptionCompletedEvent(
                    transcript=f"question {index}",
                    turn_id=turn_id,
                    turn_revision=0,
                ),
            )

        st = service._state(conn_id)
        assert st.pending_response_turn_id == "turn_0"
        assert [request.turn_id for request in st.deferred_response_requests] == [
            f"turn_{index}" for index in range(1, 9)
        ]
        assert overflow_events[-1].type == "error"
        assert overflow_events[-1].error.type == "response_queue_full"
        service.unregister(conn_id)

    def test_queued_delete_promotes_a_held_successor_turn(self):
        tracker = SpeculativeTurnTracker()
        cancel_scope = CancelScope()
        prompt_queue = Queue()
        service = RealtimeService(
            text_prompt_queue=prompt_queue,
            speculative_turns=tracker,
            cancel_scope=cancel_scope,
        )
        conn_id = service.register()
        queued = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_queued", turn_revision=0),
        )[0]
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="first", turn_id="turn_queued", turn_revision=0),
        )
        assert prompt_queue.get_nowait().turn_id == "turn_queued"
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_successor", turn_revision=0, interrupt_response=False),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="second", turn_id="turn_successor", turn_revision=0),
        )
        st = service._state(conn_id)
        assert st.pending_response_turn_id == "turn_queued"
        assert len(st.deferred_response_requests) == 1
        assert st.deferred_response_requests[0].turn_id == "turn_successor"
        assert prompt_queue.empty()

        service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id=queued.item_id),
        )

        successor = prompt_queue.get_nowait()
        assert successor.turn_id == "turn_successor"
        assert successor.cancel_generation == cancel_scope.generation == 1
        assert st.pending_response_turn_id == "turn_successor"
        assert st.deferred_response_requests == []
        service.unregister(conn_id)

    def test_barge_in_releases_successor_promoted_by_queued_delete(self):
        tracker = SpeculativeTurnTracker()
        cancel_scope = CancelScope()
        prompt_queue = Queue()
        service = RealtimeService(
            text_prompt_queue=prompt_queue,
            speculative_turns=tracker,
            cancel_scope=cancel_scope,
        )
        conn_id = service.register()
        created_items = {}
        for turn_id in ("turn_active", "turn_deleted", "turn_promoted"):
            created_items[turn_id] = service.dispatch_pipeline_event(
                conn_id,
                SpeechStartedEvent(turn_id=turn_id, turn_revision=0, interrupt_response=False),
            )[0]
            service.dispatch_pipeline_event(
                conn_id,
                TranscriptionCompletedEvent(transcript=turn_id, turn_id=turn_id, turn_revision=0),
            )
            if turn_id == "turn_active":
                prompt_queue.get_nowait()
                service.dispatch_pipeline_event(
                    conn_id,
                    AssistantTextEvent(text="active reply", turn_id=turn_id, turn_revision=0),
                )

        state = service._state(conn_id)
        promoted_chat_id = state.input_item_chat_ids[created_items["turn_promoted"].item_id]
        service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(
                type="conversation.item.delete",
                item_id=created_items["turn_deleted"].item_id,
            ),
        )

        assert state.pending_response_turn_id == "turn_promoted"
        assert promoted_chat_id in state.runtime_config.chat._protected_response_user_ids

        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_interrupt", turn_revision=0, interrupt_response=True),
        )

        assert not state.in_response
        assert not state.response_pending
        assert promoted_chat_id not in state.runtime_config.chat._protected_response_user_ids
        service.unregister(conn_id)

    def test_delete_restores_compacted_history_and_removes_only_exact_user(self, service, conn_id):
        from speech_to_speech.LLM.chat import Chat, CompactionResult, make_assistant_message

        st = service._state(conn_id)
        st.runtime_config.chat = Chat(size=2)
        service.text_prompt_queue = None
        audio_items = []
        for index in range(3):
            started = service.dispatch_pipeline_event(
                conn_id,
                SpeechStartedEvent(turn_id=f"turn_{index}", turn_revision=0),
            )[0]
            audio_items.append(started.item_id)
            service.dispatch_pipeline_event(
                conn_id,
                TranscriptionCompletedEvent(
                    transcript=f"user-{index}",
                    turn_id=f"turn_{index}",
                    turn_revision=0,
                ),
            )
            st.runtime_config.chat.add_item(make_assistant_message(f"assistant-{index}"))
        st.runtime_config.chat.trim_if_needed(
            lambda _snapshot: CompactionResult(user_summary="summary-user", assistant_summary="summary-assistant")
        )
        assert st.runtime_config.chat._compact_thread is not None
        st.runtime_config.chat._compact_thread.join(timeout=2.0)

        deleted = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id=audio_items[0]),
        )

        assert isinstance(deleted[0], ConversationItemDeletedEvent)
        texts = [
            part.text
            for item in st.runtime_config.chat.buffer
            for part in getattr(item, "content", [])
            if getattr(part, "text", None)
        ]
        assert "user-0" not in texts
        assert "assistant-0" in texts
        assert "user-1" in texts
        assert "user-2" in texts
        assert "summary-user" not in texts

    def test_lossy_eviction_keeps_protocol_visible_item_deletable(self):
        service = RealtimeService(chat_size=1)
        conn_id = service.register()
        for item_id, text in (("msg_old", "old"), ("msg_new", "new")):
            service.handle_conversation_item_create(
                conn_id,
                ConversationItemCreateEvent(
                    type="conversation.item.create",
                    item={
                        "id": item_id,
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                ),
            )
        st = service._state(conn_id)
        st.runtime_config.chat.trim_if_needed()
        assert [item.id for item in st.runtime_config.chat.buffer] == ["msg_new"]

        events = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id="msg_old"),
        )

        assert isinstance(events[0], ConversationItemDeletedEvent)
        assert st.protocol_item_ids == ["msg_new"]
        assert [item.id for item in st.runtime_config.chat.buffer] == ["msg_new"]
        service.unregister(conn_id)

    def test_missing_item_returns_correlated_error(self, service, conn_id):
        events = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(
                type="conversation.item.delete",
                event_id="event_delete",
                item_id="item_missing",
            ),
        )

        assert len(events) == 1
        assert isinstance(events[0], RealtimeErrorEvent)
        assert events[0].error.type == "item_not_found"
        assert events[0].error.event_id == "event_delete"

    def test_guarded_missing_item_delete_poison_is_correlated_and_sticky(self, service, conn_id):
        _activate_home_assistant_guard(service, conn_id)

        events = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(
                type="conversation.item.delete",
                event_id="guarded_delete_missing",
                item_id="msg_missing",
            ),
        )

        assert isinstance(events[0], RealtimeErrorEvent)
        assert events[0].error.type == "invalid_conversation_item"
        assert events[0].error.event_id == "guarded_delete_missing"
        assert service._state(conn_id).runtime_config.home_assistant_guard_failed

    def test_client_item_rebinds_explicit_response_away_from_an_old_audio_turn(self):
        tracker = SpeculativeTurnTracker()
        cancel_scope = CancelScope()
        prompt_queue = Queue()
        service = RealtimeService(
            text_prompt_queue=prompt_queue,
            speculative_turns=tracker,
            cancel_scope=cancel_scope,
        )
        conn_id = service.register()
        old_audio = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_old", turn_revision=0),
        )[0]
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="old", turn_id="turn_old", turn_revision=0),
        )
        prompt_queue.get_nowait()
        service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(text="old reply", turn_id="turn_old", turn_revision=0),
        )
        service.finish_response(conn_id)
        service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={
                    "id": "msg_new",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "new request"}],
                },
            ),
        )
        service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        prompt_queue.get_nowait()
        st = service._state(conn_id)
        generation_before_delete = cancel_scope.generation

        old_deleted = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id=old_audio.item_id),
        )

        assert [event.type for event in old_deleted] == ["conversation.item.deleted"]
        assert st.in_response
        assert st.active_response_turn_id is None
        assert st.active_response_input_item_id == "msg_new"
        assert cancel_scope.generation == generation_before_delete

        exact_deleted = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id="msg_new"),
        )
        assert [event.type for event in exact_deleted] == ["conversation.item.deleted", "response.done"]
        assert st.in_response is False
        service.unregister(conn_id)

    def test_response_input_clears_obsolete_audio_turn_ownership(self):
        cancel_scope = CancelScope()
        prompt_queue = Queue()
        service = RealtimeService(text_prompt_queue=prompt_queue, cancel_scope=cancel_scope)
        conn_id = service.register()
        old_audio = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_old", turn_revision=0),
        )[0]
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="old", turn_id="turn_old", turn_revision=0),
        )
        prompt_queue.get_nowait()
        service.finish_response(conn_id)
        service.handle_response_create(
            conn_id,
            ResponseCreateEvent(
                type="response.create",
                response={
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "new request"}],
                        }
                    ]
                },
            ),
        )
        prompt_queue.get_nowait()
        generation_before_delete = cancel_scope.generation

        deleted = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id=old_audio.item_id),
        )

        assert [event.type for event in deleted] == ["conversation.item.deleted"]
        assert service._state(conn_id).in_response
        assert cancel_scope.generation == generation_before_delete

    def test_response_input_reuse_retires_deleted_audio_identity_before_later_delete(self):
        service = RealtimeService(text_prompt_queue=Queue(), cancel_scope=CancelScope())
        conn_id = service.register()
        state = service._state(conn_id)
        created = service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={
                    "id": "msg_inline_reuse",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "old audio transcript"}],
                },
            ),
        )
        assert created[0].type == "conversation.item.created"
        state.audio_input_item_ids.add("msg_inline_reuse")
        state.input_item_chat_ids["msg_inline_reuse"] = "msg_inline_reuse"
        state.input_item_turn_ids["msg_inline_reuse"] = "turn_inline_old"
        state.turn_input_item_ids["turn_inline_old"] = "msg_inline_reuse"
        assert (
            service.handle_conversation_item_delete(
                conn_id,
                ConversationItemDeleteEvent(type="conversation.item.delete", item_id="msg_inline_reuse"),
            )[0].type
            == "conversation.item.deleted"
        )

        response = service.handle_response_create(
            conn_id,
            ResponseCreateEvent(
                type="response.create",
                response={
                    "input": [
                        {
                            "id": "msg_inline_reuse",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "new inline request"}],
                        }
                    ]
                },
            ),
        )
        assert isinstance(response, ResponseCreatedEvent)
        assert "msg_inline_reuse" not in state.audio_input_item_ids
        assert "msg_inline_reuse" not in state.deleted_input_item_ids

        deleted_again = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id="msg_inline_reuse"),
        )
        assert [event.type for event in deleted_again] == ["conversation.item.deleted", "response.done"]
        assert state.runtime_config.chat.user_message("msg_inline_reuse") is None
        service.unregister(conn_id)

    def test_initial_active_response_owner_survives_later_turn_hard_eviction(self):
        from openai.types.realtime.realtime_conversation_item_function_call import (
            RealtimeConversationItemFunctionCall,
        )

        prompt_queue = Queue()
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(
            text_prompt_queue=prompt_queue,
            chat_size=1,
            speculative_turns=tracker,
            cancel_scope=CancelScope(),
        )
        conn_id = service.register()
        first_audio = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_a", turn_revision=0, interrupt_response=False),
        )[0]
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="turn A", turn_id="turn_a", turn_revision=0),
        )
        prompt_queue.get_nowait()
        state = service._state(conn_id)
        first_chat_id = state.input_item_chat_ids[first_audio.item_id]
        service.response._ensure_response(conn_id)

        for turn_id, transcript in (("turn_b", "turn B"), ("turn_c", "turn C")):
            service.dispatch_pipeline_event(
                conn_id,
                SpeechStartedEvent(turn_id=turn_id, turn_revision=0, interrupt_response=False),
            )
            service.dispatch_pipeline_event(
                conn_id,
                TranscriptionCompletedEvent(transcript=transcript, turn_id=turn_id, turn_revision=0),
            )

        chat = state.runtime_config.chat
        assert chat.user_message(first_chat_id) is not None
        recorded = chat.add_response_item(
            RealtimeConversationItemFunctionCall(
                type="function_call",
                id="fc_active_owner",
                call_id="call_active_owner",
                name="lookup",
                arguments="{}",
            ),
            after_user_id=first_chat_id,
        )
        assert isinstance(recorded, RealtimeConversationItemFunctionCall)
        assert "call_active_owner" in chat._pending_tool_calls
        service.unregister(conn_id)

    def test_clearing_initial_pending_response_releases_its_eviction_lease(self):
        prompt_queue = Queue()
        service = RealtimeService(
            text_prompt_queue=prompt_queue,
            chat_size=1,
            speculative_turns=SpeculativeTurnTracker(),
            cancel_scope=CancelScope(),
        )
        conn_id = service.register()
        started = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_pending", turn_revision=0, interrupt_response=False),
        )[0]
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(
                transcript="pending request",
                turn_id="turn_pending",
                turn_revision=0,
            ),
        )
        state = service._state(conn_id)
        chat_id = state.input_item_chat_ids[started.item_id]
        assert chat_id in state.runtime_config.chat._protected_response_user_ids

        service.response.clear_pending_requests(conn_id)

        assert chat_id not in state.runtime_config.chat._protected_response_user_ids
        service.unregister(conn_id)

    def test_response_input_rejects_a_duplicate_live_id_without_false_deletion(self, service, conn_id):
        created = service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={
                    "id": "msg_duplicate",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "original"}],
                },
            ),
        )
        assert created[0].type == "conversation.item.created"

        rejected = service.handle_response_create(
            conn_id,
            ResponseCreateEvent(
                type="response.create",
                response={
                    "input": [
                        {
                            "id": "msg_duplicate",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "duplicate"}],
                        }
                    ]
                },
            ),
        )

        assert isinstance(rejected, RealtimeErrorEvent)
        assert rejected.error.type == "duplicate_item_id"
        assert [item.content[0].text for item in service._state(conn_id).runtime_config.chat.buffer] == ["original"]

    def test_deleting_any_recorded_response_input_user_cancels_its_response(self):
        prompt_queue = Queue()
        cancel_scope = CancelScope()
        service = RealtimeService(text_prompt_queue=prompt_queue, cancel_scope=cancel_scope)
        conn_id = service.register()
        created = service.handle_response_create(
            conn_id,
            ResponseCreateEvent(
                type="response.create",
                response={
                    "input": [
                        {
                            "id": "msg_input_one",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "one"}],
                        },
                        {
                            "id": "msg_input_two",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "two"}],
                        },
                    ]
                },
            ),
        )
        assert isinstance(created, ResponseCreatedEvent)
        prompt_queue.get_nowait()

        deleted = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id="msg_input_one"),
        )

        assert [event.type for event in deleted] == ["conversation.item.deleted", "response.done"]
        assert service._state(conn_id).in_response is False
        assert cancel_scope.generation == 1
        service.unregister(conn_id)

    def test_multi_input_response_writeback_depends_on_every_serialized_user(self):
        prompt_queue = Queue()
        service = RealtimeService(text_prompt_queue=prompt_queue, cancel_scope=CancelScope())
        conn_id = service.register()
        created = service.handle_response_create(
            conn_id,
            ResponseCreateEvent(
                type="response.create",
                response={
                    "input": [
                        {
                            "id": "msg_input_one",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "one"}],
                        },
                        {
                            "id": "msg_input_two",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "two"}],
                        },
                    ]
                },
            ),
        )
        assert isinstance(created, ResponseCreatedEvent)
        request = prompt_queue.get_nowait()
        assert request.response_user_item_id == "msg_input_two"
        assert request.response_user_item_ids == {"msg_input_one", "msg_input_two"}
        response_item = service._state(conn_id).runtime_config.chat.add_response_item(
            make_assistant_message("combined answer"),
            after_user_id=request.response_user_item_id,
            owner_user_ids=request.response_user_item_ids,
        )
        assert response_item is not None and response_item.id is not None

        service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id="msg_input_one"),
        )

        assert all(item.id != response_item.id for item in service._state(conn_id).runtime_config.chat.buffer)
        service.unregister(conn_id)

    def test_partial_multi_input_delete_preserves_surviving_follow_up_owner(self):
        prompt_queue = Queue()
        service = RealtimeService(text_prompt_queue=prompt_queue, cancel_scope=CancelScope())
        conn_id = service.register()
        service.handle_response_create(
            conn_id,
            ResponseCreateEvent(
                type="response.create",
                response={
                    "input": [
                        {
                            "id": "msg_input_one",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "one"}],
                        },
                        {
                            "id": "msg_input_two",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "two"}],
                        },
                    ]
                },
            ),
        )
        prompt_queue.get_nowait()
        service.finish_response(conn_id)

        service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id="msg_input_one"),
        )
        state = service._state(conn_id)
        assert state.response_context_input_item_id == "msg_input_two"
        assert state.response_context_input_item_ids == {"msg_input_two"}

        service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        follow_up = prompt_queue.get_nowait()
        assert follow_up.response_user_item_id == "msg_input_two"
        assert follow_up.response_user_item_ids == {"msg_input_two"}
        response_item = state.runtime_config.chat.add_response_item(
            make_assistant_message("surviving answer"),
            after_user_id=follow_up.response_user_item_id,
            owner_user_ids=follow_up.response_user_item_ids,
        )
        assert response_item is not None and response_item.id is not None
        service.finish_response(conn_id)

        service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id="msg_input_two"),
        )

        assert all(item.id != response_item.id for item in state.runtime_config.chat.buffer)
        service.unregister(conn_id)

    def test_response_input_dependency_batch_cannot_exceed_protocol_bound(self, monkeypatch):
        import speech_to_speech.api.openai_realtime.service as service_module

        monkeypatch.setattr(service_module, "MAX_TRACKED_PROTOCOL_ITEMS", 2)
        service = RealtimeService(text_prompt_queue=Queue(), cancel_scope=CancelScope())
        conn_id = service.register()

        rejected = service.handle_response_create(
            conn_id,
            ResponseCreateEvent(
                type="response.create",
                response={
                    "input": [
                        {
                            "id": f"msg_input_{index}",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": str(index)}],
                        }
                        for index in range(3)
                    ]
                },
            ),
        )

        state = service._state(conn_id)
        assert isinstance(rejected, RealtimeErrorEvent)
        assert rejected.error.type == "invalid_input_item"
        assert state.runtime_config.chat.buffer == []
        assert state.protocol_item_ids == []
        assert state.response_context_input_item_ids == set()
        assert state.admitting_response_input_item_ids == set()
        service.unregister(conn_id)

    def test_response_input_dependencies_reserve_supported_turn_fifo(self, monkeypatch):
        import speech_to_speech.api.openai_realtime.service as service_module

        monkeypatch.setattr(service_module, "MAX_TRACKED_PROTOCOL_ITEMS", 4)
        monkeypatch.setattr(service_module, "MAX_DEFERRED_RESPONSE_REQUESTS", 1)
        tracker = SpeculativeTurnTracker()
        prompt_queue = Queue()
        service = RealtimeService(
            text_prompt_queue=prompt_queue,
            speculative_turns=tracker,
            cancel_scope=CancelScope(),
        )
        conn_id = service.register()

        created = service.handle_response_create(
            conn_id,
            ResponseCreateEvent(
                type="response.create",
                response={
                    "input": [
                        {
                            "id": "msg_input_active",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "active"}],
                        }
                    ]
                },
            ),
        )
        assert created is not None and created.type == "response.created"
        prompt_queue.get_nowait()

        turn_items = {}
        for turn_id in ("turn_pending", "turn_deferred"):
            turn_items[turn_id] = service.dispatch_pipeline_event(
                conn_id,
                SpeechStartedEvent(turn_id=turn_id, turn_revision=0, interrupt_response=False),
            )[0].item_id
            service.dispatch_pipeline_event(
                conn_id,
                TranscriptionCompletedEvent(transcript=turn_id, turn_id=turn_id, turn_revision=0),
            )

        state = service._state(conn_id)
        assert state.pending_response_turn_id == "turn_pending"
        assert [request.turn_id for request in state.deferred_response_requests] == ["turn_deferred"]
        for turn_id, item_id in turn_items.items():
            assert item_id in state.protocol_item_ids
            assert state.turn_input_item_ids[turn_id] == item_id
            assert tracker.is_latest(turn_id, 0)
        service.unregister(conn_id)

    def test_protocol_retirement_prunes_completed_response_context(self, monkeypatch):
        import speech_to_speech.api.openai_realtime.service as service_module

        monkeypatch.setattr(service_module, "MAX_TRACKED_PROTOCOL_ITEMS", 5)
        monkeypatch.setattr(service_module, "MAX_DEFERRED_RESPONSE_REQUESTS", 1)
        service = RealtimeService(text_prompt_queue=Queue(), cancel_scope=CancelScope())
        conn_id = service.register()
        service.handle_response_create(
            conn_id,
            ResponseCreateEvent(
                type="response.create",
                response={
                    "input": [
                        {
                            "id": f"msg_input_{index}",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": str(index)}],
                        }
                        for index in range(2)
                    ]
                },
            ),
        )
        service.finish_response(conn_id)
        state = service._state(conn_id)

        for index in range(5):
            state.record_protocol_item(f"item_later_{index}")

        assert len(state.protocol_item_ids) == 5
        assert state.response_context_input_item_id is None
        assert state.response_context_input_item_ids == set()
        service.unregister(conn_id)

    def test_multi_input_dependencies_survive_a_tool_follow_up(self):
        from openai.types.realtime.realtime_conversation_item_function_call import (
            RealtimeConversationItemFunctionCall,
        )

        prompt_queue = Queue()
        service = RealtimeService(text_prompt_queue=prompt_queue, cancel_scope=CancelScope())
        conn_id = service.register()
        service.handle_response_create(
            conn_id,
            ResponseCreateEvent(
                type="response.create",
                response={
                    "input": [
                        {
                            "id": "msg_input_one",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "one"}],
                        },
                        {
                            "id": "msg_input_two",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "two"}],
                        },
                    ]
                },
            ),
        )
        first_request = prompt_queue.get_nowait()
        chat = service._state(conn_id).runtime_config.chat
        chat.add_response_item(
            RealtimeConversationItemFunctionCall(
                type="function_call",
                call_id="call_multi",
                name="lookup",
                arguments="{}",
            ),
            after_user_id=first_request.response_user_item_id,
            owner_user_ids=first_request.response_user_item_ids,
        )
        service.finish_response(conn_id)
        created = service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={
                    "id": "fco_multi",
                    "type": "function_call_output",
                    "call_id": "call_multi",
                    "output": "result",
                },
            ),
        )
        assert created[0].type == "conversation.item.created"

        service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        follow_up = prompt_queue.get_nowait()
        assert follow_up.response_user_item_id == "msg_input_two"
        assert follow_up.response_user_item_ids == {"msg_input_one", "msg_input_two"}
        response_item = chat.add_response_item(
            make_assistant_message("follow-up answer"),
            after_user_id=follow_up.response_user_item_id,
            owner_user_ids=follow_up.response_user_item_ids,
        )
        assert response_item is not None and response_item.id is not None

        service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id="msg_input_one"),
        )

        assert all(item.id != response_item.id for item in chat.buffer)
        service.unregister(conn_id)

    def test_inline_tool_output_inherits_every_original_response_input_dependency(self):
        from openai.types.realtime.realtime_conversation_item_function_call import (
            RealtimeConversationItemFunctionCall,
        )

        prompt_queue = Queue()
        cancel_scope = CancelScope()
        service = RealtimeService(text_prompt_queue=prompt_queue, cancel_scope=cancel_scope)
        conn_id = service.register()
        service.handle_response_create(
            conn_id,
            ResponseCreateEvent(
                type="response.create",
                response={
                    "input": [
                        {
                            "id": "msg_inline_one",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "one"}],
                        },
                        {
                            "id": "msg_inline_two",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "two"}],
                        },
                    ]
                },
            ),
        )
        first_request = prompt_queue.get_nowait()
        chat = service._state(conn_id).runtime_config.chat
        chat.add_response_item(
            RealtimeConversationItemFunctionCall(
                type="function_call",
                call_id="call_inline",
                name="lookup",
                arguments="{}",
            ),
            after_user_id=first_request.response_user_item_id,
            owner_user_ids=first_request.response_user_item_ids,
        )
        service.finish_response(conn_id)

        created = service.handle_response_create(
            conn_id,
            ResponseCreateEvent(
                type="response.create",
                response={
                    "input": [
                        {
                            "id": "fco_inline",
                            "type": "function_call_output",
                            "call_id": "call_inline",
                            "output": "result",
                        }
                    ]
                },
            ),
        )

        assert isinstance(created, ResponseCreatedEvent)
        follow_up = prompt_queue.get_nowait()
        assert follow_up.response_user_item_id == "msg_inline_two"
        assert follow_up.response_user_item_ids == {"msg_inline_one", "msg_inline_two"}
        state = service._state(conn_id)
        assert state.active_response_input_item_ids == {"msg_inline_one", "msg_inline_two"}
        response_item = chat.add_response_item(
            make_assistant_message("inline follow-up"),
            after_user_id=follow_up.response_user_item_id,
            owner_user_ids=follow_up.response_user_item_ids,
        )
        assert response_item is not None and response_item.id is not None

        deleted = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id="msg_inline_one"),
        )

        assert [event.type for event in deleted] == ["conversation.item.deleted", "response.done"]
        assert cancel_scope.generation == 1
        assert all(item.id != response_item.id for item in chat.buffer)
        service.unregister(conn_id)

    def test_mixed_inline_input_anchors_writeback_after_last_canonical_user(self):
        from openai.types.realtime.realtime_conversation_item_function_call import (
            RealtimeConversationItemFunctionCall,
        )

        prompt_queue = Queue()
        service = RealtimeService(text_prompt_queue=prompt_queue, cancel_scope=CancelScope())
        conn_id = service.register()
        service.handle_response_create(
            conn_id,
            ResponseCreateEvent(
                type="response.create",
                response={
                    "input": [
                        {
                            "id": "msg_older_owner",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "older owner"}],
                        }
                    ]
                },
            ),
        )
        first_request = prompt_queue.get_nowait()
        chat = service._state(conn_id).runtime_config.chat
        chat.add_response_item(
            RealtimeConversationItemFunctionCall(
                type="function_call",
                call_id="call_mixed",
                name="lookup",
                arguments="{}",
            ),
            after_user_id=first_request.response_user_item_id,
        )
        service.finish_response(conn_id)

        created = service.handle_response_create(
            conn_id,
            ResponseCreateEvent(
                type="response.create",
                response={
                    "input": [
                        {
                            "id": "msg_new_prompt",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "new prompt"}],
                        },
                        {
                            "id": "fco_mixed",
                            "type": "function_call_output",
                            "call_id": "call_mixed",
                            "output": "result",
                        },
                    ]
                },
            ),
        )

        assert isinstance(created, ResponseCreatedEvent)
        request = prompt_queue.get_nowait()
        assert request.response_user_item_id == "msg_new_prompt"
        assert request.response_user_item_ids == {"msg_older_owner", "msg_new_prompt"}
        service.unregister(conn_id)

    def test_promoted_snapshot_excludes_protocol_id_recreated_after_admission(self):
        service = RealtimeService(text_prompt_queue=Queue(), cancel_scope=CancelScope())
        conn_id = service.register()
        for item_id, text in (("msg_target", "target"), ("msg_reused", "old value")):
            events = service.handle_conversation_item_create(
                conn_id,
                ConversationItemCreateEvent(
                    type="conversation.item.create",
                    item={
                        "id": item_id,
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                ),
            )
            assert [event.type for event in events] == ["conversation.item.created"]
        state = service._state(conn_id)
        state.turn_input_item_ids["turn_target"] = "msg_target"
        state.input_item_chat_ids["msg_target"] = "msg_target"
        request = GenerateResponseRequest(
            runtime_config=state.runtime_config,
            chat_snapshot=state.runtime_config.chat.copy(),
            response_user_item_id="msg_target",
            response_user_item_ids={"msg_target"},
            admitted_protocol_item_ids=set(state.protocol_item_ids),
            admitted_protocol_sequence=state.next_protocol_item_sequence,
            turn_id="turn_target",
            turn_revision=0,
        )

        deleted = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id="msg_reused"),
        )
        assert deleted[0].type == "conversation.item.deleted"
        recreated = service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={
                    "id": "msg_reused",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "new value"}],
                },
            ),
        )
        assert recreated[0].type == "conversation.item.created"

        service.response.resume_pending_request(conn_id, request, enqueue=False)

        promoted = state.pending_response_request
        assert promoted is not None and promoted.chat_snapshot is not None
        serialized = promoted.chat_snapshot.to_responses_api_chat()
        serialized_text = json.dumps(serialized)
        assert "target" in serialized_text
        assert "new value" not in serialized_text
        service.unregister(conn_id)

    def test_deleted_audio_id_reused_as_user_item_loses_stale_audio_identity(self, service, conn_id):
        created = service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={
                    "id": "msg_reused_audio",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "old audio transcript"}],
                },
            ),
        )
        assert created[0].type == "conversation.item.created"
        state = service._state(conn_id)
        state.audio_input_item_ids.add("msg_reused_audio")
        state.input_item_chat_ids["msg_reused_audio"] = "msg_reused_audio"
        state.input_item_turn_ids["msg_reused_audio"] = "turn_old_audio"
        state.turn_input_item_ids["turn_old_audio"] = "msg_reused_audio"

        deleted = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id="msg_reused_audio"),
        )
        assert deleted[0].type == "conversation.item.deleted"
        assert "msg_reused_audio" in state.audio_input_item_ids
        assert "msg_reused_audio" in state.deleted_input_item_ids

        recreated = service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={
                    "id": "msg_reused_audio",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "new text item"}],
                },
            ),
        )
        assert recreated[0].type == "conversation.item.created"
        assert "msg_reused_audio" not in state.audio_input_item_ids
        assert "msg_reused_audio" not in state.deleted_input_item_ids
        assert "msg_reused_audio" not in state.input_item_turn_ids
        assert "turn_old_audio" not in state.turn_input_item_ids

        deleted_again = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id="msg_reused_audio"),
        )
        assert deleted_again[0].type == "conversation.item.deleted"
        assert state.runtime_config.chat.user_message("msg_reused_audio") is None

    def test_deferred_user_reusing_deleted_audio_id_can_be_deleted_before_flush(self):
        service = RealtimeService(text_prompt_queue=Queue(), cancel_scope=CancelScope())
        conn_id = service.register()
        started = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_old_audio", turn_revision=0),
        )[0]
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(
                transcript="old audio transcript",
                turn_id="turn_old_audio",
                turn_revision=0,
            ),
        )
        service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id=started.item_id),
        )
        state = service._state(conn_id)
        assert started.item_id in state.audio_input_item_ids
        assert started.item_id in state.deleted_input_item_ids

        active = service.handle_response_create(
            conn_id,
            ResponseCreateEvent(
                type="response.create",
                response={
                    "input": [
                        {
                            "id": "msg_active_owner",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "active"}],
                        }
                    ]
                },
            ),
        )
        assert isinstance(active, ResponseCreatedEvent)
        deferred = service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={
                    "id": started.item_id,
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "replacement"}],
                },
            ),
        )
        assert deferred == []
        assert started.item_id not in state.audio_input_item_ids
        assert [item.id for item in state.deferred_items] == [started.item_id]

        deleted = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id=started.item_id),
        )

        assert [event.type for event in deleted] == ["conversation.item.deleted"]
        assert state.deferred_items == []
        service.finish_response(conn_id)
        assert state.runtime_config.chat.user_message(started.item_id) is None
        service.unregister(conn_id)

    def test_untranscribed_audio_delete_never_removes_the_prior_turn(self, service, conn_id):
        st = service._state(conn_id)
        first = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=0),
        )[0]
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="keep me", turn_id="turn_1", turn_revision=0),
        )
        second = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_2", turn_revision=0),
        )[0]

        deleted = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(
                type="conversation.item.delete",
                event_id="delete_second",
                item_id=second.item_id,
            ),
        )

        assert isinstance(deleted[0], ConversationItemDeletedEvent)
        assert [item.content[0].text for item in st.runtime_config.chat.buffer] == ["keep me"]
        assert st.last_item_id == first.item_id

    def test_deleted_speculative_turn_cannot_reopen_or_recreate_history(self):
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(speculative_turns=tracker)
        conn_id = service.register()
        st = service._state(conn_id)
        started = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=0),
        )[0]
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="assistant echo", turn_id="turn_1", turn_revision=0),
        )
        service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(
                type="conversation.item.delete",
                event_id="delete_turn",
                item_id=started.item_id,
            ),
        )

        assert tracker.begin_reopen_candidate("turn_1", 0) is None
        assert (
            service.dispatch_pipeline_event(
                conn_id,
                SpeechStartedEvent(turn_id="turn_1", turn_revision=1, reopened=True),
            )
            == []
        )
        assert (
            service.dispatch_pipeline_event(
                conn_id,
                TranscriptionCompletedEvent(transcript="assistant echo again", turn_id="turn_1", turn_revision=1),
            )
            == []
        )
        assert st.runtime_config.chat.buffer == []
        service.unregister(conn_id)

    def test_late_speech_stop_stays_stale_after_exact_tombstone_eviction(self):
        tracker = SpeculativeTurnTracker(max_tracked_turns=1)
        service = RealtimeService(speculative_turns=tracker)
        conn_id = service.register()
        for turn_id in ("turn_deleted_1", "turn_deleted_2"):
            started = service.dispatch_pipeline_event(
                conn_id,
                SpeechStartedEvent(turn_id=turn_id, turn_revision=0),
            )[0]
            service.handle_conversation_item_delete(
                conn_id,
                ConversationItemDeleteEvent(type="conversation.item.delete", item_id=started.item_id),
            )

        assert "turn_deleted_1" not in tracker._discarded_turn_ids
        assert (
            service.dispatch_pipeline_event(
                conn_id,
                SpeechStoppedEvent(turn_id="turn_deleted_1", turn_revision=0, audio_end_ms=10),
            )
            == []
        )
        service.unregister(conn_id)

    def test_reusing_existing_protocol_item_never_moves_the_creation_tail(self, service, conn_id):
        st = service._state(conn_id)
        st.record_protocol_item("item_audio")
        st.record_protocol_item("item_tool")
        st.record_protocol_item("item_audio")

        created = service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={
                    "id": "msg_after",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "after"}],
                },
            ),
        )[0]

        assert created.previous_item_id == "item_tool"

    def test_deleted_audio_indexes_are_bounded_and_retire_correlated_maps(self, service, conn_id, monkeypatch):
        import speech_to_speech.api.openai_realtime.service as service_module

        monkeypatch.setattr(service_module, "MAX_TRACKED_PROTOCOL_ITEMS", 2)
        st = service._state(conn_id)
        for index in range(3):
            item_id = f"item_{index}"
            turn_id = f"turn_{index}"
            st.audio_input_item_ids.add(item_id)
            st.input_item_turn_ids[item_id] = turn_id
            st.turn_input_item_ids[turn_id] = item_id
            st.record_deleted_input_item(item_id)

        assert list(st.deleted_input_item_ids) == ["item_1", "item_2"]
        assert "item_0" not in st.audio_input_item_ids
        assert "item_0" not in st.input_item_turn_ids
        assert "turn_0" not in st.turn_input_item_ids

    def test_protocol_index_retirement_permanently_discards_late_audio_turn(self, monkeypatch):
        import speech_to_speech.api.openai_realtime.service as service_module

        monkeypatch.setattr(service_module, "MAX_TRACKED_PROTOCOL_ITEMS", 2)
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(speculative_turns=tracker)
        conn_id = service.register()
        state = service._state(conn_id)
        for index in range(3):
            item_id = f"item_{index}"
            turn_id = f"turn_{index}"
            tracker.observe(turn_id, 0)
            state.audio_input_item_ids.add(item_id)
            state.input_item_turn_ids[item_id] = turn_id
            state.turn_input_item_ids[turn_id] = item_id
            state.record_protocol_item(item_id)

        assert not tracker.is_latest("turn_0", 0)
        assert (
            service.dispatch_pipeline_event(
                conn_id,
                TranscriptionCompletedEvent(transcript="must stay retired", turn_id="turn_0", turn_revision=0),
            )
            == []
        )
        assert not any(
            getattr(part, "text", None) == "must stay retired"
            for item in state.runtime_config.chat.buffer
            for part in getattr(item, "content", [])
        )
        service.unregister(conn_id)

    def test_protocol_index_retains_pending_and_active_response_owner(self, monkeypatch):
        import speech_to_speech.api.openai_realtime.service as service_module

        monkeypatch.setattr(service_module, "MAX_TRACKED_PROTOCOL_ITEMS", 2)
        tracker = SpeculativeTurnTracker()
        prompt_queue = Queue()
        service = RealtimeService(text_prompt_queue=prompt_queue, speculative_turns=tracker)
        conn_id = service.register()
        owner = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_owner", turn_revision=0, interrupt_response=False),
        )[0]
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="keep my turn", turn_id="turn_owner", turn_revision=0),
        )
        request = prompt_queue.get_nowait()
        state = service._state(conn_id)

        for item_id in ("msg_ordinary_a", "msg_ordinary_b"):
            service.handle_conversation_item_create(
                conn_id,
                ConversationItemCreateEvent(
                    type="conversation.item.create",
                    item={
                        "id": item_id,
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": item_id}],
                    },
                ),
            )

        assert owner.item_id in state.protocol_item_ids
        assert "msg_ordinary_a" not in state.protocol_item_ids
        assert state.turn_input_item_ids["turn_owner"] == owner.item_id
        assert tracker.is_latest("turn_owner", 0)

        service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(
                parts=[
                    AssistantTextPart(text="answer"),
                    AssistantToolCallPart(
                        tool={"type": "function_call", "call_id": "call_owner", "name": "lookup", "arguments": "{}"}
                    ),
                ],
                turn_id=request.turn_id,
                turn_revision=request.turn_revision,
            ),
        )

        assert state.in_response
        assert state.active_response_turn_id == "turn_owner"
        assert owner.item_id in state.protocol_item_ids
        assert state.input_item_turn_ids[owner.item_id] == "turn_owner"
        assert tracker.is_latest("turn_owner", 0)

        service.finish_response(conn_id)

        assert not state.in_response
        assert not state.response_pending
        service.unregister(conn_id)

    def test_audio_tail_delete_restores_protocol_visible_predecessor(self, service, conn_id):
        first = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=0),
        )[0]
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="first", turn_id="turn_1", turn_revision=0),
        )
        second = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_2", turn_revision=0),
        )[0]
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="second", turn_id="turn_2", turn_revision=0),
        )
        service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id=second.item_id),
        )

        created = service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={
                    "id": "msg_after",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "after"}],
                },
            ),
        )[0]

        assert created.previous_item_id == first.item_id

    def test_deferred_flush_never_uses_an_unacknowledged_item_as_predecessor(self, service, conn_id):
        service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={
                    "id": "msg_a",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "a"}],
                },
            ),
        )
        st = service._state(conn_id)
        st.in_response = True
        for item_id in ("msg_b", "msg_c"):
            service.handle_conversation_item_create(
                conn_id,
                ConversationItemCreateEvent(
                    type="conversation.item.create",
                    item={
                        "id": item_id,
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": item_id}],
                    },
                ),
            )
        service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id="msg_a"),
        )

        created = service.conversation.flush_deferred_items(conn_id)

        assert [(event.item.id, event.previous_item_id) for event in created] == [
            ("msg_b", None),
            ("msg_c", "msg_b"),
        ]

    def test_barrier_rejection_keeps_delete_event_correlation(self, service, conn_id):
        service._state(conn_id).runtime_config.transcript_barrier_pending_sequence = 1

        error = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(
                type="conversation.item.delete",
                event_id="delete_pending",
                item_id="msg_x",
            ),
        )[0]

        assert error.error.type == "transcript_barrier_pending"
        assert error.error.event_id == "delete_pending"


class TestDeferConversationItemsDuringResponse:
    """conversation.item.create is buffered while a response is generating and
    flushed, in order, once it completes — so a client item never races the LLM
    handler's chat write-back (which runs on the pipeline thread)."""

    def _text_event(self, text: str, item_id: str = "msg_x") -> ConversationItemCreateEvent:
        return ConversationItemCreateEvent(
            type="conversation.item.create",
            item={  # type: ignore[arg-type]
                "id": item_id,
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        )

    def _user_texts(self, chat) -> list[str]:
        return [i.content[0].text for i in chat.buffer if getattr(i, "role", None) == "user"]

    def test_applied_immediately_when_no_active_response(self, service, conn_id):
        st = service._state(conn_id)
        assert st.in_response is False
        events = service.handle_conversation_item_create(conn_id, self._text_event("hi"))
        assert len(events) == 1
        assert isinstance(events[0], ConversationItemCreatedEvent)
        assert self._user_texts(st.runtime_config.chat) == ["hi"]
        assert st.deferred_items == []

    def test_item_deferred_while_in_response(self, service, conn_id):
        st = service._state(conn_id)
        st.in_response = True
        events = service.handle_conversation_item_create(conn_id, self._text_event("hi"))
        assert events == []  # ack deferred too
        assert len(st.deferred_items) == 1
        assert self._user_texts(st.runtime_config.chat) == []  # not yet in chat

    def test_deferred_items_flushed_in_order_on_finish(self, service, conn_id):
        st = service._state(conn_id)
        st.in_response = True
        service.handle_conversation_item_create(conn_id, self._text_event("a", "msg_1"))
        service.handle_conversation_item_create(conn_id, self._text_event("b", "msg_2"))
        assert self._user_texts(st.runtime_config.chat) == []

        events = service.finish_response(conn_id)

        assert st.in_response is False
        assert st.deferred_items == []
        assert self._user_texts(st.runtime_config.chat) == ["a", "b"]  # arrival order preserved
        created = [e for e in events if isinstance(e, ConversationItemCreatedEvent)]
        assert len(created) == 2

    def test_function_call_output_deferred_then_pairs_after_response(self, service, conn_id):
        from openai.types.realtime.realtime_conversation_item_function_call import (
            RealtimeConversationItemFunctionCall,
        )

        st = service._state(conn_id)
        chat = st.runtime_config.chat
        # The function_call the generation produced (held in _pending_tool_calls).
        chat.add_item(
            RealtimeConversationItemFunctionCall(
                type="function_call", call_id="call_1", name="camera_snapshot", arguments="{}"
            )
        )
        st.in_response = True
        evt = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "function_call_output", "output": "ok", "call_id": "call_1"},
        )
        # Output arrives mid-response: deferred (applying now could race), no error.
        assert service.handle_conversation_item_create(conn_id, evt) == []
        assert len(st.deferred_items) == 1

        finish_events = service.finish_response(conn_id)

        # Flushed after completion → pairs cleanly, no invalid_conversation_item error.
        assert not any(isinstance(e, RealtimeErrorEvent) for e in finish_events)
        assert chat._has_call_id_in_buffer("call_1")
        assert chat.buffer[-1].type == "function_call_output"

    def test_tool_result_follow_up_keeps_original_user_deletion_owner(
        self,
        service,
        conn_id,
        text_prompt_queue,
    ):
        from openai.types.realtime.realtime_conversation_item_function_call import (
            RealtimeConversationItemFunctionCall,
        )

        st = service._state(conn_id)
        chat = st.runtime_config.chat
        started = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_owner", turn_revision=0, interrupt_response=False),
        )[0]
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="look this up", turn_id="turn_owner", turn_revision=0),
        )
        first_request = text_prompt_queue.get_nowait()
        assert first_request.response_user_item_id is not None
        service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(text="checking", turn_id="turn_owner", turn_revision=0),
        )
        chat.add_response_item(
            RealtimeConversationItemFunctionCall(
                type="function_call",
                call_id="call_owned",
                name="search",
                arguments="{}",
            ),
            after_user_id=first_request.response_user_item_id,
        )
        service.finish_response(conn_id)
        created = service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={"type": "function_call_output", "output": "result", "call_id": "call_owned"},
            ),
        )
        output_id = created[0].item.id

        assert output_id is not None
        assert chat.response_owner_for_item(output_id) == first_request.response_user_item_id
        assert st.response_context_input_item_id == started.item_id
        assert st.response_context_input_item_ids == {started.item_id}

        response_created = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        assert isinstance(response_created, ResponseCreatedEvent)
        request = text_prompt_queue.get_nowait()
        assert request.response_user_item_id == first_request.response_user_item_id
        assert st.active_response_input_item_ids == {started.item_id}
        chat.add_response_item(make_assistant_message("owned follow-up"), after_user_id=request.response_user_item_id)

        deleted = service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id=started.item_id),
        )

        assert isinstance(deleted[0], ConversationItemDeletedEvent)
        assert any(isinstance(event, ResponseDoneEvent) for event in deleted)
        assert not st.in_response
        assert all(
            getattr(part, "text", None) not in {"look this up", "owned follow-up"}
            for item in chat.buffer
            for part in getattr(item, "content", [])
        )
        assert not any(getattr(item, "call_id", None) == "call_owned" for item in chat.buffer)

    def test_guarded_deferred_batch_is_atomic_when_later_output_is_invalid(
        self,
        service,
        conn_id,
        runtime_config,
    ):
        from openai.types.realtime.realtime_conversation_item_function_call import (
            RealtimeConversationItemFunctionCall,
        )

        _activate_home_assistant_guard(service, conn_id)
        st = service._state(conn_id)
        chat = runtime_config.chat
        chat.add_item(
            RealtimeConversationItemFunctionCall(
                type="function_call",
                call_id="call_valid",
                name="home_assistant__GetLiveContext",
                arguments="{}",
            )
        )
        st.in_response = True
        for call_id in ("call_valid", "call_unknown"):
            assert (
                service.handle_conversation_item_create(
                    conn_id,
                    ConversationItemCreateEvent(
                        type="conversation.item.create",
                        item={"type": "function_call_output", "output": "ok", "call_id": call_id},
                    ),
                )
                == []
            )

        events = service.finish_response(conn_id)

        assert runtime_config.home_assistant_guard_failed
        assert st.deferred_items == []
        assert "call_valid" in chat._pending_tool_calls
        assert not any(getattr(item, "type", None) == "function_call_output" for item in chat.buffer)
        assert not any(isinstance(event, ConversationItemCreatedEvent) for event in events)
        errors = [event for event in events if isinstance(event, RealtimeErrorEvent)]
        assert len(errors) == 1 and errors[0].error.type == "invalid_conversation_item"


# ===================================================================
# Audio commit
# ===================================================================


class TestHandleAudioCommit:
    def test_commit_after_audio(self, service, conn_id):
        service._state(conn_id).audio_buffer_has_data = True
        err = service.handle_audio_commit(conn_id)
        assert err is None
        assert service._state(conn_id).audio_buffer_has_data is False

    def test_commit_empty_buffer(self, service, conn_id):
        err = service.handle_audio_commit(conn_id)
        assert isinstance(err, RealtimeErrorEvent)
        assert err.error.type == "input_audio_buffer_commit_empty"


# ===================================================================
# Response create
# ===================================================================


class TestHandleResponseCreate:
    def test_response_create_ok(self, service, conn_id):
        evt = ResponseCreateEvent(type="response.create")
        result = service.handle_response_create(conn_id, evt)
        assert isinstance(result, ResponseCreatedEvent)
        assert result.response.status == "in_progress"
        st = service._state(conn_id)
        assert st.in_response is True
        assert st.current_response_id is not None
        assert st.current_item_id is not None

    def test_explicit_response_create_prevents_duplicate_created_from_assistant_output(self, service, conn_id):
        created = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))

        assistant_events = service.dispatch_pipeline_event(conn_id, AssistantTextEvent(text="Hello"))

        assert isinstance(created, ResponseCreatedEvent)
        assert not any(isinstance(event, ResponseCreatedEvent) for event in assistant_events)

    def test_response_create_while_active(self, service, conn_id):
        service._state(conn_id).in_response = True
        evt = ResponseCreateEvent(type="response.create")
        err = service.handle_response_create(conn_id, evt)
        assert isinstance(err, RealtimeErrorEvent)
        assert err.error.type == "conversation_already_has_active_response"

    def test_response_create_rejects_pending_implicit_fifo_without_mutation(self):
        prompt_queue = Queue()
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(
            text_prompt_queue=prompt_queue,
            speculative_turns=tracker,
            cancel_scope=CancelScope(),
        )
        conn_id = service.register()
        for turn_id in ("turn_a", "turn_b"):
            service.dispatch_pipeline_event(
                conn_id,
                SpeechStartedEvent(turn_id=turn_id, turn_revision=0, interrupt_response=False),
            )
            service.dispatch_pipeline_event(
                conn_id,
                TranscriptionCompletedEvent(transcript=turn_id, turn_id=turn_id, turn_revision=0),
            )

        rejected = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))

        state = service._state(conn_id)
        assert isinstance(rejected, RealtimeErrorEvent)
        assert rejected.error.type == "conversation_already_has_active_response"
        assert state.response_pending and not state.in_response
        assert state.pending_response_turn_id == "turn_a"
        assert [request.turn_id for request in state.deferred_response_requests] == ["turn_b"]
        assert prompt_queue.get_nowait().turn_id == "turn_a"
        assert prompt_queue.empty()
        service.unregister(conn_id)

    def test_response_create_stores_overrides(self, service, conn_id, runtime_config, text_prompt_queue):
        evt = ResponseCreateEvent(
            type="response.create",
            response={
                "instructions": "override instructions",
                "tool_choice": "auto",
            },
        )
        result = service.handle_response_create(conn_id, evt)
        assert isinstance(result, ResponseCreatedEvent)
        req = text_prompt_queue.get()
        assert isinstance(req, GenerateResponseRequest)
        assert req.response is not None
        assert req.response.instructions == "override instructions"
        assert req.response.tool_choice == "auto"
        assert req.runtime_config is runtime_config

    def test_response_create_preserves_latest_user_turn_timing(self, service, conn_id, text_prompt_queue):
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(
                transcript="hello",
                language_code="en",
                turn_id="turn_1",
                turn_revision=2,
                speech_stopped_at_s=123.0,
            ),
        )
        initial_req = text_prompt_queue.get()
        assert isinstance(initial_req, GenerateResponseRequest)
        assert initial_req.turn_id == "turn_1"
        assert initial_req.turn_revision == 2
        assert initial_req.speech_stopped_at_s == 123.0
        service.finish_response(conn_id)

        result = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))

        assert isinstance(result, ResponseCreatedEvent)
        followup_req = text_prompt_queue.get()
        assert isinstance(followup_req, GenerateResponseRequest)
        assert followup_req.turn_id == "turn_1"
        assert followup_req.turn_revision == 2
        assert followup_req.speech_stopped_at_s == 123.0

    def test_response_create_rejects_complex_tool_choice(self, service, conn_id, runtime_config):
        evt = ResponseCreateEvent(
            type="response.create",
            response={
                "tool_choice": {"type": "function", "name": "my_func"},
            },
        )
        err = service.handle_response_create(conn_id, evt)
        assert isinstance(err, RealtimeErrorEvent)
        assert err.error.type == "tool_choice_not_supported"
        assert service._state(conn_id).in_response is False

    def test_response_create_accepts_valid_str_tool_choices(self, service, conn_id, text_prompt_queue):
        for choice in ("auto", "required", "none"):
            evt = ResponseCreateEvent(
                type="response.create",
                response={"tool_choice": choice},
            )
            result = service.handle_response_create(conn_id, evt)
            assert isinstance(result, ResponseCreatedEvent), f"Expected ResponseCreatedEvent for tool_choice={choice!r}"
            req = text_prompt_queue.get()
            assert isinstance(req, GenerateResponseRequest)
            assert req.response.tool_choice == choice
            service.response._end_response(conn_id)

    def test_response_create_with_image_input_items(self, service, conn_id, text_prompt_queue):
        evt = ResponseCreateEvent(
            type="response.create",
            response={
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Describe this image"},
                            {"type": "input_image", "image_url": "https://example.com/photo.jpg"},
                        ],
                    }
                ],
            },
        )
        result = service.handle_response_create(conn_id, evt)
        assert isinstance(result, ResponseCreatedEvent)
        gen_msg = text_prompt_queue.get()
        assert isinstance(gen_msg, GenerateResponseRequest)

    def test_response_create_rejects_invalid_function_call_output_in_input(self, service, conn_id, text_prompt_queue):
        evt = ResponseCreateEvent(
            type="response.create",
            response={
                "input": [
                    {"type": "function_call_output", "output": '{"x": 1}', "call_id": "call_bogus"},
                ],
            },
        )
        result = service.handle_response_create(conn_id, evt)
        assert isinstance(result, RealtimeErrorEvent)
        assert "call_bogus" in result.error.message
        assert service._state(conn_id).in_response is False

    def test_ordinary_response_input_rejection_retains_valid_prefix(self, service, conn_id):
        event = ResponseCreateEvent(
            type="response.create",
            response={
                "input": [
                    self._user_input("ordinary prefix"),
                    {
                        "id": "INVALID_SECOND_ITEM",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "invalid"}],
                    },
                ]
            },
        )

        result = service.handle_response_create(conn_id, event)

        assert isinstance(result, RealtimeErrorEvent)
        assert service._state(conn_id).in_response is False
        assert [item.content[0].text for item in service._state(conn_id).runtime_config.chat.buffer] == [
            "ordinary prefix"
        ]

    def test_private_response_input_rejection_retains_no_valid_prefix(self, service, conn_id):
        state = service._state(conn_id)
        state.runtime_config.transcript_barrier_version = 1
        state.runtime_config.transcript_barrier_nonce = "ac" * 32
        canary = "PRIVATE_REJECTED_PREFIX_CANARY"
        event = ResponseCreateEvent(
            type="response.create",
            response={
                "input": [
                    {
                        "id": "msg_private_prefix",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": canary}],
                    },
                    {
                        "id": "PRIVATE_INVALID_SECOND_ITEM",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "invalid"}],
                    },
                ]
            },
        )

        result = service.handle_response_create(conn_id, event)

        assert isinstance(result, RealtimeErrorEvent)
        assert result.error.message == "Invalid private client event."
        assert state.in_response is False
        assert state.runtime_config.chat.buffer == []
        assert canary not in str(state.runtime_config.chat.to_responses_api_chat())

    def test_double_response_create_rejected(self, service, conn_id, text_prompt_queue):
        """Second response.create is rejected because in_response is set immediately."""
        evt = ResponseCreateEvent(type="response.create")
        result1 = service.handle_response_create(conn_id, evt)
        assert isinstance(result1, ResponseCreatedEvent)
        result2 = service.handle_response_create(conn_id, evt)
        assert isinstance(result2, RealtimeErrorEvent)
        assert result2.error.type == "conversation_already_has_active_response"

    @staticmethod
    def _user_input(text):
        return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}

    def test_response_create_out_of_band_does_not_append_input_to_default_chat(
        self, service, conn_id, text_prompt_queue
    ):
        chat = service._state(conn_id).runtime_config.chat
        assert len(chat.buffer) == 0
        evt = ResponseCreateEvent(
            type="response.create",
            response={"conversation": "none", "input": [self._user_input("OOB question")]},
        )
        result = service.handle_response_create(conn_id, evt)
        assert isinstance(result, ResponseCreatedEvent)
        # Out-of-band: the default conversation is left untouched...
        assert len(chat.buffer) == 0
        # ...while the input still rides along on the queued request for the LM to use.
        req = text_prompt_queue.get()
        assert isinstance(req, GenerateResponseRequest)
        assert req.response.input is not None and len(req.response.input) == 1

    def test_response_create_in_band_appends_input_to_default_chat(self, service, conn_id, text_prompt_queue):
        chat = service._state(conn_id).runtime_config.chat
        evt = ResponseCreateEvent(type="response.create", response={"input": [self._user_input("in band")]})
        result = service.handle_response_create(conn_id, evt)
        assert isinstance(result, ResponseCreatedEvent)
        assert len(chat.buffer) == 1  # in-band input is threaded into the conversation

    def test_response_create_out_of_band_carries_null_turn(self, service, conn_id, text_prompt_queue):
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(
                transcript="hello",
                language_code="en",
                turn_id="turn_1",
                turn_revision=2,
                speech_stopped_at_s=123.0,
            ),
        )
        text_prompt_queue.get()  # drain the STT-triggered request
        service.finish_response(conn_id)

        result = service.handle_response_create(
            conn_id, ResponseCreateEvent(type="response.create", response={"conversation": "none"})
        )
        assert isinstance(result, ResponseCreatedEvent)
        req = text_prompt_queue.get()
        # Null turn identity makes every speculative-staleness gate treat it as always-latest.
        assert req.turn_id is None
        assert req.turn_revision is None
        assert req.speech_stopped_at_s is None

    def test_response_create_out_of_band_reports_null_conversation_id(self, service, conn_id):
        result = service.handle_response_create(
            conn_id, ResponseCreateEvent(type="response.create", response={"conversation": "none"})
        )
        assert isinstance(result, ResponseCreatedEvent)
        assert result.response.conversation_id is None
        done = [e for e in service.finish_response(conn_id) if isinstance(e, ResponseDoneEvent)]
        assert done and done[0].response.conversation_id is None

    def test_response_create_in_band_reports_conversation_id(self, service, conn_id):
        result = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        assert isinstance(result, ResponseCreatedEvent)
        assert result.response.conversation_id == service._state(conn_id).conversation_id


# ===================================================================
# Response cancel
# ===================================================================


class TestHandleResponseCancel:
    def test_cancel_active_response(self, service, conn_id, should_listen):
        should_listen.clear()
        service.response._ensure_response(conn_id)
        events = service.handle_response_cancel(conn_id)
        assert len(events) == 1
        assert isinstance(events[0], ResponseDoneEvent)
        assert events[0].response.status == "cancelled"
        assert events[0].response.status_details.reason == "client_cancelled"
        assert should_listen.is_set()

    def test_cancel_no_active_response(self, service, conn_id):
        events = service.handle_response_cancel(conn_id)
        assert events == []


# ===================================================================
# Outbound audio encoding
# ===================================================================


class TestEncodeAudioChunk:
    def test_first_chunk_emits_response_created_and_delta(self, service, conn_id):
        audio = _pcm_bytes(256)
        events = service.encode_audio_chunk(conn_id, audio)
        assert len(events) == 2
        assert isinstance(events[0], ResponseCreatedEvent)
        resp = events[0].response
        assert resp.status == "in_progress"
        assert resp.object == "realtime.response"
        assert resp.conversation_id is not None
        assert isinstance(events[1], ResponseAudioDeltaEvent)
        assert events[1].content_index == 0
        assert events[1].output_index == 0
        assert events[1].delta == base64.b64encode(audio).decode("ascii")

    def test_subsequent_chunks_increment_content_index(self, service, conn_id):
        service.encode_audio_chunk(conn_id, _pcm_bytes(256))  # first
        events = service.encode_audio_chunk(conn_id, _pcm_bytes(256))  # second
        assert len(events) == 1
        assert isinstance(events[0], ResponseAudioDeltaEvent)
        assert events[0].content_index == 1

    def test_assistant_output_creates_response_before_later_audio(self, service, conn_id):
        assistant_events = service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(text="Hello there"),
        )
        assert isinstance(assistant_events[0], ResponseCreatedEvent)

        audio_events = service.encode_audio_chunk(conn_id, _pcm_bytes(256))

        assert len(audio_events) == 1
        assert isinstance(audio_events[0], ResponseAudioDeltaEvent)
        assert audio_events[0].response_id == assistant_events[0].response.id

    def test_tool_outputs_do_not_rebind_streaming_audio_item(self, service, conn_id):
        assistant_events = service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(
                parts=[
                    AssistantTextPart(text="I will handle that."),
                    AssistantToolCallPart(
                        tool={"type": "function_call", "call_id": "c1", "name": "first", "arguments": "{}"}
                    ),
                    AssistantToolCallPart(
                        tool={"type": "function_call", "call_id": "c2", "name": "second", "arguments": "{}"}
                    ),
                ]
            ),
        )
        transcript = assistant_events[1]
        first_tool = assistant_events[2]
        second_tool = assistant_events[3]

        first_audio = service.encode_audio_chunk(conn_id, _pcm_bytes(256))[0]
        second_audio = service.encode_audio_chunk(conn_id, _pcm_bytes(256))[0]
        done = service.finish_response(conn_id)[0]

        assert isinstance(transcript, ResponseAudioTranscriptDoneEvent)
        assert isinstance(first_audio, ResponseAudioDeltaEvent)
        assert isinstance(second_audio, ResponseAudioDeltaEvent)
        assert isinstance(done, ResponseAudioDoneEvent)
        assert first_tool.item_id != transcript.item_id
        assert second_tool.item_id != transcript.item_id
        assert first_audio.item_id == second_audio.item_id == done.item_id == transcript.item_id
        assert (
            first_audio.output_index == second_audio.output_index == done.output_index == transcript.output_index == 0
        )
        assert [first_audio.content_index, second_audio.content_index] == [0, 1]

    def test_tool_first_reserves_output_zero_for_later_audio(self, service, conn_id):
        assistant_events = service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(
                parts=[
                    AssistantToolCallPart(
                        tool={"type": "function_call", "call_id": "c1", "name": "first", "arguments": "{}"}
                    ),
                    AssistantTextPart(text="Here you go."),
                ]
            ),
        )
        tool_event = assistant_events[1]
        transcript = assistant_events[2]

        audio_delta = service.encode_audio_chunk(conn_id, _pcm_bytes(256))[0]
        audio_done = service.finish_response(conn_id)[0]

        assert isinstance(tool_event, ResponseFunctionCallArgumentsDoneEvent)
        assert isinstance(transcript, ResponseAudioTranscriptDoneEvent)
        assert isinstance(audio_delta, ResponseAudioDeltaEvent)
        assert isinstance(audio_done, ResponseAudioDoneEvent)
        assert tool_event.output_index == 1
        assert transcript.output_index == audio_delta.output_index == audio_done.output_index == 0
        assert tool_event.item_id != transcript.item_id
        assert transcript.item_id == audio_delta.item_id == audio_done.item_id

    def test_audio_created_first_prevents_duplicate_created_from_assistant_output(self, service, conn_id):
        audio_events = service.encode_audio_chunk(conn_id, _pcm_bytes(256))

        assistant_events = service.dispatch_pipeline_event(conn_id, AssistantTextEvent(text="Hello"))

        assert sum(isinstance(event, ResponseCreatedEvent) for event in audio_events) == 1
        assert not any(isinstance(event, ResponseCreatedEvent) for event in assistant_events)

    def test_response_created_includes_metadata(self, service, conn_id):
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
            metadata={"key": "value"},
        )
        events = service.encode_audio_chunk(conn_id, _pcm_bytes(256))
        resp = events[0].response
        assert resp.metadata == {"key": "value"}


# ===================================================================
# Finish audio response
# ===================================================================


class TestFinishAudioResponse:
    def test_finish_emits_audio_done_and_response_done(self, service, conn_id):
        service.encode_audio_chunk(conn_id, _pcm_bytes(256))
        events = service.finish_response(conn_id)
        assert len(events) == 2
        assert isinstance(events[0], ResponseAudioDoneEvent)
        assert events[0].content_index == 0
        assert isinstance(events[1], ResponseDoneEvent)
        assert events[1].response.status == "completed"

    def test_finish_text_only_skips_audio_done(self, service, conn_id):
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
            output_modalities=["text"],
        )
        service.response._ensure_response(conn_id)
        events = service.finish_response(conn_id)
        assert len(events) == 1
        assert isinstance(events[0], ResponseDoneEvent)
        assert events[0].response.status == "completed"
        assert not any(isinstance(e, ResponseAudioDoneEvent) for e in events)

    def test_finish_with_cancel_status(self, service, conn_id):
        service.encode_audio_chunk(conn_id, _pcm_bytes(256))
        events = service.finish_response(conn_id, status="cancelled", reason="turn_detected")
        done = events[1]
        assert done.response.status == "cancelled"
        assert done.response.status_details.reason == "turn_detected"

    def test_tool_only_response_skips_orphaned_audio_done(self, service, conn_id):
        service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(tools=[{"type": "function_call", "call_id": "c1", "name": "tool", "arguments": "{}"}]),
        )

        events = service.finish_response(conn_id)

        assert len(events) == 1
        assert isinstance(events[0], ResponseDoneEvent)
        assert not any(isinstance(event, ResponseAudioDoneEvent) for event in events)

    def test_finish_resets_state(self, service, conn_id):
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
            metadata={"k": "v"},
        )
        service.response._ensure_response(conn_id)
        service.finish_response(conn_id)
        st = service._state(conn_id)
        assert st.in_response is False
        assert st.current_response_id is None
        assert st.current_item_id is None
        assert st.current_response_params is None
        assert st.audio_output_started is False


# ===================================================================
# Pipeline text translation
# ===================================================================


class TestDispatchPipelineEvent:
    def test_poisoned_enabled_barrier_drops_inflight_assistant_text_and_tools(
        self,
        service,
        conn_id,
        runtime_config,
    ):
        runtime_config.transcript_barrier_version = 1
        runtime_config.transcript_barrier_nonce = "91" * 32
        service.poison_transcript_barrier(conn_id, "test_failure")

        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(
                text="PRIVATE_ASSISTANT_DRAIN_CONTENT",
                tools=[
                    {
                        "type": "function_call",
                        "call_id": "call_private_drain",
                        "name": "private_tool",
                        "arguments": "{}",
                    }
                ],
            ),
        )

        assert events == []
        assert service._state(conn_id).current_response_id is None

    # -- speech_started --

    def test_speech_started_emits_event(self, service, conn_id):
        events = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(),
        )
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, InputAudioBufferSpeechStartedEvent)
        assert evt.audio_start_ms == 0
        assert evt.item_id.startswith("item_")

    def test_speech_started_cancels_active_response(self, service, conn_id):
        service.encode_audio_chunk(conn_id, _pcm_bytes(256))
        events = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(),
        )
        cancel_events = [e for e in events if isinstance(e, (ResponseAudioDoneEvent, ResponseDoneEvent))]
        assert len(cancel_events) == 2
        done = [e for e in cancel_events if isinstance(e, ResponseDoneEvent)][0]
        assert done.response.status == "cancelled"
        assert done.response.status_details.reason == "turn_detected"
        speech = [e for e in events if isinstance(e, InputAudioBufferSpeechStartedEvent)]
        assert len(speech) == 1

    def test_speech_started_no_response_emits_only_started(self, service, conn_id):
        """speech_started without active response emits only the started event."""
        events = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(),
        )
        assert len(events) == 1
        assert isinstance(events[0], InputAudioBufferSpeechStartedEvent)

    def test_speech_started_does_not_cancel_when_interrupt_disabled(self, service, conn_id):
        """With interrupt_response=False, speech_started emits the started event but does NOT cancel the active response."""
        from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

        service._state(conn_id).runtime_config.session.audio.input.turn_detection = ServerVad(
            type="server_vad",
            interrupt_response=False,
        )
        _, response_item_id = service.response._ensure_response(conn_id)
        events = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(),
        )
        assert len(events) == 1
        assert isinstance(events[0], InputAudioBufferSpeechStartedEvent)
        assert service._state(conn_id).in_response is True
        assert service._state(conn_id).current_item_id == response_item_id

    def test_speech_started_internal_non_interrupt_does_not_cancel(self, service, conn_id):
        audio_events = service.encode_audio_chunk(conn_id, _pcm_bytes(256))
        response_item_id = audio_events[-1].item_id
        events = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(interrupt_response=False),
        )

        assert len(events) == 1
        assert isinstance(events[0], InputAudioBufferSpeechStartedEvent)
        assert service._state(conn_id).in_response is True
        assert service._state(conn_id).current_item_id == response_item_id
        done_events = service.finish_response(conn_id)
        assert done_events[0].item_id == response_item_id

    def test_consecutive_speech_cycles_get_distinct_item_ids(self, service, conn_id):
        """Each speech_started/stopped cycle generates a new unique item_id."""
        started_1 = service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        stopped_1 = service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent())

        started_2 = service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        stopped_2 = service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent())

        id_1 = started_1[0].item_id
        id_2 = started_2[0].item_id
        assert id_1 != id_2
        assert stopped_1[0].item_id == id_1
        assert stopped_2[0].item_id == id_2

    # -- speech_stopped --

    def test_speech_stopped_emits_event(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        events = service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(),
        )
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, InputAudioBufferSpeechStoppedEvent)
        assert evt.audio_end_ms == 0

    def test_speech_stopped_same_item_id_as_started(self, service, conn_id):
        started = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(),
        )
        stopped = service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(),
        )
        assert started[0].item_id == stopped[0].item_id

    def test_speech_stopped_stores_duration(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=2.5),
        )
        assert service._state(conn_id).input_audio_duration_s == 2.5

    def test_speech_stopped_zero_duration_not_stored(self, service, conn_id):
        """Phantom trigger (duration_s=0) emits stopped event but doesn't overwrite duration."""
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        events = service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(),
        )
        assert len(events) == 1
        assert isinstance(events[0], InputAudioBufferSpeechStoppedEvent)
        assert service._state(conn_id).input_audio_duration_s == 0.0

    # -- assistant_text --

    def test_assistant_text_emits_transcript_done(self, service, conn_id):
        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(text="Hello there"),
        )
        assert len(events) == 2
        assert isinstance(events[0], ResponseCreatedEvent)
        evt = events[1]
        assert isinstance(evt, ResponseAudioTranscriptDoneEvent)
        assert evt.content_index == 0
        assert evt.output_index == 0
        assert evt.transcript == "Hello there"

    def test_assistant_text_with_tools(self, service, conn_id):
        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(
                text="Let me check",
                tools=[
                    {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": '{"city": "Paris"}'},
                    {"type": "function_call", "call_id": "c2", "name": "get_time", "arguments": "{}"},
                ],
            ),
        )
        assert len(events) == 4
        assert isinstance(events[0], ResponseCreatedEvent)
        assert isinstance(events[1], ResponseAudioTranscriptDoneEvent)
        assert events[1].output_index == 0
        assert isinstance(events[2], ResponseFunctionCallArgumentsDoneEvent)
        assert events[2].output_index == 1
        assert events[2].name == "get_weather"
        assert events[2].call_id == "c1"
        assert json.loads(events[2].arguments) == {"city": "Paris"}
        assert isinstance(events[3], ResponseFunctionCallArgumentsDoneEvent)
        assert events[3].output_index == 2

    def test_assistant_text_tools_only(self, service, conn_id):
        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(
                text="",
                tools=[{"type": "function_call", "call_id": "c1", "name": "f1", "arguments": "{}"}],
            ),
        )
        assert len(events) == 2
        assert isinstance(events[0], ResponseCreatedEvent)
        assert isinstance(events[1], ResponseFunctionCallArgumentsDoneEvent)
        assert events[1].output_index == 1
        assert events[1].response_id == events[0].response.id

    def test_partless_assistant_event_is_ignored_without_opening_response(self, service, conn_id):
        service._state(conn_id).last_item_id = "item_old"

        events = service.dispatch_pipeline_event(conn_id, AssistantTextEvent())

        assert events == []
        assert service._state(conn_id).last_item_id == "item_old"
        assert service._state(conn_id).current_response_id is None
        assert service._state(conn_id).in_response is False

    def test_audio_text_stays_on_output_zero_when_interleaved_with_tools(self, service, conn_id):
        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(
                parts=[
                    AssistantTextPart(text="before"),
                    AssistantToolCallPart(
                        tool={"type": "function_call", "call_id": "c1", "name": "first", "arguments": "{}"}
                    ),
                    AssistantTextPart(text="after first"),
                    AssistantToolCallPart(
                        tool={"type": "function_call", "call_id": "c2", "name": "second", "arguments": "{}"}
                    ),
                ]
            ),
        )

        assert [event.type for event in events] == [
            "response.created",
            "response.output_audio_transcript.done",
            "response.function_call_arguments.done",
            "response.output_audio_transcript.done",
            "response.function_call_arguments.done",
        ]
        assert [event.output_index for event in events[1:]] == [0, 1, 0, 2]
        assert events[1].item_id == events[3].item_id
        assert len({event.item_id for event in events[1:]}) == 3

        audio_delta = service.encode_audio_chunk(conn_id, _pcm_bytes(256))[0]
        audio_done = service.finish_response(conn_id)[0]

        assert isinstance(audio_delta, ResponseAudioDeltaEvent)
        assert isinstance(audio_done, ResponseAudioDoneEvent)
        assert audio_delta.item_id == audio_done.item_id == events[1].item_id
        assert audio_delta.output_index == audio_done.output_index == 0

    def test_audio_text_reuses_output_zero_across_pipeline_events(self, service, conn_id):
        first = service.dispatch_pipeline_event(conn_id, AssistantTextEvent(text="before"))
        second = service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(tools=[{"type": "function_call", "call_id": "c1", "name": "tool", "arguments": "{}"}]),
        )
        third = service.dispatch_pipeline_event(conn_id, AssistantTextEvent(text="after"))

        assert isinstance(first[0], ResponseCreatedEvent)
        assert [first[1].output_index, second[0].output_index, third[0].output_index] == [0, 1, 0]
        assert first[1].item_id == third[0].item_id
        assert second[0].item_id != first[1].item_id

    def test_output_indices_restart_after_response_finishes(self, service, conn_id):
        first = service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(
                tools=[
                    {"type": "function_call", "call_id": "c1", "name": "first", "arguments": "{}"},
                    {"type": "function_call", "call_id": "c2", "name": "second", "arguments": "{}"},
                ]
            ),
        )
        service.finish_response(conn_id)

        second = service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(tools=[{"type": "function_call", "call_id": "c3", "name": "third", "arguments": "{}"}]),
        )

        assert [event.output_index for event in first[1:]] == [1, 2]
        assert second[1].output_index == 1

    def test_text_only_interleaving_closes_each_text_item_before_response_done(self, service, conn_id):
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
            output_modalities=["text"],
        )
        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(
                parts=[
                    AssistantTextPart(text="before"),
                    AssistantToolCallPart(
                        tool={"type": "function_call", "call_id": "c1", "name": "tool", "arguments": "{}"}
                    ),
                    AssistantTextPart(text="after"),
                ]
            ),
        )

        assert [event.type for event in events] == [
            "response.created",
            "response.output_text.delta",
            "response.function_call_arguments.done",
            "response.output_text.delta",
        ]
        assert [event.output_index for event in events[1:]] == [0, 1, 2]

        done_events = service.finish_response(conn_id)
        text_done = [event for event in done_events if isinstance(event, ResponseTextDoneEvent)]
        assert [(event.output_index, event.text) for event in text_done] == [(0, "before"), (2, "after")]
        assert isinstance(done_events[-1], ResponseDoneEvent)

    def test_assistant_text_text_only_emits_text_events(self, service, conn_id):
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
            output_modalities=["text"],
        )
        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(text="Hello there"),
        )
        # on_assistant_text creates the implicit response, then streams only
        # the delta; the matching done is emitted once at close.
        assert len(events) == 2
        assert isinstance(events[0], ResponseCreatedEvent)
        assert isinstance(events[1], ResponseTextDeltaEvent)
        assert events[1].content_index == 0
        assert events[1].output_index == 0
        assert events[1].delta == "Hello there"
        assert not any(isinstance(e, ResponseTextDoneEvent) for e in events)
        assert not any(isinstance(e, ResponseAudioTranscriptDoneEvent) for e in events)

        done_events = service.finish_response(conn_id)
        text_done = [e for e in done_events if isinstance(e, ResponseTextDoneEvent)]
        assert len(text_done) == 1
        assert text_done[0].content_index == 0
        assert text_done[0].output_index == 0
        assert text_done[0].text == "Hello there"

    def test_text_only_done_concatenates_streamed_parts(self, service, conn_id):
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
            output_modalities=["text"],
        )
        service.dispatch_pipeline_event(conn_id, AssistantTextEvent(text="Hello there. "))
        service.dispatch_pipeline_event(conn_id, AssistantTextEvent(text="How are you?"))
        done_events = service.finish_response(conn_id)
        text_done = [e for e in done_events if isinstance(e, ResponseTextDoneEvent)]
        assert len(text_done) == 1
        # done.text concatenates the raw streamed parts verbatim (== sum of deltas).
        assert text_done[0].text == "Hello there. How are you?"

    def test_text_only_no_text_done_on_cancel(self, service, conn_id):
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
            output_modalities=["text"],
        )
        service.dispatch_pipeline_event(conn_id, AssistantTextEvent(text="partial"))
        done_events = service.finish_response(conn_id, status="cancelled", reason="client_cancelled")
        assert not any(isinstance(e, ResponseTextDoneEvent) for e in done_events)
        assert any(isinstance(e, ResponseDoneEvent) for e in done_events)

    def test_assistant_text_text_only_keeps_tool_events(self, service, conn_id):
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
            output_modalities=["text"],
        )
        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(
                text="Let me check",
                tools=[{"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{}"}],
            ),
        )
        # No per-chunk done: created, delta, then the tool event at output_index 1.
        assert isinstance(events[0], ResponseCreatedEvent)
        assert isinstance(events[1], ResponseTextDeltaEvent)
        assert not any(isinstance(e, ResponseTextDoneEvent) for e in events)
        tool_event = events[2]
        assert isinstance(tool_event, ResponseFunctionCallArgumentsDoneEvent)
        assert tool_event.output_index == 1
        assert tool_event.name == "get_weather"

    def test_assistant_text_waits_for_pending_reopen_and_drops_confirmed_stale_turn(
        self,
        runtime_config,
        should_listen,
    ):
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(should_listen=should_listen, speculative_turns=tracker)
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config
        tracker.observe("turn_1", 0)
        candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
        done = Event()
        result = {}

        def dispatch():
            result["events"] = service.dispatch_pipeline_event(
                conn_id,
                AssistantTextEvent(text="stale", turn_id="turn_1", turn_revision=0),
            )
            done.set()

        thread = Thread(target=dispatch)
        thread.start()

        assert not done.wait(0.05)
        assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)
        assert done.wait(1.0)
        thread.join(timeout=1.0)

        assert result["events"] == []
        assert service._state(conn_id).current_response_id is None
        service.unregister(conn_id)

    def test_assistant_text_waits_for_pending_reopen_and_emits_cancelled_reopen(
        self,
        runtime_config,
        should_listen,
    ):
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(should_listen=should_listen, speculative_turns=tracker)
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config
        tracker.observe("turn_1", 0)
        candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
        done = Event()
        result = {}

        def dispatch():
            result["events"] = service.dispatch_pipeline_event(
                conn_id,
                AssistantTextEvent(text="latest", turn_id="turn_1", turn_revision=0),
            )
            done.set()

        thread = Thread(target=dispatch)
        thread.start()

        assert not done.wait(0.05)
        tracker.cancel_reopen_candidate("turn_1", candidate_revision)
        assert done.wait(1.0)
        thread.join(timeout=1.0)

        assert len(result["events"]) == 2
        assert isinstance(result["events"][0], ResponseCreatedEvent)
        assert isinstance(result["events"][1], ResponseAudioTranscriptDoneEvent)
        assert result["events"][1].transcript == "latest"
        assert tracker.is_committed("turn_1", 0)
        service.unregister(conn_id)

    def test_assistant_text_retries_reopen_without_holding_private_failure_lock(
        self,
        runtime_config,
        should_listen,
        monkeypatch: pytest.MonkeyPatch,
    ):
        tracker = SpeculativeTurnTracker()
        cancel_scope = CancelScope()
        service = RealtimeService(
            should_listen=should_listen,
            speculative_turns=tracker,
            cancel_scope=cancel_scope,
        )
        assert service.verify_cancel_scope_wiring(cancel_scope, cancel_scope)
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config
        _activate_home_assistant_guard(service, conn_id)
        tracker.observe("turn_1", 0)
        original_check = service._is_stale_turn_event
        reopen_started = Event()
        candidate_revision: list[int] = []

        def start_reopen_after_first_check(event, *, wait_for_pending_reopen=True):
            if not reopen_started.is_set():
                candidate = tracker.begin_reopen_candidate("turn_1", 0)
                assert candidate is not None
                candidate_revision.append(candidate)
                reopen_started.set()
                return False
            return original_check(event, wait_for_pending_reopen=wait_for_pending_reopen)

        monkeypatch.setattr(service, "_is_stale_turn_event", start_reopen_after_first_check)
        dispatched = Event()
        result: dict[str, object] = {}

        def dispatch() -> None:
            result["events"] = service.dispatch_pipeline_event(
                conn_id,
                AssistantTextEvent(text="must stay private", turn_id="turn_1", turn_revision=0),
            )
            dispatched.set()

        dispatch_thread = Thread(target=dispatch)
        dispatch_thread.start()
        assert reopen_started.wait(1.0)

        poisoned = Event()

        def poison() -> None:
            runtime_config.fail_home_assistant_guard()
            poisoned.set()

        poison_thread = Thread(target=poison)
        poison_thread.start()
        assert poisoned.wait(0.25), "speculative wait held the private-failure lock"
        tracker.cancel_reopen_candidate("turn_1", candidate_revision[0])
        assert dispatched.wait(1.0)
        dispatch_thread.join(timeout=1.0)
        poison_thread.join(timeout=1.0)

        assert result["events"] == []
        assert runtime_config.home_assistant_guard_failed
        assert service._state(conn_id).current_response_id is None
        service.unregister(conn_id)

    def test_guard_failure_cannot_linearize_inside_terminal_deferred_flush(
        self,
        service,
        conn_id,
        runtime_config,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _activate_home_assistant_guard(service, conn_id)
        st = service._state(conn_id)
        st.in_response = True
        service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={
                    "id": "msg_guarded",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "accepted first"}],
                },
            ),
        )
        original_flush = service.conversation.flush_deferred_items
        flush_entered = Event()
        release_flush = Event()

        def blocked_flush(flush_conn_id: str):
            flush_entered.set()
            assert release_flush.wait(1.0)
            return original_flush(flush_conn_id)

        monkeypatch.setattr(service.conversation, "flush_deferred_items", blocked_flush)
        finished = Event()
        result: dict[str, object] = {}

        def finish() -> None:
            result["events"] = service.finish_response(conn_id)
            finished.set()

        finish_thread = Thread(target=finish)
        finish_thread.start()
        assert flush_entered.wait(1.0)
        poisoned = Event()

        def poison() -> None:
            runtime_config.fail_home_assistant_guard()
            poisoned.set()

        poison_thread = Thread(target=poison)
        poison_thread.start()
        assert not poisoned.wait(0.05), "failure crossed the terminal sink boundary"
        release_flush.set()
        assert finished.wait(1.0)
        assert poisoned.wait(1.0)
        finish_thread.join(timeout=1.0)
        poison_thread.join(timeout=1.0)

        user_texts = [
            item.content[0].text for item in runtime_config.chat.buffer if getattr(item, "role", None) == "user"
        ]
        assert user_texts == ["accepted first"]
        assert any(isinstance(event, ConversationItemCreatedEvent) for event in result["events"])
        assert runtime_config.home_assistant_guard_failed
        service.unregister(conn_id)

    def test_token_usage_waits_for_pending_reopen_and_drops_confirmed_stale_turn(
        self,
        runtime_config,
        should_listen,
    ):
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(should_listen=should_listen, speculative_turns=tracker)
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config
        tracker.observe("turn_1", 0)
        candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
        done = Event()
        result = {}

        def dispatch():
            result["events"] = service.dispatch_pipeline_event(
                conn_id,
                TokenUsageEvent(input_tokens=10, output_tokens=5, turn_id="turn_1", turn_revision=0),
            )
            done.set()

        thread = Thread(target=dispatch)
        thread.start()

        assert not done.wait(0.05)
        assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)
        assert done.wait(1.0)
        thread.join(timeout=1.0)

        assert result["events"] == []
        assert service._state(conn_id).response_usage.input_tokens == 0
        assert service._state(conn_id).response_usage.output_tokens == 0
        service.unregister(conn_id)

    def test_try_dispatch_assistant_text_defers_pending_reopen(self, runtime_config, should_listen):
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(should_listen=should_listen, speculative_turns=tracker)
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config
        tracker.observe("turn_1", 0)
        candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)

        event = AssistantTextEvent(text="latest", turn_id="turn_1", turn_revision=0)

        assert service.try_dispatch_pipeline_event(conn_id, event) is None
        assert service._state(conn_id).current_response_id is None

        tracker.cancel_reopen_candidate("turn_1", candidate_revision)
        events = service.try_dispatch_pipeline_event(conn_id, event)

        assert events is not None
        assert len(events) == 2
        assert isinstance(events[0], ResponseCreatedEvent)
        assert isinstance(events[1], ResponseAudioTranscriptDoneEvent)
        assert events[1].transcript == "latest"
        assert tracker.is_committed("turn_1", 0)
        service.unregister(conn_id)

    def test_try_dispatch_assistant_text_defers_reopen_grace(self, runtime_config, should_listen):
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(should_listen=should_listen, speculative_turns=tracker)
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config
        tracker.observe("turn_1", 0)
        tracker.start_reopen_grace("turn_1", 0, grace_s=0.05)

        event = AssistantTextEvent(text="latest", turn_id="turn_1", turn_revision=0)

        assert service.should_defer_pipeline_event(event)
        assert service.try_dispatch_pipeline_event(conn_id, event) is None
        assert service._state(conn_id).current_response_id is None

        sleep(0.06)
        events = service.try_dispatch_pipeline_event(conn_id, event)

        assert events is not None
        assert len(events) == 2
        assert isinstance(events[0], ResponseCreatedEvent)
        assert isinstance(events[1], ResponseAudioTranscriptDoneEvent)
        assert events[1].transcript == "latest"
        assert tracker.is_committed("turn_1", 0)
        service.unregister(conn_id)

    def test_try_dispatch_token_usage_defers_pending_reopen(self, runtime_config, should_listen):
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(should_listen=should_listen, speculative_turns=tracker)
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config
        tracker.observe("turn_1", 0)
        candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)

        event = TokenUsageEvent(input_tokens=10, output_tokens=5, turn_id="turn_1", turn_revision=0)

        assert service.try_dispatch_pipeline_event(conn_id, event) is None
        assert service._state(conn_id).response_usage.input_tokens == 0

        assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)
        assert service.try_dispatch_pipeline_event(conn_id, event) == []
        assert service._state(conn_id).response_usage.input_tokens == 0
        assert service._state(conn_id).response_usage.output_tokens == 0
        service.unregister(conn_id)

    # -- partial_transcription --

    def test_partial_transcription_emits_delta(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        e1 = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="hel"),
        )
        e2 = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="lo"),
        )
        assert isinstance(e1[0], ConversationItemInputAudioTranscriptionDeltaEvent)
        assert e1[0].content_index == 0
        assert e1[0].delta == "hel"
        assert isinstance(e2[0], ConversationItemInputAudioTranscriptionDeltaEvent)
        assert e2[0].content_index == 1

    # -- transcription_completed --

    def test_transcription_completed_emits_event(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=3.2),
        )
        events = service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="hello world"),
        )
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, ConversationItemInputAudioTranscriptionCompletedEvent)
        assert evt.content_index == 0
        assert evt.transcript == "hello world"
        assert evt.usage.seconds == 3.2
        assert evt.usage.type == "duration"
        assert service._state(conn_id).response_pending is True

    def test_empty_transcription_completed_emits_event_without_response(
        self,
        service,
        conn_id,
        runtime_config,
        text_prompt_queue,
    ):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=1.1),
        )
        events = service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="", language_code="en"),
        )

        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, ConversationItemInputAudioTranscriptionCompletedEvent)
        assert evt.transcript == ""
        assert evt.usage.seconds == 1.1
        assert text_prompt_queue.empty()
        assert runtime_config.chat.buffer == []
        assert service._state(conn_id).response_pending is False

    def test_private_barrier_final_never_enters_chat_or_generation(
        self,
        service,
        conn_id,
        runtime_config,
        text_prompt_queue,
    ):
        nonce = "cd" * 32
        runtime_config.transcript_barrier_version = 1
        runtime_config.transcript_barrier_nonce = nonce
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent(duration_s=1.2))

        events = service.dispatch_pipeline_event(
            conn_id,
            TranscriptBarrierCompletedEvent(transcript="Do you recognize me?", language_code="en"),
        )

        assert len(events) == 1
        assert isinstance(events[0], TranscriptBarrierCompletedServerEvent)
        assert events[0].nonce == nonce
        assert events[0].sequence == 1
        assert events[0].transcript == "Do you recognize me?"
        assert runtime_config.chat.buffer == []
        assert text_prompt_queue.empty()
        assert service._state(conn_id).response_pending is False
        assert runtime_config.transcript_barrier_pending is True
        assert runtime_config.transcript_barrier_pending_transcript == "Do you recognize me?"

    def test_private_barrier_exact_accept_is_the_only_history_entry(
        self,
        service,
        conn_id,
        runtime_config,
        text_prompt_queue,
    ):
        nonce = "34" * 32
        transcript = "Do you recognize me?"
        runtime_config.transcript_barrier_version = 1
        runtime_config.transcript_barrier_nonce = nonce
        completed = service.dispatch_pipeline_event(
            conn_id,
            TranscriptBarrierCompletedEvent(transcript=transcript),
        )[0]
        assert isinstance(completed, TranscriptBarrierCompletedServerEvent)

        resolved = service.handle_transcript_barrier_resolve(
            conn_id,
            TranscriptBarrierResolveEvent.model_validate(
                {
                    "type": "reachy.transcript_barrier.resolve",
                    "version": 1,
                    "nonce": nonce,
                    "sequence": completed.sequence,
                    "input_item_id": completed.item_id,
                    "action": "accept",
                    "item": {
                        "id": "msg_identity_1",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": transcript}],
                    },
                }
            ),
        )

        assert [event.type for event in resolved] == [
            "conversation.item.created",
            "reachy.transcript_barrier.resolved",
        ]
        assert isinstance(resolved[-1], TranscriptBarrierResolvedServerEvent)
        assert resolved[-1].action == "accepted"
        assert resolved[-1].replacement_item_id == "msg_identity_1"
        assert runtime_config.transcript_barrier_pending is False
        assert runtime_config.transcript_barrier_pending_transcript is None
        assert runtime_config.chat.buffer[-1].content[0].text == transcript
        assert text_prompt_queue.empty()

    def test_private_barrier_discard_scrubs_without_history(
        self,
        service,
        conn_id,
        runtime_config,
    ):
        nonce = "56" * 32
        runtime_config.transcript_barrier_version = 1
        runtime_config.transcript_barrier_nonce = nonce
        completed = service.dispatch_pipeline_event(
            conn_id,
            TranscriptBarrierCompletedEvent(transcript="ordinary non-identity turn"),
        )[0]
        assert isinstance(completed, TranscriptBarrierCompletedServerEvent)

        resolved = service.handle_transcript_barrier_resolve(
            conn_id,
            TranscriptBarrierResolveEvent.model_validate(
                {
                    "type": "reachy.transcript_barrier.resolve",
                    "version": 1,
                    "nonce": nonce,
                    "sequence": completed.sequence,
                    "input_item_id": completed.item_id,
                    "action": "discard",
                }
            ),
        )

        assert len(resolved) == 1
        assert isinstance(resolved[0], TranscriptBarrierResolvedServerEvent)
        assert resolved[0].action == "discarded"
        assert resolved[0].replacement_item_id is None
        assert runtime_config.transcript_barrier_pending is False
        assert runtime_config.transcript_barrier_pending_transcript is None
        assert runtime_config.chat.buffer == []

    @pytest.mark.parametrize("violation", ["wrong_text", "response_create", "second_final"])
    def test_private_barrier_pending_protocol_violations_poison_and_scrub(
        self,
        service,
        conn_id,
        runtime_config,
        violation,
    ):
        nonce = "78" * 32
        transcript = "private canary transcript"
        runtime_config.transcript_barrier_version = 1
        runtime_config.transcript_barrier_nonce = nonce
        completed = service.dispatch_pipeline_event(
            conn_id,
            TranscriptBarrierCompletedEvent(transcript=transcript),
        )[0]
        assert isinstance(completed, TranscriptBarrierCompletedServerEvent)

        if violation == "wrong_text":
            events = service.handle_transcript_barrier_resolve(
                conn_id,
                TranscriptBarrierResolveEvent.model_validate(
                    {
                        "type": "reachy.transcript_barrier.resolve",
                        "version": 1,
                        "nonce": nonce,
                        "sequence": completed.sequence,
                        "input_item_id": completed.item_id,
                        "action": "accept",
                        "item": {
                            "id": "msg_wrong",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "different"}],
                        },
                    }
                ),
            )
        elif violation == "response_create":
            result = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
            assert result is not None
            events = [result]
        else:
            events = service.dispatch_pipeline_event(
                conn_id,
                TranscriptBarrierCompletedEvent(transcript="second final"),
            )

        assert events[0].type in {"error", "reachy.transcript_barrier.failed"}
        assert runtime_config.transcript_barrier_failed is True
        assert runtime_config.transcript_barrier_pending is False
        assert runtime_config.transcript_barrier_pending_transcript is None
        assert runtime_config.chat.buffer == []

    def test_private_final_rejects_an_active_response_and_scrubs_deferred_items(
        self,
        service,
        conn_id,
        runtime_config,
    ):
        runtime_config.transcript_barrier_version = 1
        runtime_config.transcript_barrier_nonce = "79" * 32
        service.response._ensure_response(conn_id)
        deferred = ConversationItemCreateEvent.model_validate(
            {
                "type": "conversation.item.create",
                "item": {
                    "id": "msg_deferred_private_boundary",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "must not mutate history"}],
                },
            }
        )
        assert service.handle_conversation_item_create(conn_id, deferred) == []

        events = service.dispatch_pipeline_event(
            conn_id,
            TranscriptBarrierCompletedEvent(transcript="private final"),
        )

        assert isinstance(events[0], TranscriptBarrierFailedServerEvent)
        assert runtime_config.transcript_barrier_failed is True
        assert runtime_config.transcript_barrier_pending is False
        assert service._state(conn_id).deferred_items == []
        assert runtime_config.chat.buffer == []

    def test_response_finish_cannot_flush_deferred_history_while_private_final_is_pending(
        self,
        service,
        conn_id,
        runtime_config,
    ):
        runtime_config.transcript_barrier_version = 1
        runtime_config.transcript_barrier_nonce = "80" * 32
        completed = service.dispatch_pipeline_event(
            conn_id,
            TranscriptBarrierCompletedEvent(transcript="pending private final"),
        )[0]
        assert isinstance(completed, TranscriptBarrierCompletedServerEvent)
        st = service._state(conn_id)
        st.in_response = True
        st.current_response_id = "resp_stale"
        st.current_item_id = "item_stale"
        st.deferred_items.append(
            ConversationItemCreateEvent.model_validate(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "id": "msg_deferred_stale",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "deferred mutation"}],
                    },
                }
            ).item
        )

        service.finish_response(conn_id, status="cancelled")

        assert runtime_config.transcript_barrier_pending is True
        assert runtime_config.chat.buffer == []
        assert len(st.deferred_items) == 1

    def test_private_replacement_item_id_cannot_be_reused(self, service, conn_id, runtime_config):
        nonce = "81" * 32
        runtime_config.transcript_barrier_version = 1
        runtime_config.transcript_barrier_nonce = nonce

        def resolve(transcript: str):
            completed = service.dispatch_pipeline_event(
                conn_id,
                TranscriptBarrierCompletedEvent(transcript=transcript),
            )[0]
            assert isinstance(completed, TranscriptBarrierCompletedServerEvent)
            return service.handle_transcript_barrier_resolve(
                conn_id,
                TranscriptBarrierResolveEvent.model_validate(
                    {
                        "type": "reachy.transcript_barrier.resolve",
                        "version": 1,
                        "nonce": nonce,
                        "sequence": completed.sequence,
                        "input_item_id": completed.item_id,
                        "action": "accept",
                        "item": {
                            "id": "msg_private_reused",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": transcript}],
                        },
                    }
                ),
            )

        first = resolve("first private final")
        second = resolve("second private final")

        assert first[-1].type == "reachy.transcript_barrier.resolved"
        assert second[0].type == "error"
        assert runtime_config.transcript_barrier_failed is True
        assert len(runtime_config.chat.buffer) == 1

    def test_private_barrier_whitespace_final_is_content_free_and_inert(
        self,
        service,
        conn_id,
        runtime_config,
        text_prompt_queue,
    ):
        runtime_config.transcript_barrier_version = 1
        runtime_config.transcript_barrier_nonce = "ef" * 32
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())

        events = service.dispatch_pipeline_event(conn_id, TranscriptBarrierDiscardedEvent())

        assert len(events) == 1
        assert isinstance(events[0], TranscriptBarrierDiscardedServerEvent)
        assert events[0].sequence == 1
        assert not hasattr(events[0], "transcript")
        assert runtime_config.chat.buffer == []
        assert text_prompt_queue.empty()

    def test_private_barrier_oversize_final_poisoned_without_echoing_content(
        self,
        service,
        conn_id,
        runtime_config,
        text_prompt_queue,
    ):
        runtime_config.transcript_barrier_version = 1
        runtime_config.transcript_barrier_nonce = "12" * 32
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())

        events = service.dispatch_pipeline_event(
            conn_id,
            TranscriptBarrierCompletedEvent(transcript="x" * (TRANSCRIPT_BARRIER_MAX_CHARS + 1)),
        )

        assert len(events) == 1
        assert isinstance(events[0], TranscriptBarrierFailedServerEvent)
        assert events[0].reason == "transcript_too_large"
        assert runtime_config.transcript_barrier_failed is True
        assert runtime_config.chat.buffer == []
        assert text_prompt_queue.empty()

    def test_create_response_false_stores_transcript_for_explicit_response(
        self,
        service,
        conn_id,
        runtime_config,
        text_prompt_queue,
    ):
        from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

        runtime_config.session.audio.input.turn_detection = ServerVad(
            type="server_vad",
            create_response=False,
        )

        events = service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(
                transcript="hello world",
                language_code="en",
                turn_id="turn_1",
                turn_revision=2,
                speech_stopped_at_s=123.0,
            ),
        )

        assert len(events) == 1
        assert isinstance(events[0], ConversationItemInputAudioTranscriptionCompletedEvent)
        assert text_prompt_queue.empty()
        assert service._state(conn_id).response_pending is False
        assert runtime_config.chat.buffer[-1].content[0].text == "hello world"

        result = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))

        assert isinstance(result, ResponseCreatedEvent)
        request = text_prompt_queue.get_nowait()
        assert isinstance(request, GenerateResponseRequest)
        assert request.turn_id == "turn_1"
        assert request.turn_revision == 2
        assert request.speech_stopped_at_s == 123.0
        assert request.runtime_config.chat.buffer[-1].content[0].text == "hello world"

    def test_revised_transcription_replaces_speculative_user_message(self, runtime_config, should_listen):
        text_prompt_queue = Queue()
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(
            text_prompt_queue=text_prompt_queue,
            should_listen=should_listen,
            speculative_turns=tracker,
        )
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config

        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=1.0, turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="hello", turn_id="turn_1", turn_revision=0),
        )

        tracker.observe("turn_1", 1)
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=1, reopened=True),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=2.0, turn_id="turn_1", turn_revision=1),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="hello again", turn_id="turn_1", turn_revision=1),
        )

        user_items = [item for item in runtime_config.chat.buffer if getattr(item, "role", None) == "user"]
        assert len(user_items) == 1
        assert user_items[0].content[0].text == "hello again"
        first_req = text_prompt_queue.get_nowait()
        second_req = text_prompt_queue.get_nowait()
        assert first_req.turn_revision == 0
        assert second_req.turn_revision == 1
        assert service._state(conn_id).response_usage.audio_duration_s == 2.0
        service.unregister(conn_id)

    def test_empty_revised_transcription_removes_speculative_user_message(self, runtime_config, should_listen):
        text_prompt_queue = Queue()
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(
            text_prompt_queue=text_prompt_queue,
            should_listen=should_listen,
            speculative_turns=tracker,
        )
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config

        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=1.0, turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="hello", turn_id="turn_1", turn_revision=0),
        )

        tracker.observe("turn_1", 1)
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=1, reopened=True),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=2.0, turn_id="turn_1", turn_revision=1),
        )
        empty_events = service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="", turn_id="turn_1", turn_revision=1),
        )

        user_items = [item for item in runtime_config.chat.buffer if getattr(item, "role", None) == "user"]
        assert user_items == []
        first_req = text_prompt_queue.get_nowait()
        assert first_req.turn_revision == 0
        assert text_prompt_queue.empty()
        assert any(isinstance(event, ResponseCreatedEvent) for event in empty_events)
        assert any(isinstance(event, ResponseDoneEvent) for event in empty_events)
        assert not service._state(conn_id).in_response
        assert not service._state(conn_id).response_pending
        assert service.total_usage.audio_duration_s == 2.0
        assert service.total_usage.responses_cancelled == 1
        service.unregister(conn_id)

    def test_empty_active_revision_closes_lane_before_the_next_turn(self, runtime_config, should_listen):
        text_prompt_queue = Queue()
        tracker = SpeculativeTurnTracker()
        cancel_scope = CancelScope()
        service = RealtimeService(
            text_prompt_queue=text_prompt_queue,
            should_listen=should_listen,
            speculative_turns=tracker,
            cancel_scope=cancel_scope,
        )
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config

        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_empty", turn_revision=0, interrupt_response=False),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="false start", turn_id="turn_empty", turn_revision=0),
        )
        assert text_prompt_queue.get_nowait().turn_revision == 0
        service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(text="stale reply", turn_id="turn_empty", turn_revision=0),
        )
        assert service._state(conn_id).in_response

        tracker.observe("turn_empty", 1)
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(
                turn_id="turn_empty",
                turn_revision=1,
                reopened=True,
                interrupt_response=False,
            ),
        )
        terminal = service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="", turn_id="turn_empty", turn_revision=1),
        )

        assert any(isinstance(event, ResponseDoneEvent) and event.response.status == "cancelled" for event in terminal)
        assert not service._state(conn_id).in_response
        assert not service._state(conn_id).response_pending
        assert cancel_scope.generation == 1

        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_next", turn_revision=0, interrupt_response=False),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="real question", turn_id="turn_next", turn_revision=0),
        )
        successor = text_prompt_queue.get_nowait()
        assert successor.turn_id == "turn_next"
        assert service._state(conn_id).response_pending
        service.unregister(conn_id)

    def test_empty_first_revision_tracks_audio_for_later_nonempty_reopen(self, runtime_config, should_listen):
        text_prompt_queue = Queue()
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(
            text_prompt_queue=text_prompt_queue,
            should_listen=should_listen,
            speculative_turns=tracker,
        )
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config

        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=1.0, turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="", turn_id="turn_1", turn_revision=0),
        )

        tracker.observe("turn_1", 1)
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=1, reopened=True),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=2.0, turn_id="turn_1", turn_revision=1),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="hello again", turn_id="turn_1", turn_revision=1),
        )

        user_items = [item for item in runtime_config.chat.buffer if getattr(item, "role", None) == "user"]
        assert len(user_items) == 1
        assert user_items[0].content[0].text == "hello again"
        req = text_prompt_queue.get_nowait()
        assert req.turn_revision == 1
        assert text_prompt_queue.empty()
        assert service._state(conn_id).response_usage.audio_duration_s == 2.0
        service.unregister(conn_id)

    def test_stale_transcription_revision_is_ignored(self, runtime_config, should_listen):
        text_prompt_queue = Queue()
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(
            text_prompt_queue=text_prompt_queue,
            should_listen=should_listen,
            speculative_turns=tracker,
        )
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config
        tracker.observe("turn_1", 1)

        events = service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="stale", turn_id="turn_1", turn_revision=0),
        )

        assert events == []
        assert runtime_config.chat.buffer == []
        assert text_prompt_queue.empty()
        service.unregister(conn_id)

    def test_stale_assistant_text_dropped_after_unanswered_reopen(self, runtime_config, should_listen):
        text_prompt_queue = Queue()
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(
            text_prompt_queue=text_prompt_queue,
            should_listen=should_listen,
            speculative_turns=tracker,
        )
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config

        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=1.0, turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="hello", turn_id="turn_1", turn_revision=0),
        )

        # The VAD reopens an unanswered turn past the grace window through the
        # same candidate protocol it uses for an in-grace reopen.
        candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
        assert candidate_revision == 1
        assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)

        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(text="stale", turn_id="turn_1", turn_revision=0),
        )

        assert events == []
        assert service._state(conn_id).current_response_id is None

        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=1, reopened=True),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=2.5, turn_id="turn_1", turn_revision=1),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="hello and more", turn_id="turn_1", turn_revision=1),
        )

        user_items = [item for item in runtime_config.chat.buffer if getattr(item, "role", None) == "user"]
        assert len(user_items) == 1
        assert user_items[0].content[0].text == "hello and more"
        service.unregister(conn_id)

    # -- response_failed --

    def test_zero_output_success_emits_created_and_done_for_pending_response(self):
        prompt_queue = Queue()
        service = RealtimeService(
            text_prompt_queue=prompt_queue,
            speculative_turns=SpeculativeTurnTracker(),
            cancel_scope=CancelScope(),
        )
        conn_id = service.register()
        for turn_id, transcript in (("turn_empty", "empty response"), ("turn_successor", "next response")):
            service.dispatch_pipeline_event(
                conn_id,
                SpeechStartedEvent(turn_id=turn_id, turn_revision=0, interrupt_response=False),
            )
            service.dispatch_pipeline_event(
                conn_id,
                TranscriptionCompletedEvent(transcript=transcript, turn_id=turn_id, turn_revision=0),
            )
        request = prompt_queue.get_nowait()
        assert request.turn_id == "turn_empty"

        events = service.finish_response(conn_id)

        assert [event.type for event in events] == ["response.created", "response.done"]
        assert events[0].response.id == events[1].response.id
        assert events[1].response.status == "completed"
        state = service._state(conn_id)
        successor = prompt_queue.get_nowait()
        assert successor.turn_id == "turn_successor"
        assert not state.in_response and state.response_pending
        assert state.pending_response_request is successor
        service.unregister(conn_id)

    def test_response_failed_emits_error_and_failed_done(self, service, conn_id):
        service.response._ensure_response(conn_id)
        generation = service.cancel_scope.generation
        service._state(conn_id).active_response_cancel_generation = generation
        events = service.dispatch_pipeline_event(
            conn_id,
            ResponseFailedEvent(message="input must not be empty", cancel_generation=generation),
        )
        # A top-level error event carries the reason (response.done can't). The
        # response stays active until its normal terminal sentinel reaches the
        # audio side, preventing that sentinel from closing a successor.
        err = events[0]
        assert isinstance(err, RealtimeErrorEvent)
        assert err.error.message == "input must not be empty"
        assert err.error.type == "response_failed"
        assert not any(isinstance(event, ResponseDoneEvent) for event in events)
        assert service._state(conn_id).in_response is True

        terminal_events = service.finish_response(conn_id)
        done = [e for e in terminal_events if isinstance(e, ResponseDoneEvent)]
        assert len(done) == 1
        assert done[0].response.status == "failed"
        # Slot released so the next response is not locked out.
        assert service._state(conn_id).in_response is False

    def test_response_failure_terminal_promotes_successor_exactly_once(self):
        tracker = SpeculativeTurnTracker()
        prompt_queue = Queue()
        service = RealtimeService(
            text_prompt_queue=prompt_queue,
            speculative_turns=tracker,
            cancel_scope=CancelScope(),
        )
        conn_id = service.register()
        failed_request = None
        for turn_id, transcript in (("turn_failed", "first"), ("turn_successor", "second")):
            service.dispatch_pipeline_event(
                conn_id,
                SpeechStartedEvent(turn_id=turn_id, turn_revision=0, interrupt_response=False),
            )
            service.dispatch_pipeline_event(
                conn_id,
                TranscriptionCompletedEvent(transcript=transcript, turn_id=turn_id, turn_revision=0),
            )
            if turn_id == "turn_failed":
                failed_request = prompt_queue.get_nowait()
                assert failed_request.turn_id == turn_id

        assert failed_request is not None

        state = service._state(conn_id)
        assert state.response_pending
        assert [request.turn_id for request in state.deferred_response_requests] == ["turn_successor"]

        failure_events = service.dispatch_pipeline_event(
            conn_id,
            ResponseFailedEvent(
                message="provider failed",
                turn_id="turn_failed",
                turn_revision=0,
                cancel_generation=failed_request.cancel_generation,
            ),
        )

        assert len(failure_events) == 2
        assert isinstance(failure_events[0], ResponseCreatedEvent)
        assert isinstance(failure_events[1], RealtimeErrorEvent)
        created_response_id = failure_events[0].response.id
        assert state.in_response
        assert state.active_response_turn_id == "turn_failed"
        assert state.pending_response_turn_id == "turn_successor"
        assert prompt_queue.empty()

        terminal_events = service.finish_response(conn_id)

        done = next(event for event in terminal_events if isinstance(event, ResponseDoneEvent))
        assert done.response.status == "failed"
        assert done.response.id == created_response_id
        successor = prompt_queue.get_nowait()
        assert successor.turn_id == "turn_successor"
        assert state.response_pending
        assert state.pending_response_request is successor
        service.unregister(conn_id)

    def test_late_failure_cannot_activate_or_poison_promoted_successor(self):
        tracker = SpeculativeTurnTracker()
        prompt_queue = Queue()
        service = RealtimeService(
            text_prompt_queue=prompt_queue,
            speculative_turns=tracker,
            cancel_scope=CancelScope(),
        )
        conn_id = service.register()
        old_request = None
        for turn_id in ("turn_old", "turn_successor"):
            service.dispatch_pipeline_event(
                conn_id,
                SpeechStartedEvent(turn_id=turn_id, turn_revision=0, interrupt_response=False),
            )
            service.dispatch_pipeline_event(
                conn_id,
                TranscriptionCompletedEvent(transcript=turn_id, turn_id=turn_id, turn_revision=0),
            )
            if turn_id == "turn_old":
                old_request = prompt_queue.get_nowait()
                service.dispatch_pipeline_event(
                    conn_id,
                    AssistantTextEvent(text="old reply", turn_id=turn_id, turn_revision=0),
                )

        assert old_request is not None

        service.finish_response(conn_id)
        successor = prompt_queue.get_nowait()
        state = service._state(conn_id)
        assert successor.turn_id == "turn_successor"
        assert state.response_pending and not state.in_response

        events = service.dispatch_pipeline_event(
            conn_id,
            ResponseFailedEvent(
                message="late old failure",
                turn_id="turn_old",
                turn_revision=0,
                cancel_generation=old_request.cancel_generation,
            ),
        )

        assert events == []
        assert state.response_pending and not state.in_response
        assert state.pending_response_turn_id == "turn_successor"
        assert not state.response_failure_pending
        service.unregister(conn_id)

    def test_idless_late_cancelled_failure_cannot_poison_successor(self):
        prompt_queue = Queue()
        cancel_scope = CancelScope()
        service = RealtimeService(
            text_prompt_queue=prompt_queue,
            speculative_turns=SpeculativeTurnTracker(),
            cancel_scope=cancel_scope,
        )
        conn_id = service.register()
        created = service.handle_response_create(
            conn_id,
            ResponseCreateEvent(
                type="response.create",
                response={
                    "input": [
                        {
                            "id": "msg_explicit_old",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "old explicit"}],
                        }
                    ]
                },
            ),
        )
        assert isinstance(created, ResponseCreatedEvent)
        old_request = prompt_queue.get_nowait()
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_successor", turn_revision=0, interrupt_response=False),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(
                transcript="successor",
                turn_id="turn_successor",
                turn_revision=0,
            ),
        )

        service.handle_conversation_item_delete(
            conn_id,
            ConversationItemDeleteEvent(type="conversation.item.delete", item_id="msg_explicit_old"),
        )
        successor = prompt_queue.get_nowait()
        state = service._state(conn_id)
        assert successor.turn_id == "turn_successor"
        assert state.response_pending and not state.in_response
        assert successor.cancel_generation != old_request.cancel_generation

        events = service.dispatch_pipeline_event(
            conn_id,
            ResponseFailedEvent(
                message="late id-less old failure",
                cancel_generation=old_request.cancel_generation,
            ),
        )

        assert events == []
        assert state.response_pending and not state.in_response
        assert state.pending_response_turn_id == "turn_successor"
        assert not state.response_failure_pending
        service.unregister(conn_id)

    def test_response_failed_without_active_response_is_noop(self, service, conn_id):
        # No active response (e.g. already closed): nothing to fail, emit nothing.
        events = service.dispatch_pipeline_event(
            conn_id,
            ResponseFailedEvent(message="too late"),
        )
        assert events == []

    def test_private_response_failure_redacts_pipeline_error(self, service, conn_id, caplog):
        canary = "PRIVATE_PIPELINE_FAILURE_CANARY"
        state = service._state(conn_id)
        state.runtime_config.transcript_barrier_version = 1
        state.runtime_config.transcript_barrier_nonce = "ab" * 32
        service.response._ensure_response(conn_id)
        generation = service.cancel_scope.generation
        state.active_response_cancel_generation = generation

        with caplog.at_level(logging.INFO):
            events = service.dispatch_pipeline_event(
                conn_id,
                ResponseFailedEvent(message=canary, cancel_generation=generation),
            )

        error = events[0]
        assert isinstance(error, RealtimeErrorEvent)
        assert error.error.message == "Private response failed."
        assert canary not in caplog.text

    # -- unknown --

    def test_unknown_type_returns_empty(self, service, conn_id):
        from speech_to_speech.pipeline.events import PipelineEvent

        events = service.dispatch_pipeline_event(conn_id, PipelineEvent(type="something_else"))
        assert events == []


# ===================================================================
# Error helper
# ===================================================================


class TestMakeError:
    def test_make_error(self, service):
        err = service.make_error("oops", "my_error")
        assert isinstance(err, RealtimeErrorEvent)
        assert err.error.message == "oops"
        assert err.error.type == "my_error"
        assert err.event_id.startswith("event_")


# ===================================================================
# ID and state management
# ===================================================================


class TestIdAndStateManagement:
    def test_last_item_id_tracks_all_items(self, service, conn_id):
        st = service._state(conn_id)
        assert st.last_item_id is None

        # 1) speech_started sets last_item_id via dispatch_pipeline_event
        events = service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        input_id = events[0].item_id
        assert st.last_item_id == input_id

        # 2) assistant_text sets last_item_id via dispatch_pipeline_event
        events = service.dispatch_pipeline_event(conn_id, AssistantTextEvent(text="hi"))
        output_id = st.current_item_id
        assert st.last_item_id == output_id

        # 3) handle_conversation_item_create updates last_item_id
        service.response._end_response(conn_id)
        evt = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "id": "msg_manual",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "x"}],
            },
        )
        events = service.handle_conversation_item_create(conn_id, evt)
        assert st.last_item_id == events[0].item.id
        assert events[0].previous_item_id == output_id

    def test_content_index_resets_on_new_item(self, service, conn_id):
        service.response._start_item(conn_id)
        assert service.response._next_content_index(conn_id) == 0
        assert service.response._next_content_index(conn_id) == 1

        service.response._start_item(conn_id)
        assert service.response._next_content_index(conn_id) == 0

        service.response._ensure_response(conn_id)
        assert service.response._next_content_index(conn_id) == 0
        assert service.response._next_content_index(conn_id) == 1

        service.response._end_response(conn_id)
        service.response._ensure_response(conn_id)
        assert service.response._next_content_index(conn_id) == 0


# ===================================================================
# interrupt_response_enabled property
# ===================================================================


class TestInterruptResponseEnabled:
    def test_default_true_when_no_turn_detection(self, runtime_config):
        runtime_config.session.audio.input.turn_detection = None
        assert runtime_config.interrupt_response_enabled is True

    def test_true_when_server_vad_interrupt_true(self, runtime_config):
        from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

        runtime_config.session.audio.input.turn_detection = ServerVad(
            type="server_vad",
            interrupt_response=True,
        )
        assert runtime_config.interrupt_response_enabled is True

    def test_false_when_server_vad_interrupt_false(self, runtime_config):
        from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

        runtime_config.session.audio.input.turn_detection = ServerVad(
            type="server_vad",
            interrupt_response=False,
        )
        assert runtime_config.interrupt_response_enabled is False

    def test_default_true_when_server_vad_interrupt_none(self, runtime_config):
        from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

        runtime_config.session.audio.input.turn_detection = ServerVad(
            type="server_vad",
            interrupt_response=None,
        )
        assert runtime_config.interrupt_response_enabled is True

    def test_reads_dict_turn_detection(self, runtime_config):
        runtime_config.session.audio.input.turn_detection = {
            "type": "server_vad",
            "interrupt_response": False,
        }
        assert runtime_config.interrupt_response_enabled is False

    def test_dict_defaults_to_true(self, runtime_config):
        runtime_config.session.audio.input.turn_detection = {
            "type": "server_vad",
        }
        assert runtime_config.interrupt_response_enabled is True


# ===================================================================
# create_response_enabled property
# ===================================================================


class TestCreateResponseEnabled:
    def test_default_true_when_no_turn_detection(self, runtime_config):
        runtime_config.session.audio.input.turn_detection = None
        assert runtime_config.create_response_enabled is True

    def test_true_when_server_vad_create_response_true(self, runtime_config):
        from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

        runtime_config.session.audio.input.turn_detection = ServerVad(
            type="server_vad",
            create_response=True,
        )
        assert runtime_config.create_response_enabled is True

    def test_false_when_server_vad_create_response_false(self, runtime_config):
        from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

        runtime_config.session.audio.input.turn_detection = ServerVad(
            type="server_vad",
            create_response=False,
        )
        assert runtime_config.create_response_enabled is False

    def test_default_true_when_server_vad_create_response_none(self, runtime_config):
        from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

        runtime_config.session.audio.input.turn_detection = ServerVad(
            type="server_vad",
            create_response=None,
        )
        assert runtime_config.create_response_enabled is True

    def test_reads_dict_turn_detection(self, runtime_config):
        runtime_config.session.audio.input.turn_detection = {
            "type": "server_vad",
            "create_response": False,
        }
        assert runtime_config.create_response_enabled is False

    def test_dict_defaults_to_true(self, runtime_config):
        runtime_config.session.audio.input.turn_detection = {
            "type": "server_vad",
        }
        assert runtime_config.create_response_enabled is True


# ===================================================================
# Usage metrics tracking (tokens + audio duration)
# ===================================================================


class TestUsageMetricsTracking:
    # -- token accumulation --

    def test_token_usage_accumulates_in_conn_state(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=10, output_tokens=20),
        )
        usage = service._state(conn_id).response_usage
        assert usage.input_tokens == 10
        assert usage.output_tokens == 20

    def test_token_usage_accumulates_multiple(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=5, output_tokens=10),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=3, output_tokens=7),
        )
        usage = service._state(conn_id).response_usage
        assert usage.input_tokens == 8
        assert usage.output_tokens == 17

    def test_late_cancelled_generation_usage_cannot_charge_successor(self):
        cancel_scope = CancelScope()
        service = RealtimeService(text_prompt_queue=Queue(), cancel_scope=cancel_scope)
        conn_id = service.register()
        service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        assert service._state(conn_id).active_response_cancel_generation == 0

        cancel_scope.cancel()
        service.handle_response_cancel(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=100, output_tokens=50, cancel_generation=0),
        )
        usage = service._state(conn_id).response_usage
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

        service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        assert service._state(conn_id).active_response_cancel_generation == 1

        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=100, output_tokens=50, cancel_generation=0),
        )
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=7, output_tokens=3, cancel_generation=1),
        )
        assert usage.input_tokens == 7
        assert usage.output_tokens == 3
        service.unregister(conn_id)

    def test_later_turn_usage_same_generation_cannot_charge_active_response(self):
        cancel_scope = CancelScope()
        service = RealtimeService(text_prompt_queue=Queue(), cancel_scope=cancel_scope)
        conn_id = service.register()
        service.response.resume_pending_request(
            conn_id,
            GenerateResponseRequest(
                runtime_config=service._state(conn_id).runtime_config,
                turn_id="turn_active",
                turn_revision=0,
                cancel_generation=cancel_scope.generation,
            ),
            enqueue=False,
        )
        service.response._ensure_response(conn_id)

        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(
                input_tokens=100,
                output_tokens=50,
                turn_id="turn_next",
                turn_revision=0,
                cancel_generation=cancel_scope.generation,
            ),
        )

        usage = service._state(conn_id).response_usage
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        service.unregister(conn_id)

    def test_token_usage_emits_no_events(self, service, conn_id):
        events = service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=10, output_tokens=20),
        )
        assert events == []

    def test_response_done_reflects_token_usage(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=100, output_tokens=50),
        )
        events = service.finish_response(conn_id)
        done_evt = next(event for event in events if isinstance(event, ResponseDoneEvent))
        assert isinstance(done_evt, ResponseDoneEvent)
        assert done_evt.response.usage.input_tokens == 100
        assert done_evt.response.usage.output_tokens == 50
        assert done_evt.response.usage.total_tokens == 150

    def test_response_created_has_zero_tokens(self, service, conn_id):
        """ResponseCreatedEvent is emitted before any tokens are produced."""
        events = service.encode_audio_chunk(conn_id, _pcm_bytes(256))
        created_evt = events[0]
        assert isinstance(created_evt, ResponseCreatedEvent)
        assert created_evt.response.usage.input_tokens == 0
        assert created_evt.response.usage.output_tokens == 0
        assert created_evt.response.usage.total_tokens == 0

    def test_end_response_rolls_into_global(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=10, output_tokens=20),
        )
        service.response._end_response(conn_id)
        assert service.total_usage.input_tokens == 10
        assert service.total_usage.output_tokens == 20
        usage = service._state(conn_id).response_usage
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_multiple_responses_accumulate_global(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=10, output_tokens=20),
        )
        service.response._end_response(conn_id)

        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=5, output_tokens=15),
        )
        service.response._end_response(conn_id)

        assert service.total_usage.input_tokens == 15
        assert service.total_usage.output_tokens == 35

    def test_unregister_rolls_partial_tokens_into_global(self, service):
        cid = service.register()
        service.response._ensure_response(cid)
        service.dispatch_pipeline_event(
            cid,
            TokenUsageEvent(input_tokens=7, output_tokens=3),
        )
        service.unregister(cid)
        assert service.total_usage.input_tokens == 7
        assert service.total_usage.output_tokens == 3

    def test_unregister_without_active_response_no_leak(self, service):
        cid = service.register()
        service.unregister(cid)
        assert service.total_usage.input_tokens == 0
        assert service.total_usage.output_tokens == 0

    def test_finish_response_resets_per_response_tokens(self, service, conn_id):
        """After finish_response, per-response counters are zero."""
        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=50, output_tokens=25),
        )
        service.finish_response(conn_id)
        usage = service._state(conn_id).response_usage
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert service.total_usage.input_tokens == 50
        assert service.total_usage.output_tokens == 25

    # -- audio duration accumulation --

    def test_transcription_completed_accumulates_duration(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent(duration_s=2.5))
        service.dispatch_pipeline_event(conn_id, TranscriptionCompletedEvent(transcript="hi"))
        assert service._state(conn_id).response_usage.audio_duration_s == 2.5

    def test_multiple_transcriptions_accumulate_duration(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent(duration_s=1.0))
        service.dispatch_pipeline_event(conn_id, TranscriptionCompletedEvent(transcript="a"))

        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent(duration_s=2.0))
        service.dispatch_pipeline_event(conn_id, TranscriptionCompletedEvent(transcript="b"))

        assert service._state(conn_id).response_usage.audio_duration_s == 3.0

    def test_end_response_rolls_duration_into_global(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent(duration_s=4.0))
        service.dispatch_pipeline_event(conn_id, TranscriptionCompletedEvent(transcript="x"))
        service.response._ensure_response(conn_id)
        service.response._end_response(conn_id)
        assert service.total_usage.audio_duration_s == 4.0
        assert service._state(conn_id).response_usage.audio_duration_s == 0.0

    def test_unregister_rolls_duration_into_global(self, service):
        cid = service.register()
        service.dispatch_pipeline_event(cid, SpeechStartedEvent())
        service.dispatch_pipeline_event(cid, SpeechStoppedEvent(duration_s=1.5))
        service.dispatch_pipeline_event(cid, TranscriptionCompletedEvent(transcript="y"))
        service.unregister(cid)
        assert service.total_usage.audio_duration_s == 1.5

    # -- responses_completed / responses_cancelled --

    def test_responses_completed_increments(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.finish_response(conn_id)
        assert service.total_usage.responses_completed == 1
        assert service.total_usage.responses_cancelled == 0

    def test_responses_cancelled_increments(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.finish_response(conn_id, status="cancelled", reason="turn_detected")
        assert service.total_usage.responses_cancelled == 1
        assert service.total_usage.responses_completed == 0

    def test_multiple_responses_accumulate_status_counters(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.finish_response(conn_id)
        service.response._ensure_response(conn_id)
        service.finish_response(conn_id, status="cancelled", reason="client_cancelled")
        service.response._ensure_response(conn_id)
        service.finish_response(conn_id)
        assert service.total_usage.responses_completed == 2
        assert service.total_usage.responses_cancelled == 1

    # -- tool_calls --

    def test_tool_calls_increments(self, service, conn_id):
        service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(
                text="",
                tools=[
                    {"type": "function_call", "call_id": "c1", "name": "f1", "arguments": "{}"},
                    {"type": "function_call", "call_id": "c2", "name": "f2", "arguments": "{}"},
                ],
            ),
        )
        assert service._state(conn_id).response_usage.tool_calls == 2

    def test_tool_calls_rolls_into_global(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(
                text="",
                tools=[{"type": "function_call", "call_id": "c1", "name": "f1", "arguments": "{}"}],
            ),
        )
        service.finish_response(conn_id)
        assert service.total_usage.tool_calls == 1
        assert service._state(conn_id).response_usage.tool_calls == 0

    # -- connections --

    def test_connections_increments(self, service):
        assert service.total_usage.connections == 0
        cid1 = service.register()
        assert service.total_usage.connections == 1
        cid2 = service.register()
        assert service.total_usage.connections == 2
        service.unregister(cid1)
        service.unregister(cid2)

    # -- turns --

    def test_turns_increments(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        assert service._state(conn_id).response_usage.turns == 3

    def test_turns_rolls_into_global(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.response._ensure_response(conn_id)
        service.response._end_response(conn_id)
        assert service.total_usage.turns == 1
        assert service._state(conn_id).response_usage.turns == 0

    # -- errors_by_type --

    def test_errors_by_type_increments(self, service):
        service.make_error("msg", "type_a")
        service.make_error("msg", "type_a")
        service.make_error("msg", "type_b")
        assert service.total_usage.errors_by_type == {"type_a": 2, "type_b": 1}

    def test_total_errors_in_get_usage(self, service):
        service.make_error("msg", "type_a")
        service.make_error("msg", "type_b")
        usage = service.get_usage()
        assert usage["total_errors"] == 2
        assert usage["errors_by_type"] == {"type_a": 1, "type_b": 1}

    # -- get_usage --

    def test_get_usage(self, service, conn_id):
        # Speech cycle before response so speech_started doesn't cancel anything
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent(duration_s=3.0))
        service.dispatch_pipeline_event(conn_id, TranscriptionCompletedEvent(transcript="z"))

        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=10, output_tokens=20),
        )
        service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(
                text="hi",
                tools=[{"type": "function_call", "call_id": "c1", "name": "f1", "arguments": "{}"}],
            ),
        )
        service.finish_response(conn_id)
        service.make_error("oops", "some_error")
        usage = service.get_usage()
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 20
        assert usage["total_tokens"] == 30
        assert usage["audio_duration_s"] == 3.0
        assert usage["responses_completed"] == 1
        assert usage["responses_cancelled"] == 0
        assert usage["tool_calls"] == 1
        assert usage["turns"] == 1
        assert usage["connections"] >= 1
        assert usage["total_errors"] == 1
        assert usage["errors_by_type"] == {"some_error": 1}


# ===================================================================
# Chat image lifecycle
# ===================================================================


class TestChatImageLifecycle:
    """Tests for Chat.strip_images()."""

    def _make_chat(self):
        from speech_to_speech.LLM.chat import Chat

        return Chat(size=10)

    def _user_msg(self, *parts):
        from openai.types.realtime.realtime_conversation_item_user_message import (
            Content as UserContent,
        )
        from openai.types.realtime.realtime_conversation_item_user_message import (
            RealtimeConversationItemUserMessage,
        )

        content = []
        for p in parts:
            if p[0] == "text":
                content.append(UserContent(type="input_text", text=p[1]))
            elif p[0] == "image":
                content.append(UserContent(type="input_image", image_url=p[1]))
        return RealtimeConversationItemUserMessage(type="message", role="user", content=content)

    def test_strip_images_removes_image_parts(self):
        from speech_to_speech.LLM.chat import make_assistant_message

        chat = self._make_chat()
        chat.add_item(self._user_msg(("text", "What is this?"), ("image", "data:image/png;base64,abc")))
        chat.add_item(make_assistant_message("It's a cat."))
        chat.strip_images()
        user_msg = chat.buffer[0]
        assert len(user_msg.content) == 1
        assert user_msg.content[0].type == "input_text"
        assert user_msg.content[0].text == "What is this?"

    def test_strip_images_noop_on_text_only(self):
        from speech_to_speech.LLM.chat import make_assistant_message, make_user_message

        chat = self._make_chat()
        chat.add_item(make_user_message("hello"))
        chat.add_item(make_assistant_message("hi"))
        chat.strip_images()
        assert chat.buffer[0].content[0].text == "hello"
        assert chat.buffer[1].content[0].text == "hi"

    def test_strip_then_new_image_cycle(self):
        from speech_to_speech.LLM.chat import make_assistant_message

        chat = self._make_chat()
        chat.add_item(self._user_msg(("text", "look"), ("image", "old_url")))
        chat.add_item(make_assistant_message("I see it."))
        chat.strip_images()
        assert len(chat.buffer[0].content) == 1
        assert chat.buffer[0].content[0].type == "input_text"

        chat.add_item(self._user_msg(("text", "now this"), ("image", "new_url")))
        last_user = chat.buffer[-1]
        assert any(p.image_url == "new_url" for p in last_user.content)


# ===================================================================
# Chat tool call tracking
# ===================================================================


class TestChatToolCallTracking:
    """Tests for Chat._pending_tool_calls and append_tool_output."""

    def _make_chat(self, size=10):
        from speech_to_speech.LLM.chat import Chat

        return Chat(size=size)

    def _fc(self, call_id="call_1", name="f1"):
        from openai.types.realtime.realtime_conversation_item_function_call import (
            RealtimeConversationItemFunctionCall,
        )

        if not call_id.startswith("call_"):
            call_id = f"call_{call_id}"
        return RealtimeConversationItemFunctionCall(type="function_call", call_id=call_id, name=name, arguments="{}")

    def _fco(self, call_id="call_1"):
        from openai.types.realtime.realtime_conversation_item_function_call_output import (
            RealtimeConversationItemFunctionCallOutput,
        )

        if not call_id.startswith("call_"):
            call_id = f"call_{call_id}"
        return RealtimeConversationItemFunctionCallOutput(
            type="function_call_output", call_id=call_id, output='{"ok": true}'
        )

    def _user(self, text):
        from speech_to_speech.LLM.chat import make_user_message

        return make_user_message(text)

    def _assistant(self, text):
        from speech_to_speech.LLM.chat import make_assistant_message

        return make_assistant_message(text)

    def test_add_item_registers_pending_tool_call(self):
        chat = self._make_chat()
        fc = self._fc()
        chat.add_item(fc)
        assert "call_1" in chat._pending_tool_calls
        assert chat._pending_tool_calls["call_1"] is fc

    def test_append_tool_output_clears_pending(self):
        chat = self._make_chat()
        chat.add_item(self._fc())
        assert "call_1" in chat._pending_tool_calls
        chat.append_tool_output("call_1", self._fco())
        assert "call_1" not in chat._pending_tool_calls
        assert chat.buffer[-1].type == "function_call_output"

    def test_append_tool_output_reinjects_evicted_call(self):
        chat = self._make_chat(size=1)
        chat.add_item(self._user("hi"))
        chat.add_item(self._fc("call_x"))
        chat.add_item(self._assistant("ok"))
        chat.add_item(self._user("more"))
        chat.trim_if_needed()
        assert not any(getattr(e, "call_id", None) == "call_x" for e in chat.buffer)
        assert "call_x" in chat._pending_tool_calls

        chat.append_tool_output("call_x", self._fco("call_x"))
        assert chat._has_call_id_in_buffer("call_x")
        types = [e.type for e in chat.buffer]
        assert "function_call" in types
        assert "function_call_output" in types
        fc_idx = next(i for i, e in enumerate(chat.buffer) if e.type == "function_call")
        fco_idx = next(i for i, e in enumerate(chat.buffer) if e.type == "function_call_output")
        assert fc_idx < fco_idx

    def test_append_tool_output_rejects_unknown_call_id(self):
        from speech_to_speech.LLM.chat import ChatItemError

        chat = self._make_chat()
        with pytest.raises(ChatItemError, match="call_nope"):
            chat.append_tool_output("call_nope", self._fco("call_nope"))
        assert not any(getattr(e, "type", None) == "function_call_output" for e in chat.buffer)

    def test_copy_preserves_pending_tool_calls(self):
        chat = self._make_chat()
        chat.add_item(self._fc("call_a"))
        clone = chat.copy()
        assert "call_a" in clone._pending_tool_calls
        clone._pending_tool_calls.pop("call_a")
        assert "call_a" in chat._pending_tool_calls

    def test_reset_clears_pending_tool_calls(self):
        chat = self._make_chat()
        chat.add_item(self._fc())
        assert chat._pending_tool_calls
        chat.reset()
        assert chat._pending_tool_calls == {}
        assert chat.buffer == []

    # -- turn-based eviction --

    def test_eviction_removes_complete_turn(self):
        chat = self._make_chat(size=1)
        chat.add_item(self._user("turn 1"))
        chat.add_item(self._assistant("thinking"))
        chat.add_item(self._fc("c1"))
        chat.add_item(self._fco("c1"))
        chat.add_item(self._assistant("done"))
        assert len(chat.buffer) == 5

        chat.add_item(self._user("turn 2"))
        chat.trim_if_needed()
        from openai.types.realtime.realtime_conversation_item_user_message import (
            RealtimeConversationItemUserMessage,
        )

        user_msgs = [e for e in chat.buffer if isinstance(e, RealtimeConversationItemUserMessage)]
        assert len(user_msgs) == 1
        assert user_msgs[0].content[0].text == "turn 2"
        assert not any(getattr(e, "call_id", None) == "call_c1" and e.type == "function_call" for e in chat.buffer)

    def test_eviction_preserves_size_user_turns(self):
        from openai.types.realtime.realtime_conversation_item_user_message import (
            RealtimeConversationItemUserMessage,
        )

        chat = self._make_chat(size=2)
        chat.add_item(self._user("t1"))
        chat.add_item(self._assistant("r1"))
        chat.add_item(self._user("t2"))
        chat.add_item(self._assistant("let me check"))
        chat.add_item(self._fc("c2"))
        chat.add_item(self._fco("c2"))
        chat.add_item(self._assistant("here"))
        assert chat._user_turn_count == 2

        chat.add_item(self._user("t3"))
        chat.trim_if_needed()
        assert chat._user_turn_count == 2
        user_texts = [e.content[0].text for e in chat.buffer if isinstance(e, RealtimeConversationItemUserMessage)]
        assert user_texts == ["t2", "t3"]

    def test_pending_tool_calls_cleaned_after_reinjection(self):
        chat = self._make_chat(size=1)
        chat.add_item(self._user("hi"))
        chat.add_item(self._fc("call_z"))
        chat.add_item(self._user("bye"))
        assert "call_z" in chat._pending_tool_calls

        chat.append_tool_output("call_z", self._fco("call_z"))
        assert chat._has_call_id_in_buffer("call_z")


def test_home_assistant_guard_binds_first_session_contract_and_rejects_duplicate(
    service: RealtimeService,
    conn_id: str,
    runtime_config,
) -> None:
    ready = _activate_home_assistant_guard(service, conn_id)

    assert ready.nonce == "19" * 32
    assert ready.tool_count == 2
    assert runtime_config.home_assistant_guard_operational
    assert runtime_config.home_assistant_guard_tool_names == (
        "home_assistant__GetLiveContext",
        "get_local_time",
    )
    assert "reachy_home_assistant_guard" not in (runtime_config.session.model_extra or {})

    duplicate = service.handle_session_update(
        conn_id,
        SessionUpdateEvent(type="session.update", session={"type": "realtime", "instructions": "changed"}),
    )
    assert isinstance(duplicate, RealtimeErrorEvent)
    assert duplicate.error.type == "invalid_home_assistant_guard"
    assert runtime_config.home_assistant_guard_failed


def test_combined_private_handshake_is_atomic_and_ordered(
    service: RealtimeService,
    conn_id: str,
    runtime_config,
) -> None:
    service.home_assistant_guard_supported = True
    tools = _home_assistant_tools()[:1]
    digest, tool_count, _names = session_contract("private", tools)

    result = service.handle_session_update(
        conn_id,
        SessionUpdateEvent(
            type="session.update",
            session={
                "type": "realtime",
                "instructions": "private",
                "tools": tools,
                "reachy_private_transcript_barrier": {"version": 1, "nonce": "21" * 32},
                "reachy_home_assistant_guard": {
                    "version": 1,
                    "nonce": "22" * 32,
                    "session_contract_sha256": digest,
                    "tool_count": tool_count,
                },
            },
        ),
    )

    assert isinstance(result, list)
    assert [event.type for event in result] == [
        "reachy.transcript_barrier.ready",
        "reachy.home_assistant_guard.ready",
    ]
    assert runtime_config.transcript_barrier_operational
    assert runtime_config.home_assistant_guard_operational


def test_invalid_combined_private_handshake_emits_no_partial_ready(
    service: RealtimeService,
    conn_id: str,
    runtime_config,
) -> None:
    service.home_assistant_guard_supported = True
    tools = _home_assistant_tools()[:1]

    result = service.handle_session_update(
        conn_id,
        SessionUpdateEvent(
            type="session.update",
            session={
                "type": "realtime",
                "instructions": "private",
                "tools": tools,
                "reachy_private_transcript_barrier": {"version": 1, "nonce": "31" * 32},
                "reachy_home_assistant_guard": {
                    "version": 1,
                    "nonce": "32" * 32,
                    "session_contract_sha256": "00" * 32,
                    "tool_count": 1,
                },
            },
        ),
    )

    assert isinstance(result, RealtimeErrorEvent)
    assert runtime_config.transcript_barrier_enabled is False
    assert runtime_config.home_assistant_guard_enabled is False
    assert runtime_config.transcript_barrier_failed
    assert runtime_config.home_assistant_guard_failed


def test_required_guard_rejects_input_before_handshake(
    service: RealtimeService,
    conn_id: str,
    runtime_config,
    text_prompt_queue: Queue,
) -> None:
    service.home_assistant_guard_supported = True
    service.home_assistant_guard_required = True

    result = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))

    assert isinstance(result, RealtimeErrorEvent)
    assert result.error.type == "invalid_home_assistant_guard"
    assert runtime_config.home_assistant_guard_failed
    assert text_prompt_queue.empty()


def test_ordinary_session_does_not_infer_guard_from_tool_names(
    service: RealtimeService,
    conn_id: str,
    runtime_config,
) -> None:
    result = service.handle_session_update(
        conn_id,
        SessionUpdateEvent(
            type="session.update",
            session={
                "type": "realtime",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "home_assistant__GetLiveContext",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
        ),
    )

    assert result is None
    assert runtime_config.home_assistant_guard_enabled is False
    assert runtime_config.home_assistant_guard_failed is False


def test_guarded_response_override_must_match_or_explicitly_disable_tools(
    service: RealtimeService,
    conn_id: str,
    runtime_config,
    text_prompt_queue: Queue,
) -> None:
    _activate_home_assistant_guard(service, conn_id)
    changed = _home_assistant_tools()
    changed[0] = {**changed[0], "description": "Changed authority."}

    rejected = service.handle_response_create(
        conn_id,
        ResponseCreateEvent(type="response.create", response={"tools": changed}),
    )

    assert isinstance(rejected, RealtimeErrorEvent)
    assert rejected.error.type == "invalid_home_assistant_guard"
    assert runtime_config.home_assistant_guard_failed
    assert text_prompt_queue.empty()


def test_guarded_action_response_changed_instructions_rejects_bound_tools(
    service: RealtimeService,
    conn_id: str,
    runtime_config,
    text_prompt_queue: Queue,
) -> None:
    _activate_home_assistant_guard(service, conn_id)

    rejected = service.handle_response_create(
        conn_id,
        ResponseCreateEvent(type="response.create", response={"instructions": "Changed authority."}),
    )

    assert isinstance(rejected, RealtimeErrorEvent)
    assert rejected.error.type == "invalid_home_assistant_guard"
    assert runtime_config.home_assistant_guard_failed
    assert text_prompt_queue.empty()


def test_guarded_action_response_accepts_omitted_or_exact_instructions(
    service: RealtimeService,
    conn_id: str,
    text_prompt_queue: Queue,
) -> None:
    _activate_home_assistant_guard(service, conn_id)

    for response in ({"metadata": {"case": "omitted"}}, {"instructions": "Use exposed tools."}):
        created = service.handle_response_create(
            conn_id,
            ResponseCreateEvent(type="response.create", response=response),
        )
        assert isinstance(created, ResponseCreatedEvent)
        request = text_prompt_queue.get_nowait()
        assert isinstance(request, GenerateResponseRequest)
        service.finish_response(conn_id)


def test_guarded_tools_disabled_private_narration_is_allowed(
    service: RealtimeService,
    conn_id: str,
    text_prompt_queue: Queue,
) -> None:
    _activate_home_assistant_guard(service, conn_id)

    created = service.handle_response_create(
        conn_id,
        ResponseCreateEvent(
            type="response.create",
            response={
                "conversation": "none",
                "instructions": "Say one safe sentence.",
                "tool_choice": "none",
            },
        ),
    )

    assert isinstance(created, ResponseCreatedEvent)
    request = text_prompt_queue.get_nowait()
    assert isinstance(request, GenerateResponseRequest)
    assert request.response is not None and request.response.tool_choice == "none"


def test_guarded_invalid_function_output_poison_is_sticky(
    service: RealtimeService,
    conn_id: str,
    runtime_config,
    text_prompt_queue: Queue,
) -> None:
    _activate_home_assistant_guard(service, conn_id)
    event = ConversationItemCreateEvent(
        type="conversation.item.create",
        item={"type": "function_call_output", "output": '{"result":42}', "call_id": "call_unknown"},
    )

    events = service.handle_conversation_item_create(conn_id, event)
    later = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))

    assert len(events) == 1 and isinstance(events[0], RealtimeErrorEvent)
    assert events[0].error.type == "invalid_conversation_item"
    assert runtime_config.home_assistant_guard_failed
    assert runtime_config.chat.buffer == []
    assert isinstance(later, RealtimeErrorEvent)
    assert later.error.type == "private_session_failed"
    assert text_prompt_queue.empty()


@pytest.mark.parametrize(
    "event",
    [
        TranscriptionCompletedEvent(transcript="PRIVATE_STT_CANARY", turn_id="turn-race", turn_revision=0),
        AssistantTextEvent(text="PRIVATE_ASSISTANT_CANARY", turn_id="turn-race", turn_revision=0),
    ],
    ids=["transcription", "assistant-text"],
)
def test_guard_failure_after_stale_check_still_blocks_synchronous_pipeline_sinks(
    service: RealtimeService,
    conn_id: str,
    runtime_config,
    text_prompt_queue: Queue,
    monkeypatch: pytest.MonkeyPatch,
    event,
) -> None:
    """The final failure check and every synchronous sink share one lock."""
    _activate_home_assistant_guard(service, conn_id)

    def poison_during_stale_check(*_args, **_kwargs) -> bool:
        runtime_config.fail_home_assistant_guard()
        return False

    monkeypatch.setattr(service, "_is_stale_turn_event", poison_during_stale_check)

    emitted = service.dispatch_pipeline_event(conn_id, event)

    assert emitted == []
    assert runtime_config.home_assistant_guard_failed
    assert runtime_config.chat.buffer == []
    assert text_prompt_queue.empty()
