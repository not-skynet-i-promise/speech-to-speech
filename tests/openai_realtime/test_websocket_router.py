"""Integration tests for api.openai_realtime.websocket_router.

Uses Starlette's synchronous TestClient with WebSocket support to exercise
the full FastAPI app produced by ``create_app``. Each test gets a fresh
PipelineUnit pool (size 1, matching the single-session semantics of the
old tests) so there is no cross-test state.
"""

import asyncio
import base64
import logging
import time
from queue import Empty, Queue
from threading import Event as ThreadingEvent

import pytest
from openai.types.realtime.realtime_session_create_request import RealtimeSessionCreateRequest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketState

import speech_to_speech.api.openai_realtime.websocket_router as router_module
from speech_to_speech.api.openai_realtime.home_assistant_guard import (
    HOME_ASSISTANT_SELECTOR_REJECTED,
    session_contract,
)
from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.api.openai_realtime.service import CHUNK_SIZE_BYTES, RealtimeService
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.control import SESSION_END, PipelineControlMessage, is_control_message
from speech_to_speech.pipeline.events import (
    AssistantTextEvent,
    ResponseFailedEvent,
    SpeechStartedEvent,
    TokenUsageEvent,
    TranscriptBarrierCompletedEvent,
)
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END, AudioOutput

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def short_drain_timeout(monkeypatch):
    """Shorten the SESSION_END drain warning threshold so tests don't wait 10s.

    The constant only controls when the release task logs a warning about a
    slow-draining unit — there is no longer a release-anyway timeout, so the
    unit stays unavailable until SESSION_END actually drains.
    """
    monkeypatch.setattr(router_module, "SESSION_END_DRAIN_TIMEOUT_S", 0.1)


@pytest.fixture
def setup():
    """Return (app, service, input_queue, output_queue, text_output_queue,
    should_listen, stop_event, response_playing, cancel_scope) for a pool of one.

    There is no real handler chain in this fixture, so SESSION_END enqueued by
    the route handler on disconnect never reaches output_queue. Tests that need
    the release task to complete (verifying unit.session is cleared and the
    service unregistered) must drain SESSION_END themselves — see
    `_simulate_session_end_drain` below.
    """
    text_prompt_queue: Queue = Queue()
    should_listen = ThreadingEvent()
    should_listen.set()
    cancel_scope = CancelScope()
    service = RealtimeService(
        text_prompt_queue=text_prompt_queue,
        should_listen=should_listen,
        cancel_scope=cancel_scope,
        home_assistant_guard_supported=True,
    )
    assert service.verify_cancel_scope_wiring(cancel_scope, cancel_scope)
    input_queue: Queue = Queue()
    output_queue: Queue = Queue()
    text_output_queue: Queue = Queue()
    stop_event = ThreadingEvent()
    response_playing = ThreadingEvent()
    unit = PipelineUnit(
        index=0,
        service=service,
        cancel_scope=cancel_scope,
        should_listen=should_listen,
        response_playing=response_playing,
        input_queue=input_queue,
        output_queue=output_queue,
        text_output_queue=text_output_queue,
        text_prompt_queue=text_prompt_queue,
        handlers=[],
    )
    app = create_app(pool=[unit], stop_event=stop_event)
    return (
        app,
        service,
        input_queue,
        output_queue,
        text_output_queue,
        should_listen,
        stop_event,
        response_playing,
        cancel_scope,
    )


def _simulate_session_end_drain(input_queue: Queue, output_queue: Queue, timeout: float = 1.0) -> None:
    """Wait for SESSION_END to land in input_queue (from the route handler's
    release path) and forward it to output_queue — simulating the handler chain.
    The send loop will then observe SESSION_END and set `session.drained`,
    letting the release task complete (unregister + clear `unit.session`).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            item = input_queue.get(timeout=0.05)
        except Empty:
            continue
        if isinstance(item, PipelineControlMessage) and is_control_message(item, SESSION_END.kind):
            output_queue.put(item)
            return
    raise AssertionError("SESSION_END did not appear on input_queue within timeout")


def _pcm_bytes(n_samples: int) -> bytes:
    return b"\x00" * (n_samples * 2)


class _FakeWebSocket:
    application_state = WebSocketState.CONNECTED

    def __init__(self):
        self.sent: list[dict] = []
        self.scope: dict[str, object] = {}
        self.closed: tuple[int, str] | None = None

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)

    async def close(self, *, code: int, reason: str) -> None:
        self.closed = (code, reason)
        self.application_state = WebSocketState.DISCONNECTED


def test_private_websocket_transport_exception_is_content_free(caplog):
    canary = "PRIVATE_WEBSOCKET_TRANSPORT_CANARY"
    websocket = _FakeWebSocket()
    router_module._mark_websocket_private(websocket)

    async def fail(_payload: dict) -> None:
        raise RuntimeError(canary)

    websocket.send_json = fail
    event = router_module.build_error_event("fixed", error_type="fixed")

    with caplog.at_level(logging.ERROR):
        asyncio.run(router_module._send_event(websocket, event))

    assert canary not in caplog.text
    assert "Failed to send private event to client; content redacted" in caplog.text


# ===================================================================
# Connection
# ===================================================================


class TestConnection:
    def test_connect_receives_session_created(self, setup):
        app, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                msg = ws.receive_json()
                assert msg["type"] == "session.created"
                assert msg["event_id"].startswith("event_")
                assert "session" in msg

    def test_second_connection_rejected(self, setup):
        app, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws1:
                ws1.receive_json()  # session.created
                with client.websocket_connect("/v1/realtime") as ws2:
                    msg = ws2.receive_json()
                    assert msg["type"] == "error"
                    # Rejection uses the stateless build_error_event helper —
                    # the error type identifies pool exhaustion specifically.
                    assert msg["error"]["type"] == "session_limit_reached"


# ===================================================================
# Client event dispatch
# ===================================================================


class TestClientEventDispatch:
    def test_audio_append_forwarded_to_input_queue(self, setup):
        app, _, input_queue, *_ = setup
        audio_b64 = base64.b64encode(_pcm_bytes(512)).decode("ascii")
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                ws.send_json(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64,
                    }
                )
                time.sleep(0.1)
                item = input_queue.get(timeout=1)
                assert isinstance(item, tuple) and len(item) == 2
                chunk, rt_cfg = item
                assert isinstance(chunk, bytes)
                assert len(chunk) == CHUNK_SIZE_BYTES

    def test_session_update_applied(self, setup):
        app, service, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                ws.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "audio": {"output": {"voice": "coral"}},
                        },
                    }
                )
                time.sleep(0.1)
                cid = service.connection_ids[0]
                assert service._state(cid).runtime_config.session.audio.output.voice == "coral"

    def test_private_transcript_barrier_handshake_is_exact_and_acknowledged(self, setup):
        app, service, *_ = setup
        nonce = "ab" * 32
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                ws.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "reachy_private_transcript_barrier": {"version": 1, "nonce": nonce},
                        },
                    }
                )

                ready = ws.receive_json()
                assert ready == {
                    "type": "reachy.transcript_barrier.ready",
                    "event_id": ready["event_id"],
                    "version": 1,
                    "nonce": nonce,
                }
                assert service.transcript_barrier_enabled() is True

    def test_home_assistant_guard_ready_precedes_audio_and_missing_guard_closes(self, setup):
        app, service, input_queue, *_ = setup
        service.home_assistant_guard_required = True
        tools = [
            {
                "type": "function",
                "name": "home_assistant__GetLiveContext",
                "parameters": {"type": "object", "properties": {"area": {"type": "string"}}},
            }
        ]
        contract = RealtimeSessionCreateRequest(type="realtime", instructions="private", tools=tools)
        digest, count, _names = session_contract(contract.instructions, contract.tools)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                ws.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "instructions": "private",
                            "tools": tools,
                            "reachy_home_assistant_guard": {
                                "version": 1,
                                "nonce": "31" * 32,
                                "session_contract_sha256": digest,
                                "tool_count": count,
                            },
                        },
                    }
                )
                ready = ws.receive_json()
                assert ready["type"] == "reachy.home_assistant_guard.ready"
                assert ready["session_contract_sha256"] == digest
                assert service.home_assistant_guard_enabled() is True
                assert not any(isinstance(item, tuple) for item in tuple(input_queue.queue))

            _simulate_session_end_drain(setup[2], setup[3])
            deadline = time.monotonic() + 1.0
            while service.connection_ids and time.monotonic() < deadline:
                time.sleep(0.01)
            assert service.connection_ids == []

            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                ws.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "instructions": "private",
                            "tools": tools,
                        },
                    }
                )
                error = ws.receive_json()
                assert error["type"] == "error"
                assert error["error"]["type"] == "invalid_home_assistant_guard"

    def test_response_create_cannot_add_home_assistant_authority_after_session_start(self, setup):
        app, service, *_ = setup
        service.home_assistant_guard_required = True
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                ws.send_json(
                    {
                        "type": "response.create",
                    }
                )

                error = ws.receive_json()
                assert error["type"] == "error"
                assert error["error"]["type"] == "invalid_home_assistant_guard"
                assert service.sensitive_content() is True

    def test_required_backend_rejects_non_object_first_event_content_free(self, setup):
        app, service, *_ = setup
        service.home_assistant_guard_required = True
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                ws.send_json(["not", "an", "event"])

                error = ws.receive_json()
                assert error["type"] == "error"
                assert error["error"] == {
                    "type": "invalid_home_assistant_guard",
                    "code": None,
                    "event_id": None,
                    "message": "Home Assistant guard protocol violation.",
                    "param": None,
                }
                assert service.sensitive_content() is True

    def test_private_transcript_barrier_malformed_or_duplicate_handshake_fails_closed(self, setup):
        app, service, *_ = setup
        nonce = "cd" * 32
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                ws.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "reachy_private_transcript_barrier": {"version": 1, "nonce": nonce},
                        },
                    }
                )
                assert ws.receive_json()["type"] == "reachy.transcript_barrier.ready"

                ws.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "reachy_private_transcript_barrier": {"version": 1, "nonce": nonce},
                        },
                    }
                )
                error = ws.receive_json()
                assert error["type"] == "error"
                assert error["error"]["type"] == "invalid_transcript_barrier"
                cid = service.connection_ids[0]
                assert service.transcript_barrier_failed(cid) is True

    def test_malformed_first_private_handshake_is_redacted_poisoned_and_closed(
        self,
        setup,
        caplog,
    ):
        app, service, input_queue, *_ = setup
        nonce = "c1" * 32
        canary = "PRIVATE_INVALID_ACTIVATION_MODEL_CANARY"
        with caplog.at_level(logging.ERROR):
            with TestClient(app) as client:
                with client.websocket_connect("/v1/realtime") as ws:
                    ws.receive_json()
                    ws.send_json(
                        {
                            "type": "session.update",
                            "session": {
                                "type": "realtime",
                                "model": {"private": canary},
                                "reachy_private_transcript_barrier": {
                                    "version": 1,
                                    "nonce": nonce,
                                },
                            },
                        }
                    )
                    error = ws.receive_json()
                    conn_id = service.connection_ids[0]
                    assert service.transcript_barrier_failed(conn_id)

        assert error["error"]["type"] == "invalid_transcript_barrier"
        assert error["error"]["message"] == "Private transcript barrier protocol violation."
        assert canary not in str(error)
        assert canary not in caplog.text
        assert not any(isinstance(item, tuple) for item in tuple(input_queue.queue))

    def test_private_activation_cannot_return_before_ready_and_admit_later_audio(
        self,
        setup,
        caplog,
    ):
        app, service, input_queue, *_ = setup
        canary = "PRIVATE_TRANSCRIPTION_ACTIVATION_MODEL_CANARY"
        with caplog.at_level(logging.INFO):
            with TestClient(app) as client:
                with client.websocket_connect("/v1/realtime") as ws:
                    ws.receive_json()
                    ws.send_json(
                        {
                            "type": "session.update",
                            "session": {
                                "type": "transcription",
                                "model": canary,
                                "reachy_private_transcript_barrier": {
                                    "version": 1,
                                    "nonce": "c4" * 32,
                                },
                            },
                        }
                    )
                    error = ws.receive_json()
                    conn_id = service.connection_ids[0]

                    assert error["error"]["type"] == "invalid_transcript_barrier"
                    assert error["error"]["message"] == ("Private transcript barrier protocol violation.")
                    assert service.transcript_barrier_failed(conn_id)

        assert canary not in caplog.text
        assert canary not in str(error)
        assert not any(isinstance(item, tuple) for item in tuple(input_queue.queue))

    def test_private_route_and_send_loop_exceptions_are_content_free(
        self,
        setup,
        monkeypatch,
        caplog,
    ):
        app, service, _input_queue, _output_queue, text_output_queue, *_ = setup
        nonce = "c2" * 32
        route_canary = "PRIVATE_ROUTE_FAILURE_CANARY"
        send_canary = "PRIVATE_SEND_LOOP_FAILURE_CANARY"
        with caplog.at_level(logging.ERROR):
            with TestClient(app) as client:
                with client.websocket_connect("/v1/realtime") as ws:
                    ws.receive_json()
                    ws.send_json(
                        {
                            "type": "session.update",
                            "session": {
                                "type": "realtime",
                                "reachy_private_transcript_barrier": {
                                    "version": 1,
                                    "nonce": nonce,
                                },
                            },
                        }
                    )
                    assert ws.receive_json()["type"] == "reachy.transcript_barrier.ready"

                    def fail_route(*_args, **_kwargs):
                        raise RuntimeError(route_canary)

                    monkeypatch.setattr(service, "handle_audio_append", fail_route)
                    audio_b64 = base64.b64encode(_pcm_bytes(512)).decode("ascii")
                    ws.send_json({"type": "input_audio_buffer.append", "audio": audio_b64})

                # Use a fresh session for the separately-owned send-loop exception.
                _simulate_session_end_drain(setup[2], setup[3])
                deadline = time.monotonic() + 1.0
                while service.connection_ids and time.monotonic() < deadline:
                    time.sleep(0.01)

                with client.websocket_connect("/v1/realtime") as ws:
                    ws.receive_json()
                    ws.send_json(
                        {
                            "type": "session.update",
                            "session": {
                                "type": "realtime",
                                "reachy_private_transcript_barrier": {
                                    "version": 1,
                                    "nonce": "c3" * 32,
                                },
                            },
                        }
                    )
                    assert ws.receive_json()["type"] == "reachy.transcript_barrier.ready"

                    def fail_send(*_args, **_kwargs):
                        raise RuntimeError(send_canary)

                    monkeypatch.setattr(service, "dispatch_pipeline_event", fail_send)
                    text_output_queue.put(AssistantTextEvent(text="private"))
                    time.sleep(0.1)

        assert route_canary not in caplog.text
        assert send_canary not in caplog.text
        assert "Private client pipeline error; content redacted" in caplog.text
        assert "Private pipeline send loop error; content redacted" in caplog.text

    def test_private_handshake_redacts_malformed_client_event_on_wire(self, setup, caplog):
        app, _service, *_ = setup
        nonce = "ce" * 32
        canary = "PII_JOSH"
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                ws.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "reachy_private_transcript_barrier": {"version": 1, "nonce": nonce},
                        },
                    }
                )
                assert ws.receive_json()["type"] == "reachy.transcript_barrier.ready"

                with caplog.at_level(logging.ERROR):
                    ws.send_json(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "id": canary,
                                "type": "message",
                                "role": "invalid",
                                "content": [{"type": "input_text", "text": canary}],
                            },
                        }
                    )
                    error = ws.receive_json()

        assert error["type"] == "error"
        assert error["error"]["message"] == "Unknown or invalid private client event."
        assert canary not in str(error)
        assert canary not in caplog.text

    @pytest.mark.parametrize(
        ("event", "error_type"),
        [
            (
                {
                    "type": "conversation.item.create",
                    "item": {
                        "id": "PRIVATE_SEMANTIC_CONV_CANARY",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "private"}],
                    },
                },
                "invalid_conversation_item",
            ),
            (
                {
                    "type": "response.create",
                    "response": {
                        "input": [
                            {
                                "id": "PRIVATE_SEMANTIC_RESP_CANARY",
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": "private"}],
                            }
                        ]
                    },
                },
                "invalid_input_item",
            ),
        ],
    )
    def test_private_handshake_redacts_semantic_client_errors_on_wire(self, setup, event, error_type):
        app, _service, *_ = setup
        nonce = "cf" * 32
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                ws.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "reachy_private_transcript_barrier": {"version": 1, "nonce": nonce},
                        },
                    }
                )
                assert ws.receive_json()["type"] == "reachy.transcript_barrier.ready"

                ws.send_json(event)
                error = ws.receive_json()

        assert error["type"] == "error"
        assert error["error"] == {
            "code": None,
            "event_id": None,
            "message": "Invalid private client event.",
            "param": None,
            "type": error_type,
        }
        assert "CANARY" not in str(error)

    def test_private_response_input_semantic_failure_is_atomic(self, setup):
        app, service, *_ = setup
        nonce = "d0" * 32
        canary = "PRIVATE_REJECTED_PREFIX_CANARY"
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                ws.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "reachy_private_transcript_barrier": {"version": 1, "nonce": nonce},
                        },
                    }
                )
                assert ws.receive_json()["type"] == "reachy.transcript_barrier.ready"

                ws.send_json(
                    {
                        "type": "response.create",
                        "response": {
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
                    }
                )
                error = ws.receive_json()
                state = service._state(service.connection_ids[0])

                assert error["error"]["message"] == "Invalid private client event."
                assert "CANARY" not in str(error)
                assert state.runtime_config.chat.buffer == []
                assert canary not in str(state.runtime_config.chat.to_responses_api_chat())

    def test_private_session_updates_redact_model_values(self, setup, caplog):
        app, _service, *_ = setup
        nonce = "d1" * 32
        activation_canary = "PRIVATE_ACTIVATION_MODEL_CANARY"
        update_canary = "PRIVATE_POST_HANDSHAKE_MODEL_CANARY"
        with caplog.at_level(logging.INFO):
            with TestClient(app) as client:
                with client.websocket_connect("/v1/realtime") as ws:
                    ws.receive_json()
                    ws.send_json(
                        {
                            "type": "session.update",
                            "session": {
                                "type": "realtime",
                                "model": activation_canary,
                                "reachy_private_transcript_barrier": {"version": 1, "nonce": nonce},
                            },
                        }
                    )
                    assert ws.receive_json()["type"] == "reachy.transcript_barrier.ready"
                    ws.send_json(
                        {
                            "type": "session.update",
                            "session": {"type": "realtime", "model": update_canary},
                        }
                    )
                    time.sleep(0.01)

        assert activation_canary not in caplog.text
        assert update_canary not in caplog.text
        assert caplog.text.count("Private session model updated; content redacted") == 2

    def test_private_transcript_barrier_exact_resolution_round_trip(self, setup):
        app, service, _, _, text_output_queue, *_ = setup
        nonce = "ef" * 32
        transcript = "Who am I?"
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                ws.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "reachy_private_transcript_barrier": {"version": 1, "nonce": nonce},
                        },
                    }
                )
                assert ws.receive_json()["type"] == "reachy.transcript_barrier.ready"

                text_output_queue.put(TranscriptBarrierCompletedEvent(transcript=transcript))
                completed = ws.receive_json()
                assert completed["type"] == "reachy.transcript_barrier.completed"
                assert completed["transcript"] == transcript

                ws.send_json(
                    {
                        "type": "reachy.transcript_barrier.resolve",
                        "version": 1,
                        "nonce": nonce,
                        "sequence": completed["sequence"],
                        "input_item_id": completed["item_id"],
                        "action": "accept",
                        "item": {
                            "id": "msg_barrier_1",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": transcript}],
                        },
                    }
                )
                assert ws.receive_json()["type"] == "conversation.item.created"
                resolved = ws.receive_json()
                assert resolved["type"] == "reachy.transcript_barrier.resolved"
                assert resolved["action"] == "accepted"
                assert resolved["replacement_item_id"] == "msg_barrier_1"
                cid = service.connection_ids[0]
                assert service._state(cid).runtime_config.chat.buffer[-1].content[0].text == transcript

    def test_private_transcript_barrier_invalid_resolution_closes_without_echo(self, setup):
        app, service, _, _, text_output_queue, *_ = setup
        nonce = "01" * 32
        canary = "PRIVATE_TRANSCRIPT_CANARY"
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                ws.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "reachy_private_transcript_barrier": {"version": 1, "nonce": nonce},
                        },
                    }
                )
                ws.receive_json()
                text_output_queue.put(TranscriptBarrierCompletedEvent(transcript=canary))
                completed = ws.receive_json()

                ws.send_json(
                    {
                        "type": "reachy.transcript_barrier.resolve",
                        "version": 1,
                        "nonce": nonce,
                        "sequence": completed["sequence"],
                        "input_item_id": completed["item_id"],
                        "action": "accept",
                        "item": {
                            "id": "invalid-id",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": canary}],
                        },
                    }
                )
                error = ws.receive_json()
                assert error["type"] == "error"
                assert canary not in str(error)
                cid = service.connection_ids[0]
                cfg = service._state(cid).runtime_config
                assert cfg.transcript_barrier_failed is True
                assert cfg.transcript_barrier_pending_transcript is None

    def test_disconnect_scrubs_pending_private_transcript_before_handler_drain(self, setup):
        app, service, _, _, text_output_queue, *_ = setup
        nonce = "02" * 32
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                ws.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "reachy_private_transcript_barrier": {"version": 1, "nonce": nonce},
                        },
                    }
                )
                ws.receive_json()
                text_output_queue.put(TranscriptBarrierCompletedEvent(transcript="DISCONNECT_PRIVATE_CANARY"))
                assert ws.receive_json()["type"] == "reachy.transcript_barrier.completed"
                conn_id = service.connection_ids[0]
                cfg = service._state(conn_id).runtime_config
                assert cfg.transcript_barrier_pending is True

            # The handler chain has not consumed SESSION_END in this fixture,
            # so unregister is deliberately still pending. Scrubbing is not.
            assert conn_id in service.connection_ids
            assert cfg.transcript_barrier_pending is False
            assert cfg.transcript_barrier_pending_transcript is None
            assert cfg.transcript_barrier_failed is True

    def test_conversation_item_create_returns_events(self, setup):
        app, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                ws.send_json(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "ping"}],
                        },
                    }
                )
                msg = ws.receive_json()
                assert msg["type"] == "conversation.item.created"
                assert msg["item"]["content"][0]["text"] == "ping"

    def test_response_create_error_when_active(self, setup):
        app, service, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                conn_id = list(service._conns.keys())[0]
                service.response._ensure_response(conn_id)
                ws.send_json({"type": "response.create"})
                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert "another response is in progress" in msg["error"]["message"].lower()

    def test_response_cancel_returns_events(self, setup):
        app, service, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                conn_id = list(service._conns.keys())[0]
                service.response._ensure_response(conn_id)
                ws.send_json({"type": "response.cancel"})
                assert ws.receive_json()["type"] == "response.done"

    def test_response_cancel_flushes_queues(self, setup):
        app, service, _, output_queue, text_output_queue, _, _, response_playing, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                conn_id = list(service._conns.keys())[0]
                service.response._ensure_response(conn_id)
                response_playing.set()
                output_queue.put(_pcm_bytes(256))
                output_queue.put(_pcm_bytes(256))
                text_output_queue.put(AssistantTextEvent(text="stale"))
                ws.send_json({"type": "response.cancel"})
                assert ws.receive_json()["type"] == "response.done"
                time.sleep(0.1)
                assert output_queue.empty()
                assert text_output_queue.empty()
                assert not response_playing.is_set()
                assert cancel_scope.discarding

    def test_response_cancel_spurious_does_not_set_discarding(self, setup):
        """response.cancel when no response is active must NOT enable discarding,
        otherwise it would stick True forever (no __RESPONSE_DONE__ to clear it)."""
        app, service, _, _, _, _, _, _, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                assert not service._state(list(service._conns.keys())[0]).in_response
                ws.send_json({"type": "response.cancel"})
                time.sleep(0.1)
                assert not cancel_scope.discarding

    def test_response_cancel_retires_pending_automatic_generation(self, setup):
        app, service, _, _, _, _, _, response_playing, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                conn_id = list(service._conns.keys())[0]
                state = service._state(conn_id)
                state.response_pending = True
                stale_generation = cancel_scope.generation
                response_playing.set()

                ws.send_json({"type": "response.cancel"})
                time.sleep(0.1)

                assert state.response_pending is False
                assert state.in_response is False
                assert cancel_scope.generation == stale_generation + 1
                assert cancel_scope.discarding
                assert not response_playing.is_set()

    def test_response_cancel_late_audio_is_discarded(self, setup):
        """Audio arriving after response.cancel is silently dropped (discard guard)."""
        app, service, _, output_queue, _, _, _, response_playing, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                conn_id = list(service._conns.keys())[0]
                service.response._ensure_response(conn_id)
                response_playing.set()
                ws.send_json({"type": "response.cancel"})
                assert ws.receive_json()["type"] == "response.done"
                time.sleep(0.1)
                assert cancel_scope.discarding
                output_queue.put(_pcm_bytes(256))
                time.sleep(0.15)
                # No response.created or audio delta should appear; only
                # __RESPONSE_DONE__ will eventually clear the guard.
                output_queue.put(AUDIO_RESPONSE_DONE)
                time.sleep(0.15)
                assert not cancel_scope.discarding

    def test_unknown_event_returns_error(self, setup):
        app, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                ws.send_json({"type": "bogus.event"})
                msg = ws.receive_json()
                assert msg["type"] == "error"


# ===================================================================
# Send loop (pipeline -> client)
# ===================================================================


class TestSendLoop:
    def test_audio_output_ignores_session_end_control_message(self, setup):
        app, _, _, output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                output_queue.put(SESSION_END)
                output_queue.put(_pcm_bytes(256))

                msg1 = ws.receive_json()
                assert msg1["type"] == "response.created"
                msg2 = ws.receive_json()
                assert msg2["type"] == "response.output_audio.delta"

    def test_audio_output_sends_response_created_and_delta(self, setup):
        app, _, _, output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                output_queue.put(_pcm_bytes(256))
                msg1 = ws.receive_json()
                assert msg1["type"] == "response.created"
                assert msg1["response"]["status"] == "in_progress"
                msg2 = ws.receive_json()
                assert msg2["type"] == "response.output_audio.delta"
                assert "delta" in msg2

    def test_audio_output_batches_immediately_available_chunks(self, setup):
        app, _, _, output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                output_queue.put(_pcm_bytes(256))
                output_queue.put(_pcm_bytes(256))
                output_queue.put(PIPELINE_END)

                msg1 = ws.receive_json()
                assert msg1["type"] == "response.created"

                msg2 = ws.receive_json()
                assert msg2["type"] == "response.output_audio.delta"
                decoded = base64.b64decode(msg2["delta"])
                assert len(decoded) == len(_pcm_bytes(512))

                msg3 = ws.receive_json()
                msg4 = ws.receive_json()
                types = {msg3["type"], msg4["type"]}
                assert "response.output_audio.done" in types
                assert "response.done" in types

    def test_end_marker_sends_finish_events(self, setup):
        app, _, _, output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                output_queue.put(_pcm_bytes(256))
                ws.receive_json()  # response.created
                ws.receive_json()  # audio delta
                output_queue.put(PIPELINE_END)
                msg1 = ws.receive_json()
                msg2 = ws.receive_json()
                types = {msg1["type"], msg2["type"]}
                assert "response.output_audio.done" in types
                assert "response.done" in types

    def test_text_output_sends_pipeline_events(self, setup):
        app, _, _, _, text_output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                text_output_queue.put(SpeechStartedEvent())
                msg = ws.receive_json()
                assert msg["type"] == "input_audio_buffer.speech_started"
                assert msg["audio_start_ms"] == 0

    def test_barge_in_discard_clears_after_response_done(self, setup):
        """After barge-in sets discarding=True, __RESPONSE_DONE__ must clear it back to False."""
        app, service, _, output_queue, text_output_queue, _, _, response_playing, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                conn_id = list(service._conns.keys())[0]
                service.response._ensure_response(conn_id)
                response_playing.set()
                # Trigger barge-in
                text_output_queue.put(SpeechStartedEvent())
                assert ws.receive_json()["type"] == "response.done"
                assert ws.receive_json()["type"] == "input_audio_buffer.speech_started"
                time.sleep(0.1)
                assert cancel_scope.discarding
                output_queue.put(AUDIO_RESPONSE_DONE)
                time.sleep(0.15)
                assert not cancel_scope.discarding

    def test_speech_started_cancels_pending_implicit_response(self, setup):
        app, service, _, output_queue, text_output_queue, _, _, response_playing, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                conn_id = list(service._conns.keys())[0]
                stale_generation = cancel_scope.generation
                service._state(conn_id).response_pending = True

                text_output_queue.put(SpeechStartedEvent())
                msg = ws.receive_json()

                assert msg["type"] == "input_audio_buffer.speech_started"
                time.sleep(0.15)
                assert cancel_scope.discarding
                assert cancel_scope.generation == stale_generation + 1
                assert service._state(conn_id).response_pending is False
                assert service._state(conn_id).in_response is False
                assert not response_playing.is_set()

                output_queue.put(AudioOutput(audio=AUDIO_RESPONSE_DONE, cancel_generation=stale_generation))
                time.sleep(0.15)
                assert not cancel_scope.discarding

    def test_queued_selector_rejection_wins_before_speech_start_cancellation(self, setup):
        app, service, _, _, text_output_queue, _, _, response_playing, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                conn_id = service.connection_ids[0]
                state = service._state(conn_id)
                cfg = state.runtime_config
                cfg.home_assistant_guard_version = 1
                cfg.home_assistant_guard_nonce = "cd" * 32
                cfg.home_assistant_guard_contract_sha256 = "ef" * 32
                cfg.home_assistant_guard_tool_count = 1
                cfg.home_assistant_guard_tool_names = ("home_assistant__GetLiveContext",)
                service.response._ensure_response(conn_id)
                response_playing.set()
                generation = cancel_scope.generation

                text_output_queue.put(SpeechStartedEvent())
                text_output_queue.put(
                    ResponseFailedEvent(
                        message=HOME_ASSISTANT_SELECTOR_REJECTED,
                        cancel_generation=generation,
                    )
                )

                error = ws.receive_json()
                done = ws.receive_json()
                assert error["type"] == "error"
                assert error["error"]["type"] == "home_assistant_selector_rejected"
                assert done["type"] == "response.done"
                assert done["response"]["status"] == "failed"
                time.sleep(0.1)
                assert cfg.home_assistant_guard_failed is True
                assert cancel_scope.is_stale(generation)
                assert text_output_queue.empty()

    def test_late_selector_rejection_from_cancelled_generation_is_discarded(self, setup):
        app, service, _, _, text_output_queue, _, _, _, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                conn_id = service.connection_ids[0]
                state = service._state(conn_id)
                cfg = state.runtime_config
                cfg.home_assistant_guard_version = 1
                cfg.home_assistant_guard_nonce = "cd" * 32
                cfg.home_assistant_guard_contract_sha256 = "ef" * 32
                cfg.home_assistant_guard_tool_count = 1
                cfg.home_assistant_guard_tool_names = ("home_assistant__GetLiveContext",)
                stale_generation = cancel_scope.generation
                cancel_scope.cancel()
                service.response._ensure_response(conn_id)

                text_output_queue.put(
                    ResponseFailedEvent(
                        message=HOME_ASSISTANT_SELECTOR_REJECTED,
                        cancel_generation=stale_generation,
                    )
                )
                time.sleep(0.15)

                assert cfg.home_assistant_guard_failed is False
                assert state.in_response is True
                assert text_output_queue.empty()

    def test_guarded_semantic_invalid_conversation_item_closes_before_followup(self, setup):
        app, service, *_ = setup
        tools = [
            {
                "type": "function",
                "name": "home_assistant__GetLiveContext",
                "parameters": {"type": "object", "properties": {"area": {"type": "string"}}},
            }
        ]
        contract = RealtimeSessionCreateRequest(type="realtime", instructions="private", tools=tools)
        digest, count, _names = session_contract(contract.instructions, contract.tools)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                ws.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "instructions": "private",
                            "tools": tools,
                            "reachy_home_assistant_guard": {
                                "version": 1,
                                "nonce": "31" * 32,
                                "session_contract_sha256": digest,
                                "tool_count": count,
                            },
                        },
                    }
                )
                assert ws.receive_json()["type"] == "reachy.home_assistant_guard.ready"
                conn_id = service.connection_ids[0]
                cfg = service._state(conn_id).runtime_config

                ws.send_json(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": "call_unknown",
                            "output": '{"result": 42}',
                        },
                    }
                )
                error = ws.receive_json()

                assert error["type"] == "error"
                assert error["error"] == {
                    "type": "invalid_conversation_item",
                    "code": None,
                    "event_id": None,
                    "message": "Home Assistant guard protocol violation.",
                    "param": None,
                }
                assert cfg.home_assistant_guard_failed is True
                assert service.text_prompt_queue is not None
                assert service.text_prompt_queue.empty()

    def test_speech_started_does_not_cancel_pending_when_internal_non_interrupt(self, setup):
        app, service, _, _, text_output_queue, _, _, _, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                conn_id = list(service._conns.keys())[0]
                service._state(conn_id).response_pending = True

                text_output_queue.put(SpeechStartedEvent(interrupt_response=False))
                msg = ws.receive_json()

                assert msg["type"] == "input_audio_buffer.speech_started"
                time.sleep(0.15)
                assert not cancel_scope.discarding
                assert service._state(conn_id).response_pending is True

    def test_stale_tagged_audio_is_dropped_after_interruption(self, setup):
        app, _, _, output_queue, _, _, _, _, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                stale_generation = cancel_scope.generation
                cancel_scope.cancel()
                current_generation = cancel_scope.generation
                output_queue.put(AudioOutput(audio=_pcm_bytes(64), cancel_generation=stale_generation))
                output_queue.put(AudioOutput(audio=_pcm_bytes(512), cancel_generation=current_generation))

                assert ws.receive_json()["type"] == "response.created"
                delta = ws.receive_json()

                assert delta["type"] == "response.output_audio.delta"
                assert len(base64.b64decode(delta["delta"])) == len(_pcm_bytes(512))

    def test_current_generation_text_survives_stuck_discarding(self, setup):
        """Regression: a fresh response's transcript must survive a stuck discard guard.

        A superseded speculative turn can leave ``cancel_scope.discarding`` stuck True
        (its TTS dropped the stale ``EndOfResponse`` without emitting AUDIO_RESPONSE_DONE,
        so ``response_done()`` never cleared the flag). The next response's audio is tagged
        with the current generation and streams fine, but the assistant text used to be
        blanket-dropped while discarding — leaving audio + ``response.done`` with no
        ``response.output_audio_transcript.done``. The text is now discarded by the same
        generation-aware rule as audio, so a current-generation transcript is kept.
        """
        app, _, _, output_queue, text_output_queue, _, _, _, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                cancel_scope.cancel()  # discarding=True, generation bumped; sentinel never arrived
                current_generation = cancel_scope.generation
                assert cancel_scope.discarding

                text_output_queue.put(AssistantTextEvent(text="hello there", cancel_generation=current_generation))
                output_queue.put(AudioOutput(audio=_pcm_bytes(256), cancel_generation=current_generation))
                output_queue.put(AudioOutput(audio=AUDIO_RESPONSE_DONE, cancel_generation=current_generation))

                types: list[str] = []
                transcript = None
                for _ in range(8):
                    msg = ws.receive_json()
                    types.append(msg["type"])
                    if msg["type"] == "response.output_audio_transcript.done":
                        transcript = msg["transcript"]
                    if msg["type"] == "response.done":
                        break
                assert "response.output_audio_transcript.done" in types
                assert transcript == "hello there"

    def test_stale_tagged_response_done_does_not_finish_current_response(self, setup):
        app, service, _, output_queue, _, _, _, _, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                conn_id = list(service._conns.keys())[0]
                stale_generation = cancel_scope.generation
                service.response._ensure_response(conn_id)
                service.finish_response(conn_id, status="cancelled")
                cancel_scope.cancel()
                current_response_id, _ = service.response._ensure_response(conn_id)

                output_queue.put(AudioOutput(audio=AUDIO_RESPONSE_DONE, cancel_generation=stale_generation))
                time.sleep(0.15)

                state = service._state(conn_id)
                assert state.in_response
                assert state.current_response_id == current_response_id

    def test_response_done_drains_pending_token_usage_before_finish(self, setup):
        app, service, _, output_queue, text_output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                conn_id = list(service._conns.keys())[0]

                text_output_queue.put(
                    AssistantTextEvent(
                        text="",
                        tools=[{"type": "function_call", "call_id": "c1", "name": "f1", "arguments": "{}"}],
                    )
                )
                text_output_queue.put(TokenUsageEvent(input_tokens=10, output_tokens=5))
                output_queue.put(AUDIO_RESPONSE_DONE)

                assert ws.receive_json()["type"] == "response.created"
                assert ws.receive_json()["type"] == "response.function_call_arguments.done"
                assert ws.receive_json()["type"] == "response.done"

                assert service.total_usage.input_tokens == 10
                assert service.total_usage.output_tokens == 5
                assert service._state(conn_id).response_usage.input_tokens == 0
                assert service._state(conn_id).response_usage.output_tokens == 0

    def test_response_completion_drain_sends_pending_tool_before_done(self, setup):
        _, service, input_queue, output_queue, text_output_queue, should_listen, _, response_playing, cancel_scope = (
            setup
        )
        unit = PipelineUnit(
            index=0,
            service=service,
            cancel_scope=cancel_scope,
            should_listen=should_listen,
            response_playing=response_playing,
            input_queue=input_queue,
            output_queue=output_queue,
            text_output_queue=text_output_queue,
            text_prompt_queue=Queue(),
            handlers=[],
        )
        conn_id = service.register()
        response_id, _ = service.response._ensure_response(conn_id)
        text_output_queue.put(
            AssistantTextEvent(
                text="",
                tools=[
                    {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "play_emotion",
                        "arguments": '{"emotion":"loving"}',
                    }
                ],
            )
        )
        text_output_queue.put(TokenUsageEvent(input_tokens=10, output_tokens=5))
        ws = _FakeWebSocket()

        asyncio.run(router_module._drain_pending_response_events(ws, unit, conn_id))
        done_events = service.finish_response(conn_id)

        assert [payload["type"] for payload in ws.sent] == ["response.function_call_arguments.done"]
        assert [event.type for event in done_events] == ["response.done"]
        assert ws.sent[0]["response_id"] == response_id
        assert done_events[0].response.id == response_id
        assert done_events[0].response.usage.input_tokens == 10
        assert done_events[0].response.usage.output_tokens == 5
        assert text_output_queue.empty()

    def test_response_completion_drain_orders_guard_failure_before_audio_done(self, setup):
        _, service, input_queue, output_queue, text_output_queue, should_listen, _, response_playing, cancel_scope = (
            setup
        )
        unit = PipelineUnit(
            index=0,
            service=service,
            cancel_scope=cancel_scope,
            should_listen=should_listen,
            response_playing=response_playing,
            input_queue=input_queue,
            output_queue=output_queue,
            text_output_queue=text_output_queue,
            text_prompt_queue=Queue(),
            handlers=[],
        )
        conn_id = service.register()
        state = service._state(conn_id)
        state.runtime_config.home_assistant_guard_version = 1
        state.runtime_config.home_assistant_guard_nonce = "cd" * 32
        state.runtime_config.home_assistant_guard_contract_sha256 = "ef" * 32
        state.runtime_config.home_assistant_guard_tool_count = 1
        state.runtime_config.home_assistant_guard_tool_names = ("home_assistant__GetLiveContext",)
        service.response._ensure_response(conn_id)
        text_output_queue.put(SpeechStartedEvent())
        text_output_queue.put(ResponseFailedEvent(message=HOME_ASSISTANT_SELECTOR_REJECTED))
        output_queue.put(AUDIO_RESPONSE_DONE)
        ws = _FakeWebSocket()

        asyncio.run(router_module._drain_pending_response_events(ws, unit, conn_id))
        late_completion = service.finish_response(conn_id)

        assert [payload["type"] for payload in ws.sent] == ["error", "response.done"]
        assert ws.sent[0]["error"]["type"] == "home_assistant_selector_rejected"
        assert ws.sent[0]["error"]["message"] == "Home Assistant guard protocol violation."
        assert ws.sent[1]["response"]["status"] == "failed"
        assert state.runtime_config.home_assistant_guard_failed is True
        assert state.in_response is False
        assert late_completion == []
        assert output_queue.get_nowait() == AUDIO_RESPONSE_DONE
        assert isinstance(text_output_queue.get_nowait(), SpeechStartedEvent)
        assert text_output_queue.empty()
        assert ws.closed == (1008, "Private session failed")

    def test_response_completion_drain_preserves_usage_across_non_response_boundary(self, setup):
        _, service, input_queue, output_queue, text_output_queue, should_listen, _, response_playing, cancel_scope = (
            setup
        )
        unit = PipelineUnit(
            index=0,
            service=service,
            cancel_scope=cancel_scope,
            should_listen=should_listen,
            response_playing=response_playing,
            input_queue=input_queue,
            output_queue=output_queue,
            text_output_queue=text_output_queue,
            text_prompt_queue=Queue(),
            handlers=[],
        )
        conn_id = service.register()
        response_id, _ = service.response._ensure_response(conn_id)
        text_output_queue.put(
            AssistantTextEvent(
                text="",
                tools=[{"type": "function_call", "call_id": "c1", "name": "play_emotion", "arguments": "{}"}],
            )
        )
        text_output_queue.put(SpeechStartedEvent())
        text_output_queue.put(TokenUsageEvent(input_tokens=10, output_tokens=5))
        text_output_queue.put(AssistantTextEvent(text="queued after boundary"))
        ws = _FakeWebSocket()

        asyncio.run(router_module._drain_pending_response_events(ws, unit, conn_id))
        done_events = service.finish_response(conn_id)

        assert [payload["type"] for payload in ws.sent] == ["response.function_call_arguments.done"]
        assert ws.sent[0]["response_id"] == response_id
        assert done_events[0].response.usage.input_tokens == 10
        assert done_events[0].response.usage.output_tokens == 5

        boundary = text_output_queue.get_nowait()
        queued_assistant = text_output_queue.get_nowait()
        assert isinstance(boundary, SpeechStartedEvent)
        assert isinstance(queued_assistant, AssistantTextEvent)
        assert queued_assistant.text == "queued after boundary"
        assert text_output_queue.empty()

    def test_speech_started_does_not_cancel_when_interrupt_disabled(self, setup):
        """With interrupt_response=False, speech during playback should NOT cancel or flush."""
        from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

        app, service, _, output_queue, text_output_queue, _, _, response_playing, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                conn_id = list(service._conns.keys())[0]
                service._state(conn_id).runtime_config.session.audio.input.turn_detection = ServerVad(
                    type="server_vad",
                    interrupt_response=False,
                )
                _, response_item_id = service.response._ensure_response(conn_id)
                response_playing.set()
                text_output_queue.put(SpeechStartedEvent())
                msg = ws.receive_json()
                assert msg["type"] == "input_audio_buffer.speech_started"
                time.sleep(0.15)
                assert response_playing.is_set(), "response_playing should remain set"
                assert not cancel_scope.discarding, "cancel_scope should not be discarding"
                assert service._state(conn_id).in_response, "response should still be active"
                assert service._state(conn_id).current_item_id == response_item_id


# ===================================================================
# Cleanup
# ===================================================================


class TestCleanup:
    def test_new_connection_resets_discard_after_invalidating_generation(self, setup):
        """connect-time _clean_unit cancels+resets: stale work is invalidated, discarding cleared."""
        app, _, *_rest, cancel_scope = setup
        cancel_scope.cancel()
        assert cancel_scope.discarding
        assert cancel_scope.generation == 1
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                assert not cancel_scope.discarding
                assert cancel_scope.generation == 2

    def test_disconnect_bumps_cancel_scope_generation(self, setup):
        """_clean_unit() on disconnect calls cancel() so in-flight generations go stale."""
        app, _, _, _, _, _, _, _, cancel_scope = setup
        assert cancel_scope.generation == 0
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                assert cancel_scope.generation == 1
            # disconnect triggers _clean_unit again + drain (short timeout in tests)
            time.sleep(0.3)
        assert cancel_scope.generation == 2

    def test_disconnect_unregisters(self, setup):
        app, service, input_queue, output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                assert len(service._conns) == 1
            # Simulate the handler chain consuming SESSION_END so the release
            # task can complete and unregister the session.
            _simulate_session_end_drain(input_queue, output_queue)
            time.sleep(0.3)
            assert len(service._conns) == 0

    def test_last_disconnect_cancels_and_clears_response_state(self, setup):
        app, service, input_queue, output_queue, text_output_queue, _, _, response_playing, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                conn_id = list(service._conns.keys())[0]
                service.response._ensure_response(conn_id)
                response_playing.set()
                output_queue.put(_pcm_bytes(256))
                text_output_queue.put(AssistantTextEvent(text="stale"))
            _simulate_session_end_drain(input_queue, output_queue)
            time.sleep(0.3)

        assert not cancel_scope.discarding
        assert cancel_scope.generation == 2
        assert not response_playing.is_set()
        assert text_output_queue.empty()


# ===================================================================
# Pool semantics (new in pool refactor)
# ===================================================================


def _make_unit(index: int) -> PipelineUnit:
    text_prompt_queue: Queue = Queue()
    should_listen = ThreadingEvent()
    should_listen.set()
    cancel_scope = CancelScope()
    service = RealtimeService(
        text_prompt_queue=text_prompt_queue,
        should_listen=should_listen,
        cancel_scope=cancel_scope,
    )
    assert service.verify_cancel_scope_wiring(cancel_scope, cancel_scope)
    return PipelineUnit(
        index=index,
        service=service,
        cancel_scope=cancel_scope,
        should_listen=should_listen,
        response_playing=ThreadingEvent(),
        input_queue=Queue(),
        output_queue=Queue(),
        text_output_queue=Queue(),
        text_prompt_queue=text_prompt_queue,
        handlers=[],
    )


class TestPool:
    def test_pool_endpoint_reports_idle_state(self):
        pool = [_make_unit(0), _make_unit(1)]
        app = create_app(pool=pool, stop_event=ThreadingEvent())
        with TestClient(app) as client:
            r = client.get("/v1/pool")
            assert r.status_code == 200
            data = r.json()
            assert data["size"] == 2
            assert data["in_use"] == 0
            assert [u["session_id"] for u in data["units"]] == [None, None]

    def test_two_clients_claim_two_slots_third_rejected(self):
        pool = [_make_unit(0), _make_unit(1)]
        app = create_app(pool=pool, stop_event=ThreadingEvent())
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws1:
                ws1.receive_json()  # session.created
                with client.websocket_connect("/v1/realtime") as ws2:
                    ws2.receive_json()  # session.created (different unit)
                    with client.websocket_connect("/v1/realtime") as ws3:
                        msg = ws3.receive_json()
                        assert msg["type"] == "error"
                        assert msg["error"]["type"] == "session_limit_reached"
                    # Pool now reports 2 in_use
                    r = client.get("/v1/pool")
                    assert r.json()["in_use"] == 2

    def test_usage_aggregates_errors_by_type_across_units(self):
        pool = [_make_unit(0), _make_unit(1)]
        pool[0].service.total_usage.record_error("foo")
        pool[0].service.total_usage.record_error("foo")
        pool[1].service.total_usage.record_error("bar")
        app = create_app(pool=pool, stop_event=ThreadingEvent())
        with TestClient(app) as client:
            data = client.get("/v1/usage").json()
            assert data["errors_by_type"] == {"foo": 2, "bar": 1}
            assert data["total_errors"] == 3
