"""Unit tests for the chat-completions LLM backend.

These run without a GPU or a live server: the OpenAI client is faked at the
module level, so the streaming/non-streaming parse logic and the format
converters are exercised purely in-process.

Run with pytest, or standalone:  python tests/test_chat_completions_backend.py
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from types import SimpleNamespace

import httpx
import pytest
from openai.types.realtime import ResponseCreatedEvent, ResponseCreateEvent, SessionUpdateEvent
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
    RealtimeConversationItemUserMessage,
)
from openai.types.realtime.realtime_conversation_item_user_message import Content as UserContent
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.realtime.realtime_session_create_request import RealtimeSessionCreateRequest
from openai.types.responses import ResponseFunctionToolCall

import speech_to_speech.LLM.base_openai_compatible_language_model as base_mod
import speech_to_speech.LLM.chat_completions_language_model as ccm
from speech_to_speech.api.openai_realtime.home_assistant_guard import (
    HOME_ASSISTANT_SELECTOR_REJECTED,
    MAX_GUARDED_PROVIDER_EVENTS,
)
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.transcript_barrier import TranscriptBarrierReadyEvent
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.chat_completions_language_model import (
    ChatCompletionsApiModelHandler,
    _to_chat_tool_choice,
    _to_chat_tools,
)
from speech_to_speech.LLM.tool_call.function_tool import MAX_TOOL_CALLS_PER_RESPONSE
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import (
    EndOfResponse,
    GenerateResponseRequest,
    LLMResponseChunk,
    TokenUsage,
)

# ── Fakes ────────────────────────────────────────────────────────────────────


class _FakeStream:
    """Iterable stand-in for openai.Stream; yields preset chunks."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        pass


class _PoisoningStream(_FakeStream):
    """Flip the barrier after provider iteration but before normalized events commit."""

    def __init__(self, chunks, runtime_config: RuntimeConfig):
        super().__init__(chunks)
        self._runtime_config = runtime_config

    def __iter__(self):
        yield from self._chunks
        self._runtime_config.transcript_barrier_failed = True


class _FailingStream(_FakeStream):
    """Raise a transport or parser failure while the provider stream is read."""

    def __init__(self, failure: Exception):
        super().__init__([])
        self._failure = failure

    def __iter__(self):
        raise self._failure
        yield  # pragma: no cover - make this an iterator without exposing content


# Make the handler's ``isinstance(resp, Stream)`` check recognise our fake as a
# stream. Non-streaming fakes stay plain SimpleNamespace, so they still take the
# non-stream branch.
ccm.Stream = _FakeStream


class _FakeCompletions:
    def __init__(self):
        self.next_result = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self.next_result


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self, *a, **k):
        self.chat = _FakeChat()
        self.last_options = None

    def with_options(self, **kwargs):
        self.last_options = kwargs
        return self


def _make_handler(stream=True, *, cancel_scope: CancelScope | None = None):
    """Build a handler whose warmup hits the fake client (no network)."""
    orig_openai = base_mod.OpenAI
    base_mod.OpenAI = _FakeClient
    try:
        h = ChatCompletionsApiModelHandler(
            threading.Event(),
            queue.Queue(),
            queue.Queue(),
            setup_kwargs=dict(
                model_name="test-model",
                base_url="http://fake/v1",
                api_key="k",
                stream=stream,
                disable_thinking=True,
                compact_history=False,
                cancel_scope=cancel_scope,
            ),
        )
    finally:
        base_mod.OpenAI = orig_openai
    return h


def test_warmup_uses_request_scoped_sdk_retries():
    handler = _make_handler()

    assert handler.client.last_options == {"max_retries": base_mod.WARMUP_MAX_RETRIES}


def test_cancelled_queued_request_cannot_start_after_private_barrier_ready():
    text_prompt_queue: queue.Queue = queue.Queue()
    cancel_scope = CancelScope()
    service = RealtimeService(text_prompt_queue=text_prompt_queue, cancel_scope=cancel_scope)
    assert service.verify_cancel_scope_wiring(cancel_scope, cancel_scope)
    conn_id = service.register()

    created = service.handle_response_create(
        conn_id,
        ResponseCreateEvent(type="response.create", response={"conversation": "none"}),
    )
    assert isinstance(created, ResponseCreatedEvent)
    request = text_prompt_queue.get(timeout=1.0)
    assert isinstance(request, GenerateResponseRequest)
    assert request.cancel_generation == 0

    cancel_scope.cancel()
    service.handle_response_cancel(conn_id)
    ready = service.handle_session_update(
        conn_id,
        SessionUpdateEvent(
            type="session.update",
            session={
                "type": "realtime",
                "reachy_private_transcript_barrier": {"version": 1, "nonce": "cd" * 32},
            },
        ),
    )
    assert isinstance(ready, TranscriptBarrierReadyEvent)

    handler = _make_handler(stream=False, cancel_scope=cancel_scope)
    handler.client.chat.completions.last_kwargs = None
    outputs = list(handler.process(request))

    assert handler.client.chat.completions.last_kwargs is None
    assert outputs == [EndOfResponse(cancel_generation=0)]
    service.unregister(conn_id)


def test_cancel_scope_serializes_response_admission_cancel_and_private_activation():
    scope = CancelScope()

    with scope.response_admission(0) as (admitted, generation):
        assert admitted is True
        assert generation == 0
        with scope.private_activation_guard() as quiescent:
            assert quiescent is False
        scope.cancel()

    with scope.response_admission(0) as (admitted, generation):
        assert admitted is False
        assert generation == 0
    with scope.private_activation_guard() as quiescent:
        assert quiescent is True


def _chunk(content=None, tool_calls=None, usage=None, finish_reason=None, *, choice_index=0):
    choices = []
    if content is not None or tool_calls is not None or finish_reason is not None:
        choices = [
            SimpleNamespace(
                index=choice_index,
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ]
    return SimpleNamespace(choices=choices, usage=usage)


def _tc_delta(index, id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments))


def _drive(
    handler,
    *,
    tools=None,
    tool_choice=None,
    user="Hallo",
    chat=None,
    response=None,
    instructions="Du bist ein Roboter.",
    private_barrier=False,
    home_assistant_guard=False,
):
    chat = chat or Chat(10)
    if user:
        chat.add_item(make_user_message(user))
    session = RealtimeSessionCreateRequest(type="realtime", instructions=instructions)
    if tools is not None:
        session.tools = tools
    if tool_choice is not None:
        session.tool_choice = tool_choice
    rc = RuntimeConfig(
        chat=chat,
        session=session,
        transcript_barrier_version=1 if private_barrier else None,
        transcript_barrier_nonce="ab" * 32 if private_barrier else None,
        home_assistant_guard_version=1 if home_assistant_guard else None,
        home_assistant_guard_nonce="cd" * 32 if home_assistant_guard else None,
        home_assistant_guard_contract_sha256="ef" * 32 if home_assistant_guard else None,
        home_assistant_guard_tool_count=len(tools or ()) if home_assistant_guard else 0,
        home_assistant_guard_tool_names=(
            tuple(str(tool["name"]) for tool in tools or ()) if home_assistant_guard else ()
        ),
    )
    req = GenerateResponseRequest(
        runtime_config=rc, response=response, language_code="de", turn_id="t", turn_revision=0
    )
    text, tools_out, usage, end = "", [], None, None
    for out in handler.process(req):
        if isinstance(out, LLMResponseChunk):
            text += out.text
            tools_out += list(out.tools)
        elif isinstance(out, TokenUsage):
            usage = (out.input_tokens, out.output_tokens)
        elif isinstance(out, EndOfResponse):
            end = out
    return text, tools_out, usage, chat, end


# ── Converter tests ──────────────────────────────────────────────────────────


def test_to_chat_tools_flat_to_nested():
    out = _to_chat_tools([{"type": "function", "name": "f", "description": "d", "parameters": {"type": "object"}}])
    assert out == [
        {"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}
    ]


def test_to_chat_tools_passthrough_and_none():
    nested = [{"type": "function", "function": {"name": "f"}}]
    assert _to_chat_tools(nested) == nested
    assert _to_chat_tools(None) is None
    assert _to_chat_tools([]) is None


def test_to_chat_tool_choice():
    assert _to_chat_tool_choice("auto") == "auto"
    assert _to_chat_tool_choice("required") == "required"
    assert _to_chat_tool_choice({"type": "function", "name": "f"}) == {"type": "function", "function": {"name": "f"}}


def test_build_extra_body_variants():
    f = ChatCompletionsApiModelHandler._build_extra_body
    assert f("http://x/v1", True, None) == {"chat_template_kwargs": {"enable_thinking": False}}
    assert f("http://x/v1", True, "none") == {"reasoning_effort": "none"}  # explicit effort wins
    assert f("https://api.openai.com/v1", True, "none") is None  # official OpenAI: no extra_body
    assert f("https://api.openai.com/v1/", True, "none") is None  # trailing slash still official
    assert f("http://x/v1", True, "") == {"chat_template_kwargs": {"enable_thinking": False}}  # empty effort ignored
    assert f("http://x/v1", False, None) is None
    assert f(None, True, None) is None


def test_chat_messages_encodes_tool_arguments_as_string():
    """to_transformers_chat emits arguments as a dict; the chat API needs a string."""
    chat = Chat(10)
    chat.add_item(make_user_message("Kopf links"))
    chat.add_item(
        RealtimeConversationItemFunctionCall(
            type="function_call", name="move_head", arguments='{"direction": "left"}', call_id="call_1", id="fc_1"
        )
    )
    chat.add_item(
        RealtimeConversationItemFunctionCallOutput(type="function_call_output", call_id="call_1", output="ok")
    )
    messages = ChatCompletionsApiModelHandler._chat_messages(chat)
    tool_call_msgs = [m for m in messages if m.get("tool_calls")]
    assert tool_call_msgs, "expected an assistant message carrying tool_calls"
    args = tool_call_msgs[0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str), f"arguments must be a JSON string, got {type(args)}"
    assert json.loads(args) == {"direction": "left"}


def test_chat_messages_strips_tool_output_name():
    """to_transformers_chat adds a tool name for HF templates; Chat Completions
    tool messages only accept role/tool_call_id/content."""
    chat = Chat(10)
    chat.add_item(make_user_message("Search for x"))
    chat.add_item(
        RealtimeConversationItemFunctionCall(
            type="function_call",
            name="search",
            arguments='{"q": "x"}',
            call_id="call_1",
            id="fc_1",
            status="completed",
        )
    )
    chat.add_item(
        RealtimeConversationItemFunctionCallOutput(type="function_call_output", call_id="call_1", output="found")
    )

    messages = ChatCompletionsApiModelHandler._chat_messages(chat)
    tool_message = [m for m in messages if m.get("role") == "tool"][0]
    assert tool_message == {"role": "tool", "tool_call_id": "call_1", "content": "found"}


def test_chat_messages_converts_image_and_text_parts_to_chat_shape():
    """to_transformers_chat emits Realtime-shaped parts (input_text / input_image
    with a bare-string image_url); the Chat Completions API needs text / image_url
    with a nested object."""
    chat = Chat(10)
    chat.add_item(
        RealtimeConversationItemUserMessage(
            type="message",
            role="user",
            content=[
                UserContent(type="input_text", text="What is this?"),
                UserContent(type="input_image", image_url="https://example.com/img.png", detail="auto"),
            ],
        )
    )
    messages = ChatCompletionsApiModelHandler._chat_messages(chat)
    user = [m for m in messages if m.get("role") == "user"][0]
    assert isinstance(user["content"], list)
    parts = {p["type"]: p for p in user["content"]}
    assert parts["text"]["text"] == "What is this?"
    assert parts["image_url"]["image_url"] == {"url": "https://example.com/img.png", "detail": "auto"}
    # No Realtime-shaped parts leak through.
    assert all(p["type"] not in ("input_text", "input_image") for p in user["content"])


# ── Streaming / non-streaming parse tests ─────────────────────────────────────


def test_streaming_text_and_usage():
    h = _make_handler(stream=True)
    h.client.chat.completions.create = lambda **k: _FakeStream(
        [
            _chunk(content="Hallo. "),
            _chunk(content="Wie geht es dir?"),
            _chunk(usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5)),
        ]
    )
    text, tools, usage, chat, _end = _drive(h)
    assert "Hallo" in text and "Wie geht es dir" in text
    assert usage == (12, 5)
    assert tools == []
    # assistant text was stored back into the conversation history
    assert any(getattr(i, "role", None) == "assistant" for i in chat.buffer)


def test_streaming_tool_call_accumulates_arguments():
    h = _make_handler(stream=True)
    # Arguments arrive split across deltas, as real servers stream them.
    h.client.chat.completions.create = lambda **k: _FakeStream(
        [
            _chunk(tool_calls=[_tc_delta(0, id="srv_1", name="move_head", arguments='{"direction"')]),
            _chunk(tool_calls=[_tc_delta(0, arguments=': "left"}')]),
            _chunk(usage=SimpleNamespace(prompt_tokens=20, completion_tokens=8)),
        ]
    )
    text, tools, usage, chat, _end = _drive(
        h,
        tools=[{"type": "function", "name": "move_head", "parameters": {"type": "object"}}],
        tool_choice="required",
    )
    assert len(tools) == 1
    tc = tools[0]
    assert isinstance(tc, ResponseFunctionToolCall)
    assert tc.name == "move_head"
    assert json.loads(tc.arguments) == {"direction": "left"}  # reassembled from two deltas
    assert usage == (20, 8)
    # the function_call was stored in history with a freshly minted call_id
    assert chat._pending_tool_calls, "tool call should be recorded in chat history"


def test_tool_call_recorded_before_chunk_is_emitted():
    """Regression: a fast client can return function_call_output before the
    deferred end-of-turn write-back runs. The call must already be in history
    the instant its chunk is yielded, otherwise the output is rejected with
    'No function_call with call_id ... found' and the model re-issues the call."""
    h = _make_handler(stream=True)
    h.client.chat.completions.create = lambda **k: _FakeStream(
        [
            _chunk(content="Sure."),
            _chunk(tool_calls=[_tc_delta(0, id="srv_1", name="camera_snapshot", arguments="{}")]),
            _chunk(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2)),
        ]
    )
    chat = Chat(10)
    chat.add_item(make_user_message("take a photo"))
    session = RealtimeSessionCreateRequest(type="realtime", instructions="Du bist ein Roboter.")
    session.tools = [{"type": "function", "name": "camera_snapshot", "parameters": {"type": "object"}}]
    rc = RuntimeConfig(chat=chat, session=session)
    req = GenerateResponseRequest(runtime_config=rc, language_code="de", turn_id="t", turn_revision=0)

    emitted_call_id = None
    for out in h.process(req):
        if isinstance(out, LLMResponseChunk) and out.tools:
            emitted_call_id = out.tools[0].call_id
            # At the moment the client receives the call, it must exist in history.
            assert emitted_call_id in chat._pending_tool_calls, (
                "function_call must be recorded BEFORE its chunk is forwarded to the client"
            )
            # A fast client returning the output here must pair cleanly (no raise).
            chat.add_item(
                RealtimeConversationItemFunctionCallOutput(
                    type="function_call_output", call_id=emitted_call_id, output="ok"
                )
            )
    assert emitted_call_id is not None, "a tool call should have been emitted"
    assert chat._has_call_id_in_buffer(emitted_call_id), "call+output should be paired in the buffer"


def test_private_poison_during_provider_stream_blocks_text_and_tool_history_writeback():
    """A post-handshake poison may race provider completion on another thread.

    Provider text and a tool call are normalized only after the streaming iterator
    finishes, so poisoning at that boundary must prevent both the trailing message
    commit and the eager function-call commit.
    """
    handler = _make_handler(stream=True)
    chat = Chat(10)
    chat.add_item(make_user_message("private request"))
    session = RealtimeSessionCreateRequest(type="realtime", instructions="Be concise.")
    session.tools = [{"type": "function", "name": "private_tool", "parameters": {"type": "object"}}]
    runtime_config = RuntimeConfig(
        chat=chat,
        session=session,
        transcript_barrier_version=1,
        transcript_barrier_nonce="ab" * 32,
    )
    request = GenerateResponseRequest(runtime_config=runtime_config, turn_id="private", turn_revision=0)
    handler.client.chat.completions.create = lambda **_kwargs: _PoisoningStream(
        [
            _chunk(content="PRIVATE_ASSISTANT_HISTORY_CANARY"),
            _chunk(tool_calls=[_tc_delta(0, id="srv_private", name="private_tool", arguments="{}")]),
        ],
        runtime_config,
    )

    outputs = list(handler.process(request))

    assert runtime_config.transcript_barrier_failed is True
    assert not any(getattr(item, "role", None) == "assistant" for item in chat.buffer)
    assert not any(getattr(item, "type", None) == "function_call" for item in chat.buffer)
    assert not any(isinstance(output, LLMResponseChunk) and output.tools for output in outputs)


def _guard_tools():
    return [
        {
            "type": "function",
            "name": "home_assistant__GetLiveContext",
            "parameters": {"type": "object", "properties": {"area": {"type": "string"}}},
        },
        {
            "type": "function",
            "name": "get_local_time",
            "parameters": {"type": "object", "properties": {}},
        },
    ]


def test_home_assistant_guard_releases_one_complete_native_call_after_validation():
    handler = _make_handler(stream=True)
    handler.client.chat.completions.create = lambda **_kwargs: _FakeStream(
        [
            _chunk(content="Let me check."),
            _chunk(
                tool_calls=[
                    _tc_delta(
                        0,
                        id="provider_ha",
                        name="home_assistant__GetLiveContext",
                        arguments='{"area":"bedroom"}',
                    )
                ]
            ),
            _chunk(finish_reason="tool_calls"),
            _chunk(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=4)),
        ]
    )

    text, tools, usage, chat, end = _drive(
        handler,
        tools=_guard_tools(),
        home_assistant_guard=True,
    )

    assert text == "Let me check."
    assert [tool.name for tool in tools] == ["home_assistant__GetLiveContext"]
    assert json.loads(tools[0].arguments) == {"area": "bedroom"}
    assert usage == (7, 4)
    assert end is not None and end.error is None
    assert len(chat._pending_tool_calls) == 1


def test_home_assistant_guard_preserves_complete_ordinary_speech():
    handler = _make_handler(stream=True)
    handler.client.chat.completions.create = lambda **_kwargs: _FakeStream(
        [
            _chunk(content="Hello there."),
            _chunk(finish_reason="stop"),
            _chunk(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2)),
        ]
    )

    text, tools, usage, chat, end = _drive(
        handler,
        tools=_guard_tools(),
        home_assistant_guard=True,
    )

    assert text == "Hello there."
    assert tools == []
    assert usage == (5, 2)
    assert end is not None and end.error is None
    assert any(getattr(item, "role", None) == "assistant" for item in chat.buffer)


@pytest.mark.parametrize(
    ("session_choice", "response_choice", "provider_call", "accepted"),
    [
        ("none", None, False, True),
        ("none", None, True, False),
        ("required", None, False, False),
        ("required", None, True, True),
        ("none", "auto", True, True),
        ("required", "none", True, False),
    ],
)
def test_home_assistant_guard_enforces_effective_tool_choice(
    session_choice,
    response_choice,
    provider_call,
    accepted,
):
    handler = _make_handler(stream=True)
    if provider_call:
        chunks = [
            _chunk(
                tool_calls=[
                    _tc_delta(
                        0,
                        id="provider_ha",
                        name="home_assistant__GetLiveContext",
                        arguments="{}",
                    )
                ]
            ),
            _chunk(finish_reason="tool_calls"),
        ]
    else:
        chunks = [_chunk(content="Hello."), _chunk(finish_reason="stop")]
    handler.client.chat.completions.create = lambda **_kwargs: _FakeStream(chunks)
    response = (
        RealtimeResponseCreateParams(conversation="none", tool_choice=response_choice)
        if response_choice is not None
        else None
    )

    text, tools, _usage, chat, end = _drive(
        handler,
        tools=_guard_tools(),
        tool_choice=session_choice,
        response=response,
        home_assistant_guard=True,
    )

    if accepted:
        assert end is not None and end.error is None
        assert ([tool.name for tool in tools] == ["home_assistant__GetLiveContext"]) is provider_call
        assert (text == "Hello.") is (not provider_call)
    else:
        assert text == ""
        assert tools == []
        assert end is not None and end.error == HOME_ASSISTANT_SELECTOR_REJECTED
        assert chat._pending_tool_calls == {}


@pytest.mark.parametrize(
    "chunks",
    [
        [
            _chunk(content="Checking now.", choice_index=0),
            _chunk(
                tool_calls=[
                    _tc_delta(
                        0,
                        id="provider_ha",
                        name="home_assistant__GetLiveContext",
                        arguments='{"area":"bedroom"}',
                    )
                ],
                choice_index=1,
            ),
            _chunk(finish_reason="tool_calls", choice_index=1),
        ],
        [
            _chunk(tool_calls=[_tc_delta(0, id="provider_ha", name="home_assistant__GetLiveContext", arguments="{}")]),
            _chunk(finish_reason="tool_calls"),
            _chunk(content="late payload"),
        ],
        [
            _chunk(content="Checking now.", choice_index=None),
            _chunk(finish_reason="stop", choice_index=None),
        ],
    ],
    ids=["cross-choice-splice", "payload-after-terminal", "missing-choice-index"],
)
def test_home_assistant_guard_rejects_ambiguous_raw_stream_envelope(chunks):
    handler = _make_handler(stream=True)
    handler.client.chat.completions.create = lambda **_kwargs: _FakeStream(chunks)

    text, tools, usage, chat, end = _drive(
        handler,
        tools=_guard_tools(),
        home_assistant_guard=True,
    )

    assert text == ""
    assert tools == []
    assert usage is None
    assert end is not None and end.error == HOME_ASSISTANT_SELECTOR_REJECTED
    assert not any(getattr(item, "role", None) == "assistant" for item in chat.buffer)
    assert chat._pending_tool_calls == {}


def test_home_assistant_guard_bounds_raw_tool_only_frames_before_normalization():
    class CountingToolStream(_FakeStream):
        def __init__(self):
            super().__init__([])
            self.frames_read = 0

        def __iter__(self):
            for index in range(MAX_GUARDED_PROVIDER_EVENTS + 2):
                self.frames_read += 1
                yield _chunk(
                    tool_calls=[
                        _tc_delta(
                            0,
                            id="provider_ha" if index == 0 else None,
                            name="home_assistant__GetLiveContext" if index == 0 else None,
                            arguments=" ",
                        )
                    ]
                )

    stream = CountingToolStream()
    handler = _make_handler(stream=True)
    handler.client.chat.completions.create = lambda **_kwargs: stream

    text, tools, usage, chat, end = _drive(
        handler,
        tools=_guard_tools(),
        home_assistant_guard=True,
    )

    assert stream.frames_read == MAX_GUARDED_PROVIDER_EVENTS + 1
    assert text == ""
    assert tools == []
    assert usage is None
    assert end is not None and end.error == HOME_ASSISTANT_SELECTOR_REJECTED
    assert chat._pending_tool_calls == {}


@pytest.mark.parametrize("stream", [True, False])
def test_home_assistant_guard_rejects_oversized_native_call_container_without_iterating(stream):
    class IterationBombToolCalls(list):
        def __iter__(self):
            raise AssertionError("guarded oversized tool-call containers must not be iterated")

    oversized = IterationBombToolCalls(
        _tc_delta(
            index,
            id=f"provider_{index}",
            name="home_assistant__GetLiveContext",
            arguments="{}",
        )
        for index in range(MAX_TOOL_CALLS_PER_RESPONSE + 1)
    )
    handler = _make_handler(stream=stream)
    if stream:
        handler.client.chat.completions.create = lambda **_kwargs: _FakeStream(
            [_chunk(tool_calls=oversized), _chunk(finish_reason="tool_calls")]
        )
    else:
        handler.client.chat.completions.create = lambda **_kwargs: SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="", tool_calls=oversized),
                    finish_reason="tool_calls",
                )
            ],
            usage=None,
        )

    text, tools, usage, chat, end = _drive(
        handler,
        tools=_guard_tools(),
        home_assistant_guard=True,
    )

    assert text == ""
    assert tools == []
    assert usage is None
    assert end is not None and end.error == HOME_ASSISTANT_SELECTOR_REJECTED
    assert not any(getattr(item, "role", None) == "assistant" for item in chat.buffer)
    assert chat._pending_tool_calls == {}


def test_home_assistant_guard_rejects_more_than_the_bounded_distinct_stream_indexes():
    handler = _make_handler(stream=True)
    handler.client.chat.completions.create = lambda **_kwargs: _FakeStream(
        [
            _chunk(
                tool_calls=[
                    _tc_delta(0, id="provider_0", name="home_assistant__GetLiveContext", arguments="{}"),
                    _tc_delta(1, id="provider_1", name="get_local_time", arguments="{}"),
                ]
            ),
            _chunk(tool_calls=[_tc_delta(2, id="provider_2", name="home_assistant__GetLiveContext", arguments="{}")]),
            _chunk(finish_reason="tool_calls"),
        ]
    )

    text, tools, usage, chat, end = _drive(
        handler,
        tools=_guard_tools(),
        home_assistant_guard=True,
    )

    assert text == ""
    assert tools == []
    assert usage is None
    assert end is not None and end.error == HOME_ASSISTANT_SELECTOR_REJECTED
    assert chat._pending_tool_calls == {}


def test_home_assistant_guard_bounds_cumulative_tool_fragments_during_normalization():
    handler = _make_handler(stream=True)
    runtime_config = RuntimeConfig(
        chat=Chat(10),
        session=RealtimeSessionCreateRequest(type="realtime"),
        home_assistant_guard_version=1,
        home_assistant_guard_nonce="cd" * 32,
        home_assistant_guard_contract_sha256="ef" * 32,
        home_assistant_guard_tool_count=1,
        home_assistant_guard_tool_names=("home_assistant__GetLiveContext",),
    )
    events = list(
        handler._iter_stream_events(
            _FakeStream(
                [
                    _chunk(
                        tool_calls=[
                            _tc_delta(
                                0,
                                id="provider_ha",
                                name="home_assistant__GetLiveContext",
                                arguments="x" * 9_000,
                            )
                        ]
                    ),
                    _chunk(tool_calls=[_tc_delta(0, arguments="y" * 9_000)]),
                    _chunk(finish_reason="tool_calls"),
                ]
            ),
            redact_private_content=runtime_config,
        )
    )

    accumulated = [event.item.arguments for event in events if isinstance(event, base_mod.ToolCall)]
    assert accumulated == ["x" * 9_000]
    assert any(isinstance(event, base_mod.MalformedProviderOutput) for event in events)


def test_home_assistant_guard_polls_cancellation_between_raw_tool_frames():
    cancel_scope = CancelScope()

    class CancellingToolStream(_FakeStream):
        def __init__(self):
            super().__init__([])
            self.frames_read = 0

        def __iter__(self):
            for index in range(MAX_GUARDED_PROVIDER_EVENTS):
                self.frames_read += 1
                yield _chunk(
                    tool_calls=[
                        _tc_delta(
                            0,
                            id="provider_ha" if index == 0 else None,
                            name="home_assistant__GetLiveContext" if index == 0 else None,
                            arguments=" ",
                        )
                    ]
                )
                if index == 0:
                    cancel_scope.cancel()

    stream = CancellingToolStream()
    handler = _make_handler(stream=True, cancel_scope=cancel_scope)
    handler.client.chat.completions.create = lambda **_kwargs: stream

    text, tools, usage, chat, end = _drive(
        handler,
        tools=_guard_tools(),
        home_assistant_guard=True,
    )

    assert stream.frames_read == 2
    assert text == ""
    assert tools == []
    assert usage is None
    assert end is not None and end.error is None
    assert chat._pending_tool_calls == {}


@pytest.mark.parametrize(
    "failure",
    [httpx.ReadTimeout("timed out"), ValueError("malformed provider frame")],
    ids=["read-timeout", "normalizer-error"],
)
def test_home_assistant_guard_maps_provider_failures_to_content_free_selector_rejection(failure):
    handler = _make_handler(stream=True)
    handler.client.chat.completions.create = lambda **_kwargs: _FailingStream(failure)

    text, tools, usage, chat, end = _drive(
        handler,
        tools=_guard_tools(),
        home_assistant_guard=True,
    )

    assert text == ""
    assert tools == []
    assert usage is None
    assert end is not None and end.error == HOME_ASSISTANT_SELECTOR_REJECTED
    assert not any(getattr(item, "role", None) == "assistant" for item in chat.buffer)
    assert chat._pending_tool_calls == {}


def test_home_assistant_guard_keeps_deliberate_cancellation_non_poisoning_on_provider_failure():
    cancel_scope = CancelScope()

    class CancelledFailureStream(_FakeStream):
        def __iter__(self):
            cancel_scope.cancel()
            raise httpx.ReadTimeout("cancelled blocked read")
            yield  # pragma: no cover - make this an iterator without exposing content

    handler = _make_handler(stream=True, cancel_scope=cancel_scope)
    handler.client.chat.completions.create = lambda **_kwargs: CancelledFailureStream([])

    text, tools, usage, chat, end = _drive(
        handler,
        tools=_guard_tools(),
        home_assistant_guard=True,
    )

    assert text == ""
    assert tools == []
    assert usage is None
    assert end is not None and end.error is None
    assert chat._pending_tool_calls == {}


@pytest.mark.parametrize(
    ("content", "tool_name", "arguments"),
    [
        ("Let me check. home_assistant__GetLiveContext(area='bedroom')", None, None),
        ("The Get_Local_Time tool says noon.", None, None),
        ("<tool_call>private</tool_call>", None, None),
        ("Let me use GetLiveContext to check.", "home_assistant__GetLiveContext", "{}"),
        ("Let me use `GetLiveContext` to check.", "home_assistant__GetLiveContext", "{}"),
        ("<function_call>GetLiveContext</function_call>", "home_assistant__GetLiveContext", "{}"),
        (r"Let me use \u0047etLiveContext to check.", "home_assistant__GetLiveContext", "{}"),
        ('{"name":"x","arguments":{}}', None, None),
        ("{'name':'x','arguments':{}}", None, None),
        ('Checking {"area":"bedroom"}.', "home_assistant__GetLiveContext", "{}"),
        ("Checking area=bedroom.", "home_assistant__GetLiveContext", "{}"),
        ("Checking fetch_state('bedroom').", "home_assistant__GetLiveContext", "{}"),
        ("Checking.", "unregistered_tool", "{}"),
        ("Checking.", "home_assistant__GetLiveContext", "[]"),
        ("Checking.", "home_assistant__GetLiveContext", '{"value":"' + "x" * 16_384 + '"}'),
        ("Checking. Still checking.", "home_assistant__GetLiveContext", "{}"),
    ],
)
def test_home_assistant_guard_rejects_complete_unsafe_selector_without_any_sink(
    content,
    tool_name,
    arguments,
):
    handler = _make_handler(stream=True)
    chunks = [_chunk(content=content)]
    if tool_name is not None:
        chunks.append(_chunk(tool_calls=[_tc_delta(0, id="provider_bad", name=tool_name, arguments=arguments)]))
    chunks.append(_chunk(finish_reason="tool_calls" if tool_name is not None else "stop"))
    handler.client.chat.completions.create = lambda **_kwargs: _FakeStream(chunks)

    text, tools, usage, chat, end = _drive(
        handler,
        tools=_guard_tools(),
        home_assistant_guard=True,
    )

    assert text == ""
    assert tools == []
    assert usage is None
    assert end is not None and end.error == HOME_ASSISTANT_SELECTOR_REJECTED
    assert not any(getattr(item, "role", None) == "assistant" for item in chat.buffer)
    assert chat._pending_tool_calls == {}


@pytest.mark.parametrize("stream", [True, False])
def test_home_assistant_guard_rejects_missing_tool_name_before_normalization(stream):
    handler = _make_handler(stream=stream)
    if stream:
        handler.client.chat.completions.create = lambda **_kwargs: _FakeStream(
            [
                _chunk(tool_calls=[_tc_delta(0, id="provider_bad", arguments='{"area":"bedroom"}')]),
                _chunk(finish_reason="tool_calls"),
            ]
        )
    else:
        handler.client.chat.completions.create = lambda **_kwargs: SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="provider_bad",
                                function=SimpleNamespace(name=None, arguments='{"area":"bedroom"}'),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=4),
        )

    text, tools, usage, chat, end = _drive(
        handler,
        tools=_guard_tools(),
        home_assistant_guard=True,
    )

    assert text == ""
    assert tools == []
    assert usage is None
    assert end is not None and end.error == HOME_ASSISTANT_SELECTOR_REJECTED
    assert chat._pending_tool_calls == {}


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    ("provider_id", "arguments"),
    [(None, "{}"), ("provider_bad", None), ("provider_bad", "")],
    ids=["missing-id", "missing-arguments", "empty-arguments"],
)
def test_home_assistant_guard_rejects_incomplete_native_call_before_normalization(
    stream,
    provider_id,
    arguments,
):
    handler = _make_handler(stream=stream)
    if stream:
        handler.client.chat.completions.create = lambda **_kwargs: _FakeStream(
            [
                _chunk(
                    tool_calls=[
                        _tc_delta(
                            0,
                            id=provider_id,
                            name="home_assistant__GetLiveContext",
                            arguments=arguments,
                        )
                    ]
                ),
                _chunk(finish_reason="tool_calls"),
            ]
        )
    else:
        handler.client.chat.completions.create = lambda **_kwargs: SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id=provider_id,
                                function=SimpleNamespace(
                                    name="home_assistant__GetLiveContext",
                                    arguments=arguments,
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=4),
        )

    text, tools, usage, chat, end = _drive(
        handler,
        tools=_guard_tools(),
        home_assistant_guard=True,
    )

    assert text == ""
    assert tools == []
    assert usage is None
    assert end is not None and end.error == HOME_ASSISTANT_SELECTOR_REJECTED
    assert chat._pending_tool_calls == {}


def test_home_assistant_guard_rejects_changed_provider_call_id_between_fragments():
    handler = _make_handler(stream=True)
    handler.client.chat.completions.create = lambda **_kwargs: _FakeStream(
        [
            _chunk(
                tool_calls=[
                    _tc_delta(
                        0,
                        id="provider_a",
                        name="home_assistant__GetLiveContext",
                        arguments="{",
                    )
                ]
            ),
            _chunk(tool_calls=[_tc_delta(0, id="provider_b", arguments="}")]),
            _chunk(finish_reason="tool_calls"),
        ]
    )

    text, tools, usage, chat, end = _drive(
        handler,
        tools=_guard_tools(),
        home_assistant_guard=True,
    )

    assert text == ""
    assert tools == []
    assert usage is None
    assert end is not None and end.error == HOME_ASSISTANT_SELECTOR_REJECTED
    assert chat._pending_tool_calls == {}


@pytest.mark.parametrize(
    "chunks",
    [
        [
            _chunk(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1)),
            _chunk(content="Hello."),
            _chunk(finish_reason="stop"),
        ],
        [
            _chunk(
                content="Hello.",
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            ),
            _chunk(finish_reason="stop"),
        ],
        [
            _chunk(content="Hello."),
            _chunk(finish_reason="stop"),
            _chunk(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1)),
            _chunk(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1)),
        ],
    ],
    ids=["early-usage", "choice-bearing-usage", "duplicate-trailing-usage"],
)
def test_home_assistant_guard_rejects_noncanonical_usage_frames(chunks):
    handler = _make_handler(stream=True)
    handler.client.chat.completions.create = lambda **_kwargs: _FakeStream(chunks)

    text, tools, usage, chat, end = _drive(
        handler,
        tools=_guard_tools(),
        home_assistant_guard=True,
    )

    assert text == ""
    assert tools == []
    assert usage is None
    assert end is not None and end.error == HOME_ASSISTANT_SELECTOR_REJECTED
    assert chat._pending_tool_calls == {}


@pytest.mark.parametrize(
    "tool_calls",
    [
        [
            _tc_delta(0, id="provider_same", name="get_local_time", arguments="{}"),
            _tc_delta(1, id="provider_same", name="home_assistant__GetLiveContext", arguments="{}"),
        ],
        [_tc_delta(1, id="provider_sparse", name="home_assistant__GetLiveContext", arguments="{}")],
    ],
    ids=["reused-provider-id", "sparse-tool-index"],
)
def test_home_assistant_guard_marks_reused_identity_or_sparse_index_malformed(tool_calls):
    handler = _make_handler(stream=True)
    runtime_config = RuntimeConfig(
        chat=Chat(10),
        session=RealtimeSessionCreateRequest(type="realtime"),
        home_assistant_guard_version=1,
        home_assistant_guard_nonce="cd" * 32,
        home_assistant_guard_contract_sha256="ef" * 32,
        home_assistant_guard_tool_count=2,
        home_assistant_guard_tool_names=("home_assistant__GetLiveContext", "get_local_time"),
    )

    events = list(
        handler._iter_stream_events(
            _FakeStream(
                [
                    _chunk(tool_calls=tool_calls),
                    _chunk(finish_reason="tool_calls"),
                ]
            ),
            redact_private_content=runtime_config,
        )
    )

    assert any(isinstance(event, base_mod.MalformedProviderOutput) for event in events)


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("finish_reason", [None, "length", "content_filter", "stop"])
def test_home_assistant_guard_requires_one_tool_calls_completion(stream, finish_reason):
    handler = _make_handler(stream=stream)
    if stream:
        chunks = [
            _chunk(tool_calls=[_tc_delta(0, id="provider_bad", name="home_assistant__GetLiveContext", arguments="{}")])
        ]
        if finish_reason is not None:
            chunks.append(_chunk(finish_reason=finish_reason))
        handler.client.chat.completions.create = lambda **_kwargs: _FakeStream(chunks)
    else:
        handler.client.chat.completions.create = lambda **_kwargs: SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="provider_bad",
                                function=SimpleNamespace(
                                    name="home_assistant__GetLiveContext",
                                    arguments="{}",
                                ),
                            )
                        ],
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=4),
        )

    text, tools, usage, chat, end = _drive(
        handler,
        tools=_guard_tools(),
        home_assistant_guard=True,
    )

    assert text == ""
    assert tools == []
    assert usage is None
    assert end is not None and end.error == HOME_ASSISTANT_SELECTOR_REJECTED
    assert chat._pending_tool_calls == {}


def test_home_assistant_guard_accepts_nonstreaming_complete_native_call():
    handler = _make_handler(stream=False)
    handler.client.chat.completions.create = lambda **_kwargs: SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Let me check.",
                    tool_calls=[
                        SimpleNamespace(
                            id="provider_ha",
                            function=SimpleNamespace(
                                name="home_assistant__GetLiveContext",
                                arguments='{"area":"bedroom"}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=4),
    )

    text, tools, usage, chat, end = _drive(
        handler,
        tools=_guard_tools(),
        home_assistant_guard=True,
    )

    assert text == "Let me check."
    assert [(tool.name, json.loads(tool.arguments)) for tool in tools] == [
        ("home_assistant__GetLiveContext", {"area": "bedroom"})
    ]
    assert usage == (7, 4)
    assert end is not None and end.error is None
    assert len(chat._pending_tool_calls) == 1


def test_home_assistant_guard_rejects_mixed_native_calls_but_preserves_non_ha_branch():
    handler = _make_handler(stream=True)
    handler.client.chat.completions.create = lambda **_kwargs: _FakeStream(
        [
            _chunk(
                tool_calls=[
                    _tc_delta(0, id="provider_ha", name="home_assistant__GetLiveContext", arguments="{}"),
                    _tc_delta(1, id="provider_time", name="get_local_time", arguments="{}"),
                ]
            ),
            _chunk(finish_reason="tool_calls"),
        ]
    )
    text, tools, _usage, chat, end = _drive(
        handler,
        tools=_guard_tools(),
        home_assistant_guard=True,
    )
    assert text == ""
    assert tools == []
    assert end is not None and end.error == HOME_ASSISTANT_SELECTOR_REJECTED
    assert chat._pending_tool_calls == {}

    handler.client.chat.completions.create = lambda **_kwargs: _FakeStream(
        [
            _chunk(content="One moment."),
            _chunk(tool_calls=[_tc_delta(0, id="provider_time", name="get_local_time", arguments="{}")]),
            _chunk(finish_reason="tool_calls"),
        ]
    )
    text, tools, _usage, _chat, end = _drive(
        handler,
        tools=_guard_tools(),
        home_assistant_guard=True,
    )
    assert text == "One moment."
    assert [tool.name for tool in tools] == ["get_local_time"]
    assert end is not None and end.error is None


def test_non_streaming_tool_call():
    h = _make_handler(stream=False)
    h.client.chat.completions.create = lambda **k: SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id="srv_9",
                            function=SimpleNamespace(name="move_head", arguments='{"direction": "right"}'),
                        )
                    ],
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
    )
    text, tools, usage, chat, _end = _drive(
        h,
        tools=[{"type": "function", "name": "move_head", "parameters": {"type": "object"}}],
        tool_choice="required",
    )
    assert len(tools) == 1 and tools[0].name == "move_head"
    assert json.loads(tools[0].arguments) == {"direction": "right"}
    assert usage == (7, 3)


def test_streaming_refusal_is_spoken_and_stored():
    """A refusal streams as delta.refusal (content None); it must be surfaced as
    assistant text and written to history, not silently dropped."""
    h = _make_handler(stream=True)
    h.client.chat.completions.create = lambda **k: _FakeStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, refusal="I cannot help with that.", tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            _chunk(usage=SimpleNamespace(prompt_tokens=4, completion_tokens=6)),
        ]
    )
    text, tools, usage, chat, _end = _drive(h)
    assert "I cannot help with that." in text
    assert any(getattr(i, "role", None) == "assistant" for i in chat.buffer)


def test_non_streaming_refusal_is_spoken_and_stored():
    h = _make_handler(stream=False)
    h.client.chat.completions.create = lambda **k: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, refusal="No can do.", tool_calls=[]))],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=2),
    )
    text, tools, usage, chat, _end = _drive(h)
    assert text == "No can do."
    assert any(getattr(i, "role", None) == "assistant" for i in chat.buffer)


def test_non_streaming_empty_choices_completes_cleanly():
    """A valid response with no choices (e.g. content filter) completes with no
    assistant text and no error, instead of raising IndexError."""
    h = _make_handler(stream=False)
    h.client.chat.completions.create = lambda **k: SimpleNamespace(
        choices=[], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0)
    )
    text, tools, usage, chat, end = _drive(h)
    assert text == ""
    assert tools == []
    assert end is not None and end.error is None  # clean end, not a generation failure


def test_tools_converted_to_chat_format_on_request():
    """The request sent to the server must carry Chat-Completions-shaped tools."""
    h = _make_handler(stream=True)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeStream([_chunk(content="ok.")])

    h.client.chat.completions.create = fake_create
    _drive(h, tools=[{"type": "function", "name": "f", "parameters": {"type": "object"}}], tool_choice="auto")
    assert captured["tools"] == [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    assert captured["tool_choice"] == "auto"
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}


# ── Text-only (output_modalities=["text"]) ────────────────────────────────────


def test_text_only_streaming_preserves_raw_deltas():
    """With output_modalities=["text"], deltas are forwarded verbatim: no
    remove_unspeechable (emoji/markdown survive) and no sentence batching."""
    h = _make_handler(stream=True)
    h.client.chat.completions.create = lambda **k: _FakeStream(
        [
            _chunk(content="# Title 🎉\n"),
            _chunk(content="- one\n- two 😀\n"),
            _chunk(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4)),
        ]
    )
    text, tools, usage, chat, end = _drive(h, response=RealtimeResponseCreateParams(output_modalities=["text"]))
    # Raw markdown layout and emoji preserved end-to-end.
    assert text == "# Title 🎉\n- one\n- two 😀\n"
    assert tools == []
    assert usage == (3, 4)
    # Raw assistant text is committed to history (not the filtered TTS string).
    assert any(getattr(i, "role", None) == "assistant" for i in chat.buffer), "assistant turn should be stored"


def test_text_only_tool_call_in_same_delta_not_dropped():
    """In text-only mode a delta can carry both content and a tool_call fragment;
    the tool_call must still be accumulated despite the verbatim-forward `continue`."""
    h = _make_handler(stream=True)
    h.client.chat.completions.create = lambda **k: _FakeStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="Looking it up. ",
                            tool_calls=[_tc_delta(0, id="srv_1", name="search", arguments='{"q":"x"}')],
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            _chunk(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5)),
        ]
    )
    text, tools, usage, chat, _end = _drive(
        h,
        tools=[{"type": "function", "name": "search", "parameters": {"type": "object"}}],
        response=RealtimeResponseCreateParams(output_modalities=["text"]),
    )
    assert "Looking it up." in text
    assert len(tools) == 1 and tools[0].name == "search"  # not dropped by the text-only continue
    assert json.loads(tools[0].arguments) == {"q": "x"}


def test_non_streaming_text_only_preserves_symbols():
    h = _make_handler(stream=False)
    h.client.chat.completions.create = lambda **k: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="**bold** 🎉", tool_calls=[]))],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=2),
    )
    text, tools, usage, chat, end = _drive(h, response=RealtimeResponseCreateParams(output_modalities=["text"]))
    assert text == "**bold** 🎉"  # symbols not stripped


# ── tool_choice decoupled from tools ──────────────────────────────────────────


def test_tool_choice_sent_without_tools():
    """A session-level tool_choice must reach the server even when no tools list
    is supplied (e.g. tool_choice="none" to suppress tool use)."""
    h = _make_handler(stream=True)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeStream([_chunk(content="ok.")])

    h.client.chat.completions.create = fake_create
    _drive(h, tool_choice="none")
    assert "tools" not in captured
    assert captured["tool_choice"] == "none"


# ── Error propagation ─────────────────────────────────────────────────────────


def test_empty_input_emits_failed_end_of_response():
    """No instructions and no conversation input → terminating EndOfResponse with
    an error, instead of an opaque provider 400."""
    h = _make_handler(stream=True)
    called = {"n": 0}

    def fake_create(**kwargs):
        called["n"] += 1
        return _FakeStream([_chunk(content="should not happen")])

    h.client.chat.completions.create = fake_create
    # Empty chat + empty instructions => nothing to send.
    text, tools, usage, chat, end = _drive(h, user="", instructions="", chat=Chat(10))
    assert called["n"] == 0, "no API call should be made when there is nothing to send"
    assert end is not None and end.error is not None
    assert text == ""


def test_generation_error_emits_failed_end_of_response():
    """An exception during generation is caught and surfaced on EndOfResponse.error
    so the response is closed instead of leaving the pipeline stuck."""
    h = _make_handler(stream=True)

    def boom(**kwargs):
        raise RuntimeError("kaboom")

    h.client.chat.completions.create = boom
    text, tools, usage, chat, end = _drive(h)
    assert end is not None and end.error is not None
    assert "kaboom" in end.error


def test_private_barrier_generation_error_scrubs_exception_content(caplog):
    h = _make_handler(stream=True)
    canary = "PRIVATE_GENERATION_CANARY"

    def boom(**kwargs):
        raise RuntimeError(canary)

    h.client.chat.completions.create = boom
    with caplog.at_level(logging.DEBUG):
        _text, _tools, _usage, _chat, end = _drive(h, private_barrier=True)

    assert end is not None
    assert end.error == "Language model generation failed in private transcript mode."
    assert canary not in caplog.text


def test_generated_content_log_holds_guard_through_concurrent_poison(monkeypatch):
    handler = _make_handler(stream=False)
    chat = Chat(10)
    chat.add_item(make_user_message("ordinary before poison"))
    runtime_config = RuntimeConfig(
        chat=chat,
        session=RealtimeSessionCreateRequest(type="realtime", instructions="Reply briefly."),
    )
    request = GenerateResponseRequest(runtime_config=runtime_config, turn_id="turn", turn_revision=0)
    poison_attempted = threading.Event()
    poison_completed = threading.Event()
    poison_thread: threading.Thread | None = None
    original_debug = base_mod.logger.debug

    def capture_debug(message, *args, **kwargs):
        nonlocal poison_thread
        if message == "Clean text: %s":

            def poison() -> None:
                poison_attempted.set()
                with runtime_config.transcript_barrier_state_guard():
                    runtime_config.transcript_barrier_failed = True
                    runtime_config.chat.enable_private_content_logging()
                poison_completed.set()

            poison_thread = threading.Thread(target=poison)
            poison_thread.start()
            assert poison_attempted.wait(timeout=1.0)
            assert not poison_completed.wait(timeout=0.05)
        original_debug(message, *args, **kwargs)

    monkeypatch.setattr(base_mod.logger, "debug", capture_debug)
    outputs = list(handler.process(request))

    assert any(isinstance(output, EndOfResponse) for output in outputs)
    assert poison_thread is not None
    poison_thread.join(timeout=1.0)
    assert not poison_thread.is_alive()
    assert poison_completed.is_set()
    assert runtime_config.transcript_barrier_failed is True


def test_generated_tool_log_holds_guard_through_concurrent_poison(monkeypatch):
    handler = _make_handler(stream=False)
    tool_canary = "ORDINARY_TOOL_BEFORE_POISON"
    handler.client.chat.completions.next_result = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_provider",
                            function=SimpleNamespace(name=tool_canary, arguments="{}"),
                        )
                    ],
                )
            )
        ],
        usage=None,
    )
    chat = Chat(10)
    chat.add_item(make_user_message("ordinary before poison"))
    runtime_config = RuntimeConfig(chat=chat, session=RealtimeSessionCreateRequest(type="realtime"))
    request = GenerateResponseRequest(runtime_config=runtime_config, turn_id="turn", turn_revision=0)
    poison_attempted = threading.Event()
    poison_completed = threading.Event()
    poison_thread: threading.Thread | None = None
    original_info = base_mod.logger.info

    def capture_info(message, *args, **kwargs):
        nonlocal poison_thread
        if message == "Tools: %s":

            def poison() -> None:
                poison_attempted.set()
                with runtime_config.transcript_barrier_state_guard():
                    runtime_config.transcript_barrier_failed = True
                    runtime_config.chat.enable_private_content_logging()
                poison_completed.set()

            poison_thread = threading.Thread(target=poison)
            poison_thread.start()
            assert poison_attempted.wait(timeout=1.0)
            assert not poison_completed.wait(timeout=0.05)
            assert tool_canary in str(args)
        original_info(message, *args, **kwargs)

    monkeypatch.setattr(base_mod.logger, "info", capture_info)
    list(handler.process(request))

    assert poison_thread is not None
    poison_thread.join(timeout=1.0)
    assert not poison_thread.is_alive()
    assert poison_completed.is_set()


def test_generation_error_log_holds_guard_through_concurrent_poison(monkeypatch):
    handler = _make_handler(stream=True)
    error_canary = "ORDINARY_ERROR_BEFORE_POISON"
    chat = Chat(10)
    chat.add_item(make_user_message("ordinary before poison"))
    runtime_config = RuntimeConfig(chat=chat, session=RealtimeSessionCreateRequest(type="realtime"))
    request = GenerateResponseRequest(runtime_config=runtime_config, turn_id="turn", turn_revision=0)
    handler.client.chat.completions.create = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(error_canary))
    poison_attempted = threading.Event()
    poison_completed = threading.Event()
    poison_thread: threading.Thread | None = None

    def capture_exception(message, *_args, **_kwargs):
        nonlocal poison_thread
        assert message == "LLM generation failed; ending the current response"

        def poison() -> None:
            poison_attempted.set()
            with runtime_config.transcript_barrier_state_guard():
                runtime_config.transcript_barrier_failed = True
                runtime_config.chat.enable_private_content_logging()
            poison_completed.set()

        poison_thread = threading.Thread(target=poison)
        poison_thread.start()
        assert poison_attempted.wait(timeout=1.0)
        assert not poison_completed.wait(timeout=0.05)

    monkeypatch.setattr(base_mod.logger, "exception", capture_exception)
    outputs = list(handler.process(request))

    assert poison_thread is not None
    poison_thread.join(timeout=1.0)
    assert not poison_thread.is_alive()
    assert poison_completed.is_set()
    end = next(output for output in outputs if isinstance(output, EndOfResponse))
    assert end.error in {
        f"Language model generation failed: {error_canary}",
        "Language model generation failed in private transcript mode.",
    }


# ── Out-of-band (conversation="none") responses ───────────────────────────────


def test_out_of_band_does_not_commit_to_default_conversation():
    """Out-of-band output is emitted but never written back to the default chat."""
    h = _make_handler(stream=True)
    h.client.chat.completions.create = lambda **k: _FakeStream(
        [_chunk(content="Background note."), _chunk(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))]
    )
    chat = Chat(10)
    text, tools, usage, chat, end = _drive(
        h, chat=chat, response=RealtimeResponseCreateParams(conversation="none", output_modalities=["text"])
    )
    assert "Background note." in text
    # Default conversation keeps only the seeded user turn — no assistant commit.
    assert not any(getattr(i, "role", None) == "assistant" for i in chat.buffer)


def test_private_out_of_band_validation_error_is_content_free(caplog):
    handler = _make_handler(stream=True)
    canary = "PRIVATE_OOB_LLM_EXCEPTION_CANARY"
    orphan = RealtimeConversationItemFunctionCallOutput(
        type="function_call_output",
        call_id=canary,
        output="{}",
    )
    response = RealtimeResponseCreateParams(conversation="none", input=[orphan])

    with caplog.at_level(logging.INFO):
        _text, _tools, _usage, _chat, end = _drive(
            handler,
            response=response,
            private_barrier=True,
        )

    assert end is not None
    assert end.error == "Private out-of-band response rejected."
    assert canary not in caplog.text


# ── Standalone runner (no pytest required) ────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
