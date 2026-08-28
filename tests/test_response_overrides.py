from __future__ import annotations

from collections.abc import Iterator
from threading import Event, Lock
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock

from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.language_model import (
    BaseLanguageModelHandler,
    LanguageModelHandler,
    StreamContext,
    _CancelCriteria,
)
from speech_to_speech.pipeline.messages import GenerateResponseRequest, LLMResponseChunk


class _RecordingLocalHandler(BaseLanguageModelHandler):
    def _load_model(
        self,
        model_name: str,
        device: str,
        torch_dtype: str,
        gen_kwargs: dict[str, Any],
    ) -> None:
        pass

    def _generate(
        self,
        chat: Chat,
        language_code: Optional[str],
        gen: int | None,
        ctx: StreamContext,
        runtime_config: RuntimeConfig | None = None,
        response: RealtimeResponseCreateParams | None = None,
    ) -> Iterator[LLMResponseChunk]:
        self.seen_chat = chat.copy(deep=True)
        self.seen_function_tools = list(ctx.function_tools)
        return
        yield


def test_local_backend_preserves_explicitly_empty_response_overrides():
    handler = object.__new__(_RecordingLocalHandler)
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.enable_lang_prompt = False
    handler.compactor = None
    handler.tokenizer = SimpleNamespace(encode=lambda _text: [])
    chat = Chat(10)
    chat.add_item(make_user_message("Answer without tools."))
    session = RealtimeSessionCreateRequest(
        type="realtime",
        instructions="SESSION INSTRUCTIONS",
        tools=[{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
    )
    request = GenerateResponseRequest(
        runtime_config=RuntimeConfig(chat=chat, session=session),
        response=RealtimeResponseCreateParams(instructions="", tools=[]),
    )

    list(handler.process(request))

    assert [item.type for item in handler.seen_chat.buffer] == ["message"]
    assert handler.seen_function_tools == []


def test_local_transformer_cancellation_during_serialization_skips_inference():
    handler = object.__new__(LanguageModelHandler)
    handler.backend = "transformers"
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.stop_event = Event()
    handler._cancel_criteria = _CancelCriteria()
    handler._transformers_lock = Lock()
    handler.gen_kwargs = {}
    handler.device = "cpu"
    handler.pipe = MagicMock()
    handler.streamer = []

    chat = Chat(10)
    chat.add_item(make_user_message("PRIVATE_LOCAL_INPUT"))
    request = GenerateResponseRequest(
        runtime_config=RuntimeConfig(
            chat=chat,
            session=RealtimeSessionCreateRequest(type="realtime", instructions="test"),
        )
    )

    class CancellingTokenizer:
        def apply_chat_template(self, _messages: Any, *, tokenize: bool, **_kwargs: Any) -> Any:
            if tokenize:
                return [1]
            request.cancel()
            return "PRIVATE_SERIALIZED_PROMPT"

    handler.tokenizer = CancellingTokenizer()
    ctx = StreamContext(request=request, cancel_event=request.cancel_event)

    output = list(handler._generate(chat, None, None, ctx, request.runtime_config))

    assert output == []
    assert ctx.cancelled
    handler.pipe.assert_not_called()
