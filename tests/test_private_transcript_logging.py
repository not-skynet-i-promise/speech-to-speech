"""Cross-pipeline canaries for the sticky private-content logging boundary."""

import logging
from collections.abc import Iterator
from queue import Queue
from threading import Event, Thread

import numpy as np
from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import Chat, make_assistant_message, make_user_message
from speech_to_speech.LLM.language_model import BaseLanguageModelHandler, StreamContext
from speech_to_speech.pipeline.messages import (
    PIPELINE_END,
    GenerateResponseRequest,
    LLMResponseChunk,
    Transcription,
    TTSInput,
    VADAudio,
)
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.TTS.facebookmms_handler import FacebookMMSTTSHandler


def _private_config() -> RuntimeConfig:
    config = RuntimeConfig()
    config.transcript_barrier_version = 1
    config.transcript_barrier_nonce = "ab" * 32
    config.chat.enable_private_content_logging()
    return config


def _rejected_private_config() -> RuntimeConfig:
    config = RuntimeConfig()
    config.transcript_barrier_failed = True
    config.chat.enable_private_content_logging()
    return config


class _ExplodingHandler(BaseHandler[TTSInput, bytes]):
    def process(self, _input: TTSInput):
        raise RuntimeError("PRIVATE_BASE_HANDLER_EXCEPTION_CANARY")
        yield b"unreachable"


class _BlockingExplodingHandler(BaseHandler[TTSInput, bytes]):
    def __init__(self, stop_event, queue_in, queue_out, started: Event, release: Event):
        self.started = started
        self.release = release
        super().__init__(stop_event, queue_in, queue_out)

    def process(self, _input: TTSInput):
        self.started.set()
        assert self.release.wait(timeout=2.0)
        raise RuntimeError("PRIVATE_INFLIGHT_HANDLER_CANARY")
        yield b"unreachable"


class _BlockingCleanupHandler(BaseHandler[TTSInput, bytes]):
    def __init__(self, stop_event, queue_in, queue_out, started: Event, release: Event):
        self.started = started
        self.release = release
        super().__init__(stop_event, queue_in, queue_out)

    def process(self, _input: TTSInput):
        self.started.set()
        assert self.release.wait(timeout=2.0)
        if False:
            yield b"unreachable"

    def cleanup(self) -> None:
        raise RuntimeError("PRIVATE_INFLIGHT_CLEANUP_CANARY")


class _BlockingExplodingSTTHandler(BaseSTTHandler):
    def __init__(self, stop_event, queue_in, queue_out, started: Event, release: Event):
        self.started = started
        self.release = release
        super().__init__(stop_event, queue_in, queue_out)

    def process(self, _input) -> Iterator[Transcription]:
        self.started.set()
        assert self.release.wait(timeout=2.0)
        raise RuntimeError("PRIVATE_STT_HANDLER_EXCEPTION_CANARY")
        yield Transcription(text="unreachable")


class _PoisoningLanguageModelHandler(BaseLanguageModelHandler):
    """Provider-free local handler that poisons immediately before write-back."""

    def _load_model(self, model_name, device, torch_dtype, gen_kwargs):
        raise AssertionError("test bypasses setup")

    def _generate(
        self,
        chat,
        language_code,
        gen,
        ctx: StreamContext,
        runtime_config=None,
        response=None,
    ) -> Iterator[LLMResponseChunk]:
        assert runtime_config is not None
        ctx.generated_text = "PRIVATE_LOCAL_ASSISTANT_HISTORY_CANARY"
        ctx.tools = [
            ResponseFunctionToolCall(
                type="function_call",
                call_id="call_private",
                name="private_tool",
                arguments="{}",
            )
        ]
        runtime_config.transcript_barrier_failed = True
        if False:
            yield LLMResponseChunk()


def test_generic_handler_exception_is_content_free_in_private_mode(caplog):
    queue_in: Queue = Queue()
    queue_out: Queue = Queue()
    handler = _ExplodingHandler(Event(), queue_in, queue_out)
    worker = Thread(target=handler.run)

    with caplog.at_level(logging.ERROR, logger="speech_to_speech.baseHandler"):
        worker.start()
        queue_in.put(TTSInput(text="private", runtime_config=_private_config()))
        queue_in.put(PIPELINE_END)
        worker.join(timeout=2.0)

    assert worker.is_alive() is False
    assert "PRIVATE_BASE_HANDLER_EXCEPTION_CANARY" not in caplog.text
    assert "private content redacted" in caplog.text


def test_generic_handler_drops_work_after_rejected_private_activation(caplog):
    queue_in: Queue = Queue()
    queue_out: Queue = Queue()
    handler = _ExplodingHandler(Event(), queue_in, queue_out)
    worker = Thread(target=handler.run)

    with caplog.at_level(logging.DEBUG, logger="speech_to_speech.baseHandler"):
        worker.start()
        queue_in.put(TTSInput(text="private", runtime_config=_rejected_private_config()))
        queue_in.put(PIPELINE_END)
        worker.join(timeout=2.0)

    assert worker.is_alive() is False
    assert "PRIVATE_BASE_HANDLER_EXCEPTION_CANARY" not in caplog.text
    assert "dropping input after private barrier failure" in caplog.text


def test_generic_handler_rechecks_privacy_when_poisoned_while_processing(caplog):
    queue_in: Queue = Queue()
    queue_out: Queue = Queue()
    started = Event()
    release = Event()
    handler = _BlockingExplodingHandler(Event(), queue_in, queue_out, started, release)
    worker = Thread(target=handler.run)
    runtime_config = RuntimeConfig()

    with caplog.at_level(logging.ERROR, logger="speech_to_speech.baseHandler"):
        worker.start()
        queue_in.put(TTSInput(text="ordinary before poison", runtime_config=runtime_config))
        assert started.wait(timeout=2.0)
        with runtime_config.transcript_barrier_state_guard():
            runtime_config.transcript_barrier_failed = True
            runtime_config.chat.enable_private_content_logging()
        release.set()
        queue_in.put(PIPELINE_END)
        worker.join(timeout=2.0)

    assert worker.is_alive() is False
    assert "PRIVATE_INFLIGHT_HANDLER_CANARY" not in caplog.text
    assert "private content redacted" in caplog.text


def test_generic_handler_latches_poison_before_final_cleanup(caplog):
    queue_in: Queue = Queue()
    queue_out: Queue = Queue()
    started = Event()
    release = Event()
    handler = _BlockingCleanupHandler(Event(), queue_in, queue_out, started, release)
    worker = Thread(target=handler.run)
    runtime_config = RuntimeConfig()

    with caplog.at_level(logging.ERROR, logger="speech_to_speech.baseHandler"):
        worker.start()
        queue_in.put(TTSInput(text="ordinary before poison", runtime_config=runtime_config))
        assert started.wait(timeout=2.0)
        with runtime_config.transcript_barrier_state_guard():
            runtime_config.transcript_barrier_failed = True
            runtime_config.chat.enable_private_content_logging()
        release.set()
        queue_in.put(PIPELINE_END)
        worker.join(timeout=2.0)

    assert worker.is_alive() is False
    assert "PRIVATE_INFLIGHT_CLEANUP_CANARY" not in caplog.text
    assert "final cleanup failed; private content redacted" in caplog.text


def test_stt_exception_uses_live_barrier_guard_after_concurrent_poison(caplog):
    queue_in: Queue = Queue()
    queue_out: Queue = Queue()
    started = Event()
    release = Event()
    service = RealtimeService()
    connection_id = service.register()
    handler = _BlockingExplodingSTTHandler(Event(), queue_in, queue_out, started, release)
    handler.set_transcript_barrier_enabled(service.transcript_barrier_private)
    handler.set_transcript_barrier_failed(service.transcript_barrier_poisoned)
    handler.set_transcript_barrier_state_guard(service.transcript_barrier_pipeline_state_guard)
    worker = Thread(target=handler.run)

    with caplog.at_level(logging.ERROR, logger="speech_to_speech.baseHandler"):
        worker.start()
        queue_in.put(VADAudio(audio=np.zeros(160, dtype=np.float32)))
        assert started.wait(timeout=2.0)
        service.poison_transcript_barrier(connection_id, "test_failure")
        release.set()
        queue_in.put(PIPELINE_END)
        worker.join(timeout=2.0)

    assert worker.is_alive() is False
    assert "PRIVATE_STT_HANDLER_EXCEPTION_CANARY" not in caplog.text
    assert "private content redacted" in caplog.text


def test_facebook_mms_exception_has_no_private_traceback(caplog):
    class _ExplodingTokenizer:
        def __call__(self, *_args, **_kwargs):
            raise RuntimeError("PRIVATE_TTS_EXCEPTION_CANARY")

    handler = object.__new__(FacebookMMSTTSHandler)
    handler.language = "en"
    handler.tokenizer = _ExplodingTokenizer()
    handler.device = "cpu"

    with caplog.at_level(logging.DEBUG, logger="speech_to_speech.TTS.facebookmms_handler"):
        assert handler.generate_audio("PRIVATE_TTS_TEXT_CANARY", redact_content=True) is None

    assert "PRIVATE_TTS_EXCEPTION_CANARY" not in caplog.text
    assert "PRIVATE_TTS_TEXT_CANARY" not in caplog.text
    assert "private content redacted" in caplog.text


def test_facebook_mms_rechecks_privacy_when_poisoned_during_generation(caplog):
    class _TensorInputs:
        input_ids = __import__("torch").tensor([[1]])
        attention_mask = __import__("torch").tensor([[1]])

    class _Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return _TensorInputs()

    started = Event()
    release = Event()

    class _BlockingModel:
        def __call__(self, **_kwargs):
            started.set()
            assert release.wait(timeout=2.0)
            raise RuntimeError("PRIVATE_FACEBOOK_PROVIDER_CANARY")

    handler = object.__new__(FacebookMMSTTSHandler)
    handler.language = "en"
    handler.tokenizer = _Tokenizer()
    handler.model = _BlockingModel()
    handler.device = "cpu"
    runtime_config = RuntimeConfig()
    result: list[object] = []

    def generate() -> None:
        result.append(
            handler.generate_audio(
                "ordinary before poison",
                runtime_config=runtime_config,
            )
        )

    worker = Thread(target=generate)
    with caplog.at_level(logging.DEBUG, logger="speech_to_speech.TTS.facebookmms_handler"):
        worker.start()
        assert started.wait(timeout=2.0)
        with runtime_config.transcript_barrier_state_guard():
            runtime_config.transcript_barrier_failed = True
            runtime_config.chat.enable_private_content_logging()
        release.set()
        worker.join(timeout=2.0)

    assert worker.is_alive() is False
    assert result == [None]
    assert "PRIVATE_FACEBOOK_PROVIDER_CANARY" not in caplog.text
    assert "private content redacted" in caplog.text


def test_compactor_exception_is_content_free_after_private_handshake(caplog):
    chat = Chat(size=2)
    chat.enable_private_content_logging()
    for index in range(4):
        chat.add_item(make_user_message(f"private user {index}"))
        chat.add_item(make_assistant_message(f"private assistant {index}"))

    def explode(_snapshot):
        raise RuntimeError("PRIVATE_COMPACTOR_EXCEPTION_CANARY")

    with caplog.at_level(logging.ERROR, logger="speech_to_speech.LLM.chat"):
        chat.trim_if_needed(explode)
        assert chat._compact_thread is not None
        chat._compact_thread.join(timeout=2.0)

    assert "PRIVATE_COMPACTOR_EXCEPTION_CANARY" not in caplog.text
    assert "private content redacted" in caplog.text


def test_local_llm_poison_during_generation_blocks_assistant_and_tool_history_writeback():
    handler = object.__new__(_PoisoningLanguageModelHandler)
    handler.speculative_turns = None
    handler.cancel_scope = None
    handler.enable_lang_prompt = False
    handler.compactor = None
    handler.tokenizer = type("Tokenizer", (), {"encode": staticmethod(lambda _text: [])})()

    chat = Chat(10)
    chat.add_item(make_user_message("private request"))
    runtime_config = RuntimeConfig(
        chat=chat,
        session=RealtimeSessionCreateRequest(type="realtime"),
        transcript_barrier_version=1,
        transcript_barrier_nonce="ab" * 32,
    )

    list(handler.process(GenerateResponseRequest(runtime_config=runtime_config)))

    assert runtime_config.transcript_barrier_failed is True
    assert not any(getattr(item, "role", None) == "assistant" for item in chat.buffer)
    assert not any(getattr(item, "type", None) == "function_call" for item in chat.buffer)
