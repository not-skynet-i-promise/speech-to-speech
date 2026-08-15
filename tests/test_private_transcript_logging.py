"""Cross-pipeline canaries for the sticky private-content logging boundary."""

import logging
from collections.abc import Iterator
from queue import Empty, Queue
from threading import Event, Thread

import numpy as np
import pytest
from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.responses import ResponseFunctionToolCall

import speech_to_speech.LLM.chat as chat_module
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import Chat, make_assistant_message, make_user_message
from speech_to_speech.LLM.language_model import (
    BaseLanguageModelHandler,
    StreamContext,
    _CancelCriteria,
)
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import (
    PIPELINE_END,
    EndOfResponse,
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


class _NeverCalledLanguageModelHandler(BaseLanguageModelHandler):
    """Provider-free local handler whose generation hook must stay unreachable."""

    def _load_model(self, model_name, device, torch_dtype, gen_kwargs):
        raise AssertionError("test bypasses setup")

    def _generate(self, *args, **kwargs):
        raise AssertionError("stale queued request reached local model generation")
        yield LLMResponseChunk()


class _ExplodingLanguageModelHandler(BaseLanguageModelHandler):
    """Provider-free handler used to exercise the terminal failure boundary."""

    failure: BaseException

    def _load_model(self, model_name, device, torch_dtype, gen_kwargs):
        raise AssertionError("test bypasses setup")

    def _generate(self, *args, **kwargs):
        raise self.failure
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


def test_compactor_exception_log_is_linearized_with_late_privacy(monkeypatch):
    chat = Chat(size=2)
    for index in range(4):
        chat.add_item(make_user_message(f"ordinary user {index}"))
        chat.add_item(make_assistant_message(f"ordinary assistant {index}"))
    privacy_attempted = Event()
    privacy_completed = Event()
    privacy_thread: Thread | None = None

    def explode(_snapshot):
        raise RuntimeError("COMPACTOR_EXCEPTION_BEFORE_PRIVACY")

    def capture_exception(message, *_args, **_kwargs):
        nonlocal privacy_thread
        assert message == "Chat compaction failed; chat unchanged"

        def enable_privacy() -> None:
            privacy_attempted.set()
            chat.enable_private_content_logging()
            privacy_completed.set()

        privacy_thread = Thread(target=enable_privacy)
        privacy_thread.start()
        assert privacy_attempted.wait(timeout=1.0)
        assert not privacy_completed.wait(timeout=0.05)

    monkeypatch.setattr(chat_module.logger, "exception", capture_exception)
    chat.trim_if_needed(explode)
    assert chat._compact_thread is not None
    chat._compact_thread.join(timeout=2.0)

    assert not chat._compact_thread.is_alive()
    assert privacy_thread is not None
    privacy_thread.join(timeout=1.0)
    assert not privacy_thread.is_alive()
    assert privacy_completed.is_set()


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


def test_local_llm_drops_request_cancelled_before_dequeue():
    handler = object.__new__(_NeverCalledLanguageModelHandler)
    handler.speculative_turns = None
    handler.cancel_scope = CancelScope()

    request = GenerateResponseRequest(
        runtime_config=RuntimeConfig(),
        cancel_generation=handler.cancel_scope.generation,
    )
    handler.cancel_scope.cancel()

    assert list(handler.process(request)) == [EndOfResponse(cancel_generation=0)]


class _ImmediateTimeoutStreamer:
    def __init__(self):
        self.ended = False

    def end(self):
        self.ended = True

    def __iter__(self):
        return self

    def __next__(self):
        raise Empty


class _FailingEndStreamer(_ImmediateTimeoutStreamer):
    def end(self):
        self.ended = True
        raise RuntimeError("PRIVATE_STREAMER_END_FAILURE_CANARY")


def test_transformers_worker_outliving_join_keeps_private_activation_blocked(monkeypatch):
    handler = object.__new__(_NeverCalledLanguageModelHandler)
    handler.cancel_scope = CancelScope()
    worker_started = Event()
    release_worker = Event()

    def blocked_worker() -> None:
        worker_started.set()
        assert release_worker.wait(timeout=2.0)

    with handler.cancel_scope.response_admission(0) as (admitted, _generation):
        assert admitted is True
        streamer = _ImmediateTimeoutStreamer()
        worker, worker_state = handler._start_transformers_generation(blocked_worker, streamer, _private_config())
        assert worker_started.wait(timeout=1.0)
        real_join = worker.join
        monkeypatch.setattr(worker, "join", lambda timeout=None: real_join(timeout=0.01))
        handler._finish_transformers_generation(
            worker,
            streamer,
            _CancelCriteria(),
            worker_state,
        )
        assert worker.is_alive()

    with handler.cancel_scope.private_activation_guard() as quiescent:
        assert quiescent is False

    release_worker.set()
    real_join(timeout=1.0)
    assert not worker.is_alive()
    with handler.cancel_scope.private_activation_guard() as quiescent:
        assert quiescent is True


def test_transformers_generations_use_isolated_streamers_and_cancel_criteria():
    handler = object.__new__(_NeverCalledLanguageModelHandler)
    handler.tokenizer = object()

    first_streamer, first_criteria = handler._new_transformers_streaming_state()
    second_streamer, second_criteria = handler._new_transformers_streaming_state()
    first_criteria.cancel()

    assert first_streamer is not second_streamer
    assert first_criteria is not second_criteria
    assert first_criteria(None, None) is True
    assert second_criteria(None, None) is False


def test_private_transformers_worker_exception_is_redacted_and_unblocks_streamer(caplog, capsys):
    canary = "PRIVATE_LOCAL_TRANSFORMERS_WORKER_EXCEPTION_CANARY"
    handler = object.__new__(_NeverCalledLanguageModelHandler)
    handler.cancel_scope = CancelScope()
    streamer = _ImmediateTimeoutStreamer()

    def explode() -> None:
        raise RuntimeError(canary)

    caplog.set_level(logging.DEBUG)
    worker, worker_state = handler._start_transformers_generation(explode, streamer, _private_config())
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert worker_state.failed is True
    assert streamer.ended is True
    assert canary not in caplog.text
    assert canary not in capsys.readouterr().err
    assert "private content redacted" in caplog.text
    with pytest.raises(RuntimeError, match="^Local Transformers generation worker failed$"):
        handler._finish_transformers_generation(worker, streamer, _CancelCriteria(), worker_state)
    with handler.cancel_scope.private_activation_guard() as quiescent:
        assert quiescent is True


def test_transformers_thread_constructor_failure_releases_activation_lease(monkeypatch):
    handler = object.__new__(_NeverCalledLanguageModelHandler)
    handler.cancel_scope = CancelScope()

    def fail_constructor(*_args, **_kwargs):
        raise RuntimeError("thread construction failed")

    monkeypatch.setattr("speech_to_speech.LLM.language_model.Thread", fail_constructor)
    with pytest.raises(RuntimeError, match="^thread construction failed$"):
        handler._start_transformers_generation(
            lambda: None,
            _ImmediateTimeoutStreamer(),
            _private_config(),
        )

    with handler.cancel_scope.private_activation_guard() as quiescent:
        assert quiescent is True


@pytest.mark.parametrize("failure_site", ["constructor", "start"])
def test_private_transformers_thread_base_exception_is_normalized(monkeypatch, failure_site, capsys):
    canary = f"PRIVATE_THREAD_{failure_site.upper()}_BASE_EXCEPTION_CANARY"
    handler = object.__new__(_NeverCalledLanguageModelHandler)
    handler.cancel_scope = CancelScope()

    class ThreadFault(BaseException):
        pass

    if failure_site == "constructor":

        def fail_thread(*_args, **_kwargs):
            raise ThreadFault(canary)

        monkeypatch.setattr("speech_to_speech.LLM.language_model.Thread", fail_thread)
    else:

        class FailStartThread:
            def __init__(self, *_args, **_kwargs):
                pass

            def start(self):
                raise ThreadFault(canary)

        monkeypatch.setattr("speech_to_speech.LLM.language_model.Thread", FailStartThread)

    with pytest.raises(RuntimeError, match="^Local Transformers generation worker could not start$"):
        handler._start_transformers_generation(
            lambda: None,
            _ImmediateTimeoutStreamer(),
            _private_config(),
        )

    assert canary not in capsys.readouterr().err
    with handler.cancel_scope.private_activation_guard() as quiescent:
        assert quiescent is True


def test_private_transformers_worker_survives_logger_failure_without_excepthook(monkeypatch, capsys):
    target_canary = "PRIVATE_LOCAL_WORKER_TARGET_CANARY"
    logger_canary = "PRIVATE_LOCAL_WORKER_LOGGER_CANARY"
    handler = object.__new__(_NeverCalledLanguageModelHandler)
    handler.cancel_scope = CancelScope()
    streamer = _ImmediateTimeoutStreamer()

    def explode_target() -> None:
        raise RuntimeError(target_canary)

    def explode_logger(*_args, **_kwargs) -> None:
        raise RuntimeError(logger_canary)

    monkeypatch.setattr("speech_to_speech.LLM.language_model.logger.error", explode_logger)
    worker, worker_state = handler._start_transformers_generation(
        explode_target,
        streamer,
        _private_config(),
    )
    worker.join(timeout=1.0)
    stderr = capsys.readouterr().err

    assert not worker.is_alive()
    assert worker_state.failed is True
    assert streamer.ended is True
    assert target_canary not in stderr
    assert logger_canary not in stderr
    assert "Exception in thread" not in stderr
    with handler.cancel_scope.private_activation_guard() as quiescent:
        assert quiescent is True


def test_private_generation_base_exception_returns_content_free_terminal(capsys):
    canary = "PRIVATE_GENERATION_BASE_EXCEPTION_CANARY"

    class GenerationFault(BaseException):
        pass

    handler = object.__new__(_ExplodingLanguageModelHandler)
    handler.speculative_turns = None
    handler.cancel_scope = None
    handler.enable_lang_prompt = False
    handler.compactor = None
    handler.tokenizer = type("Tokenizer", (), {"encode": staticmethod(lambda _text: [])})()
    handler.failure = GenerationFault(canary)
    runtime_config = _private_config()
    runtime_config.chat.add_item(make_user_message("private request"))

    outputs = list(handler.process(GenerateResponseRequest(runtime_config=runtime_config)))

    assert outputs == [
        EndOfResponse(error="Language model generation failed in private transcript mode."),
    ]
    assert canary not in capsys.readouterr().err


@pytest.mark.parametrize("failure_site", ["logger", "guard"])
def test_private_generation_failure_always_returns_content_free_terminal(
    monkeypatch,
    capsys,
    failure_site,
):
    secondary_canary = f"PRIVATE_{failure_site.upper()}_FAILURE_CANARY"
    handler = object.__new__(_ExplodingLanguageModelHandler)
    handler.speculative_turns = None
    handler.cancel_scope = None
    handler.enable_lang_prompt = False
    handler.compactor = None
    handler.tokenizer = type("Tokenizer", (), {"encode": staticmethod(lambda _text: [])})()
    handler.failure = RuntimeError("PRIVATE_GENERATION_FAILURE_CANARY")
    runtime_config = _private_config()
    runtime_config.chat.add_item(make_user_message("private request"))

    if failure_site == "logger":

        def fail_logger(*_args, **_kwargs):
            raise RuntimeError(secondary_canary)

        monkeypatch.setattr("speech_to_speech.LLM.language_model.logger.error", fail_logger)
    else:

        class GuardFault(BaseException):
            pass

        def fail_guard(_self):
            raise GuardFault(secondary_canary)

        monkeypatch.setattr(RuntimeConfig, "transcript_barrier_content_guard", fail_guard)

    outputs = list(handler.process(GenerateResponseRequest(runtime_config=runtime_config)))
    stderr = capsys.readouterr().err

    assert outputs == [
        EndOfResponse(error="Language model generation failed in private transcript mode."),
    ]
    assert "PRIVATE_GENERATION_FAILURE_CANARY" not in stderr
    assert secondary_canary not in stderr
    assert "Exception in thread" not in stderr


def test_private_streamer_end_failure_cannot_strand_transformers_consumer(caplog, capsys):
    target_canary = "PRIVATE_TRANSFORMERS_TARGET_FAILURE_CANARY"
    handler = object.__new__(_NeverCalledLanguageModelHandler)
    handler.cancel_scope = CancelScope()
    handler.speculative_turns = None
    handler.stop_event = Event()
    streamer = _FailingEndStreamer()

    def explode_target() -> None:
        raise RuntimeError(target_canary)

    worker, worker_state = handler._start_transformers_generation(
        explode_target,
        streamer,
        _private_config(),
    )
    ctx = StreamContext()
    outputs: list[LLMResponseChunk] = []
    consumer = Thread(
        target=lambda: outputs.extend(
            handler._stream_tokens(
                streamer,
                None,
                None,
                ctx,
                _private_config(),
                worker_state=worker_state,
            )
        )
    )
    consumer.start()
    consumer.join(timeout=1.0)
    worker.join(timeout=1.0)

    assert not consumer.is_alive()
    assert not worker.is_alive()
    assert outputs == []
    assert worker_state.failed is True
    assert worker_state.completed.is_set()
    with pytest.raises(RuntimeError, match="^Local Transformers generation worker failed$"):
        handler._finish_transformers_generation(worker, streamer, _CancelCriteria(), worker_state)
    stderr = capsys.readouterr().err
    assert target_canary not in caplog.text
    assert "PRIVATE_STREAMER_END_FAILURE_CANARY" not in caplog.text
    assert target_canary not in stderr
    assert "PRIVATE_STREAMER_END_FAILURE_CANARY" not in stderr
    assert "Exception in thread" not in stderr
    with handler.cancel_scope.private_activation_guard() as quiescent:
        assert quiescent is True
