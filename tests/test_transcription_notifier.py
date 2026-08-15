import logging
from queue import Queue
from threading import Event

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.events import (
    PartialTranscriptionEvent,
    TranscriptBarrierCompletedEvent,
    TranscriptBarrierDiscardedEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.messages import GenerateResponseRequest, PartialTranscription, Transcription
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier


def _notifier(
    text_output_queue: Queue | None = None,
    runtime_config: RuntimeConfig | None = None,
    should_listen: Event | None = None,
    barrier_enabled: bool = False,
) -> TranscriptionNotifier:
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(
        text_output_queue=text_output_queue,
        runtime_config=runtime_config,
        should_listen=should_listen,
        transcript_barrier_enabled=lambda: barrier_enabled,
    )
    return notifier


def test_empty_final_transcription_still_emits_completion_after_partial():
    text_output_queue = Queue()
    notifier = _notifier(text_output_queue=text_output_queue)

    assert list(notifier.process(PartialTranscription(text="Yeah."))) == []
    assert list(notifier.process(Transcription(text="", language_code="en", speech_stopped_at_s=123.0))) == []

    partial = text_output_queue.get_nowait()
    completed = text_output_queue.get_nowait()

    assert isinstance(partial, PartialTranscriptionEvent)
    assert partial.delta == "Yeah."
    assert isinstance(completed, TranscriptionCompletedEvent)
    assert completed.transcript == ""
    assert completed.language_code == "en"
    assert completed.speech_stopped_at_s == 123.0
    assert text_output_queue.empty()


def test_empty_final_transcription_does_not_trigger_legacy_generation():
    runtime_config = RuntimeConfig()
    should_listen = Event()
    notifier = _notifier(runtime_config=runtime_config, should_listen=should_listen)

    assert list(notifier.process(Transcription(text="", language_code="en"))) == []
    assert should_listen.is_set()


def test_non_empty_final_transcription_still_triggers_legacy_generation():
    runtime_config = RuntimeConfig()
    should_listen = Event()
    notifier = _notifier(runtime_config=runtime_config, should_listen=should_listen)

    result = list(notifier.process(Transcription(text="hello", language_code="en", speech_stopped_at_s=123.0)))

    assert len(result) == 1
    assert isinstance(result[0], GenerateResponseRequest)
    assert result[0].runtime_config is runtime_config
    assert result[0].language_code == "en"
    assert result[0].speech_stopped_at_s == 123.0
    assert not should_listen.is_set()


def test_non_empty_final_transcription_logs_full_text_at_info(caplog):
    notifier = _notifier()
    transcript = "hello " * 30

    with caplog.at_level(logging.INFO, logger="speech_to_speech.STT.transcription_notifier"):
        assert list(notifier.process(Transcription(text=transcript, language_code="en"))) == []

    assert "Transcription completed (language=en): " + transcript in caplog.text


def test_empty_final_transcription_reenables_listening_without_runtime_config():
    should_listen = Event()
    notifier = _notifier(should_listen=should_listen)

    assert list(notifier.process(Transcription(text="", language_code="en"))) == []

    assert should_listen.is_set()


def test_private_barrier_suppresses_partials_and_reserves_the_final_without_logging(caplog):
    text_output_queue = Queue()
    notifier = _notifier(text_output_queue=text_output_queue, barrier_enabled=True)
    canary = "Josh private transcript canary"

    with caplog.at_level(logging.DEBUG, logger="speech_to_speech.STT.transcription_notifier"):
        assert list(notifier.process(PartialTranscription(text=canary))) == []
        assert list(notifier.process(Transcription(text=canary, language_code="en"))) == []

    event = text_output_queue.get_nowait()
    assert isinstance(event, TranscriptBarrierCompletedEvent)
    assert event.transcript == canary
    assert text_output_queue.empty()
    assert canary not in caplog.text


def test_poisoned_private_session_keeps_stt_redaction_sticky_while_work_drains(caplog):
    text_output_queue = Queue()
    runtime_config = RuntimeConfig()
    runtime_config.transcript_barrier_version = 1
    runtime_config.transcript_barrier_nonce = "ab" * 32
    runtime_config.transcript_barrier_failed = True
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(
        text_output_queue=text_output_queue,
        transcript_barrier_enabled=lambda: runtime_config.transcript_barrier_enabled,
    )
    canary = "PRIVATE_STT_AFTER_POISON_CANARY"

    with caplog.at_level(logging.DEBUG, logger="speech_to_speech.STT.transcription_notifier"):
        assert list(notifier.process(PartialTranscription(text=canary))) == []
        assert list(notifier.process(Transcription(text=canary, language_code="en"))) == []

    event = text_output_queue.get_nowait()
    assert isinstance(event, TranscriptBarrierCompletedEvent)
    assert event.transcript == canary
    assert canary not in caplog.text


def test_private_barrier_discards_whitespace_without_a_placeholder_or_response():
    text_output_queue = Queue()
    should_listen = Event()
    notifier = _notifier(
        text_output_queue=text_output_queue,
        should_listen=should_listen,
        barrier_enabled=True,
    )

    assert list(notifier.process(Transcription(text=" \t\n", language_code="en"))) == []

    assert isinstance(text_output_queue.get_nowait(), TranscriptBarrierDiscardedEvent)
    assert text_output_queue.empty()
    assert should_listen.is_set()
