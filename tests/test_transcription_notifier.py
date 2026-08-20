import logging
from queue import Queue
from threading import Event, Thread

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
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
    barrier_failed: bool = False,
) -> TranscriptionNotifier:
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(
        text_output_queue=text_output_queue,
        runtime_config=runtime_config,
        should_listen=should_listen,
        transcript_barrier_enabled=lambda: barrier_enabled,
        transcript_barrier_failed=lambda: barrier_failed,
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
    runtime_config.transcript_barrier_failed = True
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(
        text_output_queue=text_output_queue,
        transcript_barrier_enabled=lambda: runtime_config.transcript_barrier_private,
        transcript_barrier_failed=lambda: runtime_config.transcript_barrier_failed,
    )
    canary = "PRIVATE_STT_AFTER_POISON_CANARY"

    with caplog.at_level(logging.DEBUG, logger="speech_to_speech.STT.transcription_notifier"):
        assert list(notifier.process(PartialTranscription(text=canary))) == []
        assert list(notifier.process(Transcription(text=canary, language_code="en"))) == []

    assert text_output_queue.empty()
    assert canary not in caplog.text


def test_home_assistant_guard_redacts_live_stt_and_drops_it_after_poison(caplog) -> None:
    text_output_queue = Queue()
    runtime_config = RuntimeConfig(
        home_assistant_guard_version=1,
        home_assistant_guard_nonce="ab" * 32,
        home_assistant_guard_contract_sha256="cd" * 32,
        home_assistant_guard_tool_count=1,
        home_assistant_guard_tool_names=("home_assistant__GetLiveContext",),
    )
    notifier = _notifier(text_output_queue=text_output_queue, runtime_config=runtime_config)
    canary = "PRIVATE_HOME_TRANSCRIPT_CANARY"

    with caplog.at_level(logging.INFO, logger="speech_to_speech.STT.transcription_notifier"):
        initial = list(notifier.process(Transcription(text=canary, language_code="en")))
        runtime_config.fail_home_assistant_guard()
        assert list(notifier.process(Transcription(text="LATE_PRIVATE_CANARY", language_code="en"))) == []

    assert len(initial) == 1 and isinstance(initial[0], GenerateResponseRequest)
    event = text_output_queue.get_nowait()
    assert isinstance(event, TranscriptionCompletedEvent)
    assert event.transcript == canary
    assert text_output_queue.empty()
    assert canary not in caplog.text
    assert "LATE_PRIVATE_CANARY" not in caplog.text


def test_realtime_state_guard_linearizes_notifier_side_effects_before_later_poison(caplog):
    service = RealtimeService()
    connection_id = service.register()
    poison_attempted = Event()
    poison_completed = Event()
    poison_thread: Thread | None = None

    class PoisoningQueue(Queue):
        def put(self, item, block=True, timeout=None):
            nonlocal poison_thread

            def poison() -> None:
                poison_attempted.set()
                service.poison_transcript_barrier(connection_id, "test_failure")
                poison_completed.set()

            poison_thread = Thread(target=poison)
            poison_thread.start()
            assert poison_attempted.wait(timeout=1.0)
            assert not poison_completed.wait(timeout=0.05)
            return super().put(item, block=block, timeout=timeout)

    text_output_queue = PoisoningQueue()
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(
        text_output_queue=text_output_queue,
        transcript_barrier_state_guard=service.transcript_barrier_pipeline_state_guard,
    )

    with caplog.at_level(logging.INFO, logger="speech_to_speech.STT.transcription_notifier"):
        assert list(notifier.process(Transcription(text="ordinary before poison", language_code="en"))) == []

    assert poison_thread is not None
    poison_thread.join(timeout=1.0)
    assert not poison_thread.is_alive()
    assert poison_completed.is_set()
    assert service.transcript_barrier_failed(connection_id)
    assert isinstance(text_output_queue.get_nowait(), TranscriptionCompletedEvent)


def test_broken_realtime_state_guard_fails_closed_without_content(caplog):
    text_output_queue = Queue()
    notifier = object.__new__(TranscriptionNotifier)

    def broken_guard():
        raise RuntimeError("PRIVATE_GUARD_FAILURE_CANARY")

    notifier.setup(
        text_output_queue=text_output_queue,
        transcript_barrier_state_guard=broken_guard,
    )

    with caplog.at_level(logging.DEBUG, logger="speech_to_speech.STT.transcription_notifier"):
        assert list(notifier.process(Transcription(text="PRIVATE_TRANSCRIPT_CANARY", language_code="en"))) == []

    assert text_output_queue.empty()
    assert "PRIVATE_GUARD_FAILURE_CANARY" not in caplog.text
    assert "PRIVATE_TRANSCRIPT_CANARY" not in caplog.text
    assert "private content redacted" in caplog.text


def test_realtime_state_guard_without_one_live_connection_drops_stale_content(caplog):
    text_output_queue = Queue()
    service = RealtimeService()
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(
        text_output_queue=text_output_queue,
        transcript_barrier_state_guard=service.transcript_barrier_pipeline_state_guard,
    )

    with caplog.at_level(logging.DEBUG, logger="speech_to_speech.STT.transcription_notifier"):
        assert list(notifier.process(Transcription(text="STALE_TRANSCRIPT_CANARY", language_code="en"))) == []

    assert text_output_queue.empty()
    assert "STALE_TRANSCRIPT_CANARY" not in caplog.text


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
