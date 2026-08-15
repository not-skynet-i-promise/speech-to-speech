import io
import logging
from contextlib import contextmanager
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
from rich.text import Text

from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.STT import parakeet_tdt_handler
from speech_to_speech.STT.parakeet_tdt_handler import ParakeetTDTSTTHandler
from speech_to_speech.STT.smart_progressive_streaming import SmartProgressiveStreamingHandler


def test_show_progressive_transcription_returns_combined_text(monkeypatch):
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.streaming_handler = SimpleNamespace(
        transcribe_incremental=lambda audio: SimpleNamespace(
            fixed_text="I just wanted",
            active_text="to check in",
        )
    )
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    result = handler._show_progressive_transcription(np.zeros(16000, dtype=np.float32))

    assert result == "I just wanted to check in"


def test_private_barrier_suppresses_progressive_console_content(monkeypatch):
    calls = []
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.set_transcript_barrier_enabled(lambda: True)
    handler.streaming_handler = SimpleNamespace(
        transcribe_incremental=lambda audio: SimpleNamespace(
            fixed_text="PRIVATE NAME CANARY",
            active_text="PRIVATE TRANSCRIPT CANARY",
        )
    )
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: calls.append(args))

    result = handler._show_progressive_transcription(np.zeros(16000, dtype=np.float32))

    assert result == "PRIVATE NAME CANARY PRIVATE TRANSCRIPT CANARY"
    assert calls == []


def test_progressive_console_holds_live_barrier_guard_through_content_sink(monkeypatch):
    calls = []
    service = RealtimeService()
    connection_id = service.register()
    poison_attempted = Event()
    poison_completed = Event()
    poison_thread: Thread | None = None
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.set_transcript_barrier_enabled(service.transcript_barrier_private)
    handler.set_transcript_barrier_failed(service.transcript_barrier_poisoned)
    handler.set_transcript_barrier_state_guard(service.transcript_barrier_pipeline_state_guard)
    handler.streaming_handler = SimpleNamespace(
        transcribe_incremental=lambda audio: SimpleNamespace(
            fixed_text="PRIVATE STT RACE",
            active_text="CANARY",
        )
    )

    def capture_print(*args, **kwargs):
        nonlocal poison_thread

        def poison() -> None:
            poison_attempted.set()
            service.poison_transcript_barrier(connection_id, "test_failure")
            poison_completed.set()

        poison_thread = Thread(target=poison)
        poison_thread.start()
        assert poison_attempted.wait(timeout=1.0)
        assert not poison_completed.wait(timeout=0.05)
        calls.append((args, kwargs))

    monkeypatch.setattr(parakeet_tdt_handler.console, "print", capture_print)

    result = handler._show_progressive_transcription(np.zeros(16000, dtype=np.float32))

    assert result == "PRIVATE STT RACE CANARY"
    assert len(calls) == 1
    assert poison_thread is not None
    poison_thread.join(timeout=1.0)
    assert not poison_thread.is_alive()
    assert poison_completed.is_set()
    assert service.transcript_barrier_failed(connection_id)

    handler._show_progressive_transcription(np.zeros(16000, dtype=np.float32))
    assert len(calls) == 1


def test_broken_live_barrier_guard_suppresses_progressive_content(monkeypatch, caplog):
    calls = []
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.streaming_handler = SimpleNamespace(
        transcribe_incremental=lambda audio: SimpleNamespace(
            fixed_text="PRIVATE BROKEN GUARD",
            active_text="CANARY",
        )
    )

    def broken_guard():
        raise RuntimeError("PRIVATE STT GUARD ERROR CANARY")

    handler.set_transcript_barrier_state_guard(broken_guard)
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: calls.append(args))

    with caplog.at_level(logging.ERROR, logger="speech_to_speech.STT.base_stt_handler"):
        result = handler._show_progressive_transcription(np.zeros(16000, dtype=np.float32))

    assert result == "PRIVATE BROKEN GUARD CANARY"
    assert calls == []
    assert "PRIVATE BROKEN GUARD CANARY" not in caplog.text
    assert "PRIVATE STT GUARD ERROR CANARY" not in caplog.text
    assert "private content redacted" in caplog.text


def test_live_barrier_guard_without_exactly_one_connection_suppresses_content(monkeypatch):
    calls = []
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.streaming_handler = SimpleNamespace(
        transcribe_incremental=lambda audio: SimpleNamespace(
            fixed_text="STALE CONNECTION",
            active_text="CANARY",
        )
    )
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: calls.append(args))

    for connection_count in (0, 2):
        service = RealtimeService()
        for _ in range(connection_count):
            service.register()
        handler.set_transcript_barrier_state_guard(service.transcript_barrier_pipeline_state_guard)

        result = handler._show_progressive_transcription(np.zeros(16000, dtype=np.float32))

        assert result == "STALE CONNECTION CANARY"

    assert calls == []


def test_live_transcription_clears_terminal_line_before_each_update(monkeypatch):
    calls = []

    class FakeConsole:
        is_terminal = True
        width = 80

        def __init__(self):
            self.file = io.StringIO()

        def print(self, *args, **kwargs):
            calls.append((args, kwargs))

    handler = object.__new__(ParakeetTDTSTTHandler)
    handler._live_transcription_active = False
    fake_console = FakeConsole()
    monkeypatch.setattr(parakeet_tdt_handler, "console", fake_console)

    handler._print_live_transcription(Text("Live: first"), "first")
    handler._print_live_transcription(Text("Live: second"), "second")
    handler._clear_live_transcription_line()

    assert [args[0].plain for args, _ in calls] == ["Live: first", "Live: second"]
    assert [kwargs for _, kwargs in calls] == [{"end": ""}, {"end": ""}]
    assert fake_console.file.getvalue() == "\r\x1b[2K\r\r\x1b[2K\r\r\x1b[2K"
    assert handler._live_transcription_active is False


def test_live_transcription_truncates_terminal_updates(monkeypatch):
    calls = []

    class FakeConsole:
        is_terminal = True
        width = 14

        def __init__(self):
            self.file = io.StringIO()

        def print(self, *args, **kwargs):
            calls.append((args, kwargs))

    handler = object.__new__(ParakeetTDTSTTHandler)
    handler._live_transcription_active = False
    monkeypatch.setattr(parakeet_tdt_handler, "console", FakeConsole())

    handler._print_live_transcription(Text("Live: abcdefghijklmnopqrstuvwxyz"), "abcdefghijklmnopqrstuvwxyz")

    printed_text = calls[0][0][0]
    assert printed_text.plain == "Live: abcdef\u2026"
    assert len(printed_text.plain) == 13


def test_live_transcription_uses_lines_for_non_terminal_logs(monkeypatch):
    calls = []

    class FakeConsole:
        is_terminal = False

        def print(self, *args, **kwargs):
            calls.append((args, kwargs))

    handler = object.__new__(ParakeetTDTSTTHandler)
    handler._live_transcription_active = False
    monkeypatch.setattr(parakeet_tdt_handler, "console", FakeConsole())

    handler._print_live_transcription(Text("Live: first"), "first")

    assert calls == [((Text("Live: first"),), {})]
    assert handler._live_transcription_active is False


def test_process_yields_partial_tagged_tuple(monkeypatch):
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = True
    handler.processing_final = False

    @contextmanager
    def fake_lock(*args, **kwargs):
        yield True

    handler._compute_lock_context = fake_lock
    handler._show_progressive_transcription = lambda audio: "partial text"
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    result = list(handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32), mode="progressive")))

    assert len(result) == 1
    assert isinstance(result[0], PartialTranscription)
    assert result[0].text == "partial text"


def test_process_yields_final_transcript(monkeypatch):
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = False
    handler.backend = "nano_parakeet"
    handler.last_language = "en"
    handler.start_language = None

    @contextmanager
    def fake_lock(*args, **kwargs):
        yield True

    handler._compute_lock_context = fake_lock
    handler._process_nano_parakeet = lambda audio_input: ("I am here.", "en")
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    result = list(handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32))))

    assert len(result) == 1
    assert isinstance(result[0], Transcription)
    assert result[0].text == "I am here."
    assert result[0].language_code == "en"


def test_private_barrier_suppresses_final_console_content(monkeypatch):
    calls = []
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.set_transcript_barrier_enabled(lambda: True)
    handler.enable_live_transcription = False
    handler.backend = "nano_parakeet"
    handler.last_language = "en"
    handler.start_language = None

    @contextmanager
    def fake_lock(*args, **kwargs):
        yield True

    handler._compute_lock_context = fake_lock
    handler._process_nano_parakeet = lambda audio_input: ("PRIVATE FINAL CANARY", "en")
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: calls.append(args))

    result = list(handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32), mode="final")))

    assert result[0].text == "PRIVATE FINAL CANARY"
    assert calls == []


def test_final_console_holds_live_barrier_guard_through_all_content_sinks(monkeypatch):
    calls = []
    service = RealtimeService()
    connection_id = service.register()
    poison_attempted = Event()
    poison_completed = Event()
    poison_thread: Thread | None = None
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.set_transcript_barrier_enabled(service.transcript_barrier_private)
    handler.set_transcript_barrier_failed(service.transcript_barrier_poisoned)
    handler.set_transcript_barrier_state_guard(service.transcript_barrier_pipeline_state_guard)
    handler.enable_live_transcription = False
    handler.backend = "nano_parakeet"
    handler.last_language = "en"
    handler.start_language = None

    @contextmanager
    def fake_lock(*args, **kwargs):
        yield True

    handler._compute_lock_context = fake_lock
    handler._process_nano_parakeet = lambda audio_input: ("PRIVATE FINAL RACE CANARY", "en")

    def capture_print(*args, **kwargs):
        nonlocal poison_thread
        calls.append((args, kwargs))
        if poison_thread is not None:
            return

        def poison() -> None:
            poison_attempted.set()
            service.poison_transcript_barrier(connection_id, "test_failure")
            poison_completed.set()

        poison_thread = Thread(target=poison)
        poison_thread.start()
        assert poison_attempted.wait(timeout=1.0)
        assert not poison_completed.wait(timeout=0.05)

    monkeypatch.setattr(parakeet_tdt_handler.console, "print", capture_print)

    result = list(handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32), mode="final")))

    assert result[0].text == "PRIVATE FINAL RACE CANARY"
    assert len(calls) == 2
    assert poison_thread is not None
    poison_thread.join(timeout=1.0)
    assert not poison_thread.is_alive()
    assert poison_completed.is_set()
    assert service.transcript_barrier_failed(connection_id)

    list(handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32), mode="final")))
    assert len(calls) == 2


def test_parakeet_timing_logs_only_final_transcriptions():
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler._times = [0.01]

    assert handler.timing_log_level == logging.INFO
    assert handler.should_log_timing(Transcription(text="I am here.", language_code="en"))
    assert not handler.should_log_timing(PartialTranscription(text="I am"))


def test_final_transcription_resets_live_streaming_state(monkeypatch):
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = True
    handler.backend = "nano_parakeet"
    handler.last_language = "en"
    handler.start_language = None
    handler.processing_final = False
    handler._live_turn_key = (None, None)
    reset_calls = []
    handler.streaming_handler = SimpleNamespace(reset=lambda: reset_calls.append(True))

    @contextmanager
    def fake_lock(*args, **kwargs):
        yield True

    handler._compute_lock_context = fake_lock
    handler._process_nano_parakeet = lambda audio_input: ("I am here.", "en")
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    result = list(handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32), mode="final")))

    assert len(result) == 1
    assert isinstance(result[0], Transcription)
    assert handler.processing_final is False
    assert reset_calls == [True]


def test_turn_change_resets_live_streaming_state_before_progressive(monkeypatch):
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = True
    handler.processing_final = False
    handler._live_turn_key = ("turn_1", 0)
    reset_calls = []
    handler.streaming_handler = SimpleNamespace(reset=lambda: reset_calls.append(True))

    @contextmanager
    def fake_lock(*args, **kwargs):
        yield True

    handler._compute_lock_context = fake_lock
    handler._show_progressive_transcription = lambda audio: "new partial"
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    result = list(
        handler.process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="progressive",
                turn_id="turn_2",
                turn_revision=0,
            )
        )
    )

    assert reset_calls == [True]
    assert len(result) == 1
    assert isinstance(result[0], PartialTranscription)
    assert result[0].text == "new partial"


def test_mlx_final_ignores_fixed_text_that_exceeds_current_audio(monkeypatch):
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = True
    handler.backend = "mlx"
    handler.last_language = "en"
    handler.start_language = None
    handler.processing_final = False
    handler._live_turn_key = ("turn_3", 0)
    handler.streaming_handler = SimpleNamespace(
        fixed_sentences=["stale previous transcript"],
        fixed_end_time=10.0,
        reset=lambda: None,
    )

    @contextmanager
    def fake_lock(*args, **kwargs):
        yield True

    handler._compute_lock_context = fake_lock
    handler._process_mlx = lambda audio_input: ("new short turn", "en")
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    result = list(
        handler.process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="final",
                turn_id="turn_3",
                turn_revision=0,
            )
        )
    )

    assert len(result) == 1
    assert isinstance(result[0], Transcription)
    assert result[0].text == "new short turn"


def test_final_transcription_prevents_stale_fixed_window_on_next_progressive(monkeypatch):
    class Model:
        def __init__(self):
            self.progressive_window_lengths = []

        def transcribe(self, audio, timestamps=True):
            self.progressive_window_lengths.append(len(audio))
            return SimpleNamespace(text="new partial", timestamp={"segment": []})

    model = Model()
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = True
    handler.backend = "nano_parakeet"
    handler.last_language = "en"
    handler.start_language = None
    handler.processing_final = False
    handler.streaming_handler = SmartProgressiveStreamingHandler(model)
    handler.streaming_handler.fixed_sentences = ["previous fixed sentence"]
    handler.streaming_handler.fixed_end_time = 10.0
    handler.streaming_handler.last_transcribed_length = 20 * 16000

    @contextmanager
    def fake_lock(*args, **kwargs):
        yield True

    handler._compute_lock_context = fake_lock
    handler._process_nano_parakeet = lambda audio_input: ("previous final", "en")
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    final_result = list(handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32), mode="final")))
    progressive_audio = np.zeros(852 * 16, dtype=np.float32)
    progressive_result = list(handler.process(VADAudio(audio=progressive_audio, mode="progressive")))

    assert len(final_result) == 1
    assert isinstance(final_result[0], Transcription)
    assert model.progressive_window_lengths == [len(progressive_audio)]
    assert len(progressive_result) == 1
    assert isinstance(progressive_result[0], PartialTranscription)
    assert progressive_result[0].text == "new partial"


def test_on_session_end_resets_streaming_state():
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.start_language = "en"
    handler.enable_live_transcription = True
    handler.processing_final = True
    reset_calls = []
    handler.streaming_handler = SimpleNamespace(reset=lambda: reset_calls.append(True))

    handler.on_session_end()

    assert handler.processing_final is False
    assert handler.last_language == "en"
    assert reset_calls == [True]
