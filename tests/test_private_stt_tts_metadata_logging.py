"""Provider-free canaries for STT/TTS metadata and exception privacy sinks."""

# The optional Lightning dependency must be stubbed before importing its handler.
# ruff: noqa: E402

from __future__ import annotations

import logging
import sys
from threading import Event, Thread
from types import ModuleType, SimpleNamespace

import numpy as np

_lightning_whisper = ModuleType("lightning_whisper_mlx")
_lightning_whisper.LightningWhisperMLX = object  # type: ignore[attr-defined]
sys.modules.setdefault("lightning_whisper_mlx", _lightning_whisper)

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.pipeline.messages import TTSInput, VADAudio
from speech_to_speech.STT import (
    lightning_whisper_mlx_handler,
    mlx_audio_whisper_handler,
    whisper_stt_handler,
)
from speech_to_speech.STT.lightning_whisper_mlx_handler import LightningWhisperSTTHandler
from speech_to_speech.STT.mlx_audio_whisper_handler import MLXAudioWhisperSTTHandler
from speech_to_speech.STT.whisper_stt_handler import WhisperSTTHandler
from speech_to_speech.TTS.facebookmms_handler import FacebookMMSTTSHandler
from speech_to_speech.TTS.kokoro_handler import KokoroTTSHandler
from speech_to_speech.TTS.pocket_tts_handler import PocketTTSHandler


def _private_config() -> RuntimeConfig:
    config = RuntimeConfig()
    config.transcript_barrier_version = 1
    config.transcript_barrier_nonce = "ab" * 32
    config.chat.enable_private_content_logging()
    return config


def test_private_stt_language_outputs_are_content_free(monkeypatch, caplog):
    language_canary = "PRIVATE_STT_LANGUAGE_CANARY"
    transcript_canary = "PRIVATE_STT_TRANSCRIPT_CANARY"
    audio = VADAudio(audio=np.zeros(160, dtype=np.float32))

    mlx = object.__new__(MLXAudioWhisperSTTHandler)
    mlx.start_language = "auto"
    mlx.last_language = "en"
    mlx.model = SimpleNamespace(
        generate=lambda *_args, **_kwargs: SimpleNamespace(text=transcript_canary, language=language_canary)
    )
    mlx.set_transcript_barrier_enabled(lambda: True)

    lightning = object.__new__(LightningWhisperSTTHandler)
    lightning.start_language = "auto"
    lightning.last_language = "en"

    def transcribe(_audio, language=None):
        return {
            "text": transcript_canary,
            "language": language_canary if language is None else "en",
        }

    lightning.model = SimpleNamespace(transcribe=transcribe)
    lightning.set_transcript_barrier_enabled(lambda: True)
    monkeypatch.setattr(lightning_whisper_mlx_handler.torch.mps, "empty_cache", lambda: None)

    whisper = object.__new__(WhisperSTTHandler)
    whisper.device = "cpu"
    whisper.start_language = "auto"
    whisper.last_language = "en"
    whisper.gen_kwargs = {}
    whisper.prepare_model_inputs = lambda _audio: np.zeros((1, 2), dtype=np.float32)
    whisper.model = SimpleNamespace(generate=lambda *_args, **_kwargs: np.array([[1, 2]]))
    whisper.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(decode=lambda *_args, **_kwargs: f"<|{language_canary}|>"),
        batch_decode=lambda *_args, **_kwargs: [transcript_canary],
    )
    whisper.set_transcript_barrier_enabled(lambda: True)

    monkeypatch.setattr(mlx_audio_whisper_handler.console, "print", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(lightning_whisper_mlx_handler.console, "print", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(whisper_stt_handler.console, "print", lambda *_args, **_kwargs: None)

    with caplog.at_level(logging.DEBUG):
        assert len(list(mlx.process(audio))) == 1
        assert len(list(lightning.process(audio))) == 1
        assert len(list(whisper.process(audio))) == 1

    assert language_canary not in caplog.text
    assert transcript_canary not in caplog.text
    assert "private content redacted" in caplog.text


def test_stt_language_log_holds_live_guard_through_concurrent_poison(monkeypatch):
    language_canary = "ORDINARY_LANGUAGE_BEFORE_POISON"
    service = RealtimeService()
    connection_id = service.register()
    handler = object.__new__(MLXAudioWhisperSTTHandler)
    handler.start_language = "auto"
    handler.last_language = "en"
    handler.model = SimpleNamespace(
        generate=lambda *_args, **_kwargs: SimpleNamespace(text="ordinary", language=language_canary)
    )
    handler.set_transcript_barrier_enabled(service.transcript_barrier_private)
    handler.set_transcript_barrier_failed(service.transcript_barrier_poisoned)
    handler.set_transcript_barrier_state_guard(service.transcript_barrier_pipeline_state_guard)
    monkeypatch.setattr(mlx_audio_whisper_handler.console, "print", lambda *_args, **_kwargs: None)
    poison_attempted = Event()
    poison_completed = Event()
    poison_thread: Thread | None = None

    def capture_warning(message, *_args, **_kwargs):
        nonlocal poison_thread

        def poison() -> None:
            poison_attempted.set()
            service.poison_transcript_barrier(connection_id, "test_failure")
            poison_completed.set()

        poison_thread = Thread(target=poison)
        poison_thread.start()
        assert poison_attempted.wait(timeout=1.0)
        assert not poison_completed.wait(timeout=0.05)
        assert language_canary in str(message)

    monkeypatch.setattr(mlx_audio_whisper_handler.logger, "warning", capture_warning)
    assert len(list(handler.process(VADAudio(audio=np.zeros(160, dtype=np.float32))))) == 1

    assert poison_thread is not None
    poison_thread.join(timeout=1.0)
    assert not poison_thread.is_alive()
    assert poison_completed.is_set()
    assert service.transcript_barrier_failed(connection_id)


def test_private_tts_language_voice_and_exception_sinks_are_content_free(monkeypatch, caplog):
    language_canary = "PRIVATE_TTS_LANGUAGE_CANARY"
    voice_canary = "PRIVATE_TTS_VOICE_CANARY"
    exception_canary = "PRIVATE_TTS_EXCEPTION_CANARY"
    config = _private_config()

    pocket = object.__new__(PocketTTSHandler)
    pocket.speculative_turns = None
    pocket.cancel_scope = None
    pocket.model = SimpleNamespace(sample_rate=16000, generate_audio_stream=lambda *_args, **_kwargs: iter(()))
    pocket.sample_rate = 16000
    pocket.blocksize = 512
    pocket.voice_state = object()
    pocket.max_tokens = 1

    facebook = object.__new__(FacebookMMSTTSHandler)
    facebook.speculative_turns = None
    facebook.cancel_scope = None
    facebook.language = "en"
    facebook.generate_audio = lambda *_args, **_kwargs: None
    facebook.load_model = lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyError(language_canary))

    kokoro = object.__new__(KokoroTTSHandler)
    kokoro.cancel_scope = None
    kokoro.lang_code = "b"
    kokoro.voice = voice_canary
    kokoro.model = SimpleNamespace(
        _get_pipeline=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(exception_canary))
    )
    kokoro._pipeline = lambda *_args, **_kwargs: iter(())
    kokoro.speed = 1.0
    kokoro.blocksize = 512

    monkeypatch.setattr("speech_to_speech.TTS.pocket_tts_handler.console.print", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("speech_to_speech.TTS.facebookmms_handler.console.print", lambda *_args, **_kwargs: None)

    with caplog.at_level(logging.DEBUG):
        assert (
            list(pocket.process(TTSInput(text="private", language_code=language_canary, runtime_config=config))) == []
        )
        assert (
            list(facebook.process(TTSInput(text="private", language_code=language_canary, runtime_config=config))) == []
        )
        assert list(kokoro._process_mlx("private", "fr", config)) == []

    assert language_canary not in caplog.text
    assert voice_canary not in caplog.text
    assert exception_canary not in caplog.text
    assert "content redacted" in caplog.text


def test_kokoro_cleanup_uses_sticky_private_logging(caplog):
    exception_canary = "PRIVATE_KOKORO_CLEANUP_CANARY"
    handler = object.__new__(KokoroTTSHandler)
    handler._private_content_logging = True
    handler._initial_voice = "initial"
    handler._initial_lang_code = "b"
    handler.backend = "mlx"
    handler.model = SimpleNamespace(
        _get_pipeline=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(exception_canary))
    )

    with caplog.at_level(logging.WARNING):
        handler.on_session_end()

    assert exception_canary not in caplog.text
    assert "private content redacted" in caplog.text


def test_kokoro_language_log_holds_guard_through_concurrent_poison(monkeypatch):
    voice_canary = "ORDINARY_KOKORO_VOICE_BEFORE_POISON"
    runtime_config = RuntimeConfig()
    handler = object.__new__(KokoroTTSHandler)
    handler.cancel_scope = None
    handler.lang_code = "b"
    handler.voice = voice_canary
    handler.speed = 1.0
    handler.blocksize = 512

    class _Pipeline:
        def load_voice(self, _voice):
            return object()

        def __call__(self, **_kwargs):
            return iter(())

    pipeline = _Pipeline()
    handler.model = SimpleNamespace(_get_pipeline=lambda *_args, **_kwargs: pipeline)
    handler._pipeline = pipeline
    poison_attempted = Event()
    poison_completed = Event()
    poison_thread: Thread | None = None
    original_info = logging.getLogger("speech_to_speech.TTS.kokoro_handler").info

    def capture_info(message, *_args, **_kwargs):
        nonlocal poison_thread
        if str(message).startswith("Language change detected:"):

            def poison() -> None:
                poison_attempted.set()
                with runtime_config.transcript_barrier_state_guard():
                    runtime_config.transcript_barrier_failed = True
                    runtime_config.chat.enable_private_content_logging()
                poison_completed.set()

            poison_thread = Thread(target=poison)
            poison_thread.start()
            assert poison_attempted.wait(timeout=1.0)
            assert not poison_completed.wait(timeout=0.05)
            assert voice_canary in str(message)
        original_info(message, *_args, **_kwargs)

    monkeypatch.setattr("speech_to_speech.TTS.kokoro_handler.logger.info", capture_info)
    assert list(handler._process_mlx("ordinary", "fr", runtime_config)) == []

    assert poison_thread is not None
    poison_thread.join(timeout=1.0)
    assert not poison_thread.is_alive()
    assert poison_completed.is_set()
