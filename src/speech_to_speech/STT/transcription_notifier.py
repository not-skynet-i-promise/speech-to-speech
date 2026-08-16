from __future__ import annotations

import logging
from contextlib import AbstractContextManager, ExitStack, contextmanager
from queue import Queue
from threading import Event
from typing import Callable, Iterator, Union

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import make_user_message
from speech_to_speech.pipeline.events import (
    PartialTranscriptionEvent,
    TranscriptBarrierCompletedEvent,
    TranscriptBarrierDiscardedEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.handler_types import LLMIn, STTOut
from speech_to_speech.pipeline.messages import GenerateResponseRequest, PartialTranscription, Transcription
from speech_to_speech.pipeline.queue_types import TextEventItem

logger = logging.getLogger(__name__)


class TranscriptionNotifier(BaseHandler[STTOut, Union[STTOut, LLMIn]]):
    """Sits between STT and LLM.

    For **realtime mode** (no ``runtime_config``): emits transcription events
    on ``text_output_queue`` for protocol translation but yields nothing -- the
    ``RealtimeService`` builds ``GenerateResponseRequest`` directly.

    For **legacy mode** (``runtime_config`` provided): appends the user
    message to ``runtime_config.chat`` and yields a
    ``GenerateResponseRequest`` so the LLM handler receives a uniform input
    type regardless of pipeline mode.
    """

    def setup(
        self,
        text_output_queue: Queue[TextEventItem] | None = None,
        runtime_config: RuntimeConfig | None = None,
        should_listen: Event | None = None,
        transcript_barrier_enabled: Callable[[], bool] | None = None,
        transcript_barrier_failed: Callable[[], bool] | None = None,
        transcript_barrier_state_guard: Callable[[], AbstractContextManager[tuple[bool, bool]]] | None = None,
    ) -> None:
        self.text_output_queue = text_output_queue
        self.runtime_config = runtime_config
        self.should_listen = should_listen
        self.transcript_barrier_enabled = transcript_barrier_enabled or (lambda: False)
        self.transcript_barrier_failed = transcript_barrier_failed or (lambda: False)
        self.transcript_barrier_state_guard = transcript_barrier_state_guard

    @contextmanager
    def _barrier_state_guard(self) -> Iterator[tuple[bool, bool]]:
        """Hold the live barrier lock across every content-bearing side effect."""
        with ExitStack() as stack:
            try:
                if self.transcript_barrier_state_guard is not None:
                    state = stack.enter_context(self.transcript_barrier_state_guard())
                elif self.runtime_config is not None:
                    stack.enter_context(self.runtime_config.transcript_barrier_state_guard())
                    state = (
                        self.runtime_config.transcript_barrier_private,
                        self.runtime_config.transcript_barrier_failed,
                    )
                else:
                    state = (
                        self.transcript_barrier_enabled(),
                        self.transcript_barrier_failed(),
                    )
                if (
                    not isinstance(state, tuple)
                    or len(state) != 2
                    or type(state[0]) is not bool
                    or type(state[1]) is not bool
                ):
                    raise ValueError("invalid transcript barrier state")
            except Exception:
                logger.error("Transcript barrier state unavailable; private content redacted")
                yield True, True
                return
            yield state

    def process(self, transcription: STTOut) -> Iterator[Union[STTOut, LLMIn]]:
        with self._barrier_state_guard() as (barrier_private, barrier_failed):
            if barrier_failed:
                logger.debug("Dropping transcription after private barrier failure")
                return
            if isinstance(transcription, PartialTranscription):
                if barrier_private:
                    logger.debug("Private transcript barrier suppressed a partial transcription")
                    return
                if self.text_output_queue and transcription.text:
                    self.text_output_queue.put(
                        PartialTranscriptionEvent(
                            delta=str(transcription.text),
                            turn_id=transcription.turn_id,
                            turn_revision=transcription.turn_revision,
                        )
                    )
                    logger.debug("Partial transcription: %s", str(transcription.text)[:80])
                return

            if isinstance(transcription, Transcription):
                text = transcription.text
                language_code = transcription.language_code
                turn_id = transcription.turn_id
                turn_revision = transcription.turn_revision
                speech_stopped_at_s = transcription.speech_stopped_at_s
            else:
                text = transcription
                language_code = None
                turn_id = None
                turn_revision = None
                speech_stopped_at_s = None

            transcript = str(text)
            if barrier_private:
                if self.text_output_queue is not None:
                    if transcript.strip():
                        self.text_output_queue.put(
                            TranscriptBarrierCompletedEvent(
                                transcript=transcript,
                                language_code=language_code,
                                turn_id=turn_id,
                                turn_revision=turn_revision,
                                speech_stopped_at_s=speech_stopped_at_s,
                            )
                        )
                        logger.debug("Private transcript barrier completed one transcription")
                    else:
                        self.text_output_queue.put(
                            TranscriptBarrierDiscardedEvent(
                                turn_id=turn_id,
                                turn_revision=turn_revision,
                            )
                        )
                        logger.debug("Private transcript barrier discarded an empty transcription")
                if not transcript.strip() and self.should_listen is not None:
                    self.should_listen.set()
                return

            # Always close the client-visible transcription item. Empty final STT
            # results should not trigger the LLM, but clients may already have
            # received partial deltas and still need a completed event.
            if self.text_output_queue is not None:
                self.text_output_queue.put(
                    TranscriptionCompletedEvent(
                        transcript=transcript,
                        language_code=language_code,
                        turn_id=turn_id,
                        turn_revision=turn_revision,
                        speech_stopped_at_s=speech_stopped_at_s,
                    )
                )

            if not transcript:
                logger.debug("Transcription completed with empty transcript")
                if self.should_listen is not None:
                    self.should_listen.set()
                    logger.debug("Empty transcription completed; listening re-enabled")
                return

            if language_code:
                logger.info("Transcription completed (language=%s): %s", language_code, transcript)
            else:
                logger.info("Transcription completed: %s", transcript)

            if self.runtime_config is not None:
                self.runtime_config.chat.add_item(make_user_message(transcript))
                yield GenerateResponseRequest(
                    runtime_config=self.runtime_config,
                    language_code=language_code,
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                    speech_stopped_at_s=speech_stopped_at_s,
                )
