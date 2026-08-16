from __future__ import annotations

import logging
from collections import Counter, OrderedDict
from contextlib import AbstractContextManager, ExitStack, contextmanager
from time import perf_counter
from typing import Any, Callable, Iterator

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)


class BaseSTTHandler(BaseHandler[STTIn, STTOut]):
    """Base STT handler with speculative-turn stale input filtering."""

    _MAX_COMPLETED_FINAL_REVISIONS = 2048

    speculative_turns: SpeculativeTurnTracker | None = None
    final_revision_settle_s: float = 0.0
    _transcript_barrier_enabled: Callable[[], bool] = staticmethod(lambda: False)
    _transcript_barrier_failed: Callable[[], bool] = staticmethod(lambda: False)
    _transcript_barrier_state_guard: Callable[[], AbstractContextManager[tuple[bool, bool]]] | None = None

    def set_transcript_barrier_enabled(self, enabled: Callable[[], bool]) -> None:
        """Install the single-session content-redaction gate after construction."""
        self._transcript_barrier_enabled = enabled

    def set_transcript_barrier_failed(self, failed: Callable[[], bool]) -> None:
        """Install the single-session fail-closed output gate."""
        self._transcript_barrier_failed = failed

    def set_transcript_barrier_state_guard(
        self,
        state_guard: Callable[[], AbstractContextManager[tuple[bool, bool]]],
    ) -> None:
        """Install the live barrier lock used around raw-content side effects."""
        self._transcript_barrier_state_guard = state_guard

    @property
    def transcript_barrier_enabled(self) -> bool:
        return self._transcript_barrier_enabled()

    @contextmanager
    def _transcript_barrier_state(self) -> Iterator[tuple[bool, bool]]:
        """Read one fail-closed state while holding the live barrier lock."""
        stack = ExitStack()
        try:
            if self._transcript_barrier_state_guard is None:
                state = (self._transcript_barrier_enabled(), self._transcript_barrier_failed())
            else:
                state = stack.enter_context(self._transcript_barrier_state_guard())
            if (
                not isinstance(state, tuple)
                or len(state) != 2
                or type(state[0]) is not bool
                or type(state[1]) is not bool
            ):
                raise ValueError("invalid transcript barrier state")
        except Exception:
            try:
                stack.close()
            except Exception:
                pass
            logger.error("STT transcript barrier state unavailable; private content redacted")
            yield True, True
            return

        with stack:
            yield state

    @contextmanager
    def transcript_content_allowed(self) -> Iterator[bool]:
        """Hold the live barrier lock from policy selection through one content sink."""
        with self._transcript_barrier_state() as (private, failed):
            yield not private and not failed

    @contextmanager
    def _input_private_logging_guard(self, item: object) -> Iterator[bool]:
        """Use the same live state for generic STT exception and cleanup logs."""
        if self._transcript_barrier_state_guard is None:
            with super()._input_private_logging_guard(item) as private:
                try:
                    failed = self._transcript_barrier_failed()
                except Exception:
                    failed = True
                yield private or failed
            return
        with self._transcript_barrier_state() as (private, failed):
            yield self._private_content_logging or private or failed

    def should_process_input(self, item: STTIn) -> bool:
        if self._transcript_barrier_failed():
            logger.debug("Dropping STT input after private barrier failure")
            return False
        mode = getattr(item, "mode", None)
        turn_id = getattr(item, "turn_id", None)
        turn_revision = getattr(item, "turn_revision", None)
        if self._is_completed_final_revision(item):
            queued_drops = self._drop_stale_queued_inputs()
            self._log_stale_turn_item(item, "input-after-final", queued_drops=queued_drops)
            return False
        if mode == "progressive" and self._has_queued_final_for_revision(item):
            self._log_stale_turn_item(item, "progressive-before-final")
            return False

        wait_for_stability = mode == "final"
        gate_start = perf_counter()
        is_latest = self._is_latest_turn_item(
            item,
            wait_for_pending_reopen=True,
            wait_for_stability=wait_for_stability,
        )
        gate_wait_s = perf_counter() - gate_start
        if gate_wait_s >= 0.05:
            logger.info(
                "%s: STT input gate waited %.3fs for turn=%s rev=%s mode=%s latest=%s age=%.3fs queue=%s",
                self.__class__.__name__,
                gate_wait_s,
                turn_id,
                turn_revision,
                mode,
                is_latest,
                self._item_age_s(item),
                self._safe_qsize(),
            )

        if not is_latest:
            queued_drops = self._drop_stale_queued_inputs()
            self._log_stale_turn_item(item, "input", queued_drops=queued_drops)
            return False
        return True

    def should_emit_output(self, output: STTOut) -> bool:
        if self._transcript_barrier_failed():
            logger.debug("Dropping STT output after private barrier failure")
            return False
        if isinstance(output, PartialTranscription) and self._is_completed_final_revision(output):
            self._log_stale_turn_item(output, "output-after-final")
            return False

        if not self._is_latest_turn_item(output, wait_for_pending_reopen=True, wait_for_stability=False):
            self._log_stale_turn_item(output, "output")
            return False
        return True

    def before_emit_output(self, output: STTOut) -> None:
        if isinstance(output, Transcription):
            self._mark_completed_final_revision(output)

    def _is_latest_turn_item(
        self,
        item: object,
        *,
        wait_for_pending_reopen: bool,
        wait_for_stability: bool,
    ) -> bool:
        if self.speculative_turns is None:
            return True
        turn_id = getattr(item, "turn_id", None)
        turn_revision = getattr(item, "turn_revision", None)
        if turn_id is None or turn_revision is None:
            return True

        if wait_for_stability:
            is_latest = self.speculative_turns.is_latest_after_stability_window(
                turn_id,
                turn_revision,
                self.final_revision_settle_s,
            )
        elif wait_for_pending_reopen:
            is_latest = self.speculative_turns.is_latest_after_pending_reopen(turn_id, turn_revision)
        else:
            is_latest = self.speculative_turns.is_latest(turn_id, turn_revision)
        return is_latest

    def _drop_stale_queued_inputs(self) -> int:
        if self.speculative_turns is None or not hasattr(self.queue_in, "mutex") or not hasattr(self.queue_in, "queue"):
            return 0

        dropped = 0
        with self.queue_in.mutex:
            kept: list[Any] = []
            while self.queue_in.queue:
                queued_item = self.queue_in.queue.popleft()
                if isinstance(queued_item, VADAudio) and (
                    self._is_completed_final_revision(queued_item)
                    or (queued_item.mode == "progressive" and self._has_queued_final_for_revision_locked(queued_item))
                    or not self._is_latest_turn_item(
                        queued_item,
                        wait_for_pending_reopen=False,
                        wait_for_stability=False,
                    )
                ):
                    dropped += 1
                else:
                    kept.append(queued_item)
            self.queue_in.queue.extend(kept)
            if dropped:
                self.queue_in.not_full.notify_all()
        return dropped

    def _log_stale_turn_item(self, item: object, stage: str, *, queued_drops: int = 0) -> None:
        turn_id = getattr(item, "turn_id", None)
        turn_revision = getattr(item, "turn_revision", None)
        if turn_id is None or turn_revision is None:
            return

        if not hasattr(self, "_stale_drop_counts"):
            self._stale_drop_counts: Counter[tuple[str, str, int]] = Counter()
        key = (stage, turn_id, turn_revision)
        self._stale_drop_counts[key] += 1

        message = "%s: dropping stale STT %s for turn=%s rev=%s age=%.3fs"
        args: tuple[object, ...] = (
            self.__class__.__name__,
            stage,
            turn_id,
            turn_revision,
            self._item_age_s(item),
        )
        if queued_drops:
            message += " (+%d queued)"
            args = (*args, queued_drops)

        if self._stale_drop_counts[key] == 1:
            logger.info(message, *args)
        else:
            logger.debug(message, *args)

    def _item_age_s(self, item: object) -> float:
        created_at_s = getattr(item, "created_at_s", None)
        if not isinstance(created_at_s, float):
            return 0.0
        return max(0.0, perf_counter() - created_at_s)

    def _safe_qsize(self) -> int | str:
        try:
            return self.queue_in.qsize()
        except NotImplementedError:
            return "unknown"

    def _has_queued_final_for_revision(self, item: object) -> bool:
        if not hasattr(self.queue_in, "mutex") or not hasattr(self.queue_in, "queue"):
            return False
        with self.queue_in.mutex:
            return self._has_queued_final_for_revision_locked(item)

    def _has_queued_final_for_revision_locked(self, item: object) -> bool:
        key = self._revision_key(item)
        if key is None:
            return False
        return any(
            isinstance(queued_item, VADAudio) and queued_item.mode == "final" and self._revision_key(queued_item) == key
            for queued_item in self.queue_in.queue
        )

    def _revision_key(self, item: object) -> tuple[str, int] | None:
        turn_id = getattr(item, "turn_id", None)
        turn_revision = getattr(item, "turn_revision", None)
        if not isinstance(turn_id, str) or not isinstance(turn_revision, int):
            return None
        return (turn_id, turn_revision)

    def _completed_final_revisions(self) -> OrderedDict[tuple[str, int], None]:
        if not hasattr(self, "_completed_final_revision_keys"):
            self._completed_final_revision_keys: OrderedDict[tuple[str, int], None] = OrderedDict()
        return self._completed_final_revision_keys

    def _is_completed_final_revision(self, item: object) -> bool:
        key = self._revision_key(item)
        return key is not None and key in self._completed_final_revisions()

    def _mark_completed_final_revision(self, output: Transcription) -> None:
        key = self._revision_key(output)
        if key is None:
            return
        completed = self._completed_final_revisions()
        completed[key] = None
        completed.move_to_end(key)
        while len(completed) > self._MAX_COMPLETED_FINAL_REVISIONS:
            completed.popitem(last=False)

    def on_session_end(self) -> None:
        if hasattr(self, "_completed_final_revision_keys"):
            self._completed_final_revision_keys.clear()
