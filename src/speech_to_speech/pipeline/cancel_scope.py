from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock


class CancelScope:
    """Unified cancellation signal for the speech-to-speech pipeline.

    Uses a generation counter so pipeline threads (LLM, TTS) can detect
    cancellation without brief-pulse timing games, and an internal
    ``discarding`` flag so the async send loop can drop stale output.

    Cancellation, response admission, and private-barrier activation share one
    lock. This gives queued work one exact ordering: it is admitted before a
    cancel (and remains visible as active), or it observes the newer generation
    and is rejected before provider/model execution.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._gen: int = 0
        self._discarding: bool = False
        self._discarded_generation: int | None = None
        self._active_responses: int = 0

    @property
    def generation(self) -> int:
        """Current generation number.  Pipeline threads capture this at
        the start of each response and compare with ``is_stale``."""
        with self._lock:
            return self._gen

    def cancel(self) -> None:
        """Cancel the current response.

        Increments the generation (so pipeline threads see their captured
        generation as stale) and enables the send-loop discard guard.
        """
        with self._lock:
            # prevent overflow... after 4 billion generations, we'll wrap around xD...
            self._discarded_generation = self._gen
            self._gen = (self._gen + 1) & 0xFFFFFFFF
            self._discarding = True

    def response_done(self, generation: int | None = None) -> None:
        """Pipeline acknowledged completion.  Clears the discard guard."""
        with self._lock:
            if (
                generation is not None
                and self._discarded_generation is not None
                and generation not in {self._discarded_generation, self._gen}
            ):
                return
            self._discarding = False
            self._discarded_generation = None

    def new_response(self) -> None:
        """An explicit ``response.create`` starts a new response.
        Clears the discard guard."""
        with self._lock:
            self._discarding = False
            self._discarded_generation = None

    def is_stale(self, gen: int) -> bool:
        """Return True if *gen* has been superseded by a ``cancel`` call."""
        with self._lock:
            return gen != self._gen

    @property
    def discarding(self) -> bool:
        """Whether the send loop should silently drop stale output."""
        with self._lock:
            return self._discarding

    @contextmanager
    def response_admission(self, generation: int | None) -> Iterator[tuple[bool, int]]:
        """Atomically admit one response against the current generation.

        The admission lease stays set until the handler's generator exits, but
        the lock is not held during model/provider work. A local worker that
        survives its generator holds an additional lease through actual thread
        exit. Private activation takes the same lock and refuses readiness
        while any response or worker lease is active.
        """
        with self._lock:
            resolved_generation = self._gen if generation is None else generation
            admitted = resolved_generation == self._gen
            if admitted:
                self._active_responses += 1
        try:
            yield admitted, resolved_generation
        finally:
            if admitted:
                with self._lock:
                    self._active_responses -= 1

    @contextmanager
    def private_activation_guard(self) -> Iterator[bool]:
        """Serialize barrier readiness with response admission and cancellation."""
        with self._lock:
            yield self._active_responses == 0

    def response_worker_started(self) -> None:
        """Keep private activation blocked while an admitted worker is alive."""
        with self._lock:
            self._active_responses += 1

    def response_worker_done(self) -> None:
        """Release one live-worker lease after the worker has actually exited."""
        with self._lock:
            if self._active_responses <= 0:
                raise RuntimeError("Response worker lease underflow")
            self._active_responses -= 1

    def reset(self) -> None:
        """Clear discard state (e.g. on new session connect)."""
        with self._lock:
            self._discarding = False
            self._discarded_generation = None
