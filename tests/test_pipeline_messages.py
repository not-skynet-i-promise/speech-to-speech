from threading import Event, Thread

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline import PartialTranscriptionEvent
from speech_to_speech.pipeline.messages import GenerateResponseRequest, ResponsePrefetchTransaction


def test_partial_transcription_event_retains_delta_constructor():
    event = PartialTranscriptionEvent(delta="hello")

    assert event.delta == "hello"


def test_prefetch_transaction_commits_cleanup_once_across_completion_order():
    completed_first: list[str] = []
    transaction = ResponsePrefetchTransaction()
    transaction.complete(lambda: completed_first.append("cleanup"))
    assert completed_first == []
    transaction.claim()
    transaction.claim()
    assert completed_first == ["cleanup"]

    claimed_first: list[str] = []
    transaction = ResponsePrefetchTransaction()
    transaction.claim()
    transaction.complete(lambda: claimed_first.append("cleanup"))
    assert claimed_first == ["cleanup"]


def test_prefetch_transaction_discards_deferred_cleanup():
    completed: list[str] = []
    transaction = ResponsePrefetchTransaction()
    transaction.complete(lambda: completed.append("cleanup"))
    transaction.discard()
    assert transaction.claim() is False
    assert completed == []


def test_prefetch_transaction_aborts_only_unclaimed_work():
    aborted: list[str] = []
    transaction = ResponsePrefetchTransaction()
    transaction.register_abort(lambda: aborted.append("running"))
    transaction.discard()
    transaction.discard()
    transaction.register_abort(lambda: aborted.append("late"))
    assert aborted == ["running", "late"]

    claimed = ResponsePrefetchTransaction()
    claimed.register_abort(lambda: aborted.append("must not run"))
    claimed.claim()
    claimed.discard()
    assert aborted == ["running", "late"]


def test_prefetch_transaction_abort_failure_does_not_skip_remaining_cleanup():
    transaction = ResponsePrefetchTransaction()
    aborted: list[str] = []

    def fail_to_close() -> None:
        raise RuntimeError("close failed")

    transaction.register_abort(fail_to_close)
    transaction.register_abort(lambda: aborted.append("next callback"))

    transaction.discard()

    assert transaction.discarded
    assert aborted == ["next callback"]


def test_provider_start_is_atomic_with_request_cancellation():
    """Cancellation cannot acknowledge inside the provider-start boundary."""
    request = GenerateResponseRequest(runtime_config=RuntimeConfig())
    provider_entered = Event()
    release_provider_start = Event()
    cancellation_returned = Event()
    result: list[tuple[bool, object]] = []

    def start_provider() -> object:
        provider_entered.set()
        assert release_provider_start.wait(1.0)
        return "provider-started"

    starter = Thread(target=lambda: result.append(request.start_provider_request(start_provider)))
    starter.start()
    assert provider_entered.wait(1.0)

    canceller = Thread(target=lambda: (request.cancel(), cancellation_returned.set()))
    canceller.start()
    assert not cancellation_returned.wait(0.05)

    release_provider_start.set()
    starter.join(timeout=1.0)
    canceller.join(timeout=1.0)

    assert result == [(True, "provider-started")]
    assert cancellation_returned.is_set()
    assert request.is_cancelled
