from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import blake2b
from threading import Condition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PendingReopen:
    base_revision: int
    candidate_revision: int


@dataclass(frozen=True)
class _ReopenGrace:
    revision: int
    deadline: float


class SpeculativeTurnTracker:
    """Thread-safe revision tracker for raw-audio speculative turns."""

    _PENDING_REOPEN_WAIT_TIMEOUT_S = 2.0
    _MAX_TRACKED_TURNS = 2048
    _DISCARD_FILTER_BYTES = 1 << 17

    def __init__(self, max_tracked_turns: int = _MAX_TRACKED_TURNS) -> None:
        self._condition = Condition()
        self._max_tracked_turns = max_tracked_turns
        self._latest_revision: OrderedDict[str, int] = OrderedDict()
        self._committed_revision: dict[str, int] = {}
        self._pending_reopen: dict[str, _PendingReopen] = {}
        self._reopen_grace: dict[str, _ReopenGrace] = {}
        self._discarded_turn_ids: OrderedDict[str, None] = OrderedDict()
        # A fixed-size, fail-closed filter prevents an evicted tombstone from
        # reviving late pipeline events. False positives only discard a turn;
        # they never re-expose content or output from a deleted turn.
        self._discard_filter = bytearray(self._DISCARD_FILTER_BYTES)

    def _discard_filter_positions(self, turn_id: str) -> tuple[int, int, int, int]:
        digest = blake2b(turn_id.encode("utf-8"), digest_size=16).digest()
        bit_count = len(self._discard_filter) * 8
        return (
            int.from_bytes(digest[0:4], "big") % bit_count,
            int.from_bytes(digest[4:8], "big") % bit_count,
            int.from_bytes(digest[8:12], "big") % bit_count,
            int.from_bytes(digest[12:16], "big") % bit_count,
        )

    def _mark_discarded_locked(self, turn_id: str) -> None:
        for position in self._discard_filter_positions(turn_id):
            self._discard_filter[position // 8] |= 1 << (position % 8)

    def _is_discarded_locked(self, turn_id: str) -> bool:
        if turn_id in self._discarded_turn_ids:
            return True
        return all(
            self._discard_filter[position // 8] & (1 << (position % 8))
            for position in self._discard_filter_positions(turn_id)
        )

    def _owns_revision_locked(self, turn_id: str, revision: int) -> bool:
        return not self._is_discarded_locked(turn_id) and self._latest_revision.get(turn_id, revision) == revision

    def observe(self, turn_id: str | None, revision: int | None) -> None:
        if turn_id is None or revision is None:
            return
        with self._condition:
            if self._is_discarded_locked(turn_id):
                return
            current = self._latest_revision.get(turn_id, -1)
            if revision > current:
                self._latest_revision[turn_id] = revision
                self._latest_revision.move_to_end(turn_id)
                self._prune_tracked_turns()
                logger.debug("Observed speculative turn %s revision %d", turn_id, revision)
                self._condition.notify_all()

    def discard(self, turn_id: str | None) -> None:
        """Make every current or later revision of one turn permanently stale."""
        if turn_id is None:
            return
        with self._condition:
            self._committed_revision.pop(turn_id, None)
            self._pending_reopen.pop(turn_id, None)
            self._reopen_grace.pop(turn_id, None)
            self._latest_revision[turn_id] = -1
            self._latest_revision.move_to_end(turn_id)
            self._mark_discarded_locked(turn_id)
            self._discarded_turn_ids[turn_id] = None
            self._discarded_turn_ids.move_to_end(turn_id)
            if self._max_tracked_turns > 0:
                while len(self._discarded_turn_ids) > self._max_tracked_turns:
                    retired_turn_id, _ = self._discarded_turn_ids.popitem(last=False)
                    self._latest_revision.pop(retired_turn_id, None)
            self._condition.notify_all()

    def is_latest(self, turn_id: str | None, revision: int | None) -> bool:
        if turn_id is None or revision is None:
            return True
        with self._condition:
            return self._owns_revision_locked(turn_id, revision)

    def is_latest_after_pending_reopen(self, turn_id: str | None, revision: int | None) -> bool:
        if turn_id is None or revision is None:
            return True
        with self._condition:
            self._wait_for_pending_reopen_locked(turn_id, revision, self._PENDING_REOPEN_WAIT_TIMEOUT_S)
            return self._owns_revision_locked(turn_id, revision)

    def try_is_latest_after_pending_reopen(self, turn_id: str | None, revision: int | None) -> bool | None:
        """Non-blocking variant of ``is_latest_after_pending_reopen``.

        Returns ``None`` when a matching reopen candidate is still pending and
        the caller should retry after it resolves.
        """
        if turn_id is None or revision is None:
            return True
        with self._condition:
            if self._is_discarded_locked(turn_id):
                return False
            if self._has_pending_reopen_locked(turn_id, revision):
                return None
            return self._owns_revision_locked(turn_id, revision)

    def is_latest_after_reopen_grace(self, turn_id: str | None, revision: int | None) -> bool:
        if turn_id is None or revision is None:
            return True
        with self._condition:
            self._wait_for_reopen_gate_locked(turn_id, revision)
            return self._owns_revision_locked(turn_id, revision)

    def try_is_latest_after_reopen_grace(self, turn_id: str | None, revision: int | None) -> bool | None:
        if turn_id is None or revision is None:
            return True
        with self._condition:
            if self._is_discarded_locked(turn_id):
                return False
            if (
                self._has_pending_reopen_locked(turn_id, revision)
                or self._reopen_grace_remaining_locked(
                    turn_id,
                    revision,
                )
                > 0
            ):
                return None
            return self._owns_revision_locked(turn_id, revision)

    def commit_if_latest_after_pending_reopen(self, turn_id: str | None, revision: int | None) -> bool:
        if turn_id is None or revision is None:
            return True
        with self._condition:
            self._wait_for_pending_reopen_locked(turn_id, revision, self._PENDING_REOPEN_WAIT_TIMEOUT_S)
            if self._is_discarded_locked(turn_id):
                return False
            latest = self._latest_revision.get(turn_id, revision)
            if revision != latest:
                return False
            self._committed_revision[turn_id] = revision
            logger.debug("Committed speculative turn %s revision %d", turn_id, revision)
            self._condition.notify_all()
            return True

    def commit_if_latest_after_reopen_grace(self, turn_id: str | None, revision: int | None) -> bool:
        if turn_id is None or revision is None:
            return True
        with self._condition:
            self._wait_for_reopen_gate_locked(turn_id, revision)
            if self._is_discarded_locked(turn_id):
                return False
            latest = self._latest_revision.get(turn_id, revision)
            if revision != latest:
                return False
            self._committed_revision[turn_id] = revision
            logger.debug("Committed speculative turn %s revision %d", turn_id, revision)
            self._condition.notify_all()
            return True

    def try_commit_if_latest_after_pending_reopen(self, turn_id: str | None, revision: int | None) -> bool | None:
        """Non-blocking variant of ``commit_if_latest_after_pending_reopen``.

        Returns ``None`` when a matching reopen candidate is still pending and
        the caller should retry after it resolves.
        """
        if turn_id is None or revision is None:
            return True
        with self._condition:
            if self._is_discarded_locked(turn_id):
                return False
            if self._has_pending_reopen_locked(turn_id, revision):
                return None
            latest = self._latest_revision.get(turn_id, revision)
            if revision != latest:
                return False
            self._committed_revision[turn_id] = revision
            logger.debug("Committed speculative turn %s revision %d", turn_id, revision)
            self._condition.notify_all()
            return True

    def try_commit_if_latest_after_reopen_grace(self, turn_id: str | None, revision: int | None) -> bool | None:
        if turn_id is None or revision is None:
            return True
        with self._condition:
            if self._is_discarded_locked(turn_id):
                return False
            if (
                self._has_pending_reopen_locked(turn_id, revision)
                or self._reopen_grace_remaining_locked(
                    turn_id,
                    revision,
                )
                > 0
            ):
                return None
            latest = self._latest_revision.get(turn_id, revision)
            if revision != latest:
                return False
            self._committed_revision[turn_id] = revision
            logger.debug("Committed speculative turn %s revision %d", turn_id, revision)
            self._condition.notify_all()
            return True

    def has_pending_reopen(self, turn_id: str | None, revision: int | None) -> bool:
        if turn_id is None or revision is None:
            return False
        with self._condition:
            return self._has_pending_reopen_locked(turn_id, revision)

    def has_pending_reopen_or_grace(self, turn_id: str | None, revision: int | None) -> bool:
        if turn_id is None or revision is None:
            return False
        with self._condition:
            return (
                self._has_pending_reopen_locked(turn_id, revision)
                or self._reopen_grace_remaining_locked(
                    turn_id,
                    revision,
                )
                > 0
            )

    def start_reopen_grace(self, turn_id: str | None, revision: int | None, grace_s: float) -> None:
        if turn_id is None or revision is None or grace_s <= 0:
            return
        with self._condition:
            if not self._owns_revision_locked(turn_id, revision):
                return
            if self._committed_revision.get(turn_id, -1) >= revision:
                return
            deadline = time.monotonic() + grace_s
            existing = self._reopen_grace.get(turn_id)
            if existing is None or existing.revision != revision or deadline > existing.deadline:
                self._reopen_grace[turn_id] = _ReopenGrace(revision=revision, deadline=deadline)
                logger.debug(
                    "Started speculative reopen grace for turn %s revision %d: %.0fms",
                    turn_id,
                    revision,
                    grace_s * 1000,
                )
                self._condition.notify_all()

    def is_latest_after_stability_window(
        self,
        turn_id: str | None,
        revision: int | None,
        settle_s: float,
    ) -> bool:
        if turn_id is None or revision is None:
            return True
        if settle_s <= 0:
            return self.is_latest_after_pending_reopen(turn_id, revision)
        with self._condition:
            deadline = time.monotonic() + settle_s
            while self._owns_revision_locked(turn_id, revision):
                if self._has_pending_reopen_locked(turn_id, revision):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            self._wait_for_pending_reopen_locked(turn_id, revision, self._PENDING_REOPEN_WAIT_TIMEOUT_S)
            return self._owns_revision_locked(turn_id, revision)

    def commit(self, turn_id: str | None, revision: int | None) -> None:
        if turn_id is None or revision is None:
            return
        with self._condition:
            if self._is_discarded_locked(turn_id):
                return
            pending = self._pending_reopen.get(turn_id)
            if pending is not None and pending.base_revision == revision:
                logger.debug(
                    "Deferring speculative turn %s revision %d commit while reopen is pending", turn_id, revision
                )
                return
            latest = self._latest_revision.get(turn_id, revision)
            if revision == latest:
                self._committed_revision[turn_id] = revision
                logger.debug("Committed speculative turn %s revision %d", turn_id, revision)
                self._condition.notify_all()

    def is_committed(self, turn_id: str | None, revision: int | None = None) -> bool:
        if turn_id is None:
            return False
        with self._condition:
            committed = self._committed_revision.get(turn_id)
            if committed is None:
                return False
            return revision is None or committed >= revision

    def begin_reopen_candidate(self, turn_id: str | None, revision: int | None) -> int | None:
        if turn_id is None or revision is None:
            return None
        with self._condition:
            if self._is_discarded_locked(turn_id):
                return None
            if self._committed_revision.get(turn_id, -1) >= revision:
                return None
            if self._latest_revision.get(turn_id, revision) != revision:
                return None

            pending = self._pending_reopen.get(turn_id)
            if pending is not None:
                if pending.base_revision == revision:
                    return pending.candidate_revision
                return None

            candidate_revision = revision + 1
            self._pending_reopen[turn_id] = _PendingReopen(
                base_revision=revision,
                candidate_revision=candidate_revision,
            )
            logger.debug(
                "Started speculative reopen candidate for turn %s revision %d -> %d",
                turn_id,
                revision,
                candidate_revision,
            )
            self._condition.notify_all()
            return candidate_revision

    def confirm_reopen_candidate(
        self,
        turn_id: str | None,
        base_revision: int | None,
        candidate_revision: int | None,
    ) -> bool:
        if turn_id is None or base_revision is None or candidate_revision is None:
            return False
        with self._condition:
            pending = self._pending_reopen.get(turn_id)
            if (
                pending is None
                or pending.base_revision != base_revision
                or pending.candidate_revision != candidate_revision
            ):
                return False
            if self._committed_revision.get(turn_id, -1) >= base_revision:
                del self._pending_reopen[turn_id]
                self._prune_tracked_turns()
                self._condition.notify_all()
                return False
            if self._latest_revision.get(turn_id, base_revision) != base_revision:
                del self._pending_reopen[turn_id]
                self._prune_tracked_turns()
                self._condition.notify_all()
                return False

            self._latest_revision[turn_id] = candidate_revision
            self._latest_revision.move_to_end(turn_id)
            del self._pending_reopen[turn_id]
            self._prune_tracked_turns()
            logger.debug(
                "Confirmed speculative reopen candidate for turn %s revision %d",
                turn_id,
                candidate_revision,
            )
            self._condition.notify_all()
            return True

    def cancel_reopen_candidate(self, turn_id: str | None, candidate_revision: int | None = None) -> None:
        if turn_id is None:
            return
        with self._condition:
            pending = self._pending_reopen.get(turn_id)
            if pending is None:
                return
            if candidate_revision is not None and pending.candidate_revision != candidate_revision:
                return
            del self._pending_reopen[turn_id]
            self._prune_tracked_turns()
            logger.debug("Cancelled speculative reopen candidate for turn %s", turn_id)
            self._condition.notify_all()

    def wait_for_pending_reopen(
        self,
        turn_id: str | None,
        revision: int | None,
        timeout_s: float = _PENDING_REOPEN_WAIT_TIMEOUT_S,
    ) -> None:
        if turn_id is None or revision is None:
            return
        with self._condition:
            self._wait_for_pending_reopen_locked(turn_id, revision, timeout_s)

    def _has_pending_reopen_locked(self, turn_id: str, revision: int) -> bool:
        pending = self._pending_reopen.get(turn_id)
        return pending is not None and pending.base_revision == revision

    def _reopen_grace_remaining_locked(self, turn_id: str, revision: int) -> float:
        grace = self._reopen_grace.get(turn_id)
        if grace is None or grace.revision != revision:
            return 0.0
        if not self._owns_revision_locked(turn_id, revision):
            del self._reopen_grace[turn_id]
            return 0.0
        remaining = grace.deadline - time.monotonic()
        if remaining <= 0:
            del self._reopen_grace[turn_id]
            self._prune_tracked_turns()
            return 0.0
        return remaining

    def _wait_for_reopen_gate_locked(self, turn_id: str, revision: int) -> None:
        while self._owns_revision_locked(turn_id, revision):
            self._wait_for_pending_reopen_locked(turn_id, revision, self._PENDING_REOPEN_WAIT_TIMEOUT_S)
            if not self._owns_revision_locked(turn_id, revision):
                return
            remaining = self._reopen_grace_remaining_locked(turn_id, revision)
            if remaining <= 0:
                return
            logger.debug("Waiting for speculative reopen grace turn=%s rev=%s", turn_id, revision)
            self._condition.wait(remaining)

    def _wait_for_pending_reopen_locked(self, turn_id: str, revision: int, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        pending = self._pending_reopen.get(turn_id)
        if pending is None or pending.base_revision != revision:
            return
        logger.debug("Waiting for pending speculative reopen turn=%s rev=%s", turn_id, revision)
        while pending is not None and pending.base_revision == revision:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning("Timed out waiting for pending speculative reopen turn=%s rev=%s", turn_id, revision)
                if self._pending_reopen.get(turn_id) == pending:
                    del self._pending_reopen[turn_id]
                    self._prune_tracked_turns()
                    self._condition.notify_all()
                return
            self._condition.wait(remaining)
            pending = self._pending_reopen.get(turn_id)

    def _prune_tracked_turns(self) -> None:
        if self._max_tracked_turns <= 0:
            return

        self._drop_expired_reopen_graces_locked()
        prunable_turn_ids = [
            turn_id
            for turn_id in self._latest_revision
            if turn_id not in self._pending_reopen
            and turn_id not in self._reopen_grace
            and turn_id not in self._discarded_turn_ids
        ]
        while len(prunable_turn_ids) > self._max_tracked_turns:
            turn_id = prunable_turn_ids.pop(0)
            self._latest_revision.pop(turn_id, None)
            self._committed_revision.pop(turn_id, None)
            self._reopen_grace.pop(turn_id, None)

    def _drop_expired_reopen_graces_locked(self) -> None:
        now = time.monotonic()
        for turn_id, grace in list(self._reopen_grace.items()):
            if not self._owns_revision_locked(turn_id, grace.revision) or grace.deadline <= now:
                del self._reopen_grace[turn_id]

    def reset(self) -> None:
        with self._condition:
            self._latest_revision.clear()
            self._committed_revision.clear()
            self._pending_reopen.clear()
            self._reopen_grace.clear()
            self._discarded_turn_ids.clear()
            self._discard_filter = bytearray(self._DISCARD_FILTER_BYTES)
            self._condition.notify_all()
