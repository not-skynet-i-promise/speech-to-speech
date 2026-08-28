from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from openai.types.realtime import (
    ConversationItem,
    RealtimeResponse,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemFunctionCallOutput,
    RealtimeConversationItemUserMessage,
)
from openai.types.realtime.realtime_response import Audio, AudioOutput
from openai.types.realtime.realtime_response_status import RealtimeResponseStatus
from openai.types.realtime.realtime_response_usage import RealtimeResponseUsage

from speech_to_speech.api.openai_realtime.handlers.base import RealtimeBaseHandler
from speech_to_speech.LLM.chat import ChatItemError, add_supported_item, add_supported_items_atomically
from speech_to_speech.pipeline.events import AssistantTextEvent
from speech_to_speech.pipeline.messages import AssistantTextPart, AssistantToolCallPart, GenerateResponseRequest
from speech_to_speech.utils.utils import _generate_id, is_out_of_band, response_wants_audio

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import ServerEvent, _ResponseStatus, _StatusReason

logger = logging.getLogger(__name__)


class ResponseHandler(RealtimeBaseHandler):
    """Owns the response lifecycle: create, cancel, finish, and ID management."""

    # ── ID / state helpers ────────────────────────

    def _ensure_response(self, conn_id: str) -> tuple[str, str]:
        """Ensure a response and output item exist, creating them if needed."""
        st = self._state(conn_id)
        if st.current_response_id is None:
            active_request = st.pending_response_request
            successor = self.pop_next_deferred_request(conn_id)
            st.current_response_id = _generate_id("resp")
            self._start_item(conn_id)
            st.in_response = True
            st.active_response_turn_id = st.pending_response_turn_id
            st.active_response_turn_revision = st.pending_response_turn_revision
            st.active_response_cancel_generation = (
                active_request.cancel_generation if active_request is not None else None
            )
            st.active_response_input_item_id = (
                st.turn_input_item_ids.get(st.active_response_turn_id)
                if st.active_response_turn_id is not None
                else None
            )
            st.active_response_input_item_ids = (
                {st.active_response_input_item_id} if st.active_response_input_item_id is not None else set()
            )
            if successor is None:
                st.response_pending = False
                st.pending_response_turn_id = None
                st.pending_response_turn_revision = None
                st.pending_response_request = None
                st.pending_response_enqueued = False
            else:
                st.response_pending = True
                st.pending_response_turn_id = successor.turn_id
                st.pending_response_turn_revision = successor.turn_revision
                st.pending_response_request = successor
                st.pending_response_enqueued = False
        return st.current_response_id, self._current_item_id(conn_id)

    def pop_next_deferred_request(self, conn_id: str) -> GenerateResponseRequest | None:
        """Pop the next still-current distinct turn from the response FIFO."""
        st = self._state(conn_id)
        tracker = self._service.speculative_turns
        while st.deferred_response_requests:
            request = st.deferred_response_requests.pop(0)
            if tracker is None or tracker.is_latest(request.turn_id, request.turn_revision):
                return request
        return None

    def clear_pending_requests(self, conn_id: str) -> None:
        """Forget every queued response admission after cancellation/barge-in."""
        st = self._state(conn_id)
        st.response_pending = False
        st.pending_response_turn_id = None
        st.pending_response_turn_revision = None
        st.pending_response_request = None
        st.pending_response_enqueued = False
        st.deferred_response_requests.clear()

    def discard_turn(self, conn_id: str, turn_id: str) -> list[ServerEvent]:
        """Discard one superseded speculative turn without stranding the FIFO."""

        st = self._state(conn_id)
        st.deferred_response_requests = [
            request for request in st.deferred_response_requests if request.turn_id != turn_id
        ]
        active_matches = st.in_response and st.active_response_turn_id == turn_id
        pending_matches = st.response_pending and st.pending_response_turn_id == turn_id
        if not active_matches and not pending_matches:
            return []
        if self._service.cancel_scope is not None and (active_matches or not st.in_response):
            self._service.cancel_scope.cancel()
        if active_matches or not st.in_response:
            return self.finish_response(
                conn_id,
                status="cancelled",
                reason="client_cancelled",
            )

        # A different response is active and this turn only owns the held slot.
        # Remove it without cancelling the active generation, then preserve FIFO
        # by moving the next still-current turn into the held slot.
        st.response_pending = False
        st.pending_response_turn_id = None
        st.pending_response_turn_revision = None
        st.pending_response_request = None
        st.pending_response_enqueued = False
        successor = self.pop_next_deferred_request(conn_id)
        if successor is not None:
            self.resume_pending_request(conn_id, successor, enqueue=False)
        return []

    def _end_response(self, conn_id: str, status: _ResponseStatus = "completed") -> None:
        st = self._state(conn_id)
        st.last_closed_response_turn_id = st.active_response_turn_id
        st.last_closed_response_cancel_generation = st.active_response_cancel_generation
        if status == "cancelled":
            st.response_usage.responses_cancelled += 1
        else:
            st.response_usage.responses_completed += 1
        self._service.total_usage += st.response_usage
        logger.info(
            "Response done (status=%s) — this response: input_tokens=%d, output_tokens=%d, audio=%.2fs"
            " | cumulative: input_tokens=%d, output_tokens=%d, audio=%.2fs",
            status,
            st.response_usage.input_tokens,
            st.response_usage.output_tokens,
            st.response_usage.audio_duration_s,
            self._service.total_usage.input_tokens,
            self._service.total_usage.output_tokens,
            self._service.total_usage.audio_duration_s,
        )
        st.response_usage.reset()
        st.current_response_id = None
        st.current_item_id = None
        st.content_index = 0
        st.in_response = False
        st.response_pending = False
        st.pending_response_turn_id = None
        st.pending_response_turn_revision = None
        st.pending_response_request = None
        st.pending_response_enqueued = False
        st.deferred_response_requests.clear()
        st.active_response_turn_id = None
        st.active_response_turn_revision = None
        st.active_response_cancel_generation = None
        st.active_response_input_item_id = None
        st.active_response_input_item_ids.clear()
        st.response_failure_pending = False
        st.current_response_params = None
        st.next_output_index = 0
        st.current_output_index = None
        st.current_output_kind = None
        st.current_output_item_id = None
        st.last_text_item_id = None
        st.last_text_output_index = None
        st.audio_output_started = False
        st.pending_text_outputs = []

    def _start_item(self, conn_id: str) -> str:
        """Generate a new item ID, reset content index, and store it."""
        st = self._state(conn_id)
        item_id = _generate_id("item")
        st.current_item_id = item_id
        st.content_index = 0
        st.input_audio_duration_s = 0.0
        return item_id

    def _current_item_id(self, conn_id: str) -> str:
        return self._state(conn_id).current_item_id or self._start_item(conn_id)

    def _next_content_index(self, conn_id: str) -> int:
        """Return the current content index and advance it."""
        st = self._state(conn_id)
        idx = st.content_index
        st.content_index += 1
        return idx

    def _output_part_context(self, conn_id: str, kind: str) -> tuple[int, str]:
        """Return a stable output index and item id for an ordered part.

        An audio response has one audio item because the TTS queue streams one
        continuous output without per-text-part metadata. All of its text
        therefore stays on output zero while tools receive later items. For a
        text-only response, consecutive text chunks share one assistant item,
        every tool starts a new item, and text after a tool starts a new item.
        """
        st = self._state(conn_id)
        if kind == "text" and response_wants_audio(st.current_response_params):
            item_id = self._current_item_id(conn_id)
            if st.next_output_index == 0:
                st.next_output_index = 1
            st.current_output_index = 0
            st.current_output_kind = "text"
            st.current_output_item_id = item_id
            return 0, item_id
        if kind == "tool_call" and response_wants_audio(st.current_response_params):
            # Output zero is reserved for the response's continuous audio item,
            # even when the model emits a tool before any spoken text.
            st.next_output_index = max(1, st.next_output_index)
        if (
            kind == "text"
            and st.current_output_kind == "text"
            and st.current_output_index is not None
            and st.current_output_item_id is not None
        ):
            return st.current_output_index, st.current_output_item_id

        # Keep the response's audio item stable. Audio chunks are produced on a
        # separate queue and use ``current_item_id``; rebinding it for a tool
        # output would incorrectly attribute the already-streaming audio to the
        # function-call item and reset its content index.
        item_id = self._current_item_id(conn_id) if st.next_output_index == 0 else _generate_id("item")
        output_index = st.next_output_index
        st.next_output_index += 1
        st.current_output_index = output_index
        st.current_output_kind = "text" if kind == "text" else "tool_call"
        st.current_output_item_id = item_id
        return output_index, item_id

    def _build_response(
        self,
        conn_id: str,
        status: _ResponseStatus,
        reason: _StatusReason | None = None,
    ) -> RealtimeResponse:
        """Build a fully-populated RealtimeResponse from the current connection state."""
        st = self._state(conn_id)
        status_details = None
        if reason or status in ("completed", "cancelled", "incomplete", "failed"):
            status_details = RealtimeResponseStatus(type=status, reason=reason)  # type: ignore[arg-type]

        rp = st.current_response_params
        metadata = rp.metadata if rp and rp.metadata else None

        voice: Optional[str] = None
        if rp and rp.audio and rp.audio.output and rp.audio.output.voice:
            voice = str(rp.audio.output.voice)
        if not voice:
            audio_cfg = st.runtime_config.session.audio
            audio_output = audio_cfg.output if audio_cfg is not None else None
            voice = str(audio_output.voice) if audio_output is not None and audio_output.voice else None

        # Out-of-band responses are not threaded into any conversation: report a null id.
        conversation_id = None if is_out_of_band(rp) else st.conversation_id

        return RealtimeResponse(
            id=st.current_response_id,
            object="realtime.response",
            status=status,
            status_details=status_details,
            audio=Audio(output=AudioOutput(voice=str(voice) if voice else None)),  # type: ignore[arg-type]
            conversation_id=conversation_id,
            metadata=metadata,
            usage=RealtimeResponseUsage(
                input_tokens=st.response_usage.input_tokens,
                output_tokens=st.response_usage.output_tokens,
                total_tokens=st.response_usage.input_tokens + st.response_usage.output_tokens,
            ),
        )

    # ── Public handlers ───────────────────────────

    def handle_response_create(self, conn_id: str, event: ResponseCreateEvent) -> ServerEvent | None:
        """Trigger a response.

        Returns a ``ResponseCreatedEvent`` on success, a ``RealtimeErrorEvent``
        on failure, or ``None`` if there is no text_prompt_queue.
        """
        st = self._state(conn_id)
        if event.response:
            if event.response.tool_choice and not isinstance(event.response.tool_choice, str):
                return self.make_error(
                    message="Only string tool_choice values are supported for now (auto, required, none).",
                    _type="tool_choice_not_supported",
                )
        if st.in_response or st.response_pending:
            return self.make_error(
                message="Cannot create response while another response is in progress.",
                _type="conversation_already_has_active_response",
            )

        out_of_band = is_out_of_band(event.response)

        # In-band: response.input items are added to the default conversation here so
        # they appear in history. Out-of-band: leave the default conversation untouched —
        # the input rides along on the request and seeds a throwaway chat in the LM.
        if not out_of_band and event.response and event.response.input:
            input_items = list(event.response.input)
            seen_ids = set(st.protocol_item_ids)
            seen_ids.update(item_id for item in st.deferred_items if (item_id := getattr(item, "id", None)) is not None)
            for item in input_items:
                item_id = getattr(item, "id", None)
                if item_id is None:
                    continue
                if item_id in seen_ids:
                    if st.runtime_config.home_assistant_guard_operational:
                        return self._service.poison_home_assistant_guard(conn_id, "invalid_input_item")
                    return self.make_client_content_error(
                        conn_id,
                        "Conversation item ID already exists.",
                        "duplicate_item_id",
                    )
                seen_ids.add(item_id)
            accepted_dependency_input_ids: set[str] = set()
            accepted_primary_input_id: str | None = None
            accepted_item_ids: list[str] = []

            def record_accepted_item(item: ConversationItem) -> None:
                nonlocal accepted_primary_input_id
                if item.id is None:
                    return
                st.record_protocol_item(item.id)
                accepted_item_ids.append(item.id)
                if isinstance(item, RealtimeConversationItemUserMessage):
                    st.runtime_config.chat.mark_user_message_deletable(item.id)
                    accepted_dependency_input_ids.add(item.id)
                    accepted_primary_input_id = item.id
                elif isinstance(item, RealtimeConversationItemFunctionCallOutput):
                    owner_id = st.runtime_config.chat.response_owner_for_item(item.id)
                    if owner_id is None:
                        return
                    dependency_ids = st.runtime_config.chat.response_dependencies_for_item(item.id) or {owner_id}
                    mapped_input_ids = {
                        input_id
                        for input_id in st.protocol_item_ids
                        if st.input_item_chat_ids.get(input_id, input_id) in dependency_ids
                    }
                    accepted_dependency_input_ids.update(mapped_input_ids or dependency_ids)
                    primary_input_ids = [
                        input_id
                        for input_id in st.protocol_item_ids
                        if st.input_item_chat_ids.get(input_id, input_id) == owner_id
                    ]
                    accepted_primary_input_id = primary_input_ids[-1] if primary_input_ids else owner_id

            def resolve_primary_input_id() -> str | None:
                """Resolve writeback order from canonical chat order, not batch order."""

                dependency_chat_ids = {
                    st.input_item_chat_ids.get(input_id, input_id) for input_id in accepted_dependency_input_ids
                }
                primary_chat_id = st.runtime_config.chat.latest_user_message_id(dependency_chat_ids)
                if primary_chat_id is None:
                    return accepted_primary_input_id
                protocol_candidates = [
                    input_id
                    for input_id in st.protocol_item_ids
                    if st.input_item_chat_ids.get(input_id, input_id) == primary_chat_id
                ]
                return protocol_candidates[-1] if protocol_candidates else primary_chat_id

            try:
                if st.runtime_config.sensitive_content:
                    add_supported_items_atomically(st.runtime_config.chat, input_items)
                    for item in input_items:
                        record_accepted_item(item)
                else:
                    # Preserve the default Realtime behavior: accepted prefix items
                    # remain in ordinary history if a later item is rejected. Private
                    # sessions need the stronger all-or-nothing retention boundary.
                    for item in input_items:
                        add_supported_item(st.runtime_config.chat, item)
                        record_accepted_item(item)
            except ChatItemError as exc:
                if accepted_item_ids:
                    st.response_context_input_item_id = resolve_primary_input_id()
                    st.response_context_input_item_ids = set(accepted_dependency_input_ids)
                    st.response_context_turn_id = None
                    st.response_context_turn_revision = None
                    st.response_context_speech_stopped_at_s = None
                if st.runtime_config.home_assistant_guard_operational:
                    return self._service.poison_home_assistant_guard(conn_id, "invalid_input_item")
                return self.make_client_content_error(conn_id, str(exc), "invalid_input_item")
            # response.input is newer context than any prior audio turn. Track
            # every accepted user dependency because deletion of any one
            # invalidates the response that serialized the batch. Tool outputs
            # inherit the complete dependency set of their originating call.
            st.response_context_input_item_id = resolve_primary_input_id()
            st.response_context_input_item_ids = set(accepted_dependency_input_ids)
            st.response_context_turn_id = None
            st.response_context_turn_revision = None
            st.response_context_speech_stopped_at_s = None

        st.in_response = True
        st.response_pending = False
        st.pending_response_turn_id = None
        st.pending_response_turn_revision = None
        st.pending_response_request = None
        st.pending_response_enqueued = False
        st.deferred_response_requests.clear()
        active_generation = self._service.cancel_scope.generation if self._service.cancel_scope else None
        st.active_response_turn_id = None if out_of_band else st.response_context_turn_id
        st.active_response_turn_revision = None if out_of_band else st.response_context_turn_revision
        st.active_response_cancel_generation = active_generation
        st.active_response_input_item_id = None if out_of_band else st.response_context_input_item_id
        st.active_response_input_item_ids = set() if out_of_band else set(st.response_context_input_item_ids)

        st.current_response_params = event.response
        st.current_response_id = _generate_id("resp")
        self._start_item(conn_id)

        cfg = st.runtime_config
        queue = self._queue(conn_id)
        if queue:
            # Out-of-band responses carry no turn identity: a null turn_id makes every
            # speculative-turn staleness gate treat them as always-latest, so a new user
            # turn mid-generation can never silently drop their output.
            queue.put(
                GenerateResponseRequest(
                    runtime_config=cfg,
                    chat_snapshot=cfg.chat.copy(),
                    response_user_item_id=(
                        st.input_item_chat_ids.get(
                            st.response_context_input_item_id,
                            st.response_context_input_item_id,
                        )
                        if not out_of_band and st.response_context_input_item_id is not None
                        else None
                    ),
                    response_user_item_ids=(
                        {
                            st.input_item_chat_ids.get(input_id, input_id)
                            for input_id in st.response_context_input_item_ids
                        }
                        if not out_of_band
                        else set()
                    ),
                    admitted_protocol_item_ids={
                        *st.protocol_item_ids,
                        *(item.id for item in st.deferred_items if item.id is not None),
                    },
                    admitted_protocol_sequence=st.next_protocol_item_sequence,
                    response=event.response,
                    turn_id=None if out_of_band else st.response_context_turn_id,
                    turn_revision=None if out_of_band else st.response_context_turn_revision,
                    speech_stopped_at_s=None if out_of_band else st.response_context_speech_stopped_at_s,
                    cancel_generation=active_generation,
                )
            )
        logger.debug("response.create received, LLM generation triggered")
        return ResponseCreatedEvent(
            type="response.created",
            event_id=self._next_event_id(),
            response=self._build_response(conn_id, "in_progress"),
        )

    def handle_response_cancel(self, conn_id: str) -> list[ServerEvent]:
        """Cancel the in-progress response and re-enable listening."""
        events = self.finish_response(
            conn_id,
            status="cancelled",
            reason="client_cancelled",
            preserve_pending=False,
        )
        should_listen = self._should_listen(conn_id)
        if should_listen:
            should_listen.set()
        logger.info("Response cancelled, listening re-enabled")
        return events

    def finish_response(
        self,
        conn_id: str,
        status: _ResponseStatus = "completed",
        reason: _StatusReason | None = None,
        *,
        preserve_pending: bool = True,
        enqueue_pending: bool = True,
    ) -> list[ServerEvent]:
        """Close the current response (audio/text done + response done).

        Audio responses emit ``response.output_audio.done`` for any terminal
        status. Text-only responses emit one ``response.output_text.done`` per
        contiguous text output item, but only on ``status="completed"`` — a
        cancelled or failed text response sends no audio, so it just closes
        with ``response.done``.
        """
        st = self._state(conn_id)
        if status == "completed" and st.response_failure_pending:
            status = "failed"
        deferred_requests: list[GenerateResponseRequest] = []
        if preserve_pending:
            if st.in_response:
                if st.pending_response_request is not None:
                    deferred_requests.append(st.pending_response_request)
                deferred_requests.extend(st.deferred_response_requests)
            elif st.response_pending:
                deferred_requests.extend(st.deferred_response_requests)
        events: list[ServerEvent] = []
        if not st.in_response and st.response_pending:
            self._ensure_response(conn_id)
            events.append(
                ResponseCreatedEvent(
                    type="response.created",
                    event_id=self._next_event_id(),
                    response=self._build_response(conn_id, "in_progress"),
                )
            )
        if st.in_response:
            active_response_user_id = (
                st.input_item_chat_ids.get(
                    st.active_response_input_item_id,
                    st.active_response_input_item_id,
                )
                if st.active_response_input_item_id is not None
                else None
            )
            resp_id, item_id = self._ensure_response(conn_id)
            if response_wants_audio(st.current_response_params) and st.audio_output_started:
                events.append(
                    ResponseAudioDoneEvent(
                        type="response.output_audio.done",
                        event_id=self._next_event_id(),
                        content_index=0,
                        item_id=st.last_text_item_id or item_id,
                        output_index=(st.last_text_output_index if st.last_text_output_index is not None else 0),
                        response_id=resp_id,
                    )
                )
            elif status == "completed":
                for pending in st.pending_text_outputs:
                    events.append(
                        ResponseTextDoneEvent(
                            type="response.output_text.done",
                            event_id=self._next_event_id(),
                            content_index=0,
                            item_id=str(pending["item_id"]),
                            output_index=int(pending["output_index"]),
                            response_id=resp_id,
                            text="".join(pending["parts"]),
                        )
                    )
            events.append(
                ResponseDoneEvent(
                    type="response.done",
                    event_id=self._next_event_id(),
                    response=self._build_response(conn_id, status, reason),
                )
            )
            self._end_response(conn_id, status)
            st.runtime_config.chat.release_response_turn(
                active_response_user_id,
                force=status != "completed",
            )
        elif st.response_pending:
            self.clear_pending_requests(conn_id)
        # Apply any client items that arrived mid-generation now that in_response
        # is cleared and the generation's own write-back has landed. Done outside
        # the in_response guard so a stray terminal call still drains the buffer.
        cfg = st.runtime_config
        if cfg.sensitive_content:
            with cfg.transcript_barrier_state_guard():
                if not cfg.transcript_barrier_pending and not cfg.private_protocol_failed:
                    events.extend(self._service.conversation.flush_deferred_items(conn_id))
        elif not cfg.transcript_barrier_pending:
            # An ordinary provider may still own the content guard while a
            # cancelled response is closing. Private activation is separately
            # serialized against its response lease, so do not wait here.
            if not cfg.private_protocol_failed:
                events.extend(self._service.conversation.flush_deferred_items(conn_id))
        if not cfg.private_protocol_failed:
            tracker = self._service.speculative_turns
            while deferred_requests:
                deferred_request = deferred_requests.pop(0)
                if tracker is not None and not tracker.is_latest(
                    deferred_request.turn_id,
                    deferred_request.turn_revision,
                ):
                    continue
                st.deferred_response_requests = deferred_requests
                self.resume_pending_request(conn_id, deferred_request, enqueue=enqueue_pending)
                break
        return events

    def resume_pending_request(
        self,
        conn_id: str,
        request: GenerateResponseRequest,
        *,
        enqueue: bool = True,
    ) -> None:
        """Re-admit one held successor after its preceding response closes."""
        st = self._state(conn_id)
        generation = self._service.cancel_scope.generation if self._service.cancel_scope else None
        target_input_id = st.turn_input_item_ids.get(request.turn_id) if request.turn_id is not None else None
        target_chat_id = st.input_item_chat_ids.get(target_input_id) if target_input_id is not None else None
        later_chat_ids: set[str] = set()
        for later_request in st.deferred_response_requests:
            if later_request.turn_id is None:
                continue
            later_input_id = st.turn_input_item_ids.get(later_request.turn_id)
            if later_input_id is None:
                continue
            later_chat_id = st.input_item_chat_ids.get(later_input_id)
            if later_chat_id is not None:
                later_chat_ids.add(later_chat_id)
        if request.admitted_protocol_sequence is not None:
            future_protocol_ids = {
                item_id
                for item_id in st.protocol_item_ids
                if st.protocol_item_sequences.get(item_id, 0) > request.admitted_protocol_sequence
            }
        else:
            future_protocol_ids = (
                set(st.protocol_item_ids) - request.admitted_protocol_item_ids
                if request.admitted_protocol_item_ids is not None
                else set()
            )
        future_chat_ids = {st.input_item_chat_ids.get(item_id, item_id) for item_id in future_protocol_ids}
        refreshed_chat = (
            st.runtime_config.chat.snapshot_for_response_turn(
                target_chat_id,
                later_chat_ids,
                fallback_user=(
                    request.chat_snapshot.user_message(target_chat_id) if request.chat_snapshot is not None else None
                ),
                fallback_init_message=(
                    request.chat_snapshot.init_chat_message if request.chat_snapshot is not None else None
                ),
                excluded_item_ids=future_chat_ids,
            )
            if target_chat_id is not None
            else request.chat_snapshot
        )
        request = request.model_copy(update={"cancel_generation": generation, "chat_snapshot": refreshed_chat})
        st.response_pending = True
        st.pending_response_turn_id = request.turn_id
        st.pending_response_turn_revision = request.turn_revision
        st.pending_response_request = request
        st.pending_response_enqueued = False
        if enqueue:
            self.enqueue_pending_request(conn_id)

    def enqueue_pending_request(self, conn_id: str) -> None:
        """Release a held successor exactly once after cancelled output is flushed."""
        st = self._state(conn_id)
        request = st.pending_response_request
        if not st.response_pending or request is None or st.pending_response_enqueued:
            return
        queue = self._queue(conn_id)
        if queue is not None:
            queue.put(request)
            st.pending_response_enqueued = True

    # ── Pipeline event handlers ───────────────────

    def on_assistant_text(
        self,
        conn_id: str,
        event: AssistantTextEvent,
        *,
        wait_for_pending_reopen: bool = True,
    ) -> list[ServerEvent] | None:
        """Handle assistant_text: create the implicit response, then emit its ordered parts."""
        if self._service.speculative_turns:
            commit_result: bool | None
            if wait_for_pending_reopen:
                commit_result = self._service.speculative_turns.commit_if_latest_after_reopen_grace(
                    event.turn_id,
                    event.turn_revision,
                )
            else:
                commit_result = self._service.speculative_turns.try_commit_if_latest_after_reopen_grace(
                    event.turn_id,
                    event.turn_revision,
                )
            if commit_result is None:
                return None
            if not commit_result:
                logger.debug("Dropping stale assistant text for turn=%s rev=%s", event.turn_id, event.turn_revision)
                return []
        st = self._state(conn_id)
        if st.in_response:
            owner_turn_id = st.active_response_turn_id
            owner_generation = st.active_response_cancel_generation
        elif st.response_pending and st.pending_response_request is not None:
            owner_turn_id = st.pending_response_turn_id
            owner_generation = st.pending_response_request.cancel_generation
        else:
            owner_turn_id = None
            owner_generation = None
        if not st.in_response and not st.response_pending:
            closed_owner = (
                event.turn_id == st.last_closed_response_turn_id
                if event.turn_id is not None
                else (
                    event.cancel_generation is not None
                    and event.cancel_generation == st.last_closed_response_cancel_generation
                )
            )
            if closed_owner:
                logger.debug("Dropping assistant output after its response slot closed")
                return []
        if event.turn_id is not None and owner_turn_id is not None and event.turn_id != owner_turn_id:
            logger.debug(
                "Dropping assistant output for non-active turn=%s (owner=%s)",
                event.turn_id,
                owner_turn_id,
            )
            return []
        if (
            event.cancel_generation is not None
            and owner_generation is not None
            and event.cancel_generation != owner_generation
        ):
            logger.debug(
                "Dropping assistant output for non-active generation=%s (owner=%s)",
                event.cancel_generation,
                owner_generation,
            )
            return []
        if not any(
            isinstance(part, AssistantToolCallPart) or (isinstance(part, AssistantTextPart) and bool(part.text))
            for part in event.parts
        ):
            return []
        events: list[ServerEvent] = []
        need_created = st.current_response_id is None
        resp_id, _ = self._ensure_response(conn_id)
        if need_created:
            events.append(
                ResponseCreatedEvent(
                    type="response.created",
                    event_id=self._next_event_id(),
                    response=self._build_response(conn_id, "in_progress"),
                )
            )
        for part in event.parts:
            if isinstance(part, AssistantTextPart):
                if not part.text:
                    continue
                output_idx, item_id = self._output_part_context(conn_id, "text")
                st.last_text_item_id = item_id
                st.last_text_output_index = output_idx
                if response_wants_audio(st.current_response_params):
                    events.append(
                        ResponseAudioTranscriptDoneEvent(
                            type="response.output_audio_transcript.done",
                            event_id=self._next_event_id(),
                            content_index=0,
                            item_id=item_id,
                            output_index=output_idx,
                            response_id=resp_id,
                            transcript=part.text,
                        )
                    )
                else:
                    # Stream the delta now; finish_response emits one matching
                    # done event for each contiguous text output item.
                    if not st.pending_text_outputs or st.pending_text_outputs[-1]["output_index"] != output_idx:
                        st.pending_text_outputs.append({"item_id": item_id, "output_index": output_idx, "parts": []})
                    st.pending_text_outputs[-1]["parts"].append(part.text)
                    events.append(
                        ResponseTextDeltaEvent(
                            type="response.output_text.delta",
                            event_id=self._next_event_id(),
                            content_index=0,
                            item_id=item_id,
                            output_index=output_idx,
                            response_id=resp_id,
                            delta=part.text,
                        )
                    )
            elif isinstance(part, AssistantToolCallPart):
                tool = part.tool
                output_idx, item_id = self._output_part_context(conn_id, "tool_call")
                st.response_usage.tool_calls += 1
                events.append(
                    ResponseFunctionCallArgumentsDoneEvent(
                        type="response.function_call_arguments.done",
                        event_id=self._next_event_id(),
                        call_id=tool.call_id,
                        name=tool.name,
                        arguments=tool.arguments,
                        item_id=item_id,
                        output_index=output_idx,
                        response_id=resp_id,
                    )
                )
            st.record_protocol_item(item_id)
        return events
