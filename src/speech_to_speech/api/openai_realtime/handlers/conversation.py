from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openai.types.realtime import (
    ConversationItem,
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
)
from openai.types.realtime.conversation_item import RealtimeConversationItemUserMessage
from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import (
    UsageTranscriptTextUsageDuration,
)

from speech_to_speech.api.openai_realtime.handlers.base import RealtimeBaseHandler
from speech_to_speech.LLM.chat import ChatItemError, add_supported_item, add_supported_items_atomically
from speech_to_speech.pipeline.events import PartialTranscriptionEvent, TranscriptionCompletedEvent

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import ServerEvent

logger = logging.getLogger(__name__)


class ConversationHandler(RealtimeBaseHandler):
    """Owns conversation item injection and pipeline-to-protocol translation."""

    def handle_conversation_item_create(
        self,
        conn_id: str,
        event: ConversationItemCreateEvent,
    ) -> list[ServerEvent]:
        """Inject a text message or function-call output into the LLM context.

        Items are added to the LLM chat context but do NOT trigger response
        generation on their own.  A subsequent ``response.create`` event is
        required to trigger the model.

        While a response is generating, the item is *deferred*: applying it now
        would race the LLM handler's end-of-turn chat write-back, which runs on
        the pipeline thread (e.g. a ``function_call_output`` arriving before its
        ``function_call`` is recorded, or an image stripped before the next
        turn reads it). Deferred items are flushed, in order, once the response
        completes — see :meth:`flush_deferred_items`.
        """
        st = self._state(conn_id)
        if self._item_id_exists(st, event.item):
            return [self._duplicate_item_error(conn_id, event.event_id)]
        if st.in_response:
            st.deferred_items.append(event.item)
            logger.debug("Deferred conversation item until the active response completes")
            return []
        return self._apply_item(conn_id, event.item)

    @staticmethod
    def _item_id_exists(st, item: ConversationItem) -> bool:
        item_id = getattr(item, "id", None)
        if item_id is None:
            return False
        return item_id in st.protocol_item_ids or any(
            getattr(existing, "id", None) == item_id for existing in st.deferred_items
        )

    def _duplicate_item_error(self, conn_id: str, event_id: str | None = None) -> ServerEvent:
        st = self._state(conn_id)
        if st.runtime_config.home_assistant_guard_operational:
            error = self._service.poison_home_assistant_guard(conn_id, "invalid_conversation_item")
        else:
            error = self.make_client_content_error(conn_id, "Conversation item ID already exists.", "duplicate_item_id")
        error.error.event_id = event_id
        return error

    def _apply_item(self, conn_id: str, item: ConversationItem) -> list[ServerEvent]:
        """Add one item to the chat and build its ``conversation.item.created``."""
        try:
            self._append_item(conn_id, item)
        except ChatItemError as exc:
            if self._state(conn_id).runtime_config.home_assistant_guard_operational:
                return [self._service.poison_home_assistant_guard(conn_id, "invalid_conversation_item")]
            return [self.make_client_content_error(conn_id, str(exc), "invalid_conversation_item")]

        return [] if not item else [self._item_created_event(conn_id, item)]

    def _item_created_event(self, conn_id: str, item: ConversationItem) -> ConversationItemCreatedEvent:
        """Build the acknowledgement for an item already committed to chat."""
        st = self._state(conn_id)
        previous_item_id = st.last_item_id
        event = ConversationItemCreatedEvent(
            type="conversation.item.created",
            event_id=self._next_event_id(),
            previous_item_id=previous_item_id,
            item=item,
        )
        if item.id is not None:
            st.record_protocol_item(item.id)
            if isinstance(item, RealtimeConversationItemUserMessage):
                st.runtime_config.chat.mark_user_message_deletable(item.id)
        return event

    def flush_deferred_items(self, conn_id: str) -> list[ServerEvent]:
        """Apply items buffered during a response, in arrival order.

        Called at response completion (after the generation's own write-back),
        so a ``function_call_output`` pairs with its now-recorded ``function_call``
        and an image survives the just-finished response's ``strip_images``.
        """
        st = self._state(conn_id)
        if not st.deferred_items:
            return []
        items = st.deferred_items
        st.deferred_items = []
        seen_ids = set(st.protocol_item_ids)
        for item in items:
            item_id = getattr(item, "id", None)
            if item_id is not None and item_id in seen_ids:
                return [self._duplicate_item_error(conn_id)]
            if item_id is not None:
                seen_ids.add(item_id)
        try:
            add_supported_items_atomically(st.runtime_config.chat, items)
        except ChatItemError as exc:
            if st.runtime_config.home_assistant_guard_operational:
                return [self._service.poison_home_assistant_guard(conn_id, "invalid_conversation_item")]
            return [self.make_client_content_error(conn_id, str(exc), "invalid_conversation_item")]
        return [self._item_created_event(conn_id, item) for item in items if item]

    def _append_item(self, conn_id: str, item: ConversationItem) -> None:
        """Narrow ``ConversationItem`` to ``SupportedItem`` and delegate to ``Chat.add_item``.

        Raises :class:`ChatItemError` on validation failure or unsupported type.
        """
        add_supported_item(self._state(conn_id).runtime_config.chat, item)

    def handle_conversation_item_delete(
        self,
        conn_id: str,
        event: ConversationItemDeleteEvent,
        *,
        defer_successor_enqueue: bool = False,
    ) -> list[ServerEvent]:
        """Remove one exact user item and acknowledge only after history changed."""
        st = self._state(conn_id)
        input_item = event.item_id in st.audio_input_item_ids
        protocol_item = event.item_id in st.protocol_item_ids
        chat_item_id = st.input_item_chat_ids.get(event.item_id, event.item_id)
        mapped_input = input_item and event.item_id in st.input_item_chat_ids
        removed = (
            st.runtime_config.chat.remove_user_message(chat_item_id)
            if mapped_input or (not input_item and protocol_item)
            else False
        )
        if input_item and not mapped_input:
            removed = event.item_id in st.protocol_item_ids
        if not removed and not input_item:
            for index, item in enumerate(st.deferred_items):
                if isinstance(item, RealtimeConversationItemUserMessage) and item.id in {
                    event.item_id,
                    chat_item_id,
                }:
                    del st.deferred_items[index]
                    removed = True
                    break
        if not removed:
            error = self.make_client_content_error(conn_id, "Conversation item was not found.", "item_not_found")
            error.error.event_id = event.event_id
            return [error]

        turn_id = st.input_item_turn_ids.get(event.item_id) if input_item else None
        if input_item:
            st.record_deleted_input_item(event.item_id)
            if self._service.speculative_turns is not None:
                self._service.speculative_turns.discard(turn_id)
            st.input_item_chat_ids.pop(event.item_id, None)
        if st.speculative_user_item_id == chat_item_id:
            st.speculative_user_item_id = None
        if st.speculative_input_item_id == event.item_id:
            st.speculative_input_item_id = None
        if turn_id is not None and st.speculative_turn_id == turn_id:
            st.speculative_turn_id = None
            st.speculative_turn_revision = None
        if turn_id is not None and st.speculative_user_turn_id == turn_id:
            st.speculative_user_turn_id = None
            st.speculative_user_turn_revision = None
            st.speculative_user_speech_stopped_at_s = None
            st.speculative_audio_duration_s = 0.0

        pending_matches = turn_id is not None and st.pending_response_turn_id == turn_id
        active_matches = turn_id is not None and st.active_response_turn_id == turn_id
        promote_successor = pending_matches and not st.in_response
        if pending_matches:
            st.response_pending = False
            st.pending_response_turn_id = None
            st.pending_response_turn_revision = None
            st.pending_response_request = None
            st.pending_response_enqueued = False
        if (active_matches or (pending_matches and not st.in_response)) and self._service.cancel_scope is not None:
            self._service.cancel_scope.cancel()
        # Retire the deleted item before closing its response.  Closing flushes
        # items deferred during generation, and their previous_item_id must
        # never point at the item this operation just removed.
        st.remove_protocol_item(event.item_id)
        response_events: list[ServerEvent] = []
        if active_matches:
            response_events = self._service.response.finish_response(
                conn_id,
                status="cancelled",
                reason="client_cancelled",
                enqueue_pending=not defer_successor_enqueue,
            )
        elif promote_successor:
            successor_request = self._service.response.pop_next_deferred_request(conn_id)
            if successor_request is not None:
                self._service.response.resume_pending_request(
                    conn_id,
                    successor_request,
                    enqueue=not defer_successor_enqueue,
                )
        if active_matches or pending_matches:
            should_listen = self._should_listen(conn_id)
            if should_listen is not None:
                should_listen.set()
        return [
            ConversationItemDeletedEvent(
                type="conversation.item.deleted",
                event_id=self._next_event_id(),
                item_id=event.item_id,
            ),
            *response_events,
        ]

    # ── Pipeline event handlers ────────────────────

    def on_partial_transcription(self, conn_id: str, event: PartialTranscriptionEvent) -> list[ServerEvent]:
        """Handle partial_transcription: emit transcription delta event."""
        return [
            ConversationItemInputAudioTranscriptionDeltaEvent(
                type="conversation.item.input_audio_transcription.delta",
                event_id=self._next_event_id(),
                content_index=self._next_input_content_index(conn_id),
                item_id=self._input_item_id(conn_id),
                delta=event.delta,
            )
        ]

    def on_transcription_completed(
        self,
        conn_id: str,
        event: TranscriptionCompletedEvent,
        *,
        item_id: str | None = None,
    ) -> list[ServerEvent]:
        """Handle transcription_completed: accumulate duration and emit completed event."""
        st = self._state(conn_id)
        st.response_usage.audio_duration_s += st.input_audio_duration_s
        return [
            ConversationItemInputAudioTranscriptionCompletedEvent(
                type="conversation.item.input_audio_transcription.completed",
                event_id=self._next_event_id(),
                content_index=0,
                item_id=item_id or self._input_item_id(conn_id),
                transcript=event.transcript,
                usage=UsageTranscriptTextUsageDuration(
                    seconds=st.input_audio_duration_s,
                    type="duration",
                ),
            )
        ]
