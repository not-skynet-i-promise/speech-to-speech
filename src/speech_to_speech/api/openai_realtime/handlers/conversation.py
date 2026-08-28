from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unicodedata import category

from openai.types.realtime import (
    ConversationItem,
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    ConversationItemInputAudioTranscriptionFailedEvent,
)
from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import (
    UsageTranscriptTextUsageDuration,
)
from openai.types.realtime.conversation_item_input_audio_transcription_failed_event import (
    Error as InputAudioTranscriptionError,
)
from openai.types.realtime.realtime_conversation_item_function_call_output import (
    RealtimeConversationItemFunctionCallOutput,
)
from openai.types.realtime.realtime_conversation_item_user_message import (
    RealtimeConversationItemUserMessage,
)

from speech_to_speech.api.openai_realtime.handlers.base import RealtimeBaseHandler
from speech_to_speech.LLM.chat import ChatItemError, add_supported_item
from speech_to_speech.pipeline.events import (
    PartialTranscriptionEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)
from speech_to_speech.pipeline.transcript_logging import transcript_for_log

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import ServerEvent

logger = logging.getLogger(__name__)


def _transcript_words(transcript: str) -> list[tuple[str, str]]:
    """Return display and comparison forms for whitespace-delimited words."""
    words: list[tuple[str, str]] = []
    for token in transcript.split():
        start = 0
        end = len(token)
        while start < end and category(token[start]).startswith("P"):
            start += 1
        while end > start and category(token[end - 1]).startswith("P"):
            end -= 1
        display = token[start:end]
        if display:
            words.append((display, display.casefold()))
    return words


def _stable_transcript_words(previous: str, current: str) -> list[tuple[str, str]]:
    """Find words confirmed by two hypotheses, excluding their unstable tail."""
    previous_words = _transcript_words(previous)
    current_words = _transcript_words(current)
    common_count = 0
    for previous_word, current_word in zip(previous_words, current_words):
        if previous_word[1] != current_word[1]:
            break
        common_count += 1

    # The last matching word was at the speculative edge of the previous
    # hypothesis. Hold it back until another update adds context after it.
    return current_words[: max(0, common_count - 1)]


class ConversationHandler(RealtimeBaseHandler):
    """Owns conversation item injection and pipeline-to-protocol translation."""

    @staticmethod
    def _is_image_message(item: ConversationItem) -> bool:
        return (
            isinstance(item, RealtimeConversationItemUserMessage)
            and bool(item.content)
            and all(part.type == "input_image" for part in item.content)
        )

    def _tool_followup_inputs_are_ordered(
        self,
        conn_id: str,
        items: list[ConversationItem],
    ) -> bool:
        """Return whether deferred items form a prefetch-safe tool batch.

        Image items are allowed immediately before a function output when
        ``previous_item_id`` confirms that insertion order. The field does not
        imply ownership, so every image remains an ordinary conversation item
        and must survive if the tool response is later rolled back.
        """
        st = self._state(conn_id)
        for index, item in enumerate(items):
            if isinstance(item, RealtimeConversationItemFunctionCallOutput):
                continue
            if not self._is_image_message(item) or item.id is None or index + 1 >= len(items):
                return False
            following = items[index + 1]
            if not isinstance(following, RealtimeConversationItemFunctionCallOutput):
                return False
            if st.deferred_function_output_previous_item_ids.get(following.call_id) != item.id:
                return False
        return True

    def handle_conversation_item_create(
        self,
        conn_id: str,
        event: ConversationItemCreateEvent,
    ) -> list[ServerEvent]:
        """Inject a text message or function-call output into the LLM context.

        Items are added to the LLM chat context but do NOT trigger response
        generation on their own.  A subsequent ``response.create`` event is
        required to trigger the model.

        While model generation is active, items remain deferred so a fast tool
        result cannot overtake later assistant items from the same response.
        Once logical generation completes, tool outputs can be applied to the
        internal chat immediately; their wire acknowledgements remain ordered
        behind the response's still-buffered output.
        """
        st = self._state(conn_id)
        if st.in_response:
            if isinstance(event.item, RealtimeConversationItemFunctionCallOutput):
                st.deferred_function_output_previous_item_ids[event.item.call_id] = event.previous_item_id
            st.deferred_items.append(event.item)
            if (
                st.current_response_key in st.generation_done_tool_calls
                and any(isinstance(item, RealtimeConversationItemFunctionCallOutput) for item in st.deferred_items)
                and self._tool_followup_inputs_are_ordered(conn_id, st.deferred_items)
            ):
                return self.flush_deferred_items(
                    conn_id,
                    tool_followup_inputs_only=True,
                    defer_acknowledgements=True,
                )
            logger.debug("Deferred conversation item until the active response completes")
            return []
        return self._apply_item(conn_id, event.item)

    def handle_conversation_item_delete(
        self,
        conn_id: str,
        event: ConversationItemDeleteEvent,
    ) -> list[ServerEvent]:
        """Delete one conversation item by its wire-visible protocol ID."""
        st = self._state(conn_id)
        preceding_events: list[ServerEvent] = []
        pending_acks = [item for item in st.pending_item_acks if item.id == event.item_id]
        if pending_acks:
            # The create was already applied successfully but its acknowledgement
            # was ordered behind response output. Complete that protocol operation
            # before acknowledging the subsequent deletion.
            st.pending_item_acks = [item for item in st.pending_item_acks if item.id != event.item_id]
            preceding_events.extend(self._ack_item(conn_id, item) for item in pending_acks)

        found = event.item_id in st.conversation_item_order
        chat_item_id = st.conversation_item_chat_ids.get(event.item_id, event.item_id)
        history_was_bound = st.runtime_config.chat.remove_item(chat_item_id)
        if history_was_bound:
            found = True

        removed_call_ids: set[str] = set()
        if found:
            removed_text_outputs = [
                pending for pending in st.pending_text_outputs if pending["item_id"] == event.item_id
            ]
            st.pending_text_outputs = [
                pending for pending in st.pending_text_outputs if pending["item_id"] != event.item_id
            ]
            if removed_text_outputs:
                removed = removed_text_outputs[0]
                st.deleted_response_outputs[int(removed["output_index"])] = {
                    "kind": "text",
                    "item_id": str(removed["item_id"]),
                }
                if not history_was_bound:
                    st.deleted_response_text_outputs[event.item_id] = {
                        "item_id": removed["item_id"],
                        "output_index": removed["output_index"],
                    }

            for output_index, call in tuple(st.pending_function_calls.items()):
                if call.id != event.item_id:
                    continue
                if call.call_id is not None:
                    removed_call_ids.add(call.call_id)
                del st.pending_function_calls[output_index]
                st.finished_function_call_indices.discard(output_index)
                st.deleted_response_outputs[output_index] = {
                    "kind": "function_call",
                    "item_id": str(call.id),
                    "call_id": call.call_id,
                    "name": call.name,
                }
                if not history_was_bound:
                    st.deleted_response_function_calls[event.item_id] = (output_index, call.call_id)
        for call_ids in st.generation_done_tool_calls.values():
            call_ids.difference_update(removed_call_ids)

        if event.item_id in st.input_items:
            deleted_turns = [
                turn
                for turn, tracked_item_id in st.input_item_by_turn_revision.items()
                if tracked_item_id == event.item_id
            ]
            for turn_id, turn_revision in deleted_turns:
                st.tombstone_input_terminal(turn_id, turn_revision)
            if st.current_input_item_id == event.item_id:
                # Metadata-free transcription/direct-audio terminals resolve
                # through the active item rather than a turn routing key.
                st.tombstone_input_terminal(None, None)
            st.input_items.pop(event.item_id, None)
            # Keep the turn route until its terminal arrives. The missing item
            # then suppresses that terminal instead of falling back to a newly
            # generated protocol ID and resurrecting the deleted speech.
            if st.current_input_item_id == event.item_id:
                st.current_input_item_id = None
            found = True

        if not found:
            return preceding_events or [
                self.make_error(
                    f"Conversation item '{event.item_id}' was not found",
                    "item_not_found",
                    event.event_id,
                )
            ]
        if st.pending_assistant_item_id == event.item_id:
            st.pending_assistant_item_id = None
            st.pending_assistant_output_index = None
            st.audio_output_started = False
        if st.current_item_id == event.item_id:
            st.current_item_id = None
            st.current_output_index = None
            st.current_output_kind = None
        preceding_events.extend(self._service.retract_queued_input_response(conn_id, event.item_id))
        st.forget_conversation_item(event.item_id)
        st.tombstone_conversation_item(event.item_id)
        return [
            *preceding_events,
            ConversationItemDeletedEvent(
                type="conversation.item.deleted",
                event_id=self._next_event_id(),
                item_id=event.item_id,
            ),
        ]

    def _apply_item(
        self,
        conn_id: str,
        item: ConversationItem,
        *,
        defer_acknowledgement: bool = False,
    ) -> list[ServerEvent]:
        """Add one item to the chat and build its ``conversation.item.created``."""
        try:
            self._append_item(conn_id, item)
        except ChatItemError as exc:
            return [self.make_error(str(exc), "invalid_conversation_item")]

        if not item:
            return []
        st = self._state(conn_id)
        if defer_acknowledgement:
            # The prefetching LM strips consumed images from Chat in place.
            # Keep the protocol echo immutable until it can be acknowledged in
            # order behind the origin response.
            st.pending_item_acks.append(item.model_copy(deep=True))
            return []
        return [self._ack_item(conn_id, item)]

    def _ack_item(self, conn_id: str, item: ConversationItem) -> ConversationItemCreatedEvent:
        """Build one ordered acknowledgement for an item already in the chat."""
        st = self._state(conn_id)
        event = ConversationItemCreatedEvent(
            type="conversation.item.created",
            event_id=self._next_event_id(),
            previous_item_id=st.last_item_id,
            item=item,
        )
        if item.id is not None:
            st.record_conversation_item(item.id, item.id)
        return event

    def flush_deferred_items(
        self,
        conn_id: str,
        *,
        tool_followup_inputs_only: bool = False,
        defer_acknowledgements: bool = False,
    ) -> list[ServerEvent]:
        """Apply items buffered during a response, in arrival order.

        Called as soon as model generation has committed its history, or at
        response completion as a fallback if the side-channel event is delayed.
        """
        st = self._state(conn_id)
        if not st.deferred_items:
            return []
        has_function_output = any(
            isinstance(item, RealtimeConversationItemFunctionCallOutput) for item in st.deferred_items
        )
        inputs_are_ordered = self._tool_followup_inputs_are_ordered(conn_id, st.deferred_items)
        if tool_followup_inputs_only and (not has_function_output or not inputs_are_ordered):
            return []
        items = st.deferred_items
        st.deferred_items = []
        for item in items:
            if isinstance(item, RealtimeConversationItemFunctionCallOutput):
                st.deferred_function_output_previous_item_ids.pop(item.call_id, None)
        events: list[ServerEvent] = []
        for item in items:
            events.extend(
                self._apply_item(
                    conn_id,
                    item,
                    defer_acknowledgement=defer_acknowledgements,
                )
            )
        return events

    def flush_pending_item_acks(
        self,
        conn_id: str,
        *,
        revalidate_tool_outputs: bool = False,
    ) -> list[ServerEvent]:
        """Emit acknowledgements deferred behind an active response's output."""
        st = self._state(conn_id)
        items = st.pending_item_acks
        st.pending_item_acks = []
        events: list[ServerEvent] = []
        for item in items:
            if revalidate_tool_outputs and isinstance(item, RealtimeConversationItemFunctionCallOutput):
                events.extend(self._apply_item(conn_id, item))
            else:
                events.append(self._ack_item(conn_id, item))
        return events

    def _append_item(self, conn_id: str, item: ConversationItem) -> None:
        """Narrow ``ConversationItem`` to ``SupportedItem`` and delegate to ``Chat.add_item``.

        Raises :class:`ChatItemError` on validation failure or unsupported type.
        """
        add_supported_item(self._state(conn_id).runtime_config.chat, item)

    # ── Pipeline event handlers ────────────────────

    def on_partial_transcription(self, conn_id: str, event: PartialTranscriptionEvent) -> list[ServerEvent]:
        """Stabilize a cumulative STT hypothesis into an append-only Realtime delta."""
        st = self._state(conn_id)
        item_id = self._input_item_id(conn_id, event.turn_id, event.turn_revision)
        if item_id is None:
            logger.debug(
                "Ignoring partial transcription for unknown turn=%s rev=%s",
                event.turn_id,
                event.turn_revision,
            )
            return []
        input_item = st.input_items.get(item_id)
        if input_item is None:
            logger.debug("Ignoring partial transcription for released item=%s", item_id)
            return []
        hypothesis = event.delta.strip()
        if not hypothesis or hypothesis == input_item.latest_transcript:
            return []

        previous = input_item.latest_transcript
        input_item.latest_transcript = hypothesis
        if not previous:
            return []

        stable_words = _stable_transcript_words(previous, hypothesis)
        emitted_words = _transcript_words(input_item.transcript_prefix)
        emitted_comparison = [word[1] for word in emitted_words]
        stable_comparison = [word[1] for word in stable_words]
        if stable_comparison[: len(emitted_comparison)] != emitted_comparison:
            # A word that already reached the wire was later revised. Realtime
            # has no retraction event, so wait for a future hypothesis that
            # extends the committed prefix or for the authoritative completion.
            logger.debug(
                "Withholding revised stable transcription for item=%s (emitted=%s, hypothesis=%s)",
                item_id,
                transcript_for_log(input_item.transcript_prefix),
                transcript_for_log(hypothesis),
            )
            return []

        new_words = stable_words[len(emitted_words) :]
        if not new_words:
            return []

        delta = (" " if emitted_words else "") + " ".join(word[0] for word in new_words)
        input_item.transcript_prefix += delta
        return [
            ConversationItemInputAudioTranscriptionDeltaEvent(
                type="conversation.item.input_audio_transcription.delta",
                event_id=self._next_event_id(),
                content_index=0,
                item_id=item_id,
                delta=delta,
            )
        ]

    def terminalize_input_item(
        self,
        conn_id: str,
        item_id: str,
    ) -> tuple[str, float] | None:
        """Release one input item's active transcript and routing state."""
        st = self._state(conn_id)
        input_item = st.input_items.pop(item_id, None)
        if input_item is None:
            logger.debug("Ignoring input terminal for released item=%s", item_id)
            st.input_item_by_turn_revision = {
                turn: tracked_item_id
                for turn, tracked_item_id in st.input_item_by_turn_revision.items()
                if tracked_item_id != item_id
            }
            if st.current_input_item_id == item_id:
                st.current_input_item_id = None
            return None
        duration_s = input_item.audio_duration_s
        st.input_item_by_turn_revision = {
            turn: tracked_item_id
            for turn, tracked_item_id in st.input_item_by_turn_revision.items()
            if tracked_item_id != item_id
        }
        if st.current_input_item_id == item_id:
            st.current_input_item_id = None
        return item_id, duration_s

    def _completion_input_item_id(
        self,
        conn_id: str,
        turn_id: str | None,
        turn_revision: int | None,
    ) -> str | None:
        """Resolve a final to its routed item, falling back to the active input."""
        st = self._state(conn_id)
        if turn_id is not None:
            routed_item_id = st.input_item_by_turn_revision.get((turn_id, turn_revision))
            if routed_item_id is not None:
                return routed_item_id
        return st.current_input_item_id

    def on_transcription_completed(
        self,
        conn_id: str,
        event: TranscriptionCompletedEvent,
    ) -> list[ConversationItemInputAudioTranscriptionCompletedEvent]:
        """Terminalize one transcript item and emit its authoritative final event."""
        st = self._state(conn_id)
        if st.input_terminal_was_deleted(event.turn_id, event.turn_revision):
            logger.debug(
                "Ignoring transcription completion for deleted turn=%s rev=%s",
                event.turn_id,
                event.turn_revision,
            )
            if event.turn_id is not None:
                item_id = self._completion_input_item_id(conn_id, event.turn_id, event.turn_revision)
                if item_id is not None:
                    self.terminalize_input_item(conn_id, item_id)
            return []
        item_id = self._completion_input_item_id(conn_id, event.turn_id, event.turn_revision)
        if item_id is None:
            # Preserve the pre-routing fallback for protocol-neutral pipelines
            # that do not publish speech lifecycle metadata. #485 tracks a
            # stricter standalone/ambiguous-terminal policy.
            item_id = self._service.response._current_item_id(conn_id)
            duration_s = st.input_audio_duration_s
        else:
            terminal = self.terminalize_input_item(conn_id, item_id)
            if terminal is None:
                return []
            item_id, duration_s = terminal
        st.response_usage.audio_duration_s += duration_s
        return [
            ConversationItemInputAudioTranscriptionCompletedEvent(
                type="conversation.item.input_audio_transcription.completed",
                event_id=self._next_event_id(),
                content_index=0,
                item_id=item_id,
                transcript=event.transcript,
                usage=UsageTranscriptTextUsageDuration(
                    seconds=duration_s,
                    type="duration",
                ),
            )
        ]

    def on_transcription_failed(
        self,
        conn_id: str,
        event: TranscriptionFailedEvent,
    ) -> list[ConversationItemInputAudioTranscriptionFailedEvent]:
        """Terminalize one transcript item and emit its item-scoped failure."""
        st = self._state(conn_id)
        if st.input_terminal_was_deleted(event.turn_id, event.turn_revision):
            logger.debug(
                "Ignoring transcription failure for deleted turn=%s rev=%s",
                event.turn_id,
                event.turn_revision,
            )
            if event.turn_id is not None:
                item_id = self._completion_input_item_id(conn_id, event.turn_id, event.turn_revision)
                if item_id is not None:
                    self.terminalize_input_item(conn_id, item_id)
            return []
        if event.turn_id is not None:
            item_id = st.input_item_by_turn_revision.get((event.turn_id, event.turn_revision))
        else:
            item_id = st.current_input_item_id
        if item_id is None:
            logger.debug(
                "Ignoring transcription failure for unknown turn=%s rev=%s",
                event.turn_id,
                event.turn_revision,
            )
            return []
        if self.terminalize_input_item(conn_id, item_id) is None:
            return []
        return [
            ConversationItemInputAudioTranscriptionFailedEvent(
                type="conversation.item.input_audio_transcription.failed",
                event_id=self._next_event_id(),
                content_index=0,
                item_id=item_id,
                error=InputAudioTranscriptionError(
                    type="transcription_error",
                    code="transcription_failed",
                    message=event.message,
                    param=None,
                ),
            )
        ]
