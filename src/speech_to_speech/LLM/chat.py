from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Union

from openai.types.realtime import ConversationItem
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
    RealtimeConversationItemSystemMessage,
    RealtimeConversationItemUserMessage,
)
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.realtime.realtime_conversation_item_system_message import Content as SystemContent
from openai.types.realtime.realtime_conversation_item_user_message import Content as UserContent
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.responses.response_input_image_param import ResponseInputImageParam
from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputMessageContentListParam,
)
from openai.types.responses.response_input_param import (
    FunctionCallOutput,
    ResponseFunctionToolCallParam,
    ResponseInputItemParam,
    ResponseInputParam,
    ResponseOutputMessageParam,
)
from openai.types.responses.response_input_param import (
    Message as ResponseMessage,
)
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_output_text_param import ResponseOutputTextParam
from pydantic import BaseModel

from speech_to_speech.utils.utils import _generate_id

logger = logging.getLogger(__name__)


class ChatItemError(Exception):
    """Raised when a conversation item fails validation in :meth:`Chat.add_item`."""


class CompactionResult(BaseModel):
    """Output of a :data:`CompactFn` summarization run."""

    user_summary: str
    assistant_summary: str


def _ensure_id(value: str | None, prefix: str) -> str:
    if value is None:
        return _generate_id(prefix)
    if not value.startswith(f"{prefix}_"):
        raise ChatItemError(f"ID must start with '{prefix}_', got {value!r}")
    return value


SupportedItem = Union[
    RealtimeConversationItemSystemMessage,
    RealtimeConversationItemUserMessage,
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
]


@dataclass
class _CompactionNode:
    """Reversible provenance for one summary pair.

    Nodes nest instead of flattening older summaries. Once all protocol-visible
    user IDs below a child retire, that child can collapse back to its two-item
    summary and release the original content it no longer needs to delete.
    """

    user_summary: RealtimeConversationItemUserMessage
    assistant_summary: RealtimeConversationItemAssistantMessage
    originals: list[SupportedItem | "_CompactionNode"]
    deletable_user_ids: set[str]


CompactFn = Callable[[ResponseInputParam], CompactionResult]


class Chat:
    """Manages conversation history with bounded size to avoid OOM issues.

    The buffer stores ``ConversationItem`` objects (user messages, assistant
    messages, function calls, function call outputs).  System messages are
    stored separately in ``init_chat_message`` and never placed in the buffer.

    History bounding is decided per ``add_item`` call via the ``compactor``
    argument:

    - ``compactor=None``: when the user-turn count exceeds ``size`` the oldest
      complete turn is evicted in place. Synchronous, lossy, no LLM involvement.
    - ``compactor=<fn>``: when ``size`` is exceeded, ``fn`` is invoked in a
      background thread to summarize older turns into a single user/assistant
      pair (with pending function calls preserved). Single-flight: while a
      compaction is running, additional triggers are silently bypassed.
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self.init_chat_message: RealtimeConversationItemSystemMessage | None = None
        # ``size`` is the number of user turns to keep.  When exceeded the
        # oldest complete turn (everything up to the next user message)
        # is evicted -- or, with a compactor, summarized in the background.
        self.buffer: list[SupportedItem] = []
        self._pending_tool_calls: dict[str, RealtimeConversationItemFunctionCall] = {}
        self._user_turn_count: int = 0
        self._deletable_user_ids: set[str] = set()
        self._compaction_nodes: dict[str, _CompactionNode] = {}

        # All state mutations and serializations go through _lock. Public methods
        # acquire it once; internal callers that already hold it use the
        # ``_locked`` helpers, so no reentry is needed (regular Lock is safe).
        self._lock = threading.Lock()
        self._compact_in_flight: bool = False
        self._compact_thread: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._gen_counter = 0
        self._compaction_suspended = False
        self._private_content_logging = False

    # ── Internal mutators (caller holds _lock) ─────────────────

    def _evict_oldest_turn(self) -> None:
        """Remove items from the front until the next user message boundary."""
        if not self.buffer:
            return
        removed: list[SupportedItem] = []
        first = self.buffer.pop(0)
        removed.append(first)
        if isinstance(first, RealtimeConversationItemUserMessage):
            self._user_turn_count -= 1
        while self.buffer and not isinstance(self.buffer[0], RealtimeConversationItemUserMessage):
            removed.append(self.buffer.pop(0))
        for item in removed:
            if item.id is not None:
                self._compaction_nodes.pop(item.id, None)

    def _has_call_id_in_buffer(self, call_id: str) -> bool:
        for entry in self.buffer:
            if isinstance(entry, RealtimeConversationItemFunctionCall) and entry.call_id == call_id:
                return True
        return False

    def _mark_call_completed(
        self, call_id: str, status: Literal["completed", "incomplete", "in_progress"] | None = None
    ) -> None:
        """Set ``status`` to ``"completed"`` on the matching function_call."""
        for entry in self.buffer:
            if isinstance(entry, RealtimeConversationItemFunctionCall) and entry.call_id == call_id:
                entry.status = "completed" if status is None else status
                return

    def append_tool_output(self, call_id: str, output_item: RealtimeConversationItemFunctionCallOutput) -> None:
        """Append a ``function_call_output``, re-injecting its ``function_call`` if evicted.

        Also marks the paired ``function_call`` as ``"completed"`` if its
        status was ``None``.

        Raises :class:`ChatItemError` if *call_id* is unknown.
        """
        with self._lock:
            self._append_tool_output_locked(call_id, output_item)

    def _append_tool_output_locked(self, call_id: str, output_item: RealtimeConversationItemFunctionCallOutput) -> None:
        """Body of :meth:`append_tool_output`. Caller must hold ``_lock``."""
        if self._has_call_id_in_buffer(call_id):
            self._pending_tool_calls.pop(call_id, None)
            self._mark_call_completed(call_id, output_item.status)
            self.buffer.append(output_item)
            return

        if call_id in self._pending_tool_calls:
            if self._private_content_logging:
                logger.info("Re-injecting private evicted function_call; content redacted")
            else:
                logger.info("Re-injecting evicted function_call for call_id=%s", call_id)
            fc = self._pending_tool_calls.pop(call_id)
            fc.status = "completed" if output_item.status is None else output_item.status
            self.buffer.append(fc)
            self.buffer.append(output_item)
            return

        raise ChatItemError(f"No function_call with call_id '{call_id}' found in conversation history.")

    def init_chat(self, message: RealtimeConversationItemSystemMessage) -> None:
        with self._lock:
            self.init_chat_message = message

    def add_item(self, item: SupportedItem) -> SupportedItem:
        """Validate and route a conversation item into the chat buffer.

        Does not enforce the soft size limit — call :meth:`trim_if_needed`
        explicitly after each successful generation to evict or compact old
        turns. A hard upper bound at ``2 * size`` is enforced inline as a
        runaway-client safety net: if the user-turn count exceeds it, the
        oldest complete turn is evicted (lossy, no compaction).

        Raises :class:`ChatItemError` if the item fails validation.
        """
        with self._lock:
            return self._add_item_locked(item)

    def add_items_atomically(self, items: list[SupportedItem]) -> None:
        """Add a client-supplied batch completely or restore the prior chat state."""
        with self._lock:
            buffer_before = list(self.buffer)
            pending_before = dict(self._pending_tool_calls)
            compaction_nodes_before = dict(self._compaction_nodes)
            turns_before = self._user_turn_count
            init_before = self.init_chat_message
            function_calls = {
                id(item): item
                for item in (*buffer_before, *pending_before.values())
                if isinstance(item, RealtimeConversationItemFunctionCall)
            }
            statuses_before = [(item, item.status) for item in function_calls.values()]
            try:
                for item in items:
                    self._add_item_locked(item)
            except ChatItemError:
                self.buffer = buffer_before
                self._pending_tool_calls = pending_before
                self._compaction_nodes = compaction_nodes_before
                self._user_turn_count = turns_before
                self.init_chat_message = init_before
                for function_call, status in statuses_before:
                    function_call.status = status
                raise

    def _add_item_locked(self, item: SupportedItem) -> SupportedItem:
        """Body of :meth:`add_item`; caller must hold ``_lock``."""
        if isinstance(item, RealtimeConversationItemSystemMessage):
            item.id = _ensure_id(item.id, "sys")
            self.init_chat_message = item
            logger.debug("Set system message via conversation item")

        elif isinstance(item, RealtimeConversationItemUserMessage):
            item.id = _ensure_id(item.id, "msg")
            item.content = [
                p
                for p in item.content
                if (p.type == "input_text" and p.text) or (p.type == "input_image" and p.image_url)
            ]
            if not item.content:
                raise ChatItemError("Message has no supported content. Supported modalities: input_text, input_image.")
            self.buffer.append(item)
            self._user_turn_count += 1
            logger.debug("Added user message to chat (%d parts)", len(item.content))

        elif isinstance(item, RealtimeConversationItemAssistantMessage):
            item.id = _ensure_id(item.id, "msg")
            item.content = [p for p in item.content if p.type == "output_text" and p.text]
            if not item.content:
                return item
            self.buffer.append(item)
            logger.debug("Added assistant message to chat (%d parts)", len(item.content))

        elif isinstance(item, RealtimeConversationItemFunctionCall):
            item.id = _ensure_id(item.id, "fc")
            item.call_id = _ensure_id(item.call_id, "call")
            self._pending_tool_calls[item.call_id] = item
            if self._private_content_logging:
                logger.debug("Added private function_call; content redacted")
            else:
                logger.debug("Added function_call to chat (call_id=%s)", item.call_id)

        elif isinstance(item, RealtimeConversationItemFunctionCallOutput):
            item.id = _ensure_id(item.id, "fco")
            self._append_tool_output_locked(item.call_id, item)
            if self._private_content_logging:
                logger.debug("Added private function_call_output; content redacted")
            else:
                logger.debug("Added function_call_output to chat (call_id=%s)", item.call_id)

        else:
            raise ChatItemError(f"Unsupported item type: {getattr(item, 'type', None)}")

        if self.size > 0 and self._user_turn_count > 2 * self.size:
            logger.warning(
                "Chat buffer exceeded hard cap (%d > 2 * size=%d); evicting oldest turn",
                self._user_turn_count,
                self.size,
            )
            while self._user_turn_count > 2 * self.size:
                self._evict_oldest_turn()

        return item

    def trim_if_needed(self, compactor: CompactFn | None = None) -> None:
        """Enforce the size limit after a generation completes. Fires when
        ``user_turn_count > size``.

        - ``compactor=None``: synchronous eviction of the oldest complete turn.
        - ``compactor=<fn>``: launch a background compaction (single-flight).

        Call once after each successful generation, not inside :meth:`add_item`.
        """
        with self._lock:
            if self._compaction_suspended:
                return
            if self._user_turn_count <= self.size:
                return
            if compactor is not None:
                self._maybe_trigger_compaction(compactor)
            else:
                while self._user_turn_count > self.size:
                    self._evict_oldest_turn()

    def replace_user_message_text(self, item_id: str, text: str) -> bool:
        """Replace the text content of an existing user message.

        Used by speculative turn revisions: the conversation turn remains the
        same, but the STT transcript is superseded by a transcription of a
        longer raw-audio buffer.
        """

        with self._lock:
            for item in self.buffer:
                if not isinstance(item, RealtimeConversationItemUserMessage) or item.id != item_id:
                    continue
                item.content = [UserContent(type="input_text", text=text)]
                if self._private_content_logging:
                    logger.debug("Replaced private speculative user message; content redacted")
                else:
                    logger.debug("Replaced speculative user message %s", item_id)
                return True
        return False

    def mark_user_message_deletable(self, item_id: str) -> None:
        """Preserve enough compaction provenance to delete a protocol item later."""

        with self._lock:
            self._deletable_user_ids.add(item_id)

    def retire_user_message_deletable(self, item_id: str) -> None:
        """Release provenance after a user item leaves the bounded protocol index."""

        with self._lock:
            self._deletable_user_ids.discard(item_id)
            for summary_id, node in list(self._compaction_nodes.items()):
                if item_id not in node.deletable_user_ids:
                    continue
                self._retire_from_compaction_node(node, item_id)
                if not node.deletable_user_ids:
                    self._compaction_nodes.pop(summary_id, None)

    def _retire_from_compaction_node(self, node: _CompactionNode, item_id: str) -> None:
        """Retire one ID and collapse child provenance that no longer serves deletes."""

        node.deletable_user_ids.discard(item_id)
        collapsed: list[SupportedItem | _CompactionNode] = []
        for original in node.originals:
            if not isinstance(original, _CompactionNode):
                collapsed.append(original)
                continue
            if item_id in original.deletable_user_ids:
                self._retire_from_compaction_node(original, item_id)
            if original.deletable_user_ids:
                collapsed.append(original)
            else:
                collapsed.extend((original.user_summary, original.assistant_summary))
        node.originals = collapsed

    def _restore_compaction_node_without(
        self,
        node: _CompactionNode,
        item_id: str,
    ) -> tuple[list[SupportedItem], list[_CompactionNode]]:
        """Expand only the branch containing *item_id* and preserve all siblings."""

        restored: list[SupportedItem] = []
        surviving_nodes: list[_CompactionNode] = []
        for original in node.originals:
            if isinstance(original, _CompactionNode):
                if item_id in original.deletable_user_ids:
                    nested_items, nested_nodes = self._restore_compaction_node_without(original, item_id)
                    restored.extend(nested_items)
                    surviving_nodes.extend(nested_nodes)
                else:
                    restored.extend((original.user_summary, original.assistant_summary))
                    surviving_nodes.append(original)
                continue
            if isinstance(original, RealtimeConversationItemUserMessage) and original.id == item_id:
                continue
            restored.append(original)
        return restored, surviving_nodes

    def remove_user_message(self, item_id: str) -> bool:
        """Remove an exact user message, restoring a compacted snapshot if needed."""

        with self._lock:
            for index, item in enumerate(self.buffer):
                if not isinstance(item, RealtimeConversationItemUserMessage) or item.id != item_id:
                    continue
                del self.buffer[index]
                self._user_turn_count -= 1
                self._deletable_user_ids.discard(item_id)
                self._compaction_nodes.pop(item_id, None)
                # A summary already running from an older snapshot must not
                # splice deleted user content back into live history.
                self._gen_counter += 1
                self._compact_in_flight = False
                if self._private_content_logging:
                    logger.debug("Removed private speculative user message; content redacted")
                else:
                    logger.debug("Removed speculative user message %s", item_id)
                return True
            for summary_id, node in list(self._compaction_nodes.items()):
                if item_id not in node.deletable_user_ids:
                    continue
                summary_ids = {node.user_summary.id, node.assistant_summary.id}
                indexes = [index for index, item in enumerate(self.buffer) if item.id in summary_ids]
                if not indexes:
                    self._compaction_nodes.pop(summary_id, None)
                    continue
                insert_at = min(indexes)
                restored, surviving_nodes = self._restore_compaction_node_without(node, item_id)
                self.buffer = [item for item in self.buffer if item.id not in summary_ids]
                self.buffer[insert_at:insert_at] = restored
                self._compaction_nodes.pop(summary_id, None)
                for surviving in surviving_nodes:
                    assert surviving.user_summary.id is not None
                    self._compaction_nodes[surviving.user_summary.id] = surviving
                self._deletable_user_ids.discard(item_id)
                self._user_turn_count = sum(
                    1 for item in self.buffer if isinstance(item, RealtimeConversationItemUserMessage)
                )
                self._gen_counter += 1
                self._compact_in_flight = False
                if self._private_content_logging:
                    logger.debug("Removed compacted private user message; content redacted")
                else:
                    logger.debug("Removed compacted user message %s", item_id)
                return True
            # Lossy eviction may already have removed the item from model context.
            # It remains protocol-visible until its bounded index entry retires, so
            # acknowledge the exact deletion while releasing its retained marker.
            if item_id in self._deletable_user_ids:
                self._deletable_user_ids.remove(item_id)
                return True
        return False

    def to_responses_api_chat(self, items: list[SupportedItem] | None = None) -> ResponseInputParam:
        """Serialize the chat (system prompt + buffer) for the OpenAI Responses API.

        If *items* is provided, serialize that slice instead of the live buffer
        (used by the compaction snapshot).
        """
        with self._lock:
            return self._to_responses_api_chat_locked(items if items is not None else self.buffer)

    def _to_responses_api_chat_locked(self, items: list[SupportedItem]) -> ResponseInputParam:
        """Body of :meth:`to_responses_api_chat`. Caller must hold ``_lock``."""
        buffer_items = list(items)
        result: list[ResponseInputItemParam] = []
        if self.init_chat_message:
            result.append(
                ResponseMessage(
                    content=[
                        ResponseInputTextParam(text=p.text or "A helpful AI assistant.", type="input_text")
                        for p in self.init_chat_message.content
                    ],
                    role="system",
                    type="message",
                )
            )
        for item in buffer_items:
            assert item.id is not None and item.id != "", f"item.id is {item.id}"
            if isinstance(item, RealtimeConversationItemUserMessage):
                content: ResponseInputMessageContentListParam = []
                for user_part in item.content:
                    if user_part.type == "input_text" and user_part.text is not None:
                        content.append(ResponseInputTextParam(text=user_part.text or "", type="input_text"))
                    elif user_part.type == "input_image" and user_part.image_url is not None:
                        img = ResponseInputImageParam(type="input_image", detail=user_part.detail or "auto")
                        if user_part.image_url is not None:
                            img["image_url"] = user_part.image_url
                        content.append(img)
                if content:
                    result.append(ResponseMessage(content=content, role="user", type="message"))
            elif isinstance(item, RealtimeConversationItemAssistantMessage):
                assistant_content: list[ResponseOutputTextParam] = []
                for assistant_part in item.content:
                    if assistant_part.type == "output_text" and assistant_part.text is not None:
                        assistant_content.append(
                            ResponseOutputTextParam(text=assistant_part.text, type="output_text", annotations=[])
                        )
                if assistant_content:
                    result.append(
                        ResponseOutputMessageParam(
                            id=item.id,
                            content=assistant_content,
                            role="assistant",
                            status=item.status or "completed",
                            type="message",
                        )
                    )
            elif isinstance(item, RealtimeConversationItemFunctionCall) and item.call_id is not None:
                assert item.call_id is not None and item.call_id != ""
                function_call = ResponseFunctionToolCallParam(
                    arguments=item.arguments,
                    call_id=item.call_id,
                    name=item.name,
                    type="function_call",
                    id=item.id,
                )
                if item.id is not None:
                    function_call["id"] = item.id
                if item.status is not None:
                    function_call["status"] = item.status
                result.append(function_call)
            elif isinstance(item, RealtimeConversationItemFunctionCallOutput):
                function_call_output = FunctionCallOutput(
                    call_id=item.call_id,
                    output=item.output,
                    type="function_call_output",
                )
                if item.id is not None:
                    function_call_output["id"] = item.id
                if item.status is not None:
                    function_call_output["status"] = item.status
                result.append(function_call_output)
        return result

    def to_transformers_chat(self) -> list[dict[str, Any]]:
        """Serialize the full chat for HuggingFace transformers ``apply_chat_template``.

        User messages with only text produce a plain string ``content`` value.
        User messages containing images keep ``content`` as a list of dicts so
        VLM pipelines can process them.
        """
        with self._lock:
            messages: list[TransformersChatMessage] = []
            if self.init_chat_message:
                text = " ".join(p.text for p in self.init_chat_message.content if p.text)
                messages.append(TransformersSystemMessage(content=text))
            for item in self.buffer:
                if isinstance(item, RealtimeConversationItemUserMessage):
                    has_images = any(p.type == "input_image" for p in item.content)
                    if has_images:
                        messages.append(
                            TransformersUserMessage(content=[p.model_dump(exclude_none=True) for p in item.content])
                        )
                    else:
                        text = " ".join(p.text for p in item.content if p.type == "input_text" and p.text)
                        messages.append(TransformersUserMessage(content=text))
                elif isinstance(item, RealtimeConversationItemAssistantMessage):
                    text = " ".join(p.text for p in item.content if p.text)
                    messages.append(TransformersAssistantMessage(content=text))
                elif isinstance(item, RealtimeConversationItemFunctionCall):
                    assert item.call_id is not None and item.call_id != ""
                    args: Any = item.arguments
                    try:
                        args = json.loads(args) if isinstance(args, str) else args
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    messages.append(
                        TransformersFunctionCallMessage(
                            tool_calls=[
                                TransformersToolCall(
                                    id=item.call_id,
                                    function=TransformersToolCallFunction(name=item.name, arguments=args),
                                )
                            ]
                        )
                    )
                elif isinstance(item, RealtimeConversationItemFunctionCallOutput):
                    name = ""
                    for prev in reversed(messages):
                        if isinstance(prev, TransformersFunctionCallMessage):
                            for tc in prev.tool_calls:
                                if tc.id == item.call_id:
                                    name = tc.function.name
                                    break
                            if name:
                                break
                    messages.append(
                        TransformersToolMessage(
                            tool_call_id=item.call_id,
                            name=name,
                            content=item.output,
                        )
                    )
            return [m.model_dump() for m in messages]

    def copy(self) -> Chat:
        """Return a shallow snapshot safe for concurrent read access."""
        with self._lock:
            clone = Chat(self.size)
            clone.init_chat_message = self.init_chat_message
            clone.buffer = list(self.buffer)
            clone._pending_tool_calls = dict(self._pending_tool_calls)
            clone._user_turn_count = self._user_turn_count
            clone._deletable_user_ids = set(self._deletable_user_ids)
            clone._compaction_nodes = dict(self._compaction_nodes)
            clone._compaction_suspended = self._compaction_suspended
            clone._private_content_logging = self._private_content_logging
            return clone

    def reset(
        self,
        *,
        private_content_logging: bool = False,
        suspend_compaction: bool = False,
    ) -> None:
        """Clear conversation state and atomically select post-reset privacy."""
        with self._lock:
            self._gen_counter += 1
            self._compact_in_flight = False
            self._compaction_suspended = suspend_compaction
            self._private_content_logging = private_content_logging
            self.buffer = []
            self.init_chat_message = None
            self._pending_tool_calls = {}
            self._user_turn_count = 0
            self._deletable_user_ids = set()
            self._compaction_nodes = {}

    def close(self) -> None:
        """Permanently shut down the chat. In-flight compaction splice is suppressed.

        The compaction worker (a daemon thread) is not joined: it may be blocked
        in an LLM call. Process exit reaps it.
        """
        self._shutdown.set()
        with self._lock:
            self._gen_counter += 1
            self._compact_in_flight = False

    def suppress_inflight_compaction(self) -> None:
        """Prevent an older background summary from splicing into current state."""
        with self._lock:
            self._gen_counter += 1
            self._compact_in_flight = False

    def suspend_compaction(self) -> None:
        """Freeze current and future compaction until an explicit resume."""
        with self._lock:
            self._gen_counter += 1
            self._compact_in_flight = False
            self._compaction_suspended = True

    def resume_compaction(self) -> None:
        """Allow later size enforcement after a private input is resolved."""
        with self._lock:
            if not self._shutdown.is_set():
                self._compaction_suspended = False

    def enable_private_content_logging(self) -> None:
        """Keep exception logs content-free for the lifetime of this chat."""
        with self._lock:
            self._private_content_logging = True

    def image_message_ids(self) -> set[str]:
        """IDs of user messages currently carrying ``input_image`` content."""
        with self._lock:
            return {
                item.id
                for item in self.buffer
                if isinstance(item, RealtimeConversationItemUserMessage)
                and item.id is not None
                and any(p.type == "input_image" for p in item.content)
            }

    def strip_images(self, only_ids: set[str] | None = None) -> None:
        """Remove image content parts from user messages in the buffer.

        Called after appending the assistant response so images don't persist
        across turns. With *only_ids*, strip only those message IDs — the images
        the just-completed response actually consumed (captured before the
        request was sent). This leaves intact an image a fast client injected
        mid-generation for the *next* turn, which the current response never saw.
        Without *only_ids*, every image is stripped.
        """
        with self._lock:
            for item in self.buffer:
                if isinstance(item, RealtimeConversationItemUserMessage):
                    if only_ids is not None and item.id not in only_ids:
                        continue
                    item.content = [p for p in item.content if p.type != "input_image"]

    # ── Compaction internals ──────────────────────────────────

    def _snapshot_for_compaction(
        self,
    ) -> tuple[ResponseInputParam, set[str], int]:
        """Compute the snapshot of items eligible for compaction.

        Caller must hold ``_lock``. Returns
        ``(serialized_snapshot, marker_ids, n_turns)``. ``marker_ids``
        identifies the buffer items that may be removed when the splice runs.
        Always leaves the most recent user turn untouched (it may be in-flight).
        Returns an empty result if there are fewer than 2 compactable turns.
        """
        n_turns = max(0, self._user_turn_count - 1)
        if n_turns < 2:
            return [], set(), n_turns

        # Slice up to (but not including) the (n_turns + 1)-th user message.
        user_seen = 0
        end_idx = len(self.buffer)
        for i, entry in enumerate(self.buffer):
            if isinstance(entry, RealtimeConversationItemUserMessage):
                user_seen += 1
                if user_seen == n_turns + 1:
                    end_idx = i
                    break

        items_to_compact = self.buffer[:end_idx]
        marker_ids = {entry.id for entry in items_to_compact if entry.id is not None}
        snapshot = self._to_responses_api_chat_locked(items=items_to_compact)
        # Strip image parts so the summarizer doesn't have to handle them.
        for raw in snapshot:
            if not isinstance(raw, dict) or raw.get("role") != "user":
                continue
            msg: dict[str, Any] = raw  # type: ignore[assignment]
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = [c for c in content if not (isinstance(c, dict) and c.get("type") == "input_image")]
        return snapshot, marker_ids, n_turns

    def _maybe_trigger_compaction(self, compactor: CompactFn) -> None:
        """Start a background compaction worker. Bypass silently if one is running.

        Caller must hold ``_lock``.
        """
        if self._shutdown.is_set() or self._compact_in_flight:
            return
        snapshot, marker_ids, n_turns = self._snapshot_for_compaction()
        if n_turns < 2 or not marker_ids:
            return
        gen = self._gen_counter
        self._compact_in_flight = True
        thread = threading.Thread(
            target=self._compact_worker,
            args=(compactor, snapshot, marker_ids, gen),
            daemon=True,
            name="chat-compact",
        )
        self._compact_thread = thread
        logger.info(
            "Chat compaction triggered: compacting %d turn(s) (%d item(s)), buffer size=%d",
            n_turns,
            len(marker_ids),
            len(self.buffer),
        )
        thread.start()

    def _compact_worker(
        self,
        compactor: CompactFn,
        snapshot: ResponseInputParam,
        marker_ids: set[str],
        gen: int,
    ) -> None:
        """Worker thread entry point."""
        try:
            if self._shutdown.is_set() or self._gen_counter != gen:
                return
            try:
                result = compactor(snapshot)
            except Exception:
                with self._lock:
                    if self._private_content_logging:
                        logger.error("Chat compaction failed; private content redacted")
                    else:
                        logger.exception("Chat compaction failed; chat unchanged")
                return
            if not isinstance(result, CompactionResult):
                logger.error("Compactor must return a CompactionResult, got %r", type(result).__name__)
                return
            if self._shutdown.is_set() or self._gen_counter != gen:
                return
            self._apply_compaction(result, marker_ids, gen)
        finally:
            # Don't clobber the flag if reset/close has advanced the gen.
            with self._lock:
                if self._gen_counter == gen:
                    self._compact_in_flight = False

    def _apply_compaction(
        self,
        result: CompactionResult,
        marker_ids: set[str],
        gen: int,
    ) -> None:
        """Splice the summary in front of items not consumed by compaction.

        FC/FCO pairing is left entirely to :meth:`add_item` / :meth:`append_tool_output`.
        Compaction only drops items; it never inserts an FC into the buffer.
        Pending FCs (no FCO yet) stay in ``_pending_tool_calls`` and will be
        appended adjacent to their FCO when it arrives.
        """
        with self._lock:
            if self._shutdown.is_set() or self._gen_counter != gen:
                return
            # Keep FC if its FCO is outside the compacted range -- otherwise
            # the FCO in `remaining` would be orphaned.
            fco_call_ids_in_range = {
                x.call_id
                for x in self.buffer
                if isinstance(x, RealtimeConversationItemFunctionCallOutput) and x.id in marker_ids
            }
            fc_ids_to_keep = {
                x.id
                for x in self.buffer
                if x.id in marker_ids
                and isinstance(x, RealtimeConversationItemFunctionCall)
                and x.call_id not in fco_call_ids_in_range
            }
            drop_ids = marker_ids - fc_ids_to_keep
            remaining = [x for x in self.buffer if x.id not in drop_ids]

            originals: list[SupportedItem | _CompactionNode] = []
            consumed_summary_ids: set[str] = set()
            for item in self.buffer:
                if item.id not in drop_ids or item.id in consumed_summary_ids:
                    continue
                nested = self._compaction_nodes.pop(item.id or "", None)
                if nested is not None:
                    originals.append(nested)
                    if nested.assistant_summary.id is not None:
                        consumed_summary_ids.add(nested.assistant_summary.id)
                else:
                    originals.append(item)

            user_msg = make_user_message(result.user_summary)
            user_msg.id = _generate_id("msg")
            asst_msg = make_assistant_message(result.assistant_summary)
            asst_msg.id = _generate_id("msg")

            deletable_user_ids = {
                original.id
                for original in originals
                if isinstance(original, RealtimeConversationItemUserMessage)
                and original.id is not None
                and original.id in self._deletable_user_ids
            }
            for original in originals:
                if isinstance(original, _CompactionNode):
                    deletable_user_ids.update(original.deletable_user_ids)
            if deletable_user_ids:
                self._compaction_nodes[user_msg.id] = _CompactionNode(
                    user_summary=user_msg,
                    assistant_summary=asst_msg,
                    originals=originals,
                    deletable_user_ids=deletable_user_ids,
                )

            self.buffer = [user_msg, asst_msg, *remaining]
            self._user_turn_count = sum(1 for x in self.buffer if isinstance(x, RealtimeConversationItemUserMessage))
            logger.info(
                "Chat compaction applied: buffer now %d item(s), %d user turn(s)",
                len(self.buffer),
                self._user_turn_count,
            )


# ---------------------------------------------------------------------------
# Transformers chat message models
# ---------------------------------------------------------------------------


class TransformersToolCallFunction(BaseModel):
    name: str
    arguments: dict[str, Any]


class TransformersToolCall(BaseModel):
    type: Literal["function"] = "function"
    id: str
    function: TransformersToolCallFunction


class TransformersSystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str


class TransformersUserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[dict[str, Any]]


class TransformersAssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class TransformersFunctionCallMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    tool_calls: list[TransformersToolCall]


class TransformersToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    name: str
    content: str


TransformersChatMessage = Union[
    TransformersSystemMessage,
    TransformersUserMessage,
    TransformersAssistantMessage,
    TransformersFunctionCallMessage,
    TransformersToolMessage,
]


# ---------------------------------------------------------------------------
# Factory helpers -- hide verbose constructors behind simple calls
# ---------------------------------------------------------------------------


def make_user_message(text: str) -> RealtimeConversationItemUserMessage:
    return RealtimeConversationItemUserMessage(
        type="message",
        role="user",
        content=[UserContent(type="input_text", text=text)],
    )


def make_assistant_message(text: str) -> RealtimeConversationItemAssistantMessage:
    return RealtimeConversationItemAssistantMessage(
        type="message",
        role="assistant",
        content=[AssistantContent(type="output_text", text=text)],
    )


def make_system_message(text: str) -> RealtimeConversationItemSystemMessage:
    return RealtimeConversationItemSystemMessage(
        type="message",
        role="system",
        content=[SystemContent(type="input_text", text=text)],
    )


def _require_supported_item(item: ConversationItem) -> SupportedItem:
    """Narrow one protocol item without mutating chat state."""
    # call_id on function_call items must be client-supplied: it is referenced later by
    # function_call_output items, so we cannot silently generate one here.
    if isinstance(item, RealtimeConversationItemFunctionCall) and (
        item.call_id is None or not item.call_id.startswith("call_")
    ):
        raise ChatItemError("function_call item is missing a call_id. The call_id should start with 'call_'.")

    if isinstance(
        item,
        (
            RealtimeConversationItemSystemMessage,
            RealtimeConversationItemUserMessage,
            RealtimeConversationItemAssistantMessage,
            RealtimeConversationItemFunctionCall,
            RealtimeConversationItemFunctionCallOutput,
        ),
    ):
        return item

    raise ChatItemError(f"Unsupported item type: {getattr(item, 'type', None)}")


def add_supported_item(chat: Chat, item: ConversationItem) -> None:
    """Narrow a protocol ``ConversationItem`` and add it to *chat*."""
    chat.add_item(_require_supported_item(item))


def add_supported_items_atomically(chat: Chat, items: list[ConversationItem]) -> None:
    """Validate and add a protocol-item batch without retaining a rejected prefix."""
    chat.add_items_atomically([_require_supported_item(item) for item in items])


def build_active_chat(original_chat: Chat, response: RealtimeResponseCreateParams | None) -> Chat:
    """Build the chat an *out-of-band* response generates against (caller ensures out-of-band).

    Mirrors the OpenAI realtime semantics for ``input``:

    - ``input is None`` -> a read-only **copy of the default conversation** (the
      out-of-band response reads history but never commits back).
    - ``input == []`` -> a **fresh, empty chat** (context cleared; only the
      system prompt, added later by the handler, will be present).
    - ``input == [...]`` -> a **fresh chat seeded** with those items.

    Raises :class:`ChatItemError` if an ``input`` item fails validation.
    """
    if response is not None and response.input is not None:
        fresh = Chat(original_chat.size)
        for item in response.input:
            add_supported_item(fresh, item)
        return fresh
    return original_chat.copy()
