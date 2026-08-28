from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional

import httpx
from nltk import sent_tokenize
from openai import OpenAI
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
)
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel, ConfigDict, Field

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import (
    Chat,
    ChatItemError,
    SupportedItem,
    build_active_chat,
    make_system_message,
    make_user_message,
)
from speech_to_speech.LLM.compaction_prompt import CompactGenerateFn, build_compactor
from speech_to_speech.LLM.text_prompt import build_text_system_prompt
from speech_to_speech.LLM.tool_call.function_tool import MAX_TOOL_CALLS_PER_RESPONSE
from speech_to_speech.LLM.utils import remove_unspeechable, resolve_auto_language
from speech_to_speech.LLM.voice_prompt import build_voice_system_prompt
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut
from speech_to_speech.pipeline.messages import (
    EndOfResponse,
    LLMResponseChunk,
    TokenUsage,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.utils.utils import is_out_of_band, response_wants_audio

logger = logging.getLogger(__name__)

# About 18–24 seconds of default SDK backoff before warmup fails.
WARMUP_MAX_RETRIES = 6


# ── Normalised provider events ────────────────────────────────────────────────
# Each backend's stream/response is mapped to this small vocabulary so the shared
# speech-pipeline logic (sentence batching, cancellation, history, token usage)
# lives in one place. Subclasses differ only in how they produce these events.


class TextDelta(BaseModel):
    """Incremental assistant text. Always RAW (unfiltered); the base applies
    ``remove_unspeechable`` for the audio path."""

    text: str


class AssistantMessage(BaseModel):
    """A complete assistant turn to write back to history."""

    content: list[AssistantContent]


class ToolCall(BaseModel):
    """A complete function tool call (``call_id`` / ``id`` already regenerated)."""

    item: ResponseFunctionToolCall


class Usage(BaseModel):
    """Token accounting for the turn."""

    input_tokens: int
    output_tokens: int


ProviderEvent = TextDelta | AssistantMessage | ToolCall | Usage
PrivateContentGuard = bool | RuntimeConfig


@contextmanager
def private_content_redaction(guard: PrivateContentGuard) -> Iterator[bool]:
    """Read sticky privacy state atomically with a content-sensitive operation."""
    if isinstance(guard, RuntimeConfig):
        with guard.transcript_barrier_content_guard() as private_content:
            yield private_content
        return
    yield guard


class _Turn(BaseModel):
    """Per-request context threaded through generation (immutable for the turn)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    language_code: Optional[str]
    gen: int | None
    runtime_config: Any
    response: Any
    turn_id: str | None
    turn_revision: int | None
    response_user_item_id: str | None
    response_user_item_ids: set[str]
    speech_stopped_at_s: float | None
    wants_audio: bool


class _GenState(BaseModel):
    """Mutable accumulators collected while consuming a turn's events."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tools: list[ResponseFunctionToolCall] = Field(default_factory=list)
    pending: list[SupportedItem] = Field(default_factory=list)
    clean_text: str = ""  # filtered text, kept only for the debug log
    input_tokens: int = 0
    output_tokens: int = 0


class BaseOpenAICompatibleHandler(BaseHandler[LLMIn, LLMOut], ABC):
    """Shared lifecycle for OpenAI-compatible LLM backends (Responses & Chat
    Completions).

    Subclasses implement four hooks — :meth:`warmup`,
    :meth:`_build_compaction_generate_fn`, :meth:`_serialize`, :meth:`_request`,
    :meth:`_iter_events` and :meth:`_build_optional_kwargs` — and inherit the
    request/response orchestration: speculative-turn gating, cancellation,
    sentence batching, text-only vs audio handling, history write-back, token
    usage, out-of-band handling and error termination.
    """

    # ── setup ─────────────────────────────────────────────────────────────────

    def setup(
        self,
        model_name: str = "gpt-5.4-mini",
        device: str = "cuda",
        gen_kwargs: dict[str, Any] = {},
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        stream: bool = True,
        user_role: str = "user",
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        disable_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        request_timeout_s: float = 20.0,
        stream_batch_sentences: int = 3,
        enable_lang_prompt: bool = False,
        compact_history: bool = False,
        **_kwargs: Any,
    ) -> None:
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.model_name = model_name
        self.stream = stream
        self.stream_batch_sentences = max(1, stream_batch_sentences)
        self.enable_lang_prompt = enable_lang_prompt
        self.gen_kwargs = dict(gen_kwargs)
        self.request_timeout_s = float(request_timeout_s)
        self.request_timeout = httpx.Timeout(
            self.request_timeout_s,
            connect=min(10.0, self.request_timeout_s),
        )

        self.user_role = user_role
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._extra_body = self._build_extra_body(base_url, disable_thinking, reasoning_effort)
        self.compactor = build_compactor(self._build_compaction_generate_fn()) if compact_history else None
        self.warmup()

    @staticmethod
    def _is_official_openai(base_url: Optional[str]) -> bool:
        """Whether ``base_url`` points at the official OpenAI server.

        Normalises a trailing slash so ``https://api.openai.com/v1/`` is also
        recognised; the official server rejects the provider-specific extra_body
        keys we send to vLLM / the HF router.
        """
        if base_url is None:
            return False
        return base_url.rstrip("/") == "https://api.openai.com/v1"

    @classmethod
    def _build_extra_body(
        cls,
        base_url: Optional[str],
        disable_thinking: bool,
        reasoning_effort: Optional[str],
    ) -> Optional[dict[str, Any]]:
        """Build the provider-specific ``extra_body`` used to disable reasoning.

        Providers differ in how reasoning is turned off: vLLM/Qwen honour
        ``chat_template_kwargs.enable_thinking=false``, while others (e.g. GLM via
        the HF router) ignore that and require ``reasoning_effort='none'``. A
        non-empty ``reasoning_effort`` therefore takes precedence; otherwise we fall
        back to the chat-template flag. None of this applies to the official
        OpenAI server, which rejects unknown extra_body keys.
        """
        if base_url is None or cls._is_official_openai(base_url):
            return None
        if reasoning_effort:
            return {"reasoning_effort": reasoning_effort}
        if disable_thinking:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return None

    # ── subclass hooks ──────────────────────────────────────────────────────--

    @abstractmethod
    def warmup(self) -> None:
        """Issue a cheap request so the model/connection is ready before serving."""
        ...

    @abstractmethod
    def _build_compaction_generate_fn(self) -> CompactGenerateFn:
        """Return a ``(system, user) -> text`` fn used to compact long histories."""
        ...

    @abstractmethod
    def _serialize(self, active_chat: Chat) -> Any:
        """Serialise the chat to the backend's request payload (input/messages)."""
        ...

    @abstractmethod
    def _request(self, api_input: Any, optional_kwargs: dict[str, Any]) -> Any:
        """Issue the create() call and return the response or stream."""
        ...

    @abstractmethod
    def _iter_stream_events(
        self,
        api_response: Any,
        *,
        redact_private_content: PrivateContentGuard = False,
    ) -> Iterator[ProviderEvent]:
        """Map a streaming response to normalised :data:`ProviderEvent`s."""
        ...

    @abstractmethod
    def _iter_response_events(
        self,
        api_response: Any,
        *,
        redact_private_content: PrivateContentGuard = False,
    ) -> Iterator[ProviderEvent]:
        """Map a non-streaming response to normalised :data:`ProviderEvent`s."""
        ...

    def _iter_events(
        self,
        api_response: Any,
        *,
        redact_private_content: PrivateContentGuard = False,
    ) -> Iterator[ProviderEvent]:
        """Dispatch to the stream/non-stream mapper. ``self.stream`` is the single
        source of truth (it set the request's ``stream=`` flag), so the response
        type always matches it."""
        if self.stream:
            yield from self._iter_stream_events(
                api_response,
                redact_private_content=redact_private_content,
            )
        else:
            yield from self._iter_response_events(
                api_response,
                redact_private_content=redact_private_content,
            )

    def _request_for_turn(
        self,
        api_input: Any,
        optional_kwargs: dict[str, Any],
        turn: _Turn,
    ) -> tuple[Any, bool]:
        """Return the provider response and whether this exact turn streams."""
        return self._request(api_input, optional_kwargs), self.stream

    def _events_for_turn(
        self,
        api_response: Any,
        optional_kwargs: dict[str, Any],
        turn: _Turn,
        *,
        streaming: bool,
    ) -> Iterator[ProviderEvent]:
        """Map one response; subclasses may fully validate a private envelope."""
        del optional_kwargs
        if streaming:
            return self._iter_stream_events(api_response, redact_private_content=turn.runtime_config)
        return self._iter_response_events(api_response, redact_private_content=turn.runtime_config)

    @abstractmethod
    def _build_optional_kwargs(self, req_tools: Any, req_tool_choice: Any) -> dict[str, Any]:
        """Build the per-request tools/tool_choice kwargs in the backend's shape."""
        ...

    # ── speculative-turn / cancellation gating ─────────────────────────────────

    def _turn_is_latest(self, turn_id: str | None, turn_revision: int | None) -> bool:
        return self.speculative_turns is None or self.speculative_turns.is_latest(turn_id, turn_revision)

    def _generation_is_stale(self, gen: int | None) -> bool:
        return gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen)

    def _turn_output_allowed(self, turn_id: str | None, turn_revision: int | None) -> bool:
        if self.speculative_turns is None:
            return True
        return self.speculative_turns.is_latest_after_reopen_grace(turn_id, turn_revision)

    def _turn_owns_writeback_now(self, turn_id: str | None, turn_revision: int | None) -> bool | None:
        """Check writeback ownership without waiting; ``None`` asks the caller to retry."""
        if self.speculative_turns is None:
            return True
        return self.speculative_turns.try_is_latest_after_reopen_grace(turn_id, turn_revision)

    def _fail_home_assistant_guard_for_current_turn(
        self,
        runtime_config: RuntimeConfig,
        *,
        generation: int | None,
        turn_id: str | None,
        turn_revision: int | None,
    ) -> bool:
        """Latch a guarded turn failure only while that turn still owns output.

        Speculative reopen waits must happen outside the private-failure lock.
        The nonblocking recheck inside that lock either establishes one exact
        failure-before-cancel/reopen ordering or retries after releasing it.
        """
        while True:
            if self._generation_is_stale(generation) or not self._turn_output_allowed(turn_id, turn_revision):
                return False
            with runtime_config.transcript_barrier_state_guard():
                if self._generation_is_stale(generation):
                    return False
                if self.speculative_turns is not None:
                    owns_turn = self.speculative_turns.try_is_latest_after_reopen_grace(
                        turn_id,
                        turn_revision,
                    )
                    if owns_turn is None:
                        continue
                    if not owns_turn:
                        return False
                runtime_config.fail_home_assistant_guard()
                return True

    def _apply_config(
        self,
        chat: Chat,
        instructions: Optional[str],
        wants_audio: bool = True,
    ) -> None:
        if instructions:
            builder = build_voice_system_prompt if wants_audio else build_text_system_prompt
            full_instructions = builder(instructions)
            chat.add_item(make_system_message(full_instructions))

    # ── output helpers ──────────────────────────────────────────────────────--

    def _chunk(
        self,
        turn: _Turn,
        *,
        text: str = "",
        tools: list[ResponseFunctionToolCall] | None = None,
        language_code: Optional[str] = None,
    ) -> LLMResponseChunk:
        return LLMResponseChunk(
            text=text,
            language_code=language_code if language_code is not None else turn.language_code,
            tools=tools or [],
            runtime_config=turn.runtime_config,
            response=turn.response,
            turn_id=turn.turn_id,
            turn_revision=turn.turn_revision,
            speech_stopped_at_s=turn.speech_stopped_at_s,
            cancel_generation=turn.gen,
        )

    def _record_tool_call(self, state: _GenState, turn: _Turn, item: ResponseFunctionToolCall) -> Iterator[LLMOut]:
        """Emit a tool call, persisting it (and any assistant text seen so far)
        to history *before* it is forwarded to the client.

        The function_call must already exist in the conversation by the time the
        client returns its ``function_call_output``; otherwise a fast client
        races ahead of the deferred end-of-turn write-back and the output is
        rejected ("No function_call with call_id ... found"), which makes the
        model re-issue the same tool call. The call lands in ``_pending_tool_calls``
        (not serialized until its output pairs it), so eager recording is safe.

        Out-of-band turns never touch the default conversation, and a stale turn
        records nothing (it is not forwarded to the client either)."""
        if any(previous.name == item.name and previous.arguments == item.arguments for previous in state.tools):
            with private_content_redaction(turn.runtime_config) as private_content:
                if private_content:
                    logger.warning("Skipping duplicate private tool call; content redacted")
                else:
                    logger.warning("Skipping duplicate tool call '%s'", item.name)
            return
        if len(state.tools) >= MAX_TOOL_CALLS_PER_RESPONSE:
            with private_content_redaction(turn.runtime_config) as private_content:
                if private_content:
                    logger.warning("Skipping extra private tool call; content redacted")
                else:
                    logger.warning(
                        "Skipping extra tool call '%s'; at most %d tool calls are allowed per response",
                        item.name,
                        MAX_TOOL_CALLS_PER_RESPONSE,
                    )
            return
        state.tools.append(item)
        fc_item = RealtimeConversationItemFunctionCall(
            type="function_call",
            name=item.name,
            arguments=item.arguments,
            call_id=item.call_id,
            id=item.id,
            status=item.status,
        )
        if self._generation_is_stale(turn.gen) or not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
            logger.info("LLM generation cancelled (stale speculative turn)")
            return
        with turn.runtime_config.transcript_barrier_state_guard():
            if (
                turn.runtime_config.private_protocol_failed
                or self._generation_is_stale(turn.gen)
                or not self._turn_owns_writeback_now(turn.turn_id, turn.turn_revision)
            ):
                return
            if not is_out_of_band(turn.response):
                # Flush assistant text accumulated before this call first (so history
                # order matches what the client received), then persist the call —
                # all before the chunk leaves for the client.
                chat = turn.runtime_config.chat
                for pending_item in state.pending:
                    chat.add_response_item(
                        pending_item,
                        after_user_id=turn.response_user_item_id,
                        owner_user_ids=turn.response_user_item_ids,
                    )
                state.pending.clear()
                chat.add_response_item(
                    fc_item,
                    after_user_id=turn.response_user_item_id,
                    owner_user_ids=turn.response_user_item_ids,
                )
        yield self._chunk(turn, tools=[item])

    # ── consumption ─────────────────────────────────────────────────────────--

    def _consume_streaming(self, events: Iterator[ProviderEvent], state: _GenState, turn: _Turn) -> Iterator[LLMOut]:
        cancelled = False
        printable_text = ""
        sentence_batch: list[str] = []

        def _flush(batch: list[str]) -> Iterator[LLMOut]:
            if not batch:
                return
            if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                logger.info("LLM generation cancelled (stale speculative turn)")
                return
            yield self._chunk(turn, text=" ".join(batch))

        for event in events:
            if self._generation_is_stale(turn.gen) or not self._turn_is_latest(turn.turn_id, turn.turn_revision):
                logger.info("LLM generation cancelled (interruption)")
                cancelled = True
                break

            if isinstance(event, Usage):
                state.input_tokens = event.input_tokens
                state.output_tokens = event.output_tokens
            elif isinstance(event, AssistantMessage):
                state.pending.append(
                    RealtimeConversationItemAssistantMessage(type="message", role="assistant", content=event.content)
                )
            elif isinstance(event, ToolCall):
                # Flush any pending spoken text before emitting the tool call.
                if printable_text.strip():
                    sentence_batch.append(printable_text.strip())
                    printable_text = ""
                if sentence_batch:
                    if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                        logger.info("LLM generation cancelled (stale speculative turn)")
                        cancelled = True
                        break
                    yield from _flush(sentence_batch)
                    sentence_batch = []
                yield from self._record_tool_call(state, turn, event.item)
            elif isinstance(event, TextDelta):
                if not turn.wants_audio:
                    # Text-only: forward verbatim. Keep every character (no
                    # remove_unspeechable, which strips TTS-unfriendly symbols) and
                    # don't sentence-split (sent_tokenize collapses newlines/markdown).
                    state.clean_text += event.text
                    if event.text:
                        if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                            logger.info("LLM generation cancelled (stale speculative turn)")
                            cancelled = True
                            break
                        yield self._chunk(turn, text=event.text)
                    continue
                new_text = remove_unspeechable(event.text)
                state.clean_text += new_text
                printable_text += new_text
                sentences = sent_tokenize(printable_text)
                if len(sentences) > 1:
                    for s in sentences[:-1]:
                        sentence_batch.append(s)
                        if len(sentence_batch) >= self.stream_batch_sentences:
                            if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                                logger.info("LLM generation cancelled (stale speculative turn)")
                                cancelled = True
                                break
                            yield from _flush(sentence_batch)
                            sentence_batch = []
                    if cancelled:
                        break
                    printable_text = sentences[-1]

        if not cancelled:
            if printable_text.strip():
                sentence_batch.append(printable_text.strip())
            if sentence_batch:
                if self._generation_is_stale(turn.gen):
                    logger.info("LLM generation cancelled (interruption)")
                else:
                    with private_content_redaction(turn.runtime_config) as private_content:
                        if private_content:
                            logger.debug("Generated text redacted (characters=%d)", len(state.clean_text))
                        else:
                            logger.debug("Clean text: %s", state.clean_text)
                    yield from _flush(sentence_batch)
            with private_content_redaction(turn.runtime_config) as private_content:
                if private_content:
                    logger.info("Generated tools redacted (count=%d)", len(state.tools))
                else:
                    logger.info("Tools: %s", state.tools)

    def _consume_nonstreaming(self, events: Iterator[ProviderEvent], state: _GenState, turn: _Turn) -> Iterator[LLMOut]:
        if self._generation_is_stale(turn.gen) or not self._turn_is_latest(turn.turn_id, turn.turn_revision):
            logger.info("LLM generation cancelled (interruption)")
            return
        for event in events:
            if isinstance(event, Usage):
                state.input_tokens = event.input_tokens
                state.output_tokens = event.output_tokens
            elif isinstance(event, AssistantMessage):
                state.pending.append(
                    RealtimeConversationItemAssistantMessage(type="message", role="assistant", content=event.content)
                )
            elif isinstance(event, ToolCall):
                yield from self._record_tool_call(state, turn, event.item)
            elif isinstance(event, TextDelta):
                # Text-only keeps every character verbatim; audio strips
                # TTS-unfriendly symbols via remove_unspeechable.
                spoken = event.text if not turn.wants_audio else remove_unspeechable(event.text)
                state.clean_text += spoken
                out = spoken if not turn.wants_audio else spoken.strip()
                if (
                    out
                    and not self._generation_is_stale(turn.gen)
                    and self._turn_output_allowed(turn.turn_id, turn.turn_revision)
                ):
                    yield self._chunk(turn, text=out)
        with private_content_redaction(turn.runtime_config) as private_content:
            if private_content:
                logger.debug("Generated text redacted (characters=%d)", len(state.clean_text))
                logger.info("Generated tools redacted (count=%d)", len(state.tools))
            else:
                logger.debug("Clean text: %s", state.clean_text)
                logger.info("Tools: %s", state.tools)

    # ── orchestration ─────────────────────────────────────────────────────────

    def _generate(
        self,
        active_chat: Chat,
        original_chat: Chat,
        turn: _Turn,
        optional_kwargs: dict[str, Any],
    ) -> Iterator[LLMOut]:
        api_response: Any = None
        state = _GenState()
        error_message: str | None = None
        skip_provider = False
        guarded_turn = bool(turn.runtime_config.home_assistant_guard_operational)
        try:
            api_input = self._serialize(active_chat)
        except Exception:
            if not guarded_turn:
                raise
            logger.error("Guarded provider request could not be serialized; private content redacted")
            self._fail_home_assistant_guard_for_current_turn(
                turn.runtime_config,
                generation=turn.gen,
                turn_id=turn.turn_id,
                turn_revision=turn.turn_revision,
            )
            yield EndOfResponse(
                turn_id=turn.turn_id,
                turn_revision=turn.turn_revision,
                cancel_generation=turn.gen,
            )
            return
        # Images the model actually sees this turn; only these are stripped on
        # write-back, so an image a fast client injects mid-generation for the
        # next turn survives (it is not in this serialized snapshot).
        consumed_image_ids = active_chat.image_message_ids()
        if not api_input:
            # Nothing to send: empty `instructions` and no `input` (in the response,
            # the default conversation, or the out-of-band context). The provider
            # would reject this; fail with a clear message instead of an opaque error.
            error_message = "Cannot generate a response: no instructions and no input were provided."
            if guarded_turn:
                self._fail_home_assistant_guard_for_current_turn(
                    turn.runtime_config,
                    generation=turn.gen,
                    turn_id=turn.turn_id,
                    turn_revision=turn.turn_revision,
                )
                error_message = None
                skip_provider = True

        try:
            if error_message is None and not skip_provider:
                api_response, provider_streaming = self._request_for_turn(api_input, optional_kwargs, turn)
            else:
                provider_streaming = self.stream
            if api_response is not None and not self._generation_is_stale(turn.gen):
                events = self._events_for_turn(
                    api_response,
                    optional_kwargs,
                    turn,
                    streaming=provider_streaming,
                )
                if provider_streaming:
                    yield from self._consume_streaming(events, state, turn)
                else:
                    yield from self._consume_nonstreaming(events, state, turn)
        except httpx.ReadTimeout:
            logger.warning(
                "OpenAI API read timed out after %.1fs; ending the current response",
                self.request_timeout_s,
            )
            if guarded_turn:
                self._fail_home_assistant_guard_for_current_turn(
                    turn.runtime_config,
                    generation=turn.gen,
                    turn_id=turn.turn_id,
                    turn_revision=turn.turn_revision,
                )
            elif not self._generation_is_stale(turn.gen) and self._turn_output_allowed(
                turn.turn_id,
                turn.turn_revision,
            ):
                # Canned apology carries no language_code (mirrors the prior handlers).
                yield LLMResponseChunk(
                    text="Wow I'm a bit slow today, could you repeat that?",
                    runtime_config=turn.runtime_config,
                    response=turn.response,
                    turn_id=turn.turn_id,
                    turn_revision=turn.turn_revision,
                    speech_stopped_at_s=turn.speech_stopped_at_s,
                    cancel_generation=turn.gen,
                )
        except Exception as exc:
            # Any other generation failure must still terminate the response: record
            # the error and fall through to the EndOfResponse below. Without this the
            # exception would escape process() and no EndOfResponse would be emitted,
            # leaving st.in_response stuck and locking every subsequent response.
            with private_content_redaction(turn.runtime_config) as private_content:
                if private_content:
                    logger.error("LLM generation failed; private content redacted")
                else:
                    logger.exception("LLM generation failed; ending the current response")
            if guarded_turn:
                self._fail_home_assistant_guard_for_current_turn(
                    turn.runtime_config,
                    generation=turn.gen,
                    turn_id=turn.turn_id,
                    turn_revision=turn.turn_revision,
                )
                error_message = None
            elif error_message is None:
                error_message = (
                    "Language model generation failed in private transcript mode."
                    if private_content
                    else f"Language model generation failed: {exc}"
                )
        finally:
            if api_response is not None and hasattr(api_response, "close"):
                try:
                    api_response.close()
                except Exception:
                    pass

        if (
            error_message is None
            and not self._generation_is_stale(turn.gen)
            and self._turn_output_allowed(turn.turn_id, turn.turn_revision)
        ):
            # Out-of-band responses emit output and usage but never write back to the
            # default conversation (their context was a throwaway chat).
            if not is_out_of_band(turn.response):
                while True:
                    with turn.runtime_config.transcript_barrier_state_guard():
                        if turn.runtime_config.private_protocol_failed or self._generation_is_stale(turn.gen):
                            break
                        owns_writeback = self._turn_owns_writeback_now(turn.turn_id, turn.turn_revision)
                        if owns_writeback is not None:
                            if owns_writeback:
                                # Tool calls (and any assistant text preceding them) were already
                                # written eagerly in _record_tool_call; only trailing items remain.
                                for item in state.pending:
                                    original_chat.add_response_item(
                                        item,
                                        after_user_id=turn.response_user_item_id,
                                        owner_user_ids=turn.response_user_item_ids,
                                    )
                                original_chat.strip_images(consumed_image_ids)
                                original_chat.trim_if_needed(self.compactor)
                            break
                    # A reopen began between the blocking output check and the
                    # guarded writeback fence. Wait only outside the content lock,
                    # then retry if the original revision still owns the turn.
                    if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                        break
            if state.input_tokens or state.output_tokens:
                yield TokenUsage(
                    input_tokens=state.input_tokens,
                    output_tokens=state.output_tokens,
                    turn_id=turn.turn_id,
                    turn_revision=turn.turn_revision,
                    cancel_generation=turn.gen,
                )
        with private_content_redaction(turn.runtime_config) as private_content:
            if private_content and error_message is not None:
                error_message = "Language model generation failed in private transcript mode."
            yield EndOfResponse(
                turn_id=turn.turn_id,
                turn_revision=turn.turn_revision,
                cancel_generation=turn.gen,
                error=error_message,
            )

    def process(self, request: LLMIn) -> Iterator[LLMOut]:
        """Process a language model request and yield LLMResponseChunks."""
        runtime_config = request.runtime_config
        response = request.response
        turn_id = request.turn_id
        turn_revision = request.turn_revision
        speech_stopped_at_s = request.speech_stopped_at_s
        request_generation = request.cancel_generation
        owner_generation = request_generation
        if owner_generation is None and self.cancel_scope is not None:
            owner_generation = self.cancel_scope.generation
        if (
            request_generation is not None
            and self.cancel_scope is not None
            and self.cancel_scope.is_stale(request_generation)
        ):
            logger.info("Skipping cancelled LLM request before provider execution")
            yield EndOfResponse(
                turn_id=turn_id,
                turn_revision=turn_revision,
                cancel_generation=request_generation,
            )
            return
        if not self._turn_is_latest(turn_id, turn_revision):
            logger.info("Skipping stale LLM request for turn=%s rev=%s", turn_id, turn_revision)
            yield EndOfResponse(turn_id=turn_id, turn_revision=turn_revision)
            return

        original_chat = runtime_config.chat
        request_chat = request.chat_snapshot or original_chat
        if is_out_of_band(response):
            try:
                active_chat = build_active_chat(request_chat, response)
            except ChatItemError as exc:
                with private_content_redaction(runtime_config) as private_content:
                    guarded_failure = runtime_config.home_assistant_guard_operational
                    if guarded_failure:
                        error_message = None
                    elif private_content:
                        error_message = "Private out-of-band response rejected."
                        logger.info("Out-of-band response rejected; private content redacted")
                    else:
                        error_message = str(exc)
                        logger.info("Out-of-band response rejected: %s", error_message)
                if guarded_failure:
                    self._fail_home_assistant_guard_for_current_turn(
                        runtime_config,
                        generation=owner_generation,
                        turn_id=turn_id,
                        turn_revision=turn_revision,
                    )
                yield EndOfResponse(
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                    cancel_generation=owner_generation,
                    error=error_message,
                )
                return
        else:
            active_chat = request_chat.copy()
        language_code = request.language_code
        instructions = (
            response.instructions if response and response.instructions else runtime_config.session.instructions
        ) or ""
        req_tools = (
            response.tools
            if response is not None and "tools" in response.model_fields_set
            else runtime_config.session.tools
        )
        req_tool_choice = (
            response.tool_choice if response and response.tool_choice else runtime_config.session.tool_choice
        )
        wants_audio = response_wants_audio(response)
        self._apply_config(active_chat, instructions, wants_audio)
        language_code, lang_name = resolve_auto_language(language_code)
        if lang_name and self.enable_lang_prompt:
            active_chat.add_item(make_user_message(f"Please reply to my message in {lang_name}."))

        optional_kwargs = self._build_optional_kwargs(req_tools, req_tool_choice)

        # CancelScope.is_stale(gen) is checked when the stream iterator advances; a
        # blocked read inside httpx cannot be aborted by cancel_scope.cancel() from
        # the websocket router. Mitigations: request_timeout_s / ReadTimeout.
        gen = owner_generation

        turn = _Turn(
            language_code=language_code,
            gen=gen,
            runtime_config=runtime_config,
            response=response,
            turn_id=turn_id,
            turn_revision=turn_revision,
            response_user_item_id=request.response_user_item_id,
            response_user_item_ids=set(request.response_user_item_ids),
            speech_stopped_at_s=speech_stopped_at_s,
            wants_audio=wants_audio,
        )
        if self.cancel_scope is None:
            yield from self._generate(active_chat, original_chat, turn, optional_kwargs)
            return

        with self.cancel_scope.response_admission(gen) as (admitted, admitted_generation):
            turn.gen = admitted_generation
            if not admitted:
                logger.info("Skipping cancelled LLM request before provider execution")
                yield EndOfResponse(
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                    cancel_generation=admitted_generation,
                )
                return
            yield from self._generate(active_chat, original_chat, turn, optional_kwargs)

    @property
    def timing_log_level(self) -> int:
        return logging.INFO

    def should_log_timing(self, output: LLMOut) -> bool:
        return isinstance(output, LLMResponseChunk) and self.last_time > self.min_time_to_debug
