import asyncio
import logging
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from queue import Empty, Queue
from threading import Event as ThreadingEvent
from typing import Any, Callable, TypeVar

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai.types.realtime import (
    ConversationItemCreateEvent,
    ConversationItemDeleteEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferCommitEvent,
    ResponseCancelEvent,
    ResponseCreateEvent,
    SessionUpdateEvent,
)
from starlette.websockets import WebSocketState

from speech_to_speech.api.openai_realtime.home_assistant_guard import HOME_ASSISTANT_GUARD_FIELD
from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit, SessionState
from speech_to_speech.api.openai_realtime.service import ServerEvent, build_error_event
from speech_to_speech.api.openai_realtime.transcript_barrier import (
    TRANSCRIPT_BARRIER_FIELD,
    TranscriptBarrierResolveEvent,
)
from speech_to_speech.pipeline.control import SESSION_END, PipelineControlMessage, is_control_message
from speech_to_speech.pipeline.events import (
    AssistantTextEvent,
    PartialTranscriptionEvent,
    PipelineEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TokenUsageEvent,
    TranscriptBarrierCompletedEvent,
    TranscriptBarrierDiscardedEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.log_context import pipeline_log_ctx
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END, AudioOutput

logger = logging.getLogger(__name__)
MAX_AUDIO_BATCH_BYTES = 6400
# How long the release path waits for SESSION_END to propagate through the
# handler chain back to output_queue before clearing unit.session. Tests
# monkeypatch this to a small value since their fixtures usually skip the
# real handler chain.
SESSION_END_DRAIN_TIMEOUT_S = 10.0
QItem = TypeVar("QItem")
_PRIVATE_WEBSOCKET_SCOPE_KEY = "reachy_private_content"
_OUTBOUND_LOCK_SCOPE_KEY = "reachy_outbound_lock"


def _requests_private_transcript_barrier(raw: object) -> bool:
    if not isinstance(raw, Mapping) or raw.get("type") != "session.update":
        return False
    session = raw.get("session")
    return isinstance(session, Mapping) and TRANSCRIPT_BARRIER_FIELD in session


def _requests_home_assistant_guard(raw: object) -> bool:
    if not isinstance(raw, Mapping) or raw.get("type") != "session.update":
        return False
    session = raw.get("session")
    return isinstance(session, Mapping) and HOME_ASSISTANT_GUARD_FIELD in session


def _as_event_list(result: ServerEvent | list[ServerEvent] | None) -> list[ServerEvent]:
    if result is None:
        return []
    return result if isinstance(result, list) else [result]


def _mark_websocket_private(ws: WebSocket) -> None:
    ws.scope[_PRIVATE_WEBSOCKET_SCOPE_KEY] = True


def _websocket_is_private(ws: WebSocket | None) -> bool:
    return bool(ws is not None and ws.scope.get(_PRIVATE_WEBSOCKET_SCOPE_KEY) is True)


async def _send_event_unlocked(ws: WebSocket, event: ServerEvent) -> None:
    # Skip cleanly when the ws is already closing/closed — happens during Ctrl-C
    # shutdown, where the lifespan starts closing sockets while the route handler
    # or send loop is still in flight pushing events.
    if ws.application_state != WebSocketState.CONNECTED:
        return
    try:
        await ws.send_json(event.model_dump())
    except WebSocketDisconnect:
        logger.debug("Skipped event: ws disconnected mid-send")
    except RuntimeError as e:
        # Race: ws closed between the state check above and the send. Starlette
        # raises a plain RuntimeError("Unexpected ASGI message 'websocket.send'
        # after sending 'websocket.close' ...") — harmless during shutdown.
        if _websocket_is_private(ws):
            logger.error("Failed to send private event to client; content redacted")
            return
        msg = str(e)
        if "websocket.close" in msg or "websocket.disconnect" in msg or "response already completed" in msg:
            logger.debug(f"Skipped event: ws already closed ({msg})")
        else:
            logger.error(f"Failed to send event to client: {e}")
    except Exception as e:  # noqa: BLE001
        if _websocket_is_private(ws):
            logger.error("Failed to send private event to client; content redacted")
        else:
            logger.error(f"Failed to send event to client: {e}")


async def _send_event(ws: WebSocket, event: ServerEvent) -> None:
    lock = ws.scope.get(_OUTBOUND_LOCK_SCOPE_KEY)
    if isinstance(lock, asyncio.Lock):
        async with lock:
            await _send_event_unlocked(ws, event)
    else:
        await _send_event_unlocked(ws, event)


async def _send_events_unlocked(ws: WebSocket, events: list[ServerEvent]) -> None:
    for event in events:
        await _send_event_unlocked(ws, event)


async def _send_events(ws: WebSocket, events: list[ServerEvent]) -> None:
    lock = ws.scope.get(_OUTBOUND_LOCK_SCOPE_KEY)
    if isinstance(lock, asyncio.Lock):
        async with lock:
            await _send_events_unlocked(ws, events)
    else:
        await _send_events_unlocked(ws, events)


async def _close_failed_private_session(
    ws: WebSocket | None,
    unit: PipelineUnit,
    session_id: str | None,
    *,
    lock_held: bool = False,
) -> bool:
    if (
        session_id is None
        or not unit.service.private_protocol_failed(session_id)
        or ws is None
        or ws.application_state != WebSocketState.CONNECTED
    ):
        return False
    send = _send_event_unlocked if lock_held else _send_event
    await send(
        ws,
        unit.service.make_error(
            "Private Home Assistant selector failed.",
            "home_assistant_selector_rejected",
        ),
    )
    await ws.close(code=1008, reason="Private session failed")
    return True


def _keep_audio_sentinel(item: Any) -> bool:
    return _is_audio_done(item)


def _keep_user_text_event(item: Any) -> bool:
    return isinstance(
        item,
        (
            SpeechStoppedEvent,
            PartialTranscriptionEvent,
            TranscriptionCompletedEvent,
            TranscriptBarrierCompletedEvent,
            TranscriptBarrierDiscardedEvent,
            TokenUsageEvent,
        ),
    )


def _audio_payload(item: Any) -> Any:
    return item.audio if isinstance(item, AudioOutput) else item


def _audio_generation(item: Any) -> int | None:
    return item.cancel_generation if isinstance(item, AudioOutput) else None


def _flush_queue(q: Queue[QItem], *, preserve: Callable[[QItem], bool] | None = None) -> None:
    """Drain a queue, optionally preserving items matching *preserve*.

    Preserved items are re-inserted at the **front** of the queue
    (atomically under the queue's mutex) so they are processed before
    anything a pipeline thread may have enqueued during the drain.
    """
    preserved: list[QItem] = []
    while True:
        try:
            item = q.get_nowait()
            if preserve and preserve(item):
                preserved.append(item)
        except Empty:
            break
    if preserved:
        with q.mutex:
            for item in reversed(preserved):
                q.queue.appendleft(item)
            q.not_empty.notify(len(preserved))


async def _drain_pending_response_events(
    ws: WebSocket | None,
    unit: PipelineUnit,
    session_id: str | None,
    *,
    lock_held: bool = False,
) -> None:
    if session_id is None:
        return

    preserved: list[Any] = []
    drained_assistant = 0
    drained_usage = 0
    drain_assistant_events = True
    try:
        while True:
            try:
                item = unit.text_output_queue.get_nowait()
            except Empty:
                break
            # Usage is accounting-only, so keep the old whole-queue drain behavior.
            # Assistant events are client-visible response output and stop at the
            # first non-response boundary to preserve normal text-event ordering.
            if isinstance(item, TokenUsageEvent):
                unit.service.dispatch_pipeline_event(session_id, item)
                drained_usage += 1
            elif drain_assistant_events and isinstance(item, AssistantTextEvent):
                drained_assistant += 1
                if _generation_is_discardable(unit, item.cancel_generation):
                    continue
                events = unit.service.dispatch_pipeline_event(session_id, item)
                if ws is not None and events:
                    send = _send_events_unlocked if lock_held else _send_events
                    await send(ws, events)
            else:
                preserved.append(item)
                drain_assistant_events = False
    finally:
        if preserved:
            with unit.text_output_queue.mutex:
                for item in reversed(preserved):
                    unit.text_output_queue.queue.appendleft(item)
                unit.text_output_queue.not_empty.notify(len(preserved))

    if drained_assistant or drained_usage:
        logger.debug(
            "Pipeline %d: drained %d assistant event(s) and %d token usage event(s) before response completion",
            unit.index,
            drained_assistant,
            drained_usage,
        )


async def _forward_audio_item_locked(
    unit: PipelineUnit,
    session: SessionState,
    ws: WebSocket | None,
    session_id: str,
    audio_chunk: Any,
    dequeued_generation: int,
) -> bool:
    """Process one dequeued audio item while ``session.outbound_lock`` is held.

    Returns ``True`` when the pipeline-end sentinel asks the send loop to stop.
    Rechecking cancellation after lock acquisition closes the gap where delete
    owns the transport boundary after this task has already dequeued old output.
    """
    if _is_pipeline_end(audio_chunk):
        await _drain_pending_response_events(ws, unit, session_id, lock_held=True)
        if await _close_failed_private_session(ws, unit, session_id, lock_held=True):
            return True
        if ws is not None:
            await _send_events_unlocked(ws, unit.service.finish_response(session_id))
        return True

    if _is_audio_done(audio_chunk):
        audio_generation = _audio_generation(audio_chunk)
        if audio_generation is not None and unit.cancel_scope.is_stale(audio_generation):
            unit.cancel_scope.response_done(audio_generation)
            unit.should_listen.set()
            logger.info(f"Pipeline {unit.index}: stale response complete, listening re-enabled")
            return False
        await _drain_pending_response_events(ws, unit, session_id, lock_held=True)
        if await _close_failed_private_session(ws, unit, session_id, lock_held=True):
            return False
        if ws is not None:
            await _send_events_unlocked(ws, unit.service.finish_response(session_id))
        unit.response_playing.clear()
        unit.cancel_scope.response_done(audio_generation)
        unit.should_listen.set()
        logger.info(f"Pipeline {unit.index}: response complete, listening re-enabled")
        return False

    # SESSION_END travels from input_queue through every handler to output_queue.
    if is_control_message(audio_chunk, SESSION_END.kind):
        session.drained.set()
        logger.debug(f"Pipeline {unit.index}: SESSION_END drained")
        return False
    if is_control_message(audio_chunk):
        return False
    if unit.service.private_protocol_failed(session_id):
        return False

    if _should_discard_audio(unit, audio_chunk) or (
        _audio_generation(audio_chunk) is None and unit.cancel_scope.generation != dequeued_generation
    ):
        return False

    audio_batch = bytearray(_to_audio_bytes(audio_chunk))
    while len(audio_batch) < MAX_AUDIO_BATCH_BYTES:
        try:
            next_chunk = unit.output_queue.get_nowait()
        except Empty:
            break

        if (
            _is_pipeline_end(next_chunk)
            or _is_audio_done(next_chunk)
            or is_control_message(next_chunk, SESSION_END.kind)
        ):
            session.pending_output_item = next_chunk
            break
        if _should_discard_audio(unit, next_chunk):
            continue
        next_audio = _to_audio_bytes(next_chunk)
        if len(audio_batch) + len(next_audio) > MAX_AUDIO_BATCH_BYTES:
            session.pending_output_item = next_chunk
            break
        audio_batch.extend(next_audio)

    if not unit.response_playing.is_set():
        unit.response_playing.set()
        unit.should_listen.set()
    if ws is not None:
        await _send_events_unlocked(ws, unit.service.encode_audio_chunk(session_id, bytes(audio_batch)))
    return False


def _clean_unit(unit: PipelineUnit, preserve: Callable[[Any], bool] | None = None) -> None:
    """Cancel in-flight work and flush queues for a single pipeline unit.

    All four pipeline queues are drained — input audio, transcript-to-LM,
    LM-to-TTS output, and the text-event side channel — so pending work from
    a released session cannot be picked up by handlers and leak into the next
    session that claims this unit. SESSION_END is enqueued by the route
    handler *after* this returns to serve as the soft reset signal for
    stateful handlers.
    """
    unit.cancel_scope.cancel()
    _flush_queue(unit.input_queue)
    _flush_queue(unit.text_prompt_queue)
    _flush_queue(unit.output_queue, preserve=preserve)
    _flush_queue(unit.text_output_queue, preserve=preserve)
    unit.response_playing.clear()
    unit.cancel_scope.reset()
    unit.should_listen.set()


def _to_audio_bytes(chunk: Any) -> bytes:
    chunk = _audio_payload(chunk)
    if isinstance(chunk, PipelineControlMessage):
        raise TypeError(f"unexpected control message on audio output queue: {chunk!r}")
    if isinstance(chunk, np.ndarray) or hasattr(chunk, "tobytes"):
        return chunk.tobytes()
    return chunk


def _is_audio_done(item: Any) -> bool:
    payload = _audio_payload(item)
    return isinstance(payload, bytes) and payload == AUDIO_RESPONSE_DONE


def _is_pipeline_end(item: Any) -> bool:
    payload = _audio_payload(item)
    return isinstance(payload, bytes) and payload == PIPELINE_END


def _generation_is_discardable(unit: PipelineUnit, generation: int | None) -> bool:
    """Whether output tagged with *generation* should be dropped.

    A generation is discardable if it has been superseded (``is_stale``) or if the
    cancel scope is in its post-cancel discard window and this is not the current
    live generation. Shared by audio and assistant-text so the two paths stay in
    lockstep: dropping text whenever ``discarding`` is set (without this generation
    check) silently swallows the transcript of a fresh response when ``discarding``
    lingers — e.g. a superseded speculative turn whose TTS never emitted an
    AUDIO_RESPONSE_DONE sentinel, so response_done() never cleared the flag.
    """
    if generation is not None and unit.cancel_scope.is_stale(generation):
        return True
    if unit.cancel_scope.discarding and generation != unit.cancel_scope.generation:
        return True
    return False


def _should_discard_audio(unit: PipelineUnit, item: Any) -> bool:
    return _generation_is_discardable(unit, _audio_generation(item))


async def _release_unit_after_drain(unit: PipelineUnit, session: Any, session_id: str) -> None:
    """Wait indefinitely for SESSION_END to propagate, then release the unit.

    Runs in its own asyncio task so the route handler's finally block can return
    immediately. The unit stays unavailable for new claims (unit.session != None)
    until SESSION_END travels all the way through the handler chain back to
    output_queue — observed by the send loop, which sets session.drained.

    Intentionally has no timeout-fallback release. If a handler (e.g. an LM HTTP
    call) is still busy past SESSION_END_DRAIN_TIMEOUT_S, releasing the unit
    would let a new client claim it while stale output from the previous session
    is still in flight — that output would be dispatched under the new session.
    We accept reduced pool capacity over a cross-session leak; operators can see
    stuck units in `/v1/pool` (long `released_at` age).
    """
    elapsed = 0.0
    warned = False
    while not session.drained.is_set():
        await asyncio.sleep(0.05)
        elapsed += 0.05
        if not warned and elapsed >= SESSION_END_DRAIN_TIMEOUT_S:
            logger.warning(
                f"Pipeline {unit.index}: SESSION_END not drained after {elapsed:.1f}s — "
                f"unit will remain unavailable until handlers finish (session {session_id})"
            )
            warned = True
    unit.service.unregister(session_id)
    unit.session = None
    logger.info(f"Pipeline {unit.index} released (session {session_id} ended)")


def create_app(pool: list[PipelineUnit], stop_event: ThreadingEvent) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One send loop per pipeline unit; each polls its own queues and forwards
        # to the websocket currently attached via unit.session.
        send_tasks = [asyncio.create_task(_send_loop_for(unit)) for unit in pool]
        yield
        for task in send_tasks:
            task.cancel()
        for task in send_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        for unit in pool:
            sess = unit.session
            if sess is not None:
                try:
                    await sess.websocket.close()
                except Exception:
                    pass

    app = FastAPI(lifespan=lifespan)

    def _claim_unit(ws: WebSocket) -> PipelineUnit | None:
        """Atomically (between asyncio yield points) reserve the first idle unit.

        Creates a placeholder SessionState that the caller fills in with the
        session_id after RealtimeService.register().
        """
        for unit in pool:
            if unit.session is None:
                unit.session = SessionState(websocket=ws)
                return unit
        return None

    @app.websocket("/v1/realtime")
    async def realtime_endpoint(ws: WebSocket) -> None:
        await ws.accept()

        unit = _claim_unit(ws)
        if unit is None:
            logger.warning(f"Rejected connection: all {len(pool)} pipeline slots in use")
            # Stateless error event — rejection is not chargeable to any unit's usage metrics.
            await _send_event(
                ws,
                build_error_event(
                    f"All {len(pool)} session slots are in use. Disconnect an existing client first.",
                    error_type="session_limit_reached",
                ),
            )
            await ws.close(code=1008, reason="All session slots are in use")
            return

        pipeline_log_ctx.set(unit.index)
        session_id = unit.service.register()
        # _claim_unit guarantees unit.session is not None for the returned unit.
        assert unit.session is not None
        unit.session.session_id = session_id
        ws.scope[_OUTBOUND_LOCK_SCOPE_KEY] = unit.session.outbound_lock
        logger.info(f"Client connected to pipeline {unit.index} (session {session_id})")

        if unit.service.home_assistant_guard_required:
            _mark_websocket_private(ws)

        # Defensive: drain edge queues and reset events so stale data from a
        # previous session that survived SESSION_END propagation doesn't leak.
        _clean_unit(unit)

        try:
            await _send_event(ws, unit.service.build_session_created(session_id))

            while not stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.receive_json(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                activation_requested = _requests_private_transcript_barrier(raw)
                home_assistant_requested = _requests_home_assistant_guard(raw)
                home_assistant_context = (
                    unit.service.home_assistant_guard_required
                    or home_assistant_requested
                    or unit.service.home_assistant_guard_enabled()
                    or unit.service.home_assistant_guard_failed(session_id)
                )
                if activation_requested or home_assistant_context:
                    _mark_websocket_private(ws)
                redact_private_content = _websocket_is_private(ws) or unit.service.sensitive_content()
                event = unit.service.parse_client_event(
                    raw,
                    redact_private_content=redact_private_content,
                )
                if event is None:
                    raw_type = raw.get("type") if isinstance(raw, Mapping) else None
                    if activation_requested:
                        await _send_event(
                            ws,
                            unit.service.poison_transcript_barrier(
                                session_id,
                                "invalid_transcript_barrier",
                            ),
                        )
                        await ws.close(code=1008, reason="Private transcript barrier negotiation failed")
                        return
                    if home_assistant_context:
                        error = unit.service.poison_home_assistant_guard(session_id, "invalid_home_assistant_guard")
                        if raw_type == "conversation.item.delete" and isinstance(raw, Mapping):
                            raw_event_id = raw.get("event_id")
                            if isinstance(raw_event_id, str) and raw_event_id:
                                error.error.event_id = raw_event_id
                        await _send_event(
                            ws,
                            error,
                        )
                        await ws.close(code=1008, reason="Home Assistant guard failed")
                        return
                    if raw_type == "reachy.transcript_barrier.resolve":
                        await _send_event(
                            ws,
                            unit.service.poison_transcript_barrier(
                                session_id,
                                "invalid_transcript_barrier_resolution",
                            ),
                        )
                        await ws.close(code=1008, reason="Private transcript barrier resolution failed")
                        return
                    message = (
                        "Unknown or invalid private client event."
                        if redact_private_content
                        else f"Unknown or invalid event: {raw_type}"
                    )
                    error = unit.service.make_error(message, "unknown_or_invalid_event")
                    if raw_type == "conversation.item.delete" and isinstance(raw, Mapping):
                        raw_event_id = raw.get("event_id")
                        if isinstance(raw_event_id, str) and raw_event_id:
                            error.error.event_id = raw_event_id
                    await _send_event(ws, error)
                    continue

                if unit.service.home_assistant_guard_pending(session_id) and not isinstance(event, SessionUpdateEvent):
                    error = unit.service.poison_home_assistant_guard(session_id, "invalid_home_assistant_guard")
                    if isinstance(event, ConversationItemDeleteEvent):
                        error.error.event_id = event.event_id
                    await _send_event(
                        ws,
                        error,
                    )
                    await ws.close(code=1008, reason="Home Assistant guard negotiation failed")
                    return

                if isinstance(event, InputAudioBufferAppendEvent):
                    if not unit.service.transcript_barrier_audio_allowed(session_id):
                        await _send_event(
                            ws,
                            unit.service.poison_transcript_barrier(
                                session_id,
                                "transcript_barrier_pending",
                            ),
                        )
                        await ws.close(code=1008, reason="Private transcript barrier pending")
                        return
                    chunks = unit.service.handle_audio_append(session_id, event)
                    rt_cfg = unit.service._state(session_id).runtime_config
                    for chunk in chunks:
                        unit.input_queue.put((chunk, rt_cfg))

                elif isinstance(event, InputAudioBufferCommitEvent):
                    err = unit.service.handle_audio_commit(session_id)
                    if err:
                        await _send_event(ws, err)

                elif isinstance(event, SessionUpdateEvent):
                    result = unit.service.handle_session_update(session_id, event)
                    result_events = _as_event_list(result)
                    if activation_requested and (
                        not any(item.type == "reachy.transcript_barrier.ready" for item in result_events)
                        or not unit.service.transcript_barrier_enabled()
                    ):
                        if not result_events:
                            result_events.append(
                                unit.service.poison_transcript_barrier(session_id, "invalid_transcript_barrier")
                            )
                        await _send_events(ws, result_events)
                        await ws.close(
                            code=1008,
                            reason="Private transcript barrier negotiation failed",
                        )
                        return
                    if home_assistant_context and (
                        not any(item.type == "reachy.home_assistant_guard.ready" for item in result_events)
                        or not unit.service.home_assistant_guard_enabled()
                    ):
                        if not result_events:
                            result_events.append(
                                unit.service.poison_home_assistant_guard(session_id, "invalid_home_assistant_guard")
                            )
                        await _send_events(ws, result_events)
                        await ws.close(code=1008, reason="Home Assistant guard negotiation failed")
                        return
                    if result_events:
                        await _send_events(ws, result_events)
                    if unit.service.private_protocol_failed(session_id):
                        await ws.close(code=1008, reason="Private session negotiation failed")
                        return

                elif isinstance(event, ConversationItemCreateEvent):
                    events = unit.service.handle_conversation_item_create(session_id, event)
                    if events:
                        await _send_events(ws, events)
                    if unit.service.private_protocol_failed(session_id):
                        await ws.close(code=1008, reason="Private session failed")
                        return

                elif isinstance(event, ConversationItemDeleteEvent):
                    assert unit.session is not None
                    async with unit.session.outbound_lock:
                        generation_before_delete = unit.cancel_scope.generation
                        events = unit.service.handle_conversation_item_delete(
                            session_id,
                            event,
                            defer_successor_enqueue=True,
                        )
                        cancelled_generation = unit.cancel_scope.generation != generation_before_delete
                        if cancelled_generation:
                            _flush_queue(unit.output_queue, preserve=_keep_audio_sentinel)
                            _flush_queue(unit.text_output_queue, preserve=_keep_user_text_event)
                            unit.response_playing.clear()
                            unit.should_listen.set()
                        if events:
                            await _send_events_unlocked(ws, events)
                        if cancelled_generation:
                            # Release the successor only after cancelled output is
                            # gone and the deletion acknowledgement is on the wire.
                            unit.service.enqueue_pending_response(session_id)
                    if unit.service.private_protocol_failed(session_id):
                        await ws.close(code=1008, reason="Private session failed")
                        return

                elif isinstance(event, ResponseCreateEvent):
                    result = unit.service.handle_response_create(session_id, event)
                    if result:
                        if result.type != "error":
                            unit.cancel_scope.new_response()
                        await _send_event(ws, result)
                    if unit.service.private_protocol_failed(session_id):
                        await ws.close(code=1008, reason="Private session failed")
                        return

                elif isinstance(event, TranscriptBarrierResolveEvent):
                    events = unit.service.handle_transcript_barrier_resolve(session_id, event)
                    if events:
                        await _send_events(ws, events)
                    if unit.service.transcript_barrier_failed(session_id):
                        await ws.close(code=1008, reason="Private transcript barrier resolution failed")
                        return

                elif isinstance(event, ResponseCancelEvent):
                    assert unit.session is not None
                    async with unit.session.outbound_lock:
                        state = unit.service._state(session_id)
                        was_active = state.in_response or state.response_pending
                        if was_active:
                            unit.cancel_scope.cancel()
                        _flush_queue(unit.output_queue, preserve=_keep_audio_sentinel)
                        _flush_queue(unit.text_output_queue, preserve=_keep_user_text_event)
                        events = unit.service.handle_response_cancel(session_id)
                        if events:
                            await _send_events_unlocked(ws, events)
                        unit.response_playing.clear()

                if unit.service.private_protocol_failed(session_id):
                    await _close_failed_private_session(ws, unit, session_id)
                    return

        except WebSocketDisconnect:
            logger.info(f"Client {session_id} disconnected from pipeline {unit.index}")
        except Exception as e:
            if _websocket_is_private(ws) or unit.service.sensitive_content():
                logger.error("Private client pipeline error; content redacted")
            else:
                logger.error(
                    f"Client {session_id} on pipeline {unit.index} error: {type(e).__name__}: {e}",
                    exc_info=True,
                )
        finally:
            # Hold the session reference: the send loop's snapshot will still resolve
            # to this object until we clear unit.session, so any handler output that
            # arrives during the drain window is sent to the now-closed ws (silently
            # dropped) instead of leaking to whichever client claims this unit next.
            old_session = unit.session
            if old_session is not None:
                old_session.released_at = time.monotonic()
            unit.service.scrub_private_protocols_for_disconnect(session_id)
            _clean_unit(unit)
            unit.input_queue.put(SESSION_END)
            # Spawn the drain-and-release as a separate task so the route handler's
            # finally returns immediately. Awaiting here is unreliable: after
            # WebSocketDisconnect propagates, subsequent awaits in the same task
            # can be skipped/cancelled by Starlette's runner and never resume.
            asyncio.create_task(_release_unit_after_drain(unit, old_session, session_id))

    @app.get("/v1/usage")
    async def usage_endpoint() -> dict[str, Any]:
        # Aggregate usage across the pool. Numeric fields sum; dict fields (e.g.
        # errors_by_type) merge with numeric leaves summed too, so per-unit error
        # counts don't get dropped by the first-unit's value.
        def _merge(into: dict[str, Any], src: dict[str, Any]) -> None:
            for k, v in src.items():
                if isinstance(v, (int, float)):
                    into[k] = into.get(k, 0) + v
                elif isinstance(v, dict):
                    sub = into.setdefault(k, {})
                    if isinstance(sub, dict):
                        _merge(sub, v)
                else:
                    into.setdefault(k, v)

        total: dict[str, Any] = {}
        for unit in pool:
            _merge(total, unit.service.get_usage())
        return total

    @app.get("/v1/pool")
    async def pool_endpoint() -> dict[str, Any]:
        now = time.monotonic()

        def _state(u: PipelineUnit) -> dict[str, Any]:
            s = u.session
            if s is None:
                return {"index": u.index, "state": "idle", "session_id": None}
            if s.released_at is None:
                return {"index": u.index, "state": "active", "session_id": s.session_id}
            # released by client but SESSION_END hasn't drained yet → unit
            # is still occupied; surface elapsed time so operators can spot
            # stuck handlers.
            return {
                "index": u.index,
                "state": "draining",
                "session_id": s.session_id,
                "draining_for_s": round(now - s.released_at, 2),
            }

        return {
            "size": len(pool),
            "in_use": sum(1 for u in pool if u.session is not None),
            "units": [_state(u) for u in pool],
        }

    async def _send_loop_for(unit: PipelineUnit) -> None:
        """Per-pipeline send loop. Polls this unit's output queues and forwards
        to the websocket currently attached via unit.session.

        Per-session scratch (pending_output_item) lives on SessionState, so it
        disappears together with the websocket when the session is released —
        no stale sentinel can leak into the next claim.
        """
        pipeline_log_ctx.set(unit.index)
        while not stop_event.is_set():
            try:
                # Snapshot the session once per iteration; if the route releases the
                # unit mid-iteration, we continue against the prior snapshot which is
                # consistent (its websocket is still valid until ws.close() returns).
                session = unit.session
                ws = session.websocket if session is not None else None
                session_id = session.session_id if session is not None else None

                if await _close_failed_private_session(ws, unit, session_id):
                    await asyncio.sleep(0.01)
                    continue

                # Text events first (speech_started cancels active response).
                try:
                    text_msg = unit.text_output_queue.get_nowait()
                    dequeued_generation = unit.cancel_scope.generation
                    if session is not None and ws is not None and session_id:
                        async with session.outbound_lock:
                            is_speech_start = isinstance(text_msg, SpeechStartedEvent)
                            st = unit.service._state(session_id)
                            was_in_response = is_speech_start and st.in_response
                            was_response_pending = is_speech_start and st.response_pending

                            stale_untagged_assistant = (
                                isinstance(text_msg, AssistantTextEvent)
                                and text_msg.cancel_generation is None
                                and unit.cancel_scope.generation != dequeued_generation
                            )
                            if isinstance(text_msg, AssistantTextEvent) and (
                                stale_untagged_assistant or _generation_is_discardable(unit, text_msg.cancel_generation)
                            ):
                                pass
                            elif isinstance(text_msg, PipelineEvent):
                                events = unit.service.dispatch_pipeline_event(session_id, text_msg)
                                if events:
                                    await _send_events_unlocked(ws, events)
                                if unit.service.private_protocol_failed(session_id):
                                    await ws.close(code=1008, reason="Private session failed")

                            if is_speech_start and (was_in_response or was_response_pending):
                                active_cfg = st.runtime_config
                                if text_msg.interrupt_response and active_cfg.interrupt_response_enabled:
                                    unit.cancel_scope.cancel()
                                    unit.service.response.clear_pending_requests(session_id)
                                    _flush_queue(unit.output_queue, preserve=_keep_audio_sentinel)
                                    _flush_queue(unit.text_output_queue, preserve=_keep_user_text_event)
                                    if unit.response_playing.is_set():
                                        unit.response_playing.clear()
                                    logger.info(
                                        "Pipeline %d: speech during %s: cancelled, queue flushed",
                                        unit.index,
                                        "response" if was_in_response else "pending response",
                                    )
                                else:
                                    logger.info(
                                        "Pipeline %d: speech during response: interrupt_response disabled, ignoring",
                                        unit.index,
                                    )
                except Empty:
                    pass

                try:
                    if session is not None and session.pending_output_item is not None:
                        audio_chunk = session.pending_output_item
                        session.pending_output_item = None
                    else:
                        audio_chunk = unit.output_queue.get_nowait()
                    dequeued_generation = unit.cancel_scope.generation
                    if session is None or not session_id:
                        continue
                    async with session.outbound_lock:
                        should_stop = await _forward_audio_item_locked(
                            unit,
                            session,
                            ws,
                            session_id,
                            audio_chunk,
                            dequeued_generation,
                        )
                    if should_stop:
                        break
                except Empty:
                    pass

                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                session = unit.session
                ws = session.websocket if session is not None else None
                if _websocket_is_private(ws) or (
                    session is not None and session.session_id is not None and unit.service.sensitive_content()
                ):
                    logger.error("Private pipeline send loop error; content redacted")
                else:
                    logger.error(f"Pipeline {unit.index} send loop error: {e}")
                await asyncio.sleep(0.1)

    return app
