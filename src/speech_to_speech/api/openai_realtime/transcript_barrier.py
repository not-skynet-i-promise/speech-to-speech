"""Private transcript barrier extension for trusted local realtime clients."""

from __future__ import annotations

import re
from typing import Literal, Mapping

from openai.types.realtime.realtime_conversation_item_user_message import (
    RealtimeConversationItemUserMessage,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

TRANSCRIPT_BARRIER_FIELD = "reachy_private_transcript_barrier"
TRANSCRIPT_BARRIER_VERSION: Literal[1] = 1
TRANSCRIPT_BARRIER_MAX_CHARS = 4_000
_NONCE_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MESSAGE_ID_PATTERN = re.compile(r"^msg_[A-Za-z0-9_-]{1,128}$")


def parse_transcript_barrier_request(value: object) -> str | None:
    """Return the exact nonce for one supported request, otherwise ``None``."""
    if not isinstance(value, Mapping) or set(value) != {"version", "nonce"}:
        return None
    version = value.get("version")
    nonce = value.get("nonce")
    if type(version) is not int or version != TRANSCRIPT_BARRIER_VERSION:
        return None
    if not isinstance(nonce, str) or _NONCE_PATTERN.fullmatch(nonce) is None:
        return None
    return nonce


class TranscriptBarrierReadyEvent(BaseModel):
    """Positive acknowledgement required before the opt-in client streams audio."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["reachy.transcript_barrier.ready"] = "reachy.transcript_barrier.ready"
    event_id: str
    version: Literal[1] = TRANSCRIPT_BARRIER_VERSION
    nonce: str = Field(pattern=r"^[0-9a-f]{64}$")


class TranscriptBarrierCompletedServerEvent(BaseModel):
    """Private final transcript that never enters ordinary protocol/history paths."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["reachy.transcript_barrier.completed"] = "reachy.transcript_barrier.completed"
    event_id: str
    version: Literal[1] = TRANSCRIPT_BARRIER_VERSION
    nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    sequence: int = Field(ge=1)
    item_id: str
    transcript: str = Field(min_length=1, max_length=TRANSCRIPT_BARRIER_MAX_CHARS)
    language_code: str | None = None


class TranscriptBarrierDiscardedServerEvent(BaseModel):
    """Content-free acknowledgement for an empty/whitespace-only final."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["reachy.transcript_barrier.discarded"] = "reachy.transcript_barrier.discarded"
    event_id: str
    version: Literal[1] = TRANSCRIPT_BARRIER_VERSION
    nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    sequence: int = Field(ge=1)
    item_id: str


class TranscriptBarrierFailedServerEvent(BaseModel):
    """Content-free terminal failure; the client must close the session."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["reachy.transcript_barrier.failed"] = "reachy.transcript_barrier.failed"
    event_id: str
    version: Literal[1] = TRANSCRIPT_BARRIER_VERSION
    nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    sequence: int = Field(ge=1)
    item_id: str
    reason: Literal["transcript_too_large", "overlapping_transcript"]


class TranscriptBarrierResolveEvent(BaseModel):
    """Accept one exact private final into history, or discard it content-free."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["reachy.transcript_barrier.resolve"] = "reachy.transcript_barrier.resolve"
    version: Literal[1] = TRANSCRIPT_BARRIER_VERSION
    nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    sequence: int = Field(ge=1)
    input_item_id: str
    action: Literal["accept", "discard"]
    item: RealtimeConversationItemUserMessage | None = None

    @model_validator(mode="after")
    def _validate_action_shape(self) -> "TranscriptBarrierResolveEvent":
        if self.action == "accept":
            if self.item is None or self.item.id is None or _MESSAGE_ID_PATTERN.fullmatch(self.item.id) is None:
                raise ValueError("accept requires one valid replacement message")
            if self.item.role != "user" or self.item.type != "message":
                raise ValueError("accept requires one user message")
        elif self.item is not None:
            raise ValueError("discard cannot include a replacement message")
        return self


class TranscriptBarrierResolvedServerEvent(BaseModel):
    """Content-free acknowledgement that a pending private final was resolved."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["reachy.transcript_barrier.resolved"] = "reachy.transcript_barrier.resolved"
    event_id: str
    version: Literal[1] = TRANSCRIPT_BARRIER_VERSION
    nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    sequence: int = Field(ge=1)
    input_item_id: str
    replacement_item_id: str | None = None
    action: Literal["accepted", "discarded"]
