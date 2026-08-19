"""Home Assistant selector guard for trusted local realtime clients."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

HOME_ASSISTANT_GUARD_FIELD = "reachy_home_assistant_guard"
HOME_ASSISTANT_GUARD_VERSION: Literal[1] = 1
HOME_ASSISTANT_TOOL_PREFIX = "home_assistant__"
HOME_ASSISTANT_SELECTOR_REJECTED = "Home Assistant selector output was rejected."
MAX_GUARDED_PROVIDER_EVENTS = 512
MAX_GUARDED_TEXT_CHARS = 16_384
GUARDED_TOOL_CHOICES = frozenset({"auto", "required", "none"})

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_FUNCTION_TOOL_FIELDS = frozenset({"type", "name", "description", "parameters"})


def _jsonable_tool(tool: object) -> dict[str, Any]:
    if isinstance(tool, BaseModel):
        value = tool.model_dump(mode="json", exclude_none=True)
    elif isinstance(tool, Mapping):
        value = dict(tool)
    else:
        raise ValueError("tool schema must be an object")
    if not isinstance(value, dict):
        raise ValueError("tool schema must serialize to an object")
    if set(value) - _CANONICAL_FUNCTION_TOOL_FIELDS:
        raise ValueError("tool schema contains noncanonical fields")
    return value


def registered_tool_surface_names(names: Sequence[str]) -> tuple[str, ...]:
    """Return every client and provider-facing spelling that speech must hide."""
    surface: list[str] = []
    for name in names:
        surface.append(name)
        if name.startswith(HOME_ASSISTANT_TOOL_PREFIX):
            suffix = name.removeprefix(HOME_ASSISTANT_TOOL_PREFIX)
            if suffix:
                surface.append(suffix)
    return tuple(surface)


def valid_guarded_tool_choice(value: object) -> bool:
    """Return whether the initial/response choice has supported exact semantics."""
    return value is None or (type(value) is str and value in GUARDED_TOOL_CHOICES)


def session_contract(
    instructions: str | None,
    tools: Sequence[object] | None,
) -> tuple[str, int, tuple[str, ...]]:
    """Return the canonical digest, complete tool count, and ordered names."""
    serialized_tools = [_jsonable_tool(tool) for tool in tools or ()]
    names: list[str] = []
    for tool in serialized_tools:
        if tool.get("type") != "function" or not isinstance(tool.get("name"), str) or not tool["name"]:
            raise ValueError("only named function tools are supported by the guard")
        names.append(tool["name"])
    if len(names) != len(set(names)):
        raise ValueError("guarded function tool names must be unique")
    payload = json.dumps(
        {"instructions": instructions or "", "tools": serialized_tools},
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest(), len(serialized_tools), tuple(names)


def parse_home_assistant_guard_request(
    value: object,
    *,
    expected_digest: str,
    expected_tool_count: int,
) -> str | None:
    """Return the nonce only for one exact request matching the live session."""
    if not isinstance(value, Mapping) or set(value) != {
        "version",
        "nonce",
        "session_contract_sha256",
        "tool_count",
    }:
        return None
    if type(value.get("version")) is not int or value.get("version") != HOME_ASSISTANT_GUARD_VERSION:
        return None
    nonce = value.get("nonce")
    digest = value.get("session_contract_sha256")
    tool_count = value.get("tool_count")
    if not isinstance(nonce, str) or _SHA256_PATTERN.fullmatch(nonce) is None:
        return None
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        return None
    if digest != expected_digest:
        return None
    if type(tool_count) is not int or tool_count != expected_tool_count:
        return None
    return nonce


class HomeAssistantGuardReadyEvent(BaseModel):
    """Acknowledgement required before a guarded client sends microphone audio."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["reachy.home_assistant_guard.ready"] = "reachy.home_assistant_guard.ready"
    event_id: str
    version: Literal[1] = HOME_ASSISTANT_GUARD_VERSION
    nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    session_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_count: int = Field(ge=1)
