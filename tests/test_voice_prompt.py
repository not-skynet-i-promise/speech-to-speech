import logging

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.language_model import LanguageModelHandler, StreamContext
from speech_to_speech.LLM.tool_call.function_tool import FunctionTool
from speech_to_speech.LLM.tool_call.tool_prompt import END_CODE, ENTER_CODE, build_block_regex, build_tool_system_prompt
from speech_to_speech.LLM.voice_prompt import VOICE_SYSTEM_PROMPT, build_voice_system_prompt


def test_voice_prompt_is_short_and_keeps_persona_in_session_prompt():
    prompt = build_voice_system_prompt("Be concise.")

    assert len(VOICE_SYSTEM_PROMPT.split()) < 230
    assert len(prompt.split()) < 240
    assert "The session prompt defines persona" in prompt
    assert "Match the user's intent" not in prompt


def test_voice_prompt_makes_speech_the_default_and_handles_noisy_stt():
    prompt = build_voice_system_prompt("Be concise.")

    assert "Speech is the default." in prompt
    assert "Use at most two tools when they help" in prompt
    assert "never repeat an identical call" in prompt
    assert "Use at most one tool" not in prompt
    assert "Treat transcripts as noisy." in prompt
    assert "Correct likely mishearings only if asked or meaning depends on it" in prompt
    assert "Reachy/Richie/Richy" not in prompt
    assert "If unsure whether a tool is needed, just speak." in prompt


def test_voice_prompt_requests_spoken_lead_in_and_sparing_expression_tools():
    prompt = build_voice_system_prompt("Be concise.")

    assert "Before a tool call, use a brief natural utterance" in prompt
    assert "briefly say that you will check" in prompt
    assert "For expression/background tools, speak first." in prompt
    assert "Sure, here's my best <emotion>." in prompt
    assert "Sure, here's my best sadness." not in prompt
    assert "Never mention tools." in prompt
    assert "do not add a second spoken comment" in prompt
    assert "Use motion, dance, emotion, and similar tools sparingly" in prompt


def test_local_tool_prompt_bounds_multiple_tool_calls_and_voice_order():
    prompt = build_tool_system_prompt(
        [
            FunctionTool(
                type="function",
                name="dance",
                description="Dance once.",
                parameters={"type": "object", "properties": {}},
            )
        ]
    )

    assert "Use at most two tool calls" in prompt
    assert "never repeat an identical call" in prompt
    assert "all spoken prose before the first tool call" in prompt
    assert "do not add prose between or after tool calls" in prompt
    assert "Only one tool call may appear in a response." not in prompt


def test_local_tool_prompt_allows_spoken_lead_in_before_code_block():
    prompt = build_tool_system_prompt(
        [
            FunctionTool(
                type="function",
                name="camera",
                description="Look through the camera.",
                parameters={"type": "object", "properties": {}},
            )
        ]
    )

    assert "one brief natural sentence before the tool call" in prompt
    assert "always speak first" in prompt
    assert "Sure, here's my best <emotion>." in prompt
    assert "Sure, here's my best sadness." not in prompt
    assert "fitting empathetic sentence" in prompt
    assert "do not claim tool results before a tool result is available" in prompt
    assert "Omit optional args instead of placeholder values" in prompt


def test_local_tool_parser_flushes_lead_in_before_tool_even_with_large_sentence_batch():
    handler = object.__new__(LanguageModelHandler)
    ctx = StreamContext(
        function_tools=[
            FunctionTool(
                type="function",
                name="dance",
                description="Dance once.",
                parameters={"type": "object", "properties": {}},
            )
        ],
        block_regex=build_block_regex(),
        enter_code=ENTER_CODE,
        end_code=END_CODE,
    )
    text = f"Here we go. {ENTER_CODE}dance(){END_CODE}"

    chunks, tools, remaining = handler._process_printable_text(text, None, [], ctx)

    assert [chunk.text for chunk in chunks] == ["Here we go.", ""]
    assert chunks[0].tools == []
    assert [tool.name for tool in chunks[1].tools] == ["dance"]
    assert [tool.name for tool in tools] == ["dance"]
    assert remaining == ""


def test_local_tool_parser_flushes_pending_batch_before_tool_with_empty_before_text():
    handler = object.__new__(LanguageModelHandler)
    ctx = StreamContext(
        function_tools=[
            FunctionTool(
                type="function",
                name="dance",
                description="Dance once.",
                parameters={"type": "object", "properties": {}},
            )
        ],
        block_regex=build_block_regex(),
        enter_code=ENTER_CODE,
        end_code=END_CODE,
        sentence_batch=["Queued lead-in."],
    )
    text = f"{ENTER_CODE}dance(){END_CODE}"

    chunks, tools, remaining = handler._process_printable_text(text, None, [], ctx)

    assert [chunk.text for chunk in chunks] == ["Queued lead-in.", ""]
    assert chunks[0].tools == []
    assert [tool.name for tool in chunks[1].tools] == ["dance"]
    assert [tool.name for tool in tools] == ["dance"]
    assert remaining == ""


def test_local_tool_parser_drops_repeated_tool_blocks(monkeypatch):
    monkeypatch.setattr(
        "speech_to_speech.LLM.language_model.sent_tokenize",
        lambda value: [value.strip()] if value.strip() else [],
    )
    handler = object.__new__(LanguageModelHandler)
    ctx = StreamContext(
        function_tools=[
            FunctionTool(
                type="function",
                name="dance",
                description="Dance once.",
                parameters={"type": "object", "properties": {}},
            )
        ],
        block_regex=build_block_regex(),
        enter_code=ENTER_CODE,
        end_code=END_CODE,
    )
    text = f"Watch this. {ENTER_CODE}dance(){END_CODE} Watch this. {ENTER_CODE}dance(){END_CODE}"

    chunks, tools, remaining = handler._process_printable_text(text, None, [], ctx)

    assert [chunk.text for chunk in chunks] == ["Watch this.", "", "Watch this."]
    assert [tool.name for tool in chunks[1].tools] == ["dance"]
    assert chunks[2].tools == []
    assert [tool.name for tool in tools] == ["dance"]
    assert remaining == ""


def test_local_tool_parser_caps_distinct_calls_at_two():
    handler = object.__new__(LanguageModelHandler)
    ctx = StreamContext(
        function_tools=[
            FunctionTool(
                type="function",
                name=name,
                description=f"Run {name} once.",
                parameters={"type": "object", "properties": {}},
            )
            for name in ("dance", "camera", "sleep")
        ],
        block_regex=build_block_regex(),
        enter_code=ENTER_CODE,
        end_code=END_CODE,
    )
    text = f"{ENTER_CODE}dance(){END_CODE}{ENTER_CODE}camera(){END_CODE}{ENTER_CODE}sleep(){END_CODE}"

    chunks, tools, remaining = handler._process_printable_text(text, None, [], ctx)

    assert [[tool.name for tool in chunk.tools] for chunk in chunks] == [["dance"], ["camera"]]
    assert [tool.name for tool in tools] == ["dance", "camera"]
    assert remaining == ""


def test_local_tool_parser_redacts_invalid_private_tool_identity(caplog):
    handler = object.__new__(LanguageModelHandler)
    context = StreamContext(
        function_tools=[],
        block_regex=build_block_regex(),
        enter_code=ENTER_CODE,
        end_code=END_CODE,
    )
    runtime_config = RuntimeConfig()
    runtime_config.transcript_barrier_version = 1
    runtime_config.transcript_barrier_nonce = "cd" * 32

    with caplog.at_level(logging.WARNING, logger="speech_to_speech.LLM.language_model"):
        chunks, tools, remaining = handler._process_printable_text(
            f"{ENTER_CODE}PRIVATE_LLM_TOOL_CANARY(){END_CODE}",
            None,
            [],
            context,
            runtime_config,
        )

    assert chunks == []
    assert tools == []
    assert remaining == ""
    assert "PRIVATE_LLM_TOOL_CANARY" not in caplog.text
    assert "content redacted" in caplog.text


def test_local_tool_parser_redacts_malformed_recovered_identity_in_low_level_logs(caplog):
    handler = object.__new__(LanguageModelHandler)
    context = StreamContext(
        function_tools=[],
        block_regex=build_block_regex(),
        enter_code=ENTER_CODE,
        end_code=END_CODE,
    )
    runtime_config = RuntimeConfig()
    runtime_config.transcript_barrier_version = 1
    runtime_config.transcript_barrier_nonce = "ce" * 32
    canary = "PRIVATE_MALFORMED_MODEL_CANARY"

    with caplog.at_level(logging.WARNING):
        chunks, tools, remaining = handler._process_printable_text(
            f"{ENTER_CODE}{canary}('private') broken({END_CODE}",
            None,
            [],
            context,
            runtime_config,
        )

    assert chunks == []
    assert tools == []
    assert remaining == ""
    assert canary not in caplog.text
    assert "content redacted" in caplog.text


def test_local_tool_parser_redacts_dropped_argument_keys_in_low_level_logs(caplog):
    handler = object.__new__(LanguageModelHandler)
    tool = FunctionTool(
        type="function",
        name="safe_tool",
        description="Run safely.",
        parameters={
            "type": "object",
            "properties": {"allowed": {"type": "string"}},
            "required": ["allowed"],
        },
    )
    context = StreamContext(
        function_tools=[tool],
        block_regex=build_block_regex(),
        enter_code=ENTER_CODE,
        end_code=END_CODE,
    )
    runtime_config = RuntimeConfig()
    runtime_config.transcript_barrier_version = 1
    runtime_config.transcript_barrier_nonce = "cf" * 32
    canary = "PRIVATE_DROPPED_ARGUMENT_CANARY"

    with caplog.at_level(logging.WARNING):
        chunks, tools, remaining = handler._process_printable_text(
            f"{ENTER_CODE}safe_tool(allowed='ok', {canary}='private'){END_CODE}",
            None,
            [],
            context,
            runtime_config,
        )

    assert remaining == ""
    assert [tool_call.name for tool_call in tools] == ["safe_tool"]
    assert len(chunks) == 1
    assert canary not in caplog.text
    assert "content redacted" in caplog.text


def test_local_tool_parser_preserves_detailed_default_dropped_argument_log(caplog):
    handler = object.__new__(LanguageModelHandler)
    tool = FunctionTool(
        type="function",
        name="safe_tool",
        description="Run safely.",
        parameters={
            "type": "object",
            "properties": {"allowed": {"type": "string"}},
            "required": ["allowed"],
        },
    )
    context = StreamContext(
        function_tools=[tool],
        block_regex=build_block_regex(),
        enter_code=ENTER_CODE,
        end_code=END_CODE,
    )
    canary = "DEFAULT_DROPPED_ARGUMENT_CANARY"

    with caplog.at_level(logging.WARNING):
        handler._process_printable_text(
            f"{ENTER_CODE}safe_tool(allowed='ok', {canary}='ordinary'){END_CODE}",
            None,
            [],
            context,
        )

    assert canary in caplog.text


def test_local_tool_parser_cap_and_dedup_hold_across_streaming_invocations():
    handler = object.__new__(LanguageModelHandler)
    ctx = StreamContext(
        function_tools=[
            FunctionTool(
                type="function",
                name=name,
                description=f"Run {name} once.",
                parameters={"type": "object", "properties": {}},
            )
            for name in ("dance", "camera", "sleep")
        ],
        block_regex=build_block_regex(),
        enter_code=ENTER_CODE,
        end_code=END_CODE,
    )

    first, tools, _ = handler._process_printable_text(f"{ENTER_CODE}dance(){END_CODE}", None, [], ctx)
    duplicate, tools, _ = handler._process_printable_text(f"{ENTER_CODE}dance(){END_CODE}", None, tools, ctx)
    second, tools, _ = handler._process_printable_text(f"{ENTER_CODE}camera(){END_CODE}", None, tools, ctx)
    excess, tools, _ = handler._process_printable_text(f"{ENTER_CODE}sleep(){END_CODE}", None, tools, ctx)

    assert [[tool.name for tool in chunk.tools] for chunk in first] == [["dance"]]
    assert duplicate == []
    assert [[tool.name for tool in chunk.tools] for chunk in second] == [["camera"]]
    assert excess == []
    assert [tool.name for tool in tools] == ["dance", "camera"]


def test_local_tool_parser_preserves_interleaved_text_and_tool_calls(monkeypatch):
    monkeypatch.setattr(
        "speech_to_speech.LLM.language_model.sent_tokenize",
        lambda value: [value.strip()] if value.strip() else [],
    )
    handler = object.__new__(LanguageModelHandler)
    ctx = StreamContext(
        function_tools=[
            FunctionTool(
                type="function",
                name="dance",
                description="Dance once.",
                parameters={"type": "object", "properties": {}},
            ),
            FunctionTool(
                type="function",
                name="camera",
                description="Look through the camera.",
                parameters={"type": "object", "properties": {}},
            ),
        ],
        block_regex=build_block_regex(),
        enter_code=ENTER_CODE,
        end_code=END_CODE,
    )
    text = f"First. {ENTER_CODE}dance(){END_CODE} Middle. {ENTER_CODE}camera(){END_CODE} Last."

    chunks, tools, remaining = handler._process_printable_text(text, None, [], ctx)

    assert [(chunk.text, [tool.name for tool in chunk.tools]) for chunk in chunks] == [
        ("First.", []),
        ("", ["dance"]),
        ("Middle.", []),
        ("", ["camera"]),
    ]
    assert [tool.name for tool in tools] == ["dance", "camera"]
    assert remaining.strip() == "Last."
