import json

import pytest

from speech_to_speech.LLM.tool_call import function_call as function_call_module
from speech_to_speech.LLM.tool_call.function_call import (
    FunctionToolCall,
    extract_function_calls_from_text,
    parse_function_call,
)
from speech_to_speech.LLM.tool_call.function_tool import FunctionTool

# ---------------------------------------------------------------------------
# parse_function_call – single calls
# ---------------------------------------------------------------------------


class TestParseFunctionCall:
    @pytest.mark.parametrize(
        "call_str, expected_name, expected_params",
        [
            ("mobile.home()", "mobile.home", {}),
            ("mobile.back()", "mobile.back", {}),
            ("mobile.open_app(app_name='drupe')", "mobile.open_app", {"app_name": "drupe"}),
            ("mobile.long_press(x=0.799, y=0.911)", "mobile.long_press", {"x": 0.799, "y": 0.911}),
            ("mobile.terminate(status='success')", "mobile.terminate", {"status": "success"}),
            ("answer('text')", "answer", {"__arg_0__": "text"}),
            ("pyautogui.hscroll(page=-0.1)", "pyautogui.hscroll", {"page": -0.1}),
            ("pyautogui.scroll(page=-0.1)", "pyautogui.scroll", {"page": -0.1}),
            ("pyautogui.scroll(0.13)", "pyautogui.scroll", {"__arg_0__": 0.13}),
            ("pyautogui.click(x=0.8102, y=0.9463)", "pyautogui.click", {"x": 0.8102, "y": 0.9463}),
            ("pyautogui.hotkey(keys=['ctrl', 'c'])", "pyautogui.hotkey", {"keys": ["ctrl", "c"]}),
            ("pyautogui.press(keys='enter')", "pyautogui.press", {"keys": "enter"}),
            ("pyautogui.press(keys=['enter'])", "pyautogui.press", {"keys": ["enter"]}),
            ("pyautogui.moveTo(x=0.04, y=0.405)", "pyautogui.moveTo", {"x": 0.04, "y": 0.405}),
            ("pyautogui.write(message='bread buns')", "pyautogui.write", {"message": "bread buns"}),
            ("pyautogui.dragTo(x=0.8102, y=0.9463)", "pyautogui.dragTo", {"x": 0.8102, "y": 0.9463}),
        ],
    )
    def test_single_call(self, call_str, expected_name, expected_params):
        results = parse_function_call(call_str)
        assert len(results) == 1
        assert results[0].function_name == expected_name
        assert results[0].parameters == expected_params

    def test_swipe_with_list_params(self):
        results = parse_function_call("mobile.swipe(from_coord=[0.581, 0.898], to_coord=[0.601, 0.518])")
        assert len(results) == 1
        assert results[0].function_name == "mobile.swipe"
        assert results[0].parameters["from_coord"] == [0.581, 0.898]
        assert results[0].parameters["to_coord"] == [0.601, 0.518]


# ---------------------------------------------------------------------------
# parse_function_call – multiple positional arguments
# ---------------------------------------------------------------------------


class TestPositionalArguments:
    def test_bare_identifiers(self):
        results = parse_function_call("function(arg1, arg2, arg3)")
        assert len(results) == 1
        assert results[0].function_name == "function"

    def test_mixed_positional_and_named(self):
        results = parse_function_call("function('hello', 123, x=0.5)")
        r = results[0]
        assert r.parameters["__arg_0__"] == "hello"
        assert r.parameters["__arg_1__"] == 123
        assert r.parameters["x"] == 0.5

    def test_positional_with_named_trailing(self):
        results = parse_function_call("function(arg1, arg2, named_param='value')")
        assert results[0].parameters["named_param"] == "value"

    def test_many_positional(self):
        results = parse_function_call("function(1, 2, 3, 4, 5)")
        r = results[0]
        for i in range(5):
            assert r.parameters[f"__arg_{i}__"] == i + 1

    def test_strings_with_kwargs(self):
        results = parse_function_call("function('a', 'b', 'c', x=1, y=2)")
        r = results[0]
        assert r.parameters["__arg_0__"] == "a"
        assert r.parameters["__arg_1__"] == "b"
        assert r.parameters["__arg_2__"] == "c"
        assert r.parameters["x"] == 1
        assert r.parameters["y"] == 2


# ---------------------------------------------------------------------------
# parse_function_call – only JSON-safe literals are accepted
# ---------------------------------------------------------------------------


class TestLiteralSafety:
    @pytest.mark.parametrize(
        "call_str",
        [
            "tool(payload=b'bytes')",
            "tool(value=1j)",
            "tool(value=...)",
            "tool(value=1e400)",
            "tool(value=-1e400)",
            "tool(value=-True)",
            "tool(payload={1: 'one'})",
            "tool(payload={**other})",
        ],
    )
    def test_non_json_safe_literals_are_rejected(self, call_str):
        with pytest.raises(ValueError):
            parse_function_call(call_str)

    def test_parsed_arguments_always_serialise(self):
        fc = parse_function_call("tool(a=1, b=1.5, c='x', d=None, e=True, f=[1, (2, 3)], g={'k': 'v'})")[0]
        assert json.loads(json.dumps(fc.parameters)) == {
            "a": 1,
            "b": 1.5,
            "c": "x",
            "d": None,
            "e": True,
            "f": [1, [2, 3]],
            "g": {"k": "v"},
        }


# ---------------------------------------------------------------------------
# parse_function_call – nested parens / special characters (Bug 1 fixes)
# ---------------------------------------------------------------------------


class TestNestedParens:
    def test_closing_paren_inside_string(self):
        results = parse_function_call("tool(msg='hello ) world')")
        assert len(results) == 1
        assert results[0].function_name == "tool"
        assert results[0].parameters == {"msg": "hello ) world"}

    def test_tuple_argument(self):
        results = parse_function_call("tool(x=(1, 2))")
        assert len(results) == 1
        assert results[0].parameters == {"x": [1, 2]}

    def test_dict_with_paren_in_value(self):
        results = parse_function_call("tool(a={'nested': ')'})")
        assert len(results) == 1
        assert results[0].parameters == {"a": {"nested": ")"}}

    def test_mixed_nested_structures(self):
        results = parse_function_call("tool(items=[1, (2, 3), 4])")
        assert len(results) == 1
        assert results[0].parameters == {"items": [1, [2, 3], 4]}


# ---------------------------------------------------------------------------
# parse_function_call – multi-line (multiple calls)
# ---------------------------------------------------------------------------


class TestMultiLineParsing:
    def test_two_calls_on_separate_lines(self):
        text = "mobile.wait(seconds=3)\nmobile.swipe(from_coord=[0.581, 0.898], to_coord=[0.601, 0.518])"
        results = parse_function_call(text)
        assert len(results) == 2
        assert results[0].function_name == "mobile.wait"
        assert results[1].function_name == "mobile.swipe"


# ---------------------------------------------------------------------------
# extract_function_calls_from_text
# ---------------------------------------------------------------------------


class TestExtractFromText:
    CODE_BLOCK_REGEX = r"<code>.*?</code>"

    def test_no_code_block_returns_original_text_no_calls(self):
        text = "Hello world, no code blocks here"
        outside, calls = extract_function_calls_from_text(text, block_regex=self.CODE_BLOCK_REGEX)
        assert outside == text
        assert calls == []

    def test_extracts_calls_inside_code_block(self):
        text = "Sure, I'll do that.\n<code>mobile.click(x=0.5)</code>\nDone."
        outside, calls = extract_function_calls_from_text(text, block_regex=self.CODE_BLOCK_REGEX)
        assert len(calls) == 1
        assert calls[0].function_name == "mobile.click"
        assert "mobile.click" not in outside

    def test_ignores_calls_outside_code_block(self):
        text = "mobile.click(x=0.5)\n<code>real.call(a=1)</code>\nmobile.home()"
        outside, calls = extract_function_calls_from_text(text, block_regex=self.CODE_BLOCK_REGEX)
        names = [c.function_name for c in calls]
        assert names == ["real.call"]

    def test_multiline_code_block(self):
        text = "Here:\n<code>\ndo.a()\ndo.b()\n</code>\nDone."
        outside, calls = extract_function_calls_from_text(text, block_regex=self.CODE_BLOCK_REGEX)
        names = [c.function_name for c in calls]
        assert "do.a" in names
        assert "do.b" in names
        assert len(calls) == 2

    def test_multiple_code_blocks(self):
        text = "Step 1\n<code>a.first()</code>\nStep 2\n<code>b.second()</code>\nDone"
        outside, calls = extract_function_calls_from_text(text, block_regex=self.CODE_BLOCK_REGEX)
        names = [c.function_name for c in calls]
        assert names == ["a.first", "b.second"]

    def test_outside_text_excludes_code_blocks(self):
        text = "Hello\n<code>hidden()</code>\nWorld"
        outside, _ = extract_function_calls_from_text(text, block_regex=self.CODE_BLOCK_REGEX)
        assert "<code>" not in outside
        assert "hidden" not in outside
        assert "Hello" in outside
        assert "World" in outside

    def test_no_calls_when_code_block_has_no_functions(self):
        text = "<code>just plain text</code>"
        outside, calls = extract_function_calls_from_text(text, block_regex=self.CODE_BLOCK_REGEX)
        assert calls == []

    def test_nested_parens_inside_code_block(self):
        text = "<code>tool(msg='hello ) world')</code>"
        _, calls = extract_function_calls_from_text(text, block_regex=self.CODE_BLOCK_REGEX)
        assert len(calls) == 1
        assert calls[0].parameters == {"msg": "hello ) world"}

    def test_drops_fallback_call_carrying_a_positional_argument(self):
        text = "<code>search('current weather') dance(</code>"
        _, calls = extract_function_calls_from_text(text, block_regex=self.CODE_BLOCK_REGEX)
        assert calls == []

    def test_recovers_simple_sibling_call_from_malformed_code_block(self, monkeypatch):
        fallback_used = False
        original_fallback = function_call_module._split_simple_calls_with_regex

        def spy_fallback(source: str) -> list[str]:
            nonlocal fallback_used
            fallback_used = True
            return original_fallback(source)

        monkeypatch.setattr(function_call_module, "_split_simple_calls_with_regex", spy_fallback)
        text = "Let me check.\n<code>camera(question='What is in front of me?') dance(</code>"
        outside, calls = extract_function_calls_from_text(text, block_regex=self.CODE_BLOCK_REGEX)
        assert "Let me check." in outside
        assert fallback_used
        assert len(calls) == 1
        assert calls[0].function_name == "camera"
        assert calls[0].parameters == {"question": "What is in front of me?"}


# ---------------------------------------------------------------------------
# to_realtime_function_tool_call – arg stripping & validation (Bug 2 fixes)
# ---------------------------------------------------------------------------


def _make_tool(name: str, properties: dict, required: list[str] | None = None) -> FunctionTool:
    schema = {"type": "object", "properties": properties}
    if required is not None:
        schema["required"] = required
    return FunctionTool(type="function", name=name, parameters=schema)


class TestToRealtimeToolCall:
    def test_maps_live_search_shape_with_only_one_positional_query(self):
        tool_name = "pollen_robotics_reachy_mini_search_tool__search_web"
        fc = parse_function_call(f"{tool_name}('current temperature in Chicago')")[0]
        tool = _make_tool(
            tool_name,
            {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                    "default": 3,
                },
            },
            required=["query"],
        )

        result = fc.to_realtime_function_tool_call([tool])

        assert json.loads(result.arguments) == {
            "query": "current temperature in Chicago",
        }

    def test_maps_observed_live_search_query_and_result_count_positionally(self):
        tool_name = "pollen_robotics_reachy_mini_search_tool__search_web"
        fc = parse_function_call(f"{tool_name}('current Chicago Cubs score', 3)")[0]
        tool = _make_tool(
            tool_name,
            {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                    "default": 3,
                },
            },
            required=["query"],
        )

        result = fc.to_realtime_function_tool_call([tool])

        assert json.loads(result.arguments) == {
            "query": "current Chicago Cubs score",
            "max_results": 3,
        }

    @pytest.mark.parametrize("additional_properties", [False, True])
    def test_maps_observed_person_fact_with_boolean_additional_properties(self, additional_properties):
        fc = parse_function_call("remember_person_fact('The POC test color is teal.')")[0]
        tool = FunctionTool(
            type="function",
            name="remember_person_fact",
            parameters={
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "One short third-person fact.",
                    }
                },
                "required": ["fact"],
                "additionalProperties": additional_properties,
            },
        )

        result = fc.to_realtime_function_tool_call([tool])

        assert json.loads(result.arguments) == {"fact": "The POC test color is teal."}

    def test_maps_one_positional_string_to_sole_missing_required_string(self):
        fc = parse_function_call("search('current temperature', max_results=3)")[0]
        tool = _make_tool(
            "search",
            {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            required=["query"],
        )

        result = fc.to_realtime_function_tool_call([tool])

        assert json.loads(result.arguments) == {
            "query": "current temperature",
            "max_results": 3,
        }

    def test_positional_args_stripped_when_required_present(self):
        fc = FunctionToolCall(
            function_name="greet",
            parameters={"__arg_0__": 1, "msg": "hi"},
            original_string="greet(1, msg='hi')",
        )
        tool = _make_tool("greet", {"msg": {"type": "string"}}, required=["msg"])
        result = fc.to_realtime_function_tool_call([tool])
        args = json.loads(result.arguments)
        assert "__arg_0__" not in args
        assert args == {"msg": "hi"}

    def test_undeclared_args_stripped_when_required_present(self):
        fc = FunctionToolCall(
            function_name="greet",
            parameters={"msg": "hi", "bogus": 42},
            original_string="greet(msg='hi', bogus=42)",
        )
        tool = _make_tool("greet", {"msg": {"type": "string"}}, required=["msg"])
        result = fc.to_realtime_function_tool_call([tool])
        args = json.loads(result.arguments)
        assert "bogus" not in args
        assert args == {"msg": "hi"}

    def test_raises_when_required_missing_after_strip(self):
        fc = FunctionToolCall(
            function_name="greet",
            parameters={"__arg_0__": 1, "bogus": 2},
            original_string="greet(1, bogus=2)",
        )
        tool = _make_tool("greet", {"msg": {"type": "string"}}, required=["msg"])
        with pytest.raises(ValueError, match="Missing required"):
            fc.to_realtime_function_tool_call([tool])

    def test_bare_identifier_positional_string_still_fails_closed(self):
        fc = parse_function_call("search(query)")[0]
        tool = _make_tool(
            "search",
            {"query": {"type": "string"}},
            required=["query"],
        )

        with pytest.raises(ValueError, match="Missing required"):
            fc.to_realtime_function_tool_call([tool])

    def test_reserved_positional_keyword_cannot_shadow_a_parsed_positional(self):
        with pytest.raises(ValueError, match="reserved for parser positionals"):
            parse_function_call("search('quoted', __arg_0__=bare_identifier)")

    def test_duplicate_named_argument_is_rejected(self):
        with pytest.raises(ValueError, match="Duplicate keyword argument"):
            parse_function_call("search('weather', limit=1, limit=2)")

    def test_reserved_schema_property_cannot_receive_a_recovered_positional(self):
        fc = parse_function_call("search('weather')")[0]
        tool = _make_tool(
            "search",
            {"__arg_0__": {"type": "string"}},
            required=["__arg_0__"],
        )

        with pytest.raises(ValueError, match="Malformed properties schema"):
            fc.to_realtime_function_tool_call([tool])

    def test_direct_construction_recovers_only_when_its_source_really_parses_that_way(self):
        tool = _make_tool("search", {"query": {"type": "string"}}, required=["query"])
        faithful = FunctionToolCall(
            function_name="search",
            parameters={"__arg_0__": "weather"},
            original_string="search('weather')",
        )

        assert json.loads(faithful.to_realtime_function_tool_call([tool]).arguments) == {"query": "weather"}

        forged = FunctionToolCall(
            function_name="search",
            parameters={"__arg_0__": "weather"},
            original_string="search(weather)",
        )
        with pytest.raises(ValueError, match="Missing required"):
            forged.to_realtime_function_tool_call([tool])

    @pytest.mark.parametrize("required", ["q", None, {}, ["q", "q"], [1]])
    def test_malformed_required_schema_cannot_enable_recovery(self, required):
        fc = parse_function_call("search('weather')")[0]
        tool = FunctionTool(
            type="function",
            name="search",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": required,
            },
        )

        with pytest.raises(ValueError, match="Malformed required schema"):
            fc.to_realtime_function_tool_call([tool])

    @pytest.mark.parametrize("properties", [None, ["query"]])
    def test_non_dict_properties_fail_closed_with_value_error(self, properties):
        fc = parse_function_call("search('weather')")[0]
        tool = FunctionTool(
            type="function",
            name="search",
            parameters={
                "type": "object",
                "properties": properties,
                "required": ["query"],
            },
        )

        with pytest.raises(ValueError, match="Malformed properties schema"):
            fc.to_realtime_function_tool_call([tool])

    def test_non_dict_properties_reject_named_arguments_instead_of_emitting_empty(self):
        fc = parse_function_call("search(mode='fast')")[0]
        tool = FunctionTool(
            type="function",
            name="search",
            parameters={"type": "object", "properties": None},
        )

        with pytest.raises(ValueError, match="Malformed properties schema"):
            fc.to_realtime_function_tool_call([tool])

    def test_absent_parameters_schema_still_allows_a_no_argument_call(self):
        fc = parse_function_call("dance()")[0]
        tool = FunctionTool(type="function", name="dance", parameters=None)

        result = fc.to_realtime_function_tool_call([tool])

        assert json.loads(result.arguments) == {}

    def test_schema_without_object_type_keeps_named_calls_but_declines_recovery(self):
        tool = FunctionTool(
            type="function",
            name="search",
            parameters={"properties": {"query": {"type": "string"}}, "required": ["query"]},
        )
        named = parse_function_call("search(query='weather')")[0]

        assert json.loads(named.to_realtime_function_tool_call([tool]).arguments) == {"query": "weather"}

        positional = parse_function_call("search('weather')")[0]
        with pytest.raises(ValueError, match="Missing required"):
            positional.to_realtime_function_tool_call([tool])

    @pytest.mark.parametrize(
        "extra",
        [
            {"additionalProperties": 0},
            {"allOf": [{"required": ["locale"]}]},
            {"oneOf": "not-an-array"},
            {"$ref": "#/$defs/search"},
            {"title": "Search"},
        ],
    )
    def test_top_level_schema_outside_the_supported_subset_declines_recovery(self, extra):
        fc = parse_function_call("search('weather')")[0]
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            **extra,
        }
        tool = FunctionTool(type="function", name="search", parameters=schema)

        with pytest.raises(ValueError, match="Missing required"):
            fc.to_realtime_function_tool_call([tool])

    def test_non_string_property_name_is_a_malformed_schema(self):
        fc = parse_function_call("search('weather')")[0]
        tool = _make_tool("search", {1: {"type": "string"}}, required=["query"])

        with pytest.raises(ValueError, match="Malformed properties schema"):
            fc.to_realtime_function_tool_call([tool])

    @pytest.mark.parametrize(
        "properties",
        [
            {"query": {"type": "string"}, "locale": None},
            {"query": {"type": "string"}, "locale": {"type": "string", "enum": "abc"}},
            {"query": {"type": "string"}, "locale": {"type": "string", "enum": []}},
            {"query": {"type": "string"}, "locale": {"type": "string", "enum": ["a", "a"]}},
            {"query": {"type": "string"}, "locale": {"type": "string", "description": 3}},
            {"query": {"type": "string"}, "locale": {"anyOf": [{"type": "string"}]}},
            {"query": {"type": "string"}, "locale": {"type": ["string", "null"]}},
            {"query": {"type": "string"}, "locale": {"type": "string", "pattern": "^a"}},
            {"query": {"type": "string"}, "locale": {"type": "string", "items": {"type": "string"}}},
            {"query": {"type": "string", "default": float("inf")}},
            {"query": {"type": "string"}, "limit": {"type": "string", "minimum": 1}},
            {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": True}},
            {"query": {"type": "string"}, "limit": {"type": "integer", "maximum": float("inf")}},
            {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 4, "maximum": 3},
            },
        ],
    )
    def test_schemas_outside_the_supported_subset_decline_recovery(self, properties):
        fc = parse_function_call("search('weather')")[0]
        tool = _make_tool("search", properties, required=["query"])

        with pytest.raises(ValueError, match="Missing required"):
            fc.to_realtime_function_tool_call([tool])

    def test_recovers_alongside_supported_optional_property_shapes(self):
        fc = parse_function_call("search('weather')")[0]
        tool = _make_tool(
            "search",
            {
                "query": {"type": "string", "description": "What to look up."},
                "locale": {"type": "string", "enum": ["en", "fr"], "default": "en"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 3, "default": 3},
                "sources": {"type": "array", "items": {"type": "string"}},
            },
            required=["query"],
        )

        result = fc.to_realtime_function_tool_call([tool])

        assert json.loads(result.arguments) == {"query": "weather"}

    @pytest.mark.parametrize("count", [0, 4])
    def test_parser_binds_integer_without_enforcing_schema_bounds(self, count):
        fc = parse_function_call(f"search('weather', {count})")[0]
        tool = _make_tool(
            "search",
            {
                "query": {"type": "string"},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                    "default": 3,
                },
            },
            required=["query"],
        )

        result = fc.to_realtime_function_tool_call([tool])

        assert json.loads(result.arguments) == {"query": "weather", "max_results": count}

    @pytest.mark.parametrize("call", ["search('weather', True)", "search('weather', '3')", "search('weather', 3.0)"])
    def test_declines_wrong_positional_type_for_integer_field(self, call):
        fc = parse_function_call(call)[0]
        tool = _make_tool(
            "search",
            {
                "query": {"type": "string"},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                    "default": 3,
                },
            },
            required=["query"],
        )

        with pytest.raises(ValueError, match="Missing required"):
            fc.to_realtime_function_tool_call([tool])

    def test_declines_multi_positional_recovery_when_named_argument_collides(self):
        fc = parse_function_call("search('weather', 3, max_results=2)")[0]
        tool = _make_tool(
            "search",
            {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 3},
            },
            required=["query"],
        )

        with pytest.raises(ValueError, match="Missing required"):
            fc.to_realtime_function_tool_call([tool])

    def test_declines_multi_positional_recovery_for_non_scalar_optional_field(self):
        fc = parse_function_call("search('weather', ['news'])")[0]
        tool = _make_tool(
            "search",
            {
                "query": {"type": "string"},
                "sources": {"type": "array", "items": {"type": "string"}},
            },
            required=["query"],
        )

        with pytest.raises(ValueError, match="Missing required"):
            fc.to_realtime_function_tool_call([tool])

    def test_valid_arbitrary_precision_integer_bound_does_not_escape_recovery(self):
        fc = parse_function_call("search('weather')")[0]
        tool = _make_tool(
            "search",
            {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10**400},
            },
            required=["query"],
        )

        result = fc.to_realtime_function_tool_call([tool])

        assert json.loads(result.arguments) == {"query": "weather"}

    @pytest.mark.parametrize(
        ("call", "mutate"),
        [
            # The recovered value itself no longer matches its source expression.
            ("search('observed')", lambda fc: fc.parameters.__setitem__("__arg_0__", "not-observed")),
            # An equal-but-differently-typed swap (``1`` -> ``True``) that ``==`` misses.
            ("search('weather', limit=1)", lambda fc: fc.parameters.__setitem__("limit", True)),
            # A second positional dropped to make the call look unambiguous.
            ("search('weather', 'today')", lambda fc: fc.parameters.pop("__arg_1__")),
            # The call retargeted at a different tool than the one it was parsed for.
            ("lookup('weather')", lambda fc: setattr(fc, "function_name", "search")),
        ],
    )
    def test_parameter_mutations_that_break_the_reparse_fail_closed(self, call, mutate):
        fc = parse_function_call(call)[0]
        mutate(fc)
        tool = _make_tool(
            "search",
            {"query": {"type": "string"}, "limit": {"type": "integer"}},
            required=["query"],
        )

        with pytest.raises(ValueError, match="Missing required"):
            fc.to_realtime_function_tool_call([tool])

    def test_coherently_rewritten_call_is_simply_another_parsed_call(self):
        fc = parse_function_call("search('weather')")[0]
        fc.parameters["__arg_0__"] = "traffic"
        fc.original_string = "search('traffic')"
        tool = _make_tool("search", {"query": {"type": "string"}}, required=["query"])

        result = fc.to_realtime_function_tool_call([tool])

        assert json.loads(result.arguments) == {"query": "traffic"}

    @pytest.mark.parametrize(
        ("call", "parameter", "mutated", "definition"),
        [
            (
                "search('weather', metadata={'1': 'x'})",
                "metadata",
                {1: "x"},
                {"type": "object"},
            ),
            (
                "search('weather', tags=['local'])",
                "tags",
                ("local",),
                {"type": "array", "items": {"type": "string"}},
            ),
        ],
    )
    def test_nested_json_encoding_collisions_do_not_authorize_recovery(self, call, parameter, mutated, definition):
        fc = parse_function_call(call)[0]
        fc.parameters[parameter] = mutated
        tool = _make_tool(
            "search",
            {"query": {"type": "string"}, parameter: definition},
            required=["query"],
        )

        with pytest.raises(ValueError, match="Missing required"):
            fc.to_realtime_function_tool_call([tool])

    def test_non_string_original_source_mutation_fails_at_the_value_error_boundary(self):
        fc = parse_function_call("search('weather')")[0]
        fc.original_string = 7
        tool = _make_tool("search", {"query": {"type": "string"}}, required=["query"])

        with pytest.raises(ValueError, match="Missing required"):
            fc.to_realtime_function_tool_call([tool])

    @pytest.mark.parametrize("unsafe", [b"bytes", 1j, ..., float("nan"), float("inf")])
    def test_publicly_mutated_named_values_fail_at_the_value_error_boundary(self, unsafe):
        fc = parse_function_call("tool(value='safe')")[0]
        fc.parameters["value"] = unsafe
        tool = _make_tool("tool", {"value": {"type": "string"}}, required=["value"])

        with pytest.raises(ValueError, match="not JSON-safe"):
            fc.to_realtime_function_tool_call([tool])

    def test_positional_does_not_fill_second_required_string(self):
        fc = parse_function_call("send('hello', text='hi there')")[0]
        tool = _make_tool(
            "send",
            {
                "text": {"type": "string"},
                "recipient": {"type": "string"},
            },
            required=["text", "recipient"],
        )

        with pytest.raises(ValueError, match="Missing required"):
            fc.to_realtime_function_tool_call([tool])

    def test_positional_does_not_skip_an_optional_first_property(self):
        fc = parse_function_call("search('weather')")[0]
        tool = _make_tool(
            "search",
            {
                "locale": {"type": "string"},
                "query": {"type": "string"},
            },
            required=["query"],
        )

        with pytest.raises(ValueError, match="Missing required"):
            fc.to_realtime_function_tool_call([tool])

    @pytest.mark.parametrize(
        ("call", "properties", "required"),
        [
            (
                "tool('one', 'two')",
                {"query": {"type": "string"}},
                ["query"],
            ),
            (
                "tool(1)",
                {"query": {"type": "string"}},
                ["query"],
            ),
            (
                "tool('one')",
                {"count": {"type": "integer"}},
                ["count"],
            ),
            (
                "tool('one')",
                {"query": {"type": "string"}, "locale": {"type": "string"}},
                ["query", "locale"],
            ),
            (
                "tool('one', bogus=True)",
                {"query": {"type": "string"}},
                ["query"],
            ),
        ],
    )
    def test_ambiguous_or_non_string_positional_calls_still_fail_closed(
        self,
        call,
        properties,
        required,
    ):
        fc = parse_function_call(call)[0]
        tool = _make_tool("tool", properties, required=required)

        with pytest.raises(ValueError, match="Missing required"):
            fc.to_realtime_function_tool_call([tool])

    def test_succeeds_with_no_required_after_full_strip(self):
        fc = FunctionToolCall(
            function_name="noop",
            parameters={"__arg_0__": 1, "yy": 2},
            original_string="noop(1, yy=2)",
        )
        tool = _make_tool("noop", {"x": {"type": "integer"}})
        result = fc.to_realtime_function_tool_call([tool])
        args = json.loads(result.arguments)
        assert args == {}

    def test_no_collision_with_real_arg_prefix(self):
        """A real parameter named 'arg_0' should NOT be stripped."""
        fc = FunctionToolCall(
            function_name="calc",
            parameters={"arg_0": 10, "x": 5},
            original_string="calc(arg_0=10, x=5)",
        )
        tool = _make_tool(
            "calc",
            {"arg_0": {"type": "integer"}, "x": {"type": "integer"}},
            required=["arg_0"],
        )
        result = fc.to_realtime_function_tool_call([tool])
        args = json.loads(result.arguments)
        assert args == {"arg_0": 10, "x": 5}
