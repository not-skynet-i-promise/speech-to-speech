#!/usr/bin/env python3
"""
Function parser for extracting function names, parameter names, and values from string function calls.

Uses Python's ``tokenize`` and ``ast`` modules so that nested parentheses,
strings containing ')' characters, tuples, dicts, etc. are handled correctly.
"""

import ast
import io
import json
import logging
import math
import re
import tokenize
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel

from speech_to_speech.LLM.tool_call.function_tool import FunctionTool
from speech_to_speech.LLM.tool_call.signature_from_schema import signature_from_schema
from speech_to_speech.utils.utils import _generate_id

logger = logging.getLogger(__name__)

_POSITIONAL_RE = re.compile(r"^__arg_\d+__$")
_LENIENT_CALL_RE = re.compile(
    r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*"
    r"\((?:[^()\"']+|\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*')*\)"
)

# Narrow JSON Schema subset understood well enough to place a positional value.
_SCHEMA_JSON_TYPES = frozenset({"string", "number", "integer", "boolean", "array", "object", "null"})
_SCHEMA_OBJECT_KEYS = frozenset({"type", "properties", "required", "additionalProperties"})
_SCHEMA_PROPERTY_KEYS = frozenset({"type", "description", "enum", "default", "items", "minimum", "maximum"})


# ── AST / tokenize helpers ───────────────────────────────────────────


def _split_top_level_calls(source: str) -> List[str]:
    """Split *source* into individual ``name(...)`` expression strings.

    Uses the tokenizer to walk tokens and track parenthesis depth so that
    nested parens, strings with ')' chars, etc. are handled correctly.
    """
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    calls: List[str] = []
    i = 0

    while i < len(tokens):
        tok = tokens[i]
        if tok.type != tokenize.NAME:
            i += 1
            continue

        start = i
        j = i + 1

        # Walk past dotted attribute access (e.g. ``mobile.click``)
        while j + 1 < len(tokens) and tokens[j].string == "." and tokens[j + 1].type == tokenize.NAME:
            j += 2

        if j >= len(tokens) or tokens[j].string != "(":
            i += 1
            continue

        # Track balanced parens
        depth = 0
        end = None
        k = j
        while k < len(tokens):
            t = tokens[k]
            if t.type == tokenize.OP and t.string == "(":
                depth += 1
            elif t.type == tokenize.OP and t.string == ")":
                depth -= 1
                if depth == 0:
                    end = k
                    break
            k += 1

        if end is None:
            i += 1
            continue

        calls.append(tokenize.untokenize(tokens[start : end + 1]).strip())
        i = end + 1

    return calls


def _split_simple_calls_with_regex(source: str) -> List[str]:
    """Extract complete simple ``name(args)`` spans from malformed model output.

    This fallback can recover well-formed siblings before a tokenizer error,
    but not the incomplete call that caused the tokenizer error.
    """
    return [match.group(0).strip() for match in _LENIENT_CALL_RE.finditer(source)]


def _parse_function_exprs(
    expressions: List[str],
    pattern_to_match: list[str],
    *,
    lenient_fallback: bool = False,
    redact_private_content: bool = False,
) -> List["FunctionToolCall"]:
    """Parse *expressions* into calls, optionally with the lenient fallback rules.

    The regex fallback re-scans output the tokenizer already rejected, so its
    spans are the untrustworthy ones: unparsable spans are skipped, and any call
    carrying a positional value is dropped entirely rather than reaching
    conversion, where a positional value may otherwise be recovered.
    """
    results: List[FunctionToolCall] = []
    for expr in expressions:
        try:
            call = _parse_call_expr(expr)
        except Exception:
            if lenient_fallback:
                continue
            raise
        if lenient_fallback and any(_POSITIONAL_RE.match(key) for key in call.parameters):
            if redact_private_content:
                logger.warning("Dropping malformed private recovered call; content redacted")
            else:
                logger.warning(
                    "Dropping recovered call '%s' with positional arguments from malformed output",
                    call.function_name,
                )
            continue
        if pattern_to_match and all(pattern not in call.function_name for pattern in pattern_to_match):
            continue
        results.append(call)
    return results


def _extract_function_name(node: ast.expr) -> str:
    """Return the dotted function name from a Call node's ``func`` attribute."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _extract_function_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    raise ValueError(f"Unsupported function target: {ast.dump(node)}")


def _literal_from_ast(node: ast.AST) -> Any:
    """Convert an AST node to a JSON-safe Python literal value.

    Only the literal shapes ``json.dumps`` can serialise are accepted -- ``None``,
    ``bool``, ``int``, finite ``float``, ``str``, lists/tuples of those, and dicts
    with string keys -- so a parsed call can never make argument serialisation
    fail later with an uncaught ``TypeError``. Bare identifiers keep their legacy
    representation as their own name; the string they yield is deliberately not
    treated as a quoted literal anywhere.
    """
    if isinstance(node, ast.Constant):
        constant = node.value
        if constant is None or isinstance(constant, (bool, int, str)):
            return constant
        if isinstance(constant, float) and math.isfinite(constant):
            return constant
        raise ValueError(f"Unsupported constant literal: {ast.dump(node)}")

    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal_from_ast(elt) for elt in node.elts]

    if isinstance(node, ast.Dict):
        literal: Dict[str, Any] = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                raise ValueError("Dict unpacking is not supported")
            literal_key = _literal_from_ast(key)
            if not isinstance(literal_key, str):
                raise ValueError(f"Unsupported dict key: {ast.dump(key)}")
            literal[literal_key] = _literal_from_ast(value)
        return literal

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal_from_ast(node.operand)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Unsupported unary literal: {ast.dump(node)}")
        return -value if isinstance(node.op, ast.USub) else value

    raise ValueError(f"Unsupported literal: {ast.dump(node)}")


def _json_fingerprint(value: Any) -> tuple[Any, ...] | None:
    """Return a recursive type-tagged JSON fingerprint, or ``None``.

    Exact built-in JSON types are required at every level. Type tags distinguish
    values Python equality or ``json.dumps`` otherwise collapses, including
    ``1``/``1.0``/``True``, list/tuple, and integer/string mapping keys. Mapping
    order remains part of the fingerprint because call parameters are ordered.
    """
    value_type = type(value)
    if value is None:
        return ("null",)
    if value_type is bool:
        return ("bool", value)
    if value_type is int:
        return ("int", value)
    if value_type is float:
        return ("float", value.hex()) if math.isfinite(value) else None
    if value_type is str:
        return ("str", value)
    if value_type is list:
        members: list[tuple[Any, ...]] = []
        for item in value:
            member = _json_fingerprint(item)
            if member is None:
                return None
            members.append(member)
        return ("list", tuple(members))
    if value_type is dict:
        entries: list[tuple[str, tuple[Any, ...]]] = []
        for key, item in value.items():
            if type(key) is not str:
                return None
            member = _json_fingerprint(item)
            if member is None:
                return None
            entries.append((key, member))
        return ("dict", tuple(entries))
    return None


def _is_json_safe(value: Any) -> bool:
    """Return whether *value* can be serialised as an argument value."""
    return _json_fingerprint(value) is not None


def _log_dropped_positionals(
    function_name: str,
    positional: list[tuple[str, Any]],
    *,
    redact_private_content: bool = False,
) -> None:
    """Log positional parameter keys without logging their values."""
    if positional:
        if redact_private_content:
            logger.warning("Dropping private positional arguments; content redacted")
        else:
            logger.warning(
                "Dropping positional arguments for '%s': %s",
                function_name,
                {key for key, _value in positional},
            )


def _call_from_ast(parsed: ast.Call, expr: str) -> "FunctionToolCall":
    """Build a FunctionToolCall from an already-parsed ``name(args...)`` node."""
    parameters: "OrderedDict[str, Any]" = OrderedDict()

    for idx, arg in enumerate(parsed.args):
        parameters[f"__arg_{idx}__"] = _literal_from_ast(arg)

    for kw in parsed.keywords:
        if kw.arg is None:
            raise ValueError("**kwargs are not supported")
        if _POSITIONAL_RE.match(kw.arg):
            raise ValueError(f"Keyword name is reserved for parser positionals: {kw.arg}")
        if kw.arg in parameters:
            raise ValueError(f"Duplicate keyword argument: {kw.arg}")
        parameters[kw.arg] = _literal_from_ast(kw.value)

    return FunctionToolCall(
        function_name=_extract_function_name(parsed.func),
        parameters=parameters,
        original_string=expr,
    )


def _parse_call_expr(expr: str) -> "FunctionToolCall":
    """Parse a single ``name(args...)`` expression string into a FunctionToolCall."""
    parsed = ast.parse(expr, mode="eval").body
    if not isinstance(parsed, ast.Call):
        raise ValueError(f"Expression is not a function call: {expr!r}")
    return _call_from_ast(parsed, expr)


def _positional_recovery_fields(
    schema: dict,
    properties: dict,
    required_names: list,
    positional_count: int,
) -> tuple[str, ...] | None:
    """Return the leading fields direct scalar positionals may fill, or ``None``.

    Recovery remains deliberately narrow: the object schema has one required
    first string field and positionals map only to the leading declared
    properties. ``additionalProperties`` does not affect declared-field
    binding, but must be a boolean when present. A recovered field may be a
    string or integer, which covers the observed one-fact and official search
    calls without turning this parser into a JSON Schema implementation. A
    richer but valid schema simply declines recovery.
    """
    if (
        not set(schema) <= _SCHEMA_OBJECT_KEYS
        or schema.get("type") != "object"
        or ("additionalProperties" in schema and type(schema["additionalProperties"]) is not bool)
        or len(required_names) != 1
        or not properties
        or not 1 <= positional_count <= len(properties)
    ):
        return None
    field = required_names[0]
    if next(iter(properties)) != field:
        return None
    if any(not isinstance(name, str) for name in properties):
        return None
    if not all(_supported_property_definition(definition) for definition in properties.values()):
        return None
    fields = tuple(properties)[:positional_count]
    if properties[field].get("type") != "string":
        return None
    if any(properties[name].get("type") not in {"string", "integer"} for name in fields):
        return None
    return fields


def _supported_property_definition(definition: Any) -> bool:
    """Return whether a property definition is written in the supported subset."""
    if not isinstance(definition, dict) or not set(definition) <= _SCHEMA_PROPERTY_KEYS:
        return False

    json_type = definition.get("type")
    if not isinstance(json_type, str) or json_type not in _SCHEMA_JSON_TYPES:
        return False

    if "description" in definition and not isinstance(definition["description"], str):
        return False

    if "default" in definition and not _is_json_safe(definition["default"]):
        return False

    if "enum" in definition:
        enum = definition["enum"]
        if not isinstance(enum, list) or not enum:
            return False
        members = [_json_fingerprint(value) for value in enum]
        if any(member is None for member in members) or len(set(members)) != len(members):
            return False

    if "items" in definition:
        if json_type != "array" or not _supported_property_definition(definition["items"]):
            return False

    bounds: list[int | float] = []
    for keyword in ("minimum", "maximum"):
        if keyword not in definition:
            continue
        bound = definition[keyword]
        if type(bound) not in (int, float):
            return False
        if type(bound) is float and not math.isfinite(bound):
            return False
        bounds.append(bound)
    if bounds and json_type not in {"integer", "number"}:
        return False
    if "minimum" in definition and "maximum" in definition:
        if definition["minimum"] > definition["maximum"]:
            return False

    return True


# ── Data model ───────────────────────────────────────────────────────


class FunctionToolCall(BaseModel):
    """Represents a parsed function call with its parameters."""

    function_name: str
    parameters: Dict[str, Any]
    original_string: str
    description: str = ""

    def _direct_scalar_positionals(self) -> tuple[str | int, ...] | None:
        """Return exact direct string/integer positionals from ``original_string``.

        Re-parsing here *is* the parser observation, so nothing has to be carried
        on the object: the value is returned only when ``original_string`` still
        parses to exactly the call this object describes -- same function name and
        same ordered parameters, compared type-sensitively -- and every positional
        is a direct quoted string or integer literal. Bare identifiers, booleans,
        floats, containers, and computed expressions are never recovered. Fields
        changed together into another valid call are just a different parsed call;
        ``description`` is documentation, not syntax, and is free.
        """
        try:
            node = ast.parse(self.original_string, mode="eval").body
        except (SyntaxError, TypeError, ValueError):
            return None
        if not isinstance(node, ast.Call) or not node.args:
            return None
        values: list[str | int] = []
        for argument in node.args:
            if not isinstance(argument, ast.Constant):
                return None
            value = argument.value
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, int) and not isinstance(value, bool):
                values.append(value)
            else:
                return None

        try:
            reparsed = _call_from_ast(node, self.original_string)
        except ValueError:
            return None
        if reparsed.function_name != self.function_name:
            return None
        fingerprint = _json_fingerprint(self.parameters)
        if fingerprint is None or fingerprint != _json_fingerprint(reparsed.parameters):
            return None
        return tuple(values)

    @staticmethod
    def _positionals_match_declared_types(
        fields: tuple[str, ...],
        values: tuple[str | int, ...],
        properties: dict,
    ) -> bool:
        """Return whether direct scalar values have the exact declared types."""
        for field, value in zip(fields, values, strict=True):
            definition = properties[field]
            json_type = definition["type"]
            if json_type == "string":
                if type(value) is not str:
                    return False
            elif json_type == "integer":
                if type(value) is not int:
                    return False
            else:
                return False
        return True

    def to_realtime_function_tool_call(
        self,
        function_tools: list[FunctionTool] | None = None,
        *,
        redact_private_content: bool = False,
    ) -> ResponseFunctionToolCall:
        positional = [(k, v) for k, v in self.parameters.items() if _POSITIONAL_RE.match(k)]
        arguments = {k: v for k, v in self.parameters.items() if not _POSITIONAL_RE.match(k)}

        if function_tools is not None:
            tool = next(
                (t for t in function_tools if t.name == self.function_name),
                None,
            )
            if tool is None:
                available = [t.name for t in function_tools]
                raise ValueError(f"Function '{self.function_name}' not found in available tools: {available}")

            # A missing or unusable schema keeps its long-standing meaning: nothing
            # is declared, so every argument is dropped. Only shapes this code has
            # to read to make a decision are rejected outright.
            schema = tool.parameters if isinstance(tool.parameters, dict) else {}
            properties = schema.get("properties", {})
            if not isinstance(properties, dict) or any(
                not isinstance(name, str) or _POSITIONAL_RE.match(name) for name in properties
            ):
                raise ValueError(f"Malformed properties schema for '{self.function_name}'")
            required_names = schema.get("required", [])
            if (
                not isinstance(required_names, list)
                or any(not isinstance(name, str) or _POSITIONAL_RE.match(name) for name in required_names)
                or len(set(required_names)) != len(required_names)
            ):
                raise ValueError(f"Malformed required schema for '{self.function_name}'")

            undeclared = {k for k in arguments if k not in properties}
            if undeclared:
                if redact_private_content:
                    logger.warning("Dropping private undeclared parameters; content redacted")
                else:
                    logger.warning(
                        "Dropping undeclared parameters for '%s': %s",
                        self.function_name,
                        undeclared,
                    )
                arguments = {k: v for k, v in arguments.items() if k in properties}

            missing = set(required_names) - set(arguments.keys())
            if positional and not undeclared:
                fields = _positional_recovery_fields(
                    schema,
                    properties,
                    required_names,
                    len(positional),
                )
                if fields is not None and missing == {fields[0]}:
                    values = self._direct_scalar_positionals()
                    if (
                        values is not None
                        and len(values) == len(fields)
                        and self._positionals_match_declared_types(fields, values, properties)
                    ):
                        try:
                            signature_from_schema(schema).bind_partial(*values, **arguments)
                        except (TypeError, ValueError):
                            pass
                        else:
                            arguments.update(zip(fields, values, strict=True))
                            positional = []
                            missing = set(required_names) - set(arguments)
                            if redact_private_content:
                                logger.warning(
                                    "Mapped %d private positional arguments; content redacted",
                                    len(fields),
                                )
                            else:
                                logger.warning(
                                    "Mapped %d positional arguments for '%s' to declared fields %s",
                                    len(fields),
                                    self.function_name,
                                    fields,
                                )
            if missing:
                _log_dropped_positionals(
                    self.function_name,
                    positional,
                    redact_private_content=redact_private_content,
                )
                raise ValueError(f"Missing required parameters for '{self.function_name}': {missing}")

        _log_dropped_positionals(
            self.function_name,
            positional,
            redact_private_content=redact_private_content,
        )

        try:
            serialized_arguments = json.dumps(arguments, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Arguments for '{self.function_name}' are not JSON-safe") from exc

        return ResponseFunctionToolCall(
            name=self.function_name,
            arguments=serialized_arguments,
            call_id=_generate_id("call"),
            type="function_call",
            id=_generate_id("fc"),
            status="in_progress",
        )


# ── Public API ───────────────────────────────────────────────────────


def parse_function_call(
    function_string: str,
    pattern_to_match: list[str] = [],
    *,
    redact_private_content: bool = False,
) -> List[FunctionToolCall]:
    """Parse a function call string and extract all function calls found.

    Args:
        function_string: String representation of function calls.
        pattern_to_match: If non-empty, only calls whose function name
            contains at least one of these substrings are returned.

    Returns:
        List of FunctionToolCall objects with parsed information.
    """
    function_string = function_string.strip()
    if not function_string:
        return []

    try:
        expressions = _split_top_level_calls(function_string)
    except tokenize.TokenError:
        return _parse_function_exprs(
            _split_simple_calls_with_regex(function_string),
            pattern_to_match,
            lenient_fallback=True,
            redact_private_content=redact_private_content,
        )

    return _parse_function_exprs(
        expressions,
        pattern_to_match,
        redact_private_content=redact_private_content,
    )


def parse_multiple_functions(function_strings: List[str]) -> List[FunctionToolCall]:
    """Parse multiple function call strings.

    Args:
        function_strings: List of function call strings.

    Returns:
        List of FunctionToolCall objects.
    """
    results: List[FunctionToolCall] = []
    for func_str in function_strings:
        try:
            results.extend(parse_function_call(func_str))
        except Exception:
            continue
    return results


def extract_function_calls_from_text(
    text: str,
    block_regex: str = ".*",
    *,
    redact_private_content: bool = False,
) -> Tuple[str, List[FunctionToolCall]]:
    """Extract function calls from delimited code blocks inside *text*.

    The LLM is prompted to wrap tool calls inside code-block delimiters
    (e.g. ``<code>func(x=1)</code>``).  This function finds those blocks,
    parses the function calls within them, and returns the remaining text
    (with blocks stripped) alongside the parsed calls.

    Args:
        text: Full model output potentially containing code blocks.
        block_regex: Regex matching the code-block delimiters **and** their
            content (e.g. ``r"<code>.*?</code>"``).  Only text **inside**
            matched blocks is scanned for function calls.

    Returns:
        ``(outside_text, function_calls)`` -- the text with blocks stripped
        and the parsed function calls found inside the blocks.
    """
    if not block_regex:
        return text, []

    matches = list(re.finditer(block_regex, text, flags=re.DOTALL))
    if not matches:
        return text, []

    outside = re.sub(block_regex, "", text, flags=re.DOTALL)
    inside = " ".join(match.group(0) for match in matches).strip()
    if not inside:
        return outside, []

    try:
        return outside, parse_function_call(inside, redact_private_content=redact_private_content)
    except Exception:
        return outside, []
