"""
Type coercion and input normalisation utilities for the Finance Toolkit MCP server.

Provides best-effort conversion of LLM- or user-supplied string values to the
Python types required by controller methods, as well as date-string validation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Union, get_args, get_origin

from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()

try:
    from types import UnionType
except ImportError:
    UnionType = None


def _split_list(val: str) -> list[str]:
    """Splits a comma-separated string, with or without brackets, into its values."""
    stripped = val.strip().removeprefix("[").removesuffix("]")
    return [part.strip().strip("'\"") for part in stripped.split(",") if part.strip()]


def _coerce_number(val: Any, int_allowed: bool, float_allowed: bool) -> Any:
    """
    Converts a value to a number: an int when only ints are accepted, a float when only
    floats are, and when both are an int for a whole number and a float otherwise (so
    0.1 stays 0.1 and 3 stays 3). The value is returned as given when it is not a number.
    """
    try:
        number = float(val)
    except (ValueError, TypeError):
        return val
    if int_allowed and (not float_allowed or number.is_integer()):
        return int(number)
    return number


def _coerce_item(val: Any, item_type: Any) -> Any:
    """Converts one element of a list to the element type of the list."""
    members = (
        [a for a in get_args(item_type) if a is not type(None)]
        if get_origin(item_type) is Union
        or (UnionType is not None and isinstance(item_type, UnionType))
        else [item_type]
    )
    if int in members or float in members:
        return _coerce_number(val, int in members, float in members)
    if bool in members:
        return to_boolean(val)
    return val


def coerce_value(val: Any, annotation: Any) -> Any:
    """
    Best-effort type coercion for values based on a type annotation.

    Treats None and empty strings as absent values (returning None). For boolean targets it
    delegates to to_boolean; numbers become an int or a float as the annotation accepts (a
    whole number becomes an int when both are accepted, so 0.1 is never cut to 0).
    Optional/Union annotations (including PEP 604 ``int | None``) are unwrapped.

    A parameter that accepts a list (``list[str]``, ``int | list[int]``,
    ``list[float] | None``) receives a comma-separated or bracket-notation string as a list,
    with its elements converted, since the MCP schema advertises such parameters as a
    string. A single value stays a single value when the annotation also accepts one, and
    becomes a list of one otherwise. A list of objects (option strategy legs) is passed on
    as given.

    If conversion fails or the target type is not recognised, the original value is returned.

    Args:
        val (Any): The value to coerce. Often a string coming from an external source.
        annotation (Any): A type annotation (possibly from typing).

    Returns:
        Any: The coerced value on success, None for None/empty input, or the original
        value if conversion is not possible.
    """
    if val is None or val == "":
        return None

    origin = get_origin(annotation)
    is_union = origin is Union or (
        UnionType is not None and isinstance(annotation, UnionType)
    )
    args = (
        [a for a in get_args(annotation) if a is not type(None)]
        if is_union
        else [annotation]
    )

    lists = [a for a in args if a is list or get_origin(a) is list]
    item_type = get_args(lists[0])[0] if lists and get_args(lists[0]) else str
    scalars = [a for a in args if a in (bool, int, float, str)]

    if lists and get_origin(item_type) is not dict:
        if isinstance(val, list):
            return [_coerce_item(item, item_type) for item in val]
        if isinstance(val, str):
            items = _split_list(val)
            if len(items) > 1 or not scalars:
                return [_coerce_item(item, item_type) for item in items]

    if not scalars:
        return val
    if scalars[0] is bool:
        return to_boolean(val)
    # An annotation that leads with text (str | float) keeps the text as given.
    if int in scalars or scalars[0] is float:
        return _coerce_number(val, int in scalars, float in scalars)
    return val


def accepts_text_only(annotation: Any) -> bool:
    """Whether an annotation takes text (``str`` or ``str | None``) and no list."""
    args = (
        get_args(annotation)
        if get_origin(annotation) is Union
        or (UnionType is not None and isinstance(annotation, UnionType))
        else (annotation,)
    )
    return str in args and not any(a is list or get_origin(a) is list for a in args)


def text_from_parsed_json(value: Any) -> Any:
    """
    Turns a list or object back into text for a parameter that takes text.

    The MCP SDK parses every text argument that reads as a JSON array or object into a
    list or dictionary before validation, unless the parameter is typed exactly ``str``,
    so tickers='["AAPL", "MSFT"]' arrives as a list and an optional text parameter would
    refuse it. A flat list becomes comma-separated text ("AAPL,MSFT"), which every list
    parameter accepts; a nested list or an object becomes its JSON text again.

    Args:
        value (Any): The argument as the SDK passes it on.

    Returns:
        Any: The argument as text when it is a list or dictionary, otherwise as given.
    """
    if isinstance(value, dict) or (
        isinstance(value, list) and any(isinstance(v, list | dict) for v in value)
    ):
        return json.dumps(value)
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return value


def to_boolean(value: str | int | float | bool) -> bool:
    """
    Coerce a value to a boolean, with support for common string representations.

    Args:
        value (Any): Value to coerce. If value is already a bool it is returned unchanged.
            For non-bool inputs the value is converted to a string and compared
            case-insensitively against recognized truthy values: "true", "1", and "yes".

    Returns:
        bool: Boolean interpretation of the input.
    """
    return (
        value if isinstance(value, bool) else str(value).lower() in ("true", "1", "yes")
    )


def validate_date(date_str: str, default_date: str) -> str:
    """
    Normalise a date string to YYYY-MM-DD or return a safe default.

    Args:
        date_str (str): Input date string to validate. Accepts both YYYY-MM-DD
            and YYYY/MM/DD formats.
        default_date (str): Fallback date returned when date_str is absent or
            does not match the expected pattern.

    Returns:
        str: A validated YYYY-MM-DD date string, or default_date if validation fails.
    """
    if date_str:
        cleaned = date_str.replace("/", "-")
        if re.match(r"^\d{4}-\d{2}-\d{2}$", cleaned):
            return cleaned

    logger.warning(f"Invalid date input '{date_str}', defaulting to {default_date}")
    return default_date
