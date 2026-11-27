"""
Type coercion and input normalisation utilities for the Finance Toolkit MCP server.

Provides best-effort conversion of LLM- or user-supplied string values to the
Python types required by controller methods, as well as date-string validation.
"""

from __future__ import annotations

import contextlib
import json
import math
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
    as given, or read from its JSON text.

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
    scalars = [a for a in args if a in (bool, int, float, str)]
    # A bare list next to a number and no text (a rate, or one per ticker) holds numbers.
    item_type = (
        get_args(lists[0])[0]
        if lists and get_args(lists[0])
        else next((a for a in scalars if a in (int, float) and str not in scalars), str)
    )

    # A list of objects written as JSON text ('[{"strike_price": 340}]') is read as JSON.
    if lists and get_origin(item_type) is dict and isinstance(val, str):
        with contextlib.suppress(json.JSONDecodeError):
            return json.loads(val)

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


def _numbers(value: Any) -> list[float]:
    """The numbers in a value: the value itself, or the elements of a list."""
    values = value if isinstance(value, list | tuple) else [value]

    return [
        float(item)
        for item in values
        if isinstance(item, int | float) and not isinstance(item, bool)
    ]


def refused_value_problem(
    method_name: str, arguments: dict[str, Any], limits: dict[str, Any]
) -> str | None:
    """
    Checks the arguments of a call against the values the server refuses for an indicator
    (config.yaml parameter_limits, refused_values), such as a variant whose fit takes
    longer than an assistant waits for a tool.

    Args:
        method_name (str): The indicator called.
        arguments (dict[str, Any]): The arguments the indicator receives.
        limits (dict[str, Any]): The parameter limits, of which "refused_values" lists the
            indicators ("methods"), the "parameter", its refused "values" and the "reason".

    Returns:
        str | None: Why the value is refused, or None when the call may go ahead.
    """
    for refusal in limits.get("refused_values", []):
        value = arguments.get(refusal["parameter"])
        # A list (such as the models of a comparison) is refused when one of its values is.
        values = value if isinstance(value, list | tuple) else [value]
        # A switch (american_option=true) is refused by its value written in lower case.
        refused = [
            item if isinstance(item, str) else str(item).lower()
            for item in values
            if isinstance(item, str | bool)
            and str(item).strip().lower() in refusal["values"]
        ]
        if method_name in refusal["methods"] and refused:
            return (
                f"`{refusal['parameter']}='{refused[0]}'` is not available on this server: "
                f"{refusal['reason']}"
            )

    return None


# The arguments the server adds for the size limits only (config.yaml parameter_limits): the
# number of tickers of a call and the years between its start and end date (range_years, as
# the short rate simulations take a parameter called years). They are never passed to an
# indicator. With how a product of the limits names them to the caller.
PSEUDO_PARAMETERS = {
    "tickers": "the number of tickers",
    "range_years": "the years from start_date to end_date",
}


def _product_factor(
    name: str, arguments: dict[str, Any], defaults: dict[str, Any]
) -> tuple[float, str] | None:
    """
    The value of one factor of a product limit and how the answer names it, from the
    arguments or else the indicator's defaults. A factor "a|b" takes a, or b when a is left
    out or None (fixings, or time_steps when there are no fixings); a factor "1/x" divides by
    the smallest value of x (refits every x returns). A list counts with its largest value.

    Returns:
        tuple[float, str] | None: The value and its name, or None when no alternative has a
        number (the indicator does not take the parameter).
    """
    divides = name.startswith("1/")
    for alternative in name.removeprefix("1/").split("|"):
        value = arguments.get(alternative)
        numbers = _numbers(value if value is not None else defaults.get(alternative))
        if not numbers:
            continue
        label = PSEUDO_PARAMETERS.get(alternative, f"`{alternative}`")
        if divides:
            return (
                1 / min(numbers) if min(numbers) > 0 else math.inf,
                f"1 / {label}",
            )
        return max(numbers), label
    return None


def _applies(
    product: dict[str, Any],
    method_name: str | None,
    arguments: dict[str, Any],
    defaults: dict[str, Any],
) -> bool:
    """
    Whether a product limit applies to the call: to the indicators it names ("methods", all
    when left out) and when every parameter of "when" has one of its values (the argument,
    or else the indicator's default), compared as lower case text.
    """
    if product.get("methods") and method_name not in product["methods"]:
        return False
    for parameter, values in product.get("when", {}).items():
        value = arguments.get(parameter)
        value = value if value is not None else defaults.get(parameter)
        if str(value).strip().lower() not in {str(v).lower() for v in values}:
            return False
    return True


def size_limit_problem(
    arguments: dict[str, Any],
    defaults: dict[str, Any],
    limits: dict[str, Any],
    method_name: str | None = None,
) -> str | None:
    """
    Checks the size parameters of a call against the limits of the server (config.yaml
    parameter_limits), so a call cannot simulate or download more than the server can hold.

    A product limit is a list of "parameters" whose values multiply to at most "maximum".
    A parameter may be written "a|b" (a, or b when a is left out), "1/x" (divided by x) or
    be one of the PSEUDO_PARAMETERS ("tickers", "range_years"), which the registry adds to the
    arguments of this check. "methods" restricts a product to those indicators, "when" to
    calls with those values of other parameters (such as method='monte_carlo'), and "hint"
    replaces the closing advice of the answer.

    Args:
        arguments (dict[str, Any]): The arguments the indicator receives, with the pseudo
            parameters.
        defaults (dict[str, Any]): The indicator's own default per parameter, used in a
            product for a parameter the caller left out.
        limits (dict[str, Any]): The "maximums" (per parameter, or per parameter of one
            indicator as "indicator.parameter"), the "minimums" (per parameter), the
            "lengths" (the most items of a list, per parameter or "indicator.parameter") and
            the "products".
        method_name (str | None, optional): The indicator called, for the limits of one
            indicator. Defaults to None, which checks only the limits of every indicator.

    Returns:
        str | None: What is over a limit and what to change, or None when the call is within
        them.
    """
    for name, maximum in limits.get("maximums", {}).items():
        indicator, _, parameter = name.rpartition(".")
        if indicator and indicator != method_name:
            continue
        if any(number > maximum for number in _numbers(arguments.get(parameter))):
            return (
                f"`{parameter}` may be at most {maximum:,} on this server, which keeps every call "
                "to a size it can compute next to the others. Please use a smaller value."
            )

    for name, minimum in limits.get("minimums", {}).items():
        if any(number < minimum for number in _numbers(arguments.get(name))):
            return (
                f"`{name}` must be at least {minimum:,} on this server, which keeps every call "
                "to a size it can compute next to the others. Please use a larger value."
            )

    for name, maximum in limits.get("lengths", {}).items():
        indicator, _, parameter = name.rpartition(".")
        if indicator and indicator != method_name:
            continue
        value = arguments.get(parameter)
        items = _split_list(value) if isinstance(value, str) else value
        if isinstance(items, list | tuple) and len(items) > maximum:
            return (
                f"`{parameter}` may list at most {maximum:,} values on this server "
                f"({len(items):,} given), which keeps every call to a size it can compute "
                "next to the others. Please list fewer, or split them over several calls."
            )

    for product in limits.get("products", []):
        if not _applies(product, method_name, arguments, defaults):
            continue
        given = [
            _product_factor(name, arguments, defaults) for name in product["parameters"]
        ]
        factors = [factor for factor in given if factor is not None]
        # A product leaves out an indicator that does not take every parameter of it.
        if len(factors) < len(given):
            continue
        total = math.prod(value for value, _ in factors)
        if total > product["maximum"]:
            labels = [label for _, label in factors]
            # A divisor is lowered by a larger value of its parameter.
            divisors = [
                label.removeprefix("1 / ")
                for label in labels
                if label.startswith("1 / ")
            ]
            advice = product.get("hint") or (
                "Please use smaller values"
                + (f", or a larger {' or '.join(divisors)}" if divisors else "")
                + "."
            )
            return (
                f"{' times '.join(labels)} may be at most {product['maximum']:,} on this "
                f"server ({total:,.4g} here), which keeps every call to a size it can "
                f"compute next to the others. {advice}"
            )

    return None
