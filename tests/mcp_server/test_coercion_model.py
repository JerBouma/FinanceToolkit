"""MCP Value Coercion Tests"""

import pytest

from financetoolkit.mcp_server.coercion_model import coerce_value

# pylint: disable=missing-function-docstring


@pytest.mark.parametrize(
    ("value", "annotation", "expected"),
    [
        # A fraction is never cut to an int (the Hill estimator's k=0.1).
        ("0.1", int | float, 0.1),
        ("3", int | float, 3),
        ("5", int, 5),
        ("2.5", float | None, 2.5),
        ("true", bool, True),
        # Text-first annotations keep the text.
        ("BBB:0.011", str | float | None, "BBB:0.011"),
        ("", int, None),
    ],
)
def test_scalars(value, annotation, expected):
    result = coerce_value(value, annotation)
    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize(
    ("value", "annotation", "expected"),
    [
        # A list arrives as one string and is split (EIOPA's maturity_years).
        ("1,3,7,50", list[int] | None, [1, 3, 7, 50]),
        ("10", list[int] | None, [10]),
        ("[1, 4]", int | list[int], [1, 4]),
        ("4", int | list[int], 4),
        ("MSFT, GOOGL", str | list[str] | None, ["MSFT", "GOOGL"]),
        ("MSFT", str | list[str] | None, "MSFT"),
        ("0.05,0.95", float | list[float], [0.05, 0.95]),
        (["1", "2"], list[int], [1, 2]),
        ([{"strike_price": 100}], list[dict] | None, [{"strike_price": 100}]),
    ],
)
def test_lists(value, annotation, expected):
    assert coerce_value(value, annotation) == expected
