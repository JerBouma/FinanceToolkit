"""
Dynamic tool registration for the Finance Toolkit MCP Server — Router Pattern.

Consolidates 500+ individual methods into 22 categorical master tools, each
accepting an indicator parameter (Literal enum) that routes to the
correct underlying method.
"""

from __future__ import annotations

import contextlib
import difflib
import importlib
import inspect
import re
import time
import types
import typing
from datetime import datetime, timedelta
from typing import Annotated, Any, NamedTuple

import pandas as pd
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field as PydanticField

from financetoolkit.fmp_model import parse_fiscal_period
from financetoolkit.mcp_server import series_model
from financetoolkit.mcp_server.analytics_model import UsageAnalytics
from financetoolkit.mcp_server.coercion_model import (
    coerce_value,
    to_boolean,
    validate_date,
)
from financetoolkit.mcp_server.diagnostics_model import (
    call_input_info,
    capture_call_messages,
    missing_data_notes,
    redact,
    summarize_reasons,
)
from financetoolkit.mcp_server.formatting_model import (
    country_names,
    format_result,
    has_period_columns,
    is_time_index_of,
    is_yearly,
    select_rows,
)
from financetoolkit.mcp_server.inspection_controller import ControllerInspector
from financetoolkit.mcp_server.provider_model import ToolkitProvider
from financetoolkit.utilities import validation_model
from financetoolkit.utilities.dataframe_model import filter_columns as _filter_columns
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.statistics_model import normalize_country_name

logger = get_logger()

# Types that are not representable in JSON Schema — dropped during normalization.
_OPAQUE_TYPES = {
    "list",
    "dict",
    "set",
    "tuple",
    "range",
    "ndarray",
    "Series",
    "DataFrame",
}


# Injected into each tool's Args docstring so FastMCP can surface them.
def _curve_maturity(maturity: Any) -> float | str:
    """A curve maturity as a number of years, or the label itself ("10Y", "6M")."""
    try:
        return float(maturity)
    except (TypeError, ValueError):
        label = str(maturity).strip().upper()
        if label[:-1].replace(".", "", 1).isdigit() and label[-1:] in (
            "D",
            "W",
            "M",
            "Y",
        ):
            return label
        raise


# The indicators that read the returns of the period itself rather than results within each
# period, so a blocked period (e.g. daily performance) does not apply to them.
PERIOD_BLOCK_EXEMPT = frozenset({"get_style_analysis"})
# The indicators whose own limit also takes a negative number, for the last rows.
NEGATIVE_LIMIT_METHODS = frozenset({"get_etf_holdings"})


# Injected into each tool's Args docstring so FastMCP can surface them.
_PARAM_DESCRIPTIONS: dict[str, str] = {
    "indicator": (
        "Name of the specific metric to calculate, one of the listed values. Required — "
        "search_metrics finds the indicator (and its tool) for a metric by name."
    ),
    "tickers": "Comma-separated ticker symbols, e.g. 'AAPL,MSFT,GOOGL'.",
    "countries": (
        "Country names, comma-separated or as a list, e.g. 'United States,Germany,Japan'."
    ),
    "start_date": "Start of the date range in YYYY-MM-DD format.",
    "end_date": "End of the date range in YYYY-MM-DD format.",
    "quarterly": "Return quarterly data instead of annual when True.",
    "benchmark_ticker": "Ticker used as the market benchmark, e.g. 'SPY' or '^GSPC'.",
    "growth": "Return period-over-period growth rates instead of absolute values.",
    "lag": "Number of periods to lag when computing growth rates.",
    "standardize": (
        "Return the Z-Score (standard score) instead of the raw values, i.e. how many "
        "standard deviations each value is from the mean of its own series. When "
        "combined with growth=True, the growth values are standardized instead of the "
        "raw values."
    ),
    "cumulative": (
        "Return the cumulative value compounded over time instead of the discrete "
        "value per period. Always rebased to start at 1 at the beginning of the "
        "selected date range."
    ),
    "rolling": (
        "Rolling window size in number of periods. When set, the metric is computed over "
        "a smoothly overlapping trailing window across the full history (e.g. period='monthly' "
        "and rolling=6 gives a rolling 6-month value) instead of one value per period, or "
        "(for economics indicators) a simple moving average used to smooth the raw series."
    ),
    "trailing": (
        "Trailing window size in number of periods. Sums the raw values over the trailing N "
        "periods (e.g. trailing=4 on quarterly data gives a trailing-4-quarter / TTM-style sum) "
        "instead of returning one value per period."
    ),
    "threshold_percentile": (
        "Only used when distribution='evt'. The percentile of losses above which the "
        "Generalized Pareto Distribution is fitted, e.g. 0.95 fits on the worst 5% of losses."
    ),
    "minimum_acceptable_return": (
        "The minimum acceptable return (MAR) threshold below which returns are considered "
        "downside, e.g. 0.0 for downside relative to a zero return."
    ),
    "days": "Number of calendar days used in day-count-based calculations.",
    "period": (
        "Observation frequency, e.g. 'daily', 'weekly', 'monthly', 'quarterly', or 'annual' "
        "(supported values differ per method)."
    ),
    "measure": "Sub-measure selector, e.g. 'M1', 'M2', or 'M3' for money supply.",
    "gmdb_source": (
        "Use the Global Macro Database as the data source when True, rather than the "
        "OECD. The two are independent providers with different country and period "
        "coverage; both return rates and ratios as decimal fractions."
    ),
    "annualize": (
        "Set to true to annualize a Sharpe ratio (by default it is per observation, e.g. "
        "per day, so roughly sqrt(252) smaller than the annualized figure)."
    ),
    "primary_only": (
        "Set to true to keep only the most traded listing of each company rather than "
        "every listing (e.g. ING in Amsterdam, New York, Toronto and Frankfurt)."
    ),
    "maturity_years": (
        "Comma-separated maturities in years to return, e.g. '1,2,5,10,20,30'."
    ),
    "fiscal_period": (
        "The company's fiscal quarter, e.g. '2025Q4' (its own fiscal year, which differs from "
        "the calendar year for e.g. Apple and Microsoft). Leave unset for the most recent."
    ),
    "window": "The number of periods the indicator looks back over, e.g. 14 or 20.",
    "weights": (
        "The portfolio weight per ticker as an object, e.g. {'AAPL': 0.6, 'MSFT': 0.4}; "
        "equal weights when left unset."
    ),
    "lags": "The number of lags (past periods) the test or model uses, e.g. 1 or 5.",
    "column": "The price column to use, e.g. 'Close' or 'Adj Close'.",
    "strike_price_range": (
        "Only strikes within this fraction of the current stock price, e.g. 0.05 for the "
        "strikes within 5% of it."
    ),
    "legs": (
        "The strategy as a list of legs. An option leg: {'strike_price': 340, "
        "'put_option': false, 'position': 'long' or 'short', 'premium': 4.2}; a stock leg: "
        "{'instrument': 'stock', 'position': 'long', 'premium': 336}. E.g. a bull call "
        "spread is a long call at a lower and a short call at a higher strike."
    ),
    "limit": (
        "The rows to return, like head and tail: 5 for the first five, -5 for the last five "
        "(the most recent of a time series). A table with its periods as columns (financial "
        "statements, ratios) keeps its first or last periods instead. For ETF holdings the "
        "largest (or, negative, the smallest) by weight, and for a list (e.g. news or a "
        "screener) its length."
    ),
    "gmdb_forecasts": (
        "Set to true to include the years the Global Macro Database projects (IMF World "
        "Economic Outlook, up to five years ahead); by default a yearly series ends with "
        "its last observation."
    ),
    "inflation_adjusted": "Adjust nominal values for inflation when True.",
    "usd": (
        "Return levels in millions of US dollars instead of national currency when True, "
        "which makes them comparable across countries (Global Macro Database)."
    ),
    "component": ("Part of final consumption: 'total', 'household' or 'government'."),
    "level": (
        "Level of government: 'consolidated' (longest history), 'general' (central, "
        "state and local government and social security) or 'central'."
    ),
    "bond_price": "Clean price of the bond per 100 face value.",
    "coupon_rate": "Annual coupon rate as a decimal, e.g. 0.05 for 5 %.",
    "years_to_maturity": "Years remaining until the bond matures.",
    "maturities": "Comma-separated maturities, e.g. '1M,3M,6M,1Y'.",
    "strike_rate": "Option strike price.",
    "factors_to_calculate": "Comma-separated factor names to include in the calculation.",
    "growth_rate": "Assumed constant growth rate as a decimal.",
    "perpetual_growth_rate": "Terminal (perpetual) growth rate used in DCF models.",
    "weighted_average_cost_of_capital": "WACC as a decimal, e.g. 0.09 for 9 %.",
    "series": (
        "Analyze the indicators of other tools instead of the tickers' prices: references "
        "separated by ';', each tool:indicator:tickers-or-countries with an optional "
        ":column, e.g. 'valuation:get_price_to_earnings_ratio:AAPL,MSFT; "
        "macroeconomics:get_inflation_rate:United States' or "
        "'rates:get_government_bond_yield_curve:United States,Germany:10Y'. Every ticker or "
        "country becomes a series named after it, which dependent_ticker and "
        "independent_tickers then name. Use start_date for a long enough history."
    ),
    "changes": (
        "With series: how their changes (the 'Return' column the regressions and "
        "diagnostics use) are calculated: 'percentage' (default), 'log', 'difference' (for "
        "rates and ratios) or 'none' (the series are changes already). The series "
        "themselves are the 'Adj Close' column the unit root and cointegration tests use."
    ),
    "show_columns": (
        "Comma-separated names to filter the output. "
        "For historical data use the key names visible in any response record "
        "(e.g. 'Close,Volume,Return'). "
        "For financial statements use the 'metric' field values from the response "
        "(e.g. 'Revenue,Net Income,EBITDA'). "
        "Call the tool once without this parameter to see all available names, "
        "then repeat with show_columns to reduce response size and token usage."
    ),
}


# Beyond this many characters the per-indicator default listing stops helping.
_DISPUTED_DEFAULT_DETAIL_LIMIT = 320


def _annotate_with_description(
    name: str, ann: Any, suffix: str = "", documented: str = ""
) -> Any:
    """Wrap *ann* in ``Annotated[ann, Field(description=...)]`` for JSON schema output.

    Pydantic reads the ``Field(description=...)`` metadata and includes it in the
    ``properties[name].description`` field of the generated JSON schema, which is
    what Smithery (and other MCP clients) use to display parameter descriptions.

    Args:
        name (str): The parameter name, used to look up the shared description.
        ann (Any): The annotation to wrap.
        suffix (str): Extra text appended to the description, used to spell out
            defaults that differ between the indicators of a router group.
        documented (str): The description the indicators' docstrings give, used when
            there is no shared description for the parameter.

    Returns:
        Any: The annotated type carrying the description metadata.
    """
    desc = _PARAM_DESCRIPTIONS.get(name) or documented or f"Value for {name}."

    if suffix:
        desc = f"{desc} {suffix}"

    return Annotated[ann, PydanticField(description=desc)]  # type: ignore[valid-type]


# The length of an indicator's summary in the search results.
MAXIMUM_SUMMARY_CHARACTERS = 200


# The inputs of a bond or derivative calculation whose defaults are assumptions about the
# instrument, named in the answer when the caller leaves them out.
ASSUMED_INPUTS = (
    "par_value",
    "coupon_rate",
    "yield_to_maturity",
    "bond_price",
    "volatility",
    "strike_rate",
    "notional",
)

# The yield curves the fixed income calculations take, which default to a sample curve,
# except for the calculations that default to the current market curve.
CURVE_PARAMETERS = ("spot_rates", "nominal_rates", "real_rates")
MARKET_CURVE_METHODS = ("get_option_adjusted_spread",)

# A window of at most this many days can be empty for lack of trading days alone.
SHORT_WINDOW_DAYS = 3

# How alike an unknown indicator name must be to a known one to be read as it.
NEAR_CERTAIN_MATCH = 0.9

# The most rows a note names one by one; more are counted instead.
MAXIMUM_NAMED_ROWS = 10

# The periods the toolkit resamples to, and the intervals of intraday data.
PERIODS = ("daily", "weekly", "monthly", "quarterly", "yearly", "annual")
INTRADAY_PERIODS = ("1min", "5min", "15min", "30min", "1hour")


# The most distinct docstring descriptions of one parameter that are listed per indicator.
MAXIMUM_DESCRIPTION_VERSIONS = 3

# The longest parameter description taken from a docstring.
MAXIMUM_DOCUMENTED_CHARACTERS = 300


def _docstring_arguments(docstring: str) -> dict[str, str]:
    """The description of each argument in the Args section of a Google-style docstring.

    A "Defaults to ..." sentence is left out: the schema carries the default, and where the
    indicators of a tool disagree, the description lists each one.

    Args:
        docstring (str): The cleaned docstring.

    Returns:
        dict[str, str]: Per argument name, its description on one line.
    """
    section = re.search(
        r"^Args:\s*\n(.*?)(?=^\S|\Z)", docstring, flags=re.MULTILINE | re.DOTALL
    )
    if section is None:
        return {}

    arguments: dict[str, str] = {}
    name = ""
    for line in section.group(1).splitlines():
        match = re.match(r"^\s{2,8}(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", line)
        if match and (not name or len(line) - len(line.lstrip()) <= 8):  # noqa: PLR2004
            name = match.group(1)
            arguments[name] = match.group(2).strip()
        elif name and line.strip():
            arguments[name] += " " + line.strip()

    cleaned: dict[str, str] = {}
    for argument, raw in arguments.items():
        # Up to the end of its sentence: a full stop followed by a capital or the end, so
        # "Defaults to 0.0, i.e. ..." and "Defaults to 1.5." go as a whole.
        text = re.sub(
            r"\s*Defaults to .*?(?:\.(?=\s+[A-Z`(])|\.?$)", "", raw, flags=re.DOTALL
        ).strip()
        # Pointers to the rest of the docstring mean nothing in a tool schema.
        text = re.sub(r",? (?:as )?(?:described|listed|see) above", "", text)
        if len(text) > MAXIMUM_DOCUMENTED_CHARACTERS:
            text = text[: MAXIMUM_DOCUMENTED_CHARACTERS - 3].rsplit(" ", 1)[0] + "..."
        if text:
            cleaned[argument] = text if text.endswith((".", "?", "...")) else f"{text}."

    return cleaned


def _strip_schema_titles(mcp: Any, tool_name: str) -> None:
    """Drop the titles pydantic gives every parameter ("Start Date" for start_date).

    They repeat the parameter name and add about 14,000 characters to the tool listing
    every client reads before its first call.

    Args:
        mcp (Any): The FastMCP server.
        tool_name (str): The tool whose input schema to trim.
    """
    tool_manager = getattr(mcp, "_tool_manager", None)
    tool = tool_manager.get_tool(tool_name) if tool_manager is not None else None
    if tool is None:
        return

    tool.parameters.pop("title", None)
    for schema in tool.parameters.get("properties", {}).values():
        schema.pop("title", None)


def _summarize_docstring(docstring: str) -> dict[str, str]:
    """The first sentence of a docstring and its "Also known as" aliases.

    Args:
        docstring (str): The cleaned docstring of an indicator.

    Returns:
        dict[str, str]: The ``summary`` (at most 200 characters) and the ``aliases``.
    """
    paragraph = " ".join(docstring.split("\n\n", maxsplit=1)[0].split())
    sentence = re.split(r"(?<=[a-z0-9)])\. ", paragraph)[0].rstrip(".")
    summary = (
        sentence
        if len(sentence) <= MAXIMUM_SUMMARY_CHARACTERS
        else sentence[: MAXIMUM_SUMMARY_CHARACTERS - 3].rsplit(" ", 1)[0] + "..."
    )
    aliases = re.search(
        r"Also known as:\s*(.+?)(?:\.\s*$|\.\n|$)", docstring, re.MULTILINE
    )

    return {"summary": summary, "aliases": aliases.group(1).strip() if aliases else ""}


def _unmatched_columns(
    result: object, show_columns: list[str]
) -> tuple[list[str], list[str]]:
    """The requested show_columns a result does not have, and the names it does have.

    The names are the column labels on every level and, unless the rows are dates, the row
    labels of the last index level (the line items of a financial statement).

    Args:
        result (object): The value returned by the indicator, before filtering.
        show_columns (list[str]): The requested names.

    Returns:
        tuple[list[str], list[str]]: The names not found, and the available names.
    """
    tables = (
        [result]
        if isinstance(result, pd.DataFrame | pd.Series)
        else (
            [value for value in result.values() if isinstance(value, pd.DataFrame)]
            if isinstance(result, dict)
            else []
        )
    )
    available: list[str] = []

    for table in tables:
        # The row index is part of every record too, such as a news item's Published Date.
        for name in table.index.names:
            if name is not None and str(name) not in available:
                available.append(str(name))
        axes = [table.columns] if isinstance(table, pd.DataFrame) else []
        if not isinstance(
            table.index.get_level_values(-1), pd.DatetimeIndex | pd.PeriodIndex
        ):
            axes.append(table.index)
        for axis in axes:
            levels = (
                range(axis.nlevels) if axis is not table.index else [axis.nlevels - 1]
            )
            for level in levels:
                for label in axis.get_level_values(level).unique():
                    if str(label) not in available:
                        available.append(str(label))

    if not available:
        return [], []

    return [name for name in show_columns if name not in available], available


def _table_names(method_kwargs: dict[str, Any], count: int) -> list[str]:
    """The names of the tables an indicator returns together, after the option that asked.

    Args:
        method_kwargs (dict[str, Any]): The arguments of the call.
        count (int): The number of tables.

    Returns:
        list[str]: One name per table.
    """
    if method_kwargs.get("include_daily_residuals"):
        names = ["Model", "Daily Residuals"]
    elif method_kwargs.get("show_standard_error"):
        names = ["Price", "Standard Error"]
    else:
        names = ["Result"]

    return [
        *names,
        *(f"Table {number}" for number in range(len(names) + 1, count + 1)),
    ][:count]


def _short_window_reason(category: str, start_date: str, end_date: str) -> str | None:
    """Why a market data request over a few days can be empty: no trading on those days.

    Args:
        category (str): The dispatch category of the call.
        start_date (str): The start of the window.
        end_date (str): The end of the window.

    Returns:
        str | None: The reason, or None when the window is not that short.
    """
    if category not in ("ticker", "toolkit") or not start_date or not end_date:
        return None

    with contextlib.suppress(ValueError):
        days = (
            datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)
        ).days
        if days <= SHORT_WINDOW_DAYS:
            return (
                f"There were no trading days between {start_date} and {end_date} (a "
                "weekend or holiday). Widen the range, e.g. start_date a week earlier."
            )

    return None


def _labels(result: pd.DataFrame | pd.Series) -> set[str]:
    """Every index and column label of a result, on every level, as text."""
    axes = [result.index] + (
        [result.columns] if isinstance(result, pd.DataFrame) else []
    )

    return {
        str(label)
        for axis in axes
        for level in range(axis.nlevels)
        for label in axis.get_level_values(level).unique()
    }


def _mentions_benchmark(result: object) -> bool:
    """Whether a result has a 'Benchmark' row or column, as a benchmark-based metric does.

    Args:
        result (object): The value returned by the indicator.

    Returns:
        bool: True when one of the index or column labels is 'Benchmark'.
    """
    if not isinstance(result, pd.DataFrame | pd.Series):
        return False

    axes = [result.index] + (
        [result.columns] if isinstance(result, pd.DataFrame) else []
    )

    return any(
        "Benchmark" in axis.get_level_values(level).astype(str)
        for axis in axes
        for level in range(axis.nlevels)
    )


def _describe_disputed_default(per_method: dict[str, Any]) -> str:
    """Describe a parameter whose default differs between indicators of a group.

    The tool cannot advertise one default for all of them, so the description has
    to carry what the schema cannot: which indicators require the parameter and
    what each of the others falls back to when it is left unset.

    Args:
        per_method (dict[str, Any]): Per indicator, the default it declares, with
            ``inspect.Parameter.empty`` standing for "required".

    Returns:
        str: A sentence to append to the parameter description.
    """
    required = sorted(
        name
        for name, default in per_method.items()
        if default is inspect.Parameter.empty
    )

    grouped: list[tuple[Any, list[str]]] = []

    for name, default in sorted(per_method.items()):
        if default is inspect.Parameter.empty:
            continue

        for value, names in grouped:
            if type(value) is type(default) and value == default:
                names.append(name)
                break
        else:
            grouped.append((default, [name]))

    grouped.sort(key=lambda item: (-len(item[1]), str(item[0])))

    sentences = ["Leave unset to use the default of the indicator you selected."]

    if required:
        sentences.append(f"Required by: {', '.join(required)}.")

    detail = "; ".join(f"{value!r} for {', '.join(names)}" for value, names in grouped)

    if detail and len(detail) <= _DISPUTED_DEFAULT_DETAIL_LIMIT:
        sentences.append(f"Defaults are {detail}.")
    elif detail:
        sentences.append("Defaults differ between indicators.")

    return " ".join(sentences)


def _simplify_annotation(ann: Any) -> Any:
    """Reduce complex union/collection annotations to JSON-schema-compatible scalars.

    FastMCP generates "unknown" schema entries for union types that contain
    collections (list, dict, range, ndarray) or mixed numeric types. This
    function strips those parts out and returns the simplest scalar equivalent:

    - ``int | list[int]``                → ``int``
    - ``int | float | None``             → ``float | None``
    - ``float | range | list | None``    → ``float | None``
    - ``list[str] | str | None``         → ``str | None``
    - ``float | list | dict[str,float]`` → ``float``
    """
    if ann is inspect.Parameter.empty or ann is None:
        return str

    origin = typing.get_origin(ann)
    args = typing.get_args(ann)

    # Bare list[T] → T, except a list of objects (option strategy legs), which has no
    # single-value form.
    if origin is list:
        if args and typing.get_origin(args[0]) is dict:
            return ann
        return args[0] if args else str

    # A bare opaque type outside a union falls back to str, as the union case does.
    if origin is None:
        name = getattr(ann, "__name__", "")
        if name in _OPAQUE_TYPES:
            return str

    # Union types (both `X | Y` and `Union[X, Y]`)
    is_union = origin is typing.Union or (
        hasattr(types, "UnionType") and isinstance(ann, types.UnionType)
    )
    if not is_union:
        return ann

    has_none = type(None) in args
    scalars: list[Any] = []
    for a in args:
        if a is type(None):
            continue
        a_origin = typing.get_origin(a)
        # Drop generic collections (list[str], dict[str, float], …)
        if a_origin in (list, dict, set, tuple):
            continue
        # Drop bare opaque types by name
        name = getattr(a, "__name__", "")
        if name in _OPAQUE_TYPES:
            continue
        scalars.append(a)

    if not scalars:
        # A mapping without a single-value form (portfolio weights per ticker) stays one,
        # so a caller can pass {"AAPL": 0.6, "MSFT": 0.4}.
        mapping = next(
            (
                (a if typing.get_args(a) else dict[str, float])
                for a in args
                if typing.get_origin(a) is dict or a is dict
            ),
            None,
        )
        if mapping is not None:
            return mapping | None if has_none else mapping
        return str | None if has_none else str

    # Merge int + float → float (float is the wider type)
    if int in scalars and float in scalars:
        scalars = [x for x in scalars if x is not int]

    scalar = scalars[0]

    # A number that may also be a list of numbers (coupon rates or maturities for a bond
    # sensitivity table, lags for growth) accepts both, as the indicator does.
    if scalar in (int, float) and any(
        a is list
        or (
            typing.get_origin(a) is list
            and all(arg in (int, float) for arg in typing.get_args(a))
        )
        for a in args
    ):
        combined = scalar | list[scalar]
        return combined | None if has_none else combined

    return scalar | None if has_none else scalar


class RouterGroupSpec(NamedTuple):
    """
    Immutable specification for a single router group tool.

    Each instance describes one categorical master tool — the metadata needed
    to discover controller methods, build a dispatcher wrapper, and register
    the tool on the FastMCP instance.

    Args:
        tool_name (str): Unique name for the FastMCP tool (e.g. "get_valuation_ratios").
        display_name (str): Human-readable label shown in tool descriptions.
        module_name (str): Key into module_class_map identifying the controller class.
        category (str): Dispatch category ("ticker", "standalone", "toolkit",
            "discovery", or "mixed").
        collect_method (str | None): Optional name of a collect_* method on the
            controller class whose source is parsed to discover the ordered list of
            get_* methods.
        method_override (list[str] | None): Explicit ordered list of method names.
            When set, method discovery is bypassed entirely.
        method_to_module (dict[str, dict[str, str]] | None): Per-method routing table
            used by Mixed groups, mapping method names to {"module": ..., "category": ...}
            dicts.
        index_category (str | None): Category key used when inserting the tool into
            the tool index. Defaults to module_name when None.
        description (str | None): Optional custom description for the tool. When None,
            a default description is generated from display_name and the method list.
    """

    tool_name: str
    display_name: str
    module_name: str
    category: str
    collect_method: str | None = None
    method_override: list[str] | None = None
    method_to_module: dict[str, dict[str, str]] | None = None
    index_category: str | None = None
    description: str | None = None


def _split_values(value: object) -> list[str]:
    """
    Splits tickers or countries given as a comma-separated string or as a list.

    Args:
        value (object): The value as the client sent it.

    Returns:
        list[str]: The individual values, stripped of surrounding spaces.
    """
    items = value if isinstance(value, list | tuple) else str(value).split(",")

    return [str(item).strip() for item in items if str(item).strip()]


class ToolRegistry:
    """Dynamically builds and registers categorical master tools on a FastMCP instance.

    Reads router-group specifications from ``tool_groups`` in the config dict,
    resolves ``module_class_map`` string paths to actual class objects via
    :meth:`_resolve_class_map`, introspects the relevant Finance Toolkit controller
    classes via :class:`ControllerInspector`, and calls ``mcp.add_tool()`` for each
    group. The resulting tool index can be queried via :meth:`get_tool_index`.
    """

    def __init__(
        self,
        mcp: FastMCP,
        provider: ToolkitProvider,
        inspector: ControllerInspector,
        module_class_map: dict[str, str],
        skip_methods: list[str],
        direct_methods: list[str],
        tool_groups: list[dict[str, Any]],
        blocked_periods: dict[str, list[str]] | None = None,
        method_defaults: dict[str, dict[str, Any]] | None = None,
        countries_optional: list[str] | None = None,
        analytics: UsageAnalytics | None = None,
        max_response_characters: int | None = None,
        full_text_methods: list[str] | None = None,
        tickers_optional: list[str] | None = None,
    ) -> None:
        """Initialise the registry with the FastMCP instance and shared subsystems.

        The ``module_class_map`` values may be either fully-qualified dotted class
        path strings (as read from ``config.yaml``) or already-resolved types.
        String paths are resolved to actual class objects at construction time via
        :meth:`_resolve_class_map`.

        Args:
            mcp (FastMCP): The FastMCP server instance to register tools on.
            provider (ToolkitProvider): Provider used to route actual data calls.
            inspector (ControllerInspector): Inspection helper for building tool
                signatures and discovering group methods.
            module_class_map (dict[str, str]): Mapping of module names to
                fully-qualified class path strings
                (e.g. ``"financetoolkit.ratios.ratios_controller.Ratios"``).
                Values may also be pre-resolved class objects.
            skip_methods (list[str]): List of method names to ignore when discovering
                group methods.
            direct_methods (list[str]): List of method names that should be directly
                exposed as top-level tools instead of via router groups.
            tool_groups (list[dict[str, Any]]): List of router group specifications
                from the config dict.
            blocked_periods (dict[str, list[str]] | None): Per tool, the periods it does not support.
            method_defaults (dict[str, dict[str, Any]] | None): Per method, MCP-only defaults applied
                when the caller leaves a parameter unset, plus an optional lookback_days and
                lookahead_days window used when the caller leaves the dates at the tool default.
            countries_optional (list[str] | None): Methods that accept countries but may be
                called without them; every other method that accepts countries requires them.
            analytics (UsageAnalytics | None): Counts the calls of every tool when the
                server's usage analytics are on. Defaults to None.
            max_response_characters (int | None): The size a response may have; a larger
                one is cut with a note on how to narrow it. Defaults to None (no limit).
            full_text_methods (list[str] | None): Methods whose answer is only of use in full,
                such as an earnings call transcript: they take one ticker, return one item and
                are never cut. Defaults to None.
            tickers_optional (list[str] | None): Toolkit methods that answer the same for any
                ticker (market-wide snapshots) and so may be called without tickers.
                Defaults to None.
        """
        self._mcp = mcp
        self._provider = provider
        self._inspector = inspector
        self._tool_index: dict[str, list[dict[str, str]]] = {}
        self._indicator_index: list[dict[str, str]] = []
        # The module and category of every indicator per tool, for the indicators an
        # econometrics call names as its series.
        self._routes: dict[str, dict[str, tuple[str, str]]] = {}
        self._module_class_map: dict[str, type] = self._resolve_class_map(
            module_class_map
        )
        self._skip_methods = frozenset(skip_methods)
        self._direct_methods = frozenset(direct_methods)
        self._tool_groups = tool_groups
        self._blocked_periods: dict[str, frozenset[str]] = {
            tool: frozenset(periods)
            for tool, periods in (blocked_periods or {}).items()
        }
        self._method_defaults: dict[str, dict[str, Any]] = method_defaults or {}
        self._countries_optional: frozenset[str] = frozenset(countries_optional or [])
        self._analytics = analytics
        self._max_response_characters = max_response_characters
        self._full_text_methods: frozenset[str] = frozenset(full_text_methods or [])
        self._tickers_optional: frozenset[str] = frozenset(tickers_optional or [])

    @staticmethod
    def _resolve_class_map(class_map: dict[str, str]) -> dict[str, type]:
        """Resolve a mapping of module names to fully-qualified class path strings
        into a mapping of module names to actual class objects.

        Each value in ``class_map`` should be a dotted import path such as
        ``"financetoolkit.ratios.ratios_controller.Ratios"``. Values that are
        already class objects (i.e. already resolved) are passed through unchanged.
        Entries whose class path cannot be imported or whose attribute cannot be
        found are skipped with a warning, so a partial map is returned rather than
        raising an exception.

        Args:
            class_map (dict[str, str]): Mapping of short module keys to fully-qualified
                class path strings (or already-resolved types).

        Returns:
            dict[str, type]: Mapping of module keys to resolved class objects.
        """
        resolved: dict[str, type] = {}
        for key, class_path in class_map.items():
            if isinstance(class_path, type):
                resolved[key] = class_path
                continue
            try:
                module_path, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                resolved[key] = getattr(module, class_name)
            except (ImportError, AttributeError, ValueError) as exc:
                logger.warning(
                    "Failed to resolve class '%s' for key '%s': %s",
                    class_path,
                    key,
                    exc,
                )
        return resolved

    def get_tool_index(self) -> dict[str, list[dict[str, str]]]:
        """Return the populated tool index mapping categories to registered tools.

        Returns:
            dict[str, list[dict[str, str]]]: Mapping of category name to a list
                of ``{"tool": str, "description": str}`` dicts for every tool
                registered under that category.
        """
        return self._tool_index

    def get_indicator_index(self) -> list[dict[str, str]]:
        """Return every registered indicator with its tool, summary and aliases.

        Returns:
            list[dict[str, str]]: One entry per indicator with the keys ``tool``,
                ``indicator``, ``summary`` and ``aliases``.
        """
        return self._indicator_index

    def build_router_group_specs(self) -> list[RouterGroupSpec]:
        """Build the list of ``RouterGroupSpec`` objects from the config.

        Reads the ``tool_groups`` list from the config dict and converts each
        entry into a typed ``RouterGroupSpec`` NamedTuple. Groups configured with
        ``use_direct_methods: true`` have their ``method_override`` set to
        the sorted list of ``direct_methods`` from the config.

        Returns:
            list[RouterGroupSpec]: Ordered list of group specifications ready for
                ``_register_router_group``.
        """
        specifications = []
        toolkit_methods = sorted(self._direct_methods)
        for raw in self._tool_groups:
            method_override = raw.get("method_override")
            if raw.get("use_direct_methods"):
                method_override = toolkit_methods
            specifications.append(
                RouterGroupSpec(
                    tool_name=raw["tool_name"],
                    display_name=raw["display_name"],
                    module_name=raw.get("module_name", ""),
                    category=raw["category"],
                    collect_method=raw.get("collect_method"),
                    method_override=method_override,
                    method_to_module=raw.get("method_to_module"),
                    index_category=raw.get("index_category"),
                    description=raw.get("description"),
                )
            )
        return specifications

    def _collect_series(
        self,
        references: str,
        start_date: str,
        end_date: str,
        quarterly: bool,
        benchmark_ticker: str,
    ) -> pd.DataFrame:
        """
        Retrieves the indicators an econometrics call names as its series, each through
        the tool it belongs to (and so through its cache), and arranges them as series by
        date, see series_model.

        Args:
            references (str): The references, see series_model.parse_series_references.
            start_date (str): The start date of the call.
            end_date (str): The end date of the call.
            quarterly (bool): Whether quarterly data is asked for.
            benchmark_ticker (str): The benchmark of the call.

        Returns:
            pd.DataFrame: The series, a column each, by date.

        Raises:
            ValueError: When a reference names an unknown tool or indicator.
        """
        frames = []
        for reference in series_model.parse_series_references(references):
            tools = (
                [reference.tool]
                if reference.tool
                else [
                    tool
                    for tool, routes in self._routes.items()
                    if reference.indicator in routes
                ]
            )
            if not tools or tools[0] not in self._routes:
                raise ValueError(
                    f"The series '{reference.text}' names no known "
                    f"{'tool' if reference.tool else 'indicator'}. Use search_metrics to "
                    "find an indicator and its tool."
                )
            routes = self._routes[tools[0]]
            if reference.indicator not in routes:
                raise ValueError(
                    f"`{reference.indicator}` is not an indicator of `{tools[0]}` (series "
                    f"'{reference.text}')."
                )
            module, category = routes[reference.indicator]
            by_ticker = category in (
                self._inspector.categories.get("ticker", "ticker"),
                self._inspector.categories.get("toolkit", "toolkit"),
            )
            entities = (
                [entity.upper() for entity in reference.entities]
                if by_ticker
                else [
                    normalize_country_name(entity, country_names())
                    for entity in reference.entities
                ]
            )
            result = self._provider.call_method(
                module_name=module,
                method_name=reference.indicator,
                category=category,
                tickers=entities if by_ticker else None,
                countries=None if by_ticker else entities,
                start_date=start_date,
                end_date=end_date,
                quarterly=quarterly,
                benchmark_ticker=benchmark_ticker,
            )
            resolved = reference._replace(entities=entities)
            frames.append((resolved, series_model.to_series_columns(result, resolved)))
        return series_model.combine_series(frames)

    def _build_router_wrapper(
        self,
        spec: RouterGroupSpec,
        cls: type | None,
        group_methods: list[str],
        extra_params: list[inspect.Parameter],
        method_to_cls: dict[str, type] | None = None,
        method_dispatch: dict[str, tuple[str, str]] | None = None,
    ) -> Any:
        """Build the ``**kwargs`` callable that powers a router group tool.

        Constructs a closure (``wrapper``) that dispatches to the correct
        Finance Toolkit method based on the ``indicator`` keyword argument, coerces
        inputs, calls the provider, and formats the result as Markdown. The
        closure's ``__signature__`` is replaced with a proper FastMCP-compatible
        signature derived from the group's common and extra parameters.

        Args:
            spec (RouterGroupSpec): Specification for the router group being built.
            cls (type | None): Primary controller class for the group. ``None``
                for ``Mixed`` groups where ``method_to_cls`` is used instead.
            group_methods (list[str]): Ordered list of indicator method names for
                this group.
            extra_params (list[inspect.Parameter]): Extra method-specific parameters
                to append to the tool signature beyond the common params.
            method_to_cls (dict[str, type] | None): Per-method class override used
                for ``Mixed`` groups.
            method_dispatch (dict[str, tuple[str, str]] | None): Per-method
                ``(module_name, category)`` routing table for ``Mixed`` groups.

        Returns:
            Any: A callable with a replaced ``__signature__`` ready to be passed
                to ``mcp.add_tool()``.
        """
        tool_name = spec.tool_name
        module_name = spec.module_name
        category = spec.category
        inspector = self._inspector
        provider = self._provider
        blocked_periods_for_tool = self._blocked_periods.get(tool_name, frozenset())
        max_response_characters = self._max_response_characters

        def owning_class(method_name: str) -> type:
            # Dispatch groups map each method to its controller; plain groups share
            # one class. A method with neither is a registry configuration error.
            owner = (method_to_cls.get(method_name) if method_to_cls else None) or cls

            if owner is None:
                raise ValueError(
                    f"No controller class is registered for method {method_name!r} "
                    f"of tool {tool_name!r}."
                )

            return owner

        method_param_names = {
            m: inspector.get_method_param_names(owning_class(m), m)
            for m in group_methods
        }
        # The order of every indicator's own parameters, so the inputs it requires are named
        # as its signature lists them rather than in the order the group first saw them.
        method_param_order = {
            m: list(inspect.signature(getattr(owning_class(m), m)).parameters)
            for m in group_methods
        }
        param_meta = [(p.name, p.annotation, p.default) for p in extra_params]
        all_indicators = group_methods

        # Parameters the indicators in this group disagree about, or that some of them require. One schema default cannot describe all of them, so they are advertised as unset and only forwarded when the caller actually supplies a value, leaving each indicator's own default to apply.  # noqa: E501
        parameter_defaults = inspector.collect_group_parameter_defaults(
            cls, group_methods, spec.collect_method, method_to_cls
        )
        # An MCP-only default (config.yaml method_defaults) replaces the indicator's own one, and its parameter is advertised as unset like a disputed one: a schema default would always arrive as an explicit value and so never let the MCP default apply.  # noqa: E501
        mcp_default_params: set[str] = set()
        for method_name in group_methods:
            for pname, value in self._method_defaults.get(
                f"{tool_name}.{method_name}", self._method_defaults.get(method_name, {})
            ).items():
                if method_name in parameter_defaults.get(pname, {}):
                    parameter_defaults[pname][method_name] = value
                    mcp_default_params.add(pname)
        disputed_params = (
            inspector.find_disputed_parameters(parameter_defaults) | mcp_default_params
        )

        def wrapper(**kwargs):
            """
            Dispatch a router group tool call to the correct Finance Toolkit method.

            Resolves the indicator name, coerces all typed parameters, validates
            required inputs (tickers, period), routes the call through the provider,
            and returns a formatted Markdown string suitable for LLM consumption.
            """
            raw_indicator = kwargs.pop("indicator", None)
            if not raw_indicator:
                return (
                    f"Please specify an `indicator`. "
                    f"Available for `{tool_name}`: "
                    + ", ".join(f"`{m}`" for m in all_indicators)
                )
            candidate = raw_indicator
            if (
                not candidate.startswith("get_")
                and f"get_{candidate}" in method_param_names
            ):
                candidate = f"get_{candidate}"
            # A near-certain match (get_gordon_growth_model for get_gorden_growth_model, a
            # spelling the package keeps for compatibility) is used, with a note.
            indicator_note = None
            if candidate not in method_param_names:
                near = difflib.get_close_matches(
                    candidate, all_indicators, n=2, cutoff=NEAR_CERTAIN_MATCH
                )
                if len(near) == 1:
                    indicator_note = f"`{raw_indicator}` was read as `{near[0]}`."
                    candidate = near[0]
            if candidate not in method_param_names:
                elsewhere = [
                    entry["tool"]
                    for entry in self._indicator_index
                    if entry["indicator"] == candidate
                ]
                if elsewhere:
                    return (
                        f"`{candidate}` is not part of `{tool_name}`; it is an indicator of "
                        + " and ".join(f"`{tool}`" for tool in elsewhere)
                        + f". Please call that tool with `indicator='{candidate}'`."
                    )
                close = difflib.get_close_matches(
                    candidate, all_indicators, n=3, cutoff=0.6
                )
                suggestion = (
                    (" Did you mean: " + ", ".join(close) + "?") if close else ""
                )
                return (
                    f"Unknown indicator `{raw_indicator}` for `{tool_name}`."
                    f"{suggestion}\n"
                    "Available: "
                    + ", ".join(all_indicators)
                    + ". Use `search_metrics` "
                    "to find an indicator by what it measures."
                )
            method_name = candidate

            raw_tickers = kwargs.pop("tickers", None)
            tickers = (
                [t.strip().upper() for t in _split_values(raw_tickers)]
                if raw_tickers
                else None
            )
            raw_countries = kwargs.pop("countries", None)
            # Short forms and any casing ("USA", "eurozone") become the names the sources
            # use, before a source is asked for them by name.
            countries = (
                [
                    normalize_country_name(country, country_names())
                    for country in _split_values(raw_countries)
                ]
                if raw_countries
                else None
            )

            # A market-wide snapshot (config.yaml tickers_optional) does not depend on the
            # tickers, but the Toolkit it is reached through needs one to be built.
            if not tickers and method_name in self._tickers_optional:
                tickers = ["SPY"]

            # An econometrics call can analyze the indicators of other tools instead of
            # the prices of its tickers.
            raw_series = (
                kwargs.pop("series", None) if module_name == "econometrics" else None
            )
            changes = (
                kwargs.pop("changes", None) if module_name == "econometrics" else None
            ) or "percentage"

            # Return an actionable error rather than a confusing AttributeError later.
            effective_category = category
            if method_dispatch and method_name in method_dispatch:
                _, effective_category = method_dispatch[method_name]
            if (
                effective_category
                in (
                    inspector.categories.get("ticker", "ticker"),
                    inspector.categories.get("toolkit", "toolkit"),
                )
                and not tickers
                and not raw_series
            ):
                return (
                    f"`{tool_name}` (`{method_name}`) requires a `tickers` parameter. "
                    "Please provide one or more ticker symbols, e.g. `tickers='AAPL'` "
                    "or `tickers='AAPL,MSFT'`."
                )

            # Tickers on a tool that works with countries would otherwise be dropped as an
            # unknown parameter and every country returned.
            if (
                tickers
                and not countries
                and effective_category
                == inspector.categories.get("standalone", "standalone")
            ):
                return (
                    f"`{tool_name}` (`{method_name}`) works with countries, not tickers. "
                    "Please provide `countries`, e.g. `countries='United States'` or "
                    "`countries='Germany,Japan'`."
                )

            # A date that was given but cannot be read is an error, not a reason to quietly
            # answer for the default range instead.
            for date_parameter in ("start_date", "end_date"):
                given = kwargs.get(date_parameter)
                if given and not validation_model.is_valid_date(
                    str(given).replace("/", "-")
                ):
                    return (
                        f"Invalid `{date_parameter}` for `{tool_name}` (`{method_name}`): "
                        f"'{given}'. Please use a date written as YYYY-MM-DD, e.g. '2026-01-31'."
                    )

            # The range as given, for the discovery calendars that only take one when asked.
            given_dates = {
                name: str(kwargs.get(name)).replace("/", "-")
                for name in ("start_date", "end_date")
                if kwargs.get(name)
            }
            quarterly = to_boolean(kwargs.pop("quarterly", False))
            start_date = validate_date(
                kwargs.pop("start_date", inspector.start_date) or inspector.start_date,
                default_date=(
                    datetime.now() - timedelta(days=90 * 5 if quarterly else 365 * 5)
                ).strftime("%Y-%m-%d"),
            )
            end_date = validate_date(
                kwargs.pop("end_date", inspector.end_date) or inspector.end_date,
                default_date=datetime.now().strftime("%Y-%m-%d"),
            )
            benchmark_ticker = kwargs.pop(
                "benchmark_ticker", inspector.benchmark_ticker
            )

            # show_columns is handled entirely here — never forwarded to methods
            raw_show_columns = kwargs.pop("show_columns", None)
            show_columns = (
                [c.strip() for c in str(raw_show_columns).split(",") if c.strip()]
                if raw_show_columns
                else None
            )

            accepted_params = method_param_names.get(method_name, set())

            # Without countries a country indicator returns every country it has, often sixty
            # or more columns, which is rarely what was asked and costs many tokens.
            if (
                not countries
                and "countries" in accepted_params
                and method_name not in self._countries_optional
            ):
                return (
                    f"`{tool_name}` (`{method_name}`) requires a `countries` parameter. "
                    "Please provide one or more countries, e.g. `countries='United States'` "
                    "or `countries='Germany,Japan'`."
                )

            # limit selects rows like head and tail: 5 for the first five, -5 for the last
            # five. An indicator with a limit of its own (e.g. the largest ETF holdings or
            # the length of a list) receives it instead, when it supports the value.
            raw_limit = kwargs.pop("limit", None)
            try:
                rows_limit = int(raw_limit) if raw_limit not in (None, "") else None
            except (TypeError, ValueError):
                return (
                    f"Invalid input for `{tool_name}` (`{method_name}`): limit must be a "
                    f"whole number, e.g. 5 for the first five rows or -5 for the last "
                    f"five, not {raw_limit!r}."
                )
            row_selection = None

            method_kwargs = {}
            if rows_limit is not None:
                if "limit" in method_param_names.get(method_name, set()) and (
                    rows_limit > 0 or method_name in NEGATIVE_LIMIT_METHODS
                ):
                    method_kwargs["limit"] = rows_limit
                else:
                    row_selection = rows_limit

            # Arguments the indicator does not take (another indicator of the tool does) are
            # named in the answer, so a caller does not believe they were used.
            ignored_params: list[str] = []
            given_params: set[str] = set()
            for pname, pann, pdefault in param_meta:
                if pname in kwargs:
                    val = kwargs.pop(pname)
                    advertised = (
                        None
                        if pname in disputed_params
                        else (
                            pdefault if pdefault is not inspect.Parameter.empty else ""
                        )
                    )
                    if val not in (None, "") and val != advertised:
                        given_params.add(pname)
                    if pname not in accepted_params:
                        if pname in given_params:
                            ignored_params.append(pname)
                        continue
                    # A disputed parameter is advertised as unset, so an unset value means "no preference" and must reach the indicator's own default rather than another indicator's.  # noqa: E501
                    if pname in disputed_params and val in (None, ""):
                        continue
                    method_kwargs[pname] = coerce_value(val, pann)

            # Discovery takes a range only when one is given (see provider_model).
            if effective_category == inspector.categories.get("discovery", "discovery"):
                start_date = given_dates.get("start_date", "")
                end_date = given_dates.get("end_date", "")

            # MCP-only defaults (config.yaml method_defaults) keep a broad request small: they
            # apply only to what the caller left unset, so an explicit value always wins. The
            # dates count as unset when both are still the tool defaults. A "tool.method" key
            # applies to that tool only, for a method name two tools share.
            method_defaults = self._method_defaults.get(
                f"{tool_name}.{method_name}", self._method_defaults.get(method_name, {})
            )
            if (
                "lookback_days" in method_defaults
                or "lookahead_days" in method_defaults
            ) and (start_date, end_date) in (
                (inspector.start_date, inspector.end_date),
                ("", ""),
            ):
                today = datetime.now()
                start_date = (
                    today - timedelta(days=int(method_defaults.get("lookback_days", 0)))
                ).strftime("%Y-%m-%d")
                end_date = (
                    today
                    + timedelta(days=int(method_defaults.get("lookahead_days", 0)))
                ).strftime("%Y-%m-%d")
            for pname, value in method_defaults.items():
                if (
                    pname not in ("lookback_days", "lookahead_days")
                    and pname in accepted_params
                    and method_kwargs.get(pname) in (None, "")
                ):
                    method_kwargs[pname] = value

            # A curve given as {"10": 0.043} is keyed by maturity in years, as a number; a
            # label such as "10Y" or "6M" is passed on for the indicator to read.
            curve_note = None
            for curve in CURVE_PARAMETERS:
                given = method_kwargs.get(curve)
                if isinstance(given, dict):
                    try:
                        method_kwargs[curve] = {
                            _curve_maturity(maturity): float(rate)
                            for maturity, rate in given.items()
                        }
                    except (TypeError, ValueError):
                        return (
                            f"Invalid `{curve}` for `{tool_name}` (`{method_name}`): give the "
                            "rate per maturity in years, e.g. {'1': 0.042, '10': 0.043}."
                        )
            if missing_curves := [
                curve
                for curve in CURVE_PARAMETERS
                if curve in accepted_params
                and method_kwargs.get(curve) in (None, "", {})
                and method_name not in MARKET_CURVE_METHODS
            ]:
                curve_note = (
                    f"{', '.join(f'`{curve}`' for curve in missing_curves)} "
                    f"{'was' if len(missing_curves) == 1 else 'were'} not given, so a fixed "
                    "sample curve was used for illustration, not market data. Pass a "
                    "current zero-coupon (spot) curve from the rates tool, e.g. "
                    "get_bank_of_england_yield_curve or get_hqm_corporate_bond_yield_curve "
                    "with rate='spot'; get_treasury_rates is a par curve."
                )

            # The economic inputs of a calculation that the caller left to their defaults
            # (a bond priced at an 8% yield) are named, as they decide the answer.
            assumed_inputs = [
                f"{pname}={parameter_defaults[pname][method_name]}"
                for pname in ASSUMED_INPUTS
                if pname in accepted_params
                and pname not in given_params
                and parameter_defaults.get(pname, {}).get(method_name)
                not in (None, inspect.Parameter.empty, "")
            ]

            # An indicator that needs inputs the caller left out is answered with which ones,
            # rather than with the TypeError calling it would raise.
            signature_order = method_param_order.get(method_name, [])
            missing_inputs = sorted(
                (
                    pname
                    for pname, per_method in parameter_defaults.items()
                    if per_method.get(method_name, None) is inspect.Parameter.empty
                    and method_kwargs.get(pname) in (None, "")
                ),
                key=lambda pname: (
                    signature_order.index(pname)
                    if pname in signature_order
                    else len(signature_order)
                ),
            )
            if missing_inputs:
                listed = ", ".join(f"`{pname}`" for pname in missing_inputs)
                return (
                    f"`{tool_name}` (`{method_name}`) requires {listed}. Please provide "
                    f"{'it' if len(missing_inputs) == 1 else 'them'}; the tool's parameter "
                    "descriptions explain each one."
                )

            # A transcript is of use only in full, and a full one is about as large as an
            # answer may be, so these return one item of one ticker and are never cut.
            full_text = method_name in self._full_text_methods
            if full_text:
                if tickers and len(tickers) > 1:
                    return (
                        f"`{tool_name}` (`{method_name}`) returns the full text for one ticker "
                        f"at a time. Please call it once per ticker, e.g. `tickers='{tickers[0]}'`."
                    )
                if method_kwargs.get("latest") is False:
                    return (
                        f"`{tool_name}` (`{method_name}`) returns one full text at a time. "
                        "Please pick one with `fiscal_period`, e.g. `fiscal_period='2025Q4'`; "
                        "without it, the most recent one is returned."
                    )
                if method_kwargs.get("fiscal_period") not in (None, ""):
                    try:
                        parse_fiscal_period(method_kwargs["fiscal_period"])
                    except ValueError as exc:
                        return (
                            f"Invalid input for `{tool_name}` (`{method_name}`): {exc}"
                        )

            if kwargs:
                logger.warning(
                    "Unknown parameter(s) passed to %s (%s) and ignored: %s",
                    tool_name,
                    method_name,
                    ", ".join(kwargs.keys()),
                )

            # A period the toolkit does not know would surface as a confusing missing-item
            # error from deep inside the calculation.
            if method_kwargs.get("period") not in (None, ""):
                known_periods = (
                    INTRADAY_PERIODS if method_name == "get_intraday_data" else PERIODS
                )
                if str(method_kwargs["period"]).lower() not in known_periods:
                    return (
                        f"Invalid `period` for `{tool_name}` (`{method_name}`): "
                        f"'{method_kwargs['period']}'. Please use one of: "
                        f"{', '.join(known_periods)}."
                    )

            # Validate that the requested period (if any) is not blocked for this tool
            # A rolling window reads the regular data of the period, which every period
            # has, so the block only applies to results within each period.
            if (
                blocked_periods_for_tool
                and "period" in method_kwargs
                and not method_kwargs.get("rolling")
                and method_name not in PERIOD_BLOCK_EXEMPT
            ):
                requested_period = str(method_kwargs["period"]).lower()
                if requested_period in blocked_periods_for_tool:
                    allowed = [
                        period
                        for period in (
                            "daily",
                            "weekly",
                            "monthly",
                            "quarterly",
                            "yearly",
                        )
                        if period not in blocked_periods_for_tool
                    ]
                    return (
                        f"`{tool_name}` (`{method_name}`) does not support "
                        f"`period='{requested_period}'` without a `rolling` window. "
                        f"Please use one of: {', '.join(allowed)}, or set `rolling`."
                    )

            if method_dispatch and method_name in method_dispatch:
                dispatch_module, dispatch_category = method_dispatch[method_name]
            else:
                dispatch_module, dispatch_category = module_name, category

            try:
                call_started = time.perf_counter()
                # The warnings and errors logged during the call explain an empty result,
                # e.g. a missing API key, so they are passed on rather than lost.
                with capture_call_messages() as call_messages:
                    if raw_series:
                        result = provider.call_econometrics_with_data(
                            method_name,
                            self._collect_series(
                                raw_series,
                                start_date=start_date,
                                end_date=end_date,
                                quarterly=quarterly,
                                benchmark_ticker=benchmark_ticker,
                            ),
                            changes=changes,
                            **method_kwargs,
                        )
                    else:
                        result = provider.call_method(
                            module_name=dispatch_module,
                            method_name=method_name,
                            category=dispatch_category,
                            tickers=tickers,
                            countries=countries,
                            start_date=start_date,
                            end_date=end_date,
                            quarterly=quarterly,
                            benchmark_ticker=benchmark_ticker,
                            **method_kwargs,
                        )
                    input_info = call_input_info()
                # Per-call timing at debug level, so a "tool X is slow" report can be
                # diagnosed from the logs alone (a first call pays for data collection,
                # a repeat call should be near-instant off the provider cache).
                logger.debug(
                    "Tool %s (%s) took %.0f ms for %d ticker(s)",
                    tool_name,
                    method_name,
                    (time.perf_counter() - call_started) * 1000,
                    len(tickers) if tickers else 0,
                )
                # A result and its detail (a model and its residuals, prices and their
                # standard errors) become named tables rather than one unnamed list.
                if isinstance(result, tuple) and all(
                    isinstance(part, pd.DataFrame | pd.Series) for part in result
                ):
                    result = dict(
                        zip(
                            _table_names(method_kwargs, len(result)),
                            result,
                            strict=False,
                        )
                    )
                unmatched_columns: list[str] = []
                if show_columns is not None:
                    # Matched in any casing ("close" for Close), as the names are shown
                    # in a response and typed back by hand.
                    _, available_columns = _unmatched_columns(result, show_columns)
                    by_lowercase = {name.lower(): name for name in available_columns}
                    show_columns = [
                        (
                            name
                            if name in available_columns
                            else by_lowercase.get(name.lower(), name)
                        )
                        for name in show_columns
                    ]
                    unmatched_columns, available_columns = _unmatched_columns(
                        result, show_columns
                    )
                    result = _filter_columns(result, show_columns)
                notes: list[str] = []
                if dispatch_category in ("ticker", "toolkit") and tickers:
                    notes = provider.get_transformation_notes(
                        tickers=tickers,
                        start_date=start_date,
                        end_date=end_date,
                        quarterly=quarterly,
                        benchmark_ticker=benchmark_ticker,
                        api_key=provider._api_key,
                        fred_api_key=provider._fred_api_key,
                    )
                # Tickers left out of an otherwise answered call are named, e.g. a symbol
                # the source does not know.
                notes = list(notes) + missing_data_notes(call_messages)
                # A ticker the sources do not know is dropped from the answer; when the
                # others are there, say which one is missing rather than leave it out quietly.
                if (
                    tickers
                    and len(tickers) > 1
                    and isinstance(result, pd.DataFrame | pd.Series)
                    # A regression's dependent ticker (or a pair's) is no row of the answer.
                    and not any(
                        pname.endswith(("_ticker", "_tickers"))
                        for pname in method_kwargs
                    )
                ):
                    # An empty column of its own counts as missing too.
                    if isinstance(result, pd.DataFrame) and result.columns.nlevels == 1:
                        empty = [
                            ticker
                            for ticker in tickers
                            if ticker in result.columns and result[ticker].isna().all()
                        ]
                        if len(empty) < len(result.columns):
                            result = result.drop(columns=empty)
                    labels = _labels(result)
                    absent = [
                        ticker
                        for ticker in tickers
                        if ticker not in labels
                        and not any(ticker in note for note in notes)
                    ]
                    if absent and len(absent) < len(tickers):
                        notes.append(
                            f"No data is available for {', '.join(absent)}, so "
                            f"{'it is' if len(absent) == 1 else 'they are'} left out. "
                            "Please check the symbol (e.g. ASML.AS for Amsterdam)."
                        )
                # After the check for missing tickers, which needs every row of a column.
                if row_selection and isinstance(result, dict):
                    result = {
                        name: (
                            select_rows(table, row_selection)
                            if isinstance(table, pd.DataFrame | pd.Series)
                            else table
                        )
                        for name, table in result.items()
                    }
                # A plain list, such as the expiration dates of an option chain.
                if row_selection and isinstance(result, list | tuple):
                    result = (
                        list(result[:row_selection])
                        if row_selection > 0
                        else list(result[row_selection:])
                    )
                if row_selection and isinstance(result, pd.DataFrame | pd.Series):
                    # One row per ticker (a figure over the whole period) has no first or
                    # last rows to speak of; cutting it would only drop tickers.
                    row_labels = {
                        str(label)
                        for label in result.index.get_level_values(0).unique()
                    }
                    # Metrics as rows and a column per ticker (a profile, the share
                    # float) have no first or last rows either.
                    column_labels = (
                        {str(label) for label in result.columns}
                        if isinstance(result, pd.DataFrame)
                        and result.columns.nlevels == 1
                        else set()
                    )
                    per_ticker_columns = (
                        bool(tickers)
                        and bool(column_labels)
                        and column_labels <= {*(tickers or []), "Benchmark"}
                        and not is_time_index_of(result)
                        # Named rows (an ETF's Sector) are a list a limit can shorten.
                        and result.index.name is None
                    )
                    if per_ticker_columns:
                        notes.append(
                            f"limit={row_selection} was not applied: the answer has one "
                            "column per ticker rather than a row per period; use "
                            "show_columns to pick fields."
                        )
                    elif (
                        tickers
                        and row_labels <= {*tickers, "Benchmark"}
                        and len(result.index) == len(row_labels)
                        and not (
                            isinstance(result, pd.DataFrame)
                            and has_period_columns(result)
                        )
                    ):
                        notes.append(
                            f"limit={row_selection} was not applied: the answer has one row "
                            "per ticker rather than one per period."
                        )
                    else:
                        reported = (
                            result.index[result.notna().any(axis=1)]
                            if isinstance(result, pd.DataFrame)
                            else result.index
                        )
                        selected = select_rows(result, row_selection)
                        if (
                            row_selection < 0
                            and isinstance(result, pd.DataFrame)
                            and has_period_columns(result)
                            and result.columns[-1] not in selected.columns
                        ):
                            skipped = result.columns[
                                result.columns.get_loc(selected.columns[-1]) + 1 :
                            ]
                            notes.append(
                                f"{', '.join(map(str, skipped))} "
                                f"{'is' if len(skipped) == 1 else 'are'} left out: most rows "
                                "have no value for it yet. Use start_date/end_date to see it."
                            )
                        result = selected
                        # The latest period may not be reported yet for every ticker; a row
                        # without any value at all is not worth a mention.
                        if (
                            isinstance(result, pd.DataFrame)
                            and has_period_columns(result)
                            and (
                                empty_rows := result.index[
                                    result.isna().all(axis=1)
                                    & result.index.isin(reported)
                                ]
                            ).size
                        ):
                            names = [
                                (
                                    " ".join(map(str, label))
                                    if isinstance(label, tuple)
                                    else str(label)
                                )
                                for label in empty_rows.unique()
                            ]
                            notes.append(
                                (
                                    f"{', '.join(names)} "
                                    f"{'has' if len(names) == 1 else 'have'} no "
                                    if len(names) <= MAXIMUM_NAMED_ROWS
                                    else f"{len(names):,} rows have no "
                                )
                                + "value for the selected periods (e.g. not reported yet); "
                                f"limit={row_selection - 1 if row_selection < 0 else row_selection + 1} "
                                "includes one more period."
                            )
                # Most economic series are yearly only, whatever quarterly asks for.
                if (
                    quarterly
                    and isinstance(result, pd.DataFrame | pd.Series)
                    and is_yearly(result.index)
                ):
                    notes.append(
                        f"Quarterly data is not available for `{method_name}`, so the yearly "
                        "figures are shown."
                    )
                if indicator_note:
                    notes.insert(0, indicator_note)
                if curve_note:
                    notes.append(curve_note)
                if assumed_inputs:
                    notes.append(
                        f"Assumed {', '.join(assumed_inputs)} (the "
                        f"{'default' if len(assumed_inputs) == 1 else 'defaults'}); pass "
                        "your own values for the bond in question."
                    )
                # The inputs an option model used, which it only logs.
                if method_kwargs.get("show_input_info"):
                    notes.extend(input_info)
                if ignored_params:
                    notes.append(
                        ", ".join(f"`{pname}`" for pname in ignored_params)
                        + f" {'is' if len(ignored_params) == 1 else 'are'} not used by "
                        f"`{method_name}` and {'was' if len(ignored_params) == 1 else 'were'} "
                        "ignored."
                    )
                if unmatched_columns:
                    notes.append(
                        "show_columns: "
                        + ", ".join(f"'{name}'" for name in unmatched_columns)
                        + " is not part of this answer. Available: "
                        + ", ".join(available_columns[:40])
                        + "."
                    )
                # A note on how the benchmark was chosen only confuses an answer that has
                # no benchmark in it, such as an ETF's holdings.
                if not _mentions_benchmark(result):
                    notes = [note for note in notes if "the benchmark" not in note]
                formatted = format_result(
                    result,
                    notes=notes or None,
                    reason=summarize_reasons(call_messages)
                    or _short_window_reason(dispatch_category, start_date, end_date),
                    max_characters=None if full_text else max_response_characters,
                )
                return formatted
            except (ValueError, KeyError) as exc:
                return (
                    f"Invalid input for `{tool_name}` (`{method_name}`): {redact(exc)}"
                )
            except TypeError as exc:
                return f"Parameter error for `{tool_name}` (`{method_name}`): {redact(exc)}"
            except ConnectionError as exc:
                return f"API connection failed: {redact(exc)}"
            except Exception as exc:
                logger.warning(
                    "Tool %s (%s) failed: %s",
                    tool_name,
                    method_name,
                    exc,
                    exc_info=True,
                )
                return (
                    f"`{tool_name}` (`{method_name}`) failed with error: "
                    f"{type(exc).__name__}: {redact(exc)}"
                )

        P = inspect.Parameter
        POS = P.POSITIONAL_OR_KEYWORD
        indicator_choices = group_methods
        # The choices are advertised as an enum, but any text is accepted: an indicator of
        # another tool (or a near miss) is then answered with where it is, rather than with
        # a validation error that only lists this tool's indicators.
        indicator_ann = (
            Annotated[
                str, PydanticField(json_schema_extra={"enum": list(indicator_choices)})
            ]
            if indicator_choices
            else str
        )
        indicator_default = P.empty
        indicator_param = P(
            "indicator",
            POS,
            default=indicator_default,
            annotation=_annotate_with_description("indicator", indicator_ann),
        )

        mixed_cats = None
        if category == inspector.categories["mixed"] and method_dispatch:
            mixed_cats = {cat for _, cat in method_dispatch.values()}

        common = inspector.build_common_signature_params(
            category, mixed_categories=mixed_cats
        )
        sig_params = [indicator_param] + [
            P(
                p.name,
                POS,
                default=p.default,
                annotation=_annotate_with_description(p.name, p.annotation),
            )
            for p in common
        ]

        # What the indicators' own docstrings say about each parameter, for the ones
        # without a shared description.
        versions: dict[str, dict[str, list[str]]] = {}
        for method_name in group_methods:
            owner = owning_class(method_name)
            for pname, text in _docstring_arguments(
                inspect.getdoc(getattr(owner, method_name, None)) or ""
            ).items():
                versions.setdefault(pname, {}).setdefault(text, []).append(method_name)
        # A parameter that means something else per indicator (performance's method is a
        # regression method for one and a Sharpe ratio variant for another) says so.
        documented: dict[str, str] = {
            pname: (
                " ".join(
                    f"For {', '.join(methods)}: {text}"
                    for text, methods in texts.items()
                )
                if 1 < len(texts) <= MAXIMUM_DESCRIPTION_VERSIONS
                # Many versions: the shortest is usually the generic one.
                else min(texts, key=len)
            )
            for pname, texts in versions.items()
        }

        for p in extra_params:
            ann = (
                _simplify_annotation(p.annotation)
                if p.annotation is not P.empty
                else str
            )

            if p.name in disputed_params:
                # Advertised as unset with the per-indicator defaults spelled out, because no single value is correct for the whole group.  # noqa: E501
                default = None
                annotation = _annotate_with_description(
                    p.name,
                    typing.Optional[ann],  # noqa: UP045
                    suffix=_describe_disputed_default(parameter_defaults[p.name]),
                    documented=documented.get(p.name, ""),
                )
            else:
                default = p.default if p.default is not P.empty else ""
                annotation = _annotate_with_description(
                    p.name, ann, documented=documented.get(p.name, "")
                )

            sig_params.append(P(p.name, POS, default=default, annotation=annotation))

        if module_name == "econometrics":
            sig_params.extend(
                P(
                    name,
                    POS,
                    default=None,
                    annotation=_annotate_with_description(name, str | None),
                )
                for name in ("series", "changes")
            )

        # Appended last so it does not disturb positional ordering of other params.
        sig_params.append(
            P(
                "show_columns",
                POS,
                default=None,
                annotation=_annotate_with_description("show_columns", str | None),
            )
        )

        # Every tool can select rows; a group whose indicator has a limit of its own already
        # lists it.
        if "limit" not in {p.name for p in sig_params}:
            sig_params.append(
                P(
                    "limit",
                    POS,
                    default=None,
                    annotation=_annotate_with_description("limit", int | None),
                )
            )

        wrapper.__signature__ = inspect.Signature(sig_params, return_annotation=str)
        wrapper.__annotations__ = {p.name: p.annotation for p in sig_params}
        wrapper.__annotations__["return"] = str

        return wrapper

    def _register_router_group(self, spec: RouterGroupSpec) -> bool:
        """Build and register a single router group tool on the FastMCP instance.

        Resolves the controller class(es) for the group, discovers or applies the
        method list, builds the router wrapper via ``_build_router_wrapper``, and
        calls ``mcp.add_tool()``. Also populates the internal tool index.

        Args:
            spec (RouterGroupSpec): Specification for the router group to register.

        Returns:
            bool: ``True`` if the tool was successfully registered, ``False`` if
                the group was skipped due to a missing class, empty method list, or
                an exception during tool construction.
        """
        method_to_cls = None
        method_dispatch = None
        cls = None

        if spec.category == self._inspector.categories["mixed"]:
            if not spec.method_to_module:
                logger.warning(
                    f"Mixed group '{spec.tool_name}' has no method_to_module, skipping"
                )
                return False
            method_to_cls = {}
            method_dispatch = {}
            for mname, info in spec.method_to_module.items():
                mod = info.get("module", "")
                cat = info.get("category", "")
                target_cls = self._module_class_map.get(mod)
                if target_cls is None:
                    logger.warning(
                        f"No class for module '{mod}' in mixed group '{spec.tool_name}' — skipping method"
                    )
                    continue
                method_to_cls[mname] = target_cls
                method_dispatch[mname] = (mod, cat)
            group_methods = (
                list(spec.method_override)
                if spec.method_override is not None
                else sorted(method_to_cls.keys())
            )
            extra_params = self._inspector.collect_group_extra_params(
                None, group_methods, spec.collect_method, method_to_cls=method_to_cls
            )
        else:
            cls = self._module_class_map.get(spec.module_name)
            if cls is None:
                logger.warning(
                    "No class found for module '%s' — skipping", spec.module_name
                )
                return False
            group_methods = (
                list(spec.method_override)
                if spec.method_override is not None
                else self._inspector.discover_group_methods(
                    cls, spec.collect_method, self._skip_methods
                )
            )
            extra_params = self._inspector.collect_group_extra_params(
                cls, group_methods, spec.collect_method
            )
            method_to_cls = {m: cls for m in group_methods}

        if not group_methods:
            logger.warning(f"No methods found for group '{spec.tool_name}' — skipping")
            return False

        self._routes[spec.tool_name] = {
            method_name: (method_dispatch or {}).get(
                method_name, (spec.module_name, spec.category)
            )
            for method_name in group_methods
        }

        # Each indicator by name, the first sentence of its docstring and its aliases, so
        # search_metrics can point at the indicator itself rather than only its tool.
        for method_name in group_methods:
            owner = (method_to_cls or {}).get(method_name) or cls
            self._indicator_index.append(
                {
                    "tool": spec.tool_name,
                    "indicator": method_name,
                    **_summarize_docstring(
                        inspect.getdoc(getattr(owner, method_name, None)) or ""
                    ),
                }
            )

        method_list = ", ".join(group_methods)
        # The indicators are the enum of the indicator parameter, so the tool description
        # does not repeat them; the category listing (search_by_category) does show them.
        description = spec.description or f"{spec.display_name}."
        index_description = f"{description}\n\nAvailable indicators: {method_list}."

        try:
            fn = self._build_router_wrapper(
                spec,
                cls,
                group_methods,
                extra_params,
                method_to_cls=method_to_cls,
                method_dispatch=method_dispatch,
            )
            fn.__name__ = spec.tool_name
            fn.__doc__ = description

            if self._analytics is not None:
                fn = self._analytics.counted(spec.tool_name, fn)

            self._mcp.add_tool(
                fn,
                name=spec.tool_name,
                description=description,
                annotations=ToolAnnotations(
                    title=spec.display_name,
                    readOnlyHint=True,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
            )
            index_cat = spec.index_category or spec.module_name
            if index_cat not in self._tool_index:
                self._tool_index[index_cat] = []
            self._tool_index[index_cat].append(
                {"tool": spec.tool_name, "description": index_description}
            )
            _strip_schema_titles(self._mcp, spec.tool_name)
            return True
        except Exception as exc:
            logger.warning(f"Failed to register group '{spec.tool_name}': {exc}")
            return False

    def register_all_tools(self) -> int:
        """Register every router group defined in the config.

        Iterates over all ``RouterGroupSpec`` objects produced by
        ``build_router_group_specs()`` and registers each via
        ``_register_router_group()``. Logs the outcome of each group.

        Returns:
            int: Total number of router group tools successfully registered.
        """
        total = 0
        for spec in self.build_router_group_specs():
            # Each group becomes a single router tool, so this is a success flag rather than a count; failures
            # are already logged as a warning by _register_router_group.
            registered = self._register_router_group(spec)
            if registered:
                logger.info(
                    f"Registered router group '{spec.tool_name}' ({spec.module_name})"
                )
            total += registered
        logger.info(f"Total master tools registered: {total}")
        return total
