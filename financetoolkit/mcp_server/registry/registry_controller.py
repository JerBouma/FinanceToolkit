"""
Dynamic tool registration for the Finance Toolkit MCP Server — Router Pattern.

Consolidates the indicators (1100+ metrics with their method= variants) into categorical
master tools, each accepting an indicator parameter (Literal enum) that routes to the
correct underlying method.
"""

from __future__ import annotations

import contextlib
import difflib
import functools
import importlib
import inspect
import json
import math
import os
import re
import time
import traceback
import types
import typing
from collections import Counter
from datetime import datetime, timedelta
from typing import Annotated, Any, NamedTuple

import pandas as pd
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field as PydanticField,
)

from financetoolkit import currencies_model
from financetoolkit.cache import policy_model
from financetoolkit.fmp_model import log_server_errors_once, parse_fiscal_period
from financetoolkit.helpers import rank_across_tickers
from financetoolkit.mcp_server.analytics.analytics_model import (
    UNKNOWN_INDICATOR,
    UsageAnalytics,
    record_outcome,
    run_in_thread,
)
from financetoolkit.mcp_server.inspection.inspection_controller import (
    ControllerInspector,
)
from financetoolkit.mcp_server.registry import series_model
from financetoolkit.mcp_server.registry.coercion_model import (
    accepts_text_only,
    coerce_value,
    refused_value_problem,
    size_limit_problem,
    text_from_parsed_json,
    to_boolean,
    validate_date,
)
from financetoolkit.mcp_server.registry.diagnostics_model import (
    FMP_KEY_HINT,
    FRED_KEY_HINT,
    call_input_info,
    capture_call_messages,
    current_call_messages,
    missing_data_notes,
    summarize_reasons,
)
from financetoolkit.mcp_server.registry.formatting_model import (
    SAMPLE_PERIOD_NAMES,
    SAMPLE_PERIODS,
    convert_to_period,
    country_names,
    filled_day_notes,
    format_result,
    has_period_columns,
    is_time_index_of,
    is_yearly,
    observed_period,
    parse_base_window,
    rebase_result,
    sample_rows,
    select_rows,
    signal_rows,
    skipped_periods_note,
)
from financetoolkit.mcp_server.registry.provider_model import (
    FALLBACK_BENCHMARKS,
    ToolkitProvider,
    resolve_benchmark_ticker,
)
from financetoolkit.utilities import validation_model
from financetoolkit.utilities.dataframe_model import filter_columns as _filter_columns
from financetoolkit.utilities.logger_model import get_logger, redact
from financetoolkit.utilities.sources_model import collect_sources, readable_sources
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


# The indicators that read the returns of the period itself rather than results within each
# period, so a blocked period (e.g. daily performance) does not apply to them.
PERIOD_BLOCK_EXEMPT = frozenset(
    {
        "get_style_analysis",
        "get_beta_implied_return",
        "get_return_statistics",
        "get_return_concentration",
        "get_strategy_risk",
        "get_outperformance_probability",
        "get_returns",
    }
)
# The indicators whose own limit also takes a negative number, for the last rows.
NEGATIVE_LIMIT_METHODS = frozenset({"get_holdings"})


# The most transactions a portfolio_performance call takes: every one is priced and valued
# per day, and a portfolio passed by hand rarely has more.
MAX_TRANSACTIONS = 1000

# Injected into each tool's Args docstring so FastMCP can surface them.
_PARAM_DESCRIPTIONS: dict[str, str] = {
    "indicator": (
        "Name of the specific metric to calculate, one of the listed values, or a list of up "
        "to five of them, each answered in its own table. Required — search_metrics finds "
        "the indicator (and its tool) for a metric by name."
    ),
    "tickers": "Comma-separated ticker symbols, e.g. 'AAPL,MSFT,GOOGL'.",
    "countries": (
        "Country names, comma-separated or as a list, e.g. 'United States,Germany,Japan'."
    ),
    "start_date": (
        "Start of the date range in YYYY-MM-DD format; left empty, five years before today "
        "(or the indicator's own window)."
    ),
    "end_date": (
        "End of the date range in YYYY-MM-DD format; left empty, today (or the indicator's own window)."
    ),
    "quarterly": "Return quarterly data instead of annual when True.",
    "benchmark_ticker": (
        "Ticker used as the market benchmark, e.g. 'SPY' or '^GSPC'; 'None' for no benchmark. "
        "Left empty, SPY, or QQQ when SPY is one of the tickers."
    ),
    "risk_free_rate": (
        "The US Treasury yield used as the risk-free rate: '13w' (the 13-week bill), '5y', '10y' (default) or '30y'."
    ),
    "basket": (
        "Also analyse the tickers as one company, 'Total': its statements are the sums of "
        "theirs and its ratios are calculated on those sums (per-share figures stay empty)."
    ),
    "growth": "Return period-over-period growth rates instead of absolute values.",
    "lag": "Number of periods to lag when computing growth rates.",
    "compound": (
        "With growth=true, return the compound annual growth rate (CAGR) over the lag "
        "instead of the total change, e.g. lag=5 for the five-year CAGR (lag=20 with "
        "quarterly=true). Missing when either value is zero or negative."
    ),
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
    "metrics": (
        "get_factor_analysis only: the metric(s) to analyse, a method name such as "
        "'get_price_to_earnings_ratio', a comma-separated list, or a collection of the tool, "
        "e.g. 'profitability'."
    ),
    "metric": (
        "get_indicator_summary only: the indicator to summarise, e.g. "
        "'get_relative_strength_index' (default) or 'get_stochastic_oscillator'."
    ),
    "below": (
        "get_indicator_summary only: the line the indicator falls below, e.g. 30 for an "
        "oversold RSI; adds the periods below it, the crossings into that zone and the "
        "returns after them."
    ),
    "above": (
        "get_indicator_summary only: the line the indicator rises above, e.g. 70 for an overbought RSI."
    ),
    "forward_periods": (
        "get_indicator_summary only: the trading periods after each crossing to measure the "
        "return over, e.g. [5, 21, 63] for about a week, a month and a quarter."
    ),
    "signals_only": (
        "get_signals, get_candlestick_patterns, get_williams_fractal and get_liquidity_sweep "
        "only: set to true to return only the periods with a signal, pattern, fractal or sweep, "
        "e.g. with limit=-5 for the last five."
    ),
    "show_crossings": (
        "get_indicator_summary only: set to true for a row per crossing (date, value, its "
        "percentile at the time and the forward returns) instead of the summary."
    ),
    "detail": (
        "get_factor_analysis only: returns one part for a single metric instead of the "
        "summary: 'information_coefficient', 'information_coefficient_by_month', "
        "'information_coefficient_by_year', 'group_information_coefficient', "
        "'quantile_returns', 'quantile_returns_by_date', 'quantile_return_statistics', "
        "'quantile_statistics', 'quantile_turnover', 'rank_autocorrelation', "
        "'factor_weights', 'factor_returns', 'cumulative_returns' or 'event_returns'."
    ),
    "group": (
        "get_factor_analysis only: 'sector' or 'industry' (from the company profiles) to "
        "analyse within groups: the group_information_coefficient detail, binning='group' "
        "and, with demeaned=true, a group-neutral analysis."
    ),
    "demeaned": (
        "get_factor_analysis only: set to true to compare the forward returns with the mean "
        "of the tickers in each period, or with group= with the mean of their group."
    ),
    "binning": (
        "get_factor_analysis only: how the tickers are sorted into quantiles: 'quantile' "
        "(default, equally sized), 'equal_width', 'zero_aware' (negative and positive "
        "values apart; an even number of quantiles) or 'group' (within each group)."
    ),
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
    "lags": "The number of lags (past periods) the test or model uses, e.g. 1 or 5.",
    "column": (
        "The price column to use, e.g. 'Close' or 'Adj Close'; for get_factor_analysis the "
        "output of a metric with several, e.g. 'Signal Line'."
    ),
    "strike_price_range": (
        "Only strikes within this fraction of the current stock price, e.g. 0.05 for the strikes within 5% of it."
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
        "statements, ratios) keeps its first or last periods instead, and rows per ticker (an "
        "earnings calendar) are selected per ticker. For ETF holdings the "
        "largest (or, negative, the smallest) by weight, and for a list (e.g. news or a "
        "screener) its length. 0 returns every row."
    ),
    "sample": (
        "'weekly', 'monthly', 'quarterly' or 'yearly': keeps the last row of each calendar "
        "week, month, quarter or year of a daily or intraday answer (dates as columns: the "
        "last column), including the latest period that has not ended, with the real dates "
        "of those rows. Unlike period, which resamples the prices first (a 200-day average "
        "on weekly prices is a 200-week one), the indicator is calculated on daily data and "
        "read once per period, e.g. window=200 with sample='weekly' for the 200-day average "
        "every week. Applied before limit, so limit=-10 gives the last ten periods."
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
    "functions": (
        "Functions of government (COFOG), comma-separated, e.g. 'Defence,Health,Education'; "
        "'Public Debt Transactions' or 'Total' also work. Defaults to the ten divisions."
    ),
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
        ":column and :name=value arguments of the indicator (e.g. "
        "'rates:get_outstanding_mortgage_rates:United States:Below 4%:cumulative=true'), "
        "e.g. 'valuation:get_price_to_earnings_ratio:AAPL,MSFT; "
        "macroeconomics:get_inflation_rate:United States' or "
        "'rates:get_government_bond_yield_curve:United States,Germany:10Y'. Every ticker or "
        "country becomes a series named after it, which dependent_ticker and "
        "independent_tickers then name. Use start_date for a long enough history. "
        "period='daily' reads the ratios of statement items (P/E, P/B) on every trading "
        "day, each period's figure from its fiscal period end; quarterly=true uses the "
        "quarterly statements."
    ),
    "changes": (
        "With series: how their changes (the 'Return' column the regressions and "
        "diagnostics use) are calculated: 'percentage' (default), 'log', 'difference' (for "
        "rates and ratios) or 'none' (the series are changes already). The series "
        "themselves are the 'Adj Close' column the unit root and cointegration tests use."
    ),
    "deflate_with": (
        "With series: the country whose consumer price index the indicators that take "
        "deflate_with are deflated with, e.g. 'United States' for "
        "'macroeconomics:get_commodity_prices:Gold' in real terms."
    ),
    "rank": (
        "true ranks the tickers (or countries) within every period, from the highest value "
        "(rank 1) to the lowest, and compares each with its neighbour toward the middle of "
        "the ranking: the Difference (in points for returns) and the Multiple (only when "
        "both have the same sign), so the first and last rank show how far the best and the "
        "worst stand apart, e.g. a fall more than twice as large as the next one. Needs a "
        "value per period and ticker; pick one field with show_columns where there are more."
    ),
    "rebase_to": (
        "Index a time series to 100 at its average over a base period: every column (a "
        "ticker, country or series) is divided by its mean over the rows in the window "
        "and multiplied by 100, e.g. '1340-1347' for 100 = the 1340-1347 average. A year "
        "'2020', a month '2020-03', a day '2020-03-31', a range 'START:END' in those "
        "forms ('2020-01:2020-06') or a range of years 'YYYY-YYYY' (a second part of four "
        "digits is a year, of two digits a month). A period row counts when any of its "
        "days lies in the window. Applied before limit, show_columns and rank, on all "
        "rows from start_date to end_date: a window before start_date needs an earlier "
        "start_date. Skipped with growth=true or standardize=true; meant for levels "
        "(prices, wages, indices), not for returns or rates."
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

# What the tool schemas say about a parameter whose full description above is long: one
# short sentence. The server instructions explain the parameters every tool has (limit,
# rank, rebase_to, sample, show_columns) and search_metrics with an indicator's name gives
# the full description of each parameter it takes.
_SCHEMA_PARAM_DESCRIPTIONS: dict[str, str] = {
    "indicator": (
        "The indicator, or a list of up to five; search_metrics with its name gives its parameters."
    ),
    "tickers": "Comma-separated tickers, e.g. 'AAPL,MSFT'.",
    "start_date": "YYYY-MM-DD; empty for the default range (often five years).",
    "end_date": "YYYY-MM-DD; empty for today.",
    "quarterly": "true for quarterly instead of annual data.",
    "benchmark_ticker": "SPY by default; 'None' for no benchmark.",
    "risk_free_rate": "The Treasury yield used as the risk-free rate.",
    "basket": "true adds the tickers together as one company, 'Total'.",
    "compound": "With growth=true, the compound annual growth rate over lag periods.",
    "standardize": "Return Z-scores of the values (of the growth with growth=true).",
    "cumulative": "Compound the values over time, starting at 1.",
    "rolling": "A rolling value over this many periods instead of one per period.",
    "trailing": "Sum over the trailing N periods (4 quarters for a TTM figure).",
    "threshold_percentile": "With method='evt', the loss percentile of the fit.",
    "minimum_acceptable_return": "The return below which returns count as downside.",
    "metrics": "The metric(s) to analyse: a name, a list or a collection.",
    "metric": "The indicator to summarise, e.g. 'get_relative_strength_index'.",
    "below": "The line the indicator falls below, e.g. 30 for an oversold RSI.",
    "above": "The line the indicator rises above, e.g. 70 for an overbought RSI.",
    "forward_periods": "Periods after each crossing to measure the return over.",
    "signals_only": "true keeps only the periods with a signal.",
    "show_crossings": "true lists every crossing instead of the summary.",
    "detail": "One part of the analysis of a single metric instead of the summary.",
    "group": "'sector' or 'industry', to analyse within groups.",
    "demeaned": "true compares the returns with the mean of the tickers (or group).",
    "binning": "How the tickers are sorted into quantiles.",
    "period": "'daily', 'weekly', 'monthly', 'quarterly' or 'yearly'.",
    "gmdb_source": "true uses the Global Macro Database instead of the OECD.",
    "annualize": "true annualizes the ratio.",
    "primary_only": "true keeps only the most traded listing of each company.",
    "fiscal_period": "The company's fiscal quarter, e.g. '2025Q4'; empty for the latest.",
    "column": "The price column to use, e.g. 'Close'.",
    "strike_price_range": "Strikes within this fraction of the price, e.g. 0.05.",
    "legs": (
        "The legs, e.g. [{'strike_price': 340, 'put_option': false, 'position': 'long', "
        "'premium': 4.2}]; a stock leg has 'instrument': 'stock'."
    ),
    "limit": "-5 for the last five rows, 5 for the first five, 0 for all.",
    "sample": "The last row of each 'weekly', 'monthly', 'quarterly' or 'yearly' period.",
    "gmdb_forecasts": "true adds the years the Global Macro Database projects.",
    "usd": "true gives levels in millions of US dollars.",
    "functions": "Functions of government (COFOG), e.g. 'Defence,Health'.",
    "government.level": "'consolidated', 'general' or 'central' government.",
    "series": (
        "Indicators of other tools instead of prices: tool:indicator:tickers, ';'-separated."
    ),
    "changes": "With series: 'percentage', 'log', 'difference' or 'none'.",
    "deflate_with": "With series: the country whose CPI deflates the indicators.",
    # Parameters whose name alone reads several ways.
    "risk.weights": "Weight per ticker, e.g. {'AAPL': 0.6, 'MSFT': 0.4}; empty for equal weights.",
    "portfolio.weights": "Weight per ticker, e.g. {'AAPL': 0.6, 'MSFT': 0.4}; empty for equal weights.",
    "current_weights": "The weights held now, per ticker, that turnover is measured from.",
    "show_daily": "true gives a value per trading day (limit=-1 for the latest).",
    "diluted": "true uses the diluted share count.",
    "valuation.include_dividends": "true deducts preferred dividends from net income.",
    "options.volatility": "The annual volatility to use; empty for that of the past year.",
    "put_option": "true for puts, false for calls.",
    "risk.alpha": "The tail probability, e.g. 0.05 for the worst 5% (95% confidence).",
    "portfolio.alpha": "The tail probability, e.g. 0.05 for the worst 5% (95% confidence).",
    "performance.alpha": "The tail probability, e.g. 0.05 for the worst 5% (95% confidence).",
    "econometrics.alpha": "The tail probability or size of the test, e.g. 0.05.",
    "risk.measure": "The range: 'range', 'var', 'cvar', 'tail_gini' or 'evar'.",
    "jobs.measure": "The life table column, e.g. 'death_probability' or 'life_expectancy'.",
    "risk.region": "The region of the Fama-French factors, e.g. 'Developed'.",
    "portfolio.region": "The region of the Fama-French factors, e.g. 'Developed'.",
    "performance.region": "The region of the Fama-French factors, e.g. 'Developed'.",
    "rates.maturity": "true per maturity, false per rating.",
    "rank": "true ranks the tickers or countries per period.",
    "rebase_to": "Index to 100 at a base period, e.g. '2020' or '2019-2021'.",
    "show_columns": "Comma-separated fields to keep, e.g. 'Close,Volume'.",
}

# The parameters every indicator tool has, which the server instructions explain.
UNIVERSAL_PARAMETERS = (
    "indicator",
    "rank",
    "rebase_to",
    "sample",
    "show_columns",
    "limit",
)


# The inputs that describe the portfolio of the portfolio_performance tool. They replace the
# shared descriptions there: its start_date only applies to holdings, weights and presets,
# and its weights are a portfolio rather than the weights of the portfolio tool.
_PORTFOLIO_PARAM_DESCRIPTIONS: dict[str, str] = {
    "transactions": (
        "The trades of the portfolio as a list, e.g. [{'date': '2023-01-03', 'ticker': "
        "'AAPL', 'volume': 10, 'price': 125.1}, {'date': '2024-06-03', 'ticker': 'AAPL', "
        "'volume': -5, 'price': 194.0}]. volume is negative for a sale; price (for every "
        "trade or none: the close of the day), costs (negative) and currency are optional. "
        f"At most {MAX_TRANSACTIONS} trades."
    ),
    "holdings": (
        "The shares held per ticker, e.g. {'AAPL': 10, 'MSFT': 5} or 'AAPL:10, MSFT:5', "
        "bought at the close of start_date."
    ),
    "weights": (
        "The target weight per ticker, e.g. {'VTI': 0.6, 'BND': 0.4} or 'VTI:0.6, BND:0.4': "
        "initial_value is invested on start_date and brought back to these weights as "
        "rebalancing says, with the dividends reinvested."
    ),
    "preset": (
        "A classic portfolio by name, e.g. '60/40', 'All Weather', 'Permanent', 'Golden "
        "Butterfly' or 'Three Fund' (get_presets lists them all)."
    ),
    "start_date": (
        "When holdings, weights or a preset are invested, YYYY-MM-DD; left empty, ten years "
        "ago for weights and presets and five years ago for holdings. Transactions start on "
        "their first date."
    ),
    "initial_value": "The amount invested in weights or a preset. Defaults to 10,000.",
    "rebalancing": (
        "How often weights or a preset return to their weights: 'none', 'monthly', "
        "'quarterly' or 'yearly'. Defaults to quarterly (a preset: its own frequency)."
    ),
    "currency": (
        "The currency the portfolio is valued in, e.g. 'USD' or 'EUR'; prices in other "
        "currencies are converted at each day's rate. Defaults to the currency most of the "
        "portfolio trades in."
    ),
}


# The most indicators a parameter's schema description names; beyond it, it names the first
# ones and counts the rest.
MAXIMUM_NAMED_INDICATORS = 2

# The longest shared description the schema gives whole.
MAXIMUM_SCHEMA_DESCRIPTION_CHARACTERS = 100


def _annotate_with_description(
    name: str, ann: Any, description: str | None = None
) -> Any:
    """Wrap *ann* in ``Annotated[ann, Field(description=...)]`` for JSON schema output.

    Pydantic reads the ``Field(description=...)`` metadata and includes it in the
    ``properties[name].description`` field of the generated JSON schema, which is
    what Smithery (and other MCP clients) use to display parameter descriptions.

    Args:
        name (str): The parameter name, used to look up the shared description.
        ann (Any): The annotation to wrap.
        description (str | None): A description that replaces the shared one, for a
            parameter that means something else in one tool (the start_date of a
            Portfolio) or an indicator's own parameter; empty for none.

    Returns:
        Any: The annotated type carrying the description metadata.
    """
    desc = (
        description
        if description is not None
        else _SCHEMA_PARAM_DESCRIPTIONS.get(name) or _PARAM_DESCRIPTIONS.get(name, "")
    ) or None

    # A text argument that reads as JSON reaches validation as a list or object, which
    # is turned back into text (see text_from_parsed_json).
    if accepts_text_only(ann):
        return Annotated[  # type: ignore[valid-type]
            ann,
            PydanticField(description=desc),
            BeforeValidator(text_from_parsed_json),
        ]

    return Annotated[ann, PydanticField(description=desc)]  # type: ignore[valid-type]


# A parameter at least this share of a tool's indicators take is named once per tool in the
# search results instead of in each indicator's call line, for a tool of at least this many.
COMMON_PARAMETER_SHARE = 2 / 3
MINIMUM_INDICATORS_FOR_COMMON_PARAMETERS = 3

# The length of an indicator's summary in the search results.
MAXIMUM_SUMMARY_CHARACTERS = 200
# The description search_metrics gives for an indicator's name, in characters.
MAXIMUM_DESCRIPTION_CHARACTERS = 2500
# The sections that end a docstring's description.
DESCRIPTION_SECTION = re.compile(
    r"(Args|Returns|Notes|Raises|Examples?|Referenced against|Shares similar logic|Also known as"
    r"|For more information|Which returns|As an example)\b"
)


# A window of at most this many days can be empty for lack of trading days alone.
SHORT_WINDOW_DAYS = 3

# The most tickers or countries in one call: each is collected separately, and a request for
# hundreds would hold a worker for minutes on a server many people share.
MAX_ENTITIES = 50

# The most indicators of one tool in one call: each is calculated and answered in its own
# table, so a call holds a worker (and the answer the size of a response) for all of them.
MAX_INDICATORS = 5


def _pseudo_parameters(
    tickers: list[str] | None, start_date: str, end_date: str
) -> dict[str, float]:
    """
    The pseudo parameters of the size limits (coercion_model.PSEUDO_PARAMETERS): the number
    of tickers of a call and the years from its start date to its end date (today when it has
    none). A value that is not known is left out, which leaves out the limits that use it.
    """
    pseudo: dict[str, float] = {}
    if tickers:
        pseudo["tickers"] = len(tickers)
    with contextlib.suppress(ValueError, TypeError):
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
        pseudo["range_years"] = max((end - start).days, 0) / 365.25
    return pseudo


# The parameters a series reference cannot set itself: the econometrics call sets them for
# every reference from its own parameters.
SERIES_TOOL_PARAMETERS = frozenset(
    {
        "tickers",
        "countries",
        "start_date",
        "end_date",
        "quarterly",
        "benchmark_ticker",
        "period",
        "show_daily",
        "show_columns",
        "as_series",
        "deflate_with",
    }
)

# A part of a series reference written as an argument with capitals (Window=10).
MISCASED_ARGUMENT = re.compile(r"\s*(?P<name>[A-Za-z_]\w*)\s*=")

# How alike an unknown indicator name must be to a known one to be read as it.
NEAR_CERTAIN_MATCH = 0.9

# The note a call answered by another tool carries there (an indicator asked of a tool that
# doesn't have it), passed under a name no tool parameter has.
ROUTED_NOTE = "__routed_note__"

# The size the answer of one indicator may have in a call of several, which share the size of
# a response; passed under a name no tool parameter has.
ANSWER_SIZE = "__answer_size__"


def _indicator_count(arguments: dict[str, Any]) -> int:
    """The indicators a call asks for, which the usage statistics count as a call each; a
    call of more than MAX_INDICATORS is answered with that limit, and counts once.

    Args:
        arguments (dict[str, Any]): The arguments of the call.

    Returns:
        int: The number of different indicators, at least one.
    """
    count = len(set(_split_values(arguments.get("indicator") or "")))

    return count if 1 <= count <= MAX_INDICATORS else 1


def _combine_answers(answers: dict[str, str]) -> str:
    """The answers of several indicators of one call as one answer: each under its name,
    with the notes and sources they all have once, after them.

    Args:
        answers (dict[str, str]): The answer of each indicator, by the name it was asked as.

    Returns:
        str: The answer as JSON. An indicator answered with a message instead of data (an
            unknown name, a missing input) has it under "error".
    """
    # A call that fails as a whole (no API key, no tickers) gives every indicator the same
    # message, which is then given once.
    if len(answers) > 1 and len(set(answers.values())) == 1:
        text = next(iter(answers.values()))
        try:
            json.loads(text)
        except ValueError:
            return text

    combined: dict[str, Any] = {}
    for name, text in answers.items():
        try:
            answer = json.loads(text)
        except ValueError:
            answer = {"error": text}
        combined[name] = answer if isinstance(answer, dict) else {"data": answer}

    entries = list(combined.values())
    for key in ("_notes", "_sources"):
        lists = [answer.get(key) or [] for answer in entries]
        shared = [
            item for item in lists[0] if all(item in other for other in lists[1:])
        ]
        if not shared:
            continue
        for answer in entries:
            if own := [item for item in answer.pop(key) if item not in shared]:
                answer[key] = own
        combined[key] = shared

    return json.dumps(combined, default=str, separators=(",", ":"))


# How a portfolio_performance call names the columns of a transaction, in any casing.
TRANSACTION_FIELDS = {
    "date": "Date",
    "ticker": "Ticker",
    "symbol": "Ticker",
    "volume": "Volume",
    "quantity": "Volume",
    "shares": "Volume",
    "price": "Price",
    "costs": "Costs",
    "cost": "Costs",
    "currency": "Currency",
}
REBALANCING_FREQUENCIES = ("none", "monthly", "quarterly", "yearly")
# A ticker as the sources write it ("AAPL", "ASML.AS", "BTC-USD", "EURUSD=X", "^GSPC"), checked
# before it becomes part of a web address.
TICKER_PATTERN = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=^]{0,19}$")
CURRENCY_PATTERN = re.compile(r"^[A-Za-z]{3}$")


# The most rows a note names one by one; more are counted instead.
MAXIMUM_NAMED_ROWS = 10

# The periods the toolkit resamples to, and the intervals of intraday data.
PERIODS = ("daily", "weekly", "monthly", "quarterly", "yearly", "annual")
INTRADAY_PERIODS = ("1min", "5min", "15min", "30min", "1hour")


def _normalise_period(period: Any) -> str:
    """The period in lower case, with "annual" read as "yearly"."""
    period = str(period).strip().lower()

    return "yearly" if period == "annual" else period


# The longest parameter description taken from a docstring.
MAXIMUM_DOCUMENTED_CHARACTERS = 300


def _docstring_arguments(
    docstring: str,
    maximum_characters: int | None = MAXIMUM_DOCUMENTED_CHARACTERS,
    keep_defaults: bool = False,
) -> dict[str, str]:
    """The description of each argument in the Args section of a Google-style docstring.

    A "Defaults to ..." sentence is left out: the schema carries the default, and where the
    indicators of a tool disagree, the description lists each one.

    Args:
        docstring (str): The cleaned docstring.
        maximum_characters (int | None, optional): The longest description, cut at a word
            with "..." beyond it. Defaults to MAXIMUM_DOCUMENTED_CHARACTERS; None keeps
            every description whole.
        keep_defaults (bool, optional): Whether to keep the "Defaults to ..." sentence.
            Defaults to False.

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
    # Every argument starts at the indentation of the first, so a continuation line that
    # reads "are: ..." stays part of the description before it.
    indentation = None
    for line in section.group(1).splitlines():
        match = re.match(r"^\s{2,8}(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", line)
        if match and len(line) - len(line.lstrip()) == (
            indentation if indentation is not None else len(line) - len(line.lstrip())
        ):
            indentation = len(line) - len(line.lstrip())
            name = match.group(1)
            arguments[name] = match.group(2).strip()
        elif name and line.strip():
            arguments[name] += " " + line.strip()

    cleaned: dict[str, str] = {}
    for argument, raw in arguments.items():
        # Up to the end of its sentence: a full stop followed by a capital or the end, so
        # "Defaults to 0.0, i.e. ..." and "Defaults to 1.5." go as a whole.
        text = (
            raw.strip()
            if keep_defaults
            else re.sub(
                r"\s*Defaults to .*?(?:\.(?=\s+[A-Z`(])|\.?$)", "", raw, flags=re.DOTALL
            ).strip()
        )
        # Pointers to the rest of the docstring mean nothing in a tool schema.
        text = re.sub(r",? (?:as )?(?:described|listed|see) above", "", text)
        if maximum_characters and len(text) > maximum_characters:
            text = text[: maximum_characters - 3].rsplit(" ", 1)[0] + "..."
        if text:
            cleaned[argument] = text if text.endswith((".", "?", "...")) else f"{text}."

    return cleaned


def keep_unknown_arguments(mcp: Any, tool_name: str) -> None:
    """Pass the arguments that are no parameter of a tool on to the tool.

    The MCP SDK validates the arguments with a model of the tool's parameters that drops
    any other argument, so a misspelt parameter (`perid`) would be ignored without a word
    and the answer would read as one for the given value. Kept, the tool names them in
    its notes. The input schema the clients read is built when the tool is added and
    does not change.

    Args:
        mcp (Any): The FastMCP server.
        tool_name (str): The tool whose argument model to extend.
    """
    tool_manager = getattr(mcp, "_tool_manager", None)
    tool = tool_manager.get_tool(tool_name) if tool_manager is not None else None
    metadata = getattr(tool, "fn_metadata", None)
    if metadata is None:
        return

    arguments = metadata.arg_model

    class ArgumentsWithUnknown(arguments):  # type: ignore[valid-type, misc]
        # The title keeps the name a validation error starts with ("riskArguments").
        model_config = ConfigDict(
            **{**arguments.model_config, "extra": "allow", "title": arguments.__name__}
        )

        def model_dump_one_level(self) -> dict[str, Any]:
            return {**super().model_dump_one_level(), **(self.model_extra or {})}

    metadata.arg_model = ArgumentsWithUnknown


def unknown_parameter_notes(
    tool_name: str, unknown: list[str], parameters: list[str]
) -> list[str]:
    """Name the arguments that are no parameter of a tool, with the parameter meant.

    Args:
        tool_name (str): The tool that was called.
        unknown (list[str]): The arguments that are no parameter of the tool.
        parameters (list[str]): The parameters of the tool.

    Returns:
        list[str]: A note per unknown argument, with the closest parameter when one is
            near enough to be what was meant (`perid` for `period`).
    """
    notes = []
    for pname in unknown:
        near = difflib.get_close_matches(pname, parameters, n=1, cutoff=0.8)
        notes.append(
            f"`{pname}` is not a parameter of `{tool_name}` and was ignored"
            + (f"; did you mean `{near[0]}`?" if near else ".")
        )

    return notes


def name_unknown_arguments(method: Any, tool_name: str) -> Any:
    """Let a tool that answers in text take arguments it has no parameter for.

    With `keep_unknown_arguments` the MCP SDK passes such an argument on, which a method
    without `**kwargs` would reject. The method is called with its own parameters only, and
    the answer names the others, as the indicator tools do in their notes.

    Args:
        method (Any): The tool's method, which returns text.
        tool_name (str): The name the tool is registered under.

    Returns:
        Any: The tool, with the method's signature.
    """
    parameters = list(inspect.signature(method).parameters)

    @functools.wraps(method)
    def tool(**kwargs: Any) -> str:
        notes = unknown_parameter_notes(
            tool_name,
            [pname for pname in kwargs if pname not in parameters],
            parameters,
        )
        answer = method(
            **{pname: value for pname, value in kwargs.items() if pname in parameters}
        )

        return answer + "".join(f"\n\nNote: {note}" for note in notes)

    return tool


def _strip_schema_titles(mcp: Any, tool_name: str) -> None:
    """Drop the titles pydantic gives every parameter ("Start Date" for start_date), and the
    null option and null default of a parameter that may be left out.

    The titles repeat the parameter name and the null options say what leaving a parameter
    out says already (none is required but the indicator), while together they add about a
    third to the tool listing every client reads before its first call. The server still
    accepts null for them.

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
        if "default" in schema and schema["default"] is None:
            del schema["default"]
        options = [
            option for option in schema.get("anyOf", []) if option != {"type": "null"}
        ]
        if len(options) == 1:
            del schema["anyOf"]
            schema.update(options[0])
        elif options:
            schema["anyOf"] = options


def _offer_indicator_list(mcp: Any, tool_name: str) -> None:
    """Let the indicator parameter of a tool take a list of its indicators as well as one.

    The names are listed once, under $defs, and both forms refer to them, so the list form
    adds a few dozen characters to the tool listing rather than every name again.

    Args:
        mcp (Any): The FastMCP server.
        tool_name (str): The tool whose input schema to extend.
    """
    tool_manager = getattr(mcp, "_tool_manager", None)
    tool = tool_manager.get_tool(tool_name) if tool_manager is not None else None
    schema = tool.parameters.get("properties", {}).get("indicator") if tool else None
    if not schema or "enum" not in schema:
        return

    tool.parameters.setdefault("$defs", {})["indicator"] = {
        "type": schema.pop("type", "string"),
        "enum": schema.pop("enum"),
    }
    reference = {"$ref": "#/$defs/indicator"}
    schema["anyOf"] = [
        reference,
        {"type": "array", "items": reference, "maxItems": MAX_INDICATORS},
    ]


# The most characters of one tool in the tool listing (its name, description and input
# schema as JSON), which a client that loads tools on demand reads in full.
MAXIMUM_TOOL_CHARACTERS = 12_000


def _fit_tool_listing(mcp: Any, tool_name: str, parameters: set[str]) -> None:
    """Drop the schema descriptions of indicators' parameters, the longest first and those
    without words of their own before the others, while the tool is larger than
    MAXIMUM_TOOL_CHARACTERS in the tool listing.

    The descriptions that say a parameter is required stay. search_metrics with an
    indicator's name describes each of its parameters in full.

    Args:
        mcp (Any): The FastMCP server.
        tool_name (str): The tool whose input schema to trim.
        parameters (set[str]): The parameters of the tool's indicators, whose descriptions
            may go (not those of the tool, such as the dates).
    """
    tool_manager = getattr(mcp, "_tool_manager", None)
    tool = tool_manager.get_tool(tool_name) if tool_manager is not None else None
    if tool is None:
        return

    properties = tool.parameters.get("properties", {})
    size = len(
        json.dumps(
            {
                "name": tool_name,
                "description": tool.description,
                "inputSchema": tool.parameters,
            }
        )
    )
    # The descriptions that only name indicators go first, as search_metrics gives each
    # indicator's parameters; the few words for a name that reads several ways go last.
    optional = sorted(
        (
            name
            for name in parameters
            if "Required" not in properties.get(name, {}).get("description", "Required")
        ),
        key=lambda name: (
            f"{tool_name}.{name}" in _SCHEMA_PARAM_DESCRIPTIONS
            or name in _SCHEMA_PARAM_DESCRIPTIONS,
            -len(properties[name]["description"]),
        ),
    )
    for name in optional:
        if size <= MAXIMUM_TOOL_CHARACTERS:
            break
        size -= len(json.dumps(properties[name].pop("description"))) + len(
            ', "description": '
        )


# With FT_MCP_COMPACT_TOOLS=1 a tool's description is its first sentence and its parameters
# have no descriptions: a client that loads several tools (an agent searching for its tools
# on every run) reads a few hundred characters per tool instead of tens of thousands.
# search_metrics with an indicator's name gives what it calculates and its parameters.
COMPACT_TOOLS_ENVIRONMENT_VARIABLE = "FT_MCP_COMPACT_TOOLS"


def compact_tools_enabled() -> bool:
    """Whether the tools are listed without their long descriptions.

    Returns:
        bool: True when FT_MCP_COMPACT_TOOLS is 1, true, yes or on.
    """
    return os.environ.get(COMPACT_TOOLS_ENVIRONMENT_VARIABLE, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _compact_description(description: str, first_indicator: str) -> str:
    """The first clause of a tool description and where the rest can be found.

    Args:
        description (str): The full description of the tool.
        first_indicator (str): An indicator of the tool, for the example.

    Returns:
        str: The compact description.
    """
    clause = re.split(r" \(|\. |: ", description, maxsplit=1)[0].rstrip(".")

    return (
        f"{clause}. Pass the indicator by name; search_metrics with an indicator's name "
        f"(e.g. '{first_indicator}') says what it calculates and which parameters it takes."
    )


def _strip_schema_descriptions(mcp: Any, tool_name: str) -> None:
    """Drop the parameter descriptions of a tool, except that of the indicator.

    Args:
        mcp (Any): The FastMCP server.
        tool_name (str): The tool whose input schema to trim.
    """
    tool_manager = getattr(mcp, "_tool_manager", None)
    tool = tool_manager.get_tool(tool_name) if tool_manager is not None else None
    if tool is None:
        return

    for name, schema in tool.parameters.get("properties", {}).items():
        if name != "indicator":
            schema.pop("description", None)


def _summarize_docstring(docstring: str) -> dict[str, str]:
    """The first sentence of a docstring and its "Also known as" aliases.

    Args:
        docstring (str): The cleaned docstring of an indicator.

    Returns:
        dict[str, str]: The ``summary`` (at most 200 characters) and the ``aliases``.
    """
    paragraph = " ".join(docstring.split("\n\n", maxsplit=1)[0].split())
    # "e.g." and "i.e." end no sentence.
    sentence = re.split(r"(?<=[a-z0-9)])(?<!e\.g)(?<!i\.e)\. ", paragraph)[0].rstrip(
        "."
    )
    summary = (
        sentence
        if len(sentence) <= MAXIMUM_SUMMARY_CHARACTERS
        else sentence[: MAXIMUM_SUMMARY_CHARACTERS - 3].rsplit(" ", 1)[0] + "..."
    )
    # The aliases run over as many lines as the width needs, up to the end of their list:
    # a sentence after it ("Note that ...") is no alias, unless it names more ("The
    # variants are also known as ...").
    aliases = re.search(
        r"Also known as:\s*(.+?)\s*(?:\n\s*\n|\Z)", docstring, re.DOTALL
    )
    sentences = (
        re.split(r"(?<=[\w)]{2})\.\s+(?=[A-Z`])", " ".join(aliases.group(1).split()))
        if aliases
        else []
    )

    return {
        "summary": summary,
        "aliases": ". ".join(
            sentence
            for position, sentence in enumerate(sentences)
            if position == 0 or "known as" in sentence
        ).rstrip("."),
    }


def _describe_docstring(docstring: str) -> dict[str, Any]:
    """What an indicator calculates and its parameters, for search_metrics by name.

    Args:
        docstring (str): The cleaned docstring of an indicator.

    Returns:
        dict[str, Any]: The ``description``: the first paragraphs.
    """
    # The description runs from the summary sentence up to the first section (the
    # references, the aliases or Args), so the explanation and the formula come along
    # with it; a paragraph that would take it past the limit is left out.
    paragraphs = []
    length = 0
    for paragraph in docstring.split("\n\n"):
        stripped = paragraph.strip()
        if not stripped:
            continue
        if paragraphs and DESCRIPTION_SECTION.match(stripped):
            break
        text = (
            stripped
            if stripped.startswith("-") or "\n-" in stripped
            else " ".join(stripped.split())
        )
        if paragraphs and length + len(text) > MAXIMUM_DESCRIPTION_CHARACTERS:
            break
        paragraphs.append(text)
        length += len(text)

    return {"description": "\n\n".join(paragraphs)}


def _rank_result(
    result: object, is_country: bool, entities: list[str] | None = None
) -> pd.DataFrame | None:
    """The result ranked across its tickers (or countries) per period, if it has that shape.

    A table with the periods as rows and a column per ticker is ranked as it is; one with
    the periods as columns (ratios) is turned around first. A second level with a single
    field (historical data with show_columns='Return', ratios with one metric) is dropped.

    Args:
        result (object): The answer of the indicator.
        is_country (bool): Whether the columns are countries rather than tickers.
        entities (list[str] | None, optional): The requested tickers or countries. The
            answer of a single one has its fields as columns (Close and Volume), which are
            not ranked against each other. Defaults to None, which does not check the
            columns.

    Returns:
        pd.DataFrame | None: The ranking, or None when the result has no such shape.
    """
    if isinstance(result, pd.Series) and isinstance(result.index, pd.MultiIndex):
        result = result.unstack(level=-1)
    if not isinstance(result, pd.DataFrame) or result.empty:
        return None
    if has_period_columns(result):
        result = result.T
    if result.columns.nlevels == 2:  # noqa: PLR2004
        # (field, ticker) for historical data, (ticker, metric) for turned-around ratios
        single_level = [
            level
            for level in range(2)
            if len(set(result.columns.get_level_values(level))) == 1
        ]
        if not single_level:
            return None
        result = result.droplevel(single_level[0], axis=1)
    if (
        result.columns.nlevels != 1
        or not is_time_index_of(result)
        or result.columns.size < 2  # noqa: PLR2004
        # The answer of a single ticker or country has its fields as columns.
        or (
            entities is not None
            and len(entities) == 1
            and str(entities[0]).upper()
            not in {str(column).upper() for column in result.columns}
        )
    ):
        return None
    result = result.copy()
    result.columns.name = "Country" if is_country else "Ticker"

    return rank_across_tickers(result)


# Words in an indicator's name that make its value a ratio, rate or count rather than an
# amount of money (get_price_to_earnings_ratio, get_gross_margin, get_beta, get_days_...).
DIMENSIONLESS_NAME_PARTS = (
    "ratio",
    "margin",
    "return",
    "yield",
    "growth",
    "_to_",
    "rate",
    "beta",
    "alpha",
    "score",
    "volatility",
    "percent",
    "multiple",
    "coverage",
    "turnover",
    "days",
    "drawdown",
    "correlation",
    "sharpe",
    "sortino",
    "variance",
    "skew",
    "kurtosis",
    "value_at_risk",
    "share_of",
    "weight",
)


def _ranks_amounts(method_name: str, ranking: pd.DataFrame) -> bool:
    """
    Whether a ranking compares amounts of money (a market cap, earnings per share, a price)
    rather than ratios, rates or counts, which have no currency.

    The indicator's name tells most apart; otherwise values that all lie between -1 and 1
    are taken for decimals (a rate or a return), as amounts rarely do.

    Args:
        method_name (str): The indicator that was ranked.
        ranking (pd.DataFrame): The ranking, with the ranked values in "Value".

    Returns:
        bool: Whether the ranked values are amounts of money.
    """
    if any(part in method_name.lower() for part in DIMENSIONLESS_NAME_PARTS):
        return False
    values = pd.to_numeric(ranking.get("Value"), errors="coerce").dropna()

    return not values.empty and bool((values.abs() > 1).any())


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


# The periods in a year, to annualize a standard deviation with the square root of it.
PERIODS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4}

# The note on a change that is empty because its base period has no value, with the name of
# the series (a country or ticker).
NO_CHANGE_PATTERN = re.compile(r"^No change for (.+?) in ")


def _has_fiscal_periods(result: object) -> bool:
    """Whether a result has years or quarters as its rows or columns (on any level), the
    periods a company's fiscal year is relabelled in.

    Args:
        result (object): The value returned by the indicator.

    Returns:
        bool: True when a table of the result has a yearly or quarterly period axis.
    """
    tables = (
        [result]
        if isinstance(result, pd.DataFrame | pd.Series)
        else [
            value
            for value in (result.values() if isinstance(result, dict) else [])
            if isinstance(value, pd.DataFrame | pd.Series)
        ]
    )

    return any(
        isinstance(values := axis.get_level_values(level), pd.PeriodIndex)
        and values.freqstr[0] in "YAQ"
        for table in tables
        for axis in [table.index]
        + ([table.columns] if isinstance(table, pd.DataFrame) else [])
        for level in range(axis.nlevels)
    )


def _period_name(period: str) -> str:
    """The noun of a period: 'year' for 'yearly' (or 'annual'), 'quarter' for 'quarterly'."""
    period = "yearly" if period.lower() == "annual" else period.lower()

    return SAMPLE_PERIOD_NAMES.get(period, "day" if period == "daily" else period)


def _row_period(result: pd.DataFrame | pd.Series) -> str:
    """The period each row (or, for periods as columns, each column) of a result covers."""
    dates = (
        result.columns
        if isinstance(result, pd.DataFrame) and has_period_columns(result)
        else result.index
    ).get_level_values(0)
    observed = observed_period(dates.unique())

    return _period_name(observed) if observed else "period"


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


def _unclosed_session_note(result: object) -> str | None:
    """
    A note when the last row of a daily series is a session that has not closed yet,
    whose prices are then the latest ones rather than the close.

    Args:
        result (object): The value returned by the indicator.

    Returns:
        str | None: The note, or None when the last row is a closed session, a date
            after today or the rows are no daily dates.
    """
    if (
        not isinstance(result, pd.DataFrame | pd.Series)
        or result.empty
        or not is_time_index_of(result)
    ):
        return None

    dates = result.index.get_level_values(0)

    # A year or a month is not one session.
    if isinstance(dates, pd.PeriodIndex):
        if dates.freqstr != "D":
            return None
        dates = dates.to_timestamp()

    last_date = dates.max().date()

    # A date after today is a scheduled event (the next earnings release or dividend in a
    # calendar), not a session.
    if (
        last_date <= policy_model.get_last_closed_session()
        or last_date > pd.Timestamp.today().date()
    ):
        return None

    return (
        f"The last row ({last_date}) is a session that has not closed yet, so its prices "
        "are the latest ones rather than the close; ask again after the close for the "
        "final figures."
    )


def _mentions_benchmark(result: object) -> bool:
    """Whether a result has a benchmark row or column, as a benchmark-based metric does:
    a 'Benchmark' row, or a column such as 'Benchmark Return'.

    Args:
        result (object): The value returned by the indicator.

    Returns:
        bool: True when one of the index or column labels contains 'Benchmark'.
    """
    if not isinstance(result, pd.DataFrame | pd.Series):
        return False

    axes = [result.index] + (
        [result.columns] if isinstance(result, pd.DataFrame) else []
    )

    return any(
        "Benchmark" in str(label)
        for axis in axes
        for level in range(axis.nlevels)
        for label in axis.get_level_values(level).unique()
    )


def _name_indicators(indicators: list[str], every_indicator: list[str]) -> str:
    """The indicators of a tool that take a parameter, in a few words.

    Args:
        indicators (list[str]): The indicators that take it.
        every_indicator (list[str]): The indicators of the tool.

    Returns:
        str: Empty when every indicator takes it, else the indicators (or those that do not
            take it, when that is the shorter list), the first few and a count beyond them.
    """
    if len(indicators) == len(every_indicator):
        return ""
    if len(indicators) <= MAXIMUM_NAMED_INDICATORS:
        return ", ".join(indicators)
    others = [name for name in every_indicator if name not in indicators]
    if len(others) <= MAXIMUM_NAMED_INDICATORS:
        return f"All but {', '.join(others)}"

    return ""


def _schema_parameter_description(
    tool: str,
    name: str,
    indicators: list[str],
    every_indicator: list[str],
    required: list[str],
    documented: str,
) -> str:
    """The short description of an indicator's parameter in the tool schema: which
    indicators take it and, for a parameter several tools share, a few words.

    Args:
        tool (str): The tool, for a description of the parameter in that tool only
            ("government.level").
        name (str): The parameter.
        indicators (list[str]): The indicators of the tool that take it.
        every_indicator (list[str]): The indicators of the tool.
        required (list[str]): The indicators that require it.
        documented (str): The shortest description the indicators' docstrings give, for a
            parameter every indicator takes that has no shared description.

    Returns:
        str: The description.
    """
    shared = (
        _SCHEMA_PARAM_DESCRIPTIONS.get(f"{tool}.{name}")
        or _SCHEMA_PARAM_DESCRIPTIONS.get(name)
        or _PARAM_DESCRIPTIONS.get(name, "")
    )
    words = shared if len(shared) <= MAXIMUM_SCHEMA_DESCRIPTION_CHARACTERS else ""
    named = _name_indicators(indicators, every_indicator)
    if not words and not named and len(indicators) == len(every_indicator):
        words = re.split(r"(?<=[a-z0-9)])[.;:] ", documented, maxsplit=1)[0].rstrip(".")
        if len(words) > MAXIMUM_SCHEMA_DESCRIPTION_CHARACTERS:
            words = (
                words[: MAXIMUM_SCHEMA_DESCRIPTION_CHARACTERS - 3].rsplit(" ", 1)[0]
                + "..."
            )
        words = words + "." if words and not words.endswith((".", "...")) else words
    description = (
        f"{named}: {words}" if named and words else (f"{named}." if named else words)
    )
    if required and required == indicators:
        description += " Required."
    elif required:
        named_required = _name_indicators(required, every_indicator)
        description += (
            f" Required by {named_required}."
            if named_required
            else f" Required by {len(required)} indicators."
        )

    return description.strip()


def _literal_values(annotation: Any) -> list[Any]:
    """The values of the Literal types in an annotation (Literal["a", "b"] | None).

    Args:
        annotation (Any): The annotation of a parameter.

    Returns:
        list[Any]: The values, in their order.
    """
    if typing.get_origin(annotation) is typing.Literal:
        return list(typing.get_args(annotation))

    return [
        value
        for member in typing.get_args(annotation)
        for value in _literal_values(member)
    ]


def _documented_options(text: str, refused: list[Any]) -> list[str]:
    """The options a parameter's description lists ('Options are: "historic" (the
    empirical quantile), "gaussian" ...'), for a parameter typed as plain text.

    Args:
        text (str): The description of the parameter.
        refused (list[Any]): The values the server refuses.

    Returns:
        list[str]: The quoted values after "Options are", without the explanations in
            parentheses and those the server refuses.
    """
    listed = re.search(r"Options are:?(.*?)(?:Defaults to|$)", text, re.DOTALL)
    if listed is None:
        return []
    # Explanations in parentheses can quote values of their own.
    values = re.findall(
        r"\"([^\"]+)\"|'([^']+)'", re.sub(r"\([^()]*\)", "", listed.group(1))
    )
    refused_values = {str(value).lower() for value in refused}

    return [
        value
        for value in dict.fromkeys(double or single for double, single in values)
        if value.lower() not in refused_values
    ]


def _readable_value(value: Any) -> str:
    """A default or option as a caller writes it: 'text', true, 0.05 or [1, 2]."""
    if isinstance(value, str):
        return f"'{value}'"
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _formula_variants(method: Any) -> tuple[Any, ...]:
    """
    Return the formula variants of an indicator: the values of its ``method`` parameter
    when that parameter is typed ``Literal[...]`` (``method: Literal["standard", "smart"]``),
    also inside an Optional or other union. A plain ``str`` method (an interpolation or an
    estimation option) and the other parameters (``model``, ``distribution``) select no
    formula variant, so they give none.

    The values are read from the controller's signature rather than from the MCP schema,
    which turns a Literal into the type of its values.

    Args:
        method (Any): The controller method of the indicator.

    Returns:
        tuple[Any, ...]: The values of the Literal, or an empty tuple when the method has
            no formula variants.
    """
    try:
        parameter = inspect.signature(method).parameters.get("method")
    except (TypeError, ValueError):
        return ()
    if parameter is None:
        return ()

    annotation = parameter.annotation
    origin = typing.get_origin(annotation)
    members = (
        typing.get_args(annotation)
        if origin is typing.Union or isinstance(annotation, types.UnionType)
        else (annotation,)
    )
    values = [
        value
        for member in members
        if typing.get_origin(member) is typing.Literal
        for value in typing.get_args(member)
    ]

    return tuple(dict.fromkeys(values))


def _method_key(owner: type | None, method_name: str) -> str:
    """
    Return what identifies an indicator's method: the class that defines it and its name,
    so a method inherited by several controllers is one method and two methods that share
    a name in different classes are two.

    Args:
        owner (type | None): The controller class the indicator is served from.
        method_name (str): The name of the indicator.

    Returns:
        str: The key, e.g. "financetoolkit.etf.etf_controller.ETF.get_profile".
    """
    defining = next(
        (
            klass
            for klass in getattr(owner, "__mro__", ())
            if method_name in vars(klass)
        ),
        owner,
    )

    return f"{getattr(defining, '__module__', '')}.{getattr(defining, '__qualname__', '')}.{method_name}"


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
    - ``float | str``                    → ``float | str``
    - ``DataFrame | ndarray | list[list[float]]`` → ``list[list[float]]``
    - ``Literal["garch", "gjr"]``         → ``str``
    - ``list | dict | None``               → ``list[str] | dict[str, float] | None``
    """
    if ann is inspect.Parameter.empty or ann is None:
        return str

    origin = typing.get_origin(ann)
    args = typing.get_args(ann)

    # The values a method takes (method: Literal["garch", "gjr", "egarch"]) are those of one
    # indicator, while the tool's parameter serves every indicator that has it (another
    # indicator's method="smart"), so the parameter takes the type of the values.
    if origin is typing.Literal:
        return type(args[0])

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
    for member in args:
        if member is type(None):
            continue
        a_origin = typing.get_origin(member)
        a = type(typing.get_args(member)[0]) if a_origin is typing.Literal else member
        if a_origin is typing.Literal and a in scalars:
            continue
        # Drop generic collections (list[str], dict[str, float], …)
        if a_origin in (list, dict, set, tuple):
            continue
        # Drop bare opaque types by name
        name = getattr(a, "__name__", "")
        if name in _OPAQUE_TYPES:
            continue
        scalars.append(a)

    if not scalars:
        # A matrix (a list of lists of numbers, such as a Wald restriction matrix) or a list
        # of objects (option strategy legs) has no text form, so it stays an array of arrays
        # or of objects.
        matrix = next(
            (
                a
                for a in args
                if typing.get_origin(a) is list
                and typing.get_args(a)
                and typing.get_origin(typing.get_args(a)[0]) in (list, dict)
            ),
            None,
        )
        if matrix is not None:
            return matrix | None if has_none else matrix
        # Labels in order or a score per label (the ranking of the centroid expected returns,
        # ['MSFT', 'AAPL'] or {'MSFT': 2, 'AAPL': 1}) take either form.
        if any(
            a is list or (typing.get_origin(a) is list and typing.get_args(a) == (str,))
            for a in args
        ) and any(typing.get_origin(a) is dict or a is dict for a in args):
            labels = list[str] | dict[str, float]
            return labels | None if has_none else labels
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

    if scalar not in (int, float):
        return scalar | None if has_none else scalar

    # A number accepts every other form the indicator takes, so none of them is refused
    # before the indicator sees it: a keyword or label (the Black-Litterman
    # risk_aversion='market', a maturity of '10Y'), a list (coupon rates or maturities for
    # a bond sensitivity table, lags for growth) and a value per ticker (the weight bounds of
    # the portfolio optimizers, the discount rates {'AAPL': 0.08} of a valuation model).
    # Built at runtime for the tool schema; ty cannot follow a type held in a variable.
    combined = scalar
    takes_text = str in scalars
    if any(
        a is list
        or (
            typing.get_origin(a) is list
            and all(arg in (int, float) for arg in typing.get_args(a))
        )
        for a in args
    ):
        # A list of labels next to the numbers ('2Y,10Y') keeps them as text.
        item = scalar | str if takes_text else scalar
        combined = combined | list[item]  # ty: ignore[invalid-type-form]
        # A list is also written as text ('1,2'), which coerce_value splits.
        takes_text = True
    if any(
        typing.get_origin(a) is dict
        and all(arg in (str, int, float) for arg in typing.get_args(a))
        for a in args
    ):
        combined = combined | dict[str, scalar]  # ty: ignore[invalid-type-form]
    if takes_text:
        combined = combined | str

    return combined | None if has_none else combined


class RouterGroupSpec(NamedTuple):
    """
    Immutable specification for a single router group tool.

    Each instance describes one categorical master tool — the metadata needed
    to discover controller methods, build a dispatcher wrapper, and register
    the tool on the FastMCP instance.

    Args:
        tool_name (str): Unique name for the FastMCP tool (e.g. "valuation").
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
        risk_free_rate (bool): Whether the tool takes the Toolkit's risk_free_rate, for the
            modules that use a risk-free rate (performance, risk, models, portfolio). The
            options methods take a risk_free_rate of their own (the rate as a number).
        basket (bool): Whether the tool takes the Toolkit's basket, for the tools that read
            the financial statements (the statements, ratios and models, and the series of
            econometrics), which then add up the tickers into a "Total".
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
    risk_free_rate: bool = False
    basket: bool = False


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


def _finite_number(value: object, name: str) -> float:
    """A finite number, or a ValueError naming the input it came from."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        number = math.nan
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a number, not {value!r}.")
    return number


def _ticker(value: object, name: str) -> str:
    """A ticker symbol in upper case, or a ValueError naming the input it came from."""
    ticker = str(value).strip().upper()
    if not TICKER_PATTERN.match(ticker):
        raise ValueError(f"{name} has an invalid ticker: {value!r}.")
    return ticker


def _is_currency_or_crypto(ticker: str) -> bool:
    """
    Whether a ticker is a currency pair (EURUSD, EURUSD=X) or a cryptocurrency quoted in a
    currency (BTCUSD, BTC-USD, ETH-BTC), which have prices but no financial statements. A
    share (AAPL, ASML.AS, BRK-B) is neither: a crypto pair is at least six characters
    without an exchange suffix, longer than a US share ticker.

    Args:
        ticker (str): The ticker as the caller wrote it.

    Returns:
        bool: Whether the ticker is a currency or crypto pair.
    """
    symbol = currencies_model.to_financial_modeling_prep_symbol(ticker).upper()
    if currencies_model.is_currency_pair(symbol):
        return True
    return (
        len(symbol) >= 2 * currencies_model.CURRENCY_CODE_LENGTH
        and symbol.isalnum()
        and any(
            symbol.endswith(quote) and len(symbol) > len(quote) + 1
            for quote in (
                currencies_model.CURRENCY_CODES
                | currencies_model.CRYPTO_QUOTE_CURRENCIES
            )
        )
    )


def _parse_json_text(value: object) -> object:
    """A JSON list or object sent as text, parsed; any other value as it is."""
    if isinstance(value, str) and value.strip()[:1] in ("[", "{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"The JSON could not be read: {error.msg}.") from error
    return value


def _amounts_per_ticker(value: object, name: str) -> dict[str, float]:
    """
    Weights or holdings given as {"VTI": 0.6} or as text such as "VTI:0.6, BND:0.4".

    Args:
        value (object): The value as the client sent it.
        name (str): The input, named in an error.

    Returns:
        dict[str, float]: The amount per ticker; a ticker given more than once gets the
            sum of its amounts.
    """
    value = _parse_json_text(value)
    if isinstance(value, str):
        pairs = [item.split(":", 1) for item in value.split(",") if item.strip()]
        if not pairs or any(len(pair) != 2 for pair in pairs):  # noqa: PLR2004
            raise ValueError(
                f"{name} must be given per ticker, e.g. {{'VTI': 0.6, 'BND': 0.4}} or 'VTI:0.6, BND:0.4'."
            )
        # Kept as pairs until the tickers are cleaned, so "AAPL:2,AAPL:3" adds up as
        # "AAPL:2, AAPL:3" does rather than the last one replacing the first.
        items = [(ticker, amount) for ticker, amount in pairs]
    elif isinstance(value, dict) and value:
        items = list(value.items())
    else:
        raise ValueError(
            f"{name} must be given per ticker, e.g. {{'VTI': 0.6, 'BND': 0.4}}."
        )
    amounts: dict[str, float] = {}
    for ticker, amount in items:
        symbol = _ticker(ticker, name)
        amounts[symbol] = amounts.get(symbol, 0.0) + _finite_number(
            amount, f"The {name} of {symbol}"
        )
    return amounts


def _transactions(value: object) -> list[dict[str, object]]:
    """
    Transactions given as a list of trades, e.g. [{"date": "2023-01-03", "ticker": "AAPL",
    "volume": 10, "price": 125.1}], with their fields in any casing. Prices, costs and
    currencies are given for every trade or for none: a trade without a price is priced at
    the close of its day, which applies to the whole list.

    Args:
        value (object): The value as the client sent it.

    Returns:
        list[dict[str, object]]: The trades with the column names of a Portfolio dataset.
    """
    value = _parse_json_text(value)
    if not isinstance(value, list) or not value:
        raise ValueError(
            "transactions must be a list of trades, e.g. [{'date': '2023-01-03', "
            "'ticker': 'AAPL', 'volume': 10, 'price': 125.1}]."
        )
    if len(value) > MAX_TRANSACTIONS:
        raise ValueError(
            f"transactions holds {len(value)} trades; at most {MAX_TRANSACTIONS} can be analyzed in one call."
        )
    trades: list[dict[str, object]] = []
    for number, raw in enumerate(value, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Transaction {number} is not an object with fields.")
        fields = {}
        for key, item in raw.items():
            column = TRANSACTION_FIELDS.get(str(key).strip().lower())
            if column is None:
                raise ValueError(
                    f"Transaction {number} has an unknown field {key!r}; use date, ticker, "
                    "volume and optionally price, costs and currency."
                )
            fields[column] = item
        missing = [
            name for name in ("Date", "Ticker", "Volume") if fields.get(name) is None
        ]
        if missing:
            raise ValueError(
                f"Transaction {number} has no {', '.join(name.lower() for name in missing)}."
            )
        date = str(fields["Date"]).strip().replace("/", "-")
        if not validation_model.is_valid_date(date):
            raise ValueError(
                f"Transaction {number} has an invalid date {fields['Date']!r}; use YYYY-MM-DD."
            )
        trade: dict[str, object] = {
            "Date": date,
            "Ticker": _ticker(fields["Ticker"], f"Transaction {number}"),
            "Volume": _finite_number(
                fields["Volume"], f"The volume of transaction {number}"
            ),
        }
        if fields.get("Price") is not None:
            trade["Price"] = _finite_number(
                fields["Price"], f"The price of transaction {number}"
            )
        # A trade without costs had none, rather than costs the Portfolio warns are missing.
        trade["Costs"] = (
            _finite_number(fields["Costs"], f"The costs of transaction {number}")
            if fields.get("Costs") is not None
            else 0.0
        )
        if fields.get("Currency") is not None:
            currency = str(fields["Currency"]).strip()
            if not CURRENCY_PATTERN.match(currency):
                raise ValueError(
                    f"Transaction {number} has an invalid currency {currency!r}; use a three-letter code such as 'USD'."
                )
            trade["Currency"] = currency
        trades.append(trade)
    for column in ("Price", "Currency"):
        given = sum(column in trade for trade in trades)
        if 0 < given < len(trades):
            raise ValueError(
                f"{column} is given for {given} of the {len(trades)} transactions; give it "
                "for every transaction or for none."
            )
    return trades


def _portfolio_inputs(arguments: dict[str, Any]) -> dict[str, Any]:
    """
    The portfolio a portfolio_performance call describes, read from its arguments (which
    are removed from them): exactly one of transactions, holdings, weights or preset, and
    optionally start_date, initial_value, rebalancing and currency.

    Args:
        arguments (dict[str, Any]): The call's arguments.

    Raises:
        ValueError: When the portfolio is missing, given more than once or invalid.

    Returns:
        dict[str, Any]: The validated inputs, plain values only, so they also key the
        response cache.
    """
    given = {
        name: arguments.pop(name, None)
        for name in ("transactions", "holdings", "weights", "preset")
    }
    given = {
        name: value for name, value in given.items() if value not in (None, "", [], {})
    }
    options = {
        name: arguments.pop(name, None)
        for name in ("initial_value", "rebalancing", "currency")
    }
    if len(given) != 1:
        raise ValueError(
            "Please describe the portfolio with exactly one of transactions, holdings, "
            "weights or preset"
            + (f" (received {', '.join(sorted(given))})." if given else ".")
        )
    ((kind, value),) = given.items()
    inputs: dict[str, Any] = {}
    if kind == "transactions":
        inputs["transactions"] = _transactions(value)
        tickers = {trade["Ticker"] for trade in inputs["transactions"]}
    elif kind == "preset":
        inputs["preset"] = str(value).strip()[:100]
        tickers = set()
    else:
        inputs[kind] = _amounts_per_ticker(value, kind)
        tickers = set(inputs[kind])
    if len(tickers) > MAX_ENTITIES:
        raise ValueError(
            f"The portfolio holds {len(tickers)} tickers; at most {MAX_ENTITIES} can be analyzed in one call."
        )
    if options["initial_value"] not in (None, ""):
        initial_value = _finite_number(options["initial_value"], "initial_value")
        if initial_value <= 0:
            raise ValueError("initial_value must be more than 0.")
        inputs["initial_value"] = initial_value
    if options["rebalancing"] not in (None, ""):
        rebalancing = str(options["rebalancing"]).strip().lower()
        if rebalancing not in REBALANCING_FREQUENCIES:
            raise ValueError(
                f"rebalancing must be one of {', '.join(REBALANCING_FREQUENCIES)}, not {options['rebalancing']!r}."
            )
        inputs["rebalancing"] = rebalancing
    if options["currency"] not in (None, ""):
        currency = str(options["currency"]).strip()
        if not CURRENCY_PATTERN.match(currency):
            raise ValueError(
                f"currency must be a three-letter code such as 'USD', not {currency!r}."
            )
        # Pence keep their lower-case p (GBp); other codes are written in capitals.
        inputs["currency"] = currency if currency == "GBp" else currency.upper()
    return inputs


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
        parameter_limits: dict[str, Any] | None = None,
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
            parameter_limits (dict[str, Any] | None): The most and least a size parameter may
                be ("maximums", "minimums") and the most parameters computed together may multiply to
                ("products"), see coercion_model.size_limit_problem, and the values refused
                per indicator ("refused_values"), see coercion_model.refused_value_problem.
                Defaults to None (no limits).
        """
        self._mcp = mcp
        self._provider = provider
        self._inspector = inspector
        self._tool_index: dict[str, list[dict[str, str]]] = {}
        self._indicator_index: list[dict[str, Any]] = []
        # Each router tool's own function, so a call naming another tool's indicator is
        # answered by that tool.
        self._router_functions: dict[str, Any] = {}
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
        self._parameter_limits: dict[str, Any] = parameter_limits or {}

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

    def _offered_variants(self, method_name: str, method: Any) -> tuple[Any, ...]:
        """
        The formula variants of an indicator (see _formula_variants) that the server does
        not refuse (config.yaml parameter_limits, refused_values), such as EGARCH.

        Args:
            method_name (str): The name of the indicator.
            method (Any): The controller method of the indicator.

        Returns:
            tuple[Any, ...]: The values of its method parameter that can be asked for.
        """
        refused = {
            str(value).lower()
            for refusal in self._parameter_limits.get("refused_values", [])
            if refusal["parameter"] == "method" and method_name in refusal["methods"]
            for value in refusal["values"]
        }

        return tuple(
            variant
            for variant in _formula_variants(method)
            if str(variant).lower() not in refused
        )

    def _describe_parameters(
        self, tool_name: str, method_name: str, method: Any
    ) -> tuple[dict[str, str], list[dict[str, Any]]]:
        """
        What search_metrics says about each parameter of an indicator: its whole description
        from the docstring, its options, its default (the server's own where config.yaml
        method_defaults sets one) and the values the server refuses. The tool schema only
        names the indicators that take a parameter.

        Args:
            tool_name (str): The tool of the indicator.
            method_name (str): The indicator.
            method (Any): The controller method of the indicator.

        Returns:
            tuple[dict[str, str], list[dict[str, Any]]]: Per parameter, in the order of the
                signature, its description, and the same parameters for the call line of a
                search result: ``name``, ``options``, ``default``, ``required`` and
                ``conditional`` (its description starts with "Only", e.g. "Only used with
                the GARCH method").
        """
        documented = _docstring_arguments(
            inspect.getdoc(method) or "", maximum_characters=None, keep_defaults=True
        )
        server_defaults = self._method_defaults.get(
            f"{tool_name}.{method_name}", self._method_defaults.get(method_name, {})
        )
        refused: dict[str, list[str]] = {}
        for refusal in self._parameter_limits.get("refused_values", []):
            if method_name in refusal["methods"]:
                refused.setdefault(refusal["parameter"], []).extend(refusal["values"])
        described: dict[str, str] = {}
        call: list[dict[str, Any]] = []
        for parameter in self._inspector.extract_extra_params(method):
            name = parameter.name
            text = documented.get(name) or _PARAM_DESCRIPTIONS.get(name, "")
            parts = [text] if text else []
            options = [
                value
                for value in dict.fromkeys(_literal_values(parameter.annotation))
                if str(value).lower() not in refused.get(name, ())
            ]
            # Listed unless the docstring's own list names them all already.
            if options and not all(
                f"'{value}'" in text or f'"{value}"' in text for value in options
            ):
                parts.append(
                    f"Options: {', '.join(_readable_value(value) for value in options)}."
                )
            if name in server_defaults:
                parts.append(
                    f"Default on this server: {_readable_value(server_defaults[name])}."
                )
            elif parameter.default is inspect.Parameter.empty:
                parts.append("Required.")
            elif "Defaults to" not in text and parameter.default is not None:
                parts.append(f"Default: {_readable_value(parameter.default)}.")
            if name in refused:
                parts.append(
                    f"Not available on this server: {', '.join(_readable_value(value) for value in refused[name])}."
                )
            described[name] = " ".join(parts)
            call.append(
                {
                    "name": name,
                    "options": options
                    or _documented_options(text, refused.get(name, [])),
                    "default": server_defaults.get(name, parameter.default),
                    "required": name not in server_defaults
                    and parameter.default is inspect.Parameter.empty,
                    "conditional": text.startswith("Only"),
                }
            )

        return described, call

    def get_indicator_index(self) -> list[dict[str, Any]]:
        """Return every registered indicator with its tool, summary and aliases.

        Returns:
            list[dict[str, Any]]: One entry per indicator with the keys ``tool``,
                ``indicator``, ``variants`` (the values of its formula-selecting ``method``),
                ``summary`` and ``aliases``.
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
                    risk_free_rate=bool(raw.get("risk_free_rate")),
                    basket=bool(raw.get("basket")),
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
        period: str | None = None,
        basket: bool = False,
        deflate_with: str | None = None,
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
            period (str | None, optional): The period the call asks for, passed on to every
                indicator that takes one and is published at it, so a monthly analysis reads
                the monthly series its source publishes (a monthly average, for some rates)
                rather than the last daily value of each month. Defaults to None, each
                indicator's own default.
            basket (bool, optional): Whether the indicators by ticker also add up the tickers
                into a "Total", which becomes a series of its own. Defaults to False.
            deflate_with (str | None, optional): The country whose consumer price index is
                passed on to every indicator that takes deflate_with, for its values in
                real terms. Defaults to None, nominal values.

        Returns:
            pd.DataFrame: The series, a column each, by date.

        Raises:
            ValueError: When a reference names an unknown tool or indicator, an argument
                its tool does not expose or a value over the server's limits, or when an
                indicator answers without values for a reason it logged.
        """
        parsed = series_model.parse_series_references(references)
        # Each reference collects its own tickers or countries, so together they are held
        # to the most one call may analyze.
        if (
            count := sum(len(reference.entities) for reference in parsed)
        ) > MAX_ENTITIES:
            raise ValueError(
                f"The series name {count} tickers and countries together; at most "
                f"{MAX_ENTITIES} can be analyzed in one call. Split them over several calls."
            )
        # Every reference is checked before any is collected, so a refused one costs no
        # data collection for the others.
        prepared = []
        for reference in parsed:
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
                    f"`{reference.indicator}` is not an indicator of `{tools[0]}` (series '{reference.text}')."
                )
            module, category = routes[reference.indicator]
            by_ticker = category in (
                self._inspector.categories.get("ticker", "ticker"),
                self._inspector.categories.get("toolkit", "toolkit"),
            )
            if by_ticker and (
                invalid := [
                    entity
                    for entity in reference.entities
                    if not TICKER_PATTERN.match(entity.upper())
                ]
            ):
                raise ValueError(
                    f"The series '{reference.text[:80]}' has an invalid ticker: {invalid[0][:40]!r}."
                )
            entities = (
                [entity.upper() for entity in reference.entities]
                if by_ticker
                else [
                    normalize_country_name(entity, country_names())
                    for entity in reference.entities
                ]
            )
            arguments = {
                "module_name": module,
                "method_name": reference.indicator,
                "category": category,
                "tickers": entities if by_ticker else None,
                "countries": None if by_ticker else entities,
                "start_date": start_date,
                "end_date": end_date,
                "quarterly": quarterly,
                "benchmark_ticker": benchmark_ticker,
                "toolkit_basket": basket and by_ticker,
            }
            parameters = self._inspector.get_method_param_names(
                self._module_class_map.get(module), reference.indicator
            )
            # An indicator that lists records (the economic calendar's releases) is asked
            # for them as series by the period they cover.
            if "as_series" in parameters:
                arguments["as_series"] = True
            if deflate_with and "deflate_with" in parameters:
                arguments["deflate_with"] = deflate_with
            # The reference's own arguments (cumulative=true), typed as the indicator takes them.
            # Only the parameters its tool exposes are taken: the others are the server's own
            # (self, rounding) or set by the tool from its own parameters (start_date).
            method = getattr(self._module_class_map.get(module), reference.indicator)
            signature = inspect.signature(method).parameters
            exposed = {
                parameter.name
                for parameter in self._inspector.extract_extra_params(method)
            }
            label = f"`{reference.indicator}` in the series '{reference.text[:80]}'"
            if set_by_tool := [
                name for name in reference.arguments if name in SERIES_TOOL_PARAMETERS
            ]:
                raise ValueError(
                    f"{label}: {', '.join(set_by_tool)} "
                    f"{'is' if len(set_by_tool) == 1 else 'are'} set by the tool, pass "
                    f"{'it' if len(set_by_tool) == 1 else 'them'} as a tool parameter."
                )
            # Argument names are lower case, so Window=10 would be read as a ticker or column.
            if miscased := [
                match["name"]
                for part in [*reference.entities, reference.column or ""]
                if (match := MISCASED_ARGUMENT.match(part))
                and match["name"] != match["name"].lower()
                and match["name"].lower() in exposed
            ]:
                raise ValueError(
                    f"{label}: argument names are lower case, so '{miscased[0]}=' is read "
                    f"as a ticker or column. Write it as '{miscased[0].lower()}='."
                )
            if unknown := [name for name in reference.arguments if name not in exposed]:
                raise ValueError(
                    f"`{reference.indicator}` takes no argument {', '.join(unknown)} in "
                    f"the series '{reference.text[:80]}'."
                )
            given = {
                name: coerce_value(value, signature[name].annotation)
                for name, value in reference.arguments.items()
            }
            # The same limits as a call of the indicator's own tool, with its own defaults
            # for the parameters computed together.
            if problem := size_limit_problem(
                given
                | _pseudo_parameters(
                    entities if by_ticker else None, start_date, end_date
                ),
                {
                    name: parameter.default
                    for name, parameter in signature.items()
                    if parameter.default is not inspect.Parameter.empty
                },
                self._parameter_limits,
                reference.indicator,
            ) or refused_value_problem(
                reference.indicator, given, self._parameter_limits
            ):
                raise ValueError(
                    f"`{tools[0]}` (`{reference.indicator}`) in the series '{reference.text[:80]}': {problem}"
                )
            arguments.update(given)
            prepared.append((reference, arguments, entities, by_ticker, parameters))

        frames = []
        for reference, arguments, entities, by_ticker, parameters in prepared:
            # The messages logged while the indicator runs explain an empty answer, such as
            # a method it does not know.
            messages = current_call_messages()
            logged = len(messages) if messages is not None else 0
            # The ratios of statement items have no period but show_daily, which maps every
            # period's figure onto the daily prices; a daily analysis asks for those.
            if period == "daily" and "show_daily" in parameters:
                arguments["show_daily"] = True
            result = None
            if period and "period" in parameters:
                result = self._provider.call_method(**arguments, period=period)
            # An indicator that is not published at the period (a monthly series asked for
            # weekly) is read at its own frequency instead, which the analysis resamples.
            if result is None or getattr(result, "empty", False):
                result = self._provider.call_method(**arguments)
            if (
                result is None
                or getattr(result, "empty", False)
                or (
                    isinstance(result, pd.DataFrame | pd.Series)
                    and not result.notna().to_numpy().any()
                )
            ) and (reason := summarize_reasons((messages or [])[logged:])):
                raise ValueError(
                    f"The series '{reference.text[:80]}' has no values: {reason}"
                )
            # The Total of a basket is a series of its own, next to the tickers.
            resolved = reference._replace(
                entities=entities + (["Total"] if arguments["toolkit_basket"] else [])
            )
            # The benchmark the Toolkit was built with, under which it reports a requested
            # ticker that is also the benchmark.
            benchmark = (
                resolve_benchmark_ticker(entities, benchmark_ticker)[0]
                if by_ticker
                else None
            )
            frames.append(
                (resolved, series_model.to_series_columns(result, resolved, benchmark))
            )
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
                    f"No controller class is registered for method {method_name!r} of tool {tool_name!r}."
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
        # Indicators that take **kwargs (get_factor_analysis) pass the parameters of the other
        # indicators of the tool on to the metrics they analyse.
        keyword_methods = {
            m
            for m in group_methods
            if any(
                p.kind is inspect.Parameter.VAR_KEYWORD
                for p in inspect.signature(
                    getattr(owning_class(m), m)
                ).parameters.values()
            )
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
            # Arguments that are no parameter of the tool (a misspelt one, or one of another
            # tool, such as benchmark_ticker on a country tool) are taken off before any is
            # read, so none of them is applied, and named in the answer.
            routed_note = kwargs.pop(ROUTED_NOTE, None)
            # A client can send the name too (unknown arguments are kept), so it only ever
            # makes the answer smaller.
            answer_size = kwargs.pop(ANSWER_SIZE, None)
            if not (
                isinstance(answer_size, int)
                and max_response_characters
                and 0 < answer_size < max_response_characters
            ):
                answer_size = max_response_characters
            # Several indicators of the tool (comma-separated, or a list) are each answered as
            # a call of their own with the same arguments, and the answers share the size of a
            # response.
            indicators = list(
                dict.fromkeys(_split_values(kwargs.get("indicator") or ""))
            )
            if len(indicators) > MAX_INDICATORS:
                record_outcome("invalid", tool_name)
                return (
                    f"`{tool_name}` received {len(indicators)} indicators; at most "
                    f"{MAX_INDICATORS} can be calculated in one call. Split them over "
                    "several calls."
                )
            if len(indicators) > 1:
                return _combine_answers(
                    {
                        name: wrapper(
                            **{
                                **kwargs,
                                "indicator": name,
                                ANSWER_SIZE: (
                                    answer_size // len(indicators)
                                    if answer_size
                                    else None
                                ),
                            }
                        )
                        for name in indicators
                    }
                )
            if indicators:
                kwargs["indicator"] = indicators[0]
            tool_parameters = list(wrapper.__signature__.parameters)
            # What the caller set, for a call answered by another tool: the schema defaults of
            # this tool's parameters are no arguments to pass on.
            arguments = {
                pname: value
                for pname, value in kwargs.items()
                if pname not in tool_parameters
                or value != wrapper.__signature__.parameters[pname].default
            }
            unknown_arguments = [
                pname for pname in kwargs if pname not in tool_parameters
            ]
            for pname in unknown_arguments:
                kwargs.pop(pname)

            raw_indicator = kwargs.pop("indicator", None)
            if not raw_indicator:
                record_outcome("invalid", tool_name)
                return (
                    f"Please specify an `indicator`. Available for `{tool_name}`: "
                    + ", ".join(f"`{m}`" for m in all_indicators)
                )
            candidate = raw_indicator
            if (
                not candidate.startswith("get_")
                and f"get_{candidate}" in method_param_names
            ):
                candidate = f"get_{candidate}"
            indicator_note = routed_note
            # An indicator of exactly one other tool is answered by that tool (with the same
            # arguments), before any name of this tool is taken for it: a close name here is
            # another measure (get_cash_interest_coverage_ratio for get_interest_coverage_ratio).
            elsewhere = (
                [
                    entry["tool"]
                    for entry in self._indicator_index
                    if entry["indicator"] == candidate
                ]
                if candidate not in method_param_names
                else []
            )
            if len(elsewhere) == 1 and elsewhere[0] in self._router_functions:
                return self._router_functions[elsewhere[0]](
                    **{
                        **arguments,
                        "indicator": candidate,
                        ROUTED_NOTE: f"`{candidate}` is an indicator of `{elsewhere[0]}`, "
                        f"so `{elsewhere[0]}` answered it.",
                        ANSWER_SIZE: answer_size,
                    }
                )
            # A near-certain match (get_gordon_growth_model for a misspelt get_gorden_growth_model)
            # is used, with a note, but only a misspelling: a name with a word more or less
            # (get_real_gross_domestic_product for get_gross_domestic_product) is another measure.
            if candidate not in method_param_names and not elsewhere:
                near = [
                    name
                    for name in difflib.get_close_matches(
                        candidate, all_indicators, n=2, cutoff=NEAR_CERTAIN_MATCH
                    )
                    if name.count("_") == candidate.count("_")
                ]
                if len(near) == 1:
                    indicator_note = f"`{raw_indicator}` was read as `{near[0]}`."
                    candidate = near[0]
            if candidate not in method_param_names:
                if elsewhere:
                    record_outcome("invalid", tool_name, UNKNOWN_INDICATOR)
                    return (
                        f"`{candidate}` is not part of `{tool_name}`; it is an indicator of "
                        + " and ".join(f"`{tool}`" for tool in elsewhere)
                        + f". Please call that tool with `indicator='{candidate}'`."
                    )
                close = difflib.get_close_matches(
                    candidate, all_indicators, n=3, cutoff=0.6
                )
                # Close names of other tools too, with their tool, as an indicator is often
                # looked for under a neighbouring tool (interest coverage under solvency).
                other_tools = {
                    entry["indicator"]: entry["tool"]
                    for entry in self._indicator_index
                    if entry["tool"] != tool_name
                }
                close += [
                    f"{name} (tool `{other_tools[name]}`)"
                    for name in difflib.get_close_matches(
                        candidate, list(other_tools), n=2, cutoff=0.75
                    )
                    if name not in all_indicators
                ]
                suggestion = (
                    (" Did you mean: " + ", ".join(close) + "?") if close else ""
                )
                record_outcome("invalid", tool_name, UNKNOWN_INDICATOR)
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
            for name, entities in (("tickers", tickers), ("countries", countries)):
                if entities and len(entities) > MAX_ENTITIES:
                    record_outcome("invalid", tool_name, method_name)
                    return (
                        f"`{tool_name}` received {len(entities)} {name}; at most "
                        f"{MAX_ENTITIES} can be analyzed in one call. Split them over "
                        "several calls."
                    )
            if invalid := [
                ticker for ticker in tickers or [] if not TICKER_PATTERN.match(ticker)
            ]:
                record_outcome("invalid", tool_name, method_name)
                return (
                    f"`{tool_name}` received an invalid ticker: {invalid[0][:40]!r}. A ticker "
                    "has at most 20 letters, digits and the characters . - = ^, e.g. 'AAPL', "
                    "'ASML.AS', 'BTC-USD' or '^GSPC'."
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
            deflate_with = (
                kwargs.pop("deflate_with", None)
                if module_name == "econometrics"
                else None
            )
            # A tool argument is text, so 'none' stands for changes=None (the series are
            # changes already).
            changes = None if str(changes).lower() == "none" else changes

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
                # A ticker given under another name (ticker, symbol) is pointed to tickers;
                # `ticker` can be a parameter of another indicator of the same tool.
                misnamed = [
                    pname
                    for pname in [*unknown_arguments, *kwargs]
                    if pname.lower() in ("ticker", "symbol", "symbols")
                    and pname not in method_param_names[method_name]
                ]
                record_outcome("invalid", tool_name, method_name)
                return (
                    (
                        f"`{misnamed[0]}` is not a parameter of `{method_name}`; pass the ticker symbols as `tickers`. "
                        if misnamed
                        else ""
                    )
                    + f"`{tool_name}` (`{method_name}`) requires a `tickers` parameter. "
                    "Please provide one or more ticker symbols, e.g. `tickers='AAPL'` "
                    "or `tickers='AAPL,MSFT'`."
                )

            # A portfolio_performance call describes its portfolio instead of tickers; the
            # list of classic portfolios needs none.
            portfolio_inputs = None
            if effective_category == inspector.categories.get("portfolio", "portfolio"):
                try:
                    portfolio_inputs = _portfolio_inputs(kwargs)
                except ValueError as exc:
                    if method_name != "get_presets":
                        record_outcome("invalid", tool_name, method_name)
                        return (
                            f"Invalid input for `{tool_name}` (`{method_name}`): {exc}"
                        )
                if method_name == "get_presets":
                    portfolio_inputs = {}

            # Tickers on a tool that works with countries would otherwise be dropped as an
            # unknown parameter and every country returned.
            if (
                tickers
                and not countries
                and effective_category
                == inspector.categories.get("standalone", "standalone")
            ):
                record_outcome("invalid", tool_name, method_name)
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
                    record_outcome("invalid", tool_name, method_name)
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
            benchmark_ticker = kwargs.pop("benchmark_ticker", None)
            # Left empty, the server's default benchmark applies (see below for a default
            # that is one of the tickers).
            benchmark_given = benchmark_ticker not in (None, "")
            if not benchmark_given:
                benchmark_ticker = inspector.benchmark_ticker
            # Clients write "no benchmark" as text; a ticker named NONE has no data.
            elif isinstance(
                benchmark_ticker, str
            ) and benchmark_ticker.strip().lower() in (
                "none",
                "null",
            ):
                benchmark_ticker = None
            elif benchmark_ticker is not None and not TICKER_PATTERN.match(
                str(benchmark_ticker).strip().upper()
            ):
                record_outcome("invalid", tool_name, method_name)
                return (
                    f"Invalid `benchmark_ticker` for `{tool_name}` (`{method_name}`): "
                    f"{str(benchmark_ticker)[:40]!r}. Please use a ticker such as 'SPY' or "
                    "'^GSPC', or 'None' for no benchmark."
                )
            # The default benchmark asked for as a ticker stays a ticker in every tool: the
            # Toolkit would report it as 'Benchmark' only, which a test, an optimizer or an
            # ETF's holdings leave out (and then miss it among the tickers), so the next
            # benchmark that is not one of the tickers takes its place. A benchmark the
            # caller chose is kept (see resolve_benchmark_ticker).
            benchmark_swap_note = None
            if (
                not benchmark_given
                and benchmark_ticker
                and tickers
                and benchmark_ticker.upper() in tickers
            ):
                fallback = next(
                    (
                        candidate
                        for candidate in FALLBACK_BENCHMARKS
                        if candidate.upper() not in tickers
                    ),
                    None,
                )
                benchmark_swap_note = (
                    f"{benchmark_ticker} is one of the tickers, so "
                    + (
                        f"{fallback} is the benchmark: 'Benchmark' refers to {fallback}."
                        if fallback
                        else "the results are calculated without a benchmark."
                    )
                    + " Set benchmark_ticker to choose another."
                )
                benchmark_ticker = fallback

            # The Toolkit's own setting, taken off before the method's arguments and passed under
            # its own name, so it never meets a method's own risk_free_rate (options takes the
            # rate as a number).
            toolkit_risk_free_rate = "10y"
            if spec.risk_free_rate:
                toolkit_risk_free_rate = (
                    str(kwargs.pop("risk_free_rate", None) or "10y").strip().lower()
                )
                if (
                    toolkit_risk_free_rate
                    not in validation_model.RISK_FREE_RATE_OPTIONS
                ):
                    record_outcome("invalid", tool_name, method_name)
                    return (
                        f"Invalid `risk_free_rate` for `{tool_name}` (`{method_name}`): "
                        f"'{toolkit_risk_free_rate}'. Please use one of "
                        f"{', '.join(validation_model.RISK_FREE_RATE_OPTIONS)}."
                    )

            # The Toolkit's own setting as well, so the Total is part of every statement.
            toolkit_basket = bool(
                spec.basket and to_boolean(kwargs.pop("basket", False))
            )

            # rank is handled here as well, unless the indicator ranks itself.
            rank_tickers = "rank" not in method_param_names.get(
                method_name, set()
            ) and to_boolean(kwargs.pop("rank", False))

            # rebase_to is handled here too; its window is read before any data is fetched.
            raw_rebase_to = str(kwargs.pop("rebase_to", None) or "").strip()
            try:
                base_window = (
                    parse_base_window(raw_rebase_to) if raw_rebase_to else None
                )
            except ValueError as exc:
                record_outcome("invalid", tool_name, method_name)
                return f"Invalid `rebase_to` for `{tool_name}` (`{method_name}`): {exc}"

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
                record_outcome("invalid", tool_name, method_name)
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
                record_outcome("invalid", tool_name, method_name)
                return (
                    f"Invalid input for `{tool_name}` (`{method_name}`): limit must be a "
                    f"whole number, e.g. 5 for the first five rows or -5 for the last "
                    f"five, not {raw_limit!r}."
                )
            # sample keeps the last row of every week, month, quarter or year of a daily
            # answer (a 200-day average read weekly), before limit selects from those rows.
            sample = str(kwargs.pop("sample", None) or "").strip().lower() or None
            if sample is not None and sample not in SAMPLE_PERIODS:
                record_outcome("invalid", tool_name, method_name)
                return (
                    f"Invalid input for `{tool_name}` (`{method_name}`): sample must be one "
                    f"of {', '.join(SAMPLE_PERIODS)}, not {sample!r}."
                )
            row_selection = None
            # 0 asks for every row: no selection, and no MCP default limit either (the 25
            # largest holdings of an ETF), so the indicator's own default of all rows holds.
            every_row = rows_limit == 0
            if every_row:
                rows_limit = None

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
            requested_period: str | None = None
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
                    if pname not in accepted_params and not (
                        method_name in keyword_methods and pname in given_params
                    ):
                        if pname in given_params:
                            ignored_params.append(pname)
                            if pname == "period":
                                requested_period = str(val)
                        continue
                    # A disputed parameter is advertised as unset, so an unset value means "no preference" and must reach the indicator's own default rather than another indicator's.  # noqa: E501
                    if pname in disputed_params and val in (None, ""):
                        continue
                    method_kwargs[pname] = coerce_value(val, pann)

            # Discovery takes a range only when one is given (see provider_model).
            if effective_category == inspector.categories.get("discovery", "discovery"):
                start_date = given_dates.get("start_date", "")
                end_date = given_dates.get("end_date", "")
            # A Portfolio runs from its first transaction (or start_date when one is given,
            # for holdings, weights and presets) up to today.
            if portfolio_inputs is not None:
                start_date = given_dates.get("start_date", "")
                end_date = ""

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
            # A default limit for an indicator without a limit of its own selects rows as
            # limit= does: the latest value of a weekly loan series, the last 20 levels of a
            # daily one, rather than an answer cut at the size limit. A range given with
            # start_date/end_date is answered in full. Named in the answer.
            default_limit_note = None
            if (
                "limit" in method_defaults
                and "limit" not in accepted_params
                and rows_limit is None
                and not every_row
                and not given_dates
            ):
                row_selection = int(method_defaults["limit"])
                default_limit_note = (
                    f"limit={row_selection} applies by default "
                    f"({'the last' if row_selection < 0 else 'the first'} "
                    f"{abs(row_selection)} {'row' if abs(row_selection) == 1 else 'rows'}); "
                    "limit=0 gives every row, start_date/end_date another range."
                )
            for pname, value in method_defaults.items():
                if (
                    pname not in ("lookback_days", "lookahead_days")
                    and not (pname == "limit" and every_row)
                    and pname in accepted_params
                    and method_kwargs.get(pname) in (None, "")
                ):
                    method_kwargs[pname] = value

            # A call that would simulate or download more than the server can hold is
            # refused before anything is collected. The number of tickers and the years of
            # the range count in the limits only (pseudo parameters), as the work of a
            # pairwise or rolling indicator grows with them.
            if size_problem := size_limit_problem(
                method_kwargs | _pseudo_parameters(tickers, start_date, end_date),
                {
                    pname: per_method[method_name]
                    for pname, per_method in parameter_defaults.items()
                    if method_name in per_method
                },
                self._parameter_limits,
                method_name,
            ):
                record_outcome("invalid", tool_name, method_name)
                return (
                    f"Invalid input for `{tool_name}` (`{method_name}`): {size_problem}"
                )
            if refused := refused_value_problem(
                method_name, method_kwargs, self._parameter_limits
            ):
                record_outcome("invalid", tool_name, method_name)
                return f"`{tool_name}` (`{method_name}`): {refused}"

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
                record_outcome("invalid", tool_name, method_name)
                return (
                    f"`{tool_name}` (`{method_name}`) requires {listed}. Please provide "
                    f"{'it' if len(missing_inputs) == 1 else 'them'}; search_metrics with "
                    f"'{method_name}' explains each one."
                )

            # A transcript is of use only in full, and a full one is about as large as an
            # answer may be, so these return one item of one ticker and are never cut.
            full_text = method_name in self._full_text_methods
            if full_text:
                if tickers and len(tickers) > 1:
                    record_outcome("invalid", tool_name, method_name)
                    return (
                        f"`{tool_name}` (`{method_name}`) returns the full text for one ticker "
                        f"at a time. Please call it once per ticker, e.g. `tickers='{tickers[0]}'`."
                    )
                if method_kwargs.get("latest") is False:
                    record_outcome("invalid", tool_name, method_name)
                    return (
                        f"`{tool_name}` (`{method_name}`) returns one full text at a time. "
                        "Please pick one with `fiscal_period`, e.g. `fiscal_period='2025Q4'`; "
                        "without it, the most recent one is returned."
                    )
                if method_kwargs.get("fiscal_period") not in (None, ""):
                    try:
                        parse_fiscal_period(method_kwargs["fiscal_period"])
                    except ValueError as exc:
                        record_outcome("invalid", tool_name, method_name)
                        return (
                            f"Invalid input for `{tool_name}` (`{method_name}`): {exc}"
                        )

            unknown_notes = unknown_parameter_notes(
                tool_name, unknown_arguments, tool_parameters
            )

            # A period the toolkit does not know would surface as a confusing missing-item
            # error from deep inside the calculation.
            if method_kwargs.get("period") not in (None, ""):
                known_periods = (
                    INTRADAY_PERIODS if method_name == "get_intraday_data" else PERIODS
                )
                if str(method_kwargs["period"]).lower() not in known_periods:
                    record_outcome("invalid", tool_name, method_name)
                    return (
                        f"Invalid `period` for `{tool_name}` (`{method_name}`): "
                        f"'{method_kwargs['period']}'. Please use one of: "
                        f"{', '.join(known_periods)}."
                    )
                # The methods compare the period in lower case and know "yearly" rather than "annual", so the period they receive is the normalised one, not only the one checked.  # noqa: E501
                method_kwargs["period"] = _normalise_period(method_kwargs["period"])
            if requested_period is not None:
                requested_period = _normalise_period(requested_period)

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
                    record_outcome("invalid", tool_name, method_name)
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
                # e.g. a missing API key, so they are passed on rather than lost. The
                # sources of the data are named with the answer.
                with (
                    capture_call_messages() as call_messages,
                    collect_sources() as call_sources,
                    log_server_errors_once(),
                ):
                    if raw_series:
                        result = provider.call_econometrics_with_data(
                            method_name,
                            self._collect_series(
                                raw_series,
                                start_date=start_date,
                                end_date=end_date,
                                quarterly=quarterly,
                                benchmark_ticker=benchmark_ticker,
                                period=(
                                    method_kwargs.get("period")
                                    if "period" in given_params
                                    else None
                                ),
                                basket=toolkit_basket,
                                deflate_with=deflate_with,
                            ),
                            changes=changes,
                            **method_kwargs,
                        )
                    else:
                        call_arguments = {
                            "module_name": dispatch_module,
                            "method_name": method_name,
                            "category": dispatch_category,
                            "tickers": tickers,
                            "countries": countries,
                            "start_date": start_date,
                            "end_date": end_date,
                            "quarterly": quarterly,
                            "portfolio": portfolio_inputs,
                        }
                        result = provider.call_method(
                            **call_arguments,
                            benchmark_ticker=benchmark_ticker,
                            toolkit_risk_free_rate=toolkit_risk_free_rate,
                            toolkit_basket=toolkit_basket,
                            **method_kwargs,
                        )
                        upper_tickers = {ticker.upper() for ticker in tickers or []}
                        # The Toolkit leaves out a ticker that is also the benchmark and reports it as "Benchmark"; a Toolkit method without a benchmark (a quote, a profile) would lose that ticker altogether, so it is asked again without one. Sub-module methods (a beta) need the benchmark and keep it.  # noqa: E501
                        if (
                            dispatch_category == "toolkit"
                            and benchmark_ticker
                            and benchmark_ticker.upper() in upper_tickers
                            and len(upper_tickers) > 1
                            and not _mentions_benchmark(result)
                        ):
                            benchmark_ticker = None
                            result = provider.call_method(
                                **call_arguments,
                                benchmark_ticker=None,
                                toolkit_risk_free_rate=toolkit_risk_free_rate,
                                toolkit_basket=toolkit_basket,
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
                # An indicator without a period of its own is converted to the requested
                # one, so a long daily or weekly history fits a yearly answer.
                period_note = None
                # A growth rate over each row's own period (quarter-on-quarter GDP growth) read
                # once a year would be the last quarter's growth, not the year's, so such an
                # answer is not converted or sampled to a longer period.
                # Year-over-year growth is the growth over the past year in every row, so it
                # can be read once a year.
                growth_rates = (
                    method_name.endswith("_growth")
                    or method_kwargs.get("growth") is True
                ) and method_kwargs.get("year_over_year") is not True
                if (
                    requested_period
                    and isinstance(result, pd.DataFrame | pd.Series)
                    and (converted := convert_to_period(result, requested_period))
                    is not None
                ):
                    ignored_params.remove("period")
                    if growth_rates:
                        period_note = (
                            f"`{method_name}` has no period parameter and its values are "
                            f"growth rates over each {_row_period(result)}, so period="
                            f"'{requested_period}' was not applied: the last one of a "
                            f"{_period_name(requested_period)} is not the growth over that "
                            f"{_period_name(requested_period)}."
                            + (
                                " year_over_year=true gives the growth over the past year for every row."
                                if "year_over_year" in accepted_params
                                else ""
                            )
                        )
                    else:
                        result = converted
                        period_note = (
                            f"`{method_name}` has no period parameter, so period="
                            f"'{requested_period}' shows the last value of every period "
                            "(end of period), not an average."
                        )
                unmatched_columns: list[str] = []
                # Read before show_columns can leave out the Volume and Return they need.
                filled_days = filled_day_notes(result)
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
                        toolkit_risk_free_rate=toolkit_risk_free_rate,
                        toolkit_basket=toolkit_basket,
                    )
                # The Toolkit of a request is shared by its calls, so its fiscal years can
                # be relabelled by an earlier call on financial statements; that only
                # concerns an answer with years or quarters (not daily prices).
                if not _has_fiscal_periods(result):
                    notes = [
                        note
                        for note in notes
                        if not note.startswith("Fiscal years are shown")
                    ]
                # Tickers left out of an otherwise answered call are named, e.g. a symbol
                # the source does not know. A change missing for a country or ticker the
                # answer does not show (a series calculated for every country before the
                # requested ones are chosen) is not about this answer.
                shown_labels = (
                    _labels(result)
                    if isinstance(result, pd.DataFrame | pd.Series)
                    else None
                )
                notes = (
                    list(notes)
                    + filled_days
                    + [
                        note
                        for note in missing_data_notes(call_messages)
                        if shown_labels is None
                        or not (unchanged := NO_CHANGE_PATTERN.match(note))
                        or unchanged.group(1) in shown_labels
                    ]
                )
                if benchmark_swap_note:
                    notes.append(benchmark_swap_note)
                if default_limit_note:
                    notes.append(default_limit_note)
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
                    # A factor analysis and the market cap to GDP of a basket answer for the
                    # tickers together, not per ticker.
                    and method_name
                    not in ("get_factor_analysis", "get_market_cap_to_gdp")
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
                    # An answer without any of the tickers as a label (a portfolio's figures)
                    # is not one per ticker, so a ticker absent from it is not missing.
                    absent = (
                        [
                            ticker
                            for ticker in tickers
                            if ticker not in labels
                            and not any(ticker in note for note in notes)
                        ]
                        if labels.intersection(tickers)
                        else []
                    )
                    # A currency or cryptocurrency has prices but no financial statements,
                    # so a tool that reads them has nothing for it, whatever its symbol.
                    pairs = (
                        [ticker for ticker in absent if _is_currency_or_crypto(ticker)]
                        if spec.basket
                        else []
                    )
                    others = [ticker for ticker in absent if ticker not in pairs]
                    if pairs and len(absent) < len(tickers):
                        notes.append(
                            f"{', '.join(pairs)} {'is' if len(pairs) == 1 else 'are'} left "
                            "out: currencies and cryptocurrencies have no financial "
                            f"statements, which `{method_name}` is calculated from."
                        )
                    if others and len(absent) < len(tickers):
                        notes.append(
                            f"No data is available for {', '.join(others)}, so "
                            f"{'it is' if len(others) == 1 else 'they are'} left out. "
                            "Please check the symbol (e.g. ASML.AS for Amsterdam)."
                        )
                # After the check for missing tickers (a column the base window leaves
                # empty is not a ticker without data) and before show_columns, limit and
                # rank, so the base is the average over every fetched row of the window.
                if base_window is not None:
                    skipped_by = [
                        f"{pname}=true"
                        for pname in ("growth", "standardize")
                        if method_kwargs.get(pname) is True
                    ]
                    if skipped_by:
                        notes.append(
                            f"rebase_to={raw_rebase_to!r} was not applied: "
                            f"with {' and '.join(skipped_by)} the values are changes or "
                            "standard scores, which have no level to index."
                        )
                    else:
                        result, rebase_notes = rebase_result(
                            result, base_window, raw_rebase_to
                        )
                        notes.extend(rebase_notes)
                # After the check for missing tickers, which would otherwise count a ticker
                # that show_columns leaves out as one without data.
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
                # A signal is rare, so with signals_only each ticker keeps only its own
                # periods with a signal, and a limit gives every ticker its last ones. The
                # signals (or tickers) show_columns names are chosen first, so a period
                # is kept for a signal of those and not for one that is then left out.
                if (
                    method_kwargs.get("signals_only") is True
                    and isinstance(result, pd.DataFrame)
                    and result.columns.nlevels == 2  # noqa: PLR2004
                ):
                    if show_columns is not None:
                        chosen = result.columns.get_level_values(0).isin(
                            show_columns
                        ) | result.columns.get_level_values(1).isin(show_columns)
                        if chosen.any():
                            result = result.loc[:, chosen]
                        show_columns = None
                    answered = list(dict.fromkeys(result.columns.get_level_values(1)))
                    result = signal_rows(result)
                    quiet = [
                        ticker
                        for ticker in answered
                        if ticker not in set(result.index.get_level_values(0))
                    ]
                    if quiet:
                        notes.append(
                            f"{', '.join(quiet)} {'has' if len(quiet) == 1 else 'have'} no "
                            "signal in the period, so "
                            f"{'it is' if len(quiet) == 1 else 'they are'} left out."
                        )
                if show_columns is not None:
                    # A name of the rows (an index constituent's Symbol) is no column to
                    # keep, so asked for alone it leaves only those row names: a list of
                    # them for one name, a table of them for several.
                    row_names = (
                        [
                            name
                            for name in show_columns
                            if name in result.index.names
                            and name not in unmatched_columns
                        ]
                        if isinstance(result, pd.DataFrame)
                        else []
                    )
                    if row_names and len(row_names) + len(unmatched_columns) == len(
                        show_columns
                    ):
                        rows = result.index.to_frame(index=False)[row_names]
                        result = (
                            rows[row_names[0]].tolist()
                            if len(row_names) == 1
                            else rows.set_index(row_names[0])
                        )
                    else:
                        result = _filter_columns(result, show_columns)
                # After rebase_to (its base is the average over every day of the window)
                # and before limit, so limit=-10 gives the last ten weeks.
                if sample is not None and isinstance(result, pd.DataFrame | pd.Series):
                    sampled, sample_note = sample_rows(result, sample)
                    if growth_rates and sampled is not result:
                        sample_note = (
                            f"sample={sample!r} was not applied: the values are growth rates "
                            f"over each {_row_period(result)}, and the last one of a "
                            f"{_period_name(sample)} is not the growth over that "
                            f"{_period_name(sample)}."
                            + (
                                " year_over_year=true gives the growth over the past year for every row."
                                if "year_over_year" in accepted_params
                                else ""
                            )
                        )
                    else:
                        result = sampled
                    notes.append(sample_note)
                elif sample is not None:
                    notes.append(
                        f"sample={sample!r} was not applied: the answer is not a table with dates as its rows or columns."
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
                # An answer without values (an indicator that failed) is answered with
                # why, without notes on selecting its rows or ranking it.
                no_values = isinstance(result, pd.DataFrame | pd.Series) and not (
                    result.notna().to_numpy().any()
                )
                if (
                    row_selection
                    and isinstance(result, pd.DataFrame | pd.Series)
                    and not no_values
                ):
                    # One row per ticker (a figure over the whole period) has no first or
                    # last rows to speak of; cutting it would only drop tickers. The rows of
                    # the benchmark and of a portfolio of the tickers (component VaR) count
                    # as tickers.
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
                        and column_labels
                        <= {*(tickers or []), "Benchmark", "Portfolio"}
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
                        and row_labels <= {*tickers, "Benchmark", "Portfolio"}
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
                        # A column per country or ticker over dated rows: the last rows are those of the series published latest, so a series that lags behind (a month not published yet) is empty in them, while it has a value just before.  # noqa: E501
                        if (
                            row_selection < 0
                            and isinstance(selected, pd.DataFrame)
                            and len(selected.columns) > 1
                            and is_time_index_of(selected)
                            and not isinstance(selected.index, pd.MultiIndex)
                        ):
                            lagging = [
                                (column, result[column].last_valid_index())
                                for column in selected.columns
                                if selected[column].isna().all()
                                and result[column].notna().any()
                            ]
                            if lagging:
                                names = [
                                    (
                                        " ".join(map(str, column))
                                        if isinstance(column, tuple)
                                        else str(column)
                                    )
                                    + " (latest "
                                    + (
                                        latest.strftime("%Y-%m-%d")
                                        if isinstance(latest, pd.Timestamp)
                                        else str(latest)
                                    )
                                    + ")"
                                    for column, latest in lagging
                                ]
                                notes.append(
                                    (
                                        f"{', '.join(names)} {'has' if len(names) == 1 else 'have'} no "
                                        if len(names) <= MAXIMUM_NAMED_ROWS
                                        else f"{len(names):,} columns have no "
                                    )
                                    + "value for the selected periods (e.g. not published "
                                    "yet); a larger limit or start_date/end_date shows the "
                                    "latest value."
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
                                    f"{', '.join(names)} {'has' if len(names) == 1 else 'have'} no "
                                    if len(names) <= MAXIMUM_NAMED_ROWS
                                    else f"{len(names):,} rows have no "
                                )
                                + "value for the selected periods (e.g. not reported yet); "
                                f"limit={row_selection - 1 if row_selection < 0 else row_selection + 1} "
                                "includes one more period."
                            )
                # After the row selection, so limit=-1 ranks the latest period only.
                if rank_tickers and not no_values:
                    ranking = _rank_result(
                        result,
                        is_country=bool(countries),
                        entities=countries or tickers,
                    )
                    if ranking is None:
                        notes.append(
                            "rank=true was not applied: it needs a value per period and "
                            "ticker (or country) for two or more of them; pick one field "
                            "with show_columns where there are more."
                        )
                    else:
                        result = ranking
                        # Amounts in different currencies (a market cap in US dollars and
                        # one in euros) are ranked as the numbers they are, which says so.
                        ranked_tickers = [
                            str(label)
                            for label in ranking.index.get_level_values(-1).unique()
                        ]
                        if (
                            tickers
                            and not countries
                            and dispatch_category in ("ticker", "toolkit")
                            and _ranks_amounts(method_name, ranking)
                        ):
                            currencies = {
                                ticker: currency
                                for ticker, currency in provider.get_value_currencies(
                                    tickers=tickers,
                                    start_date=start_date,
                                    end_date=end_date,
                                    quarterly=quarterly,
                                    benchmark_ticker=benchmark_ticker,
                                    api_key=provider._api_key,
                                    fred_api_key=provider._fred_api_key,
                                    toolkit_risk_free_rate=toolkit_risk_free_rate,
                                    toolkit_basket=toolkit_basket,
                                ).items()
                                if ticker in ranked_tickers
                            }
                            if len(set(currencies.values())) > 1:
                                notes.append(
                                    "The values are in each ticker's own currency ("
                                    + ", ".join(
                                        f"{ticker}: {currency}"
                                        for ticker, currency in currencies.items()
                                    )
                                    + "), so the ranking compares unlike units; "
                                    "convert them to one currency before comparing."
                                )
                # Most economic series are yearly only, whatever quarterly asks for, and a
                # series published monthly (inflation) is not made quarterly either: without
                # a period, quarterly=true gives quarterly figures only where they are published.  # noqa: E501
                if quarterly and isinstance(result, pd.DataFrame | pd.Series):
                    shown_period = (
                        "yearly"
                        if is_yearly(result.index)
                        else (
                            observed_period(result.index)
                            if "countries" in accepted_params
                            and "period" not in given_params
                            else None
                        )
                    )
                    if shown_period in ("yearly", "monthly", "weekly", "daily"):
                        notes.append(
                            f"Quarterly data is not available for `{method_name}`, so the "
                            f"{shown_period} figures are shown."
                        )
                if indicator_note:
                    notes.insert(0, indicator_note)
                if period_note:
                    notes.append(period_note)
                if isinstance(result, pd.DataFrame | pd.Series) and (
                    skipped_note := skipped_periods_note(result)
                ):
                    notes.append(skipped_note)
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
                notes.extend(unknown_notes)
                # The scale of a figure that is easily misread: a daily tracking error is
                # no annual one, a variance no volatility.
                if (
                    isinstance(result, pd.DataFrame | pd.Series)
                    and not no_values
                    and not method_kwargs.get("growth")
                    and not method_kwargs.get("standardize")
                ):
                    unit_period = method_kwargs.get("period") or (
                        "quarterly" if quarterly else "yearly"
                    )
                    rolling = method_kwargs.get("rolling")
                    if (
                        method_name == "get_tracking_error"
                        and not rolling
                        and unit_period != "daily"
                    ):
                        notes.append(
                            "The tracking error is the standard deviation of the daily return "
                            f"differences within each {_period_name(unit_period)}, a daily figure: "
                            "multiply it by the square root of 252 for an annualized one."
                        )
                    elif (
                        method_name == "get_tracking_error"
                        and rolling
                        and unit_period in PERIODS_PER_YEAR
                    ):
                        notes.append(
                            f"The tracking error is the standard deviation of the {unit_period} "
                            f"return differences over {rolling} {_period_name(unit_period)}s, a "
                            f"{unit_period} figure: multiply it by the square root of "
                            f"{PERIODS_PER_YEAR[unit_period]} for an annualized one."
                        )
                    elif method_name == "get_garch_forecast":
                        garch_period = method_kwargs.get("period") or "daily"
                        notes.append(
                            f"The forecast is the variance of the {garch_period} return at each "
                            f"step ahead; its square root is the {garch_period} volatility."
                        )
                if unmatched_columns:
                    notes.insert(
                        1 if indicator_note else 0,
                        "show_columns: "
                        + ", ".join(f"'{name}'" for name in unmatched_columns)
                        + " is not part of this answer. Available: "
                        + ", ".join(available_columns[:40])
                        + ".",
                    )
                # A forecast's rows are steps ahead, not sessions, even when the first step is today.  # noqa: E501
                if (
                    dispatch_category in ("ticker", "toolkit")
                    and "forecast" not in method_name
                ):
                    session_note = _unclosed_session_note(result)
                    if session_note:
                        notes.append(session_note)
                # A note on how the benchmark was chosen only confuses an answer that has
                # no benchmark in it, such as an ETF's holdings.
                if not _mentions_benchmark(result):
                    notes = [note for note in notes if "the benchmark" not in note]
                    requested = {ticker.upper() for ticker in tickers or []}
                    # A requested ticker that is the benchmark is then simply absent, which needs saying; as the only ticker it is kept (with another benchmark).  # noqa: E501
                    if (
                        benchmark_ticker
                        and benchmark_ticker.upper() in requested
                        and len(requested) > 1
                        and isinstance(result, pd.DataFrame | pd.Series)
                        and benchmark_ticker not in _labels(result)
                    ):
                        notes.append(
                            f"{benchmark_ticker} is the benchmark, so it is left out as a "
                            "ticker; set benchmark_ticker to another ticker to include it."
                        )
                reason = summarize_reasons(call_messages)
                answer_outcome: list[str] = []
                formatted = format_result(
                    result,
                    notes=notes or None,
                    sources=readable_sources(call_sources),
                    reason=reason
                    or _short_window_reason(dispatch_category, start_date, end_date),
                    max_characters=None if full_text else answer_size,
                    outcome=answer_outcome,
                )
                # An answer without data for want of an API key is a call that could not
                # be made, not a request without data.
                record_outcome(
                    (
                        "invalid"
                        if answer_outcome == ["empty"]
                        and reason in (FMP_KEY_HINT, FRED_KEY_HINT)
                        else (answer_outcome[0] if answer_outcome else "data")
                    ),
                    tool_name,
                    method_name,
                )
                return formatted
            except KeyError as exc:
                # A missing key is missing data (a ticker, column or date the source did not
                # return), not a mistake in the call.
                record_outcome("empty", tool_name, method_name)
                return f"No data for `{tool_name}` (`{method_name}`): {redact(exc)} is missing."
            except ValueError as exc:
                # An InvalidArgumentError, a refused value or a missing API key: a mistake in
                # the call (data errors are an empty answer through handle_errors).
                record_outcome("invalid", tool_name, method_name)
                return (
                    f"Invalid input for `{tool_name}` (`{method_name}`): {redact(exc)}"
                )
            except TypeError as exc:
                # The inputs an indicator requires are checked above, so this is unexpected.
                record_outcome("error", tool_name, method_name)
                return f"Parameter error for `{tool_name}` (`{method_name}`): {redact(exc)}"
            except ConnectionError as exc:
                record_outcome("error", tool_name, method_name)
                return f"API connection failed: {redact(exc)}"
            except Exception as exc:
                record_outcome("error", tool_name, method_name)
                # The traceback repeats the exception text, which can hold a request URL
                # with the caller's API key, so both are redacted.
                logger.warning(
                    "Tool %s (%s) failed: %s\n%s",
                    tool_name,
                    method_name,
                    redact(exc),
                    redact(traceback.format_exc()),
                )
                return f"`{tool_name}` (`{method_name}`) failed with error: {type(exc).__name__}: {redact(exc)}"

        P = inspect.Parameter
        POS = P.POSITIONAL_OR_KEYWORD
        indicator_choices = group_methods
        # The choices are advertised as an enum, but any text is accepted: an indicator of
        # another tool (or a near miss) is then answered with where it is, rather than with
        # a validation error that only lists this tool's indicators.
        # Several indicators arrive as a list, which becomes comma-separated text.
        indicator_ann = (
            Annotated[
                str,
                PydanticField(json_schema_extra={"enum": list(indicator_choices)}),
                BeforeValidator(text_from_parsed_json),
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
        if spec.risk_free_rate:
            common.append(
                P(
                    "risk_free_rate",
                    POS,
                    default="10y",
                    annotation=Annotated[
                        str,
                        PydanticField(
                            json_schema_extra={
                                "enum": list(validation_model.RISK_FREE_RATE_OPTIONS)
                            }
                        ),
                    ],
                )
            )
        if spec.basket:
            common.append(P("basket", POS, default=False, annotation=bool))
        portfolio_tool = category == inspector.categories.get("portfolio", "portfolio")
        sig_params = [indicator_param] + [
            P(
                p.name,
                POS,
                default=p.default,
                annotation=_annotate_with_description(
                    p.name,
                    p.annotation,
                    description=(
                        _PORTFOLIO_PARAM_DESCRIPTIONS.get(p.name)
                        if portfolio_tool
                        else None
                    ),
                ),
            )
            for p in common
        ]

        # The schema says which indicators take a parameter, in a few words; search_metrics
        # with an indicator's name gives the full description, options and default of each.
        for p in extra_params:
            ann = (
                _simplify_annotation(p.annotation)
                if p.annotation is not P.empty
                else str
            )
            takers = [m for m in group_methods if p.name in method_param_names[m]]
            required = [
                m
                for m in takers
                if parameter_defaults.get(p.name, {}).get(m) is P.empty
            ]
            documented = min(
                (
                    text
                    for m in takers
                    if (
                        text := _docstring_arguments(
                            inspect.getdoc(getattr(owning_class(m), m, None)) or ""
                        ).get(p.name)
                    )
                ),
                key=len,
                default="",
            )
            description = _schema_parameter_description(
                tool_name, p.name, takers, group_methods, required, documented
            )

            if p.name in disputed_params:
                # Advertised as unset, because no single default is correct for the whole group; search_metrics gives each indicator's own.  # noqa: E501
                default = None
                annotation = _annotate_with_description(
                    p.name,
                    typing.Optional[ann],  # noqa: UP045  # ty: ignore[invalid-type-form]  # fmt: skip
                    description=description,
                )
            else:
                default = p.default if p.default is not P.empty else ""
                annotation = _annotate_with_description(
                    p.name, ann, description=description
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
                for name in ("series", "changes", "deflate_with")
            )

        # Appended last so it does not disturb positional ordering of other params.
        if "rank" not in {p.name for p in sig_params}:
            sig_params.append(
                P(
                    "rank",
                    POS,
                    default=False,
                    annotation=_annotate_with_description("rank", bool),
                )
            )
        sig_params.append(
            P(
                "rebase_to",
                POS,
                default=None,
                annotation=_annotate_with_description("rebase_to", str | None),
            )
        )
        sig_params.append(
            P(
                "sample",
                POS,
                default=None,
                annotation=_annotate_with_description("sample", str | None),
            )
        )
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
        # search_metrics can point at the indicator itself rather than only its tool, its
        # formula variants, which search_metric_categories counts as metrics, and its
        # parameters, which the tool schema only names.
        entries = []
        for method_name in group_methods:
            owner = (method_to_cls or {}).get(method_name) or cls
            entries.append(
                {
                    "tool": spec.tool_name,
                    "indicator": method_name,
                    # The method itself, so an indicator several tools serve counts once
                    # and two methods that share a name (an ETF's and a company's
                    # get_profile) count as two.
                    "key": _method_key(owner, method_name),
                    "variants": self._offered_variants(
                        method_name, getattr(owner, method_name, None)
                    ),
                    **_summarize_docstring(
                        inspect.getdoc(getattr(owner, method_name, None)) or ""
                    ),
                    **_describe_docstring(
                        inspect.getdoc(getattr(owner, method_name, None)) or ""
                    ),
                }
            )
            entries[-1]["parameters"], entries[-1]["call_parameters"] = (
                self._describe_parameters(
                    spec.tool_name, method_name, getattr(owner, method_name, None)
                )
            )
        # The parameters most of the tool's indicators take (growth, lag, period), which a
        # search result names once per tool rather than in every indicator's call line.
        taken = Counter(name for entry in entries for name in entry["parameters"])
        common = [
            name
            for name, count in taken.most_common()
            if len(entries) >= MINIMUM_INDICATORS_FOR_COMMON_PARAMETERS
            and count >= COMMON_PARAMETER_SHARE * len(entries)
        ]
        for entry in entries:
            entry["common_parameters"] = common
        self._indicator_index.extend(entries)

        method_list = ", ".join(group_methods)
        # The indicators are the enum of the indicator parameter, so the tool description
        # does not repeat them; the category listing (search_by_category) does show them.
        description = spec.description or f"{spec.display_name}."
        index_description = f"{description}\n\nAvailable indicators: {method_list}."
        compact = compact_tools_enabled()
        if compact:
            description = _compact_description(description, group_methods[0])

        try:
            fn = self._build_router_wrapper(
                spec,
                cls,
                group_methods,
                extra_params,
                method_to_cls=method_to_cls,
                method_dispatch=method_dispatch,
            )
            # The tool's own parameters (the dates, growth, rolling), which apply to all of
            # its indicators, with their full descriptions for search_metrics.
            indicator_parameters = {p.name for p in extra_params}
            tool_parameters = {
                name: (
                    _PORTFOLIO_PARAM_DESCRIPTIONS.get(name)
                    if spec.category
                    == self._inspector.categories.get("portfolio", "portfolio")
                    else None
                )
                or _PARAM_DESCRIPTIONS.get(name, "")
                for name in fn.__signature__.parameters
                if name not in indicator_parameters and name not in UNIVERSAL_PARAMETERS
            }
            for entry in entries:
                entry["tool_parameters"] = tool_parameters
            fn.__name__ = spec.tool_name
            fn.__doc__ = description
            self._router_functions[spec.tool_name] = fn
            fn = run_in_thread(fn)

            if self._analytics is not None:
                fn = self._analytics.counted(spec.tool_name, fn, _indicator_count)

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
                structured_output=False,
            )
            index_cat = spec.index_category or spec.module_name
            if index_cat not in self._tool_index:
                self._tool_index[index_cat] = []
            self._tool_index[index_cat].append(
                {"tool": spec.tool_name, "description": index_description}
            )
            _strip_schema_titles(self._mcp, spec.tool_name)
            _offer_indicator_list(self._mcp, spec.tool_name)
            _fit_tool_listing(self._mcp, spec.tool_name, indicator_parameters)
            keep_unknown_arguments(self._mcp, spec.tool_name)
            if compact:
                _strip_schema_descriptions(self._mcp, spec.tool_name)
            # search_metrics explains the tool's parameters its schema leaves undescribed and
            # names the others.
            tool_manager = getattr(self._mcp, "_tool_manager", None)
            tool = tool_manager.get_tool(spec.tool_name) if tool_manager else None
            properties = tool.parameters.get("properties", {}) if tool else {}
            for name in tool_parameters:
                if properties.get(name, {}).get("description"):
                    tool_parameters[name] = ""
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
