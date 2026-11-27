"""Technicals Helpers Module"""

__docformat__ = "google"

import functools
import inspect
import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from financetoolkit.helpers import get_non_trading_days
from financetoolkit.utilities.dataframe_model import FilteredClass
from financetoolkit.utilities.logger_model import get_logger, log_each_message_once

logger = get_logger()

# The observations in a year of a ticker that trades on an exchange's trading days, and of
# one that also trades on weekends, such as a cryptocurrency.
TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365

# The indicators a collection names when it reports those without values.
MAXIMUM_REPORTED_INDICATORS = 10

PERIODS = ("intraday", "daily", "weekly", "monthly", "quarterly", "yearly")

# The indicators per collection that get_factor_analysis compares across tickers: a method name, or
# a method with the parameters of one of its variants, such as ("get_moving_average",
# {"method": "t3"}), so that each variant of a method is evaluated. The Absolute Price Oscillator
# is the MACD line, the default output of get_moving_average_convergence_divergence. Left out are
# the market-wide breadth measures (one value for all tickers), the volume-scaled ones (on
# balance volume and Archer's version, price volume trend, accumulation/distribution, Chaikin oscillator, force
# index and its running total, ease of movement, Klinger volume oscillator, market facilitation index, price-volume),
# whose level reflects how many shares trade, the running totals in price units (Williams' accumulation/distribution),
# the price volume rank (four categories), the volume profile (not per date), the pivot points, Fibonacci levels,
# price transforms and Zig Zag, which only restate the latest high, low and close or swing, Wilder's swing
# index, which is scaled by a limit move in price units and accumulates from the first date of the data, the
# Hilbert dominant cycle phase (an angle that wraps from 315 to -45 degrees every cycle, so its rank says
# nothing), and the Hilbert trend mode and the trend flags of Archer's moving averages trends and the TTM Trend
# (two values each, no ranking).
FACTOR_COLLECTIONS: dict[str, list[str | tuple[str, dict]]] = {
    "momentum": [
        "get_money_flow_index",
        "get_williams_percent_r",
        "get_aroon_indicator",
        "get_commodity_channel_index",
        "get_relative_vigor_index",
        "get_ultimate_oscillator",
        ("get_moving_average_convergence_divergence", {"method": "percentage"}),
        "get_percentage_volume_oscillator",
        "get_detrended_price_oscillator",
        "get_average_directional_index",
        "get_directional_movement_index",
        ("get_average_directional_index", {"method": "rating"}),
        "get_chande_momentum_oscillator",
        ("get_relative_strength_index", {"method": "intraday"}),
        "get_ichimoku_cloud",
        "get_stochastic_oscillator",
        "get_moving_average_convergence_divergence",
        "get_relative_strength_index",
        ("get_stochastic_oscillator", {"method": "rsi"}),
        "get_true_strength_index",
        "get_balance_of_power",
        "get_internal_bar_strength",
        "get_qstick",
        "get_awesome_oscillator",
        ("get_awesome_oscillator", {"method": "accelerator"}),
        "get_vortex_indicator",
        "get_elder_ray_index",
        "get_rate_of_change",
        ("get_rate_of_change", {"method": "absolute"}),
        "get_choppiness_index",
        "get_efficiency_ratio",
        "get_vertical_horizontal_filter",
        "get_know_sure_thing",
        "get_coppock_curve",
        "get_fisher_transform",
        "get_inverse_fisher_transform_rsi",
        "get_kdj_indicator",
        "get_premier_stochastic_oscillator",
        "get_stochastic_momentum_index",
        ("get_true_strength_index", {"method": "ergodic"}),
        "get_schaff_trend_cycle",
        "get_wavetrend_oscillator",
        "get_derivative_oscillator",
        "get_quantitative_qualitative_estimation",
        ("get_relative_strength_index", {"method": "dynamic"}),
        "get_price_zone_oscillator",
        ("get_relative_strength_index", {"method": "connors"}),
        "get_inertia",
        "get_bias",
        "get_brar",
        "get_pretty_good_oscillator",
        "get_psychological_line",
        "get_volatility_based_momentum",
        "get_correlation_trend_indicator",
        "get_forecast_oscillator",
        "get_projection_oscillator",
    ],
    "overlap": [
        "get_moving_average",
        "get_trix",
        ("get_moving_average", {"method": "kaufman_adaptive"}),
        ("get_moving_average", {"method": "t3"}),
        ("get_moving_average", {"method": "arnaud_legoux"}),
        ("get_moving_average", {"method": "volume_weighted"}),
        ("get_moving_average", {"method": "mcginley"}),
        ("get_moving_average", {"method": "variable_index_dynamic"}),
        ("get_moving_average", {"method": "fractal_adaptive"}),
        ("get_moving_average", {"method": "holt_winters"}),
        "get_holt_winters_channel",
        ("get_moving_average", {"method": "elastic_volume_weighted"}),
        ("get_moving_average", {"method": "volume_adjusted"}),
        "get_midpoint",
        "get_volume_weighted_average_price",
        "get_parabolic_sar",
        "get_gann_hilo_activator",
    ],
    "volatility": [
        "get_bollinger_bands",
        "get_bollinger_percent_b_and_bandwidth",
        "get_true_range",
        "get_average_true_range",
        "get_mass_index",
        "get_supertrend",
        "get_chandelier_exit",
        "get_keltner_channels",
        "get_donchian_channels",
        ("get_average_true_range", {"method": "normalized"}),
        ("get_average_true_range", {"method": "day_range"}),
        "get_chaikin_volatility",
        "get_relative_volatility_index",
        "get_acceleration_bands",
        "get_chande_kroll_stop",
        "get_adaptive_price_zone",
        ("get_keltner_channels", {"method": "aberration"}),
        ("get_supertrend", {"method": "profit_maximizer"}),
        "get_price_distance",
        "get_market_thermometer",
        "get_squeeze_momentum",
    ],
    "breadth": [
        "get_chaikin_money_flow",
        "get_volume_index",
        ("get_volume_index", {"method": "positive"}),
        "get_relative_volume",
        "get_volume_flow_indicator",
        "get_finite_volume_elements",
        ("get_chaikin_money_flow", {"method": "twiggs"}),
        "get_volume_zone_oscillator",
        "get_buying_selling_pressure",
    ],
    "cycle": [
        "get_hilbert_dominant_cycle_period",
        "get_hilbert_phasor_components",
        "get_hilbert_sine_wave",
        ("get_moving_average", {"method": "hilbert_trendline"}),
        "get_mesa_adaptive_moving_average",
        "get_swiss_army_knife_filter",
        ("get_moving_average", {"method": "super_smoother"}),
        "get_even_better_sinewave",
        "get_detrended_synthetic_price",
        ("get_hilbert_sine_wave", {"method": "mesa"}),
        "get_center_of_gravity_oscillator",
        ("get_relative_strength_index", {"method": "laguerre"}),
    ],
}
FACTOR_COLLECTIONS["all"] = [
    metric for metrics in FACTOR_COLLECTIONS.values() for metric in metrics
]

# Indicators in price units, which get_factor_analysis makes comparable across tickers: a
# price line (an average, band or stop) as the distance of the close from it, close / line - 1,
# and a price difference (a range or an oscillator of two averages) as a share of the close. A
# method name covers every variant of the method, and a method with parameters, such as
# ("get_swiss_army_knife_filter", {"filter": "butterworth"}), only that variant: the low-pass
# filters of the Swiss Army Knife return smoothed prices, its high-pass and band-pass filters
# oscillate around zero.
FACTOR_PRICE_LINES = (
    "get_moving_average",
    "get_holt_winters_channel",
    "get_midpoint",
    "get_price_transform",
    "get_volume_weighted_average_price",
    "get_parabolic_sar",
    "get_gann_hilo_activator",
    "get_zigzag",
    "get_ichimoku_cloud",
    "get_bollinger_bands",
    "get_supertrend",
    "get_chandelier_exit",
    "get_keltner_channels",
    "get_donchian_channels",
    "get_acceleration_bands",
    "get_chande_kroll_stop",
    "get_adaptive_price_zone",
    "get_pivot_points",
    "get_fibonacci_retracement_levels",
    "get_mesa_adaptive_moving_average",
    ("get_swiss_army_knife_filter", {"filter": "butterworth"}),
    ("get_swiss_army_knife_filter", {"filter": "gaussian"}),
)
FACTOR_PRICE_DIFFERENCES = (
    "get_true_range",
    ("get_average_true_range", {"method": "standard"}),
    ("get_average_true_range", {"method": "day_range"}),
    "get_price_distance",
    "get_market_thermometer",
    "get_squeeze_momentum",
    ("get_moving_average_convergence_divergence", {"method": "absolute"}),
    ("get_rate_of_change", {"method": "absolute"}),
    "get_qstick",
    "get_awesome_oscillator",
    "get_elder_ray_index",
    "get_detrended_price_oscillator",
    "get_force_index",
    "get_hilbert_phasor_components",
    ("get_swiss_army_knife_filter", {"filter": "high_pass"}),
    ("get_swiss_army_knife_filter", {"filter": "two_pole_high_pass"}),
    ("get_swiss_army_knife_filter", {"filter": "band_pass"}),
    "get_detrended_synthetic_price",
)


def is_factor_listed(
    entries: tuple[str | tuple[str, dict], ...], name: str, arguments: dict
) -> bool:
    """
    Returns whether an indicator, called with the given arguments, is one of the entries of FACTOR_PRICE_LINES or
    FACTOR_PRICE_DIFFERENCES.

    A method name covers every variant, and a method with parameters only the calls whose arguments have those
    values.

    Args:
        entries (tuple[str | tuple[str, dict], ...]): The method names and variants.
        name (str): The method name of the indicator.
        arguments (dict): The value of each parameter of the call, the default where none
            is given.

    Returns:
        bool: Whether the indicator is listed.
    """
    for entry in entries:
        entry_name, variant = (entry, {}) if isinstance(entry, str) else entry
        if entry_name == name and all(
            arguments.get(key) == value for key, value in variant.items()
        ):
            return True

    return False


def _has_no_values(result: object) -> bool:
    """Whether a result has numbers to show, but every one of them is NaN."""
    if isinstance(result, pd.Series):
        numbers = result if pd.api.types.is_numeric_dtype(result) else pd.Series()
    elif isinstance(result, pd.DataFrame):
        numbers = result.select_dtypes("number")
    else:
        return False

    return numbers.size > 0 and not numbers.notna().to_numpy().any()


def _reported_method(function: Callable, name: str) -> Callable:
    """
    Wraps an indicator (get_) or a collection (collect_) of the Technicals class so that a
    result without values is reported, see `report_missing_values`.
    """
    signature = inspect.signature(function)
    # The factor analysis calculates many indicators, so it reports them as a collection does.
    collection = name.startswith("collect_") or name == "get_factor_analysis"

    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        # A method called on another object (such as one standing in for the Technicals in
        # get_factor_analysis) has nothing to report on.
        if not hasattr(self, "_collecting"):
            return function(self, *args, **kwargs)

        # Each message is logged once per call, also when the indicators an indicator is built from report the same window, and again on the next call of the same instance (the MCP server keeps a Toolkit for many requests).  # noqa: E501
        with log_each_message_once():
            arguments = signature.bind_partial(self, *args, **kwargs).arguments
            period = arguments.get("period", signature.parameters.get("period"))
            if isinstance(period, inspect.Parameter):
                period = period.default
            if name == "get_factor_analysis":
                # Its period is that of the returns; the indicators are calculated on the daily data.  # noqa: E501
                period = "daily"
            outermost = not self._collecting and not self._depth

            if outermost:
                self._window_too_long = False
            if collection and not self._collecting:
                self._long_windows, self._without_values = set(), {}

            self._collecting += collection
            self._depth += 1
            try:
                result = function(self, *args, **kwargs)
            finally:
                self._collecting -= collection
                self._depth -= 1

            historical_data = (
                self._historical_data.get(period) if isinstance(period, str) else None
            )
            with_data = (
                historical_data is not None
                and not historical_data.loc[self._start_date : self._end_date]
                .dropna(how="all")
                .empty
            )

            if collection and not self._collecting:
                report_collection(self, name, period, historical_data)
            elif not collection and with_data and _has_no_values(result):
                if self._collecting:
                    self._without_values[name] = None
                elif outermost and not self._window_too_long:
                    logger.warning(
                        "%s has no values for the %s %s periods of data: its windows (or the "
                        "periods it needs to warm up) are longer than the data. Use a "
                        "higher-frequency period, shorter windows or a longer date range.",
                        name,
                        len(historical_data.dropna(how="all")),
                        period,
                    )

            return result

    return wrapper


def report_collection(
    technicals, name: str, period: str, historical_data: pd.DataFrame | None
) -> None:
    """
    Logs one message for a collection of indicators: the windows longer than the data and
    the indicators left without values, rather than one message per indicator.

    Args:
        technicals (Technicals): The Technicals instance that collected the indicators.
        name (str): The name of the collection, e.g. "collect_all_indicators".
        period (str): The period of the collection.
        historical_data (pd.DataFrame | None): The historical data of the period.
    """
    if not technicals._long_windows and not technicals._without_values:
        return

    periods = 0 if historical_data is None else len(historical_data.dropna(how="all"))
    windows = sorted(technicals._long_windows)
    without_values = list(technicals._without_values)
    # A yearly collection leaves most of the indicators without values, so the names are
    # shortened to the first ones.
    names = ", ".join(without_values[:MAXIMUM_REPORTED_INDICATORS]) + (
        f" and {len(without_values) - MAXIMUM_REPORTED_INDICATORS} more"
        if len(without_values) > MAXIMUM_REPORTED_INDICATORS
        else ""
    )
    logger.warning(
        "%s over %s %s periods of data: %s%s%s. Use a higher-frequency period, shorter "
        "windows or a longer date range.",
        name,
        periods,
        period,
        (
            f"the windows of {', '.join(map(str, windows))} periods are longer than the data"
            if windows
            else ""
        ),
        " and " if windows and without_values else "",
        (
            f"{len(without_values)} indicators have no values ({names})"
            if without_values
            else ""
        ),
    )


def report_missing_values(cls: FilteredClass) -> FilteredClass:
    """
    Class decorator that reports an indicator without values: a result that is NaN
    throughout while the data of its period is not, as when a window, a signal window
    or the warm-up of an indicator is longer than the data (such as a one- or two-row
    history). A standalone indicator logs it once per call; a collection of indicators (and
    the factor analysis) logs one message naming the long windows and the indicators without
    values.

    Args:
        cls (type): A class of Technicals indicators (a topic or the base class).

    Returns:
        type: The same class.
    """
    for name, member in list(vars(cls).items()):
        if name.startswith(("get_", "collect_")) and callable(member):
            setattr(cls, name, _reported_method(member, name))

    return cls


def calculate_per_ticker(
    historical_data: pd.DataFrame,
    columns: list[str],
    function: Callable[..., pd.Series | pd.DataFrame],
    *args,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculates an indicator per ticker over the dates the ticker traded.

    An indicator over a number of periods counts bars, so the flat bars a ticker gets on the
    dates only another ticker traded (see `get_non_trading_days`) would count as periods
    without a move: they lower an Average True Range or a Bollinger Band width, lengthen
    every window in calendar time and turn a ratio of ranges into 0 / 0. Each ticker is
    therefore calculated on its own trading days, and a result per date is placed back on
    every date of the data, NaN on the dates the ticker did not trade. A ticker that traded
    on every date is calculated on its data as it is, so a Toolkit whose tickers share one
    calendar gets the same results as a calculation over all tickers at once. A result that
    is not per date, such as the Volatility Cone, is returned as it is.

    Args:
        historical_data (pd.DataFrame): The historical data of one period with (column,
            ticker) columns.
        columns (list[str]): The columns the function takes, in its order, e.g.
            ["High", "Low", "Close"]. The tickers are those of the first column.
        function (Callable[..., pd.Series | pd.DataFrame]): The indicator, which takes a
            Series per column followed by `args` and `kwargs`.
        *args: The further positional arguments of the function, such as the window.
        **kwargs: The further keyword arguments of the function.

    Returns:
        pd.DataFrame: The indicator with a column per ticker when the function returns a
        Series, and (indicator column, ticker) columns when it returns a DataFrame.
    """
    non_trading_days = get_non_trading_days(historical_data, columns[0])

    results = {}
    for ticker in historical_data[columns[0]].columns:
        inputs = [historical_data[column][ticker] for column in columns]
        traded = ~non_trading_days[ticker]

        if traded.all():
            results[ticker] = function(*inputs, *args, **kwargs)
            continue

        inputs = [values[traded] for values in inputs]
        result = function(*inputs, *args, **kwargs)
        results[ticker] = (
            result.reindex(historical_data.index)
            if result.index.equals(inputs[0].index)
            else result
        )

    if all(isinstance(result, pd.Series) for result in results.values()):
        return pd.DataFrame(results)

    return pd.concat(results, axis=1).swaplevel(1, 0, axis=1).sort_index(axis=1)


def calculate_on_complete_bars(
    *prices: pd.Series,
    indicator: Callable[..., pd.Series | pd.DataFrame],
    **kwargs,
) -> pd.Series | pd.DataFrame:
    """
    Calculates an indicator over the bars of one ticker that have every price, and places the
    result back on every date of the prices, NaN on the incomplete bars. A pattern compares
    neighbouring bars, so an incomplete bar (a later listing or a missing day) would shift
    the bars it compares.

    Args:
        *prices (pd.Series): The prices the function takes, in its order, e.g. the Open,
            High, Low and Close of one ticker.
        indicator (Callable[..., pd.Series | pd.DataFrame]): The indicator, which takes a
            Series per price followed by `kwargs`.
        **kwargs: The further keyword arguments of the indicator.

    Returns:
        pd.Series | pd.DataFrame: The indicator on the dates of the prices.
    """
    bars = pd.concat(prices, axis=1).dropna()
    return indicator(
        *(bars.iloc[:, column] for column in range(bars.shape[1])), **kwargs
    ).reindex(prices[0].index)


def apply_on_own_closes(
    prices_close: pd.DataFrame, function: Callable[[pd.Series], pd.Series]
) -> pd.DataFrame:
    """
    Applies a function to each ticker's closes over the dates the ticker has a close, and places the result back on
    every date, NaN on the dates without a close.

    A change, moving average or rolling high of a ticker thereby spans its own trading days, not the dates only
    another ticker traded, such as the weekends of a cryptocurrency next to a stock.

    Args:
        prices_close (pd.DataFrame): The closes, a column per ticker, NaN on the dates a
            ticker did not trade.
        function (Callable[[pd.Series], pd.Series]): The calculation over one ticker's
            closes, e.g. `lambda close: close.diff()`.

    Returns:
        pd.DataFrame: The result per date and ticker.
    """
    return pd.DataFrame(
        {
            ticker: function(prices_close[ticker].dropna()).reindex(prices_close.index)
            for ticker in prices_close.columns
        },
        index=prices_close.index,
        columns=prices_close.columns,
    )


def get_observations_per_year(prices: pd.Series) -> int:
    """
    The number of daily observations in a year of a ticker, to annualize a daily measure or to express a year in
    periods.

    It is 252 for a ticker that trades on weekdays only, and 365 for one with prices on a Saturday or Sunday, such
    as a cryptocurrency that trades every calendar day. Counting the prices of each year instead would give a stock
    between 250 and 253 depending on its holidays, so a ticker on an exchange's trading days keeps the 252
    convention.

    Args:
        prices (pd.Series): The daily (or intraday) prices with a PeriodIndex or
            DatetimeIndex, NaN on the dates without a price.

    Returns:
        int: The observations per year, 252 or 365.
    """
    dates = prices.index[prices.notna().to_numpy()]

    return (
        CALENDAR_DAYS_PER_YEAR
        if bool((dates.dayofweek >= 5).any())  # noqa: PLR2004
        else TRADING_DAYS_PER_YEAR
    )


def apply_variable_smoothing(
    prices: pd.Series, smoothing_factors: pd.Series, window: int
) -> pd.Series:
    """
    Smooths prices with a smoothing factor that changes every period, as the adaptive averages do.

    These are the Variable Index Dynamic Average and the Fractal Adaptive Moving Average:

    - Average(t) = Average(t-1) + Factor(t) * (Price(t) - Average(t-1))

    The recursion is seeded with the simple average of `window` prices, placed on the
    period before the first factor that follows a full window, with the periods before it
    missing, as the Toolkit seeds its exponential moving averages (TA-Lib's convention).
    A period without a factor keeps the last average, and a period without a price is
    missing while the recursion continues from the last average on the next price.

    Args:
        prices (pd.Series): The prices to smooth.
        smoothing_factors (pd.Series): The smoothing factor per period, between 0 and 1,
            on the index of `prices`.
        window (int): The number of prices of the simple average that seeds the recursion.

    Returns:
        pd.Series: The smoothed prices, missing before the seed.
    """
    averages = [math.nan] * len(prices)
    simple_moving_average = prices.rolling(window=window).mean().to_numpy(dtype=float)
    factors = smoothing_factors.to_numpy(dtype=float)
    price_values = prices.to_numpy(dtype=float)

    # The recursion starts on the first factor whose previous period has a full window.
    eligible = np.isfinite(factors[1:]) & np.isfinite(simple_moving_average[:-1])
    if not eligible.any():
        return pd.Series(averages, index=prices.index, dtype="float64")

    start = int(np.argmax(eligible)) + 1
    previous_average = simple_moving_average[start - 1]
    averages[start - 1] = previous_average

    # The recursion runs on plain Python floats, since writing a pandas Series one element
    # at a time costs tens of microseconds per element.
    for i, (price, factor) in enumerate(
        zip(price_values[start:].tolist(), factors[start:].tolist()), start=start
    ):
        if not (math.isnan(price) or math.isnan(factor)):
            previous_average = previous_average + factor * (price - previous_average)
        averages[i] = previous_average

    return pd.Series(averages, index=prices.index, dtype="float64").where(
        prices.notna()
    )
