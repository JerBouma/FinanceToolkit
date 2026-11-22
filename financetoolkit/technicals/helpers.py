"""Technicals Helpers Module"""

__docformat__ = "google"

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from financetoolkit.helpers import get_non_trading_days

# The observations in a year of a ticker that trades on an exchange's trading days, and of
# one that also trades on weekends, such as a cryptocurrency.
TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365

PERIODS = ("intraday", "daily", "weekly", "monthly", "quarterly", "yearly")

# The indicators per collection that get_factor_analysis compares across tickers. Left out are
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
FACTOR_COLLECTIONS = {
    "momentum": [
        "get_money_flow_index",
        "get_williams_percent_r",
        "get_aroon_indicator",
        "get_commodity_channel_index",
        "get_relative_vigor_index",
        "get_ultimate_oscillator",
        "get_percentage_price_oscillator",
        "get_absolute_price_oscillator",
        "get_percentage_volume_oscillator",
        "get_detrended_price_oscillator",
        "get_average_directional_index",
        "get_directional_movement_index",
        "get_average_directional_index_rating",
        "get_chande_momentum_oscillator",
        "get_intraday_momentum_index",
        "get_ichimoku_cloud",
        "get_stochastic_oscillator",
        "get_moving_average_convergence_divergence",
        "get_relative_strength_index",
        "get_stochastic_relative_strength_index",
        "get_true_strength_index",
        "get_balance_of_power",
        "get_internal_bar_strength",
        "get_qstick",
        "get_awesome_oscillator",
        "get_accelerator_oscillator",
        "get_vortex_indicator",
        "get_elder_ray_index",
        "get_rate_of_change",
        "get_momentum",
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
        "get_smi_ergodic_indicator",
        "get_schaff_trend_cycle",
        "get_wavetrend_oscillator",
        "get_derivative_oscillator",
        "get_quantitative_qualitative_estimation",
        "get_dynamic_momentum_index",
        "get_price_zone_oscillator",
        "get_connors_relative_strength_index",
        "get_inertia",
        "get_bias",
        "get_brar",
        "get_pretty_good_oscillator",
        "get_psychological_line",
        "get_volatility_based_momentum",
    ],
    "overlap": [
        "get_moving_average",
        "get_trix",
        "get_kaufman_adaptive_moving_average",
        "get_t3_moving_average",
        "get_arnaud_legoux_moving_average",
        "get_volume_weighted_moving_average",
        "get_mcginley_dynamic",
        "get_variable_index_dynamic_average",
        "get_fractal_adaptive_moving_average",
        "get_holt_winters_moving_average",
        "get_holt_winters_channel",
        "get_elastic_volume_weighted_moving_average",
        "get_volume_adjusted_moving_average",
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
        "get_normalized_average_true_range",
        "get_average_day_range",
        "get_chaikin_volatility",
        "get_relative_volatility_index",
        "get_acceleration_bands",
        "get_chande_kroll_stop",
        "get_adaptive_price_zone",
        "get_aberration",
        "get_profit_maximizer",
        "get_price_distance",
        "get_market_thermometer",
        "get_squeeze_momentum",
    ],
    "breadth": [
        "get_chaikin_money_flow",
        "get_negative_volume_index",
        "get_positive_volume_index",
        "get_relative_volume",
        "get_volume_flow_indicator",
        "get_finite_volume_elements",
        "get_twiggs_money_flow",
        "get_volume_zone_oscillator",
        "get_buying_selling_pressure",
    ],
    "cycle": [
        "get_hilbert_dominant_cycle_period",
        "get_hilbert_phasor_components",
        "get_hilbert_sine_wave",
        "get_hilbert_instantaneous_trendline",
        "get_mesa_adaptive_moving_average",
    ],
}
FACTOR_COLLECTIONS["all"] = [
    metric for metrics in FACTOR_COLLECTIONS.values() for metric in metrics
]

# Indicators in price units, which get_factor_analysis makes comparable across tickers: a
# price line (an average, band or stop) as the distance of the close from it, close / line - 1,
# and a price difference (a range or an oscillator of two averages) as a share of the close.
FACTOR_PRICE_LINES = (
    "get_moving_average",
    "get_kaufman_adaptive_moving_average",
    "get_t3_moving_average",
    "get_arnaud_legoux_moving_average",
    "get_volume_weighted_moving_average",
    "get_mcginley_dynamic",
    "get_variable_index_dynamic_average",
    "get_fractal_adaptive_moving_average",
    "get_holt_winters_moving_average",
    "get_holt_winters_channel",
    "get_elastic_volume_weighted_moving_average",
    "get_volume_adjusted_moving_average",
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
    "get_aberration",
    "get_profit_maximizer",
    "get_pivot_points",
    "get_fibonacci_retracement_levels",
    "get_rolling_median",
    "get_rolling_percentile",
    "get_hilbert_instantaneous_trendline",
    "get_mesa_adaptive_moving_average",
)
FACTOR_PRICE_DIFFERENCES = (
    "get_true_range",
    "get_average_true_range",
    "get_average_day_range",
    "get_price_distance",
    "get_market_thermometer",
    "get_squeeze_momentum",
    "get_moving_average_convergence_divergence",
    "get_absolute_price_oscillator",
    "get_momentum",
    "get_qstick",
    "get_awesome_oscillator",
    "get_accelerator_oscillator",
    "get_elder_ray_index",
    "get_detrended_price_oscillator",
    "get_force_index",
    "get_slope",
    "get_rolling_mean_absolute_deviation",
    "get_rolling_standard_error",
    "get_hilbert_phasor_components",
)


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


def apply_on_own_closes(
    prices_close: pd.DataFrame, function: Callable[[pd.Series], pd.Series]
) -> pd.DataFrame:
    """
    Applies a function to each ticker's closes over the dates the ticker has a close, and
    places the result back on every date, NaN on the dates without a close. A change, moving
    average or rolling high of a ticker thereby spans its own trading days, not the dates
    only another ticker traded, such as the weekends of a cryptocurrency next to a stock.

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
    The number of daily observations in a year of a ticker, to annualize a daily measure or
    to express a year in periods: 252 for a ticker that trades on weekdays only, and 365 for
    one with prices on a Saturday or Sunday, such as a cryptocurrency that trades every
    calendar day. Counting the prices of each year instead would give a stock between 250
    and 253 depending on its holidays, so a ticker on an exchange's trading days keeps the
    252 convention.

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
    Smooths prices with a smoothing factor that changes every period, as the adaptive
    averages do (the Variable Index Dynamic Average and the Fractal Adaptive Moving
    Average): Average(t) = Average(t-1) + Factor(t) * (Price(t) - Average(t-1)).

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
