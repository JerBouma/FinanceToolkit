"""Look-Ahead Bias Tests

A technical indicator must be computable from information available at bar t. These
tests assert that property directly: the value an indicator reports at bar t is
identical whether it was computed over the full series or over the series truncated
at bar t, and that adding later bars never revises an earlier value.

This is regression cover for a real defect. `get_support_resistance_levels` used a
centred `argrelextrema` window and a cluster-merge loop that wrote blended levels back
onto earlier dates, so it published levels before they were knowable and rewrote 53%
of its own history as new bars arrived -- silently unusable for backtesting.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from financetoolkit.technicals import (
    breadth_model as breadth,
    momentum_model as momentum,
    overlap_model as overlap,
    volatility_model as volatility,
)

# pylint: disable=missing-function-docstring

WINDOW = 14

# Bars to truncate at. Spread across the series so warm-up transients are long past.
CUT_POINTS = [400, 500, 600, 700]

TOLERANCE = 1e-9

# Every indicator taking a single security's OHLCV, as (High, Low, Close, Open, Volume).
PER_SECURITY_INDICATORS = {
    "get_money_flow_index": lambda high, low, close, opening, volume: momentum.get_money_flow_index(
        high, low, close, volume, WINDOW
    ),
    "get_williams_percent_r": lambda high, low, close, opening, volume: momentum.get_williams_percent_r(
        high, low, close, WINDOW
    ),
    "get_aroon_indicator": lambda high, low, close, opening, volume: momentum.get_aroon_indicator(
        high, low, WINDOW
    ),
    "get_commodity_channel_index": (
        lambda high, low, close, opening, volume: momentum.get_commodity_channel_index(
            high, low, close, WINDOW
        )
    ),
    "get_relative_vigor_index": lambda high, low, close, opening, volume: momentum.get_relative_vigor_index(
        opening, close, WINDOW
    ),
    "get_force_index": lambda high, low, close, opening, volume: momentum.get_force_index(
        close, volume, WINDOW
    ),
    "get_ultimate_oscillator": lambda high, low, close, opening, volume: momentum.get_ultimate_oscillator(
        high, low, close, 7, 14, 28
    ),
    "get_percentage_price_oscillator": (
        lambda high, low, close, opening, volume: momentum.get_percentage_price_oscillator(
            close, 12, 26
        )
    ),
    "get_detrended_price_oscillator": (
        lambda high, low, close, opening, volume: momentum.get_detrended_price_oscillator(
            close, WINDOW
        )
    ),
    "get_average_directional_index": (
        lambda high, low, close, opening, volume: momentum.get_average_directional_index(
            high, low, close, WINDOW
        )
    ),
    "get_chande_momentum_oscillator": (
        lambda high, low, close, opening, volume: momentum.get_chande_momentum_oscillator(
            close, WINDOW
        )
    ),
    "get_ichimoku_cloud": lambda high, low, close, opening, volume: momentum.get_ichimoku_cloud(
        high, low, 9, 26, 52
    ),
    "get_stochastic_oscillator": (
        lambda high, low, close, opening, volume: momentum.get_stochastic_oscillator(
            high, low, close, WINDOW, 3
        )
    ),
    "get_moving_average_convergence_divergence": (
        lambda high, low, close, opening, volume: momentum.get_moving_average_convergence_divergence(
            close, 12, 26, 9
        )
    ),
    "get_relative_strength_index": (
        lambda high, low, close, opening, volume: momentum.get_relative_strength_index(
            close, WINDOW
        )
    ),
    "get_balance_of_power": lambda high, low, close, opening, volume: momentum.get_balance_of_power(
        opening, high, low, close
    ),
    "get_awesome_oscillator": lambda high, low, close, opening, volume: momentum.get_awesome_oscillator(
        high, low
    ),
    "get_vortex_indicator": lambda high, low, close, opening, volume: momentum.get_vortex_indicator(
        high, low, close, WINDOW
    ),
    "get_elder_ray_index": lambda high, low, close, opening, volume: momentum.get_elder_ray_index(
        high, low, close
    ),
    "get_rate_of_change": lambda high, low, close, opening, volume: momentum.get_rate_of_change(
        close, WINDOW
    ),
    "get_choppiness_index": lambda high, low, close, opening, volume: momentum.get_choppiness_index(
        high, low, close, WINDOW
    ),
    "get_know_sure_thing": lambda high, low, close, opening, volume: momentum.get_know_sure_thing(
        close
    ),
    "get_moving_average": lambda high, low, close, opening, volume: overlap.get_moving_average(
        close, WINDOW
    ),
    "get_exponential_moving_average": (
        lambda high, low, close, opening, volume: overlap.get_exponential_moving_average(
            close, WINDOW
        )
    ),
    "get_double_exponential_moving_average": (
        lambda high, low, close, opening, volume: overlap.get_double_exponential_moving_average(
            close, WINDOW
        )
    ),
    "get_trix": lambda high, low, close, opening, volume: overlap.get_trix(
        close, WINDOW
    ),
    "get_triangular_moving_average": (
        lambda high, low, close, opening, volume: overlap.get_triangular_moving_average(
            close, WINDOW
        )
    ),
    "get_weighted_moving_average": (
        lambda high, low, close, opening, volume: overlap.get_weighted_moving_average(
            close, WINDOW
        )
    ),
    "get_kaufman_adaptive_moving_average": (
        lambda high, low, close, opening, volume: overlap.get_kaufman_adaptive_moving_average(
            close, 10
        )
    ),
    "get_hull_moving_average": lambda high, low, close, opening, volume: overlap.get_hull_moving_average(
        close, WINDOW
    ),
    "get_volume_weighted_average_price": (
        lambda high, low, close, opening, volume: overlap.get_volume_weighted_average_price(
            high, low, close, volume, WINDOW
        )
    ),
    "get_parabolic_sar": lambda high, low, close, opening, volume: overlap.get_parabolic_sar(
        high, low
    ),
    "get_pivot_points": lambda high, low, close, opening, volume: overlap.get_pivot_points(
        high, low, close
    ),
    "get_support_resistance_levels": (
        lambda high, low, close, opening, volume: overlap.get_support_resistance_levels(
            close, 5, 0.05
        )
    ),
    "get_fibonacci_retracement_levels": (
        lambda high, low, close, opening, volume: overlap.get_fibonacci_retracement_levels(
            high.rolling(WINDOW).max(), low.rolling(WINDOW).min()
        )
    ),
    "get_true_range": lambda high, low, close, opening, volume: volatility.get_true_range(
        high, low, close
    ),
    "get_wilder_moving_average": (
        lambda high, low, close, opening, volume: volatility.get_wilder_moving_average(
            close, WINDOW
        )
    ),
    "get_average_true_range": lambda high, low, close, opening, volume: volatility.get_average_true_range(
        high, low, close, WINDOW
    ),
    "get_supertrend": lambda high, low, close, opening, volume: volatility.get_supertrend(
        high, low, close
    ),
    "get_keltner_channels": lambda high, low, close, opening, volume: volatility.get_keltner_channels(
        high, low, close, 20, 10, 2.0
    ),
    "get_donchian_channels": lambda high, low, close, opening, volume: volatility.get_donchian_channels(
        high, low, WINDOW
    ),
    "get_bollinger_bands": lambda high, low, close, opening, volume: volatility.get_bollinger_bands(
        close, 20, 2
    ),
    "get_mcclellan_oscillator": lambda high, low, close, opening, volume: breadth.get_mcclellan_oscillator(
        close, 19, 39
    ),
    "get_advancers_decliners": lambda high, low, close, opening, volume: breadth.get_advancers_decliners(
        close
    ),
    "get_on_balance_volume": lambda high, low, close, opening, volume: breadth.get_on_balance_volume(
        close, volume
    ),
    "get_accumulation_distribution_line": (
        lambda high, low, close, opening, volume: breadth.get_accumulation_distribution_line(
            high, low, close, volume
        )
    ),
    "get_chaikin_oscillator": lambda high, low, close, opening, volume: breadth.get_chaikin_oscillator(
        high, low, close, volume, 3, 10
    ),
    "get_chaikin_money_flow": lambda high, low, close, opening, volume: breadth.get_chaikin_money_flow(
        high, low, close, volume, 20
    ),
    "get_ease_of_movement": lambda high, low, close, opening, volume: breadth.get_ease_of_movement(
        high, low, volume, WINDOW
    ),
    "get_negative_volume_index": (
        lambda high, low, close, opening, volume: breadth.get_negative_volume_index(
            close, volume
        )
    ),
    "get_positive_volume_index": (
        lambda high, low, close, opening, volume: breadth.get_positive_volume_index(
            close, volume
        )
    ),
}

# Breadth indicators measured across a universe rather than one security.
CROSS_SECTIONAL_INDICATORS = {
    "get_trin": breadth.get_trin,
    "get_new_highs_new_lows": lambda close, volume: breadth.get_new_highs_new_lows(
        close, WINDOW
    ),
}


@pytest.fixture(name="historical_dataset", scope="module")
def historical_dataset_fixture():
    return pd.read_pickle(
        Path(__file__).resolve().parents[1] / "datasets" / "historical_dataset.pickle"
    )


@pytest.fixture(name="ohlcv", scope="module")
def ohlcv_fixture(historical_dataset):
    ticker = "AAPL"

    return {
        "high": historical_dataset["High"][ticker].astype(float),
        "low": historical_dataset["Low"][ticker].astype(float),
        "close": historical_dataset["Adj Close"][ticker].astype(float),
        "open": historical_dataset["Open"][ticker].astype(float),
        "volume": historical_dataset["Volume"][ticker].astype(float),
    }


@pytest.fixture(name="universe", scope="module")
def universe_fixture(historical_dataset):
    tickers = ["AAPL", "MSFT"]

    return {
        "close": historical_dataset["Adj Close"][tickers].astype(float),
        "volume": historical_dataset["Volume"][tickers].astype(float),
    }


def _values_at(result: pd.Series | pd.DataFrame, position: int) -> np.ndarray:
    """Return the indicator's value(s) at a positional index as a flat float array."""
    if isinstance(result, pd.DataFrame):
        return result.iloc[position].to_numpy(dtype=float)

    return np.array([float(result.iloc[position])])


def _largest_difference(full: np.ndarray, truncated: np.ndarray) -> float:
    """Compare two value arrays, treating a NaN-vs-number mismatch as a failure."""
    if (np.isnan(full) != np.isnan(truncated)).any():
        return np.inf

    if np.isnan(full).all():
        return 0.0

    with np.errstate(invalid="ignore"):
        difference = np.nanmax(np.abs(full - truncated))

    # An all-NaN comparison yields NaN rather than a real disagreement.
    return 0.0 if np.isnan(difference) else float(difference)


@pytest.mark.parametrize("indicator", sorted(PER_SECURITY_INDICATORS))
def test_per_security_indicator_has_no_look_ahead(indicator, ohlcv):
    calculate = PER_SECURITY_INDICATORS[indicator]
    full = calculate(
        ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["open"], ohlcv["volume"]
    )

    for cut in CUT_POINTS:
        truncated = calculate(
            ohlcv["high"].iloc[: cut + 1],
            ohlcv["low"].iloc[: cut + 1],
            ohlcv["close"].iloc[: cut + 1],
            ohlcv["open"].iloc[: cut + 1],
            ohlcv["volume"].iloc[: cut + 1],
        )

        difference = _largest_difference(
            _values_at(full, cut), _values_at(truncated, -1)
        )

        assert difference <= TOLERANCE, (
            f"{indicator} uses information from after bar {cut}: the value there "
            f"changes by {difference} once later bars are supplied."
        )


@pytest.mark.parametrize("indicator", sorted(CROSS_SECTIONAL_INDICATORS))
def test_cross_sectional_indicator_has_no_look_ahead(indicator, universe):
    calculate = CROSS_SECTIONAL_INDICATORS[indicator]
    full = calculate(universe["close"], universe["volume"])

    for cut in CUT_POINTS:
        truncated = calculate(
            universe["close"].iloc[: cut + 1], universe["volume"].iloc[: cut + 1]
        )

        difference = _largest_difference(
            _values_at(full, cut), _values_at(truncated, -1)
        )

        assert difference <= TOLERANCE, (
            f"{indicator} uses information from after bar {cut}: the value there "
            f"changes by {difference} once later bars are supplied."
        )


@pytest.mark.parametrize("indicator", sorted(PER_SECURITY_INDICATORS))
def test_per_security_indicator_never_revises_history(indicator, ohlcv):
    """Stronger than truncation: no earlier value may change as later bars arrive.

    A two-pass algorithm can publish a correct value at bar t and then rewrite it
    when more data arrives, which the truncation test alone would not catch.
    """
    calculate = PER_SECURITY_INDICATORS[indicator]
    full = calculate(
        ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["open"], ohlcv["volume"]
    )

    cut = CUT_POINTS[-1]
    truncated = calculate(
        ohlcv["high"].iloc[: cut + 1],
        ohlcv["low"].iloc[: cut + 1],
        ohlcv["close"].iloc[: cut + 1],
        ohlcv["open"].iloc[: cut + 1],
        ohlcv["volume"].iloc[: cut + 1],
    )

    overlapping = full.iloc[: cut + 1]
    revised = [
        position
        for position in range(cut + 1)
        if _largest_difference(
            _values_at(overlapping, position), _values_at(truncated, position)
        )
        > TOLERANCE
    ]

    assert not revised, (
        f"{indicator} rewrote {len(revised)} of {cut + 1} historical values once "
        f"later bars were supplied; first revised at bar {revised[0] if revised else None}."
    )
