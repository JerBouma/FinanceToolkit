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

from financetoolkit.econometrics.statistics import (
    entropy_model,
    statistics_model as statistics,
)
from financetoolkit.technicals.breadth import breadth_model as breadth
from financetoolkit.technicals.momentum import momentum_model as momentum
from financetoolkit.technicals.overlap import overlap_model as overlap
from financetoolkit.technicals.pattern import pattern_model as pattern
from financetoolkit.technicals.volatility import volatility_model as volatility

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
        opening, high, low, close, WINDOW
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
    "get_percentage_volume_oscillator": (
        lambda high, low, close, opening, volume: momentum.get_percentage_volume_oscillator(
            volume, 12, 26, 9
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
    "get_stochastic_relative_strength_index": (
        lambda high, low, close, opening, volume: momentum.get_stochastic_relative_strength_index(
            close, WINDOW, WINDOW, 3
        )
    ),
    "get_true_strength_index": lambda high, low, close, opening, volume: momentum.get_true_strength_index(
        close, 25, 13, 7
    ),
    "get_coppock_curve": lambda high, low, close, opening, volume: momentum.get_coppock_curve(
        close, 14, 11, 10
    ),
    "get_directional_movement_index": (
        lambda high, low, close, opening, volume: momentum.get_directional_movement_index(
            high, low, close, WINDOW
        )
    ),
    "get_average_directional_index_rating": (
        lambda high, low, close, opening, volume: momentum.get_average_directional_index_rating(
            high, low, close, WINDOW
        )
    ),
    "get_absolute_price_oscillator": (
        lambda high, low, close, opening, volume: momentum.get_absolute_price_oscillator(
            close, 12, 26
        )
    ),
    "get_momentum": lambda high, low, close, opening, volume: momentum.get_momentum(
        close, 10
    ),
    "get_accelerator_oscillator": (
        lambda high, low, close, opening, volume: momentum.get_accelerator_oscillator(
            high, low
        )
    ),
    "get_internal_bar_strength": (
        lambda high, low, close, opening, volume: momentum.get_internal_bar_strength(
            high, low, close
        )
    ),
    "get_intraday_momentum_index": (
        lambda high, low, close, opening, volume: momentum.get_intraday_momentum_index(
            opening, close, WINDOW
        )
    ),
    "get_qstick": lambda high, low, close, opening, volume: momentum.get_qstick(
        opening, close, 10
    ),
    "get_efficiency_ratio": lambda high, low, close, opening, volume: momentum.get_efficiency_ratio(
        close, 10
    ),
    "get_vertical_horizontal_filter": (
        lambda high, low, close, opening, volume: momentum.get_vertical_horizontal_filter(
            close, 28
        )
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
    "get_triple_exponential_moving_average": (
        lambda high, low, close, opening, volume: overlap.get_triple_exponential_moving_average(
            close, WINDOW
        )
    ),
    "get_t3_moving_average": (
        lambda high, low, close, opening, volume: overlap.get_t3_moving_average(
            close, 5, 0.7
        )
    ),
    "get_zero_lag_exponential_moving_average": (
        lambda high, low, close, opening, volume: overlap.get_zero_lag_exponential_moving_average(
            close, WINDOW
        )
    ),
    "get_arnaud_legoux_moving_average": (
        lambda high, low, close, opening, volume: overlap.get_arnaud_legoux_moving_average(
            close, 9, 6.0, 0.85
        )
    ),
    "get_volume_weighted_moving_average": (
        lambda high, low, close, opening, volume: overlap.get_volume_weighted_moving_average(
            close, volume, WINDOW
        )
    ),
    "get_mcginley_dynamic": (
        lambda high, low, close, opening, volume: overlap.get_mcginley_dynamic(
            close, WINDOW
        )
    ),
    "get_variable_index_dynamic_average": (
        lambda high, low, close, opening, volume: overlap.get_variable_index_dynamic_average(
            close, 12, 9
        )
    ),
    "get_fractal_adaptive_moving_average": (
        lambda high, low, close, opening, volume: overlap.get_fractal_adaptive_moving_average(
            high, low, 16
        )
    ),
    "get_sine_weighted_moving_average": (
        lambda high, low, close, opening, volume: overlap.get_sine_weighted_moving_average(
            close, WINDOW
        )
    ),
    "get_fibonacci_weighted_moving_average": (
        lambda high, low, close, opening, volume: overlap.get_fibonacci_weighted_moving_average(
            close, WINDOW
        )
    ),
    "get_pascal_weighted_moving_average": (
        lambda high, low, close, opening, volume: overlap.get_pascal_weighted_moving_average(
            close, WINDOW
        )
    ),
    "get_volume_weighted_average_price": (
        lambda high, low, close, opening, volume: overlap.get_volume_weighted_average_price(
            high, low, close, volume, WINDOW
        )
    ),
    "get_parabolic_sar": lambda high, low, close, opening, volume: overlap.get_parabolic_sar(
        high, low
    ),
    "get_parabolic_sar_extended": (
        lambda high, low, close, opening, volume: overlap.get_parabolic_sar(
            high, low, 0.01, 0.03, 0.25, 0.015, 0.02, 0.15, 0.0, 0.01
        )
    ),
    "get_price_transform": lambda high, low, close, opening, volume: overlap.get_price_transform(
        opening, high, low, close, "average"
    ),
    "get_midpoint": lambda high, low, close, opening, volume: overlap.get_midpoint(
        high, low, WINDOW
    ),
    "get_holt_winters_moving_average": (
        lambda high, low, close, opening, volume: overlap.get_holt_winters_moving_average(
            close
        )
    ),
    "get_holt_winters_channel": (
        lambda high, low, close, opening, volume: overlap.get_holt_winters_channel(
            close
        )
    ),
    "get_elastic_volume_weighted_moving_average": (
        lambda high, low, close, opening, volume: overlap.get_elastic_volume_weighted_moving_average(
            close, volume, 20
        )
    ),
    "get_volume_adjusted_moving_average": (
        lambda high, low, close, opening, volume: overlap.get_volume_adjusted_moving_average(
            close, volume, 8
        )
    ),
    "get_gann_hilo_activator": (
        lambda high, low, close, opening, volume: overlap.get_gann_hilo_activator(
            high, low, close
        )
    ),
    "get_zigzag": lambda high, low, close, opening, volume: overlap.get_zigzag(
        high, low, 0.03
    ),
    "get_trix_signal": lambda high, low, close, opening, volume: overlap.get_trix(
        close, WINDOW, 9
    ),
    "get_volume_weighted_average_price_anchored": (
        lambda high, low, close, opening, volume: overlap.get_volume_weighted_average_price(
            high, low, close, volume, WINDOW, "monthly"
        )
    ),
    "get_pivot_points": lambda high, low, close, opening, volume: overlap.get_pivot_points(
        high, low, close
    ),
    "get_pivot_points_demark": lambda high, low, close, opening, volume: overlap.get_pivot_points(
        high, low, close, opening, "demark", True
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
    "get_mass_index": lambda high, low, close, opening, volume: volatility.get_mass_index(
        high, low, 9, 25
    ),
    "get_supertrend": lambda high, low, close, opening, volume: volatility.get_supertrend(
        high, low, close
    ),
    "get_chandelier_exit": lambda high, low, close, opening, volume: volatility.get_chandelier_exit(
        high, low, close, 22, 3.0
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
    "get_bollinger_bands_exponential": (
        lambda high, low, close, opening, volume: volatility.get_bollinger_bands(
            close, 20, 2, "exponential"
        )
    ),
    "get_bollinger_percent_b_and_bandwidth": (
        lambda high, low, close, opening, volume: volatility.get_bollinger_percent_b_and_bandwidth(
            close, 20, 2
        )
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
    "get_price_volume_trend": lambda high, low, close, opening, volume: breadth.get_price_volume_trend(
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
    "get_linear_regression": lambda high, low, close, opening, volume: statistics.get_linear_regression(
        close, WINDOW
    ),
    "get_slope": lambda high, low, close, opening, volume: statistics.get_slope(
        close, 5
    ),
    "get_correlation_trend_indicator": (
        lambda high, low, close, opening, volume: momentum.get_correlation_trend_indicator(
            close, 20
        )
    ),
    "get_forecast_oscillator": (
        lambda high, low, close, opening, volume: momentum.get_forecast_oscillator(
            close, 5
        )
    ),
    "get_projection_oscillator": (
        lambda high, low, close, opening, volume: momentum.get_projection_oscillator(
            high, low, close, WINDOW
        )
    ),
    "get_correlation": (
        lambda high, low, close, opening, volume: statistics.get_correlation(
            high, low, 30
        )
    ),
    "get_percentile": (
        lambda high, low, close, opening, volume: statistics.get_percentile(
            close, 100, 0.5
        )
    ),
    "get_percent_rank": lambda high, low, close, opening, volume: statistics.get_percent_rank(
        close, 100
    ),
    "get_z_score": lambda high, low, close, opening, volume: statistics.get_z_score(
        close, 30
    ),
    "get_mean_absolute_deviation": (
        lambda high, low, close, opening, volume: statistics.get_mean_absolute_deviation(
            close, WINDOW
        )
    ),
    "get_standard_error": (
        lambda high, low, close, opening, volume: statistics.get_standard_error(
            close, WINDOW
        )
    ),
    "get_share_entropy": lambda high, low, close, opening, volume: pd.Series(
        np.r_[np.full(9, np.nan), entropy_model.get_share_entropy(close, 10)],
        index=close.index,
    ),
    "get_klinger_volume_oscillator": (
        lambda high, low, close, opening, volume: breadth.get_klinger_volume_oscillator(
            high, low, close, volume, 34, 55, 13
        )
    ),
    "get_market_facilitation_index": (
        lambda high, low, close, opening, volume: breadth.get_market_facilitation_index(
            high, low, volume
        )
    ),
    "get_williams_accumulation_distribution": (
        lambda high, low, close, opening, volume: breadth.get_williams_accumulation_distribution(
            high, low, close
        )
    ),
    "get_relative_volume": lambda high, low, close, opening, volume: breadth.get_relative_volume(
        volume, 20
    ),
    "get_volume_flow_indicator": (
        lambda high, low, close, opening, volume: breadth.get_volume_flow_indicator(
            high, low, close, volume, 130, 0.2, 2.5, 3
        )
    ),
    "get_finite_volume_elements": (
        lambda high, low, close, opening, volume: breadth.get_finite_volume_elements(
            high, low, close, volume, 22, 0.003
        )
    ),
    "get_twiggs_money_flow": lambda high, low, close, opening, volume: breadth.get_twiggs_money_flow(
        high, low, close, volume, 21
    ),
    "get_volume_zone_oscillator": (
        lambda high, low, close, opening, volume: breadth.get_volume_zone_oscillator(
            close, volume, WINDOW
        )
    ),
    "get_cumulative_force_index": (
        lambda high, low, close, opening, volume: breadth.get_cumulative_force_index(
            close, volume, 10
        )
    ),
    "get_price_volume": lambda high, low, close, opening, volume: breadth.get_price_volume(
        close, volume, True
    ),
    "get_price_volume_rank": lambda high, low, close, opening, volume: breadth.get_price_volume_rank(
        close, volume
    ),
    "get_buying_selling_pressure": (
        lambda high, low, close, opening, volume: breadth.get_buying_selling_pressure(
            high, low, close, volume, 40, 20
        )
    ),
    "get_archer_on_balance_volume": (
        lambda high, low, close, opening, volume: breadth.get_archer_on_balance_volume(
            close, volume, 4, 12, 2, 2
        )
    ),
    "get_volume_weighted_macd": (
        lambda high, low, close, opening, volume: momentum.get_moving_average_convergence_divergence(
            close, 12, 26, 9, "volume_weighted", "volume_weighted", volumes=volume
        )
    ),
    "get_elastic_volume_weighted_macd": (
        lambda high, low, close, opening, volume: momentum.get_moving_average_convergence_divergence(
            close, 20, 40, 9, "elastic_volume_weighted", volumes=volume
        )
    ),
    "get_connors_relative_strength_index": (
        lambda high, low, close, opening, volume: momentum.get_connors_relative_strength_index(
            close, 3, 2, 100
        )
    ),
    "get_inertia": lambda high, low, close, opening, volume: momentum.get_inertia(
        high, low, close, 20, 14, 10, "refined"
    ),
    "get_bias": lambda high, low, close, opening, volume: momentum.get_bias(close, 26),
    "get_brar": lambda high, low, close, opening, volume: momentum.get_brar(
        opening, high, low, close, 26
    ),
    "get_pretty_good_oscillator": (
        lambda high, low, close, opening, volume: momentum.get_pretty_good_oscillator(
            high, low, close, WINDOW
        )
    ),
    "get_psychological_line": (
        lambda high, low, close, opening, volume: momentum.get_psychological_line(
            opening, close, 12
        )
    ),
    "get_volatility_based_momentum": (
        lambda high, low, close, opening, volume: momentum.get_volatility_based_momentum(
            high, low, close, 12, 26
        )
    ),
    "get_archer_moving_average_trends": (
        lambda high, low, close, opening, volume: momentum.get_archer_moving_average_trends(
            close, 8, 21, 2
        )
    ),
    "get_ttm_trend": lambda high, low, close, opening, volume: momentum.get_ttm_trend(
        high, low, close, 6
    ),
    "get_candlestick_patterns": (
        lambda high, low, close, opening, volume: pattern.get_candlestick_patterns(
            opening, high, low, close
        )
    ),
    "get_heikin_ashi": lambda high, low, close, opening, volume: pattern.get_heikin_ashi(
        opening, high, low, close
    ),
    "get_williams_fractal": lambda high, low, close, opening, volume: pattern.get_williams_fractal(
        high, low, 2, 2
    ),
    "get_signal_to_noise_ratio": (
        lambda high, low, close, opening, volume: volatility.get_signal_to_noise_ratio(
            high, low, close, WINDOW
        )
    ),
    "get_price_bands": lambda high, low, close, opening, volume: volatility.get_price_bands(
        close, 20, 2, 2
    ),
    "get_trend_detection_index": (
        lambda high, low, close, opening, volume: momentum.get_trend_detection_index(
            close, 20, 2
        )
    ),
    "get_dv_intermediate_oscillator": (
        lambda high, low, close, opening, volume: momentum.get_dv_intermediate_oscillator(
            close, 252, 0.8
        )
    ),
    "get_guppy_multiple_moving_averages": (
        lambda high, low, close, opening, volume: overlap.get_guppy_multiple_moving_averages(
            close
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
