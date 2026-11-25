"""Technical Controller Tests""" ""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.technicals.pattern import pattern_model
from financetoolkit.utilities.error_model import InvalidArgumentError


def test_collect_all_indicators(recorder, technical_module):
    recorder.capture(technical_module.collect_all_indicators().round(0))
    recorder.capture(technical_module.collect_all_indicators(growth=True).round(0))
    recorder.capture(
        technical_module.collect_all_indicators(growth=True, lag=[1, 2, 3]).round(0)
    )


def test_collect_breadth_indicators(recorder, technical_module):
    recorder.capture(technical_module.collect_breadth_indicators().round(0))
    recorder.capture(technical_module.collect_breadth_indicators(growth=True).round(0))
    recorder.capture(
        technical_module.collect_breadth_indicators(growth=True, lag=[1, 2, 3]).round(0)
    )


def test_collect_momentum_indicators(recorder, technical_module):
    recorder.capture(technical_module.collect_momentum_indicators())
    recorder.capture(technical_module.collect_momentum_indicators(growth=True))
    recorder.capture(
        technical_module.collect_momentum_indicators(growth=True, lag=[1, 2, 3])
    )


def test_collect_overlap_indicators(recorder, technical_module):
    recorder.capture(technical_module.collect_overlap_indicators())
    recorder.capture(technical_module.collect_overlap_indicators(growth=True))
    recorder.capture(
        technical_module.collect_overlap_indicators(growth=True, lag=[1, 2, 3])
    )


def test_collect_volatility_indicators(recorder, technical_module):
    recorder.capture(technical_module.collect_volatility_indicators())
    recorder.capture(technical_module.collect_volatility_indicators(growth=True))
    recorder.capture(
        technical_module.collect_volatility_indicators(growth=True, lag=[1, 2, 3])
    )


def test_collect_cycle_indicators(recorder, technical_module):
    recorder.capture(technical_module.collect_cycle_indicators())
    recorder.capture(technical_module.collect_cycle_indicators(growth=True))
    recorder.capture(
        technical_module.collect_cycle_indicators(growth=True, lag=[1, 2, 3])
    )


def test_get_mcclellan_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_mcclellan_oscillator())
    recorder.capture(technical_module.get_mcclellan_oscillator(growth=True))
    recorder.capture(
        technical_module.get_mcclellan_oscillator(growth=True, lag=[1, 2, 3])
    )


def test_get_advancers_decliners(recorder, technical_module):
    recorder.capture(technical_module.get_advancers_decliners())
    recorder.capture(technical_module.get_advancers_decliners(growth=True))
    recorder.capture(
        technical_module.get_advancers_decliners(growth=True, lag=[1, 2, 3])
    )


def test_get_money_flow_index(recorder, technical_module):
    recorder.capture(technical_module.get_money_flow_index())
    recorder.capture(technical_module.get_money_flow_index(growth=True))
    recorder.capture(technical_module.get_money_flow_index(growth=True, lag=[1, 2, 3]))


def test_get_williams_percent_r(recorder, technical_module):
    recorder.capture(technical_module.get_williams_percent_r())
    recorder.capture(technical_module.get_williams_percent_r(growth=True))
    recorder.capture(
        technical_module.get_williams_percent_r(growth=True, lag=[1, 2, 3])
    )


def test_get_aroon_indicator(recorder, technical_module):
    recorder.capture(technical_module.get_aroon_indicator())
    recorder.capture(technical_module.get_aroon_indicator(growth=True))
    recorder.capture(technical_module.get_aroon_indicator(growth=True, lag=[1, 2, 3]))


def test_get_commodity_channel_index(recorder, technical_module):
    recorder.capture(technical_module.get_commodity_channel_index())
    recorder.capture(technical_module.get_commodity_channel_index(growth=True))
    recorder.capture(
        technical_module.get_commodity_channel_index(growth=True, lag=[1, 2, 3])
    )


def test_get_relative_vigor_index(recorder, technical_module):
    recorder.capture(technical_module.get_relative_vigor_index())
    recorder.capture(technical_module.get_relative_vigor_index(growth=True))
    recorder.capture(
        technical_module.get_relative_vigor_index(growth=True, lag=[1, 2, 3])
    )


def test_get_force_index(recorder, technical_module):
    recorder.capture(technical_module.get_force_index())
    recorder.capture(technical_module.get_force_index(growth=True))
    recorder.capture(technical_module.get_force_index(growth=True, lag=[1, 2, 3]))
    recorder.capture(technical_module.get_force_index(method="cumulative").round(0))
    recorder.capture(
        technical_module.get_force_index(method="cumulative", window=1).round(0)
    )
    recorder.capture(
        technical_module.get_force_index(method="cumulative", period="weekly").round(0)
    )


def test_get_ultimate_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_ultimate_oscillator())
    recorder.capture(technical_module.get_ultimate_oscillator(growth=True))
    recorder.capture(
        technical_module.get_ultimate_oscillator(growth=True, lag=[1, 2, 3])
    )


def test_get_moving_average_convergence_divergence_percentage(
    recorder, technical_module
):
    recorder.capture(
        technical_module.get_moving_average_convergence_divergence(method="percentage")
    )
    recorder.capture(
        technical_module.get_moving_average_convergence_divergence(
            method="percentage", growth=True
        )
    )
    recorder.capture(
        technical_module.get_moving_average_convergence_divergence(
            method="percentage", growth=True, lag=[1, 2, 3]
        )
    )
    recorder.capture(
        technical_module.get_moving_average_convergence_divergence(
            method="percentage", signal_window=5, moving_average="simple"
        )
    )


def test_get_moving_average_convergence_divergence_absolute_price_oscillator(
    recorder, technical_module
):
    # The Absolute Price Oscillator is the MACD line.
    def macd_line(**kwargs):
        return technical_module.get_moving_average_convergence_divergence(**kwargs)[
            "MACD Line"
        ].reindex(columns=["AAPL", "MSFT", "Benchmark"], level=0)

    recorder.capture(macd_line())
    recorder.capture(macd_line(moving_average="simple"))
    recorder.capture(macd_line(growth=True, lag=[1, 2, 3]))


def test_get_percentage_volume_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_percentage_volume_oscillator())
    recorder.capture(technical_module.get_percentage_volume_oscillator(growth=True))
    recorder.capture(
        technical_module.get_percentage_volume_oscillator(
            short_window=5, long_window=20, signal_window=5
        )
    )
    recorder.capture(
        technical_module.get_percentage_volume_oscillator(
            short_window=14, long_window=28, moving_average="simple"
        )
    )


def test_get_detrended_price_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_detrended_price_oscillator())
    recorder.capture(technical_module.get_detrended_price_oscillator(growth=True))
    recorder.capture(
        technical_module.get_detrended_price_oscillator(growth=True, lag=[1, 2, 3])
    )


def test_get_average_directional_index(recorder, technical_module):
    recorder.capture(technical_module.get_average_directional_index())
    recorder.capture(technical_module.get_average_directional_index(growth=True))
    recorder.capture(
        technical_module.get_average_directional_index(growth=True, lag=[1, 2, 3])
    )


def test_get_directional_movement_index(recorder, technical_module):
    recorder.capture(technical_module.get_directional_movement_index())
    recorder.capture(technical_module.get_directional_movement_index(window=7))
    recorder.capture(
        technical_module.get_directional_movement_index(growth=True, lag=[1, 2, 3])
    )


def test_get_average_directional_index_rating_method(recorder, technical_module):
    recorder.capture(technical_module.get_average_directional_index(method="rating"))
    recorder.capture(
        technical_module.get_average_directional_index(method="rating", period="weekly")
    )
    recorder.capture(
        technical_module.get_average_directional_index(
            method="rating", growth=True, lag=[1, 2, 3]
        )
    )


def test_get_chande_momentum_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_chande_momentum_oscillator())
    recorder.capture(technical_module.get_chande_momentum_oscillator(growth=True))
    recorder.capture(
        technical_module.get_chande_momentum_oscillator(growth=True, lag=[1, 2, 3])
    )
    recorder.capture(technical_module.get_chande_momentum_oscillator(smoothed=True))


def test_get_relative_strength_index_intraday(recorder, technical_module):
    recorder.capture(technical_module.get_relative_strength_index(method="intraday"))
    recorder.capture(
        technical_module.get_relative_strength_index(method="intraday", window=7)
    )
    recorder.capture(
        technical_module.get_relative_strength_index(
            method="intraday", growth=True, lag=[1, 2, 3]
        )
    )


def test_get_ichimoku_cloud(recorder, technical_module):
    recorder.capture(technical_module.get_ichimoku_cloud())
    recorder.capture(technical_module.get_ichimoku_cloud(growth=True))
    recorder.capture(technical_module.get_ichimoku_cloud(growth=True, lag=[1, 2, 3]))


def test_get_stochastic_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_stochastic_oscillator())
    recorder.capture(technical_module.get_stochastic_oscillator(growth=True))
    recorder.capture(
        technical_module.get_stochastic_oscillator(growth=True, lag=[1, 2, 3])
    )
    recorder.capture(technical_module.get_stochastic_oscillator(slow=True))


def test_get_moving_average_convergence_divergence(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average_convergence_divergence())
    recorder.capture(
        technical_module.get_moving_average_convergence_divergence(growth=True)
    )
    recorder.capture(
        technical_module.get_moving_average_convergence_divergence(
            growth=True, lag=[1, 2, 3]
        )
    )
    recorder.capture(
        technical_module.get_moving_average_convergence_divergence(
            moving_average="simple", signal_moving_average="simple"
        )
    )
    recorder.capture(
        technical_module.get_moving_average_convergence_divergence(fixed=True)
    )


def test_get_relative_strength_index(recorder, technical_module):
    recorder.capture(technical_module.get_relative_strength_index())
    recorder.capture(technical_module.get_relative_strength_index(growth=True))
    recorder.capture(
        technical_module.get_relative_strength_index(growth=True, lag=[1, 2, 3])
    )


def test_get_stochastic_oscillator_rsi(recorder, technical_module):
    recorder.capture(technical_module.get_stochastic_oscillator(method="rsi"))
    recorder.capture(
        technical_module.get_stochastic_oscillator(method="rsi", stochastic_window=5)
    )
    recorder.capture(
        technical_module.get_stochastic_oscillator(
            method="rsi", growth=True, lag=[1, 2, 3]
        )
    )


def test_get_true_strength_index(recorder, technical_module):
    recorder.capture(technical_module.get_true_strength_index())
    recorder.capture(technical_module.get_true_strength_index(period="weekly"))
    recorder.capture(
        technical_module.get_true_strength_index(growth=True, lag=[1, 2, 3])
    )
    recorder.capture(technical_module.get_true_strength_index(method="ergodic"))
    recorder.capture(
        technical_module.get_true_strength_index(
            method="ergodic", long_window=25, short_window=13, signal_window=7
        )
    )
    recorder.capture(
        technical_module.get_true_strength_index(method="ergodic", period="weekly")
    )


def test_get_balance_of_power(recorder, technical_module):
    recorder.capture(technical_module.get_balance_of_power())
    recorder.capture(technical_module.get_balance_of_power(growth=True))
    recorder.capture(technical_module.get_balance_of_power(growth=True, lag=[1, 2, 3]))


def test_get_internal_bar_strength(recorder, technical_module):
    recorder.capture(technical_module.get_internal_bar_strength())
    recorder.capture(technical_module.get_internal_bar_strength(period="weekly"))
    recorder.capture(
        technical_module.get_internal_bar_strength(growth=True, lag=[1, 2, 3])
    )


def test_get_qstick(recorder, technical_module):
    recorder.capture(technical_module.get_qstick())
    recorder.capture(technical_module.get_qstick(window=5))
    recorder.capture(technical_module.get_qstick(growth=True, lag=[1, 2, 3]))


def test_get_awesome_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_awesome_oscillator())
    recorder.capture(technical_module.get_awesome_oscillator(growth=True))
    recorder.capture(
        technical_module.get_awesome_oscillator(growth=True, lag=[1, 2, 3])
    )
    recorder.capture(technical_module.get_awesome_oscillator(method="accelerator"))
    recorder.capture(
        technical_module.get_awesome_oscillator(method="accelerator", signal_window=3)
    )
    recorder.capture(
        technical_module.get_awesome_oscillator(
            method="accelerator", growth=True, lag=[1, 2, 3]
        )
    )


def test_get_vortex_indicator(recorder, technical_module):
    recorder.capture(technical_module.get_vortex_indicator())
    recorder.capture(technical_module.get_vortex_indicator(growth=True))
    recorder.capture(technical_module.get_vortex_indicator(growth=True, lag=[1, 2, 3]))


def test_get_elder_ray_index(recorder, technical_module):
    recorder.capture(technical_module.get_elder_ray_index())
    recorder.capture(technical_module.get_elder_ray_index(growth=True))
    recorder.capture(technical_module.get_elder_ray_index(growth=True, lag=[1, 2, 3]))


def test_get_moving_average(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average())
    recorder.capture(technical_module.get_moving_average(growth=True))
    recorder.capture(technical_module.get_moving_average(growth=True, lag=[1, 2, 3]))


def test_get_moving_average_exponential(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="exponential"))
    recorder.capture(
        technical_module.get_moving_average(method="exponential", growth=True)
    )
    recorder.capture(
        technical_module.get_moving_average(
            method="exponential", growth=True, lag=[1, 2, 3]
        )
    )


def test_get_moving_average_double_exponential(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="double_exponential"))
    recorder.capture(
        technical_module.get_moving_average(method="double_exponential", growth=True)
    )
    recorder.capture(
        technical_module.get_moving_average(
            method="double_exponential", growth=True, lag=[1, 2, 3]
        )
    )


def test_get_trix(recorder, technical_module):
    recorder.capture(technical_module.get_trix())
    recorder.capture(technical_module.get_trix(growth=True))
    recorder.capture(technical_module.get_trix(growth=True, lag=[1, 2, 3]))


def test_get_bollinger_bands(recorder, technical_module):
    recorder.capture(technical_module.get_bollinger_bands())
    recorder.capture(technical_module.get_bollinger_bands(growth=True))
    recorder.capture(technical_module.get_bollinger_bands(growth=True, lag=[1, 2, 3]))
    recorder.capture(technical_module.get_bollinger_bands(method="price_bands"))
    recorder.capture(
        technical_module.get_bollinger_bands(
            method="price_bands", window=10, num_std_dev=1.5, short_window=3
        )
    )
    recorder.capture(
        technical_module.get_bollinger_bands(
            method="price_bands", growth=True, lag=[1, 2]
        )
    )


def test_get_bollinger_percent_b_and_bandwidth(recorder, technical_module):
    recorder.capture(technical_module.get_bollinger_percent_b_and_bandwidth())
    recorder.capture(
        technical_module.get_bollinger_percent_b_and_bandwidth(
            window=10, num_std_dev=1.5
        )
    )
    recorder.capture(
        technical_module.get_bollinger_percent_b_and_bandwidth(
            growth=True, lag=[1, 2, 3]
        )
    )


def test_get_moving_average_triangular(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="triangular"))
    recorder.capture(
        technical_module.get_moving_average(method="triangular", growth=True)
    )
    recorder.capture(
        technical_module.get_moving_average(
            method="triangular", growth=True, lag=[1, 2, 3]
        )
    )


def test_get_support_resistance_levels(recorder, technical_module):
    recorder.capture(technical_module.get_support_resistance_levels())
    recorder.capture(technical_module.get_support_resistance_levels(sensitivity=0.10))


def test_get_fibonacci_retracement_levels(recorder, technical_module):
    recorder.capture(technical_module.get_fibonacci_retracement_levels())
    recorder.capture(technical_module.get_fibonacci_retracement_levels(growth=True))
    recorder.capture(
        technical_module.get_fibonacci_retracement_levels(growth=True, lag=[1, 2, 3])
    )
    recorder.capture(
        technical_module.get_fibonacci_retracement_levels(trend="downtrend")
    )


def test_get_true_range(recorder, technical_module):
    recorder.capture(technical_module.get_true_range())
    recorder.capture(technical_module.get_true_range(growth=True))
    recorder.capture(technical_module.get_true_range(growth=True, lag=[1, 2, 3]))


def test_get_average_true_range(recorder, technical_module):
    recorder.capture(technical_module.get_average_true_range())
    recorder.capture(technical_module.get_average_true_range(growth=True))
    recorder.capture(
        technical_module.get_average_true_range(growth=True, lag=[1, 2, 3])
    )


def test_get_mass_index(recorder, technical_module):
    recorder.capture(technical_module.get_mass_index())
    recorder.capture(technical_module.get_mass_index(growth=True))
    recorder.capture(technical_module.get_mass_index(ema_window=5, sum_window=10))


def test_get_keltner_channels(recorder, technical_module):
    recorder.capture(technical_module.get_keltner_channels())
    recorder.capture(technical_module.get_keltner_channels(growth=True))
    recorder.capture(technical_module.get_keltner_channels(growth=True, lag=[1, 2, 3]))


def test_get_on_balance_volume(recorder, technical_module):
    recorder.capture(technical_module.get_on_balance_volume())
    recorder.capture(technical_module.get_on_balance_volume(growth=True))
    recorder.capture(technical_module.get_on_balance_volume(growth=True, lag=[1, 2, 3]))


def test_get_price_volume_trend(recorder, technical_module):
    recorder.capture(technical_module.get_price_volume_trend())
    recorder.capture(technical_module.get_price_volume_trend(growth=True))
    recorder.capture(technical_module.get_price_volume_trend(period="weekly"))


def test_get_accumulation_distribution_line(recorder, technical_module):
    recorder.capture(technical_module.get_accumulation_distribution_line())
    recorder.capture(technical_module.get_accumulation_distribution_line(growth=True))
    recorder.capture(
        technical_module.get_accumulation_distribution_line(growth=True, lag=[1, 2, 3])
    )


def test_get_chaikin_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_chaikin_oscillator().round(0))
    recorder.capture(technical_module.get_chaikin_oscillator(growth=True).round(0))
    recorder.capture(
        technical_module.get_chaikin_oscillator(growth=True, lag=[1, 2, 3]).round(0)
    )


def test_get_trin(recorder, technical_module):
    recorder.capture(technical_module.get_trin())
    recorder.capture(technical_module.get_trin(growth=True))
    recorder.capture(technical_module.get_trin(growth=True, lag=[1, 2, 3]))


def test_get_new_highs_new_lows(recorder, technical_module):
    recorder.capture(technical_module.get_new_highs_new_lows())
    recorder.capture(technical_module.get_new_highs_new_lows(growth=True))
    recorder.capture(
        technical_module.get_new_highs_new_lows(growth=True, lag=[1, 2, 3])
    )


def test_get_moving_average_weighted(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="weighted"))
    recorder.capture(
        technical_module.get_moving_average(method="weighted", growth=True)
    )
    recorder.capture(
        technical_module.get_moving_average(
            method="weighted", growth=True, lag=[1, 2, 3]
        )
    )


def test_get_moving_average_hull(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="hull"))
    recorder.capture(technical_module.get_moving_average(method="hull", growth=True))
    recorder.capture(
        technical_module.get_moving_average(method="hull", growth=True, lag=[1, 2, 3])
    )


def test_get_volume_weighted_average_price(recorder, technical_module):
    recorder.capture(technical_module.get_volume_weighted_average_price())
    recorder.capture(technical_module.get_volume_weighted_average_price(growth=True))
    recorder.capture(
        technical_module.get_volume_weighted_average_price(growth=True, lag=[1, 2, 3])
    )


def test_get_parabolic_sar(recorder, technical_module):
    recorder.capture(technical_module.get_parabolic_sar())
    recorder.capture(technical_module.get_parabolic_sar(growth=True))
    recorder.capture(technical_module.get_parabolic_sar(growth=True, lag=[1, 2, 3]))


def test_get_pivot_points(recorder, technical_module):
    recorder.capture(technical_module.get_pivot_points())
    recorder.capture(technical_module.get_pivot_points(growth=True))
    recorder.capture(technical_module.get_pivot_points(growth=True, lag=[1, 2, 3]))


def test_get_donchian_channels(recorder, technical_module):
    recorder.capture(technical_module.get_donchian_channels())
    recorder.capture(technical_module.get_donchian_channels(growth=True))
    recorder.capture(technical_module.get_donchian_channels(growth=True, lag=[1, 2, 3]))


def test_get_volatility_cone(recorder, technical_module):
    recorder.capture(technical_module.get_volatility_cone())
    recorder.capture(technical_module.get_volatility_cone(windows=[5, 10, 20]))


def test_get_rate_of_change(recorder, technical_module):
    recorder.capture(technical_module.get_rate_of_change())
    recorder.capture(technical_module.get_rate_of_change(growth=True))
    recorder.capture(technical_module.get_rate_of_change(growth=True, lag=[1, 2, 3]))


def test_get_rate_of_change_absolute(recorder, technical_module):
    recorder.capture(technical_module.get_rate_of_change(method="absolute"))
    recorder.capture(technical_module.get_rate_of_change(method="absolute", window=20))
    recorder.capture(
        technical_module.get_rate_of_change(
            method="absolute", growth=True, lag=[1, 2, 3]
        )
    )


def test_get_choppiness_index(recorder, technical_module):
    recorder.capture(technical_module.get_choppiness_index())
    recorder.capture(technical_module.get_choppiness_index(growth=True))
    recorder.capture(technical_module.get_choppiness_index(growth=True, lag=[1, 2, 3]))
    recorder.capture(technical_module.get_choppiness_index(range_method="true_range"))


def test_get_efficiency_ratio(recorder, technical_module):
    recorder.capture(technical_module.get_efficiency_ratio())
    recorder.capture(technical_module.get_efficiency_ratio(window=20))
    recorder.capture(technical_module.get_efficiency_ratio(growth=True, lag=[1, 2, 3]))


def test_get_vertical_horizontal_filter(recorder, technical_module):
    recorder.capture(technical_module.get_vertical_horizontal_filter())
    recorder.capture(technical_module.get_vertical_horizontal_filter(window=14))
    recorder.capture(
        technical_module.get_vertical_horizontal_filter(growth=True, lag=[1, 2, 3])
    )


def test_get_know_sure_thing(recorder, technical_module):
    recorder.capture(technical_module.get_know_sure_thing())
    recorder.capture(technical_module.get_know_sure_thing(growth=True))
    recorder.capture(technical_module.get_know_sure_thing(growth=True, lag=[1, 2, 3]))
    recorder.capture(
        technical_module.get_know_sure_thing(
            moving_average="exponential", signal_moving_average="exponential"
        )
    )


def test_get_coppock_curve(recorder, technical_module):
    recorder.capture(technical_module.get_coppock_curve())
    recorder.capture(technical_module.get_coppock_curve(period="monthly"))
    recorder.capture(technical_module.get_coppock_curve(growth=True, lag=[1, 2, 3]))


def test_get_fisher_transform(recorder, technical_module):
    recorder.capture(technical_module.get_fisher_transform())
    recorder.capture(technical_module.get_fisher_transform(window=9))
    recorder.capture(technical_module.get_fisher_transform(period="weekly"))


def test_get_inverse_fisher_transform_rsi(recorder, technical_module):
    recorder.capture(technical_module.get_inverse_fisher_transform_rsi())
    recorder.capture(
        technical_module.get_inverse_fisher_transform_rsi(rsi_window=14, wma_window=5)
    )
    recorder.capture(
        technical_module.get_inverse_fisher_transform_rsi(growth=True, lag=[1, 2])
    )


def test_get_kdj_indicator(recorder, technical_module):
    recorder.capture(technical_module.get_kdj_indicator())
    recorder.capture(technical_module.get_kdj_indicator(window=14, k_window=5))
    recorder.capture(technical_module.get_kdj_indicator(period="weekly"))


def test_get_premier_stochastic_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_premier_stochastic_oscillator())
    recorder.capture(
        technical_module.get_premier_stochastic_oscillator(window=14, smooth_window=3)
    )
    recorder.capture(
        technical_module.get_premier_stochastic_oscillator(close_column="Close")
    )


def test_get_stochastic_momentum_index(recorder, technical_module):
    recorder.capture(technical_module.get_stochastic_momentum_index())
    recorder.capture(
        technical_module.get_stochastic_momentum_index(
            window=5, long_window=20, short_window=5, signal_window=5
        )
    )
    recorder.capture(technical_module.get_stochastic_momentum_index(period="weekly"))


def test_get_schaff_trend_cycle(recorder, technical_module):
    recorder.capture(technical_module.get_schaff_trend_cycle())
    recorder.capture(
        technical_module.get_schaff_trend_cycle(
            short_window=12, long_window=26, cycle_window=10, factor=0.25
        )
    )
    recorder.capture(technical_module.get_schaff_trend_cycle(growth=True, lag=[1, 2]))


def test_get_wavetrend_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_wavetrend_oscillator())
    recorder.capture(
        technical_module.get_wavetrend_oscillator(
            channel_window=9, average_window=12, signal_window=3
        )
    )
    recorder.capture(technical_module.get_wavetrend_oscillator(period="weekly"))


def test_get_derivative_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_derivative_oscillator())
    recorder.capture(
        technical_module.get_derivative_oscillator(
            window=9, first_window=3, second_window=2, signal_window=5
        )
    )
    recorder.capture(technical_module.get_derivative_oscillator(period="weekly"))


def test_get_swing_index(recorder, technical_module):
    recorder.capture(technical_module.get_swing_index())
    recorder.capture(technical_module.get_swing_index(limit_move=10.0))
    recorder.capture(technical_module.get_swing_index(close_column="Close"))


def test_get_quantitative_qualitative_estimation(recorder, technical_module):
    recorder.capture(technical_module.get_quantitative_qualitative_estimation())
    recorder.capture(
        technical_module.get_quantitative_qualitative_estimation(
            window=10, smooth_window=3, multiplier=3.0
        )
    )
    recorder.capture(
        technical_module.get_quantitative_qualitative_estimation(period="weekly")
    )


def test_get_relative_strength_index_dynamic(recorder, technical_module):
    recorder.capture(technical_module.get_relative_strength_index(method="dynamic"))
    recorder.capture(
        technical_module.get_relative_strength_index(
            method="dynamic", window=20, minimum_window=3, maximum_window=40
        )
    )
    recorder.capture(
        technical_module.get_relative_strength_index(method="dynamic", period="weekly")
    )


def test_get_price_zone_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_price_zone_oscillator())
    recorder.capture(technical_module.get_price_zone_oscillator(window=28))
    recorder.capture(
        technical_module.get_price_zone_oscillator(growth=True, lag=[1, 2])
    )


def test_get_volume_weighted_macd(recorder, technical_module):
    recorder.capture(
        technical_module.get_moving_average_convergence_divergence(
            moving_average="volume_weighted"
        )
    )
    recorder.capture(
        technical_module.get_moving_average_convergence_divergence(
            moving_average="volume_weighted", signal_moving_average="volume_weighted"
        )
    )
    recorder.capture(
        technical_module.get_moving_average_convergence_divergence(
            short_window=20, long_window=40, moving_average="elastic_volume_weighted"
        )
    )


def test_get_relative_strength_index_connors(recorder, technical_module):
    recorder.capture(technical_module.get_relative_strength_index(method="connors"))
    recorder.capture(
        technical_module.get_relative_strength_index(
            method="connors", window=4, streak_window=3, rank_window=50
        )
    )
    recorder.capture(
        technical_module.get_relative_strength_index(method="connors", period="weekly")
    )


def test_get_inertia(recorder, technical_module):
    recorder.capture(technical_module.get_inertia())
    recorder.capture(technical_module.get_inertia(method="refined"))
    recorder.capture(technical_module.get_inertia(window=10, std_window=14))


def test_get_bias(recorder, technical_module):
    recorder.capture(technical_module.get_bias())
    recorder.capture(technical_module.get_bias(window=10, moving_average="exponential"))
    recorder.capture(technical_module.get_bias(growth=True))


def test_get_brar(recorder, technical_module):
    recorder.capture(technical_module.get_brar())
    recorder.capture(technical_module.get_brar(window=10))
    recorder.capture(technical_module.get_brar(period="weekly"))


def test_get_pretty_good_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_pretty_good_oscillator())
    recorder.capture(technical_module.get_pretty_good_oscillator(window=21))
    recorder.capture(technical_module.get_pretty_good_oscillator(growth=True))


def test_get_psychological_line(recorder, technical_module):
    recorder.capture(technical_module.get_psychological_line())
    recorder.capture(technical_module.get_psychological_line(method="open"))
    recorder.capture(technical_module.get_psychological_line(window=24))


def test_get_volatility_based_momentum(recorder, technical_module):
    recorder.capture(technical_module.get_volatility_based_momentum())
    recorder.capture(
        technical_module.get_volatility_based_momentum(window=20, atr_window=14)
    )
    recorder.capture(technical_module.get_volatility_based_momentum(period="weekly"))


def test_get_archer_moving_average_trends(recorder, technical_module):
    recorder.capture(technical_module.get_archer_moving_average_trends())
    recorder.capture(
        technical_module.get_archer_moving_average_trends(
            short_window=5, long_window=15, lookback=3, moving_average="simple"
        )
    )


def test_get_ttm_trend(recorder, technical_module):
    recorder.capture(technical_module.get_ttm_trend())
    recorder.capture(technical_module.get_ttm_trend(window=10))
    recorder.capture(technical_module.get_ttm_trend(period="weekly"))


def test_get_moving_average_kaufman_adaptive(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="kaufman_adaptive"))
    recorder.capture(
        technical_module.get_moving_average(method="kaufman_adaptive", growth=True)
    )
    recorder.capture(
        technical_module.get_moving_average(
            method="kaufman_adaptive", growth=True, lag=[1, 2, 3]
        )
    )


def test_get_chaikin_money_flow(recorder, technical_module):
    recorder.capture(technical_module.get_chaikin_money_flow())
    recorder.capture(technical_module.get_chaikin_money_flow(growth=True))
    recorder.capture(
        technical_module.get_chaikin_money_flow(growth=True, lag=[1, 2, 3])
    )


def test_get_ease_of_movement(recorder, technical_module):
    recorder.capture(technical_module.get_ease_of_movement())
    recorder.capture(technical_module.get_ease_of_movement(growth=True))
    recorder.capture(technical_module.get_ease_of_movement(growth=True, lag=[1, 2, 3]))


def test_get_klinger_volume_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_klinger_volume_oscillator().round(0))
    recorder.capture(
        technical_module.get_klinger_volume_oscillator(
            short_window=5, long_window=10, signal_window=4
        ).round(0)
    )
    recorder.capture(technical_module.get_klinger_volume_oscillator(growth=True))


def test_get_market_facilitation_index(recorder, technical_module):
    recorder.capture(technical_module.get_market_facilitation_index())
    recorder.capture(
        technical_module.get_market_facilitation_index(volume_divisor=1_000_000)
    )
    recorder.capture(technical_module.get_market_facilitation_index(growth=True))


def test_get_williams_accumulation_distribution(recorder, technical_module):
    recorder.capture(technical_module.get_williams_accumulation_distribution())
    recorder.capture(
        technical_module.get_williams_accumulation_distribution(period="weekly")
    )
    recorder.capture(
        technical_module.get_williams_accumulation_distribution(growth=True, lag=[1, 2])
    )


def test_get_relative_volume(recorder, technical_module):
    recorder.capture(technical_module.get_relative_volume())
    recorder.capture(technical_module.get_relative_volume(window=10))
    recorder.capture(technical_module.get_relative_volume(period="weekly"))


def test_get_volume_flow_indicator(recorder, technical_module):
    recorder.capture(technical_module.get_volume_flow_indicator())
    recorder.capture(
        technical_module.get_volume_flow_indicator(window=50, smooth_window=5)
    )
    recorder.capture(technical_module.get_volume_flow_indicator(growth=True))


def test_get_finite_volume_elements(recorder, technical_module):
    recorder.capture(technical_module.get_finite_volume_elements())
    recorder.capture(
        technical_module.get_finite_volume_elements(window=10, cutoff=0.01)
    )
    recorder.capture(technical_module.get_finite_volume_elements(period="weekly"))


def test_get_chaikin_money_flow_twiggs(recorder, technical_module):
    recorder.capture(technical_module.get_chaikin_money_flow(method="twiggs"))
    recorder.capture(
        technical_module.get_chaikin_money_flow(method="twiggs", window=10)
    )
    recorder.capture(
        technical_module.get_chaikin_money_flow(method="twiggs", growth=True)
    )


def test_get_volume_zone_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_volume_zone_oscillator())
    recorder.capture(technical_module.get_volume_zone_oscillator(window=28))
    recorder.capture(technical_module.get_volume_zone_oscillator(growth=True))


def test_get_price_volume(recorder, technical_module):
    recorder.capture(technical_module.get_price_volume().round(0))
    recorder.capture(technical_module.get_price_volume(signed=True).round(0))
    recorder.capture(technical_module.get_price_volume(growth=True))


def test_get_price_volume_rank(recorder, technical_module):
    recorder.capture(technical_module.get_price_volume_rank())
    recorder.capture(technical_module.get_price_volume_rank(period="weekly"))


def test_get_buying_selling_pressure(recorder, technical_module):
    recorder.capture(technical_module.get_buying_selling_pressure())
    recorder.capture(technical_module.get_buying_selling_pressure(smooth_window=1))
    recorder.capture(
        technical_module.get_buying_selling_pressure(window=20, smooth_window=10)
    )


def test_get_archer_on_balance_volume(recorder, technical_module):
    recorder.capture(technical_module.get_archer_on_balance_volume().round(0))
    recorder.capture(
        technical_module.get_archer_on_balance_volume(
            short_window=5, long_window=20, run_window=3
        ).round(0)
    )


def test_get_volume_profile(recorder, technical_module):
    recorder.capture(technical_module.get_volume_profile())
    recorder.capture(technical_module.get_volume_profile(bins=5))
    recorder.capture(technical_module.get_volume_profile(period="weekly", bins=20))


def test_get_volume_index(recorder, technical_module):
    recorder.capture(technical_module.get_volume_index())
    recorder.capture(technical_module.get_volume_index(growth=True))
    recorder.capture(technical_module.get_volume_index(growth=True, lag=[1, 2, 3]))


def test_get_volume_index_positive(recorder, technical_module):
    recorder.capture(technical_module.get_volume_index(method="positive"))
    recorder.capture(technical_module.get_volume_index(method="positive", growth=True))
    recorder.capture(
        technical_module.get_volume_index(method="positive", growth=True, lag=[1, 2, 3])
    )


def test_get_supertrend(recorder, technical_module):
    recorder.capture(technical_module.get_supertrend())
    recorder.capture(technical_module.get_supertrend(growth=True))
    recorder.capture(technical_module.get_supertrend(growth=True, lag=[1, 2, 3]))


def test_get_chandelier_exit(recorder, technical_module):
    recorder.capture(technical_module.get_chandelier_exit())
    recorder.capture(technical_module.get_chandelier_exit(growth=True))
    recorder.capture(technical_module.get_chandelier_exit(window=10, multiplier=2.0))


def test_period_ohlc_is_aggregated_from_the_adjusted_daily_prices(technical_module):
    import pandas as pd
    import pytest

    weekly = technical_module._get_historical_data("weekly", "Adj Close")
    daily = technical_module._get_historical_data("daily", "Adj Close")
    raw = technical_module._historical_data["weekly"]

    # Apple went ex-dividend on Friday 7 February 2020: the Monday open of that week is
    # scaled by the factor before the dividend, not by the one at the end of the week.
    week = pd.Period("2020-02-07", freq="W")
    days = daily.loc["2020-02-03":"2020-02-07"]
    assert weekly.loc[week, ("Open", "AAPL")] == pytest.approx(
        days[("Open", "AAPL")].iloc[0]
    )
    assert weekly.loc[week, ("High", "AAPL")] == pytest.approx(
        days[("High", "AAPL")].max()
    )
    assert weekly.loc[week, ("Low", "AAPL")] == pytest.approx(
        days[("Low", "AAPL")].min()
    )
    end_of_week_factor = (
        raw.loc[week, ("Adj Close", "AAPL")] / raw.loc[week, ("Close", "AAPL")]
    )
    assert (
        weekly.loc[week, ("Open", "AAPL")]
        < raw.loc[week, ("Open", "AAPL")] * end_of_week_factor
    )

    # The close and every other column stay as they are.
    pd.testing.assert_frame_equal(weekly["Adj Close"], raw["Adj Close"])
    pd.testing.assert_frame_equal(weekly["Volume"], raw["Volume"])


def test_conventional_default_windows():
    import inspect  # noqa: PLC0415

    from financetoolkit.technicals.technicals_controller import (  # noqa: PLC0415
        Technicals,
    )

    def defaults(method):
        return {
            name: parameter.default
            for name, parameter in inspect.signature(method).parameters.items()
        }

    macd = defaults(Technicals.get_moving_average_convergence_divergence)
    assert (macd["short_window"], macd["long_window"]) == (12, 26)
    # The window depends on the method: 13 for the Force Index and 10 for its running total.
    assert defaults(Technicals.get_force_index)["window"] is None
    # The windows depend on the method: 20 and 10 for the Keltner Channels.
    keltner = defaults(Technicals.get_keltner_channels)
    assert (keltner["window"], keltner["atr_window"], keltner["multiplier"]) == (
        None,
        None,
        2,
    )
    # Bollinger's own setting: a 20-period average with bands at 2 standard deviations.
    bollinger = defaults(Technicals.get_bollinger_bands)
    assert (bollinger["window"], bollinger["num_std_dev"]) == (20, 2)


def test_relative_vigor_index_returns_index_and_signal_line(technical_module):
    result = technical_module.get_relative_vigor_index()

    assert {"Relative Vigor Index", "Signal Line"} <= set(
        result.columns.get_level_values(0)
    )


def test_get_trix_defaults_to_a_window_of_15(technical_module):
    import inspect

    import pandas as pd

    assert (
        inspect.signature(technical_module.get_trix).parameters["window"].default == 15
    )
    pd.testing.assert_frame_equal(
        technical_module.get_trix(), technical_module.get_trix(window=15)
    )


def test_bands_and_channels_take_a_fractional_width(technical_module):
    """Bollinger Bands of 2.5 standard deviations and Keltner Channels of 1.5 ATRs."""
    import pandas as pd  # noqa: PLC0415

    for method, parameter, width, lower, middle in (
        (
            technical_module.get_bollinger_bands,
            "num_std_dev",
            2.5,
            "Lower Band",
            "Middle Band",
        ),
        (
            technical_module.get_keltner_channels,
            "multiplier",
            1.5,
            "Lower Line",
            "Middle Line",
        ),
    ):
        standard = method(rounding=10)
        fractional = method(rounding=10, **{parameter: width})
        pd.testing.assert_frame_equal(
            fractional[middle] - fractional[lower],
            (standard[middle] - standard[lower]) * width / 2,
            atol=1e-6,
        )


def test_on_balance_volume_skips_gaps_per_ticker():
    import pandas as pd

    from financetoolkit.technicals.technicals_controller import Technicals

    index = pd.period_range("2024-01-01", periods=6, freq="D")
    columns = pd.MultiIndex.from_product([["Close", "Adj Close", "Volume"], ["A", "B"]])
    nan = float("nan")
    data = pd.DataFrame(
        [
            [10.0, 20, 10.0, 20, 100.0, 300],
            [11, 21, 11, 21, 200, 100],
            [nan, 20, nan, 20, nan, 200],
            [12, 22, 12, 22, 150, 400],
            [11, 23, 11, 23, 50, 250],
            [13, 22, 13, 22, 80, 150],
        ],
        index=index,
        columns=columns,
    )
    technicals = Technicals(["A", "B"], {"daily": data}, rounding=None)

    obv = technicals.get_on_balance_volume()

    # The day after A's gap compares 12 with 11 instead of being reset.
    assert obv.loc[index[3], "A"] == 200 + 150
    assert obv.loc[index[2], "A"] != obv.loc[index[2], "A"]
    assert obv.loc[index[3], "B"] == 100 - 200 + 400


def test_candlestick_patterns_per_pattern_and_ticker(technical_module):
    patterns = technical_module.get_candlestick_patterns()

    assert list(dict.fromkeys(patterns.columns.get_level_values(0))) == list(
        pattern_model.PATTERN_LOOKBACK
    )
    values = patterns.to_numpy().ravel()
    # 2 and -2 only mark a confirmed hikkake.
    assert set(values[~pd.isna(values)]) <= {-2.0, -1.0, -0.8, 0.0, 0.8, 1.0, 2.0}
    hikkakes = patterns[["Hikkake", "Modified Hikkake"]].to_numpy().ravel()
    others = (
        patterns.drop(columns=["Hikkake", "Modified Hikkake"], level=0)
        .to_numpy()
        .ravel()
    )
    assert {-2.0, 2.0} <= set(hikkakes)
    assert set(others[~pd.isna(others)]) <= {-1.0, -0.8, 0.0, 0.8, 1.0}
    chosen = technical_module.get_candlestick_patterns(patterns="Hammer, Doji")
    assert list(dict.fromkeys(chosen.columns.get_level_values(0))) == [
        "Hammer",
        "Doji",
    ]


def test_signals_match_the_indicators_they_come_from(technical_module):
    signals = technical_module.get_signals(short_window=5, long_window=20)
    short = technical_module.get_moving_average(window=5)
    long = technical_module.get_moving_average(window=20)

    ticker = short.columns[0]
    expected = pattern_model.get_crossover(short[ticker], long[ticker])
    observed = signals["Moving Average Crossover"][ticker]
    pd.testing.assert_series_equal(
        observed.dropna(), expected.loc[observed.dropna().index], check_names=False
    )
    assert observed.abs().sum() > 0


def test_indicator_summary_matches_the_indicator(technical_module):
    rsi = technical_module.get_relative_strength_index(rounding=15)
    summary = technical_module.get_indicator_summary(
        below=30, forward_periods=[5, 21], rounding=15
    )

    for ticker in summary.index:
        values = rsi[ticker].dropna()
        assert summary.loc[ticker, "Latest"] == pytest.approx(values.iloc[-1])
        assert summary.loc[ticker, "Minimum"] == pytest.approx(values.min())
        assert summary.loc[ticker, "Observations"] == len(values)
        assert summary.loc[ticker, "Percentile"] == pytest.approx(
            (values <= values.iloc[-1]).mean()
        )
        assert summary.loc[ticker, "Periods Below"] == (values < 30).sum()
        assert summary.loc[ticker, "Average 21-Period Return"] > -1


def test_indicator_summary_crossings_are_the_rsi_signals(technical_module):
    crossings = technical_module.get_indicator_summary(
        below=30, forward_periods=5, show_crossings=True
    )
    signals = technical_module.get_signals()["RSI Oversold"]

    for ticker in crossings.index.get_level_values("Ticker").unique():
        entered = signals.index[signals[ticker].eq(-1)].to_timestamp()
        assert list(crossings.loc[ticker].index) == list(entered)
        assert crossings.loc[ticker, "5-Period Return"].notna().any()


def test_indicator_summary_needs_a_single_indicator_and_a_line(
    technical_module, monkeypatch
):
    monkeypatch.setenv("FINANCETOOLKIT_STRICT_ERRORS", "1")
    with pytest.raises(ValueError, match="single indicator"):
        technical_module.get_indicator_summary("momentum")
    with pytest.raises(ValueError, match="below or above"):
        technical_module.get_indicator_summary(forward_periods=[5])


def test_signals_only_keeps_the_periods_with_a_signal(technical_module):
    for method, kwargs in (
        (technical_module.get_signals, {"short_window": 5, "long_window": 20}),
        (technical_module.get_candlestick_patterns, {"patterns": "Engulfing"}),
    ):
        everything = method(**kwargs)
        active = method(signals_only=True, **kwargs)

        # Exactly the periods with a non-zero value, and none of the all-zero or empty ones.
        expected = everything[everything.abs().gt(0).any(axis=1)]
        pd.testing.assert_frame_equal(active, expected)
        assert 0 < len(active) < len(everything)
        assert active.abs().gt(0).any(axis=1).all()


def test_get_moving_average_rejects_an_unknown_method(technical_module):
    with pytest.raises(InvalidArgumentError, match="method must be one of"):
        technical_module.get_moving_average(method="cubic")


def _stock_with_a_cryptocurrency():
    """The test toolkit's AAPL and Benchmark next to a ticker with a price on every
    calendar day, filled as the Toolkit fills the dates a ticker did not trade."""
    import numpy as np  # noqa: PLC0415

    from financetoolkit.historical_model import (  # noqa: PLC0415
        fill_non_trading_days,
    )

    historical = pd.read_pickle("tests/datasets/historical_dataset.pickle")
    stock = historical.loc[
        :, historical.columns.get_level_values(1).isin(["AAPL", "Benchmark"])
    ]
    days = pd.period_range(stock.index[0], stock.index[-1], freq="D")
    close = pd.Series(
        100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.03, len(days)))),
        index=days,
    )
    crypto = pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.02,
            "Low": close * 0.97,
            "Close": close,
            "Adj Close": close,
            "Volume": 1e6,
            "Dividends": 0.0,
            "Return": close.pct_change(),
            "Cumulative Return": close / close.iloc[0],
        }
    )
    crypto.columns = pd.MultiIndex.from_product([crypto.columns, ["BTC"]])
    columns = stock.columns.get_level_values(0).unique()
    mixed = pd.concat([stock.reindex(days), crypto], axis=1).reindex(
        columns=pd.MultiIndex.from_product([columns, ["AAPL", "BTC", "Benchmark"]])
    )

    return stock, close, fill_non_trading_days(mixed, "Adj Close")


def test_a_stock_keeps_its_indicators_next_to_a_cryptocurrency():
    """A stock in a Toolkit with a cryptocurrency gets flat weekend bars without a Return.
    Its indicators are calculated over its own trading days, so they equal those of the
    stock alone, and are NaN on the weekends it did not trade."""
    from financetoolkit.technicals.technicals_controller import (  # noqa: PLC0415
        Technicals,
    )

    stock, _, mixed = _stock_with_a_cryptocurrency()
    alone = Technicals(["AAPL"], {"daily": stock}, rounding=None)
    together = Technicals(["AAPL", "BTC"], {"daily": mixed}, rounding=None)
    saturday = pd.Period("2022-12-24", freq="D")

    for entry in (
        "get_average_true_range",
        "get_average_directional_index",
        "get_bollinger_bands",
        "get_money_flow_index",
        "get_ease_of_movement",
        "get_relative_strength_index",
        ("get_stochastic_oscillator", {"method": "rsi"}),
        "get_true_strength_index",
        "get_coppock_curve",
        "get_bollinger_percent_b_and_bandwidth",
        "get_balance_of_power",
        "get_parabolic_sar",
        "get_supertrend",
        "get_moving_average_convergence_divergence",
        "get_on_balance_volume",
        "get_mass_index",
        "get_price_volume_trend",
        "get_chandelier_exit",
        "get_percentage_volume_oscillator",
        "get_klinger_volume_oscillator",
        ("get_chaikin_money_flow", {"method": "twiggs"}),
        "get_relative_volume",
        "get_volume_flow_indicator",
        "get_candlestick_patterns",
        "get_signals",
    ):
        method, kwargs = (entry, {}) if isinstance(entry, str) else entry
        expected = getattr(alone, method)(**kwargs)
        result = getattr(together, method)(**kwargs)
        if isinstance(expected.columns, pd.MultiIndex):
            expected = expected.xs("AAPL", level=1, axis=1)
            result = result.xs("AAPL", level=1, axis=1)
        else:
            expected, result = expected[["AAPL"]], result[["AAPL"]]

        pd.testing.assert_frame_equal(
            result.reindex(expected.index), expected, check_freq=False
        )
        assert result.loc[saturday].isna().all(), method

    # The cryptocurrency trades every day, so it has values on the weekend.
    assert together.get_average_true_range().loc[saturday, "BTC"] > 0

    cone = together.get_volatility_cone()
    pd.testing.assert_frame_equal(
        cone.xs("AAPL", level=1, axis=1),
        alone.get_volatility_cone().xs("AAPL", level=1, axis=1),
    )


def test_a_cryptocurrency_counts_a_year_in_calendar_days():
    """A ticker with prices on weekends annualizes with 365 days and measures a quarter of
    its relative strength score in 91 days; a stock keeps 252 and 63."""
    import numpy as np  # noqa: PLC0415

    from financetoolkit.technicals import helpers  # noqa: PLC0415
    from financetoolkit.technicals.seasonality import (  # noqa: PLC0415
        seasonality_model,
    )
    from financetoolkit.technicals.technicals_controller import (  # noqa: PLC0415
        Technicals,
    )

    stock, close, mixed = _stock_with_a_cryptocurrency()
    assert helpers.get_observations_per_year(close) == 365
    assert helpers.get_observations_per_year(stock[("Close", "AAPL")]) == 252

    together = Technicals(["AAPL", "BTC"], {"daily": mixed}, rounding=None)
    log_returns = np.log(close / close.shift(1))
    cone = together.get_volatility_cone().xs("BTC", level=1, axis=1)
    assert cone.loc[20, "Current"] == pytest.approx(
        log_returns.iloc[-20:].std() * np.sqrt(365)
    )

    score = seasonality_model.get_relative_strength_score(close, 365)
    assert score.iloc[-1] == pytest.approx(
        sum(
            weight * (close.iloc[-1] / close.iloc[-1 - days] - 1)
            for days, weight in zip((91, 182, 274, 365), (0.4, 0.2, 0.2, 0.2))
        )
    )


def test_a_stocks_weeks_leave_out_the_weekends_of_a_cryptocurrency():
    """The weekly Open, High and Low of a stock next to a cryptocurrency come from its own
    trading days, not from the flat bars of the dates only the cryptocurrency traded, so its
    weekly bars and the indicators on them equal those of the stock alone. In the week of
    20 June 2022 (Juneteenth on Monday) the week opens on Tuesday."""
    from financetoolkit.historical_model import (  # noqa: PLC0415
        convert_daily_to_other_period,
    )
    from financetoolkit.technicals.technicals_controller import (  # noqa: PLC0415
        Technicals,
    )

    stock, _, mixed = _stock_with_a_cryptocurrency()

    def technicals(tickers, daily):
        return Technicals(
            tickers,
            {
                "daily": daily,
                "weekly": convert_daily_to_other_period("weekly", daily),
            },
            rounding=None,
        )

    alone = technicals(["AAPL"], stock)
    together = technicals(["AAPL", "BTC"], mixed)

    week = pd.Period("2022-06-20", freq="W")
    tuesday = pd.Period("2022-06-21", freq="D")
    for close_column in ("Close", "Adj Close"):
        weekly = together._get_historical_data("weekly", close_column)
        daily = together._get_historical_data("daily", close_column)
        assert weekly.loc[week, ("Open", "AAPL")] == pytest.approx(
            daily.loc[tuesday, ("Open", "AAPL")]
        )
        expected = alone._get_historical_data("weekly", close_column)
        for column in ("Open", "High", "Low", "Close"):
            pd.testing.assert_series_equal(
                weekly[(column, "AAPL")].reindex(expected.index),
                expected[(column, "AAPL")],
                check_freq=False,
            )

    for method in (
        "get_stochastic_oscillator",
        "get_average_directional_index",
        "get_williams_percent_r",
    ):
        expected = getattr(alone, method)(period="weekly")
        result = getattr(together, method)(period="weekly")
        if isinstance(expected.columns, pd.MultiIndex):
            expected = expected.xs("AAPL", level=1, axis=1)
            result = result.xs("AAPL", level=1, axis=1)
        else:
            expected, result = expected[["AAPL"]], result[["AAPL"]]
        pd.testing.assert_frame_equal(
            result.reindex(expected.index), expected, check_freq=False
        )


def test_market_breadth_counts_a_stock_on_its_own_trading_days():
    """In a Toolkit with a stock and a cryptocurrency, the stock is not counted as unchanged
    on the weekends: only the cryptocurrency counts there, the McClellan averages skip those
    dates, and on the stock's trading days both count."""
    from financetoolkit.technicals.technicals_controller import (  # noqa: PLC0415
        Technicals,
    )

    stock, _, mixed = _stock_with_a_cryptocurrency()
    together = Technicals(["AAPL", "BTC"], {"daily": mixed}, rounding=None)
    saturday = pd.Period("2022-12-24", freq="D")
    friday = pd.Period("2022-12-23", freq="D")

    breadth = together.get_market_breadth()
    counted = breadth[["Advancers", "Decliners", "Unchanged"]].sum(axis=1)
    weekend = breadth.index.dayofweek >= 5
    assert (counted[weekend] == 1).all()
    assert (counted.loc[stock.index[1:]] == 2).all()
    assert breadth["McClellan Oscillator"][weekend].isna().all()
    assert pd.notna(breadth["McClellan Oscillator"].loc[friday])

    # The TRIN and New Highs - New Lows count the stock on its trading days only as well.
    trin = together.get_trin()
    assert trin.loc[saturday].isna().all()
    new_highs_new_lows = together.get_new_highs_new_lows(window=20)
    assert new_highs_new_lows.loc[saturday, "AAPL"] in (-1, 0, 1)


def test_market_breadth_of_stocks_ignores_filled_dates():
    """Stocks whose data holds filled dates, as next to a cryptocurrency, have the same
    breadth, TRIN and New Highs - New Lows as on their own calendar."""
    from financetoolkit.technicals.technicals_controller import (  # noqa: PLC0415
        Technicals,
    )

    stock, _, mixed = _stock_with_a_cryptocurrency()
    filled = mixed.loc[:, mixed.columns.get_level_values(1) != "BTC"]
    alone = Technicals(["AAPL", "Benchmark"], {"daily": stock}, rounding=None)
    with_filled_dates = Technicals(
        ["AAPL", "Benchmark"], {"daily": filled}, rounding=None
    )

    for method, kwargs in (
        ("get_market_breadth", {}),
        ("get_trin", {}),
        ("get_new_highs_new_lows", {"window": 20}),
    ):
        expected = getattr(alone, method)(**kwargs)
        result = getattr(with_filled_dates, method)(**kwargs)
        pd.testing.assert_frame_equal(
            result.reindex(expected.index), expected, check_freq=False
        )


def test_collected_bollinger_bands_keep_their_own_window(technical_module):
    """The collections pass their window of 14 to the indicators with a single window, but
    the Bollinger Bands keep Bollinger's 20 periods."""
    bands = technical_module.get_bollinger_bands()
    volatility = technical_module.collect_volatility_indicators()

    for line in ("Upper", "Middle", "Lower"):
        pd.testing.assert_frame_equal(
            volatility[f"Bollinger Band {line}"],
            bands[f"{line} Band"],
            check_names=False,
        )

    collected = technical_module.collect_all_indicators(rounding=4)
    pd.testing.assert_frame_equal(
        collected["Bollinger Band Upper"],
        bands["Upper Band"].round(4).loc[collected.index],
        check_names=False,
    )


def test_get_correlation_trend_indicator(recorder, technical_module):
    recorder.capture(technical_module.get_correlation_trend_indicator())
    recorder.capture(technical_module.get_correlation_trend_indicator(window=12))
    recorder.capture(
        technical_module.get_correlation_trend_indicator(growth=True, lag=[1, 2])
    )


def test_get_forecast_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_forecast_oscillator())
    recorder.capture(technical_module.get_forecast_oscillator(window=14))
    recorder.capture(technical_module.get_forecast_oscillator(growth=True, lag=[1, 2]))


def test_get_projection_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_projection_oscillator())
    recorder.capture(technical_module.get_projection_oscillator(period="weekly"))
    recorder.capture(
        technical_module.get_projection_oscillator(growth=True, lag=[1, 2])
    )


def test_get_standard_deviation_channel(recorder, technical_module):
    recorder.capture(technical_module.get_standard_deviation_channel())
    recorder.capture(
        technical_module.get_standard_deviation_channel(window=100, num_std_dev=[1, 2])
    )
    recorder.capture(
        technical_module.get_standard_deviation_channel(growth=True, lag=[1, 2])
    )


def test_standard_deviation_channel_is_fitted_over_the_date_range(technical_module):
    channel = technical_module.get_standard_deviation_channel(num_std_dev=2)

    # One straight line per ticker: the same change from each date to the next.
    line = channel["Linear Regression"]["AAPL"].dropna()
    assert np.allclose(line.diff().dropna(), line.diff().iloc[-1], atol=2e-4)
    assert (channel["Upper Band 2"] > channel["Lower Band 2"]).all().all()


def test_regression_indicator_default_windows():
    import inspect  # noqa: PLC0415

    from financetoolkit.technicals.technicals_controller import (  # noqa: PLC0415
        Technicals,
    )

    # The TA-Lib defaults, and pandas-ta-classic's where TA-Lib has no such function.
    for method, window in (
        (Technicals.get_correlation_trend_indicator, 20),
        (Technicals.get_forecast_oscillator, 5),
        (Technicals.get_projection_oscillator, 14),
    ):
        assert inspect.signature(method).parameters["window"].default == window


def test_get_signal_to_noise_ratio(recorder, technical_module):
    recorder.capture(technical_module.get_signal_to_noise_ratio())
    recorder.capture(technical_module.get_signal_to_noise_ratio(window=50))
    recorder.capture(
        technical_module.get_signal_to_noise_ratio(growth=True, lag=[1, 2])
    )


def test_get_trend_detection_index(recorder, technical_module):
    recorder.capture(technical_module.get_trend_detection_index())
    recorder.capture(technical_module.get_trend_detection_index(window=10, multiple=3))
    recorder.capture(
        technical_module.get_trend_detection_index(growth=True, lag=[1, 2])
    )


def test_get_dv_intermediate_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_dv_intermediate_oscillator())
    recorder.capture(
        technical_module.get_dv_intermediate_oscillator(
            window=100, magnitude_weight=0.5
        )
    )
    recorder.capture(
        technical_module.get_dv_intermediate_oscillator(period="weekly", window=26)
    )


def test_get_guppy_multiple_moving_averages(recorder, technical_module):
    recorder.capture(technical_module.get_guppy_multiple_moving_averages())
    recorder.capture(
        technical_module.get_guppy_multiple_moving_averages(
            short_windows=[5, 10], long_windows=[50, 100]
        )
    )
    recorder.capture(
        technical_module.get_guppy_multiple_moving_averages(growth=True, lag=[1, 2])
    )


def test_guppy_multiple_moving_averages_keep_the_order_of_their_windows(
    technical_module,
):
    result = technical_module.get_guppy_multiple_moving_averages()
    names = list(dict.fromkeys(result.columns.get_level_values(0)))

    assert names[:6] == [f"Short EMA {window}" for window in [3, 5, 8, 10, 12, 15]]
    assert names[6:12] == [f"Long EMA {window}" for window in [30, 35, 40, 45, 50, 60]]
    assert names[12:] == ["Short Group Spread", "Long Group Spread", "Group Spread"]


def test_ttr_indicator_defaults():
    import inspect  # noqa: PLC0415

    from financetoolkit.technicals.technicals_controller import (  # noqa: PLC0415
        Technicals,
    )

    def defaults(method):
        return {
            name: parameter.default
            for name, parameter in inspect.signature(method).parameters.items()
        }

    # TTR's defaults: Pee's 20 periods with a long window twice as long, Varadi's year of
    # trading days with a weight of 0.8 on the magnitude and PBands' 20 and 2 periods.
    assert defaults(Technicals.get_trend_detection_index)["multiple"] == 2
    assert defaults(Technicals.get_trend_detection_index)["window"] == 20
    dvi = defaults(Technicals.get_dv_intermediate_oscillator)
    assert (dvi["window"], dvi["magnitude_weight"]) == (252, 0.8)
    bands = defaults(Technicals.get_bollinger_bands)
    assert (bands["window"], bands["num_std_dev"], bands["short_window"]) == (20, 2, 2)


def test_get_moving_average_triple_exponential(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="triple_exponential"))
    recorder.capture(
        technical_module.get_moving_average(method="triple_exponential", window=30)
    )
    recorder.capture(
        technical_module.get_moving_average(
            method="triple_exponential", growth=True, lag=[1, 2]
        )
    )


def test_get_moving_average_zero_lag(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="zero_lag"))
    recorder.capture(technical_module.get_moving_average(method="zero_lag", window=30))
    recorder.capture(
        technical_module.get_moving_average(method="zero_lag", growth=True, lag=[1, 2])
    )


def test_get_moving_average_wilder(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="wilder"))
    recorder.capture(technical_module.get_moving_average(method="wilder", window=30))
    recorder.capture(
        technical_module.get_moving_average(method="wilder", growth=True, lag=[1, 2])
    )


def test_get_moving_average_sine(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="sine"))
    recorder.capture(technical_module.get_moving_average(method="sine", window=30))
    recorder.capture(
        technical_module.get_moving_average(method="sine", growth=True, lag=[1, 2])
    )


def test_get_moving_average_fibonacci(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="fibonacci"))
    recorder.capture(technical_module.get_moving_average(method="fibonacci", window=30))
    recorder.capture(
        technical_module.get_moving_average(method="fibonacci", growth=True, lag=[1, 2])
    )


def test_get_moving_average_pascal(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="pascal"))
    recorder.capture(technical_module.get_moving_average(method="pascal", window=30))
    recorder.capture(
        technical_module.get_moving_average(method="pascal", growth=True, lag=[1, 2])
    )


def test_get_moving_average_t3(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="t3"))
    recorder.capture(
        technical_module.get_moving_average(method="t3", window=10, volume_factor=0.8)
    )
    recorder.capture(technical_module.get_moving_average(method="t3", growth=True))


def test_get_moving_average_arnaud_legoux(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="arnaud_legoux"))
    recorder.capture(
        technical_module.get_moving_average(
            method="arnaud_legoux", window=20, sigma=4.0, offset=0.5
        )
    )
    recorder.capture(
        technical_module.get_moving_average(method="arnaud_legoux", growth=True)
    )


def test_get_moving_average_volume_weighted(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="volume_weighted"))
    recorder.capture(
        technical_module.get_moving_average(method="volume_weighted", window=30)
    )
    recorder.capture(
        technical_module.get_moving_average(method="volume_weighted", growth=True)
    )


def test_get_moving_average_mcginley(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="mcginley"))
    recorder.capture(
        technical_module.get_moving_average(method="mcginley", window=10, constant=0.6)
    )
    recorder.capture(
        technical_module.get_moving_average(method="mcginley", growth=True)
    )


def test_get_moving_average_variable_index_dynamic(recorder, technical_module):
    recorder.capture(
        technical_module.get_moving_average(method="variable_index_dynamic")
    )
    recorder.capture(
        technical_module.get_moving_average(
            method="variable_index_dynamic", window=14, cmo_window=14
        )
    )
    recorder.capture(
        technical_module.get_moving_average(
            method="variable_index_dynamic", growth=True
        )
    )


def test_get_moving_average_fractal_adaptive(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="fractal_adaptive"))
    recorder.capture(
        technical_module.get_moving_average(method="fractal_adaptive", window=10)
    )
    recorder.capture(
        technical_module.get_moving_average(method="fractal_adaptive", growth=True)
    )


def test_get_price_transform(recorder, technical_module):
    recorder.capture(technical_module.get_price_transform())
    recorder.capture(technical_module.get_price_transform(method="average"))
    recorder.capture(technical_module.get_price_transform(method="weighted_close"))


def test_get_midpoint(recorder, technical_module):
    recorder.capture(technical_module.get_midpoint())
    recorder.capture(technical_module.get_midpoint(method="high_low", window=30))
    recorder.capture(technical_module.get_midpoint(growth=True))


def test_get_moving_average_holt_winters(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="holt_winters"))
    recorder.capture(
        technical_module.get_moving_average(
            method="holt_winters",
            level_smoothing=0.3,
            velocity_smoothing=0.2,
            acceleration_smoothing=0.05,
        )
    )
    recorder.capture(
        technical_module.get_moving_average(method="holt_winters", growth=True)
    )


def test_get_holt_winters_channel(recorder, technical_module):
    recorder.capture(technical_module.get_holt_winters_channel())
    recorder.capture(
        technical_module.get_holt_winters_channel(deviation_smoothing=0.2, multiplier=2)
    )
    recorder.capture(technical_module.get_holt_winters_channel(growth=True))


def test_get_moving_average_elastic_volume_weighted(recorder, technical_module):
    recorder.capture(
        technical_module.get_moving_average(method="elastic_volume_weighted")
    )
    recorder.capture(
        technical_module.get_moving_average(method="elastic_volume_weighted", window=10)
    )
    recorder.capture(
        technical_module.get_moving_average(
            method="elastic_volume_weighted", growth=True
        )
    )


def test_get_moving_average_volume_adjusted(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="volume_adjusted"))
    recorder.capture(
        technical_module.get_moving_average(method="volume_adjusted", window=14)
    )
    recorder.capture(
        technical_module.get_moving_average(method="volume_adjusted", growth=True)
    )


def test_get_gann_hilo_activator(recorder, technical_module):
    recorder.capture(technical_module.get_gann_hilo_activator())
    recorder.capture(
        technical_module.get_gann_hilo_activator(high_window=3, low_window=3)
    )
    recorder.capture(technical_module.get_gann_hilo_activator(growth=True))


def test_get_zigzag(recorder, technical_module):
    recorder.capture(technical_module.get_zigzag())
    recorder.capture(
        technical_module.get_zigzag(sensitivity=0.1, minimum_trend_length=3)
    )


def test_get_trix_signal(recorder, technical_module):
    recorder.capture(technical_module.get_trix(window=18, signal_window=9))
    recorder.capture(technical_module.get_trix(signal_window=9, growth=True))


def test_get_parabolic_sar_extended(recorder, technical_module):
    recorder.capture(
        technical_module.get_parabolic_sar(
            af_start_short=0.01, af_increment_short=0.01, af_max_short=0.1
        )
    )
    recorder.capture(
        technical_module.get_parabolic_sar(start_value=-100, offset_on_reverse=0.01)
    )


def test_get_pivot_points_methods(recorder, technical_module):
    recorder.capture(technical_module.get_pivot_points(method="fibonacci"))
    recorder.capture(
        technical_module.get_pivot_points(method="camarilla", central_pivot_range=True)
    )
    recorder.capture(technical_module.get_pivot_points(method="woodie"))
    recorder.capture(technical_module.get_pivot_points(method="demark"))


def test_get_volume_weighted_average_price_anchored(recorder, technical_module):
    recorder.capture(technical_module.get_volume_weighted_average_price(anchor="all"))
    recorder.capture(
        technical_module.get_volume_weighted_average_price(anchor="monthly")
    )
    recorder.capture(
        technical_module.get_volume_weighted_average_price(anchor="2022-01-03")
    )


def test_get_bollinger_bands_moving_average(recorder, technical_module):
    recorder.capture(technical_module.get_bollinger_bands(moving_average="exponential"))
    recorder.capture(technical_module.get_bollinger_bands(moving_average="hull"))


def test_get_candlestick_patterns(recorder, technical_module):
    recorder.capture(technical_module.get_candlestick_patterns())
    recorder.capture(
        technical_module.get_candlestick_patterns(
            patterns="belt-hold, Inside Bar", signals_only=True
        )
    )
    recorder.capture(
        technical_module.get_candlestick_patterns(
            patterns="Three Outside, kicking by length, harami cross", signals_only=True
        )
    )
    recorder.capture(
        technical_module.get_candlestick_patterns(
            patterns="Tasuki Gap, gap three methods, Stick Sandwich, advance block",
            signals_only=True,
        )
    )
    recorder.capture(
        technical_module.get_candlestick_patterns(
            patterns="Hikkake, ladder bottom, three-line strike, rising/falling three methods",
            signals_only=True,
        )
    )


def test_get_heikin_ashi(recorder, technical_module):
    recorder.capture(technical_module.get_heikin_ashi())
    recorder.capture(technical_module.get_heikin_ashi(period="weekly"))


def test_get_williams_fractal(recorder, technical_module):
    recorder.capture(technical_module.get_williams_fractal())
    recorder.capture(
        technical_module.get_williams_fractal(
            left_bars=3, right_bars=1, signals_only=True
        )
    )


def test_get_td_sequential(recorder, technical_module):
    recorder.capture(technical_module.get_td_sequential())
    recorder.capture(technical_module.get_td_sequential(period="weekly"))


def test_get_liquidity_sweep(recorder, technical_module):
    recorder.capture(technical_module.get_liquidity_sweep())
    recorder.capture(
        technical_module.get_liquidity_sweep(
            window=5, wick_multiple=1.0, signals_only=True
        )
    )


def test_williams_fractal_is_confirmed_after_the_right_bars(technical_module):
    fractals = technical_module.get_williams_fractal(close_column="Close")
    high = technical_module._historical_data["daily"]["High"]

    ticker = high.columns[0]
    confirmed = fractals["Up Fractal"][ticker].eq(1).to_numpy()
    pivots = np.flatnonzero(confirmed) - 2
    window = high[ticker].to_numpy()
    for pivot in pivots:
        neighbours = np.delete(window[pivot - 2 : pivot + 3], 2)
        assert window[pivot] > neighbours.max()


def test_get_average_true_range_normalized(recorder, technical_module):
    recorder.capture(technical_module.get_average_true_range(method="normalized"))
    recorder.capture(
        technical_module.get_average_true_range(method="normalized", window=20)
    )
    recorder.capture(
        technical_module.get_average_true_range(method="normalized", growth=True)
    )


def test_get_average_true_range_day_range(recorder, technical_module):
    recorder.capture(technical_module.get_average_true_range(method="day_range"))
    recorder.capture(
        technical_module.get_average_true_range(method="day_range", window=5)
    )
    recorder.capture(
        technical_module.get_average_true_range(method="day_range", period="weekly")
    )


def test_get_chaikin_volatility(recorder, technical_module):
    recorder.capture(technical_module.get_chaikin_volatility())
    recorder.capture(technical_module.get_chaikin_volatility(window=14, roc_window=5))
    recorder.capture(technical_module.get_chaikin_volatility(growth=True))


def test_get_relative_volatility_index(recorder, technical_module):
    recorder.capture(technical_module.get_relative_volatility_index())
    recorder.capture(technical_module.get_relative_volatility_index(method="refined"))
    recorder.capture(
        technical_module.get_relative_volatility_index(window=10, std_window=20)
    )


def test_get_acceleration_bands(recorder, technical_module):
    recorder.capture(technical_module.get_acceleration_bands())
    recorder.capture(technical_module.get_acceleration_bands(window=10, multiplier=2.0))
    recorder.capture(technical_module.get_acceleration_bands(growth=True))


def test_get_chande_kroll_stop(recorder, technical_module):
    recorder.capture(technical_module.get_chande_kroll_stop())
    recorder.capture(
        technical_module.get_chande_kroll_stop(
            window=10, multiplier=3.0, stop_window=20
        )
    )
    recorder.capture(technical_module.get_chande_kroll_stop(growth=True))


def test_get_adaptive_price_zone(recorder, technical_module):
    recorder.capture(technical_module.get_adaptive_price_zone())
    recorder.capture(
        technical_module.get_adaptive_price_zone(window=30, multiplier=1.4)
    )
    recorder.capture(technical_module.get_adaptive_price_zone(growth=True))


def test_get_keltner_channels_aberration(recorder, technical_module):
    recorder.capture(technical_module.get_keltner_channels(method="aberration"))
    recorder.capture(
        technical_module.get_keltner_channels(
            method="aberration", window=10, atr_window=10
        )
    )
    recorder.capture(
        technical_module.get_keltner_channels(method="aberration", growth=True)
    )


def test_get_supertrend_profit_maximizer(recorder, technical_module):
    recorder.capture(technical_module.get_supertrend(method="profit_maximizer"))
    recorder.capture(
        technical_module.get_supertrend(
            method="profit_maximizer", window=20, atr_window=14, multiplier=2.0
        )
    )
    recorder.capture(
        technical_module.get_supertrend(method="profit_maximizer", period="weekly")
    )


def test_get_price_distance(recorder, technical_module):
    recorder.capture(technical_module.get_price_distance())
    recorder.capture(technical_module.get_price_distance(period="weekly"))
    recorder.capture(technical_module.get_price_distance(growth=True))


def test_get_market_thermometer(recorder, technical_module):
    recorder.capture(technical_module.get_market_thermometer())
    recorder.capture(technical_module.get_market_thermometer(window=20))
    recorder.capture(technical_module.get_market_thermometer(growth=True))


def test_get_squeeze_momentum(recorder, technical_module):
    recorder.capture(technical_module.get_squeeze_momentum())
    recorder.capture(technical_module.get_squeeze_momentum(keltner_multipliers=[1.5]))
    recorder.capture(technical_module.get_squeeze_momentum(window=10, num_std_dev=1.5))


def test_get_keltner_channels_original(recorder, technical_module):
    recorder.capture(
        technical_module.get_keltner_channels(method="original", multiplier=1)
    )
    recorder.capture(
        technical_module.get_keltner_channels(
            method="original", window=10, multiplier=1
        )
    )
    recorder.capture(technical_module.get_keltner_channels(method="original"))


def test_get_donchian_channels_include_current(recorder, technical_module):
    recorder.capture(technical_module.get_donchian_channels(include_current=True))
    recorder.capture(
        technical_module.get_donchian_channels(include_current=True, window=10)
    )


def test_keltner_and_donchian_defaults_are_unchanged(technical_module):
    keltner = technical_module.get_keltner_channels()
    assert keltner.equals(
        technical_module.get_keltner_channels(multiplier=2, method="exponential")
    )
    donchian = technical_module.get_donchian_channels()
    current = technical_module.get_donchian_channels(include_current=True)
    # The default channel ends one period before the current one.
    pd.testing.assert_frame_equal(
        donchian.iloc[1:], current.shift(1).iloc[1:], check_freq=False
    )


def test_volatility_default_windows():
    import inspect  # noqa: PLC0415

    from financetoolkit.technicals.technicals_controller import (  # noqa: PLC0415
        Technicals,
    )

    # The defaults of the primary sources, or TA-Lib's where the source has none.
    for method, defaults in (
        (Technicals.get_average_true_range, {"window": 14}),
        (Technicals.get_chaikin_volatility, {"window": 10, "roc_window": 10}),
        (Technicals.get_relative_volatility_index, {"window": 14, "std_window": 10}),
        (Technicals.get_acceleration_bands, {"window": 20, "multiplier": 4.0}),
        (
            Technicals.get_chande_kroll_stop,
            {"window": 10, "multiplier": 1.0, "stop_window": 9},
        ),
        (Technicals.get_adaptive_price_zone, {"window": 20, "multiplier": 2.0}),
        (
            Technicals.get_supertrend,
            {"window": 10, "atr_window": 10, "multiplier": 3.0},
        ),
        (Technicals.get_market_thermometer, {"window": 22}),
        (Technicals.get_squeeze_momentum, {"window": 20, "num_std_dev": 2.0}),
    ):
        parameters = inspect.signature(method).parameters
        for name, default in defaults.items():
            assert parameters[name].default == default


def test_method_dependent_default_windows(technical_module):
    # A window of None uses the default of the method it calculates.
    for name, method, windows in (
        ("get_keltner_channels", "exponential", {"window": 20, "atr_window": 10}),
        ("get_keltner_channels", "aberration", {"window": 5, "atr_window": 15}),
        ("get_chaikin_money_flow", "standard", {"window": 20}),
        ("get_chaikin_money_flow", "twiggs", {"window": 21}),
        ("get_relative_strength_index", "standard", {"window": 14}),
        ("get_relative_strength_index", "connors", {"window": 3}),
        ("get_rate_of_change", "percentage", {"window": 12}),
        ("get_rate_of_change", "absolute", {"window": 10}),
        ("get_moving_average", "simple", {"window": 14}),
        ("get_moving_average", "kaufman_adaptive", {"window": 10}),
        ("get_moving_average", "t3", {"window": 5}),
        ("get_moving_average", "fractal_adaptive", {"window": 16}),
    ):
        method_function = getattr(technical_module, name)
        pd.testing.assert_frame_equal(
            method_function(method=method),
            method_function(method=method, **windows),
        )


def test_merged_methods_reject_an_unknown_method(technical_module):
    for name in (
        "get_relative_strength_index",
        "get_stochastic_oscillator",
        "get_moving_average_convergence_divergence",
        "get_average_true_range",
        "get_average_directional_index",
        "get_rate_of_change",
        "get_hilbert_sine_wave",
        "get_supertrend",
        "get_chaikin_money_flow",
        "get_keltner_channels",
        "get_volume_index",
    ):
        with pytest.raises(InvalidArgumentError, match="method must be one of"):
            getattr(technical_module, name)(method="unknown")


def test_get_hilbert_dominant_cycle_period(recorder, technical_module):
    recorder.capture(technical_module.get_hilbert_dominant_cycle_period())
    recorder.capture(technical_module.get_hilbert_dominant_cycle_period(growth=True))
    recorder.capture(
        technical_module.get_hilbert_dominant_cycle_period(period="weekly")
    )


def test_get_hilbert_dominant_cycle_phase(recorder, technical_module):
    recorder.capture(technical_module.get_hilbert_dominant_cycle_phase())
    recorder.capture(
        technical_module.get_hilbert_dominant_cycle_phase(close_column="Close")
    )


def test_get_hilbert_phasor_components(recorder, technical_module):
    recorder.capture(technical_module.get_hilbert_phasor_components())
    recorder.capture(technical_module.get_hilbert_phasor_components(growth=True))


def test_get_hilbert_sine_wave(recorder, technical_module):
    recorder.capture(technical_module.get_hilbert_sine_wave())
    recorder.capture(technical_module.get_hilbert_sine_wave(period="weekly"))


def test_get_hilbert_trend_mode(recorder, technical_module):
    recorder.capture(technical_module.get_hilbert_trend_mode())
    recorder.capture(technical_module.get_hilbert_trend_mode(close_column="Close"))


def test_get_moving_average_hilbert_trendline(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="hilbert_trendline"))
    recorder.capture(
        technical_module.get_moving_average(method="hilbert_trendline", growth=True)
    )


def test_get_mesa_adaptive_moving_average(recorder, technical_module):
    recorder.capture(technical_module.get_mesa_adaptive_moving_average())
    recorder.capture(
        technical_module.get_mesa_adaptive_moving_average(
            fast_limit=0.3, slow_limit=0.02
        )
    )
    recorder.capture(technical_module.get_mesa_adaptive_moving_average(growth=True))


def test_get_swiss_army_knife_filter(recorder, technical_module):
    recorder.capture(technical_module.get_swiss_army_knife_filter())
    recorder.capture(
        technical_module.get_swiss_army_knife_filter(
            method="gaussian", median_price=False
        )
    )
    recorder.capture(technical_module.get_swiss_army_knife_filter(method="high_pass"))
    recorder.capture(
        technical_module.get_swiss_army_knife_filter(
            method="two_pole_high_pass", window=30
        )
    )
    recorder.capture(technical_module.get_swiss_army_knife_filter(method="butterworth"))
    recorder.capture(technical_module.get_swiss_army_knife_filter(delta=0.3))


def test_get_moving_average_super_smoother(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average(method="super_smoother"))
    recorder.capture(
        technical_module.get_moving_average(method="super_smoother", poles=3, window=20)
    )
    recorder.capture(
        technical_module.get_moving_average(method="super_smoother", period="weekly")
    )


def test_get_even_better_sinewave(recorder, technical_module):
    recorder.capture(technical_module.get_even_better_sinewave())
    recorder.capture(
        technical_module.get_even_better_sinewave(window=50, smooth_window=15)
    )


def test_get_detrended_synthetic_price(recorder, technical_module):
    recorder.capture(technical_module.get_detrended_synthetic_price())
    recorder.capture(technical_module.get_detrended_synthetic_price(window=30))


def test_get_hilbert_sine_wave_mesa(recorder, technical_module):
    recorder.capture(technical_module.get_hilbert_sine_wave(method="mesa"))
    recorder.capture(
        technical_module.get_hilbert_sine_wave(
            method="mesa", median_price=False, window=20
        )
    )


def test_get_center_of_gravity_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_center_of_gravity_oscillator())
    recorder.capture(
        technical_module.get_center_of_gravity_oscillator(
            median_price=False, close_column="Close"
        )
    )


def test_get_relative_strength_index_laguerre(recorder, technical_module):
    recorder.capture(technical_module.get_relative_strength_index(method="laguerre"))
    recorder.capture(
        technical_module.get_relative_strength_index(method="laguerre", gamma=0.8)
    )


def test_volume_profile_of_a_constant_close_keeps_the_other_tickers():
    from financetoolkit.technicals.technicals_controller import Technicals

    index = pd.period_range("2024-01-01", periods=4, freq="D")
    columns = pd.MultiIndex.from_product([["Adj Close", "Volume"], ["A", "B"]])
    data = pd.DataFrame(
        [
            [10.0, 20.0, 100.0, 300.0],
            [10.0, 21.0, 200.0, 100.0],
            [10.0, 22.0, 150.0, 200.0],
            [10.0, 23.0, 50.0, 400.0],
        ],
        index=index,
        columns=columns,
    )
    technicals = Technicals(["A", "B"], {"daily": data}, rounding=None)

    profile = technicals.get_volume_profile(bins=3)

    assert profile[("Total Volume", "A")].tolist() == [500.0, 0.0, 0.0]
    # A close on an inner edge (21) belongs to the lower bin.
    assert profile[("Total Volume", "B")].tolist() == [400.0, 200.0, 400.0]


def test_get_event_filter(recorder, technical_module):
    recorder.capture(technical_module.get_event_filter())
    recorder.capture(technical_module.get_event_filter(method="z_score", threshold=2))


def test_get_triple_barrier_labels(recorder, technical_module):
    recorder.capture(technical_module.get_triple_barrier_labels())
    recorder.capture(
        technical_module.get_triple_barrier_labels(
            events="cusum", profit_taking=2, horizon=5, vertical_barrier_zero=True
        )
    )
    meta = technical_module.get_triple_barrier_labels(side="Moving Average Crossover")
    recorder.capture(meta)

    # Meta-labels only on the crossovers, 1 or 0.
    signals = technical_module.get_signals()["Moving Average Crossover"]
    labeled = meta["Label"].notna()
    assert (signals.reindex_like(labeled)[labeled] != 0).all().all()
    assert set(meta["Label"].stack().dropna().unique()) <= {0, 1}


def test_get_fixed_horizon_labels(recorder, technical_module):
    recorder.capture(technical_module.get_fixed_horizon_labels())
    recorder.capture(
        technical_module.get_fixed_horizon_labels(
            horizon=5, threshold=0.5, volatility_scaled=True, relative_to="benchmark"
        )
    )
    relative = technical_module.get_fixed_horizon_labels(relative_to="mean")
    assert "Benchmark" not in relative.columns.get_level_values(1)


def test_get_trend_scanning_labels(recorder, technical_module):
    recorder.capture(technical_module.get_trend_scanning_labels())
    recorder.capture(
        technical_module.get_trend_scanning_labels(minimum_window=10, maximum_window=30)
    )


def test_get_tail_set_labels(recorder, technical_module):
    labels = technical_module.get_tail_set_labels()
    recorder.capture(labels)
    # Two tickers: one in each tail on every date with both scores.
    assert (labels["Label"].dropna().sum(axis=1) == 0).all()


def test_get_matrix_flag_labels(recorder, technical_module):
    recorder.capture(technical_module.get_matrix_flag_labels())
    recorder.capture(
        technical_module.get_matrix_flag_labels(
            window=25, template="cervello_royo_bull", threshold=1
        )
    )


def test_get_label_uniqueness(recorder, technical_module):
    recorder.capture(technical_module.get_label_uniqueness())
    labels = technical_module.get_fixed_horizon_labels(horizon=5)
    uniqueness = technical_module.get_label_uniqueness(labels=labels)
    recorder.capture(uniqueness)
    # A label every date over six dates: six concurrent labels in the middle, a sixth each.
    assert uniqueness["Concurrency"]["AAPL"].iloc[100] == 6
    assert uniqueness["Average Uniqueness"]["AAPL"].iloc[100] == pytest.approx(
        1 / 6, abs=1e-4
    )


def test_get_sample_weights(recorder, technical_module):
    weights = technical_module.get_sample_weights()
    recorder.capture(weights)
    recorder.capture(
        technical_module.get_sample_weights(method="time_decay", decay=0.5)
    )
    # The weights add up to the number of labels per ticker.
    pd.testing.assert_series_equal(
        weights.sum(), weights.count().astype(float), check_exact=False, atol=0.01
    )


def test_labeling_invalid_parameters(technical_module):
    with pytest.raises(InvalidArgumentError, match="method must be one of"):
        technical_module.get_event_filter(method="unknown")
    with pytest.raises(InvalidArgumentError, match="relative_to must be one of"):
        technical_module.get_fixed_horizon_labels(relative_to="unknown")
    with pytest.raises(InvalidArgumentError, match="template must be one of"):
        technical_module.get_matrix_flag_labels(template="unknown")
    assert technical_module.get_sample_weights(method="time_decay", decay=-1).empty
    assert technical_module.get_triple_barrier_labels(side="unknown").empty


def test_get_bet_sizes(recorder, technical_module):
    budget = technical_module.get_bet_sizes()
    recorder.capture(budget)
    # The EF3M fit reaches one exact solution of the moment equations from every seed, so
    # the recorded sizes do not depend on the random starting points.
    reserve = technical_module.get_bet_sizes(method="reserve")
    recorder.capture(reserve)
    pd.testing.assert_frame_equal(
        reserve, technical_module.get_bet_sizes(method="reserve", seed=7)
    )
    recorder.capture(technical_module.get_bet_sizes(method="probability"))
    recorder.capture(technical_module.get_bet_sizes(step_size=0.5))
    assert budget["Bet Size"].abs().max().max() <= 1
    with pytest.raises(InvalidArgumentError, match="method must be one of"):
        technical_module.get_bet_sizes(method="unknown")
    assert technical_module.get_bet_sizes(step_size=2).empty


# Robustness: invalid windows, columns and options raise, and a window longer than the data
# says why the indicator has no values.


@pytest.mark.parametrize(
    ("method", "arguments"),
    [
        ("get_relative_strength_index", {"window": 0}),
        ("get_moving_average", {"window": 0}),
        ("get_bollinger_bands", {"window": -1}),
        ("get_moving_average_convergence_divergence", {"long_window": 0}),
        ("get_stochastic_oscillator", {"smooth_window": 0}),
        ("get_triple_barrier_labels", {"horizon": 0}),
        ("get_relative_strength_index", {"window": 2.5}),
    ],
)
def test_a_window_below_one_period_is_rejected(technical_module, method, arguments):
    with pytest.raises(InvalidArgumentError, match="at least 1"):
        getattr(technical_module, method)(**arguments)


def test_trend_scanning_windows_are_checked(technical_module, caplog):
    with pytest.raises(InvalidArgumentError, match="minimum_window"):
        technical_module.get_trend_scanning_labels(minimum_window=2)
    with pytest.raises(InvalidArgumentError, match="maximum_window"):
        technical_module.get_trend_scanning_labels(minimum_window=10, maximum_window=5)

    # A maximum window longer than the data returns at once, without labels and with a reason.
    with caplog.at_level("WARNING", logger="financetoolkit"):
        labels = technical_module.get_trend_scanning_labels(maximum_window=5000)
    assert labels.isna().all().all()
    assert "A window of 5000 periods is longer than" in caplog.text


def test_a_window_longer_than_the_data_is_logged_once(technical_module, caplog):
    with caplog.at_level("WARNING", logger="financetoolkit"):
        technical_module.collect_overlap_indicators(window=1400)
    # Every indicator of the collection gets the window, the collection reports it once.
    assert caplog.text.count("collect_overlap_indicators over") == 1
    assert "the windows of 1400 periods are longer than the data" in caplog.text
    assert "A window of 1400 periods" not in caplog.text

    # A standalone indicator reports its window once per call, also when the same instance
    # is called again (the MCP server keeps a Toolkit for many requests).
    for _ in range(2):
        caplog.clear()
        with caplog.at_level("WARNING", logger="financetoolkit"):
            technical_module.get_moving_average(window=1400)
        assert caplog.text.count("A window of 1400 periods is longer than") == 1


@pytest.mark.parametrize(
    "method",
    [
        "get_relative_strength_index",
        "get_bollinger_bands",
        "get_moving_average_convergence_divergence",
        "get_trin",
        "get_seasonality",
    ],
)
def test_an_unknown_close_column_is_rejected(technical_module, method):
    with pytest.raises(InvalidArgumentError, match="close_column must be one of"):
        getattr(technical_module, method)(close_column="Bogus")


def test_unknown_options_raise_an_invalid_argument_error(technical_module):
    with pytest.raises(InvalidArgumentError, match="signal_moving_average"):
        technical_module.get_moving_average_convergence_divergence(
            signal_moving_average="bogus"
        )
    with pytest.raises(InvalidArgumentError, match="moving_average"):
        technical_module.get_know_sure_thing(moving_average="bogus")
    with pytest.raises(InvalidArgumentError, match="get_bogus is not a metric"):
        technical_module.get_indicator_summary(metric="get_bogus")
    with pytest.raises(InvalidArgumentError, match="get_bogus is not a metric"):
        technical_module.get_factor_analysis(metrics="get_bogus")


def test_trin_says_why_it_has_no_values(caplog):
    from financetoolkit import Toolkit

    historical = pd.read_pickle("tests/datasets/historical_dataset.pickle")
    historical = historical.loc[
        :, historical.columns.get_level_values(1).isin(["AAPL", "Benchmark"])
    ]
    toolkit = Toolkit(
        tickers=["AAPL"],
        historical=historical,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
        progress_bar=False,
    )
    toolkit._daily_risk_free_rate = pd.read_pickle(
        "tests/datasets/risk_free_rate.pickle"
    )
    toolkit._daily_treasury_data = pd.read_pickle("tests/datasets/treasury_data.pickle")
    with caplog.at_level("WARNING", logger="financetoolkit"):
        trin = toolkit.technicals.get_trin()
    assert trin.isna().all().all()
    assert "fewer than two tickers" in caplog.text


def test_a_factor_analysis_without_rankings_names_a_few_metrics(
    technical_module, monkeypatch
):
    # Three tickers can't be ranked; the message names a few of the metrics, not all of them.
    monkeypatch.setenv("FINANCETOOLKIT_STRICT_ERRORS", "1")
    with pytest.raises(
        ValueError, match=r"^\d+ metrics \(get_\w+, get_\w+, get_\w+ and others\)"
    ):
        technical_module.get_factor_analysis()


@pytest.mark.parametrize(
    "method", ["get_commodity_channel_index", "get_choppiness_index"]
)
def test_a_window_of_one_period_is_rejected_where_two_are_needed(
    technical_module, method
):
    with pytest.raises(InvalidArgumentError, match="at least 2"):
        getattr(technical_module, method)(window=1)


def test_a_yearly_collection_reports_its_long_windows_once(technical_module, caplog):
    with caplog.at_level("WARNING", logger="financetoolkit"):
        technical_module.collect_all_indicators(period="yearly")

    assert caplog.text.count("collect_all_indicators over") == 1
    assert "A window of" not in caplog.text
    assert "has no values for the" not in caplog.text


@pytest.mark.parametrize(
    ("method", "arguments"),
    [
        ("get_relative_strength_index", {}),
        ("get_trix", {"window": 2}),
    ],
)
def test_an_indicator_without_values_says_why(
    technical_module, method, arguments, caplog
):
    # The window of the RSI defaults to None (14 periods) and the TRIX smooths three
    # times, so neither has a value on four yearly prices.
    with caplog.at_level("WARNING", logger="financetoolkit"):
        result = getattr(technical_module, method)(period="yearly", **arguments)

    assert result.isna().all().all()
    assert caplog.text.count(f"{method} has no values for the") == 1


def test_the_factor_analysis_of_a_short_history_reports_its_windows_once(caplog):
    from financetoolkit import Toolkit  # noqa: PLC0415

    historical = pd.read_pickle("tests/datasets/historical_dataset.pickle").drop(
        columns=["Return", "Cumulative Return"], level=0
    )
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=historical.loc[historical.index >= pd.Period("2022-12-29", "D")],
        convert_currency=False,
        start_date="2022-12-28",
        end_date="2023-01-01",
        sleep_timer=False,
    )
    toolkit._daily_risk_free_rate = pd.read_pickle(
        "tests/datasets/risk_free_rate.pickle"
    )
    toolkit._daily_treasury_data = pd.read_pickle("tests/datasets/treasury_data.pickle")

    with caplog.at_level("WARNING", logger="financetoolkit"):
        toolkit.technicals.get_factor_analysis()

    # One message for the windows of every indicator, counted in the daily periods they use.
    assert caplog.text.count("get_factor_analysis over 2 daily periods of data") == 1
    assert "A window of" not in caplog.text
