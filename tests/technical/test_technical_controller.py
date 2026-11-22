"""Technical Controller Tests""" ""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.technicals.pattern import pattern_model


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


def test_collect_statistics_indicators(recorder, technical_module):
    recorder.capture(technical_module.collect_statistics_indicators())
    recorder.capture(technical_module.collect_statistics_indicators(growth=True))
    recorder.capture(
        technical_module.collect_statistics_indicators(growth=True, lag=[1, 2, 3])
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


def test_get_ultimate_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_ultimate_oscillator())
    recorder.capture(technical_module.get_ultimate_oscillator(growth=True))
    recorder.capture(
        technical_module.get_ultimate_oscillator(growth=True, lag=[1, 2, 3])
    )


def test_get_percentage_price_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_percentage_price_oscillator())
    recorder.capture(technical_module.get_percentage_price_oscillator(growth=True))
    recorder.capture(
        technical_module.get_percentage_price_oscillator(growth=True, lag=[1, 2, 3])
    )
    recorder.capture(
        technical_module.get_percentage_price_oscillator(
            signal_window=5, moving_average="simple"
        )
    )


def test_get_absolute_price_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_absolute_price_oscillator())
    recorder.capture(
        technical_module.get_absolute_price_oscillator(moving_average="simple")
    )
    recorder.capture(
        technical_module.get_absolute_price_oscillator(growth=True, lag=[1, 2, 3])
    )


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


def test_get_average_directional_index_rating(recorder, technical_module):
    recorder.capture(technical_module.get_average_directional_index_rating())
    recorder.capture(
        technical_module.get_average_directional_index_rating(period="weekly")
    )
    recorder.capture(
        technical_module.get_average_directional_index_rating(
            growth=True, lag=[1, 2, 3]
        )
    )


def test_get_chande_momentum_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_chande_momentum_oscillator())
    recorder.capture(technical_module.get_chande_momentum_oscillator(growth=True))
    recorder.capture(
        technical_module.get_chande_momentum_oscillator(growth=True, lag=[1, 2, 3])
    )
    recorder.capture(technical_module.get_chande_momentum_oscillator(smoothed=True))


def test_get_intraday_momentum_index(recorder, technical_module):
    recorder.capture(technical_module.get_intraday_momentum_index())
    recorder.capture(technical_module.get_intraday_momentum_index(window=7))
    recorder.capture(
        technical_module.get_intraday_momentum_index(growth=True, lag=[1, 2, 3])
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


def test_get_stochastic_relative_strength_index(recorder, technical_module):
    recorder.capture(technical_module.get_stochastic_relative_strength_index())
    recorder.capture(
        technical_module.get_stochastic_relative_strength_index(stochastic_window=5)
    )
    recorder.capture(
        technical_module.get_stochastic_relative_strength_index(
            growth=True, lag=[1, 2, 3]
        )
    )


def test_get_true_strength_index(recorder, technical_module):
    recorder.capture(technical_module.get_true_strength_index())
    recorder.capture(technical_module.get_true_strength_index(period="weekly"))
    recorder.capture(
        technical_module.get_true_strength_index(growth=True, lag=[1, 2, 3])
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


def test_get_accelerator_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_accelerator_oscillator())
    recorder.capture(technical_module.get_accelerator_oscillator(signal_window=3))
    recorder.capture(
        technical_module.get_accelerator_oscillator(growth=True, lag=[1, 2, 3])
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


def test_get_momentum(recorder, technical_module):
    recorder.capture(technical_module.get_momentum())
    recorder.capture(technical_module.get_momentum(window=20))
    recorder.capture(technical_module.get_momentum(growth=True, lag=[1, 2, 3]))


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


def test_get_smi_ergodic_indicator(recorder, technical_module):
    recorder.capture(technical_module.get_smi_ergodic_indicator())
    recorder.capture(
        technical_module.get_smi_ergodic_indicator(
            long_window=25, short_window=13, signal_window=7
        )
    )
    recorder.capture(technical_module.get_smi_ergodic_indicator(period="weekly"))


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
            window=10, smooth_window=3, factor=3.0
        )
    )
    recorder.capture(
        technical_module.get_quantitative_qualitative_estimation(period="weekly")
    )


def test_get_dynamic_momentum_index(recorder, technical_module):
    recorder.capture(technical_module.get_dynamic_momentum_index())
    recorder.capture(
        technical_module.get_dynamic_momentum_index(
            window=20, minimum_window=3, maximum_window=40
        )
    )
    recorder.capture(technical_module.get_dynamic_momentum_index(period="weekly"))


def test_get_price_zone_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_price_zone_oscillator())
    recorder.capture(technical_module.get_price_zone_oscillator(window=28))
    recorder.capture(
        technical_module.get_price_zone_oscillator(growth=True, lag=[1, 2])
    )


def test_get_kaufman_adaptive_moving_average(recorder, technical_module):
    recorder.capture(technical_module.get_kaufman_adaptive_moving_average())
    recorder.capture(technical_module.get_kaufman_adaptive_moving_average(growth=True))
    recorder.capture(
        technical_module.get_kaufman_adaptive_moving_average(growth=True, lag=[1, 2, 3])
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
        technical_module.get_volume_flow_indicator(window=50, smoothing_window=5)
    )
    recorder.capture(technical_module.get_volume_flow_indicator(growth=True))


def test_get_finite_volume_elements(recorder, technical_module):
    recorder.capture(technical_module.get_finite_volume_elements())
    recorder.capture(
        technical_module.get_finite_volume_elements(window=10, cutoff=0.01)
    )
    recorder.capture(technical_module.get_finite_volume_elements(period="weekly"))


def test_get_twiggs_money_flow(recorder, technical_module):
    recorder.capture(technical_module.get_twiggs_money_flow())
    recorder.capture(technical_module.get_twiggs_money_flow(window=10))
    recorder.capture(technical_module.get_twiggs_money_flow(growth=True))


def test_get_volume_zone_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_volume_zone_oscillator())
    recorder.capture(technical_module.get_volume_zone_oscillator(window=28))
    recorder.capture(technical_module.get_volume_zone_oscillator(growth=True))


def test_get_cumulative_force_index(recorder, technical_module):
    recorder.capture(technical_module.get_cumulative_force_index().round(0))
    recorder.capture(technical_module.get_cumulative_force_index(window=1).round(0))
    recorder.capture(
        technical_module.get_cumulative_force_index(period="weekly").round(0)
    )


def test_get_price_volume(recorder, technical_module):
    recorder.capture(technical_module.get_price_volume().round(0))
    recorder.capture(technical_module.get_price_volume(signed=True).round(0))
    recorder.capture(technical_module.get_price_volume(growth=True))


def test_get_price_volume_rank(recorder, technical_module):
    recorder.capture(technical_module.get_price_volume_rank())
    recorder.capture(technical_module.get_price_volume_rank(period="weekly"))


def test_get_buying_selling_pressure(recorder, technical_module):
    recorder.capture(technical_module.get_buying_selling_pressure())
    recorder.capture(technical_module.get_buying_selling_pressure(smoothing_window=1))
    recorder.capture(
        technical_module.get_buying_selling_pressure(window=20, smoothing_window=10)
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


def test_get_negative_volume_index(recorder, technical_module):
    recorder.capture(technical_module.get_negative_volume_index())
    recorder.capture(technical_module.get_negative_volume_index(growth=True))
    recorder.capture(
        technical_module.get_negative_volume_index(growth=True, lag=[1, 2, 3])
    )


def test_get_positive_volume_index(recorder, technical_module):
    recorder.capture(technical_module.get_positive_volume_index())
    recorder.capture(technical_module.get_positive_volume_index(growth=True))
    recorder.capture(
        technical_module.get_positive_volume_index(growth=True, lag=[1, 2, 3])
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

    assert defaults(Technicals.get_percentage_price_oscillator)["short_window"] == 12
    assert defaults(Technicals.get_percentage_price_oscillator)["long_window"] == 26
    assert defaults(Technicals.get_force_index)["window"] == 13
    keltner = defaults(Technicals.get_keltner_channels)
    assert (keltner["window"], keltner["atr_window"], keltner["atr_multiplier"]) == (
        20,
        10,
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
            "atr_multiplier",
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
    assert set(values[~pd.isna(values)]) <= {
        -1.0,
        -0.8,
        0.0,
        0.8,
        1.0,
    }
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
    assert technical_module.get_moving_average(method="cubic").empty


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

    for method in (
        "get_average_true_range",
        "get_average_directional_index",
        "get_bollinger_bands",
        "get_money_flow_index",
        "get_ease_of_movement",
        "get_relative_strength_index",
        "get_stochastic_relative_strength_index",
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
        "get_twiggs_money_flow",
        "get_relative_volume",
        "get_volume_flow_indicator",
        "get_candlestick_patterns",
        "get_signals",
    ):
        expected = getattr(alone, method)()
        result = getattr(together, method)()
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


def test_get_linear_regression(recorder, technical_module):
    recorder.capture(technical_module.get_linear_regression())
    recorder.capture(technical_module.get_linear_regression(window=20))
    recorder.capture(technical_module.get_linear_regression(growth=True, lag=[1, 2]))


def test_get_slope(recorder, technical_module):
    recorder.capture(technical_module.get_slope())
    recorder.capture(technical_module.get_slope(window=5))
    recorder.capture(technical_module.get_slope(as_angle=True))
    recorder.capture(technical_module.get_slope(growth=True, lag=[1, 2]))


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


def test_get_rolling_correlation(recorder, technical_module):
    recorder.capture(technical_module.get_rolling_correlation())
    recorder.capture(
        technical_module.get_rolling_correlation(close_column="High", against="Low")
    )
    recorder.capture(technical_module.get_rolling_correlation(against="MSFT"))
    recorder.capture(technical_module.get_rolling_correlation(growth=True, lag=[1, 2]))


def test_get_rolling_median(recorder, technical_module):
    recorder.capture(technical_module.get_rolling_median())
    recorder.capture(technical_module.get_rolling_median(window=9))
    recorder.capture(technical_module.get_rolling_median(growth=True, lag=[1, 2]))


def test_get_rolling_percentile(recorder, technical_module):
    recorder.capture(technical_module.get_rolling_percentile())
    recorder.capture(
        technical_module.get_rolling_percentile(window=30, percentile=0.75)
    )
    recorder.capture(technical_module.get_rolling_percentile(method="linear"))
    recorder.capture(technical_module.get_rolling_percentile(growth=True, lag=[1, 2]))


def test_get_percent_rank(recorder, technical_module):
    recorder.capture(technical_module.get_percent_rank())
    recorder.capture(technical_module.get_percent_rank(window=30))
    recorder.capture(technical_module.get_percent_rank(growth=True, lag=[1, 2]))


def test_get_rolling_z_score(recorder, technical_module):
    recorder.capture(technical_module.get_rolling_z_score())
    recorder.capture(technical_module.get_rolling_z_score(window=20))
    recorder.capture(technical_module.get_rolling_z_score(growth=True, lag=[1, 2]))


def test_get_rolling_mean_absolute_deviation(recorder, technical_module):
    recorder.capture(technical_module.get_rolling_mean_absolute_deviation())
    recorder.capture(technical_module.get_rolling_mean_absolute_deviation(window=30))
    recorder.capture(
        technical_module.get_rolling_mean_absolute_deviation(growth=True, lag=[1, 2])
    )


def test_get_rolling_standard_error(recorder, technical_module):
    recorder.capture(technical_module.get_rolling_standard_error())
    recorder.capture(technical_module.get_rolling_standard_error(window=30))
    recorder.capture(
        technical_module.get_rolling_standard_error(growth=True, lag=[1, 2])
    )


def test_get_rolling_entropy(recorder, technical_module):
    recorder.capture(technical_module.get_rolling_entropy())
    recorder.capture(technical_module.get_rolling_entropy(window=20, base=np.e))
    recorder.capture(technical_module.get_rolling_entropy(growth=True, lag=[1, 2]))


def test_rolling_correlation_with_an_unknown_series_is_empty(technical_module):
    assert technical_module.get_rolling_correlation(against="UNKNOWN").empty


def test_standard_deviation_channel_is_fitted_over_the_date_range(technical_module):
    channel = technical_module.get_standard_deviation_channel(num_std_dev=2)

    # One straight line per ticker: the same change from each date to the next.
    line = channel["Linear Regression"]["AAPL"].dropna()
    assert np.allclose(line.diff().dropna(), line.diff().iloc[-1], atol=2e-4)
    assert (channel["Upper Band 2"] > channel["Lower Band 2"]).all().all()


def test_statistics_default_windows():
    import inspect  # noqa: PLC0415

    from financetoolkit.technicals.technicals_controller import (  # noqa: PLC0415
        Technicals,
    )

    # The TA-Lib defaults, and pandas-ta-classic's where TA-Lib has no such function.
    for method, window in (
        (Technicals.get_linear_regression, 14),
        (Technicals.get_correlation_trend_indicator, 20),
        (Technicals.get_forecast_oscillator, 5),
        (Technicals.get_rolling_correlation, 30),
        (Technicals.get_rolling_median, 30),
        (Technicals.get_rolling_percentile, 100),
        (Technicals.get_percent_rank, 100),
        (Technicals.get_rolling_mean_absolute_deviation, 14),
        (Technicals.get_rolling_z_score, 30),
        (Technicals.get_rolling_standard_error, 14),
        (Technicals.get_rolling_entropy, 10),
        (Technicals.get_projection_oscillator, 14),
        (Technicals.get_slope, 1),
    ):
        assert inspect.signature(method).parameters["window"].default == window


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


def test_get_t3_moving_average(recorder, technical_module):
    recorder.capture(technical_module.get_t3_moving_average())
    recorder.capture(
        technical_module.get_t3_moving_average(window=10, volume_factor=0.8)
    )
    recorder.capture(technical_module.get_t3_moving_average(growth=True))


def test_get_arnaud_legoux_moving_average(recorder, technical_module):
    recorder.capture(technical_module.get_arnaud_legoux_moving_average())
    recorder.capture(
        technical_module.get_arnaud_legoux_moving_average(
            window=20, sigma=4.0, offset=0.5
        )
    )
    recorder.capture(technical_module.get_arnaud_legoux_moving_average(growth=True))


def test_get_volume_weighted_moving_average(recorder, technical_module):
    recorder.capture(technical_module.get_volume_weighted_moving_average())
    recorder.capture(technical_module.get_volume_weighted_moving_average(window=30))
    recorder.capture(technical_module.get_volume_weighted_moving_average(growth=True))


def test_get_mcginley_dynamic(recorder, technical_module):
    recorder.capture(technical_module.get_mcginley_dynamic())
    recorder.capture(technical_module.get_mcginley_dynamic(window=10, constant=0.6))
    recorder.capture(technical_module.get_mcginley_dynamic(growth=True))


def test_get_variable_index_dynamic_average(recorder, technical_module):
    recorder.capture(technical_module.get_variable_index_dynamic_average())
    recorder.capture(
        technical_module.get_variable_index_dynamic_average(window=14, cmo_window=14)
    )
    recorder.capture(technical_module.get_variable_index_dynamic_average(growth=True))


def test_get_fractal_adaptive_moving_average(recorder, technical_module):
    recorder.capture(technical_module.get_fractal_adaptive_moving_average())
    recorder.capture(technical_module.get_fractal_adaptive_moving_average(window=10))
    recorder.capture(technical_module.get_fractal_adaptive_moving_average(growth=True))


def test_get_candlestick_patterns(recorder, technical_module):
    recorder.capture(technical_module.get_candlestick_patterns())
    recorder.capture(
        technical_module.get_candlestick_patterns(
            patterns="belt-hold, Inside Bar", signals_only=True
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
