"""Technical Controller Tests""" ""
# pylint: disable=missing-function-docstring


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


def test_get_chande_momentum_oscillator(recorder, technical_module):
    recorder.capture(technical_module.get_chande_momentum_oscillator())
    recorder.capture(technical_module.get_chande_momentum_oscillator(growth=True))
    recorder.capture(
        technical_module.get_chande_momentum_oscillator(growth=True, lag=[1, 2, 3])
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


def test_get_relative_strength_index(recorder, technical_module):
    recorder.capture(technical_module.get_relative_strength_index())
    recorder.capture(technical_module.get_relative_strength_index(growth=True))
    recorder.capture(
        technical_module.get_relative_strength_index(growth=True, lag=[1, 2, 3])
    )


def test_get_balance_of_power(recorder, technical_module):
    recorder.capture(technical_module.get_balance_of_power())
    recorder.capture(technical_module.get_balance_of_power(growth=True))
    recorder.capture(technical_module.get_balance_of_power(growth=True, lag=[1, 2, 3]))


def test_get_moving_average(recorder, technical_module):
    recorder.capture(technical_module.get_moving_average())
    recorder.capture(technical_module.get_moving_average(growth=True))
    recorder.capture(technical_module.get_moving_average(growth=True, lag=[1, 2, 3]))


def test_get_exponential_moving_average(recorder, technical_module):
    recorder.capture(technical_module.get_exponential_moving_average())
    recorder.capture(technical_module.get_exponential_moving_average(growth=True))
    recorder.capture(
        technical_module.get_exponential_moving_average(growth=True, lag=[1, 2, 3])
    )


def test_get_double_exponential_moving_average(recorder, technical_module):
    recorder.capture(technical_module.get_double_exponential_moving_average())
    recorder.capture(
        technical_module.get_double_exponential_moving_average(growth=True)
    )
    recorder.capture(
        technical_module.get_double_exponential_moving_average(
            growth=True, lag=[1, 2, 3]
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


def test_get_triangular_moving_average(recorder, technical_module):
    recorder.capture(technical_module.get_triangular_moving_average())
    recorder.capture(technical_module.get_triangular_moving_average(growth=True))
    recorder.capture(
        technical_module.get_triangular_moving_average(growth=True, lag=[1, 2, 3])
    )


def test_get_support_resistance_levels(recorder, technical_module):
    recorder.capture(technical_module.get_support_resistance_levels())
    recorder.capture(technical_module.get_support_resistance_levels(sensitivity=0.10))


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


def test_get_keltner_channels(recorder, technical_module):
    recorder.capture(technical_module.get_keltner_channels())
    recorder.capture(technical_module.get_keltner_channels(growth=True))
    recorder.capture(technical_module.get_keltner_channels(growth=True, lag=[1, 2, 3]))


def test_get_on_balance_volume(recorder, technical_module):
    recorder.capture(technical_module.get_on_balance_volume())
    recorder.capture(technical_module.get_on_balance_volume(growth=True))
    recorder.capture(technical_module.get_on_balance_volume(growth=True, lag=[1, 2, 3]))


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


def test_get_weighted_moving_average(recorder, technical_module):
    recorder.capture(technical_module.get_weighted_moving_average())
    recorder.capture(technical_module.get_weighted_moving_average(growth=True))
    recorder.capture(
        technical_module.get_weighted_moving_average(growth=True, lag=[1, 2, 3])
    )


def test_get_hull_moving_average(recorder, technical_module):
    recorder.capture(technical_module.get_hull_moving_average())
    recorder.capture(technical_module.get_hull_moving_average(growth=True))
    recorder.capture(
        technical_module.get_hull_moving_average(growth=True, lag=[1, 2, 3])
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
