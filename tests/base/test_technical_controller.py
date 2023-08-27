"""Technical Controller Tests""" ""
import pandas as pd

from financetoolkit import Toolkit

historical = pd.read_pickle("tests/base/datasets/historical_dataset.pickle")

print(historical)

toolkit = Toolkit(tickers=["AAPL", "MSFT"], historical=historical)

technical_module = toolkit.technical

# pylint: disable=missing-function-docstring


def test_collect_all_indicators(recorder):
    recorder.capture(technical_module.collect_all_indicators())
    recorder.capture(technical_module.collect_all_indicators(growth=True))
    recorder.capture(
        technical_module.collect_all_indicators(growth=True, lag=[1, 2, 3])
    )


def test_collect_breadth_indicators(recorder):
    recorder.capture(technical_module.collect_breadth_indicators())
    recorder.capture(technical_module.collect_breadth_indicators(growth=True))
    recorder.capture(
        technical_module.collect_breadth_indicators(growth=True, lag=[1, 2, 3])
    )


def test_collect_momentum_indicators(recorder):
    recorder.capture(technical_module.collect_momentum_indicators())
    recorder.capture(technical_module.collect_momentum_indicators(growth=True))
    recorder.capture(
        technical_module.collect_momentum_indicators(growth=True, lag=[1, 2, 3])
    )


def test_collect_overlap_indicators(recorder):
    recorder.capture(technical_module.collect_overlap_indicators())
    recorder.capture(technical_module.collect_overlap_indicators(growth=True))
    recorder.capture(
        technical_module.collect_overlap_indicators(growth=True, lag=[1, 2, 3])
    )


def test_collect_volatility_indicators(recorder):
    recorder.capture(technical_module.collect_volatility_indicators())
    recorder.capture(technical_module.collect_volatility_indicators(growth=True))
    recorder.capture(
        technical_module.collect_volatility_indicators(growth=True, lag=[1, 2, 3])
    )


def test_get_mcclellan_oscillator(recorder):
    recorder.capture(technical_module.get_mcclellan_oscillator())
    recorder.capture(technical_module.get_mcclellan_oscillator(growth=True))
    recorder.capture(
        technical_module.get_mcclellan_oscillator(growth=True, lag=[1, 2, 3])
    )


def test_get_money_flow_index(recorder):
    recorder.capture(technical_module.get_money_flow_index())


def test_get_williams_percent_r(recorder):
    recorder.capture(technical_module.get_williams_percent_r())


def test_get_aroon_indicator(recorder):
    result = technical_module.get_aroon_indicator()

    recorder.capture(result[0])
    recorder.capture(result[1])


def test_get_commodity_channel_index(recorder):
    recorder.capture(technical_module.get_commodity_channel_index())


def test_get_relative_vigor_index(recorder):
    recorder.capture(technical_module.get_relative_vigor_index())


def test_get_force_index(recorder):
    recorder.capture(technical_module.get_force_index())


def test_get_ultimate_oscillator(recorder):
    recorder.capture(technical_module.get_ultimate_oscillator())


def test_get_percentage_price_oscillator(recorder):
    recorder.capture(technical_module.get_percentage_price_oscillator())


def test_get_detrended_price_oscillator(recorder):
    recorder.capture(technical_module.get_detrended_price_oscillator())


def test_get_average_directional_index(recorder):
    recorder.capture(technical_module.get_average_directional_index())


def test_get_chande_momentum_oscillator(recorder):
    recorder.capture(technical_module.get_chande_momentum_oscillator())


def test_get_ichimoku_cloud(recorder):
    result = technical_module.get_ichimoku_cloud()

    recorder.capture(result[0])
    recorder.capture(result[1])
    recorder.capture(result[2])
    recorder.capture(result[3])


def test_get_stochastic_oscillator(recorder):
    result = technical_module.get_stochastic_oscillator()

    recorder.capture(result[0])
    recorder.capture(result[1])


def test_get_moving_average_convergence_divergence(recorder):
    result = technical_module.get_moving_average_convergence_divergence()

    recorder.capture(result[0])
    recorder.capture(result[1])


def test_get_relative_strength_index(recorder):
    recorder.capture(technical_module.get_relative_strength_index())


def test_get_balance_of_power(recorder):
    recorder.capture(technical_module.get_balance_of_power())


def test_get_moving_average(recorder):
    recorder.capture(technical_module.get_moving_average())


def test_get_exponential_moving_average(recorder):
    recorder.capture(technical_module.get_exponential_moving_average())


def test_get_double_exponential_moving_average(recorder):
    recorder.capture(technical_module.get_double_exponential_moving_average())


def test_get_trix(recorder):
    recorder.capture(technical_module.get_trix())


def test_get_bollinger_bands(recorder):
    result = technical_module.get_bollinger_bands()

    recorder.capture(result[0])
    recorder.capture(result[1])
    recorder.capture(result[2])


def test_get_triangular_moving_average(recorder):
    recorder.capture(technical_module.get_triangular_moving_average())


def test_get_true_range(recorder):
    recorder.capture(technical_module.get_true_range())


def test_get_average_true_range(recorder):
    recorder.capture(technical_module.get_average_true_range())


def test_get_keltner_channels(recorder):
    result = technical_module.get_keltner_channels()

    recorder.capture(result[0])
    recorder.capture(result[1])
    recorder.capture(result[2])


def test_get_on_balance_volume(recorder):
    recorder.capture(technical_module.get_on_balance_volume())


def test_get_accumulation_distribution_line(recorder):
    recorder.capture(technical_module.get_accumulation_distribution_line())


def test_get_chaikin_oscillator(recorder):
    recorder.capture(technical_module.get_chaikin_oscillator())
