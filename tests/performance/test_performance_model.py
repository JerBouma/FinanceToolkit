"""Performance Model Tests"""

import pandas as pd

from financetoolkit.performance import performance_model

# pylint: disable=missing-function-docstring


def test_get_covariance(recorder):
    recorder.capture(
        performance_model.get_covariance(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
        )
    )


def test_get_beta(recorder):
    recorder.capture(
        performance_model.get_beta(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
        )
    )


def test_get_rolling_beta(recorder):
    recorder.capture(
        performance_model.get_rolling_beta(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
            window_size=2,
        )
    )


def test_get_capital_asset_pricing_model(recorder):
    recorder.capture_list(
        performance_model.get_capital_asset_pricing_model(
            risk_free_rate=pd.Series([0.01, 0.02, 0.01, 0.0, 0.006]),
            beta=pd.Series([1.0, 1.19, 1.5, 1, 1.03]),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
        )
    )


def test_obtain_fama_and_french_dataset(recorder):
    dataset = performance_model.obtain_fama_and_french_dataset()

    # This is done given that the dataset can change over time
    # it is only important to check if the data remains the same
    # for a short period of time
    recorder.capture(dataset.round(0).iloc[:100])


def test_get_factor_asset_correlations(recorder):
    recorder.capture(
        performance_model.get_factor_asset_correlations(
            factors=pd.DataFrame(
                [[0.05, 0.03], [0.06, 0.02]], columns=["Mkt-RF", "SMB"]
            ),
            excess_return=pd.Series([0.3, 0.2]),
        )
    )


def test_get_fama_and_french_model_multi(recorder):
    regression_results, residuals, _ = (
        performance_model.get_fama_and_french_model_multi(
            excess_returns=pd.Series([0.3, 0.2]),
            factor_dataset=pd.DataFrame(
                [[0.05, 0.03], [0.06, 0.02]], columns=["Mkt-RF", "SMB"]
            ),
        )
    )

    for result_values in regression_results:
        regression_results[result_values] = round(regression_results[result_values], 2)

    residuals = residuals.round(0)

    recorder.capture(pd.DataFrame((regression_results, residuals)))


def test_get_fama_and_french_model_single(recorder):
    result = performance_model.get_fama_and_french_model_single(
        excess_returns=pd.Series([0.3, 0.2]), factor=pd.Series([0.06, 0.02])
    )

    recorder.capture(pd.DataFrame(result))


def test_get_alpha(recorder):
    recorder.capture(
        performance_model.get_alpha(
            asset_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
        )
    )


def test_get_jensens_alpha(recorder):
    recorder.capture(
        performance_model.get_jensens_alpha(
            asset_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            risk_free_rate=pd.Series([0.01, 0.02, 0.01, 0.0, 0.006]),
            beta=pd.Series([1.0, 1.19, 1.5, 1, 1.03]),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
        )
    )


def test_get_treynor_ratio(recorder):
    recorder.capture(
        performance_model.get_treynor_ratio(
            asset_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            risk_free_rate=pd.Series([0.01, 0.02, 0.01, 0.0, 0.006]),
            beta=pd.Series([1.0, 1.19, 1.5, 1, 1.03]),
        )
    )


def test_get_sharpe_ratio(recorder):
    recorder.capture(
        round(
            performance_model.get_sharpe_ratio(
                excess_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06])
            ),
            4,
        )
    )


def test_get_rolling_sharpe_ratio(recorder):
    recorder.capture(
        performance_model.get_rolling_sharpe_ratio(
            excess_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), window_size=2
        ).round(4)
    )


def test_get_sortino_ratio(recorder):
    recorder.capture(
        performance_model.get_sortino_ratio(
            excess_returns=pd.Series([0.3, 0.2, 0.1, -0.05, 0.06])
        )
    )


def test_get_ulcer_performance_index(recorder):
    recorder.capture(
        performance_model.get_ulcer_performance_index(
            excess_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            ulcer_index=pd.Series([0.01, 0.02, 0.01, 0.0, 0.006]),
        )
    )


def test_get_m2_ratio(recorder):
    recorder.capture(
        performance_model.get_m2_ratio(
            asset_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            risk_free_rate=pd.Series([0.01, 0.02, 0.01, 0.0, 0.006]),
            asset_standard_deviation=pd.Series([0.01, 0.02, 0.01, 0.0, 0.006]),
        )
    )


def test_get_tracking_error(recorder):
    recorder.capture(
        performance_model.get_tracking_error(
            asset_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
        )
    )


def test_get_information_ratio(recorder):
    recorder.capture(
        performance_model.get_information_ratio(
            asset_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
        )
    )


def test_get_compound_growth_rate(recorder):
    recorder.capture(
        performance_model.get_compound_growth_rate(
            prices=pd.Series([100, 200, 300, 400, 500]),
            periods=5,
        )
    )
