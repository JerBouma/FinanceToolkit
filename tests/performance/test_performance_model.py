"""Performance Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.performance import performance_model
from financetoolkit.risk import cvar_model

# pylint: disable=missing-function-docstring

RESIDUAL_STD_TOLERANCE = 0.002


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


def test_get_rolling_alpha(recorder):
    recorder.capture(
        performance_model.get_rolling_alpha(
            asset_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
            window_size=2,
        ).round(4)
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


def test_get_probabilistic_sharpe_ratio(recorder):
    recorder.capture(
        round(
            performance_model.get_probabilistic_sharpe_ratio(
                sharpe_ratio=0.15,
                benchmark_sharpe_ratio=0.0,
                skewness=-0.3,
                kurtosis=4.0,
                n_observations=252,
            ),
            4,
        )
    )
    recorder.capture(
        performance_model.get_probabilistic_sharpe_ratio(
            sharpe_ratio=pd.Series([0.1, 0.2, -0.05]),
            benchmark_sharpe_ratio=0.0,
            skewness=pd.Series([0.1, -0.2, 0.05]),
            kurtosis=pd.Series([3.2, 3.5, 4.1]),
            n_observations=pd.Series([100, 150, 80]),
        ).round(4)
    )


def test_get_deflated_sharpe_ratio(recorder):
    recorder.capture(
        round(
            performance_model.get_deflated_sharpe_ratio(
                sharpe_ratio=0.15,
                sharpe_ratio_variance=0.0025,
                n_trials=10,
                n_observations=252,
                skewness=-0.3,
                kurtosis=4.0,
            ),
            4,
        )
    )
    # With a single trial, the Deflated Sharpe Ratio must reduce exactly to
    # the Probabilistic Sharpe Ratio against a benchmark of 0.
    recorder.capture(
        round(
            performance_model.get_deflated_sharpe_ratio(
                sharpe_ratio=0.15,
                sharpe_ratio_variance=0.0025,
                n_trials=1,
                n_observations=252,
                skewness=-0.3,
                kurtosis=4.0,
            ),
            4,
        )
    )
    recorder.capture(
        performance_model.get_deflated_sharpe_ratio(
            sharpe_ratio=pd.Series([0.1, 0.2, -0.05]),
            sharpe_ratio_variance=pd.Series([0.002, 0.003, 0.001]),
            n_trials=20,
            n_observations=pd.Series([100, 150, 80]),
            skewness=pd.Series([0.1, -0.2, 0.05]),
            kurtosis=pd.Series([3.2, 3.5, 4.1]),
        ).round(4)
    )


def test_get_sortino_ratio(recorder):
    recorder.capture(
        performance_model.get_sortino_ratio(
            excess_returns=pd.Series([0.3, 0.2, 0.1, -0.05, 0.06])
        )
    )


def test_get_rolling_sortino_ratio(recorder):
    recorder.capture(
        performance_model.get_rolling_sortino_ratio(
            excess_returns=pd.Series([0.3, 0.2, -0.1, -0.05, 0.06, -0.2, 0.1]),
            window_size=3,
        ).round(4)
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
            benchmark_standard_deviation=pd.Series([0.02, 0.03, 0.015, 0.005, 0.01]),
        )
    )


def test_get_rolling_m2_ratio(recorder):
    recorder.capture(
        performance_model.get_rolling_m2_ratio(
            asset_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            risk_free_rate=pd.Series([0.01, 0.02, 0.01, 0.0, 0.006]),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
            window_size=2,
        ).round(4)
    )


def test_get_tracking_error(recorder):
    recorder.capture(
        performance_model.get_tracking_error(
            asset_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
        )
    )


def test_get_tracking_error_dataframe(recorder):
    # Regression test: the plain (non "within period") DataFrame branch used to index
    # into the whole asset_returns DataFrame instead of the individual column, which
    # misaligned the resulting per-column Series against the (date-indexed) output
    # DataFrame and silently produced all-NaN results.
    recorder.capture(
        performance_model.get_tracking_error(
            asset_returns=pd.DataFrame(
                {
                    "AAPL": [0.3, 0.2, 0.1, 0, 0.06],
                    "MSFT": [0.28, 0.22, 0.05, 0.01, 0.04],
                }
            ),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
        )
    )


def test_get_rolling_tracking_error(recorder):
    recorder.capture(
        performance_model.get_rolling_tracking_error(
            asset_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
            window_size=2,
        ).round(4)
    )


def test_get_information_ratio(recorder):
    recorder.capture(
        performance_model.get_information_ratio(
            asset_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
        )
    )


def test_get_rolling_information_ratio(recorder):
    recorder.capture(
        performance_model.get_rolling_information_ratio(
            asset_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
            window_size=2,
        ).round(4)
    )


def test_get_compound_growth_rate(recorder):
    recorder.capture(
        performance_model.get_compound_growth_rate(
            prices=pd.Series([100, 200, 300, 400, 500]),
            periods=5,
        )
    )


def _build_within_period_data():
    """Builds a small, deterministic "within period" (period, date) Multi Index
    dataset used to test the CAPM-residual-based functions."""
    periods = ["2020", "2020", "2020", "2020", "2021", "2021", "2021", "2021"]
    dates = pd.to_datetime(
        [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-06",
            "2021-01-01",
            "2021-01-02",
            "2021-01-03",
            "2021-01-06",
        ]
    )
    index = pd.MultiIndex.from_arrays([periods, dates], names=["Period", "Date"])

    excess_returns = pd.DataFrame(
        {"AAPL": [0.02, -0.01, 0.03, 0.00, 0.01, 0.04, -0.02, 0.02]}, index=index
    )
    benchmark_excess_returns = pd.Series(
        [0.01, -0.02, 0.02, 0.01, 0.005, 0.03, -0.01, 0.015], index=index, name="Mkt"
    )
    beta = pd.DataFrame({"AAPL": [1.2, 0.9]}, index=["2020", "2021"])

    return excess_returns, benchmark_excess_returns, beta


def test_get_capm_residuals(recorder):
    excess_returns, benchmark_excess_returns, beta = _build_within_period_data()

    residuals = performance_model.get_capm_residuals(
        excess_returns, beta, benchmark_excess_returns
    )

    # The residual for each row must equal Excess Return - Beta * Benchmark Excess
    # Return, using that row's *own period's* Beta (not a single global Beta).
    for sub_period in ["2020", "2021"]:
        expected = (
            excess_returns.loc[sub_period, "AAPL"]
            - beta.loc[sub_period, "AAPL"] * benchmark_excess_returns.loc[sub_period]
        )
        pd.testing.assert_series_equal(
            residuals.loc[sub_period, "AAPL"], expected, check_names=False
        )

    recorder.capture(residuals.round(6))


def test_get_appraisal_ratio(recorder):
    excess_returns, benchmark_excess_returns, beta = _build_within_period_data()
    capm_residuals = performance_model.get_capm_residuals(
        excess_returns, beta, benchmark_excess_returns
    )
    jensens_alpha = pd.Series([0.02, -0.01], index=["2020", "2021"], name="AAPL")

    appraisal_ratio = performance_model.get_appraisal_ratio(
        jensens_alpha, capm_residuals["AAPL"]
    )

    expected = jensens_alpha / capm_residuals["AAPL"].groupby(level=0).std()
    pd.testing.assert_series_equal(appraisal_ratio, expected, check_names=False)

    recorder.capture(appraisal_ratio.round(6))


def test_get_fama_decomposition(recorder):
    periods = ["2020", "2021", "2022"]
    asset_returns = pd.DataFrame({"AAPL": [0.15, -0.03, 0.20]}, index=periods)
    risk_free_rate = pd.Series([0.01, 0.01, 0.01], index=periods)
    beta = pd.DataFrame({"AAPL": [1.2, 1.2, 1.2]}, index=periods)
    benchmark_returns = pd.Series([0.08, -0.05, 0.12], index=periods)
    asset_standard_deviation = pd.DataFrame({"AAPL": [0.25, 0.22, 0.28]}, index=periods)
    benchmark_standard_deviation = pd.Series([0.18, 0.18, 0.18], index=periods)

    selectivity, diversification = performance_model.get_fama_decomposition(
        asset_returns,
        risk_free_rate,
        beta,
        benchmark_returns,
        asset_standard_deviation,
        benchmark_standard_deviation,
    )

    # Selectivity + Diversification must reconstruct Jensen's Alpha exactly.
    jensens_alpha = performance_model.get_jensens_alpha(
        asset_returns, risk_free_rate, beta, benchmark_returns
    )
    pd.testing.assert_frame_equal(
        (selectivity + diversification).astype(float), jensens_alpha.astype(float)
    )

    recorder.capture(selectivity.round(6))
    recorder.capture(diversification.round(6))


def test_get_fama_decomposition_type_error():
    with pytest.raises(TypeError):
        performance_model.get_fama_decomposition(
            pd.DataFrame({"A": [0.1]}),
            0.01,
            "not a beta",
            0.05,
            pd.DataFrame({"A": [0.2]}),
            0.15,
        )


def test_get_adjusted_sharpe_ratio(recorder):
    # A Sharpe Ratio of 0 should always produce an Adjusted Sharpe Ratio of 0,
    # regardless of skewness or kurtosis.
    assert performance_model.get_adjusted_sharpe_ratio(0.0, -1.5, 6.0) == 0.0

    # Normal-distribution-like skew (0) and raw kurtosis (3) should leave the
    # Sharpe Ratio (almost) unadjusted, since the correction terms vanish.
    unadjusted = performance_model.get_adjusted_sharpe_ratio(0.2, 0.0, 3.0)
    assert unadjusted == pytest.approx(0.2, abs=1e-9)

    recorder.capture(
        round(
            performance_model.get_adjusted_sharpe_ratio(
                sharpe_ratio=0.15, skewness=-0.3, kurtosis=4.5
            ),
            6,
        )
    )
    recorder.capture(
        performance_model.get_adjusted_sharpe_ratio(
            sharpe_ratio=pd.Series([0.1, 0.2, -0.05]),
            skewness=pd.Series([0.1, -0.2, 0.05]),
            kurtosis=pd.Series([3.2, 3.5, 4.1]),
        ).round(6)
    )


def test_get_starr_ratio(recorder):
    # excess_returns represents already-aggregated (e.g. one value per period) excess
    # returns, divided elementwise by the (period-level) CVaR of the raw returns.
    periods = ["2020", "2021", "2022"]
    excess_returns = pd.Series([0.05, -0.02, 0.08], index=periods)
    returns = pd.Series(
        [0.05, -0.10, 0.02, 0.03, -0.08, 0.07, -0.02, 0.01, 0.04, -0.15]
    )

    starr_ratio = performance_model.get_starr_ratio(excess_returns, returns, alpha=0.2)

    expected = excess_returns / abs(cvar_model.get_cvar_historic(returns, 0.2))
    pd.testing.assert_series_equal(starr_ratio, expected)

    recorder.capture(starr_ratio.round(6))


def test_get_starr_ratio_series_denominator():
    # When both excess_returns and returns are plain (non-period) Series, the CVaR
    # denominator collapses to a scalar and the result stays Series-shaped.
    returns = pd.Series(
        [0.05, -0.10, 0.02, 0.03, -0.08, 0.07, -0.02, 0.01, 0.04, -0.15]
    )
    excess_returns = returns - 0.001

    starr_ratio = performance_model.get_starr_ratio(excess_returns, returns, alpha=0.2)

    expected = excess_returns / abs(cvar_model.get_cvar_historic(returns, 0.2))
    pd.testing.assert_series_equal(starr_ratio, expected)


def test_get_rachev_ratio(recorder):
    returns = pd.Series(
        [0.05, -0.10, 0.02, 0.03, -0.08, 0.07, -0.02, 0.01, 0.04, -0.15]
    )

    rachev_ratio = performance_model.get_rachev_ratio(returns, alpha=0.2)

    right_tail = -cvar_model.get_cvar_historic(-returns, 0.2)
    left_tail = -cvar_model.get_cvar_historic(returns, 0.2)
    assert rachev_ratio == pytest.approx(right_tail / left_tail)

    recorder.capture(round(rachev_ratio, 6))


def test_get_treynor_mazuy_model(recorder):
    np.random.seed(42)
    n = 100
    benchmark_excess_returns = pd.Series(np.random.normal(0.0003, 0.01, n))
    excess_returns = (
        0.0001
        + 1.1 * benchmark_excess_returns
        + 8.0 * benchmark_excess_returns**2
        + np.random.normal(0, 0.001, n)
    )

    result, residuals = performance_model.get_treynor_mazuy_model(
        excess_returns, benchmark_excess_returns
    )

    # Cross-check against an independent Numpy OLS fit of the same regression.
    design_matrix = np.column_stack(
        [
            np.ones(n),
            benchmark_excess_returns.to_numpy(),
            benchmark_excess_returns.to_numpy() ** 2,
        ]
    )
    coefficients, *_ = np.linalg.lstsq(
        design_matrix, excess_returns.to_numpy(), rcond=None
    )
    assert result["Alpha"] == pytest.approx(coefficients[0], abs=1e-6)
    assert result["Beta"] == pytest.approx(coefficients[1], abs=1e-6)
    assert result["Gamma"] == pytest.approx(coefficients[2], abs=1e-6)
    assert residuals.std() < RESIDUAL_STD_TOLERANCE

    recorder.capture(pd.Series({key: round(value, 4) for key, value in result.items()}))


def test_get_treynor_mazuy_model_type_error():
    with pytest.raises(TypeError):
        performance_model.get_treynor_mazuy_model(
            pd.Series([0.1, 0.2]).to_numpy(), pd.Series([0.1, 0.2])
        )


def test_get_treynor_mazuy_model_insufficient_observations():
    result, residuals = performance_model.get_treynor_mazuy_model(
        pd.Series([0.01, 0.02]), pd.Series([0.01, 0.02])
    )
    assert np.isnan(result["Alpha"])
    assert residuals.isna().all()


def test_get_henriksson_merton_model(recorder):
    np.random.seed(24)
    n = 100
    benchmark_excess_returns = pd.Series(np.random.normal(0.0003, 0.01, n))
    up_market = benchmark_excess_returns.clip(lower=0)
    excess_returns = (
        0.0001
        + 0.9 * benchmark_excess_returns
        + 0.6 * up_market
        + np.random.normal(0, 0.001, n)
    )

    result, residuals = performance_model.get_henriksson_merton_model(
        excess_returns, benchmark_excess_returns
    )

    design_matrix = np.column_stack(
        [np.ones(n), benchmark_excess_returns.to_numpy(), up_market.to_numpy()]
    )
    coefficients, *_ = np.linalg.lstsq(
        design_matrix, excess_returns.to_numpy(), rcond=None
    )
    assert result["Alpha"] == pytest.approx(coefficients[0], abs=1e-6)
    assert result["Beta"] == pytest.approx(coefficients[1], abs=1e-6)
    assert result["Up Market Beta"] == pytest.approx(coefficients[2], abs=1e-6)
    assert residuals.std() < RESIDUAL_STD_TOLERANCE

    recorder.capture(pd.Series({key: round(value, 4) for key, value in result.items()}))


def test_get_henriksson_merton_model_type_error():
    with pytest.raises(TypeError):
        performance_model.get_henriksson_merton_model(
            pd.Series([0.1, 0.2]).to_numpy(), pd.Series([0.1, 0.2])
        )
