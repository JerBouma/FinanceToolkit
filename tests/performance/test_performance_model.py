"""Performance Model Tests"""

import io
import zipfile

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


def test_obtain_fama_and_french_dataset(monkeypatch):
    text = (
        "This file was created by CMPT_ME_BEME_OP_INV_RETS_DAILY using the 202609 CRSP database.\n"
        "The Tbill return is the simple daily rate.\n\n"
        ",Mkt-RF,SMB,HML,RMW,CMA,RF\n"
        "19630701,   -0.67,    0.02,   -0.35,    0.03,    0.13,    0.012\n"
        "19630702,    0.79,   -0.28,    0.28,   -0.08,   -0.21,    0.012\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("F-F_Research_Data_5_Factors_2x3_daily.csv", text)
    requested = []

    def fake_get_request(url, timeout):  # noqa: ARG001
        requested.append(url)
        return type("Response", (), {"content": buffer.getvalue()})()

    monkeypatch.setattr(performance_model, "get_request", fake_get_request)
    monkeypatch.setattr(performance_model, "_factor_files", {})

    dataset = performance_model.obtain_fama_and_french_dataset()

    # The United States five factor file, read from percent into decimals by day.
    assert requested == [performance_model.factor_url("United States", "five")]
    assert list(dataset.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert [str(day) for day in dataset.index] == ["1963-07-01", "1963-07-02"]
    assert dataset.loc["1963-07-01", "Mkt-RF"] == pytest.approx(-0.0067)
    assert dataset.loc["1963-07-02", "RF"] == pytest.approx(0.00012)


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
    # Three non-collinear points, so the regression keeps a residual degree of
    # freedom. A perfect two-point fit leaves the p-value and standard error
    # undefined, and scipy reports that as 0.0 before 1.18 and NaN from 1.18 on.
    result = performance_model.get_fama_and_french_model_single(
        excess_returns=pd.Series([0.3, 0.2, 0.15]),
        factor=pd.Series([0.06, 0.02, 0.03]),
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
    # A single trial must reduce exactly to the Probabilistic Sharpe Ratio at 0.
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
    # Regression test: the plain DataFrame branch used to index the whole frame.
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

    # Each row's residual must use its own period's Beta, not one global Beta.
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

    # The residual std is pointwise (daily), so it is rescaled to the period frequency
    # of jensens_alpha by multiplying with the square root of the number of
    # observations in that period, matching how volatility is annualized elsewhere.
    residual_std = capm_residuals["AAPL"].groupby(level=0).std()
    observations_per_period = capm_residuals["AAPL"].groupby(level=0).size()
    expected = jensens_alpha / (residual_std * np.sqrt(observations_per_period))
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
    # A Sharpe Ratio of 0 stays 0 whatever the skewness or kurtosis.
    assert performance_model.get_adjusted_sharpe_ratio(0.0, -1.5, 6.0) == 0.0

    # Skew 0 and raw kurtosis 3 make the correction terms vanish.
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
    # Aggregated excess returns divided elementwise by the period-level CVaR.
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
    # Plain Series on both sides collapse the CVaR denominator to a scalar.
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


def test_get_beta_of_an_asset_with_a_shorter_history():
    """The benchmark variance is taken over the dates the asset has returns, too."""
    rng = np.random.default_rng(0)
    # The benchmark is three times as volatile before the asset is listed.
    benchmark = pd.Series(np.r_[rng.normal(0, 0.03, 250), rng.normal(0, 0.01, 250)])
    asset = benchmark + rng.normal(0, 0.002, 500)
    asset[:250] = np.nan
    returns = pd.DataFrame({"Listed Later": asset, "Half": 0.5 * benchmark})

    beta = performance_model.get_beta(returns, benchmark)

    assert beta["Listed Later"] == pytest.approx(1, abs=0.01)
    assert beta["Half"] == pytest.approx(0.5)
    assert performance_model.get_beta(asset, benchmark) == pytest.approx(
        beta["Listed Later"]
    )

    within_period = performance_model.get_beta(
        returns.set_axis(
            pd.MultiIndex.from_arrays([np.repeat([1, 2], 250), returns.index])
        ),
        benchmark.set_axis(
            pd.MultiIndex.from_arrays([np.repeat([1, 2], 250), returns.index])
        ),
    )

    assert np.isnan(within_period.loc[1, "Listed Later"])
    assert within_period.loc[2, "Listed Later"] == pytest.approx(1, abs=0.01)


def test_a_factor_file_is_downloaded_once_per_session(monkeypatch):
    """Every factor model call reads the file, which is downloaded only once."""
    downloads = []

    def fake_get_request(url, timeout):  # noqa: ARG001
        downloads.append(url)
        return type("Response", (), {"content": b"zip"})()

    monkeypatch.setattr(performance_model, "get_request", fake_get_request)
    monkeypatch.setattr(performance_model, "_factor_files", {})

    assert performance_model.download_factor_file("https://example.com/f.zip") == b"zip"
    assert performance_model.download_factor_file("https://example.com/f.zip") == b"zip"
    assert downloads == ["https://example.com/f.zip"]


def test_get_compound_annual_growth_rate_uses_calendar_time():
    # A price that doubles over two calendar years grows by sqrt(2) - 1 a year, whether it
    # is observed every day (a cryptocurrency) or on weekdays only (a stock).
    every_day = pd.date_range("2020-01-01", "2022-01-01", freq="D")
    weekdays = every_day[every_day.dayofweek < 5]  # noqa: PLR2004
    years = (every_day[-1] - every_day[0]).days / 365.25

    crypto = pd.Series(np.linspace(100, 200, len(every_day)), index=every_day)
    stock = pd.Series(np.linspace(50, 100, len(weekdays)), index=weekdays)

    # The union calendar leaves the stock without a price on the weekends, the last day
    # (a Saturday) included, so it is measured from its own first to its own last close.
    prices = pd.concat({"Crypto": crypto, "Stock": stock}, axis=1)
    prices.index = prices.index.to_period("D")

    cagr = performance_model.get_compound_annual_growth_rate(prices)

    assert cagr["Crypto"] == pytest.approx(2 ** (1 / years) - 1)
    stock_years = (weekdays[-1] - weekdays[0]).days / 365.25
    assert cagr["Stock"] == pytest.approx(2 ** (1 / stock_years) - 1)
    assert cagr["Stock"] == pytest.approx(0.4142, abs=1e-3)

    # A single price spans no time, so it has no growth rate.
    assert np.isnan(performance_model.get_compound_annual_growth_rate(crypto.iloc[:1]))


def test_get_rolling_beta_masks_the_benchmark_per_asset():
    # Asset B has no returns for the first three dates, so its benchmark variance must
    # cover only the dates on which B has a return, as the covariance does.
    benchmark = pd.Series([0.01, -0.02, 0.03, 0.015, -0.01, 0.02, -0.005])
    returns = pd.DataFrame(
        {
            "A": [0.02, -0.01, 0.025, 0.01, -0.02, 0.03, 0.0],
            "B": [np.nan, np.nan, np.nan, 0.02, -0.015, 0.01, -0.01],
        }
    )

    rolling_beta = performance_model.get_rolling_beta(returns, benchmark, 4)

    # The first full window of B (dates 3-6) has four returns.
    assert np.isnan(rolling_beta["B"].iloc[5])
    window = slice(3, 7)
    expected = (
        returns["B"].iloc[window].cov(benchmark.iloc[window])
        / benchmark.iloc[window].var()
    )
    assert rolling_beta["B"].iloc[6] == pytest.approx(expected)
    expected_a = (
        returns["A"].iloc[window].cov(benchmark.iloc[window])
        / benchmark.iloc[window].var()
    )
    assert rolling_beta["A"].iloc[6] == pytest.approx(expected_a)


def test_get_sortino_ratio_needs_two_observations():
    index = pd.MultiIndex.from_arrays(
        [
            pd.PeriodIndex(["2019", "2020", "2020", "2020"], freq="Y"),
            pd.PeriodIndex(
                ["2019-12-31", "2020-01-02", "2020-01-03", "2020-01-06"], freq="D"
            ),
        ]
    )
    excess_returns = pd.DataFrame({"A": [0.01, 0.02, -0.01, 0.005]}, index=index)

    sortino_ratio = performance_model.get_sortino_ratio(excess_returns)

    # A single observation would report only its sign (+1 or -1).
    assert np.isnan(sortino_ratio.loc[pd.Period("2019", "Y"), "A"])
    assert sortino_ratio.loc[pd.Period("2020", "Y"), "A"] == pytest.approx(
        np.mean([0.02, -0.01, 0.005]) / np.sqrt(0.01**2 / 3)
    )
    assert np.isnan(performance_model.get_sortino_ratio(pd.Series([-0.01])))
    assert (
        performance_model.get_rolling_sortino_ratio(excess_returns["A"], 1).isna().all()
    )


def test_get_tracking_error_and_information_ratio_per_column():
    returns = pd.DataFrame(
        {"A": [0.01, 0.02, -0.01, 0.03], "B": [0.0, -0.01, 0.02, 0.01]}
    )
    benchmark = pd.Series([0.005, 0.01, -0.005, 0.02])
    active = returns.sub(benchmark, axis=0)

    tracking_error = performance_model.get_tracking_error(returns, benchmark)
    information_ratio = performance_model.get_information_ratio(returns, benchmark)

    assert isinstance(tracking_error, pd.Series)
    assert isinstance(information_ratio, pd.Series)
    pd.testing.assert_series_equal(tracking_error, active.std())
    pd.testing.assert_series_equal(information_ratio, active.mean() / active.std())


def test_capture_ratios_use_geometric_averages():
    benchmark = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01], name="Benchmark")
    returns = pd.DataFrame({"A": [0.03, -0.02, 0.01, -0.01, 0.02]})

    def geometric(values):
        values = np.asarray(values)
        return np.prod(1 + values) ** (1 / len(values)) - 1

    up = performance_model.get_upside_capture_ratio(returns, benchmark)
    down = performance_model.get_downside_capture_ratio(returns, benchmark)

    assert up["A"] == pytest.approx(
        geometric([0.03, 0.01, 0.02]) / geometric([0.02, 0.03, 0.01])
    )
    assert down["A"] == pytest.approx(
        geometric([-0.02, -0.01]) / geometric([-0.01, -0.02])
    )
    assert performance_model.get_upside_capture_ratio(
        returns["A"], benchmark
    ) == pytest.approx(up["A"])


def test_get_returns_keeps_periods_without_data_missing():
    dates = pd.period_range("2023-01-02", "2023-12-29", freq="D")
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(
        {"A": rng.normal(0, 0.01, len(dates)), "B": rng.normal(0, 0.01, len(dates))},
        index=dates,
    )
    # B lists in the second quarter.
    returns.loc[:"2023-03-31", "B"] = np.nan

    quarterly = performance_model.get_returns(returns, "quarterly")
    cumulative = performance_model.get_returns(returns, "quarterly", cumulative=True)

    # A quarter without any return is missing, not a return of exactly 0.
    assert np.isnan(quarterly.loc[pd.Period("2023Q1", "Q"), "B"])
    expected = np.prod(1 + returns.loc["2023-04-01":"2023-06-30", "B"]) - 1
    assert quarterly.loc[pd.Period("2023Q2", "Q"), "B"] == pytest.approx(expected)
    assert np.isnan(cumulative.loc[pd.Period("2023Q1", "Q"), "B"])
    assert cumulative.loc[pd.Period("2023Q2", "Q"), "B"] == pytest.approx(1 + expected)
    assert quarterly["A"].notna().all()


def test_get_win_rate_leaves_out_days_without_both_returns():
    asset = pd.DataFrame(
        {
            "A": [0.02, np.nan, 0.01, -0.01, 0.03],
            "B": [np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    benchmark = pd.Series([0.01, 0.01, 0.02, np.nan, 0.01], name="Benchmark")

    win_rate = performance_model.get_win_rate(asset, benchmark)

    # Days 2 (no asset return) and 4 (no benchmark return) count neither as win nor loss.
    assert win_rate["A"] == pytest.approx(2 / 3)
    # Without a single comparable day there is no win rate, not a win rate of 0.
    assert np.isnan(win_rate["B"])
    assert performance_model.get_win_rate(asset["A"], benchmark) == pytest.approx(2 / 3)

    # Within periods, each period counts only its own comparable days.
    index = pd.MultiIndex.from_product([["2023", "2024"], range(5)])
    stacked = pd.concat([asset, asset]).set_index(index)
    stacked_benchmark = pd.concat([benchmark, benchmark]).set_axis(index)
    per_period = performance_model.get_win_rate(stacked, stacked_benchmark)
    assert per_period["A"].tolist() == pytest.approx([2 / 3, 2 / 3])
    assert per_period["B"].isna().all()


def test_rolling_omega_ratio_matches_every_window_and_is_infinite_without_losses():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(200, 2)), columns=["A", "B"])
    returns.iloc[50, 1] = np.nan

    def window_omega(window):
        gains = window[window > 0].sum()
        losses = -window[window < 0].sum()
        return gains / losses

    pd.testing.assert_frame_equal(
        performance_model.get_rolling_omega_ratio(returns, 20),
        returns.rolling(20).apply(window_omega, raw=True),
        rtol=1e-12,
    )
    assert (
        performance_model.get_rolling_omega_ratio(pd.Series([0.01, 0.02, 0.0, 0.0]), 2)
        .iloc[1:]
        .isna()
        .all()
    )


def test_omega_ratio_without_losses_is_nan():
    gains_only = pd.Series([0.01, 0.02, 0.005])

    assert np.isnan(performance_model.get_omega_ratio(gains_only))
    assert np.isnan(performance_model.get_omega_ratio(pd.Series([0.0, 0.0])))
    assert np.isnan(performance_model.get_rolling_omega_ratio(gains_only, 2).iloc[-1])


def test_burke_drawdown_measure_of_a_series_without_returns_is_nan():
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.01], "B": [np.nan] * 3})

    measure = performance_model.get_burke_drawdown_measure(returns)

    assert measure["A"] > 0
    assert np.isnan(measure["B"])


def test_beta_implied_return_leaves_no_residual_for_a_leveraged_market():
    days = pd.period_range("2024-01-01", periods=300, freq="D")
    rng = np.random.default_rng(1)
    market = pd.Series(rng.normal(0, 0.01, len(days)), index=days)
    # The asset is twice the market, the other asset half of it plus its own noise.
    returns = pd.DataFrame(
        {
            "DOUBLE": 2 * market,
            "OWN": 0.5 * market + rng.normal(0, 0.01, len(days)),
        }
    )

    result = performance_model.get_beta_implied_return(
        returns, market, "daily", beta_window=50
    )

    explained = result.iloc[50:]
    assert explained["Beta"]["DOUBLE"].to_numpy() == pytest.approx(2.0)
    assert explained["Residual"]["DOUBLE"].to_numpy() == pytest.approx(0.0, abs=1e-12)
    # Without a full window before it a day has no beta and so no implied return.
    assert result["Implied Return"].iloc[:50].isna().all().all()
    assert (
        explained["Residual"]
        - (explained["Actual Return"] - explained["Implied Return"])
    ).abs().max().max() == pytest.approx(0.0)


def test_beta_implied_return_uses_the_beta_known_before_the_period():
    days = pd.period_range("2024-01-01", periods=120, freq="D")
    market = pd.Series(np.tile([0.01, -0.01], 60), index=days)
    # Beta 1 for the first 60 days, then 3: the shift in beta must not reach back.
    asset = pd.Series(np.r_[market.iloc[:60], 3 * market.iloc[60:]], index=days)
    returns = asset.to_frame("A")

    daily = performance_model.get_beta_implied_return(
        returns, market, "daily", beta_window=20
    )
    # The first day of the new regime is explained by the old beta.
    assert daily["Beta"]["A"].iloc[60] == pytest.approx(1.0)

    monthly = performance_model.get_beta_implied_return(
        returns, market, "monthly", beta_window=20
    )
    # March starts on day 60 (a leap year), so its beta is the one of the window before it.
    assert monthly.loc["2024-03", ("Beta", "A")] == pytest.approx(1.0)
    assert monthly.loc["2024-04", ("Beta", "A")] == pytest.approx(3.0)
    assert monthly["Actual Return"]["A"].to_numpy() == pytest.approx(
        performance_model.get_returns(returns, "monthly")["A"].to_numpy()
    )


# A small history with four wins, three losses and a zero return (counted as neither).
TRADES = pd.Series([0.02, 0.01, -0.01, 0.0, 0.03, -0.02, -0.01, 0.02])


def test_return_statistics_match_a_hand_calculation():
    statistics = performance_model.get_return_statistics(TRADES)

    assert statistics["Win Rate"] == pytest.approx(4 / 7)
    assert statistics["Average Win"] == pytest.approx(0.02)
    assert statistics["Average Loss"] == pytest.approx(-0.04 / 3)
    assert statistics["Payoff Ratio"] == pytest.approx(1.5)
    assert statistics["Profit Factor"] == pytest.approx(2.0)
    assert statistics["Expectancy"] == pytest.approx(0.04 / 7)
    assert statistics["Max Consecutive Wins"] == 2
    assert statistics["Max Consecutive Losses"] == 2

    # From the peak after the fifth return to the trough after the seventh.
    wealth = (1 + TRADES).cumprod()
    drawdown = wealth.iloc[6] / wealth.iloc[4] - 1
    assert statistics["Recovery Factor"] == pytest.approx(
        (wealth.iloc[-1] - 1) / abs(drawdown)
    )


def test_a_history_without_losses_has_no_profit_factor():
    statistics = performance_model.get_return_statistics(pd.Series([0.01, 0.02, 0.0]))

    assert np.isnan(statistics["Profit Factor"])
    assert np.isnan(statistics["Payoff Ratio"])
    assert np.isnan(statistics["Recovery Factor"])
    assert statistics["Max Consecutive Losses"] == 0


def test_a_losing_history_keeps_the_sign_of_its_recovery_factor():
    statistics = performance_model.get_return_statistics(pd.Series([0.01, -0.05, 0.01]))

    assert statistics["Recovery Factor"] < 0


def test_return_statistics_per_period_and_per_column():
    returns = pd.DataFrame({"A": TRADES, "B": TRADES * 2})
    returns.index = pd.MultiIndex.from_arrays([["2020"] * 4 + ["2021"] * 4, range(8)])

    table = performance_model.get_return_statistics_table(returns)

    assert list(table.index) == ["2020", "2021"]
    assert table.loc["2020", ("Win Rate", "A")] == pytest.approx(2 / 3)
    assert table.loc["2021", ("Average Win", "B")] == pytest.approx(0.05)


def test_the_discrete_kelly_criterion_matches_a_hand_calculation():
    win_rate, payoff_ratio = 4 / 7, 1.5

    assert performance_model.get_kelly_criterion(TRADES) == pytest.approx(
        win_rate - (1 - win_rate) / payoff_ratio
    )
    assert performance_model.get_kelly_criterion(TRADES, fraction=0.5) == pytest.approx(
        (win_rate - (1 - win_rate) / payoff_ratio) / 2
    )


def test_the_continuous_kelly_criterion_is_the_mean_over_the_variance():
    assert performance_model.get_kelly_criterion(
        TRADES, method="continuous"
    ) == pytest.approx(TRADES.mean() / TRADES.var(ddof=1))


def test_the_kelly_criterion_without_losses_is_undefined():
    assert np.isnan(performance_model.get_kelly_criterion(pd.Series([0.01, 0.02])))


def test_the_rolling_kelly_criterion_matches_each_window():
    returns = pd.DataFrame({"A": TRADES})

    for method in ("discrete", "continuous"):
        rolling = performance_model.get_rolling_kelly_criterion(returns, 5, method)

        assert rolling["A"].iloc[:4].isna().all()
        assert rolling["A"].iloc[-1] == pytest.approx(
            performance_model.get_kelly_criterion(TRADES.iloc[-5:], method)
        )


def test_an_unknown_kelly_method_raises():
    with pytest.raises(ValueError, match="discrete"):
        performance_model.get_kelly_criterion(TRADES, method="binary")


def lagged_market_returns(size=750, seed=1):
    """An asset that takes half its market reaction a day late, as a thinly traded stock does."""
    rng = np.random.default_rng(seed)
    benchmark = pd.Series(rng.normal(0, 0.01, size))
    asset = (
        0.6 * benchmark
        + 0.6 * benchmark.shift(1).fillna(0)
        + rng.normal(0, 0.004, size)
    )

    return pd.DataFrame({"Late": asset, "Same": 1.2 * benchmark}), benchmark


def test_dimson_and_scholes_williams_betas_match_their_regressions():
    sm = pytest.importorskip("statsmodels.api")
    returns, benchmark = lagged_market_returns()
    shifted = pd.concat(
        {shift: benchmark.shift(-shift) for shift in (-1, 0, 1)}, axis=1
    )
    sample = pd.concat([returns["Late"], shifted], axis=1).dropna()

    dimson = sm.OLS(sample["Late"], sm.add_constant(sample[[-1, 0, 1]])).fit()
    slopes = [
        sm.OLS(sample["Late"], sm.add_constant(sample[shift])).fit().params[shift]
        for shift in (-1, 0, 1)
    ]
    autocorrelation = np.corrcoef(sample[0], sample[-1])[0, 1]

    assert performance_model.get_beta_estimate(returns, benchmark, "dimson")[
        "Late"
    ] == pytest.approx(dimson.params[[-1, 0, 1]].sum())
    assert performance_model.get_beta_estimate(returns, benchmark, "scholes_williams")[
        "Late"
    ] == pytest.approx(sum(slopes) / (1 + 2 * autocorrelation))


def test_the_lagged_betas_recover_a_late_market_reaction():
    returns, benchmark = lagged_market_returns()

    ordinary = performance_model.get_beta(returns, benchmark)
    dimson = performance_model.get_beta_estimate(returns, benchmark, "dimson")

    # The OLS beta only sees the same-day half of the reaction of 1.2.
    assert ordinary["Late"] == pytest.approx(0.6, abs=0.05)
    assert dimson["Late"] == pytest.approx(1.2, abs=0.05)
    assert dimson["Same"] == pytest.approx(1.2)


def test_blume_and_vasicek_betas_move_towards_their_priors():
    returns, benchmark = lagged_market_returns()
    ordinary = performance_model.get_beta(returns, benchmark)

    blume = performance_model.get_beta_estimate(returns, benchmark, "blume")
    vasicek = performance_model.get_beta_estimate(returns, benchmark, "vasicek")

    assert blume.to_dict() == pytest.approx((0.67 * ordinary + 0.33).to_dict())
    prior = ordinary.mean()
    # Each beta lies between its own estimate and the cross-sectional mean.
    for ticker in returns:
        low, high = sorted([ordinary[ticker], prior])
        assert low <= vasicek[ticker] <= high

    # A single asset is shrunk towards the market's beta of 1.
    single = performance_model.get_beta_estimate(
        returns[["Late"]], benchmark, "vasicek"
    )
    assert ordinary["Late"] < single["Late"] < 1


def test_upside_and_downside_betas_split_by_the_benchmark_direction():
    benchmark = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.015, -0.005])
    asset = benchmark.where(benchmark > 0, 2 * benchmark)
    returns = asset.to_frame("Asymmetric")

    upside = performance_model.get_beta_estimate(returns, benchmark, "upside")
    downside = performance_model.get_beta_estimate(returns, benchmark, "downside")

    assert upside["Asymmetric"] == pytest.approx(1)
    assert downside["Asymmetric"] == pytest.approx(2)


def test_beta_estimates_per_period_and_rolling():
    returns, benchmark = lagged_market_returns(size=500)
    periods = pd.MultiIndex.from_arrays([np.repeat([2024, 2025], 250), returns.index])

    within_period = performance_model.get_beta_estimate(
        returns.set_axis(periods), benchmark.set_axis(periods), "dimson"
    )
    rolling = performance_model.get_beta_estimate(
        returns, benchmark, "dimson", window_size=250
    )

    assert within_period.index.tolist() == [2024, 2025]
    assert rolling["Late"].iloc[:249].isna().all()
    assert within_period.loc[2025].to_numpy() == pytest.approx(
        rolling.iloc[-1].to_numpy(), rel=0.05
    )


def test_an_unknown_beta_method_is_refused():
    returns, benchmark = lagged_market_returns(size=10)

    with pytest.raises(ValueError, match="must be one of"):
        performance_model.get_beta_estimate(returns, benchmark, "bloomberg")


def test_morningstar_risk_adjusted_return_matches_morningstar_illustration():
    # Morningstar's "MRAR Illustrated" factsheet: two funds with the same compounded excess
    # return of 9.38%, of which the steadier earns an MRAR of 9.37% and the other 9.10%.
    excess_returns = (
        pd.DataFrame(
            {
                "A": [0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1],
                "B": [0.1, 2, -0.9, 0.5, 3.82, 0.6, 0.7, 0, -0.2, -1.5, 1, 3],
            }
        )
        / 100
    )

    mrar = performance_model.get_morningstar_risk_adjusted_return(excess_returns)
    morningstar_return = performance_model.get_morningstar_risk_adjusted_return(
        excess_returns, gamma=0
    )

    assert mrar["A"] == pytest.approx(0.0937, abs=5e-5)
    assert mrar["B"] == pytest.approx(0.0910, abs=5e-5)
    assert morningstar_return["A"] == pytest.approx(0.0938, abs=5e-5)
    # Morningstar Risk, the return given up for the variation, is never negative.
    assert (morningstar_return - mrar).ge(0).all()


def test_morningstar_risk_adjusted_return_of_steady_returns_is_their_annual_return():
    excess_returns = pd.Series([0.01] * 24)

    assert performance_model.get_morningstar_risk_adjusted_return(
        excess_returns
    ) == pytest.approx(1.01**12 - 1)


def test_geometric_excess_return_divides_by_the_risk_free_return():
    returns = pd.DataFrame({"A": [0.02, -0.01]})
    risk_free_rate = pd.Series([0.01, 0.01])

    excess_returns = performance_model.get_geometric_excess_return(
        returns, risk_free_rate
    )

    np.testing.assert_allclose(excess_returns["A"], [1.02 / 1.01 - 1, 0.99 / 1.01 - 1])


def test_rolling_morningstar_risk_adjusted_return_matches_every_full_window():
    rng = np.random.default_rng(3)
    excess_returns = pd.DataFrame({"A": rng.normal(0.005, 0.04, 40)})
    excess_returns.iloc[5, 0] = np.nan

    rolling = performance_model.get_rolling_morningstar_risk_adjusted_return(
        excess_returns, 12
    )

    # A window holding the missing month is not full, so it has no value.
    assert rolling["A"].iloc[:17].isna().all()
    for end in range(17, 40):
        assert rolling["A"].iloc[end] == pytest.approx(
            performance_model.get_morningstar_risk_adjusted_return(
                excess_returns["A"].iloc[end - 11 : end + 1]
            )
        )


def test_morningstar_risk_adjusted_return_refuses_a_negative_gamma():
    with pytest.raises(ValueError, match="cannot be negative"):
        performance_model.get_morningstar_risk_adjusted_return(
            pd.Series([0.01, 0.02]), gamma=-1
        )


def test_lo_annualization_factor_matches_the_ar1_closed_form():
    # Lo (2002): with AR(1) returns ρ_k = ρ^k and η(q) has a closed form.
    for rho in (-0.3, 0.0, 0.2, 0.5):
        q = 12
        closed_form = q / np.sqrt(
            q + 2 * rho / (1 - rho) * (q - (1 - rho**q) / (1 - rho))
        )
        assert performance_model.get_lo_annualization_factor(
            [rho**k for k in range(1, q)], q
        ) == pytest.approx(closed_form)


def test_lo_annualization_factor_by_hand():
    # q = 3, ρ_1 = 0.1, ρ_2 = 0.05: 3 / SQRT(3 + 2 · (2 · 0.1 + 1 · 0.05)) = 3 / SQRT(3.5).
    assert performance_model.get_lo_annualization_factor(
        [0.1, 0.05], 3
    ) == pytest.approx(3 / np.sqrt(3.5))
    # Without autocorrelation the factor is the usual SQRT(q).
    assert performance_model.get_lo_annualization_factor([], 252) == pytest.approx(
        np.sqrt(252)
    )
    # A variance ratio that is not positive has no factor.
    assert np.isnan(performance_model.get_lo_annualization_factor([-0.9, -0.9], 3))


def test_lo_sharpe_ratio_approaches_the_true_ar1_sharpe_ratio():
    rng = np.random.default_rng(0)
    phi, q, n = 0.5, 12, 100_000
    shocks = rng.normal(0, 1, n)
    values = np.empty(n)
    values[0] = shocks[0]
    for t in range(1, n):
        values[t] = phi * values[t - 1] + shocks[t]
    excess_returns = pd.Series(0.1 + values)

    yearly_variance = (q + 2 * sum((q - k) * phi**k for k in range(1, q))) / (
        1 - phi**2
    )
    true_sharpe_ratio = q * 0.1 / np.sqrt(yearly_variance)

    lo = performance_model.get_lo_sharpe_ratio(excess_returns, q, lags=q - 1)

    assert lo == pytest.approx(true_sharpe_ratio, rel=0.05)
    # The SQRT(q) annualization overstates it by over half.
    assert np.sqrt(q) * excess_returns.mean() / excess_returns.std() > 1.5 * lo


def test_rolling_lo_sharpe_ratio_matches_every_window():
    rng = np.random.default_rng(1)
    excess_returns = pd.DataFrame(
        {"A": rng.normal(0.01, 0.05, 60), "B": rng.normal(0.0, 0.05, 60)}
    )
    rolling = performance_model.get_rolling_lo_sharpe_ratio(excess_returns, 36, 12)

    assert rolling.iloc[:35].isna().all().all()
    for end in (35, 47, 59):
        assert rolling["A"].iloc[end] == pytest.approx(
            performance_model.get_lo_sharpe_ratio(
                excess_returns["A"].iloc[end - 35 : end + 1], 12
            )
        )
    # Windows below the 30-observation minimum have no ratio.
    assert (
        performance_model.get_rolling_lo_sharpe_ratio(excess_returns, 24, 12)
        .isna()
        .all()
        .all()
    )


def test_lo_sharpe_ratio_bias_correction_by_hand():
    rng = np.random.default_rng(3)
    excess_returns = pd.Series(rng.normal(0.001, 0.01, 40))
    values = excess_returns.to_numpy()
    n, q, lags = len(values), 252, 3
    deviation = values - values.mean()
    # Moran (1948): the Box-Jenkins autocorrelation plus (n - k) / (n (n - 1)).
    rho = [
        np.sum(deviation[:-k] * deviation[k:]) / np.sum(deviation**2)
        + (n - k) / (n * (n - 1))
        for k in range(1, lags + 1)
    ]
    eta = q / np.sqrt(q + 2 * sum((q - k) * rho[k - 1] for k in range(1, lags + 1)))

    assert performance_model.get_lo_sharpe_ratio(
        excess_returns, q, lags
    ) == pytest.approx(values.mean() / values.std(ddof=1) * eta)
    # Fewer than 30 observations give no ratio.
    assert np.isnan(performance_model.get_lo_sharpe_ratio(excess_returns.iloc[:29], q))


def test_lo_sharpe_ratio_is_unbiased_for_independent_short_samples():
    # Without the bias correction the median η(q) / SQRT(q) of a quarter of independent
    # daily returns is about 1.07; with it, it is close to 1.
    rng = np.random.default_rng(4)
    samples = rng.normal(0.001, 0.01, (63, 1000))
    ratios = [
        performance_model.get_lo_sharpe_ratio(pd.Series(sample), 252)
        / (np.sqrt(252) * sample.mean() / sample.std(ddof=1))
        for sample in samples.T
    ]

    assert np.nanmedian(ratios) == pytest.approx(1, abs=0.03)


def test_sharpe_ratio_standard_error_methods():
    rng = np.random.default_rng(2)
    normal = pd.Series(rng.normal(0.05, 1, 100_000))
    sharpe_ratio = normal.mean() / normal.std()
    iid = np.sqrt((1 + sharpe_ratio**2 / 2) / len(normal))

    # Lo (2002), equation 9, and the other two agree for independent normal returns.
    assert performance_model.get_sharpe_ratio_standard_error(normal) == pytest.approx(
        iid
    )
    for method in ("non_normal", "gmm"):
        assert performance_model.get_sharpe_ratio_standard_error(
            normal, method
        ) == pytest.approx(iid, rel=0.01)

    # Without lags the GMM error is the non-normal (Mertens) error.
    fat_tailed = pd.Series(rng.standard_t(5, 5000) * 0.01 + 0.0005)
    assert performance_model.get_sharpe_ratio_standard_error(
        fat_tailed, "gmm", lags=0
    ) == pytest.approx(
        performance_model.get_sharpe_ratio_standard_error(fat_tailed, "non_normal"),
        rel=0.01,
    )

    with pytest.raises(ValueError, match="iid, non_normal or gmm"):
        performance_model.get_sharpe_ratio_standard_error(normal, "other")


def test_sharpe_ratio_standard_error_widens_with_autocorrelation():
    rng = np.random.default_rng(3)
    shocks = rng.normal(0, 1, 5000)
    values = np.empty(5000)
    values[0] = shocks[0]
    for t in range(1, 5000):
        values[t] = 0.5 * values[t - 1] + shocks[t]
    excess_returns = pd.Series(0.1 + values)

    # With ρ = 0.5 the variance of the mean triples: (1 + ρ) / (1 − ρ).
    iid = performance_model.get_sharpe_ratio_standard_error(excess_returns)
    gmm = performance_model.get_sharpe_ratio_standard_error(
        excess_returns, "gmm", lags=30
    )
    assert gmm / iid == pytest.approx(np.sqrt(3), rel=0.1)


def test_minimum_track_record_length_by_hand():
    sharpe_ratio, skewness, kurtosis = 0.1, -0.5, 6.0
    by_hand = (
        1
        + (1 - skewness * sharpe_ratio + (kurtosis - 1) / 4 * sharpe_ratio**2)
        * (1.6448536269514722 / sharpe_ratio) ** 2
    )

    minimum = performance_model.get_minimum_track_record_length(
        sharpe_ratio, 0.0, skewness, kurtosis
    )
    assert minimum == pytest.approx(by_hand)

    # With MinTRL observations the Probabilistic Sharpe Ratio is the confidence level.
    assert performance_model.get_probabilistic_sharpe_ratio(
        sharpe_ratio, 0.0, skewness, kurtosis, minimum
    ) == pytest.approx(0.95)

    # A Sharpe ratio at or below the benchmark never becomes significant.
    assert np.isnan(
        performance_model.get_minimum_track_record_length(0.1, 0.2, 0.0, 3.0)
    )
    frame = performance_model.get_minimum_track_record_length(
        pd.Series([0.1, -0.1]), 0.0, 0.0, 3.0
    )
    assert frame.iloc[0] == pytest.approx(1 + 1.005 * (1.6448536269514722 / 0.1) ** 2)
    assert np.isnan(frame.iloc[1])

    with pytest.raises(ValueError, match="between 0 and 1"):
        performance_model.get_minimum_track_record_length(0.1, 0.0, 0.0, 3.0, 1.5)


def test_coskewness_and_cokurtosis_by_hand():
    rng = np.random.default_rng(4)
    market = pd.Series(rng.standard_t(6, 3000) * 0.01)
    asset = 0.5 * market + 3 * market**2 + pd.Series(rng.normal(0, 0.01, 3000))
    returns = pd.DataFrame({"Asset": asset, "Market": market})

    a, m = asset - asset.mean(), market - market.mean()
    beta = (a * m).sum() / (m**2).sum()
    residual = a - beta * m

    coskewness = performance_model.get_coskewness(returns, market)
    cokurtosis = performance_model.get_cokurtosis(returns, market)

    assert coskewness["Asset"] == pytest.approx(
        (a * m**2).mean() / (a.std(ddof=0) * m.var(ddof=0))
    )
    assert cokurtosis["Asset"] == pytest.approx(
        (a * m**3).mean() / (a.std(ddof=0) * m.std(ddof=0) ** 3)
    )
    # Harvey and Siddique (2000): the residual of the market regression.
    assert performance_model.get_coskewness(returns, market, "residual")[
        "Asset"
    ] == pytest.approx(
        (residual * m**2).mean() / (np.sqrt((residual**2).mean()) * (m**2).mean())
    )
    assert performance_model.get_cokurtosis(returns, market, "residual")[
        "Asset"
    ] == pytest.approx(
        (residual * m**3).mean()
        / (np.sqrt((residual**2).mean()) * (m**2).mean() ** 1.5)
    )
    # With itself, the co-moments are the market's own skewness and (raw) kurtosis.
    assert coskewness["Market"] == pytest.approx((m**3).mean() / m.std(ddof=0) ** 3)
    assert cokurtosis["Market"] == pytest.approx((m**4).mean() / m.std(ddof=0) ** 4)


def test_coskewness_per_period_uses_each_period_on_its_own():
    rng = np.random.default_rng(5)
    index = pd.MultiIndex.from_arrays(
        [np.repeat(["2020", "2021"], 250), np.arange(500)]
    )
    market = pd.Series(rng.normal(0, 0.01, 500), index=index)
    returns = pd.DataFrame({"A": market * 2 + rng.normal(0, 0.01, 500)}, index=index)
    returns.iloc[:5, 0] = np.nan

    per_period = performance_model.get_coskewness(returns, market)

    for period in ("2020", "2021"):
        assert per_period.loc[period, "A"] == pytest.approx(
            performance_model.get_coskewness(
                returns.loc[period].reset_index(drop=True),
                market.loc[period].reset_index(drop=True),
            )["A"]
        )


def test_get_calmar_ratio_divides_the_return_by_the_absolute_maximum_drawdown():
    result = performance_model.get_calmar_ratio(
        returns=pd.Series([0.2, 0.1, 0.3]),
        maximum_drawdown=pd.Series([-0.1, -0.5, np.nan]),
    )

    # A period without a drawdown has no ratio and is dropped.
    pd.testing.assert_series_equal(result, pd.Series([2.0, 0.2]))


def test_get_average_drawdown_averages_the_periods_under_water():
    # Wealth 1.1, 0.55, 0.66, 1.32 against a running peak of 1.1, 1.1, 1.1, 1.32: two
    # days under water, 50% and 40% below the peak.
    returns = pd.Series([0.1, -0.5, 0.2, 1.0])

    assert performance_model.get_average_drawdown(returns) == pytest.approx(-0.45)

    # On a level (a rate or a spread) the drawdown is the distance to the running peak.
    levels = pd.Series([1.0, 3.0, 2.0, 1.0, 4.0])
    assert performance_model.get_average_drawdown(
        levels, method="level"
    ) == pytest.approx(-1.5)

    with pytest.raises(ValueError, match="method"):
        performance_model.get_average_drawdown(returns, method="price")


def test_get_average_drawdown_per_sub_period():
    index = pd.MultiIndex.from_tuples(
        [("2020", 1), ("2020", 2), ("2021", 1), ("2021", 2)]
    )
    returns = pd.DataFrame({"A": [0.1, -0.5, -0.2, 0.1]}, index=index)

    result = performance_model.get_average_drawdown(returns)

    # Each sub-period starts with a fresh peak: 2020 falls 50% from 1.1, 2021 falls 20%
    # and recovers to 12% below the starting value of 1.
    assert result.loc["2020", "A"] == pytest.approx(-0.5)
    assert result.loc["2021", "A"] == pytest.approx((-0.2 + 0.8 * 1.1 - 1) / 2)


def test_get_sterling_ratio_adds_the_adjustment_to_the_average_drawdown():
    result = performance_model.get_sterling_ratio(
        returns=pd.Series([0.2, 0.3]),
        average_drawdown=pd.Series([-0.1, -0.2]),
    )
    pd.testing.assert_series_equal(result, pd.Series([1.0, 1.0]))

    assert performance_model.get_sterling_ratio(
        pd.Series([0.2]), pd.Series([-0.1]), adjustment=0
    ).iloc[0] == pytest.approx(2.0)


def test_get_burke_ratio_divides_the_excess_return_by_the_drawdown_measure():
    result = performance_model.get_burke_ratio(
        excess_returns=pd.Series([0.3, 0.1]),
        burke_drawdown_measure=pd.Series([0.15, np.nan]),
    )

    pd.testing.assert_series_equal(result, pd.Series([2.0]))


def test_get_kappa_ratio_scales_by_the_lower_partial_moment():
    excess_returns = pd.Series([0.02, -0.01, 0.03, -0.02])
    mean = 0.005

    # Order 2 is the Sortino ratio: the root of the mean squared shortfall.
    assert performance_model.get_kappa_ratio(excess_returns, order=2) == pytest.approx(
        mean / ((0.01**2 + 0.02**2) / 4) ** (1 / 2)
    )
    assert performance_model.get_kappa_ratio(excess_returns) == pytest.approx(
        mean / ((0.01**3 + 0.02**3) / 4) ** (1 / 3)
    )

    frame = pd.DataFrame({"A": excess_returns, "B": -excess_returns})
    result = performance_model.get_kappa_ratio(frame, order=2)
    assert result["A"] == pytest.approx(0.005 / np.sqrt(0.000125))
    assert result["B"] == pytest.approx(-0.005 / np.sqrt((0.02**2 + 0.03**2) / 4))

    with pytest.raises(TypeError):
        performance_model.get_kappa_ratio([0.01, -0.02])


def test_get_kappa_ratio_per_sub_period():
    index = pd.MultiIndex.from_tuples(
        [("2020", 1), ("2020", 2), ("2021", 1), ("2021", 2)]
    )
    frame = pd.DataFrame({"A": [0.02, -0.01, 0.03, -0.02]}, index=index)

    result = performance_model.get_kappa_ratio(frame, order=1)

    # Order 1: the mean over the mean shortfall.
    assert result.loc["2020", "A"] == pytest.approx(0.005 / 0.005)
    assert result.loc["2021", "A"] == pytest.approx(0.005 / 0.01)


def test_get_covariance_matrix_is_the_sample_covariance():
    returns = pd.DataFrame({"X": [1.0, 2.0, 3.0], "Y": [2.0, 4.0, 7.0]})

    result = performance_model.get_covariance_matrix(returns)

    # Sample statistics (ddof=1): var(X) = 2 / 2, cov(X, Y) = (7/3 + 8/3) / 2.
    assert result.loc["X", "X"] == pytest.approx(1.0)
    assert result.loc["X", "Y"] == pytest.approx(2.5)
    assert result.loc["Y", "X"] == result.loc["X", "Y"]

    with pytest.raises(TypeError):
        performance_model.get_covariance_matrix(returns["X"])
