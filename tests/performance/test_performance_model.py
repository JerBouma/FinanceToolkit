"""Performance Model Tests"""

import io
import warnings
import zipfile

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from financetoolkit.performance import performance_model
from financetoolkit.risk import risk_model
from financetoolkit.risk.helpers import (
    compound_risk_free_rate,
    get_observations_per_year,
)
from financetoolkit.risk.value_at_risk import cvar_model
from financetoolkit.utilities.error_model import InvalidArgumentError

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


def test_get_beta_rolling(recorder):
    recorder.capture(
        performance_model.get_beta(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
            window=2,
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

    def fake_get_request(url, timeout, extra_headers):  # noqa: ARG001
        # Dartmouth turns away requests that present themselves as a browser.
        assert extra_headers == performance_model.TOOLKIT_HEADERS
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


def test_get_alpha_rolling(recorder):
    recorder.capture(
        performance_model.get_alpha(
            asset_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
            window=2,
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


def test_get_sharpe_ratio_rolling(recorder):
    recorder.capture(
        performance_model.get_sharpe_ratio(
            excess_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), window=2
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


def test_get_sortino_ratio_rolling(recorder):
    recorder.capture(
        performance_model.get_sortino_ratio(
            excess_returns=pd.Series([0.3, 0.2, -0.1, -0.05, 0.06, -0.2, 0.1]),
            window=3,
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


def test_get_m2_ratio_rolling(recorder):
    recorder.capture(
        performance_model.get_m2_ratio(
            asset_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            risk_free_rate=pd.Series([0.01, 0.02, 0.01, 0.0, 0.006]),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
            window=2,
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


def test_get_tracking_error_rolling(recorder):
    recorder.capture(
        performance_model.get_tracking_error(
            asset_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
            window=2,
        ).round(4)
    )


def test_get_information_ratio(recorder):
    recorder.capture(
        performance_model.get_information_ratio(
            asset_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
        )
    )


def test_get_information_ratio_rolling(recorder):
    recorder.capture(
        performance_model.get_information_ratio(
            asset_returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            benchmark_returns=pd.Series([0.31, 0.19, 0.5, 0, 0.03]),
            window=2,
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

    def fake_get_request(url, timeout, extra_headers):  # noqa: ARG001
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


def test_get_beta_rolling_masks_the_benchmark_per_asset():
    # Asset B has no returns for the first three dates, so its benchmark variance must
    # cover only the dates on which B has a return, as the covariance does.
    benchmark = pd.Series([0.01, -0.02, 0.03, 0.015, -0.01, 0.02, -0.005])
    returns = pd.DataFrame(
        {
            "A": [0.02, -0.01, 0.025, 0.01, -0.02, 0.03, 0.0],
            "B": [np.nan, np.nan, np.nan, 0.02, -0.015, 0.01, -0.01],
        }
    )

    rolling_beta = performance_model.get_beta(returns, benchmark, window=4)

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
    with pytest.raises(
        InvalidArgumentError, match="window is a number of observations"
    ):
        performance_model.get_sortino_ratio(excess_returns["A"], window=1)


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

    # A date without a return (a day only another ticker traded on) is left out of the
    # windows of that column, which run over its own dates.
    expected = pd.concat(
        {
            column: returns[column].dropna().rolling(20).apply(window_omega, raw=True)
            for column in returns.columns
        },
        axis=1,
    ).reindex(returns.index)
    pd.testing.assert_frame_equal(
        performance_model.get_omega_ratio(returns, window=20), expected, rtol=1e-12
    )
    assert (
        performance_model.get_omega_ratio(pd.Series([0.01, 0.02, 0.0, 0.0]), window=2)
        .iloc[1:]
        .isna()
        .all()
    )


def test_omega_ratio_without_losses_is_nan():
    gains_only = pd.Series([0.01, 0.02, 0.005])

    assert np.isnan(performance_model.get_omega_ratio(gains_only))
    assert np.isnan(performance_model.get_omega_ratio(pd.Series([0.0, 0.0])))
    assert np.isnan(performance_model.get_omega_ratio(gains_only, window=2).iloc[-1])


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
        rolling = performance_model.get_kelly_criterion(returns, method, window=5)

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

    rolling = performance_model.get_morningstar_risk_adjusted_return(
        excess_returns, window=12
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
    rolling = performance_model.get_lo_sharpe_ratio(excess_returns, 12, window=36)

    assert rolling.iloc[:35].isna().all().all()
    for end in (35, 47, 59):
        assert rolling["A"].iloc[end] == pytest.approx(
            performance_model.get_lo_sharpe_ratio(
                excess_returns["A"].iloc[end - 35 : end + 1], 12
            )
        )
    # Windows below the 30-observation minimum have no ratio.
    assert (
        performance_model.get_lo_sharpe_ratio(excess_returns, 12, window=24)
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

    with pytest.raises(ValueError, match="'iid', 'non_normal' or 'gmm'"):
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


def test_get_average_largest_drawdown_averages_the_deepest_episodes():
    # Wealth 1.1, 0.55, 0.66, 1.32, 1.188, 1.2474, 1.4969, 1.0478: three drawdowns from their
    # peaks, of 50% (1.1 to 0.55), 10% (1.32 to 1.188) and 30% (1.4969 to 1.0478, not recovered).
    returns = pd.Series([0.1, -0.5, 0.2, 1.0, -0.1, 0.05, 0.2, -0.3])

    assert performance_model.get_average_largest_drawdown(returns, 1) == pytest.approx(
        -0.5
    )
    assert performance_model.get_average_largest_drawdown(returns, 2) == pytest.approx(
        -0.4
    )
    # With fewer drawdowns than d, all of them: Bacon's average drawdown.
    for largest in (3, 10):
        assert performance_model.get_average_largest_drawdown(
            returns, largest
        ) == pytest.approx((-0.5 - 0.1 - 0.3) / 3)

    # The same depths as the drawdown table of the risk module.
    table = risk_model.get_drawdown_table(returns.to_frame("A"), top=3)
    assert performance_model.get_average_largest_drawdown(returns) == pytest.approx(
        table["Drawdown"].mean()
    )

    # On a level (a rate or a spread) the drawdown is the distance to the running peak.
    levels = pd.Series([1.0, 3.0, 2.0, 1.0, 4.0, 3.5])
    assert performance_model.get_average_largest_drawdown(
        levels, method="level"
    ) == pytest.approx((-2.0 - 0.5) / 2)

    # Without a drawdown it is 0, without returns NaN.
    assert performance_model.get_average_largest_drawdown(pd.Series([0.1, 0.2])) == 0
    assert np.isnan(performance_model.get_average_largest_drawdown(pd.Series([np.nan])))

    with pytest.raises(ValueError, match="method"):
        performance_model.get_average_largest_drawdown(returns, method="price")
    with pytest.raises(ValueError, match="largest_drawdowns"):
        performance_model.get_average_largest_drawdown(returns, 0)


def test_get_average_largest_drawdown_per_sub_period():
    index = pd.MultiIndex.from_tuples(
        [("2020", 1), ("2020", 2), ("2020", 3), ("2021", 1), ("2021", 2)]
    )
    returns = pd.DataFrame(
        {"A": [-0.1, 0.2, -0.05, -0.2, 0.1], "B": [0.1, 0.1, 0.1, 0.1, np.nan]},
        index=index,
    )

    result = performance_model.get_average_largest_drawdown(returns)

    # Each sub-period starts with a fresh peak: 2020 has a drawdown of 10% from the starting
    # value of 1 and one of 5% from 1.08, 2021 one of 20%, which 10% does not end.
    assert result.loc["2020", "A"] == pytest.approx((-0.1 - 0.05) / 2)
    assert result.loc["2021", "A"] == pytest.approx(-0.2)
    assert (result["B"] == 0).all()


def test_get_sterling_ratio_adds_the_adjustment_to_the_average_largest_drawdown():
    result = performance_model.get_sterling_ratio(
        returns=pd.Series([0.2, 0.3]),
        average_largest_drawdown=pd.Series([-0.1, -0.2]),
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


def test_sharpe_ratio_series_keeps_within_period_groups():
    periods = pd.PeriodIndex(["2026-01"] * 3 + ["2026-02"] * 3, freq="M")
    index = pd.MultiIndex.from_arrays([periods, [0, 1, 2, 0, 1, 2]])
    returns = pd.Series([-0.02, 0.0, 0.02, 0.02, 0.04, 0.06], index=index, name="AAPL")
    expected = pd.Series(
        [0.0, 2.0], index=pd.PeriodIndex(["2026-01", "2026-02"], freq="M"), name="AAPL"
    )

    actual = performance_model.get_sharpe_ratio(returns)
    pd.testing.assert_series_equal(actual, expected)
    frame_result = performance_model.get_sharpe_ratio(returns.to_frame())["AAPL"]
    pd.testing.assert_series_equal(actual, frame_result)

    # The Safety-First Ratio is the Sharpe ratio above the threshold: means of -0.02 and
    # 0.02 over a standard deviation of 0.02.
    pd.testing.assert_series_equal(
        performance_model.get_safety_first_ratio(
            returns, minimum_acceptable_return=0.02
        ),
        pd.Series([-1.0, 1.0], index=expected.index, name="AAPL"),
    )


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        # January: a mean of 0; February: 0.01 over a downside deviation of 0.01 / sqrt(3).
        (performance_model.get_sortino_ratio, [0.0, np.sqrt(3)]),
        # Mean gains of 0.02 / 3 and 0.04 / 3 over downside deviations of 0.02 / sqrt(3)
        # and 0.01 / sqrt(3).
        (
            performance_model.get_upside_potential_ratio,
            [1 / np.sqrt(3), 4 / np.sqrt(3)],
        ),
        # Order 3: February's root of the lower partial moment is 0.01 / cbrt(3).
        (performance_model.get_kappa_ratio, [0.0, np.cbrt(3)]),
    ],
)
def test_downside_ratios_of_a_series_keep_within_period_groups(ratio, expected):
    periods = pd.PeriodIndex(["2026-01"] * 3 + ["2026-02"] * 3, freq="M")
    index = pd.MultiIndex.from_arrays([periods, [0, 1, 2, 0, 1, 2]])
    returns = pd.Series([-0.02, 0.0, 0.02, -0.01, 0.01, 0.03], index=index, name="AAPL")
    expected = pd.Series(
        expected, index=pd.PeriodIndex(["2026-01", "2026-02"], freq="M"), name="AAPL"
    )

    actual = ratio(returns)
    pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_series_equal(actual, ratio(returns.to_frame())["AAPL"])


def test_beta_implied_return_sets_the_residual_against_its_history():
    days = pd.period_range("2024-01-01", periods=400, freq="D")
    rng = np.random.default_rng(3)
    market = pd.Series(rng.normal(0, 0.01, len(days)), index=days)
    own = rng.normal(0, 0.01, len(days))
    # A large stock-specific fall on the last day, the largest since day 120.
    own[120], own[-1] = -0.2, -0.15
    returns = pd.DataFrame({"A": market + own}, index=days)

    result = performance_model.get_beta_implied_return(
        returns, market, "daily", beta_window=50
    )

    assert result[("Residual Largest Since", "A")].iloc[-1] == str(days[120])
    residuals = result[("Residual", "A")].dropna()
    # The share of the earlier residuals with a smaller size.
    assert result[("Residual Percentile", "A")].iloc[-1] == pytest.approx(
        np.mean(residuals.abs().iloc[:-1] < abs(residuals.iloc[-1]))
    )
    # A year of residuals is needed first: the first 252 have no percentile.
    assert (
        result[("Residual Percentile", "A")].dropna().index[0] == residuals.index[252]
    )


def test_residual_value_uses_the_market_cap_of_the_previous_close():
    days = pd.period_range("2024-01-29", periods=8, freq="D")
    market_cap = pd.DataFrame(
        {"A": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0], "B": np.nan},
        index=days,
    )
    residual = pd.DataFrame({"A": 0.01, "B": 0.02}, index=days)

    daily = performance_model.get_residual_value(residual, market_cap, "daily")

    assert np.isnan(daily["A"].iloc[0])
    assert daily["A"].iloc[1:].to_numpy() == pytest.approx(
        0.01 * market_cap["A"].iloc[:-1].to_numpy()
    )
    # Without a market cap the value is NaN, not 0.
    assert daily["B"].isna().all()

    monthly_residual = pd.DataFrame(
        {"A": [0.1, 0.2], "B": [0.1, 0.2]},
        index=pd.PeriodIndex(["2024-01", "2024-02"], freq="M"),
    )
    monthly = performance_model.get_residual_value(
        monthly_residual, market_cap, "monthly"
    )
    # February is valued at the last close of January (31 January: 120).
    assert np.isnan(monthly.loc["2024-01", "A"])
    assert monthly.loc["2024-02", "A"] == pytest.approx(0.2 * 120.0)


def test_get_appraisal_ratio_rolling_uses_the_regression_of_each_window():
    index = pd.period_range("2020-01", periods=24, freq="M")
    rng = np.random.default_rng(3)
    benchmark = pd.Series(rng.normal(0.01, 0.04, 24), index=index)
    excess = pd.DataFrame({"A": 0.8 * benchmark + rng.normal(0, 0.02, 24)}, index=index)
    jensens_alpha = pd.DataFrame({"A": rng.normal(0.01, 0.01, 24)}, index=index)

    ratio = performance_model.get_appraisal_ratio(
        jensens_alpha,
        excess_returns=excess,
        benchmark_excess_returns=benchmark,
        window=12,
    )

    # Each ratio divides by the residual standard deviation of the OLS regression over its
    # own window of 12 returns, the first one at the 12th return.
    expected = pd.Series(np.nan, index=index)
    for end in range(11, 24):
        y = excess["A"].iloc[end - 11 : end + 1].to_numpy()
        x = benchmark.iloc[end - 11 : end + 1].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        residuals = y - slope * x - intercept
        expected.iloc[end] = jensens_alpha["A"].iloc[end] / np.sqrt(
            (residuals**2).sum() / 11
        )
    pd.testing.assert_series_equal(ratio["A"], expected, check_names=False)
    assert ratio["A"].iloc[:11].isna().all()
    assert ratio["A"].iloc[11:].notna().all()


def _crypto_and_stock_returns():
    rng = np.random.default_rng(4)
    index = pd.period_range("2022-01-01", "2022-12-31", freq="D")
    weekday = index.dayofweek < 5  # noqa: PLR2004
    return pd.DataFrame(
        {
            "Crypto": rng.normal(0.001, 0.03, len(index)),
            "Stock": np.where(weekday, rng.normal(0.0005, 0.01, len(index)), np.nan),
        },
        index=index,
    )


def test_rolling_ratios_of_a_stock_next_to_a_cryptocurrency_use_its_own_returns():
    returns = _crypto_and_stock_returns()
    stock = returns["Stock"].dropna()

    for function in (
        performance_model.get_sharpe_ratio,
        performance_model.get_sortino_ratio,
        performance_model.get_kelly_criterion,
    ):
        mixed = function(returns, window=20)
        alone = function(stock.to_frame(), window=20)

        # The weekends the stock does not trade on do not break every window of it.
        assert mixed["Stock"].notna().sum() == len(stock) - 19
        pd.testing.assert_series_equal(mixed["Stock"].dropna(), alone["Stock"].dropna())
        assert mixed["Crypto"].notna().sum() == len(returns) - 19


def test_get_jensens_alpha_needs_matching_types():
    returns = pd.DataFrame({"A": [0.01, 0.02]})

    with pytest.raises(TypeError):
        performance_model.get_jensens_alpha(
            returns, 0.0, pd.Series([1.0, 1.0]), pd.Series([0.01, 0.01])
        )


def test_only_a_ticker_trading_on_every_weekend_counts_365_days_a_year():
    returns = _crypto_and_stock_returns()
    index = returns.index
    # An exchange open from Sunday to Thursday and a stock with a one-off Saturday session.
    returns["Sunday to Thursday"] = np.where(
        (index.dayofweek < 4) | (index.dayofweek == 6), 0.001, np.nan  # noqa: PLR2004
    )
    returns["One Saturday"] = returns["Stock"]
    returns.loc["2022-01-08", "One Saturday"] = 0.002

    assert get_observations_per_year(returns).to_dict() == {
        "Crypto": 365.0,
        "Stock": 252.0,
        "Sunday to Thursday": 252.0,
        "One Saturday": 252.0,
    }


def test_the_period_risk_free_rate_does_not_depend_on_the_tickers():
    returns = _crypto_and_stock_returns()
    annualized_rate = pd.Series(0.04, index=returns.index)

    mixed = compound_risk_free_rate(annualized_rate, returns, "monthly")
    stock = compound_risk_free_rate(
        annualized_rate, returns[["Stock"]].dropna(), "monthly"
    )
    days_in_month = returns.index.asfreq("M").value_counts().sort_index()

    # Every calendar day earns 1 / 365 of the year: a full year earns the annualized rate
    # and a month its days, whether the dates are the trading days of the stock or every
    # calendar day of the cryptocurrency. Only a weekend at the turn of a month moves to
    # the month of the next trading day.
    yearly = compound_risk_free_rate(annualized_rate, returns, "yearly")
    assert yearly.iloc[0] == pytest.approx(0.04, rel=1e-12)
    assert mixed.to_numpy() == pytest.approx(
        (1.04 ** (days_in_month / 365) - 1).to_numpy(), rel=1e-12
    )
    weekday_turns = [
        month
        for month in stock.index[1:]
        if month.end_time.dayofweek < 5  # noqa: PLR2004
        and (month - 1).end_time.dayofweek < 5  # noqa: PLR2004
    ]
    assert len(weekday_turns) > 3  # noqa: PLR2004
    assert stock[weekday_turns].to_numpy() == pytest.approx(
        mixed[weekday_turns].to_numpy(), rel=1e-12
    )


def test_the_rolling_standard_deviation_pairs_the_asset_with_the_benchmark_dates():
    days = pd.period_range("2024-01-01", "2024-03-31", freq="D")
    weekday = days.dayofweek < 5  # noqa: PLR2004
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(
        {
            "Crypto": rng.normal(0, 0.03, len(days)),
            "Stock": np.where(weekday, rng.normal(0, 0.01, len(days)), np.nan),
        },
        index=days,
    )
    benchmark = pd.Series(
        np.where(weekday, rng.normal(0, 0.01, len(days)), np.nan), index=days
    )

    result = performance_model.get_standard_deviation(returns, benchmark, window=10)
    benchmark_deviation = performance_model.get_standard_deviation(benchmark, window=10)
    beta = performance_model.get_beta(returns, benchmark, window=10)

    # The stock's windows hold its own trading days, as without the cryptocurrency.
    stock = returns["Stock"].dropna()
    pd.testing.assert_series_equal(
        result["Stock"].dropna(), stock.rolling(10).std().dropna(), check_names=False
    )
    # The cryptocurrency's weekend returns are compounded into Monday, the sample of its
    # Beta, so its total risk ratio is at least its absolute Beta.
    crypto = (1 + returns["Crypto"]).groupby(np.cumsum(weekday[::-1])[::-1]).transform(
        "prod"
    ) - 1
    crypto = crypto[weekday]
    expected = crypto.rolling(10).std().dropna()
    pd.testing.assert_series_equal(
        result["Crypto"].dropna(), expected, check_names=False
    )
    total_risk_ratio = result.div(benchmark_deviation, axis=0)
    assert (total_risk_ratio + 1e-12 >= beta.abs())[beta.notna()].all().all()


def test_the_safety_first_ratio_is_the_sharpe_ratio_above_the_threshold():
    returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])

    result = performance_model.get_safety_first_ratio(
        returns, minimum_acceptable_return=0.005
    )

    assert result == pytest.approx((returns.mean() - 0.005) / returns.std(ddof=1))
    assert performance_model.get_safety_first_ratio(returns) == pytest.approx(
        performance_model.get_sharpe_ratio(returns)
    )


def test_constant_returns_have_no_sharpe_or_safety_first_ratio():
    # A standard deviation of exactly 0 (0.01) and one of rounding noise once a constant
    # risk-free rate is subtracted (0.001 - 0.0001) both leave the ratio undefined, not infinite.
    returns = pd.DataFrame({"Flat": [0.01] * 250, "Noisy": [0.001 - 0.0001] * 250})
    within_period = returns.set_axis(
        pd.MultiIndex.from_product([["2024", "2025"], range(125)])
    )

    assert performance_model.get_sharpe_ratio(returns).isna().all()
    assert np.isnan(performance_model.get_sharpe_ratio(returns["Flat"]))
    assert performance_model.get_sharpe_ratio(within_period).isna().all().all()
    assert performance_model.get_sharpe_ratio(returns, window=20).isna().all().all()
    assert performance_model.get_safety_first_ratio(returns).isna().all()
    assert (
        performance_model.get_safety_first_ratio(returns, window=20).isna().all().all()
    )

    # Returns with variation keep their ratio.
    varying = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    assert performance_model.get_sharpe_ratio(varying) == pytest.approx(
        varying.mean() / varying.std()
    )


def test_ratios_without_a_denominator_are_nan_rather_than_infinite():
    # Gains only: no downside deviation, lower partial moment or losses to divide by.
    gains = pd.DataFrame({"Gains": [0.01, 0.02, 0.005, 0.03] * 10})
    within_period = gains.set_axis(
        pd.MultiIndex.from_product([["2024", "2025"], range(20)])
    )

    assert performance_model.get_sortino_ratio(gains).isna().all()
    assert np.isnan(performance_model.get_sortino_ratio(gains["Gains"]))
    assert performance_model.get_sortino_ratio(within_period).isna().all().all()
    assert performance_model.get_sortino_ratio(gains, window=5).isna().all().all()
    assert performance_model.get_kappa_ratio(gains).isna().all()
    assert performance_model.get_kappa_ratio(within_period).isna().all().all()
    assert performance_model.get_gain_to_pain_ratio(gains).isna().all()
    assert performance_model.get_omega_ratio(gains).isna().all()
    assert performance_model.get_upside_potential_ratio(gains).isna().all()

    # A return series that tracks its benchmark exactly has no tracking error.
    benchmark = pd.Series([0.01, -0.02, 0.015, -0.005] * 10, name="Benchmark")
    tracking = pd.DataFrame({"Tracker": benchmark.to_numpy()})
    assert performance_model.get_information_ratio(tracking, benchmark).isna().all()
    assert (
        performance_model.get_information_ratio(tracking, benchmark, window=5)
        .isna()
        .all()
        .all()
    )

    # A beta of 0 leaves the Treynor ratio undefined.
    assert (
        performance_model.get_treynor_ratio(pd.Series([0.05]), 0.02, pd.Series([0.0]))
        .isna()
        .all()
    )

    # A loss keeps the Sortino ratio defined.
    mixed = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    assert np.isfinite(performance_model.get_sortino_ratio(mixed))


def test_the_upside_potential_ratio_averages_both_moments_over_every_return():
    returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.0])

    # Sortino, van der Meer & Plantinga (1999): (0.05 / 5) / sqrt(0.0005 / 5).
    result = performance_model.get_upside_potential_ratio(returns)
    assert result == pytest.approx(0.01 / np.sqrt(0.0001))

    rolling = performance_model.get_upside_potential_ratio(returns, window=5)
    assert rolling.iloc[-1] == pytest.approx(result)


def test_the_upside_potential_ratio_without_a_loss_or_with_one_return_is_nan():
    assert np.isnan(
        performance_model.get_upside_potential_ratio(pd.Series([0.01, 0.02, 0.0]))
    )
    assert np.isnan(performance_model.get_upside_potential_ratio(pd.Series([-0.01])))
    assert (
        performance_model.get_upside_potential_ratio(
            pd.Series([0.01, 0.02, 0.03]), window=2
        )
        .isna()
        .all()
    )


def test_the_pain_ratio_without_a_drawdown_is_nan():
    # Without a drawdown there is no pain to divide by.
    assert performance_model.get_pain_ratio(
        pd.Series([0.05, 0.02]), pd.Series([0.0, 0.01])
    ).to_list() == [pytest.approx(2.0)]


# A fixed series of daily returns, with the quantstats 0.0.86 values on it as the reference.
QUANTSTATS_RETURNS = pd.Series(
    [0.012, -0.004, 0.009, 0.015, -0.011, 0.003, 0.007, -0.002, 0.011, -0.008]
    + [0.006, 0.004, -0.013, 0.010, 0.005, -0.001, 0.008, 0.002, -0.006, 0.013]
)

# PerformanceAnalytics' portfolio_bacon data (Bacon, 2008): 24 monthly returns of a portfolio
# and its benchmark. The expected values are the outputs in PerformanceAnalytics 2.0.4's
# tests/Examples/PerformanceAnalytics-Ex.Rout.save.
BACON_PORTFOLIO = pd.Series(
    [
        0.003,
        0.026,
        0.011,
        -0.010,
        0.015,
        0.025,
        0.016,
        0.067,
        -0.014,
        0.040,
        -0.005,
        0.081,
    ]
    + [
        0.040,
        -0.037,
        -0.061,
        0.017,
        -0.049,
        -0.022,
        0.070,
        0.058,
        -0.065,
        0.024,
        -0.005,
        -0.009,
    ]
)
BACON_BENCHMARK = pd.Series(
    [
        0.002,
        0.025,
        0.018,
        -0.011,
        0.014,
        0.018,
        0.014,
        0.065,
        -0.015,
        0.042,
        -0.006,
        0.083,
    ]
    + [
        0.039,
        -0.038,
        -0.062,
        0.015,
        -0.048,
        0.021,
        0.060,
        0.056,
        -0.067,
        0.019,
        -0.003,
        0.000,
    ]
)


def _bacon_beta():
    return np.cov(BACON_PORTFOLIO, BACON_BENCHMARK)[0, 1] / BACON_BENCHMARK.var()


def test_smart_sharpe_and_sortino_ratios_match_quantstats():
    returns = QUANTSTATS_RETURNS
    penalty = performance_model.get_autocorrelation_penalty(returns)

    assert penalty == pytest.approx(1.7127888057741563, abs=1e-12)
    assert performance_model.get_sharpe_ratio(returns) / penalty == pytest.approx(
        0.21389937978423693, abs=1e-12
    )
    sortino_ratio = performance_model.get_sortino_ratio(returns)
    assert sortino_ratio / penalty == pytest.approx(0.38637723096820564, abs=1e-12)
    assert sortino_ratio / np.sqrt(2) == pytest.approx(0.4679509613087476, abs=1e-12)


def test_probabilistic_sortino_ratios_match_quantstats():
    returns = QUANTSTATS_RETURNS
    sortino_ratio = performance_model.get_sortino_ratio(returns)

    def probabilistic(ratio):
        return performance_model.get_probabilistic_sharpe_ratio(
            ratio, 0.0, returns.skew(), returns.kurtosis() + 3, len(returns)
        )

    assert probabilistic(sortino_ratio) == pytest.approx(0.9913569550407476, abs=1e-12)
    assert probabilistic(sortino_ratio / np.sqrt(2)) == pytest.approx(
        0.963053346398112, abs=1e-12
    )


def test_autocorrelation_penalty_per_period_and_rolling():
    index = pd.MultiIndex.from_product([["2024", "2025"], range(20)])
    within = pd.DataFrame(
        {"A": np.tile(QUANTSTATS_RETURNS, 2), "B": np.tile(-QUANTSTATS_RETURNS, 2)},
        index=index,
    )
    penalty = performance_model.get_autocorrelation_penalty(within)

    # The absolute autocorrelation is the same for the series and its mirror image.
    np.testing.assert_allclose(penalty.to_numpy(), 1.7127888057741563, atol=1e-12)

    rolling = performance_model.get_autocorrelation_penalty(
        QUANTSTATS_RETURNS.to_frame("A"), window=20
    )
    assert rolling["A"].iloc[-1] == pytest.approx(1.7127888057741563, abs=1e-12)
    assert rolling["A"].iloc[:-1].isna().all()

    # Without variation there is no autocorrelation to penalize.
    assert np.isnan(
        performance_model.get_autocorrelation_penalty(pd.Series([0.01] * 5))
    )


def test_annualized_return_matches_performance_analytics():
    # Return.annualized(portfolio_bacon[, 1]) with 12 observations a year.
    expected = np.prod(1 + BACON_PORTFOLIO) ** (12 / 24) - 1
    assert performance_model.get_annualized_return(
        BACON_PORTFOLIO, 12
    ) == pytest.approx(expected, abs=1e-15)
    assert np.isnan(
        performance_model.get_annualized_return(pd.Series([np.nan, np.nan]), 12)
    )


def test_modified_and_alternative_jensens_alpha_match_performance_analytics():
    beta = _bacon_beta()
    annualized_portfolio = performance_model.get_annualized_return(BACON_PORTFOLIO, 12)
    annualized_benchmark = performance_model.get_annualized_return(BACON_BENCHMARK, 12)
    jensens_alpha = pd.Series([annualized_portfolio - beta * annualized_benchmark])
    benchmark_volatility = BACON_BENCHMARK.std() * np.sqrt(12)

    # AppraisalRatio(method = "modified") and (method = "alternative").
    assert performance_model.get_modified_jensens_alpha(
        jensens_alpha, pd.Series([beta])
    )[0] == pytest.approx(-0.01418576, abs=1e-8)
    assert performance_model.get_modified_jensens_alpha(
        jensens_alpha, pd.Series([beta]), benchmark_volatility
    )[0] == pytest.approx(-0.1066928, abs=1e-7)

    # TreynorRatio(modified = TRUE).
    assert performance_model.get_treynor_ratio(
        pd.Series([annualized_portfolio]), 0.0, pd.Series([beta]), benchmark_volatility
    )[0] == pytest.approx(0.7806747, abs=1e-7)


def test_m2_excess_return_matches_performance_analytics():
    # MSquared uses population standard deviations.
    m2_ratio = performance_model.get_m2_ratio(
        pd.Series([performance_model.get_annualized_return(BACON_PORTFOLIO, 12)]),
        0.0,
        pd.Series([BACON_PORTFOLIO.std(ddof=0) * np.sqrt(12)]),
        BACON_BENCHMARK.std(ddof=0) * np.sqrt(12),
    )
    annualized_benchmark = performance_model.get_annualized_return(BACON_BENCHMARK, 12)

    assert m2_ratio[0] == pytest.approx(0.10062, abs=1e-6)
    assert performance_model.get_m2_excess_return(
        m2_ratio, annualized_benchmark, "geometric"
    )[0] == pytest.approx(-0.01553103, abs=1e-8)
    assert performance_model.get_m2_excess_return(
        m2_ratio, annualized_benchmark, "arithmetic"
    )[0] == pytest.approx(-0.01736344, abs=1e-8)

    with pytest.raises(ValueError, match="arithmetic"):
        performance_model.get_m2_excess_return(m2_ratio, 0.0, "other")


def test_omega_excess_return_matches_performance_analytics():
    minimum_acceptable_return = 0.005

    def downside_deviation(returns):
        shortfall = (returns - minimum_acceptable_return).clip(upper=0)
        return np.sqrt((shortfall**2).mean()) * np.sqrt(12)

    assert performance_model.get_omega_excess_return(
        pd.Series([performance_model.get_annualized_return(BACON_PORTFOLIO, 12)]),
        pd.Series([downside_deviation(BACON_PORTFOLIO)]),
        downside_deviation(BACON_BENCHMARK),
    )[0] == pytest.approx(0.08053795, abs=1e-8)


def test_omega_sharpe_ratio_and_gain_to_pain_ratio_match_the_references():
    # OmegaSharpeRatio(portfolio_bacon[, 1], MAR = 0.005) and quantstats' gain_to_pain_ratio.
    assert performance_model.get_gain_to_pain_ratio(
        BACON_PORTFOLIO, 0.005
    ) == pytest.approx(0.2917933, abs=1e-7)
    assert performance_model.get_gain_to_pain_ratio(
        QUANTSTATS_RETURNS
    ) == pytest.approx(1.3333333333333335, abs=1e-12)

    # The Omega-Sharpe ratio is the Omega ratio minus 1.
    assert performance_model.get_gain_to_pain_ratio(
        BACON_PORTFOLIO, 0.005
    ) == pytest.approx(performance_model.get_omega_ratio(BACON_PORTFOLIO, 0.005) - 1)


def test_d_ratio_matches_performance_analytics():
    assert performance_model.get_d_ratio(BACON_PORTFOLIO) == pytest.approx(
        0.4013329, abs=1e-7
    )
    # No losses gives 0, no gains leaves the ratio undefined.
    assert performance_model.get_d_ratio(pd.Series([0.01, 0.02])) == 0
    assert np.isnan(performance_model.get_d_ratio(pd.Series([-0.01, -0.02])))

    within = pd.DataFrame(
        {"A": np.tile(BACON_PORTFOLIO, 2)},
        index=pd.MultiIndex.from_product([["2024", "2025"], range(24)]),
    )
    np.testing.assert_allclose(
        performance_model.get_d_ratio(within)["A"].to_numpy(), 0.4013329, atol=1e-7
    )


def test_prospect_ratio_matches_performance_analytics():
    gains = BACON_PORTFOLIO.clip(lower=0).sum()
    losses = BACON_PORTFOLIO.clip(upper=0).sum()
    downside_deviation = np.sqrt((BACON_PORTFOLIO.clip(upper=0) ** 2).mean())

    # ProspectRatio(portfolio_bacon[, 1], MAR = 0), as its R code calculates it.
    assert performance_model.get_prospect_ratio(BACON_PORTFOLIO) == pytest.approx(
        (gains + 2.25 * losses) / (downside_deviation * 24), abs=1e-12
    )

    # With a MAR the documented formula subtracts the MAR from the average.
    shortfall = (BACON_PORTFOLIO - 0.05).clip(upper=0)
    assert performance_model.get_prospect_ratio(BACON_PORTFOLIO, 0.05) == pytest.approx(
        ((gains + 2.25 * losses) / 24 - 0.05) / np.sqrt((shortfall**2).mean()),
        abs=1e-12,
    )
    assert np.isnan(performance_model.get_prospect_ratio(pd.Series([0.01, 0.02])))


def test_modified_burke_ratio_divides_by_the_root_mean_square_drawdown():
    excess_returns = pd.Series([0.1, -0.05])
    burke_drawdown_measure = pd.Series([0.2, 0.4])

    np.testing.assert_allclose(
        performance_model.get_burke_ratio(
            excess_returns, burke_drawdown_measure, n_observations=24
        ),
        performance_model.get_burke_ratio(excess_returns, burke_drawdown_measure)
        * np.sqrt(24),
    )


def test_return_statistics_extensions_match_quantstats_and_performance_analytics():
    statistics = performance_model.get_return_statistics(QUANTSTATS_RETURNS)

    # quantstats 0.0.86 on the same returns (no return is exactly 0, so the win
    # conventions agree), and PerformanceAnalytics' mean.stderr, mean.LCL and mean.UCL.
    expected = {
        "Expectancy": 0.003,
        "Best Return": 0.015,
        "Worst Return": -0.013,
        "Exposure": 1.0,
        "CPC Index": 1.9055555555555557,
        "Profit Ratio": 0.6765285996055227,
        "Outlier Win Ratio": 1.8100952380952378,
        "Outlier Loss Ratio": 1.963111111111111,
        "Risk of Ruin": 3.4015902868396653e-14,
        "Mean Return": 0.003,
        "Standard Error": 0.001831019273232089,
        "Lower Confidence Limit": -0.0008323673829599815,
        "Upper Confidence Limit": 0.006832367382959982,
    }
    for statistic, value in expected.items():
        assert statistics[statistic] == pytest.approx(value, rel=1e-12, abs=1e-15)

    # The Risk-Adjusted Return is the annualized return over the exposure.
    annualized_return = (1 + QUANTSTATS_RETURNS).prod() ** (252 / 20) - 1
    assert statistics["Risk-Adjusted Return"] == pytest.approx(annualized_return)


def test_return_statistics_leave_zero_returns_out_of_the_wins():
    returns = pd.Series([0.02, 0.0, -0.01, 0.0, 0.03, -0.02])
    statistics = performance_model.get_return_statistics(returns, periods_per_year=12)

    assert statistics["Exposure"] == pytest.approx(4 / 6)
    # Average win 0.025 over 2 wins, average loss 0.015 over 2 losses.
    assert statistics["Profit Ratio"] == pytest.approx((0.025 / 2) / (0.015 / 2))
    assert statistics["Outlier Win Ratio"] == pytest.approx(
        returns.quantile(0.99) / 0.025
    )
    assert statistics["Risk-Adjusted Return"] == pytest.approx(
        ((1 + returns).prod() ** (12 / 6) - 1) / (4 / 6)
    )


def test_capture_ratio_methods_match_empyrical_and_performance_analytics():
    portfolio, benchmark = BACON_PORTFOLIO, BACON_BENCHMARK

    # empyrical 0.7.3 on the monthly returns (period="monthly", 12 a year).
    assert performance_model.get_upside_capture_ratio(
        portfolio, benchmark, method="annualized", periods_per_year=12
    ) == pytest.approx(0.9478779356059414, rel=1e-12)
    assert performance_model.get_downside_capture_ratio(
        portfolio, benchmark, method="annualized", periods_per_year=12
    ) == pytest.approx(0.9864739725390025, rel=1e-12)
    assert performance_model.get_capture_ratio(
        portfolio, benchmark, method="annualized", periods_per_year=12
    ) == pytest.approx(0.8787532646895315, rel=1e-12)

    # PerformanceAnalytics' UpDownRatios; its Down Capture also counts the month the
    # benchmark returned 0, which the Toolkit leaves out.
    assert performance_model.get_upside_capture_ratio(
        portfolio, benchmark, method="arithmetic"
    ) == pytest.approx(0.9592668024439919, rel=1e-12)
    down = benchmark < 0
    assert performance_model.get_downside_capture_ratio(
        portfolio, benchmark, method="arithmetic"
    ) == pytest.approx(portfolio[down].sum() / benchmark[down].sum(), rel=1e-12)
    assert performance_model.get_up_down_number(
        portfolio, benchmark, market="up"
    ) == pytest.approx(0.9333333333333333)
    assert performance_model.get_up_down_number(
        portfolio, benchmark, market="down"
    ) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="'geometric', 'annualized' or 'arithmetic'"):
        performance_model.get_capture_ratio(portfolio, benchmark, method="median")


def test_rolling_capture_ratio_uses_the_returns_of_each_window():
    portfolio = BACON_PORTFOLIO.to_frame("Portfolio")
    rolling = performance_model.get_capture_ratio(
        portfolio, BACON_BENCHMARK, window=12, side="up_down"
    )

    assert rolling["Portfolio"].iloc[:11].isna().all()
    expected = performance_model.get_upside_capture_ratio(
        BACON_PORTFOLIO.iloc[-12:], BACON_BENCHMARK.iloc[-12:]
    ) / performance_model.get_downside_capture_ratio(
        BACON_PORTFOLIO.iloc[-12:], BACON_BENCHMARK.iloc[-12:]
    )
    assert rolling["Portfolio"].iloc[-1] == pytest.approx(expected)


def test_up_and_down_market_win_rates_match_performance_analytics():
    portfolio, benchmark = BACON_PORTFOLIO, BACON_BENCHMARK

    # PerformanceAnalytics' Up and Down Percent; empyrical's batting average counts the
    # month the benchmark returned 0 as an up market (0.6875 there).
    assert performance_model.get_win_rate(portfolio, benchmark) == pytest.approx(
        0.7083333333333334
    )
    assert performance_model.get_win_rate(
        portfolio, benchmark, market="up"
    ) == pytest.approx(0.7333333333333333)
    assert performance_model.get_win_rate(
        portfolio, benchmark, market="down"
    ) == pytest.approx(0.75)


def test_outperformance_probability_matches_performance_analytics():
    probability = performance_model.get_outperformance_probability(
        BACON_PORTFOLIO.to_frame("Portfolio"), BACON_BENCHMARK, windows=[1, 3, 6, 12]
    )["Portfolio"]

    # PerformanceAnalytics' table.ProbOutPerformance, transcribed and checked against
    # its recorded output on the edhec data.
    assert probability.to_list() == pytest.approx(
        [
            0.7083333333333334,
            0.6363636363636364,
            0.6842105263157895,
            0.46153846153846156,
        ]
    )

    # Without a benchmark each window return is compared with 0.
    absolute = performance_model.get_outperformance_probability(
        BACON_PORTFOLIO.to_frame("Portfolio"), windows=[1]
    )
    assert absolute.loc[1, "Portfolio"] == pytest.approx(
        (BACON_PORTFOLIO > 0).sum() / (BACON_PORTFOLIO != 0).sum()
    )


def test_bull_bear_beta_matches_performance_analytics_and_empyrical():
    results = performance_model.get_bull_bear_beta(
        BACON_PORTFOLIO, BACON_BENCHMARK, periods_per_year=12
    )

    # PerformanceAnalytics' CAPM.beta.bull and CAPM.beta.bear, and empyrical's
    # up_alpha_beta and down_alpha_beta (period="monthly").
    assert results["Bull Beta"] == pytest.approx(1.0743229309870772, rel=1e-12)
    assert results["Bear Beta"] == pytest.approx(0.9813284687198197, rel=1e-12)
    assert results["Timing Ratio"] == pytest.approx(
        1.0743229309870772 / 0.9813284687198197
    )
    assert results["Bull Alpha"] == pytest.approx(-0.04426955, abs=1e-8)
    assert results["Bear Alpha"] == pytest.approx(-0.00100136, abs=1e-8)


def test_beta_fragility_heuristic_matches_empyrical():
    # empyrical's beta_fragility_heuristic on Bacon's monthly returns.
    assert performance_model.get_beta_fragility_heuristic(
        BACON_PORTFOLIO, BACON_BENCHMARK
    ) == pytest.approx(-0.0021866666666666666, abs=1e-15)
    assert np.isnan(
        performance_model.get_beta_fragility_heuristic(
            pd.Series([0.01, 0.02]), pd.Series([0.01, 0.03])
        )
    )


def test_stability_and_r_squared_match_empyrical_and_quantstats():
    assert performance_model.get_stability_of_returns(BACON_PORTFOLIO) == pytest.approx(
        0.5879117296650359, rel=1e-12
    )
    assert performance_model.get_r_squared(
        BACON_PORTFOLIO, BACON_BENCHMARK
    ) == pytest.approx(0.9397088580814588, rel=1e-12)

    frame = pd.DataFrame({"A": BACON_PORTFOLIO, "B": BACON_PORTFOLIO * 2})
    rolling = performance_model.get_r_squared(frame, BACON_BENCHMARK, window=24)
    assert rolling.iloc[-1].to_list() == pytest.approx([0.9397088580814588] * 2)


def test_risk_decomposition_matches_performance_analytics():
    n = len(BACON_PORTFOLIO)
    risks = performance_model.get_risk_decomposition(
        BACON_PORTFOLIO, BACON_BENCHMARK, observations=12
    )

    # PerformanceAnalytics' SystematicRisk (0.132806); its SpecificRisk (0.03293109)
    # divides by n where the Toolkit divides by n - 1.
    assert risks["Systematic Risk"] == pytest.approx(0.132806, abs=1e-6)
    assert risks["Specific Risk"] == pytest.approx(
        0.03293109 * np.sqrt(n / (n - 1)), abs=1e-7
    )
    assert risks["Total Risk"] == pytest.approx(BACON_PORTFOLIO.std() * np.sqrt(12))


def test_capital_market_line_and_regression_epsilon_match_performance_analytics():
    risk_free = pd.Series(0.0, index=BACON_BENCHMARK.index)
    line = performance_model.get_capital_market_line(
        BACON_PORTFOLIO, BACON_BENCHMARK, risk_free
    )

    # PerformanceAnalytics' CAPM.CML.slope, and the reciprocal of its CAPM.SML.slope.
    assert line["Capital Market Line Slope"] == pytest.approx(0.26162488955600705)
    assert line["Security Market Line Slope"] == pytest.approx(0.010041666666666667)
    assert line["Capital Market Line Return"] == pytest.approx(
        0.26162488955600705 * BACON_PORTFOLIO.std()
    )

    # PerformanceAnalytics' CAPM.epsilon (-0.01313932) uses the annualized returns.
    epsilon = performance_model.get_regression_epsilon(
        BACON_PORTFOLIO,
        BACON_BENCHMARK,
        (1 + BACON_PORTFOLIO).prod() ** (12 / 24) - 1,
        (1 + BACON_BENCHMARK).prod() ** (12 / 24) - 1,
    )
    assert epsilon == pytest.approx(-0.013139323809325809, rel=1e-10)


def test_log_returns_add_up_to_the_cumulative_return():
    log_returns = performance_model.get_log_returns(QUANTSTATS_RETURNS)

    assert log_returns.sum() == pytest.approx(np.log((1 + QUANTSTATS_RETURNS).prod()))


def test_geltner_returns_match_performance_analytics():
    # Return.Geltner: (R_t - ρ1 R_t-1) / (1 - ρ1) with ρ1 the lag-1 sample autocorrelation.
    returns = BACON_PORTFOLIO
    deviations = returns - returns.mean()
    autocorrelation = (deviations * deviations.shift(1)).sum() / (deviations**2).sum()
    expected = (returns - autocorrelation * returns.shift(1)) / (1 - autocorrelation)

    unsmoothed = performance_model.get_geltner_returns(returns)

    assert np.isnan(unsmoothed.iloc[0])
    np.testing.assert_allclose(unsmoothed.iloc[1:], expected.iloc[1:], atol=1e-15)

    # Each column on its own history: a later start leaves the other column unchanged.
    frame = pd.DataFrame({"A": returns, "B": returns.where(returns.index >= 5)})
    np.testing.assert_allclose(
        performance_model.get_geltner_returns(frame)["A"].iloc[1:], expected.iloc[1:]
    )
    assert performance_model.get_geltner_returns(frame)["B"].iloc[:6].isna().all()


def test_smoothing_index_matches_the_maximum_likelihood_of_statsmodels():
    rng = np.random.default_rng(5)
    innovations = rng.normal(0, 0.02, 122)
    returns = pd.Series(
        0.6 * innovations[2:] + 0.3 * innovations[1:-1] + 0.1 * innovations[:-2]
    )

    smoothing = performance_model.get_smoothing_index(returns)

    # statsmodels ARIMA(0, 0, 2) without a mean, exact maximum likelihood, normalized.
    np.testing.assert_allclose(
        smoothing[["Theta 0", "Theta 1", "Theta 2"]],
        [0.61095141, 0.31326356, 0.07578503],
        atol=1e-6,
    )
    assert smoothing["Smoothing Index"] == pytest.approx(0.47713905, abs=1e-6)

    # Returns without smoothing have an index close to 1, too few returns give NaN.
    assert (
        performance_model.get_smoothing_index(pd.Series(rng.normal(0, 0.02, 500)))[
            "Smoothing Index"
        ]
        > 0.8
    )  # noqa: PLR2004
    assert performance_model.get_smoothing_index(pd.Series([0.01] * 5)).isna().all()


def test_serenity_index_is_the_ulcer_performance_index_over_the_pitfall():
    excess_returns = pd.Series([0.2, -0.1])
    ulcer_index = pd.Series([0.05, 0.1])
    conditional_drawdown_at_risk = pd.Series([-0.15, -0.3])
    volatility = pd.Series([0.2, 0.3])

    serenity_index = performance_model.get_serenity_index(
        excess_returns, ulcer_index, conditional_drawdown_at_risk, volatility
    )

    np.testing.assert_allclose(
        serenity_index, [0.2 / (0.05 * 0.75), -0.1 / (0.1 * 1.0)]
    )


def test_risk_measure_ratio_matches_skfolio():
    # skfolio 1.14: mean(returns) / measure, with the compounded drawdowns.
    expected = {
        "mad": 0.44776119402985076,
        "lpm": 0.8955223880597015,
        "gmd": 0.31456953642384106,
        "worst": 0.23076923076923078,
        "max_drawdown": 0.2307692307692286,
        "average_drawdown": 0.9935866706480108,
        "ulcer": 0.6022708609459754,
    }

    for risk_measure, value in expected.items():
        assert performance_model.get_risk_measure_ratio(
            QUANTSTATS_RETURNS, QUANTSTATS_RETURNS, risk_measure
        ) == pytest.approx(value, rel=1e-10), risk_measure

    within = pd.DataFrame(
        {"A": np.tile(QUANTSTATS_RETURNS, 2)},
        index=pd.MultiIndex.from_product([["2024", "2025"], range(20)]),
    )
    np.testing.assert_allclose(
        performance_model.get_risk_measure_ratio(within, within, "mad")["A"],
        expected["mad"],
    )

    with pytest.raises(ValueError, match="risk_measure must be one of"):
        performance_model.get_risk_measure(QUANTSTATS_RETURNS, "other")


def test_trailing_returns_match_ffn():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2021-01-01", "2024-06-28")
    prices = pd.Series(
        100 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(dates))), index=dates
    )

    trailing_returns = performance_model.get_trailing_returns(prices)

    # ffn 1.4.1 PerformanceStats: mtd, three_month, six_month, ytd, one_year and incep.
    expected = {
        "Month to Date": 0.06527024676855553,
        "3 Months": 0.1084229852355938,
        "6 Months": 0.21728091531533877,
        "Year to Date": 0.21728091531533877,
        "1 Year": 0.2407967411419052,
        "Since Inception (Annualized)": 0.1794947378121643,
    }
    for label, value in expected.items():
        assert trailing_returns[label] == pytest.approx(value, rel=1e-10), label

    # The 3 year window starts at the close of June 2021, annualized over its calendar days.
    start = prices.loc[:"2021-06-30"]
    years = (dates[-1] - start.index[-1]).days / 365.25
    assert trailing_returns["3 Years (Annualized)"] == pytest.approx(
        (prices.iloc[-1] / start.iloc[-1]) ** (1 / years) - 1
    )
    assert np.isnan(trailing_returns["5 Years (Annualized)"])


def test_calendar_returns_compound_the_months_and_years():
    dates = pd.bdate_range("2023-11-15", "2025-02-14").to_period("D")
    returns = pd.Series(np.linspace(-0.01, 0.012, len(dates)), index=dates)

    calendar = performance_model.get_calendar_returns(returns)
    monthly = performance_model.get_returns(returns, "monthly")
    yearly = performance_model.get_returns(returns, "yearly")

    assert list(calendar.columns) == [*performance_model.MONTH_NAMES, "Year"]
    assert calendar.loc["2024", "Mar"] == pytest.approx(monthly.loc["2024-03"])
    np.testing.assert_allclose(calendar["Year"], yearly)
    # Months without returns are NaN rather than 0.
    assert calendar.loc["2023", "Jan":"Oct"].isna().all()
    assert calendar.loc["2025", "Mar":"Dec"].isna().all()


def test_bootstrap_statistics_summarize_the_simulated_histories():
    returns = QUANTSTATS_RETURNS
    excess_returns = QUANTSTATS_RETURNS - 0.0001

    statistics = performance_model.get_bootstrap_statistics(
        returns, excess_returns, 252, n_bootstrap=500, seed=3
    )

    positions = performance_model.get_bootstrap_samples(20, 500, "resample", seed=3)
    samples = returns.to_numpy()[positions]
    cumulative = np.prod(1 + samples, axis=1) - 1

    assert statistics.loc["Cumulative Return", "Mean"] == pytest.approx(
        cumulative.mean()
    )
    assert statistics.loc["Cumulative Return", "5%"] == pytest.approx(
        np.percentile(cumulative, 5)
    )
    assert statistics.loc["Annualized Volatility", "Median"] == pytest.approx(
        np.median(samples.std(axis=1, ddof=1) * np.sqrt(252))
    )

    # Shuffling keeps the returns, so only the order (and the drawdown) changes.
    shuffled = performance_model.get_bootstrap_statistics(
        returns, excess_returns, 252, n_bootstrap=500, method="shuffle"
    )
    assert shuffled.loc["Cumulative Return", "Std"] == pytest.approx(0, abs=1e-15)
    assert shuffled.loc["Sharpe Ratio", "Std"] == pytest.approx(0, abs=1e-12)
    assert shuffled.loc["Maximum Drawdown", "Std"] > 0

    with pytest.raises(ValueError, match="'resample' or 'shuffle'"):
        performance_model.get_bootstrap_samples(20, 10, "other")


def test_forecast_cone_matches_pyfolio_on_the_same_paths():
    returns = QUANTSTATS_RETURNS

    cone = performance_model.get_forecast_cone(returns, horizon=5, simulations=300)

    # pyfolio's summarize_paths: the mean and standard deviation of the cumulative paths.
    positions = performance_model.get_bootstrap_samples(
        20, 300, "resample", length=5, seed=0
    )
    wealth = np.cumprod(1 + returns.to_numpy()[positions], axis=1)
    np.testing.assert_allclose(cone["Mean"], wealth.mean(axis=0))
    np.testing.assert_allclose(
        cone["Upper 1.5"], wealth.mean(axis=0) + 1.5 * wealth.std(axis=0, ddof=1)
    )
    assert list(cone.columns) == [
        "Lower 2",
        "Lower 1.5",
        "Lower 1",
        "Mean",
        "Upper 1",
        "Upper 1.5",
        "Upper 2",
    ]


def test_probability_of_backtest_overfitting_matches_ffn():
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(rng.normal(0.0005, 0.01, (480, 6)))

    result = performance_model.get_probability_of_backtest_overfitting(returns, 8)

    # ffn 1.4.1 calc_prob_backtest_overfitting(n_blocks=8, full_output=True).
    assert result["Probability of Backtest Overfitting"] == pytest.approx(
        0.4857142857142857, abs=1e-12
    )
    assert result["Average Out-of-Sample Rank"] == pytest.approx(
        0.5061224489795918, abs=1e-12
    )
    assert result["Combinations"] == 70  # noqa: PLR2004

    # A strategy that is best in every block is never overfit.
    returns[0] = returns[0] + 0.01
    assert (
        performance_model.get_probability_of_backtest_overfitting(returns, 8)[
            "Probability of Backtest Overfitting"
        ]
        == 0
    )

    with pytest.raises(ValueError, match="even"):
        performance_model.get_probability_of_backtest_overfitting(returns, 7)


def test_probabilistic_momentum_matches_ffn():
    difference = QUANTSTATS_RETURNS - BACON_PORTFOLIO.iloc[:20]
    information_ratio = difference.mean() / difference.std()

    # ffn 1.4.1 calc_prob_mom(QUANTSTATS_RETURNS, the first 20 returns of portfolio_bacon).
    assert performance_model.get_probabilistic_momentum(
        information_ratio, 20
    ) == pytest.approx(0.14496828978990775, abs=1e-12)

    probability = performance_model.get_probabilistic_momentum(
        pd.Series([0.0, information_ratio]), pd.Series([20, 1])
    )
    assert probability.iloc[0] == pytest.approx(0.5)
    assert np.isnan(probability.iloc[1])


def test_rolling_factor_regression_matches_pyfolio():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2021-01-01", periods=60)
    factors = pd.DataFrame(
        rng.normal(0, 0.01, (60, 2)), index=dates, columns=["Mkt-RF", "SMB"]
    )
    excess_returns = pd.Series(
        0.0002 + factors @ [1.1, 0.3] + rng.normal(0, 0.005, 60), index=dates
    )

    regression = performance_model.get_factor_regression(
        excess_returns, factors, window=30
    )

    # pyfolio 0.10.2 rolling_regression(rolling_window=29), whose windows hold 30 days.
    assert regression.index[0] == pd.Timestamp("2021-02-11")
    np.testing.assert_allclose(
        regression.iloc[-1][["Intercept", "Mkt-RF Slope", "SMB Slope"]],
        [5.146401340738521e-05, 1.1124490824735467, 0.3731852476859474],
        rtol=1e-9,
    )
    # A complete window equals one ordinary least squares fit with its R Squared.
    window = slice(10, 40)
    design = np.column_stack([np.ones(30), factors.iloc[window]])
    coefficients, residuals = np.linalg.lstsq(
        design, excess_returns.iloc[window], rcond=None
    )[:2]
    np.testing.assert_allclose(
        regression.iloc[10][["Intercept", "Mkt-RF Slope", "SMB Slope"]], coefficients
    )
    total = (
        (excess_returns.iloc[window] - excess_returns.iloc[window].mean()) ** 2
    ).sum()
    assert regression.iloc[10]["R Squared"] == pytest.approx(1 - residuals[0] / total)


def test_get_variability_weighted_return_by_hand():
    # Constant returns follow the exponential path exactly, so the VWR is the annualized return.
    constant = pd.Series([0.001] * 50)
    assert performance_model.get_variability_weighted_return(
        constant, 252
    ) == pytest.approx(1.001**252 - 1)

    returns = pd.Series([0.02, -0.01, 0.03, -0.02])
    values = np.cumprod(1 + returns.to_numpy())
    average = np.log(values[-1]) / 4
    deviations = values / np.exp(average * np.arange(1, 5)) - 1
    expected = (np.exp(average * 12) - 1) * (1 - (deviations.std(ddof=1) / 0.2) ** 2)
    assert performance_model.get_variability_weighted_return(
        returns, 12
    ) == pytest.approx(expected)

    frame = pd.DataFrame({"A": returns, "B": [0.01, np.nan, 0.01, 0.01]})
    result = performance_model.get_variability_weighted_return(
        frame, pd.Series({"A": 12.0, "B": 12.0})
    )
    assert result["A"] == pytest.approx(expected)
    assert result["B"] == pytest.approx(1.01**12 - 1)
    assert np.isnan(performance_model.get_variability_weighted_return(pd.Series([0.1])))


def test_get_return_concentration_by_hand():
    dates = pd.period_range("2024-01-30", periods=6, freq="D")
    returns = pd.Series([0.01, 0.03, 0.0, -0.01, -0.01, -0.02], index=dates)
    concentration = performance_model.get_return_concentration(returns)

    def hhi(values):
        weights = np.array(values) / np.sum(values)
        count = len(values)
        return ((weights**2).sum() - 1 / count) / (1 - 1 / count)

    assert concentration["Positive Concentration"] == pytest.approx(
        hhi([0.01, 0.03, 0.0])
    )
    assert concentration["Negative Concentration"] == pytest.approx(
        hhi([0.01, 0.01, 0.02])
    )
    # Two returns in January and four in February: too few months for a concentration.
    assert np.isnan(concentration["Time Concentration"])
    # Equal returns are not concentrated at all.
    assert performance_model.get_return_concentration(
        pd.Series([0.01] * 3, index=dates[:3])
    )["Positive Concentration"] == pytest.approx(0)


def test_get_strategy_risk_by_hand():
    # Symmetric bets: SR = (2p - 1) / (2 x SQRT(p (1 - p))) x SQRT(n).
    sharpe_ratio = performance_model.get_binomial_sharpe_ratio(0.01, -0.01, 0.55, 252)
    assert sharpe_ratio == pytest.approx(0.1 / (2 * np.sqrt(0.2475)) * np.sqrt(252))

    # The implied precision and frequency give back the target Sharpe ratio.
    precision = performance_model.get_implied_precision(0.02, -0.01, 52, 2.0)
    assert performance_model.get_binomial_sharpe_ratio(
        0.02, -0.01, precision, 52
    ) == pytest.approx(2.0)
    frequency = performance_model.get_implied_frequency(0.02, -0.01, 0.4, 2.0)
    assert performance_model.get_binomial_sharpe_ratio(
        0.02, -0.01, 0.4, frequency
    ) == pytest.approx(2.0)
    # A losing bet on average reaches no positive target.
    assert np.isnan(performance_model.get_implied_frequency(0.01, -0.02, 0.5, 2.0))

    returns = pd.Series([0.02, -0.01, 0.03, 0.0, -0.02, 0.01, 0.015, -0.005])
    risk = performance_model.get_strategy_risk(returns, frequency=52)
    assert risk["Precision"] == pytest.approx(4 / 8)
    assert risk["Average Gain"] == pytest.approx(0.075 / 4)
    assert risk["Average Loss"] == pytest.approx(-0.035 / 4)
    implied = performance_model.get_implied_precision(0.075 / 4, -0.035 / 4, 52, 2.0)
    assert risk["Implied Precision"] == pytest.approx(implied)
    assert risk["Probability of Failure"] == pytest.approx(
        stats.norm.cdf((implied - 0.5) / np.sqrt(0.25 / 8))
    )
    assert performance_model.get_strategy_risk(pd.Series([0.01, 0.02])).isna().all()


def sharpe_difference_returns() -> tuple[pd.Series, pd.Series]:
    """Two correlated, fat-tailed and serially correlated excess return series."""
    shocks = np.random.default_rng(3).standard_t(5, (751, 2)) * 0.01
    first = 0.0005 + shocks[1:, 0] + 0.3 * shocks[:-1, 0]
    second = 0.0002 + 0.6 * shocks[1:, 0] + 0.8 * shocks[1:, 1]

    return pd.Series(first), pd.Series(second)


@pytest.mark.parametrize("lags", [0, 3, 8])
def test_ledoit_wolf_sharpe_difference_matches_a_hac_delta_method(lags):
    # Ledoit and Wolf (2008, section 3.1): the long-run covariance of the moments from
    # statsmodels' Bartlett estimate (a sum, so divided by n) times n / (n - 4), and the
    # gradient of the Sharpe difference taken numerically.
    sandwich = pytest.importorskip("statsmodels.stats.sandwich_covariance")
    first, second = sharpe_difference_returns()
    a, b = first.to_numpy(), second.to_numpy()
    n = len(a)
    moments = np.array([a.mean(), b.mean(), (a**2).mean(), (b**2).mean()])
    deviations = np.column_stack(
        [a - moments[0], b - moments[1], a**2 - moments[2], b**2 - moments[3]]
    )
    covariance = sandwich.S_hac_simple(deviations, nlags=lags) / (n - 4)

    def difference(v):
        return v[0] / np.sqrt(v[2] - v[0] ** 2) - v[1] / np.sqrt(v[3] - v[1] ** 2)

    step = np.eye(4) * 1e-7
    gradient = np.array(
        [(difference(moments + s) - difference(moments - s)) / 2e-7 for s in step]
    )
    standard_error = np.sqrt(gradient @ covariance @ gradient / n)

    result = performance_model.get_sharpe_ratio_difference(
        first, second, "ledoit_wolf", lags=lags
    )

    assert result["Sharpe Difference"] == pytest.approx(difference(moments), abs=1e-14)
    assert result["Standard Error"] == pytest.approx(standard_error, rel=1e-7)
    assert result["P-Value"] == pytest.approx(
        2 * stats.norm.sf(abs(difference(moments) / standard_error)), rel=1e-6
    )


def test_memmel_sharpe_difference_is_the_normal_delta_method():
    # Jobson and Korkie (1981) with Memmel (2003): Var(SR_i) = (1 + SR_i^2 / 2) / n and
    # Cov(SR_a, SR_b) = (ρ + SR_a·SR_b·ρ^2 / 2) / n under joint normality.
    first, second = sharpe_difference_returns()
    n = len(first)
    sharpe_a = first.mean() / first.std(ddof=1)
    sharpe_b = second.mean() / second.std(ddof=1)
    correlation = first.corr(second)
    covariance = (
        np.array(
            [
                [
                    1 + sharpe_a**2 / 2,
                    correlation + sharpe_a * sharpe_b * correlation**2 / 2,
                ],
                [
                    correlation + sharpe_a * sharpe_b * correlation**2 / 2,
                    1 + sharpe_b**2 / 2,
                ],
            ]
        )
        / n
    )
    standard_error = np.sqrt(np.array([1, -1]) @ covariance @ np.array([1, -1]))

    result = performance_model.get_sharpe_ratio_difference(first, second)

    assert result["Sharpe Difference"] == pytest.approx(sharpe_a - sharpe_b, abs=1e-14)
    assert result["Standard Error"] == pytest.approx(standard_error, rel=1e-12)


def test_sharpe_difference_tests_hold_their_size_on_equal_sharpe_ratios():
    # Jointly normal returns with the same Sharpe ratio: both tests reject at 5% about 5% of
    # the time (400 draws, so within 2.5% to 8%).
    rng = np.random.default_rng(7)
    rejections = {"memmel": 0, "ledoit_wolf": 0}

    for _ in range(400):
        draws = rng.multivariate_normal([0.05, 0.05], [[1, 0.5], [0.5, 1]], size=250)
        for method in rejections:
            result = performance_model.get_sharpe_ratio_difference(
                pd.Series(draws[:, 0]), pd.Series(draws[:, 1]), method
            )
            rejections[method] += result["P-Value"] < 0.05  # noqa: PLR2004

    assert all(0.025 <= count / 400 <= 0.08 for count in rejections.values())


def test_sharpe_difference_test_per_period():
    first, second = sharpe_difference_returns()
    index = pd.MultiIndex.from_arrays(
        [
            pd.period_range("2020-01-01", periods=750, freq="D").asfreq("Y"),
            pd.period_range("2020-01-01", periods=750, freq="D"),
        ]
    )
    returns = pd.DataFrame({"A": first.to_numpy()}, index=index)
    benchmark = pd.Series(second.to_numpy(), index=index)

    result = performance_model.get_sharpe_ratio_difference_test(returns, benchmark)

    assert list(result.index.astype(str)) == ["2020", "2021", "2022"]
    assert result.loc[pd.Period("2021", "Y"), ("A", "Z-Statistic")] == pytest.approx(
        performance_model.get_sharpe_ratio_difference(
            returns.loc[pd.Period("2021", "Y"), "A"],
            benchmark.loc[pd.Period("2021", "Y")],
        )["Z-Statistic"]
    )
    # Fewer than five shared dates give no test.
    assert (
        performance_model.get_sharpe_ratio_difference(first[:4], second[:4])
        .isna()
        .all()
    )

    with pytest.raises(ValueError, match="'memmel' or 'ledoit_wolf'"):
        performance_model.get_sharpe_ratio_difference_test(returns, benchmark, "jk")


def test_sharpe_ratio_difference_is_nan_without_variation():
    # A constant return has no Sharpe ratio: its standard deviation is rounding noise, which
    # gave the Memmel difference a value in the 1e15.
    other = pd.Series(np.random.default_rng(4).normal(0.0005, 0.01, 500))
    constant = pd.Series(np.full(500, 0.0003))

    for method in ("memmel", "ledoit_wolf"):
        result = performance_model.get_sharpe_ratio_difference(constant, other, method)
        assert result.isna().all()


def test_rolling_capture_ratio_of_a_series_is_a_series():
    returns = pd.read_pickle("tests/datasets/historical_dataset.pickle")["Return"].iloc[
        1:
    ]

    single = performance_model.get_capture_ratio(
        returns["AAPL"], returns["Benchmark"], window=63
    )
    as_frame = performance_model.get_capture_ratio(
        returns[["AAPL"]], returns["Benchmark"], window=63
    )

    assert isinstance(single, pd.Series)
    pd.testing.assert_series_equal(single, as_frame["AAPL"])


def test_regression_ratios_of_too_few_observations_are_nan():
    index = pd.period_range("2020-01", periods=24, freq="M")
    rng = np.random.default_rng(3)
    # The benchmark repeats every return once, so every window of two has no variance.
    benchmark = pd.Series(np.repeat(rng.normal(0.01, 0.04, 12), 2), index=index)
    excess = pd.DataFrame({"A": 0.8 * benchmark + rng.normal(0, 0.02, 24)}, index=index)

    beta = performance_model.get_beta(excess, benchmark, window=2)
    assert not np.isinf(beta.to_numpy(float)).any()
    assert beta["A"].iloc[1::2].isna().all()

    # A line through two returns fits them exactly, so there is no residual risk.
    jensens_alpha = pd.DataFrame({"A": rng.normal(0.01, 0.01, 24)}, index=index)
    appraisal_ratio = performance_model.get_appraisal_ratio(
        jensens_alpha,
        excess_returns=excess,
        benchmark_excess_returns=benchmark,
        window=2,
    )
    assert appraisal_ratio["A"].isna().all()

    # Three returns per side clear the default minimum but not a minimum of 20.
    days = pd.Series([0.01, 0.02, 0.015, -0.01, -0.02, -0.03])
    asset = 1.2 * days + pd.Series([0.001, -0.002, 0.003, 0.002, -0.001, 0.0])
    assert not np.isnan(performance_model.get_bull_bear_beta(asset, days)["Bull Beta"])
    twenty_days = performance_model.get_bull_bear_beta(
        asset, days, minimum_observations=20
    )
    assert all(np.isnan(value) for value in twenty_days.values())


def test_a_window_of_no_observations_raises():
    rng = np.random.default_rng(2)
    returns = pd.DataFrame({"A": rng.normal(0, 0.01, 60)})
    styles = pd.DataFrame(rng.normal(0, 0.01, (60, 2)), columns=["B", "C"])

    for window in (0, -1):
        with pytest.raises(InvalidArgumentError, match="window is a number"):
            performance_model.get_kelly_criterion(returns, window=window)
        with pytest.raises(InvalidArgumentError, match="window is a number"):
            performance_model.get_omega_ratio(returns, window=window)
        with pytest.raises(InvalidArgumentError, match="window is a number"):
            performance_model.get_style_analysis(returns["A"], styles, window=window)
    # An Omega ratio of one return is defined; a Kelly criterion needs wins and losses.
    assert performance_model.get_omega_ratio(returns, window=1).notna().any().any()
    with pytest.raises(InvalidArgumentError, match="at least 2"):
        performance_model.get_kelly_criterion(returns, window=1)


def test_autocorrelation_penalty_of_two_returns_does_not_warn():
    returns = pd.Series(np.random.default_rng(4).normal(0, 0.01, 40))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        penalty = performance_model.get_autocorrelation_penalty(returns, window=2)

    assert len(penalty) == len(returns)
