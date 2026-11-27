"""Forecast Evaluation Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.econometrics.forecasting import forecast_evaluation_model

SIGNIFICANCE_LEVEL = 0.05


def test_get_diebold_mariano_test_better_forecast(recorder):
    rng = np.random.default_rng(9)
    n = 500
    actual = rng.standard_normal(n)
    forecast_good = actual + rng.standard_normal(n) * 0.1
    forecast_bad = actual + rng.standard_normal(n) * 1.0
    result = forecast_evaluation_model.get_diebold_mariano_test(
        pd.Series(actual), pd.Series(forecast_good), pd.Series(forecast_bad)
    )
    assert result["Diebold-Mariano Statistic"] < 0
    assert result["P-Value"] < SIGNIFICANCE_LEVEL
    recorder.capture(result.round(4))


def test_get_diebold_mariano_test_equal_forecasts(recorder):
    rng = np.random.default_rng(9)
    n = 500
    actual = rng.standard_normal(n)
    forecast_a = actual + rng.standard_normal(n) * 0.5
    forecast_b = actual + rng.standard_normal(n) * 0.5
    recorder.capture(
        forecast_evaluation_model.get_diebold_mariano_test(
            pd.Series(actual), pd.Series(forecast_a), pd.Series(forecast_b)
        ).round(4)
    )


def test_get_diebold_mariano_test_dataframe(recorder):
    rng = np.random.default_rng(9)
    n = 500
    actual = rng.standard_normal(n)
    forecast_good = actual + rng.standard_normal(n) * 0.1
    forecast_bad = actual + rng.standard_normal(n) * 1.0
    actual_df = pd.DataFrame({"AAPL": actual, "MSFT": actual})
    forecast_a_df = pd.DataFrame({"AAPL": forecast_good, "MSFT": forecast_good})
    forecast_b_df = pd.DataFrame({"AAPL": forecast_bad, "MSFT": forecast_bad})
    recorder.capture(
        forecast_evaluation_model.get_diebold_mariano_test(
            actual_df, forecast_a_df, forecast_b_df
        ).round(4)
    )


def test_get_diebold_mariano_test_absolute_loss(recorder):
    rng = np.random.default_rng(9)
    n = 500
    actual = rng.standard_normal(n)
    forecast_a = actual + rng.standard_normal(n) * 0.2
    forecast_b = actual + rng.standard_normal(n) * 0.8
    recorder.capture(
        forecast_evaluation_model.get_diebold_mariano_test(
            pd.Series(actual),
            pd.Series(forecast_a),
            pd.Series(forecast_b),
            loss="absolute",
        ).round(4)
    )


def test_get_diebold_mariano_test_invalid_loss():
    try:
        forecast_evaluation_model.get_diebold_mariano_test(
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            loss="bad",
        )
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_diebold_mariano_test_too_few_observations(recorder):
    actual = pd.Series([1.0, 2.0, 3.0])
    forecast_a = pd.Series([1.1, 2.1, 3.1])
    forecast_b = pd.Series([0.9, 1.9, 2.9])
    recorder.capture(
        forecast_evaluation_model.get_diebold_mariano_test(
            actual, forecast_a, forecast_b
        )
    )


def test_get_rmse_hand_computed():
    # actual - forecast = [-1, 1, -2, 1] -> squared = [1, 1, 4, 1] -> mean 1.75
    actual = pd.Series([10.0, 12.0, 11.0, 13.0])
    forecast = pd.Series([11.0, 11.0, 13.0, 12.0])
    rmse = forecast_evaluation_model.get_rmse(actual, forecast)
    assert abs(rmse - np.sqrt(1.75)) < 1e-9


def test_get_rmse_constant_error():
    # A constant error of 2 everywhere gives RMSE = 2 exactly.
    actual = pd.Series([1.0, 2.0, 3.0, 4.0])
    forecast = actual - 2.0
    assert abs(forecast_evaluation_model.get_rmse(actual, forecast) - 2.0) < 1e-9


def test_get_rmse_zero_for_perfect_forecast():
    actual = pd.Series([1.0, 2.0, 3.0])
    assert forecast_evaluation_model.get_rmse(actual, actual) == 0.0


def test_get_rmse_dataframe():
    actual = pd.DataFrame({"A": [10.0, 12.0], "B": [1.0, 2.0]})
    forecast = pd.DataFrame({"A": [11.0, 11.0], "B": [1.0, 2.0]})
    result = forecast_evaluation_model.get_rmse(actual, forecast)
    assert isinstance(result, pd.Series)
    assert abs(result["A"] - np.sqrt((1**2 + 1**2) / 2)) < 1e-9
    assert result["B"] == 0.0


def test_get_rmse_invalid_type():
    try:
        forecast_evaluation_model.get_rmse([1, 2, 3], [1, 2, 3])
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


def test_get_mae_hand_computed():
    # |actual - forecast| = [1, 1, 2, 1] -> mean 1.25
    actual = pd.Series([10.0, 12.0, 11.0, 13.0])
    forecast = pd.Series([11.0, 11.0, 13.0, 12.0])
    mae = forecast_evaluation_model.get_mae(actual, forecast)
    assert abs(mae - 1.25) < 1e-9


def test_get_mae_constant_error():
    actual = pd.Series([1.0, 2.0, 3.0, 4.0])
    forecast = actual - 3.0
    assert abs(forecast_evaluation_model.get_mae(actual, forecast) - 3.0) < 1e-9


def test_get_mae_zero_for_perfect_forecast():
    actual = pd.Series([1.0, 2.0, 3.0])
    assert forecast_evaluation_model.get_mae(actual, actual) == 0.0


def test_get_mae_dataframe():
    actual = pd.DataFrame({"A": [10.0, 12.0], "B": [1.0, 2.0]})
    forecast = pd.DataFrame({"A": [11.0, 11.0], "B": [1.0, 2.0]})
    result = forecast_evaluation_model.get_mae(actual, forecast)
    assert isinstance(result, pd.Series)
    assert abs(result["A"] - 1.0) < 1e-9
    assert result["B"] == 0.0


def test_get_mae_invalid_type():
    try:
        forecast_evaluation_model.get_mae([1, 2, 3], [1, 2, 3])
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


def _naive_last_value_forecast(train: pd.Series, forecast_steps: int) -> pd.Series:
    """A trivial dummy forecaster used to hand-verify the out-of-sample harness: it
    just repeats the last training observation `forecast_steps` times."""
    return pd.Series([train.iloc[-1]] * forecast_steps)


def test_get_out_of_sample_validation_no_leakage():
    # Confirm the split is chronological and training excludes every holdout index.
    series = pd.Series(np.arange(20.0), index=pd.RangeIndex(20))
    result = forecast_evaluation_model.get_out_of_sample_validation(
        series, _naive_last_value_forecast, train_fraction=0.8
    )
    assert result["Holdout Observations"] == 4

    split = int(len(series) * 0.8)
    train_index = set(series.index[:split])
    holdout_index = set(series.index[split:])
    assert train_index.isdisjoint(holdout_index)
    assert len(train_index) + len(holdout_index) == len(series)


def test_get_out_of_sample_validation_naive_forecaster_hand_computed():
    # train=[0..15], holdout=[16..19]; repeating 15 gives errors [1, 2, 3, 4].
    series = pd.Series(np.arange(20.0))
    result = forecast_evaluation_model.get_out_of_sample_validation(
        series, _naive_last_value_forecast, train_fraction=0.8
    )

    expected_errors = np.array([1.0, 2.0, 3.0, 4.0])
    expected_rmse = np.sqrt(np.mean(expected_errors**2))
    expected_mae = np.mean(expected_errors)

    assert abs(result["RMSE"] - expected_rmse) < 1e-9
    assert abs(result["MAE"] - expected_mae) < 1e-9
    assert result["Holdout Observations"] == 4


def test_get_out_of_sample_validation_plugged_into_arima(recorder):
    # End-to-end wiring check against a real Part-1 forecasting function.
    from financetoolkit.econometrics.forecasting import time_series_model

    rng = np.random.default_rng(1)
    n = 200
    e = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.6 * y[t - 1] + e[t]

    result = forecast_evaluation_model.get_out_of_sample_validation(
        pd.Series(y),
        time_series_model.get_arima_forecast,
        train_fraction=0.9,
        p=1,
        d=0,
        q=0,
    )
    assert result["Holdout Observations"] == 20
    assert result["RMSE"] > 0
    assert result["MAE"] > 0
    recorder.capture(result.round(4))


def test_get_out_of_sample_validation_invalid_type():
    try:
        forecast_evaluation_model.get_out_of_sample_validation(
            [1, 2, 3], _naive_last_value_forecast
        )
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


def test_get_out_of_sample_validation_invalid_train_fraction():
    series = pd.Series(np.arange(20.0))
    try:
        forecast_evaluation_model.get_out_of_sample_validation(
            series, _naive_last_value_forecast, train_fraction=1.5
        )
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_out_of_sample_validation_mismatched_forecast_length():
    def _wrong_length_forecaster(train: pd.Series, forecast_steps: int) -> pd.Series:
        return pd.Series([0.0] * (forecast_steps - 1))

    series = pd.Series(np.arange(20.0))
    try:
        forecast_evaluation_model.get_out_of_sample_validation(
            series, _wrong_length_forecaster, train_fraction=0.8
        )
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_volatility_forecast_uses_information_up_to_previous_period():
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0, 0.01, 120))
    lambda_, window_size = 0.94, 10

    ewma = forecast_evaluation_model.get_volatility_forecast(
        returns, "ewma", window_size, lambda_
    )
    rolling = forecast_evaluation_model.get_volatility_forecast(
        returns, "rolling", window_size, lambda_
    )

    # By construction: σₜ² = λ·σₜ₋₁² + (1 - λ)·rₜ₋₁², seeded with σ₁² = r₀².
    expected_ewma = [np.nan, returns[0] ** 2]
    for t in range(2, len(returns)):
        expected_ewma.append(
            lambda_ * expected_ewma[-1] + (1 - lambda_) * returns[t - 1] ** 2
        )
    np.testing.assert_allclose(ewma.to_numpy(), expected_ewma, rtol=1e-12)

    # The rolling forecast for t is the sample variance of rₜ₋w..rₜ₋₁.
    for t in range(window_size, len(returns)):
        assert np.isclose(
            rolling[t], returns.iloc[t - window_size : t].var(ddof=1), rtol=1e-12
        )
    assert rolling.iloc[:window_size].isna().all()

    # Changing rₜ leaves both forecasts for t unchanged, and changes those for t + 1.
    shocked = returns.copy()
    shocked[60] = 0.2
    for method in ("ewma", "rolling"):
        base = forecast_evaluation_model.get_volatility_forecast(
            returns, method, window_size, lambda_
        )
        moved = forecast_evaluation_model.get_volatility_forecast(
            shocked, method, window_size, lambda_
        )
        assert np.isclose(base[60], moved[60])
        assert not np.isclose(base[61], moved[61])


def test_get_volatility_forecast_dataframe_uses_each_columns_own_history():
    rng = np.random.default_rng(4)
    returns = pd.DataFrame(
        {"Old": rng.normal(0, 0.01, 80), "Young": rng.normal(0, 0.02, 80)}
    )
    returns.loc[:39, "Young"] = np.nan

    for method in ("ewma", "rolling"):
        result = forecast_evaluation_model.get_volatility_forecast(
            returns, method, 10, 0.94
        )
        alone = forecast_evaluation_model.get_volatility_forecast(
            returns["Old"], method, 10, 0.94
        )
        young_alone = forecast_evaluation_model.get_volatility_forecast(
            returns["Young"].dropna(), method, 10, 0.94
        )
        pd.testing.assert_series_equal(result["Old"], alone, check_names=False)
        pd.testing.assert_series_equal(
            result["Young"].dropna(), young_alone.dropna(), check_names=False
        )


def test_get_diebold_mariano_test_weighs_the_first_horizon_lags_equally():
    rng = np.random.default_rng(5)
    actual = pd.Series(rng.normal(size=80))
    forecast_a = (
        actual + pd.Series(rng.normal(0, 1, 80)).rolling(3, min_periods=1).mean()
    )
    forecast_b = actual + rng.normal(0, 1.1, 80)

    for horizon in (1, 2, 4):
        result = forecast_evaluation_model.get_diebold_mariano_test(
            actual, forecast_a, forecast_b, horizon=horizon
        )

        # Diebold & Mariano (1995): gamma_0 + 2 * the sum of the first h - 1
        # autocovariances, unweighted, with the Harvey, Leybourne & Newbold correction.
        d = ((actual - forecast_a) ** 2 - (actual - forecast_b) ** 2).to_numpy()
        demeaned, n = d - d.mean(), len(d)
        gammas = [np.sum(demeaned[k:] * demeaned[: n - k]) / n for k in range(horizon)]
        variance = gammas[0] + 2 * sum(gammas[1:])
        correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
        expected = d.mean() / np.sqrt(variance / n) * correction

        assert np.isclose(result["Diebold-Mariano Statistic"], expected)


def test_get_diebold_mariano_test_falls_back_to_bartlett_weights():
    # Alternating loss differentials have a negative first autocovariance large enough
    # to make the rectangular long-run variance negative at horizon 2.
    actual = pd.Series(np.zeros(20))
    forecast_a = pd.Series(np.tile([1.0, 0.0], 10) + np.linspace(0, 0.1, 20))
    forecast_b = pd.Series(np.full(20, 0.5))

    d = (forecast_a**2 - forecast_b**2).to_numpy()
    demeaned, n = d - d.mean(), len(d)
    rectangular = np.sum(demeaned**2) / n + 2 * np.sum(demeaned[1:] * demeaned[:-1]) / n
    assert rectangular <= 0

    result = forecast_evaluation_model.get_diebold_mariano_test(
        actual, forecast_a, forecast_b, horizon=2, small_sample_correction=False
    )
    bartlett = np.sum(demeaned**2) / n + np.sum(demeaned[1:] * demeaned[:-1]) / n
    assert np.isclose(
        result["Diebold-Mariano Statistic"], d.mean() / np.sqrt(bartlett / n)
    )


def _gaussian_ar1(n, mean, deviation, autocorrelation, seed):
    rng = np.random.default_rng(seed)
    z = np.empty(n)
    z[0] = mean + rng.standard_normal() * deviation / np.sqrt(1 - autocorrelation**2)
    for t in range(1, n):
        z[t] = (
            mean
            + autocorrelation * (z[t - 1] - mean)
            + deviation * rng.standard_normal()
        )
    return z


@pytest.mark.parametrize(
    ("mean", "deviation", "autocorrelation"), [(0.0, 1.0, 0.0), (0.1, 1.2, 0.2)]
)
def test_get_berkowitz_test_is_the_exact_ar1_likelihood_ratio(
    mean, deviation, autocorrelation
):
    arima = pytest.importorskip("statsmodels.tsa.arima.model")
    from scipy import special, stats  # noqa: PLC0415

    z = _gaussian_ar1(800, mean, deviation, autocorrelation, seed=11)
    result = forecast_evaluation_model.get_berkowitz_test(pd.Series(special.ndtr(z)))

    # The exact Gaussian AR(1) maximum likelihood of statsmodels' state space ARIMA.
    reference = arima.ARIMA(z, order=(1, 0, 0), trend="c").fit()
    statistic = 2 * (reference.llf - np.sum(stats.norm.logpdf(z)))

    assert result["Berkowitz Statistic"] == pytest.approx(statistic, rel=1e-5)
    assert result["Mean"] == pytest.approx(reference.params[0], abs=1e-4)
    assert result["Autocorrelation"] == pytest.approx(reference.params[1], abs=1e-4)
    assert result["Standard Deviation"] == pytest.approx(
        np.sqrt(reference.params[2]), rel=1e-4
    )
    assert result["P-Value"] == pytest.approx(stats.chi2.sf(statistic, 3), rel=1e-4)


def test_get_berkowitz_test_tail_is_a_censored_normal_fit(recorder):
    from scipy import optimize, special, stats  # noqa: PLC0415

    z = _gaussian_ar1(800, 0.1, 1.2, 0.0, seed=4)
    result = forecast_evaluation_model.get_berkowitz_test(
        pd.Series(special.ndtr(z)), alpha=0.05
    )

    cutoff = special.ndtri(0.05)
    tail, censored = z[z < cutoff], np.sum(z >= cutoff)

    def negative_log_likelihood(parameters):
        return -(
            np.sum(stats.norm.logpdf(tail, parameters[0], parameters[1]))
            + censored * stats.norm.logsf(cutoff, parameters[0], parameters[1])
        )

    fit = optimize.minimize(
        negative_log_likelihood,
        [0, 1],
        method="L-BFGS-B",
        bounds=[(-10, 10), (0.01, 3)],
    )
    statistic = 2 * (negative_log_likelihood([0, 1]) - fit.fun)

    assert result["Berkowitz Statistic"] == pytest.approx(statistic, rel=1e-6)
    assert result["Tail Observations"] == len(tail)
    assert result["P-Value"] < SIGNIFICANCE_LEVEL
    recorder.capture(result.round(4))


def test_get_probability_integral_transform_is_uniform_for_the_true_model():
    rng = np.random.default_rng(3)
    returns = pd.DataFrame({"A": rng.standard_normal(1500) * 0.01})

    pit = forecast_evaluation_model.get_probability_integral_transform(
        returns, model="gaussian", window_size=250
    )["A"].dropna()

    assert len(pit) == 1250
    assert (
        forecast_evaluation_model.get_berkowitz_test(pit)["P-Value"]
        > SIGNIFICANCE_LEVEL
    )

    garch_pit = forecast_evaluation_model.get_probability_integral_transform(
        returns["A"].iloc[:400], model="garch", window_size=300, refit_every=100
    ).dropna()
    assert len(garch_pit) == 100
    assert garch_pit.between(0, 1).all()


def test_get_directional_accuracy_test_by_hand():
    from scipy import stats  # noqa: PLC0415

    actual = pd.Series([0.02, -0.01, 0.03, 0.01, -0.02, -0.04, 0.05, 0.01, -0.01, 0.02])
    forecast = pd.Series(
        [0.01, 0.01, 0.02, -0.01, -0.01, -0.02, 0.01, 0.02, 0.01, 0.01]
    )

    # Pesaran-Timmermann: 7 of 10 signs right, 6 positive returns and 7 positive forecasts.
    p, p_y, p_x, n = 0.7, 0.6, 0.7, 10
    p_star = p_y * p_x + (1 - p_y) * (1 - p_x)
    variance = p_star * (1 - p_star) / n - (
        (2 * p_y - 1) ** 2 * p_x * (1 - p_x) / n
        + (2 * p_x - 1) ** 2 * p_y * (1 - p_y) / n
        + 4 * p_y * p_x * (1 - p_y) * (1 - p_x) / n**2
    )
    pesaran = forecast_evaluation_model.get_directional_accuracy_test(actual, forecast)
    assert pesaran["Pesaran-Timmermann Statistic"] == pytest.approx(
        (p - p_star) / np.sqrt(variance)
    )
    assert pesaran["Directional Accuracy"] == pytest.approx(0.7)
    assert pesaran["P-Value"] == pytest.approx(
        stats.norm.sf(pesaran["Pesaran-Timmermann Statistic"])
    )

    # Anatolyev-Gerko: the long-short return over that of random positions.
    signs = np.sign(forecast.to_numpy())
    y = actual.to_numpy()
    share_long = (1 + signs.mean()) / 2
    expected = (np.mean(signs * y) - signs.mean() * y.mean()) / np.sqrt(
        4 * share_long * (1 - share_long) * np.sum((y - y.mean()) ** 2) / n**2
    )
    anatolyev = forecast_evaluation_model.get_directional_accuracy_test(
        actual, forecast, method="anatolyev_gerko"
    )
    assert anatolyev["Anatolyev-Gerko Statistic"] == pytest.approx(expected)


def test_get_directional_accuracy_test_detects_a_useful_forecast(recorder):
    rng = np.random.default_rng(8)
    signal = rng.standard_normal(1000)
    actual = pd.DataFrame({"A": 0.3 * signal + rng.standard_normal(1000)})
    forecast = pd.DataFrame({"A": signal})
    noise = pd.DataFrame({"A": rng.standard_normal(1000)})

    useful = forecast_evaluation_model.get_directional_accuracy_test(actual, forecast)
    useless = forecast_evaluation_model.get_directional_accuracy_test(actual, noise)
    assert useful.loc["P-Value", "A"] < SIGNIFICANCE_LEVEL
    assert useless.loc["P-Value", "A"] > SIGNIFICANCE_LEVEL
    recorder.capture(pd.concat([useful, useless], axis=1).round(4))


def test_get_return_forecast_uses_the_returns_before_each_period():
    returns = pd.Series(np.arange(1.0, 11.0))
    momentum = forecast_evaluation_model.get_return_forecast(returns, "momentum", 3)
    assert momentum.iloc[3] == pytest.approx(2.0)

    # On an exact AR(1) the rolling fit recovers it and forecasts the next value.
    autoregressive_series = pd.Series(0.5 ** np.arange(20) + 1.0)
    forecast = forecast_evaluation_model.get_return_forecast(
        autoregressive_series, "autoregressive", 5
    )
    assert forecast.iloc[10] == pytest.approx(autoregressive_series.iloc[10])


def _mcs_losses():
    rng = np.random.default_rng(7)
    common = rng.standard_normal((500, 1))
    return pd.DataFrame(
        np.abs(common + rng.standard_normal((500, 6)) * 0.8) + np.linspace(0, 0.25, 6),
        columns=[f"M{i}" for i in range(6)],
    )


@pytest.mark.parametrize(("statistic", "arch_method"), [("max", "max"), ("range", "R")])
def test_get_model_confidence_set_matches_the_arch_package(
    monkeypatch, statistic, arch_method
):
    bootstrap = pytest.importorskip("arch.bootstrap")
    from financetoolkit.portfolio.construction.optimization import (  # noqa: PLC0415
        estimation_model,
    )

    losses = _mcs_losses()
    reference = bootstrap.MCS(
        losses, size=0.1, reps=500, block_size=8, method=arch_method, seed=1
    )
    reference._testing = True  # arch keeps the resampled periods for testing
    reference.compute()

    # The same resampled periods as arch, so the p-values must agree exactly.
    indices = iter(reference._bootstrap_indices)
    monkeypatch.setattr(
        estimation_model, "get_bootstrap_indices", lambda *args, **kwargs: next(indices)
    )
    result = forecast_evaluation_model.get_model_confidence_set(
        losses, alpha=0.1, statistic=statistic, n_bootstrap=500, block_size=8
    )

    np.testing.assert_allclose(
        result["MCS P-Value"],
        reference.pvalues["Pvalue"].reindex(result.index),
        atol=1e-12,
    )
    assert set(result.index[result["In Confidence Set"]]) == set(reference.included)


def test_get_model_confidence_set_keeps_the_best_models(recorder):
    result = forecast_evaluation_model.get_model_confidence_set(_mcs_losses())

    assert result.index[0] == "M0"
    assert result["Rank"].tolist() == list(range(1, 7))
    assert result["MCS P-Value"].is_monotonic_decreasing
    assert not result.loc["M5", "In Confidence Set"]
    recorder.capture(result.round(4))

    with pytest.raises(ValueError, match="'max' or 'range'"):
        forecast_evaluation_model.get_model_confidence_set(
            _mcs_losses(), statistic="sum"
        )


def test_hong_li_constants_of_the_quartic_kernel():
    from scipy import integrate  # noqa: PLC0415

    squared, boundary, variance = forecast_evaluation_model._hong_li_constants()

    assert squared == pytest.approx(5 / 7, rel=1e-12)
    assert forecast_evaluation_model._quartic_kernel_mass(1.0) == pytest.approx(1.0)
    # ∫_0^1 ∫_-1^b k^2 / (∫_-1^b k)^2 by a plain Riemann sum.
    b = np.linspace(0, 1, 2001)[1:] - 0.00025
    u = np.linspace(-1, 1, 40001)
    k2 = (15 / 16 * (1 - u**2) ** 2) ** 2
    inner = np.array([np.trapezoid(k2[u <= x], u[u <= x]) for x in b])
    mass = forecast_evaluation_model._quartic_kernel_mass(b)
    assert boundary == pytest.approx(np.mean(inner / mass**2), rel=1e-4)
    assert variance == pytest.approx(
        2
        * integrate.quad(
            lambda x: integrate.quad(
                lambda v: forecast_evaluation_model._quartic_kernel(x + v)
                * forecast_evaluation_model._quartic_kernel(v),
                -1,
                1,
            )[0]
            ** 2,
            -1,
            1,
        )[0]
        ** 2,
        rel=1e-10,
    )


def test_hong_li_distance_matches_a_fine_grid():
    rng = np.random.default_rng(3)
    pit = pd.Series(rng.random(300))
    result = forecast_evaluation_model.get_hong_li_test(pit, lags=1)

    # M(1) on a 1,500 x 1,500 midpoint grid, independent of the Gauss-Legendre panels.
    values = pit.to_numpy()
    h = result["Bandwidth"]
    grid = (np.arange(1500) + 0.5) / 1500
    kernel = forecast_evaluation_model._boundary_kernel(grid, values, h)
    density = kernel[:, 1:] @ kernel[:, :-1].T / 299
    distance = np.mean((density - 1) ** 2)
    squared, boundary, variance = forecast_evaluation_model._hong_li_constants()
    centring = ((1 / h - 2) * squared + 2 * boundary) ** 2 - 1
    expected = h * (299 * distance - centring) / np.sqrt(variance)

    assert result["Q(1) Statistic"] == pytest.approx(expected, abs=2e-3)
    assert h == pytest.approx(np.std(values, ddof=1) * 300 ** (-1 / 6), rel=1e-12)


def test_hong_li_test_is_centred_under_the_null_and_rejects_dependence():
    rng = np.random.default_rng(4)
    statistics = [
        forecast_evaluation_model.get_hong_li_test(pd.Series(rng.random(400)), 1)[
            "Q(1) Statistic"
        ]
        for _ in range(30)
    ]
    assert abs(np.mean(statistics)) < 0.6

    # PITs that follow each other closely (an AR(1) of their normal quantiles) are rejected.
    z = np.zeros(400)
    for t in range(1, 400):
        z[t] = 0.6 * z[t - 1] + 0.8 * rng.standard_normal()
    from scipy import stats as scipy_stats  # noqa: PLC0415

    dependent = forecast_evaluation_model.get_hong_li_test(
        pd.Series(scipy_stats.norm.cdf(z))
    )
    assert dependent["W P-Value"] < 1e-4
    assert (
        forecast_evaluation_model.get_hong_li_test(pd.Series(rng.random(10)))[
            "W Statistic"
        ]
        != forecast_evaluation_model.get_hong_li_test(pd.Series(rng.random(10)))[
            "W Statistic"
        ]
    )


def test_hong_li_p_values_follow_the_simulated_null_distribution():
    rng = np.random.default_rng(21)
    results = pd.DataFrame(
        [
            forecast_evaluation_model.get_hong_li_test(pd.Series(rng.random(300)), 4)
            for _ in range(40)
        ]
    )

    # Under the null the p-values are uniform: W, with a standard deviation of about 1.8,
    # rejects about 18% of the time at 5% against the standard normal distribution.
    assert results["W P-Value"].mean() == pytest.approx(0.5, abs=0.15)
    assert (results["W P-Value"] < 0.05).mean() < 0.15
    assert results["W Statistic"].std() > 1.3

    # The p-value of a Q(j) is that of W with one lag, and the p-values fall as W rises.
    first = results.iloc[0]
    assert first["Q(1) P-Value"] == pytest.approx(
        forecast_evaluation_model.get_hong_li_p_value(first["Q(1) Statistic"], 300, 1)
    )
    p_values = [
        forecast_evaluation_model.get_hong_li_p_value(statistic, 1000, 4)
        for statistic in (-2.0, 0.0, 2.0, 6.0)
    ]
    assert p_values == sorted(p_values, reverse=True)
    # A larger W is needed with more lags, since the Q(j) are correlated.
    assert forecast_evaluation_model.get_hong_li_p_value(
        3.0, 1000, 10
    ) > forecast_evaluation_model.get_hong_li_p_value(3.0, 1000, 1)
    assert np.isnan(forecast_evaluation_model.get_hong_li_p_value(np.nan, 1000, 4))


def test_hong_li_test_is_nan_for_pits_without_variation():
    # A bandwidth of 0 would ask for an integration grid of infinitely many panels.
    for constant in (np.full(300, 0.5), np.zeros(300)):
        result = forecast_evaluation_model.get_hong_li_test(pd.Series(constant), 4)
        assert result.drop("Observations").isna().all()
        assert result["Observations"] == 300
