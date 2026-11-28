"""Realized Volatility Model Tests"""

import math

import numpy as np
import pandas as pd
import pytest

from financetoolkit.risk.volatility import realized_volatility_model

# How far apart the four range-based estimators may sit on a calm path.
MAGNITUDE_AGREEMENT_RATIO = 2

# The HAR-RV forecast should track a HAR-structured series at least this closely.
HAR_RV_MINIMUM_CORRELATION = 0.9


def _generate_ohlc(n: int = 260, seed: int = 42) -> pd.DataFrame:
    """Generates a synthetic daily OHLC dataset with a PeriodIndex, mirroring the shape
    of `self._historical_data["daily"]` used by the Risk controller."""
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2021-01-01", periods=n, freq="D").to_period("D")
    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.015))
    open_ = close * np.exp(rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.006, n)))

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close}, index=dates
    )


def test_get_parkinson_volatility(recorder):
    ohlc = _generate_ohlc()
    result = realized_volatility_model.get_parkinson_volatility(
        ohlc["High"], ohlc["Low"], period="yearly"
    )
    recorder.capture(result.round(4))


def test_get_parkinson_volatility_dataframe(recorder):
    ohlc_aapl = _generate_ohlc(seed=1)
    ohlc_msft = _generate_ohlc(seed=2)
    high = pd.DataFrame({"AAPL": ohlc_aapl["High"], "MSFT": ohlc_msft["High"]})
    low = pd.DataFrame({"AAPL": ohlc_aapl["Low"], "MSFT": ohlc_msft["Low"]})

    result = realized_volatility_model.get_parkinson_volatility(
        high, low, period="quarterly"
    )
    recorder.capture(result.round(4))


def test_get_garman_klass_volatility(recorder):
    ohlc = _generate_ohlc()
    result = realized_volatility_model.get_garman_klass_volatility(
        ohlc["Open"], ohlc["High"], ohlc["Low"], ohlc["Close"], period="yearly"
    )
    recorder.capture(result.round(4))


def test_get_rogers_satchell_volatility(recorder):
    ohlc = _generate_ohlc()
    result = realized_volatility_model.get_rogers_satchell_volatility(
        ohlc["Open"], ohlc["High"], ohlc["Low"], ohlc["Close"], period="yearly"
    )
    recorder.capture(result.round(4))


def test_get_yang_zhang_volatility(recorder):
    ohlc = _generate_ohlc()
    result = realized_volatility_model.get_yang_zhang_volatility(
        ohlc["Open"], ohlc["High"], ohlc["Low"], ohlc["Close"], period="yearly"
    )
    recorder.capture(result.round(4))


def test_estimators_agree_in_magnitude(recorder):
    # The four range-based estimators should agree in magnitude on a calm path.
    ohlc = _generate_ohlc(n=500)

    parkinson = realized_volatility_model.get_parkinson_volatility(
        ohlc["High"], ohlc["Low"], period="yearly"
    )
    garman_klass = realized_volatility_model.get_garman_klass_volatility(
        ohlc["Open"], ohlc["High"], ohlc["Low"], ohlc["Close"], period="yearly"
    )
    rogers_satchell = realized_volatility_model.get_rogers_satchell_volatility(
        ohlc["Open"], ohlc["High"], ohlc["Low"], ohlc["Close"], period="yearly"
    )
    yang_zhang = realized_volatility_model.get_yang_zhang_volatility(
        ohlc["Open"], ohlc["High"], ohlc["Low"], ohlc["Close"], period="yearly"
    )

    combined = pd.concat(
        [parkinson, garman_klass, rogers_satchell, yang_zhang],
        axis=1,
        keys=["parkinson", "garman_klass", "rogers_satchell", "yang_zhang"],
    ).dropna()

    # All estimates should be within the same order of magnitude of each other.
    ratio = combined.max(axis=1) / combined.min(axis=1)
    recorder.capture(bool((ratio < MAGNITUDE_AGREEMENT_RATIO).all()))


def test_invalid_period_raises():
    ohlc = _generate_ohlc()

    try:
        realized_volatility_model.get_parkinson_volatility(
            ohlc["High"], ohlc["Low"], period="daily"
        )
        raise AssertionError("Expected a ValueError to be raised.")
    except ValueError:
        pass


def test_invalid_type_raises():
    try:
        realized_volatility_model.get_parkinson_volatility([1, 2, 3], [1, 2, 3], "yearly")  # type: ignore
        raise AssertionError("Expected a TypeError to be raised.")
    except TypeError:
        pass


def test_too_few_observations(recorder):
    # One day leaves the Variance components with zero degrees of freedom.
    ohlc = _generate_ohlc(n=1)
    result = realized_volatility_model.get_yang_zhang_volatility(
        ohlc["Open"], ohlc["High"], ohlc["Low"], ohlc["Close"], period="yearly"
    )
    recorder.capture(bool(result.isna().all()))


def test_get_har_rv_forecast_recovers_relationship(recorder):
    # A genuine daily/weekly/monthly HAR structure, which the forecast should track.
    rng = np.random.default_rng(4)
    n = 2000
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    true_b0, true_bD, true_bW, true_bM = 0.5, 0.4, 0.3, 0.2
    rv = np.full(n, 1.0)
    for t in range(30, n):
        d = rv[t - 1]
        w = rv[t - 5 : t].mean()
        m = rv[t - 22 : t].mean()
        rv[t] = true_b0 + true_bD * d + true_bW * w + true_bM * m + rng.normal(0, 0.05)
        rv[t] = max(rv[t], 0.01)
    rv_series = pd.Series(rv, index=dates)

    forecast = realized_volatility_model.get_har_rv_forecast(rv_series, horizon=1)
    valid = pd.concat([rv_series.shift(-1), forecast], axis=1).dropna()
    valid.columns = ["actual", "forecast"]
    correlation = valid["actual"].corr(valid["forecast"])

    assert correlation > HAR_RV_MINIMUM_CORRELATION
    recorder.capture(round(correlation, 4))


def test_get_har_rv_forecast_dataframe(recorder):
    ohlc = _generate_ohlc(n=200)
    returns = ohlc["Close"].pct_change().dropna()
    realized_variance = pd.DataFrame({"AAPL": returns**2, "MSFT": returns**2})
    result = realized_volatility_model.get_har_rv_forecast(realized_variance)
    recorder.capture(result.dropna().round(6).head())


def test_get_har_rv_forecast_too_few_observations(recorder):
    realized_variance = pd.Series([0.0001, 0.0002, 0.0003, 0.0001, 0.0002])
    result = realized_volatility_model.get_har_rv_forecast(realized_variance)
    recorder.capture(bool(result.isna().all()))


def test_get_har_rv_forecast_invalid_type():
    try:
        realized_volatility_model.get_har_rv_forecast(123)  # type: ignore
        raise AssertionError("Expected a TypeError to be raised.")
    except TypeError:
        pass


def test_get_yang_zhang_volatility_needs_a_period_index():
    dates = pd.bdate_range("2023-01-02", periods=60)
    rng = np.random.default_rng(5)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60))), index=dates)
    open_ = close.shift(1).fillna(100) * np.exp(rng.normal(0, 0.002, 60))
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995

    # Every grouping step converts the index the same way, with the same clear error.
    with pytest.raises(TypeError, match="PeriodIndex"):
        realized_volatility_model.get_yang_zhang_volatility(
            open_, high, low, close, "monthly"
        )

    period_index = dates.to_period("D")
    result = realized_volatility_model.get_yang_zhang_volatility(
        open_.set_axis(period_index),
        high.set_axis(period_index),
        low.set_axis(period_index),
        close.set_axis(period_index),
        "monthly",
    )

    assert len(result) == 3  # noqa: PLR2004
    assert result.notna().all()


@pytest.mark.parametrize("price", ["Open", "High", "Low", "Close"])
def test_range_based_estimators_leave_out_a_bad_bar(price):
    ohlc = _generate_ohlc()
    bad_day = ohlc.index[100]
    bad = ohlc.copy()
    bad.loc[bad_day, price] = 0.0
    # The same bar left out by hand: every price of that day missing.
    without = ohlc.copy()
    without.loc[bad_day] = np.nan

    for estimator, prices in [
        (realized_volatility_model.get_parkinson_volatility, ["High", "Low"]),
        (
            realized_volatility_model.get_garman_klass_volatility,
            ["Open", "High", "Low", "Close"],
        ),
        (
            realized_volatility_model.get_rogers_satchell_volatility,
            ["Open", "High", "Low", "Close"],
        ),
    ]:
        result = estimator(*[bad[column] for column in prices], period="yearly")
        expected = (
            estimator(*[without[column] for column in prices], period="yearly")
            if price in prices
            else estimator(*[ohlc[column] for column in prices], period="yearly")
        )

        assert np.isfinite(result).all()
        pd.testing.assert_series_equal(result, expected)


def test_yang_zhang_volatility_leaves_out_a_bad_open():
    ohlc = _generate_ohlc()
    bad = ohlc.copy()
    bad.loc[ohlc.index[100], "Open"] = 0.0
    negative = ohlc.copy()
    negative.loc[ohlc.index[100], "Open"] = -1.0

    def yang_zhang(data):
        return realized_volatility_model.get_yang_zhang_volatility(
            data["Open"], data["High"], data["Low"], data["Close"], period="yearly"
        )

    # The day still traded: its Close remains the base of the next overnight return, and
    # only the terms that need its Open are left out, which barely moves the year.
    assert np.isfinite(yang_zhang(bad)).all()
    pd.testing.assert_series_equal(yang_zhang(bad), yang_zhang(negative))
    pd.testing.assert_series_equal(
        yang_zhang(bad), yang_zhang(ohlc), check_exact=False, rtol=0.01
    )


def test_get_garman_klass_yang_zhang_volatility(recorder):
    ohlc = _generate_ohlc()
    result = realized_volatility_model.get_garman_klass_yang_zhang_volatility(
        ohlc["Open"], ohlc["High"], ohlc["Low"], ohlc["Close"], period="monthly"
    )
    recorder.capture(result.round(4))

    garman_klass = realized_volatility_model.get_garman_klass_volatility(
        ohlc["Open"], ohlc["High"], ohlc["Low"], ohlc["Close"], period="monthly"
    )

    # Garman-Klass plus the squared overnight return; the scaling cancels in the ratio to Garman-Klass.
    january = ohlc.loc["2021-01"]
    overnight = np.log(january["Open"] / ohlc["Close"].shift(1).loc["2021-01"]) ** 2
    session = (
        0.5 * np.log(january["High"] / january["Low"]) ** 2
        - (2 * np.log(2) - 1) * np.log(january["Close"] / january["Open"]) ** 2
    )
    # The first day has no previous Close, so its overnight return (and the day) is left out.
    assert (result.iloc[0] / garman_klass.iloc[0]) ** 2 == pytest.approx(
        (overnight + session).iloc[1:].mean() / session.mean()
    )


def _intraday_returns(days: int, bars: int, seed: int, jump: float = 0.0) -> pd.Series:
    """Intraday log returns of a driftless Brownian motion with a daily variance of 1e-4, a
    jump of `jump` in the middle of every day, on minute periods."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01 / np.sqrt(bars), (days, bars))
    returns[:, bars // 2] += jump
    start_times = pd.bdate_range("2021-01-04", periods=days) + pd.Timedelta(
        hours=9, minutes=30
    )
    index = pd.PeriodIndex(
        np.concatenate(
            [pd.date_range(start, periods=bars, freq="min") for start in start_times]
        ),
        freq="min",
    )
    return pd.Series(returns.ravel(), index=index)


def test_get_intraday_returns_start_each_day_from_its_open():
    index = pd.PeriodIndex(
        [
            "2025-12-30 09:30",
            "2025-12-30 10:30",
            "2025-12-31 09:30",
            "2025-12-31 10:30",
        ],
        freq="min",
    )
    open_prices = pd.Series([100.0, 101.0, 103.0, 0.0], index=index)
    close_prices = pd.Series([101.0, 102.0, 102.0, 104.0], index=index)

    returns = realized_volatility_model.get_intraday_returns(open_prices, close_prices)

    # The second day starts from its Open, not from the first day's last Close, and the bar
    # with an Open of 0 is left out.
    np.testing.assert_allclose(
        returns.to_numpy(), np.log([101 / 100, 102 / 101, 102 / 103])
    )


def test_get_realized_measures_against_a_hand_computation():
    returns = np.array([0.01, -0.02, 0.005, 0.0, -0.01, 0.03, -0.005, 0.01])
    index = pd.period_range("2025-12-31 09:30", periods=len(returns), freq="min")

    measures = realized_volatility_model.get_realized_measures(
        pd.Series(returns, index=index)
    ).iloc[0]

    # Written out term by term, with the absolute moments of a standard normal from the
    # gamma function: mu_p = 2^(p/2) Gamma((p + 1) / 2) / Gamma(1/2).
    m = len(returns)
    mu_1 = np.sqrt(2) * math.gamma(1) / math.gamma(0.5)
    mu_4_3 = 2 ** (2 / 3) * math.gamma(7 / 6) / math.gamma(0.5)
    bipower = sum(abs(returns[j] * returns[j - 1]) for j in range(1, m))
    tripower = sum(
        abs(returns[j] * returns[j - 1] * returns[j - 2]) ** (4 / 3)
        for j in range(2, m)
    )

    assert measures["Realized Variance"] == pytest.approx(0.00165)
    assert measures["Positive Semivariance"] == pytest.approx(0.001125)
    assert measures["Negative Semivariance"] == pytest.approx(0.000525)
    assert measures["Bipower Variation"] == pytest.approx(
        mu_1**-2 * m / (m - 1) * bipower
    )
    assert measures["Tripower Quarticity"] == pytest.approx(
        m * mu_4_3**-3 * m / (m - 2) * tripower
    )
    assert measures["Jump Variation"] == pytest.approx(
        0.00165 - measures["Bipower Variation"]
    )
    assert measures["Observations"] == m


def test_realized_measures_estimate_the_integrated_variance():
    measures = realized_volatility_model.get_realized_measures(
        _intraday_returns(days=500, bars=390, seed=0)
    )

    # A daily variance of 1e-4 and quarticity of 1e-8, which all three estimate without bias.
    assert measures["Realized Variance"].mean() == pytest.approx(1e-4, rel=0.01)
    assert measures["Bipower Variation"].mean() == pytest.approx(1e-4, rel=0.01)
    assert measures["Tripower Quarticity"].mean() == pytest.approx(1e-8, rel=0.03)
    np.testing.assert_allclose(
        measures["Positive Semivariance"] + measures["Negative Semivariance"],
        measures["Realized Variance"],
    )


def _jump_test(returns: pd.Series) -> pd.DataFrame:
    measures = realized_volatility_model.get_realized_measures(returns)
    return realized_volatility_model.get_jump_test(
        measures["Realized Variance"],
        measures["Bipower Variation"],
        measures["Tripower Quarticity"],
        measures["Observations"],
    )


def test_jump_test_keeps_its_size_and_finds_jumps():
    without_jumps = _jump_test(_intraday_returns(days=1000, bars=390, seed=0))
    with_jumps = _jump_test(_intraday_returns(days=200, bars=390, seed=1, jump=0.005))

    # Under the null the statistic is standard normal and about 5% of the days are flagged.
    assert without_jumps["z-Statistic"].mean() == pytest.approx(0, abs=0.08)
    assert without_jumps["z-Statistic"].std() == pytest.approx(1, abs=0.08)
    assert (without_jumps["p-Value"] < 0.05).mean() == pytest.approx(0.05, abs=0.02)
    assert (without_jumps["Significant Jump Variation"] > 0).mean() == pytest.approx(
        (without_jumps["p-Value"] < 0.05).mean()
    )
    # A jump of half the daily volatility is found on almost every day.
    assert (with_jumps["p-Value"] < 0.05).mean() > 0.9  # noqa: PLR2004


def test_jump_test_has_no_statistic_without_movement_or_returns():
    result = realized_volatility_model.get_jump_test(
        realized_variance=pd.Series([0.0, 1e-4]),
        bipower_variation=pd.Series([0.0, 1e-4]),
        tripower_quarticity=pd.Series([0.0, np.nan]),
        observations=pd.Series([78, 2]),
    )

    # The relative jump of the second day is defined, its statistic is not.
    assert result["Relative Jump"].tolist()[1] == 0
    assert result.drop(columns="Relative Jump").isna().all().all()
    assert np.isnan(result["Relative Jump"].iloc[0])
    with pytest.raises(ValueError, match="alpha"):
        realized_volatility_model.get_jump_test(
            pd.Series([1e-4]),
            pd.Series([1e-4]),
            pd.Series([1e-8]),
            pd.Series([78]),
            1.5,
        )
