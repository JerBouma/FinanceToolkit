"""Realized Volatility Model Tests"""

import numpy as np
import pandas as pd

from financetoolkit.risk import realized_volatility_model

# pylint: disable=missing-function-docstring

# The four range-based estimators should agree within this factor of each other on a
# calm synthetic price path, kept as a named constant to avoid a PLR2004 warning.
MAGNITUDE_AGREEMENT_RATIO = 2

# The HAR-RV fitted forecast should track a genuinely HAR-structured synthetic RV
# series with at least this much correlation.
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
    # All four range-based estimators should broadly agree in magnitude with each
    # other on the same, reasonably calm synthetic price path.
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
    # A single day means the overnight/open-to-close Variance components have zero
    # degrees of freedom (only one, NaN-shifted, observation), which should yield NaN
    # rather than raise.
    ohlc = _generate_ohlc(n=1)
    result = realized_volatility_model.get_yang_zhang_volatility(
        ohlc["Open"], ohlc["High"], ohlc["Low"], ohlc["Close"], period="yearly"
    )
    recorder.capture(bool(result.isna().all()))


def test_get_har_rv_forecast_recovers_relationship(recorder):
    # Simulate a Realized Variance series with a genuine daily/weekly/monthly HAR
    # structure and check the fitted forecast tracks the actual next-day RV well
    # (high correlation, roughly unbiased), which is what the model is meant to do.
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
