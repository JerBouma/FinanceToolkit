"""Statistics Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.econometrics.statistics import statistics_model
from tests.technical.sample_prices import sample_prices


def random_prices() -> pd.DataFrame:
    """80 bars of a random walk with a high and low around each close."""
    rng = np.random.default_rng(7)
    close = 100 + rng.normal(0, 1, 80).cumsum()
    return pd.DataFrame(
        {
            "High": close + 1 + rng.random(80),
            "Low": close - 1 - rng.random(80),
            "Close": close,
        }
    )


def last_three(values: pd.Series) -> list[float]:
    return values.iloc[-3:].tolist()


def test_get_linear_regression(recorder):
    prices = sample_prices()
    result = statistics_model.get_linear_regression(prices["Close"], 10).round(4)

    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_linear_regression_matches_talib():
    """TA-Lib's LINEARREG, LINEARREG_SLOPE, LINEARREG_INTERCEPT, LINEARREG_ANGLE and TSF (14)."""
    result = statistics_model.get_linear_regression(random_prices()["Close"], 14)

    assert result["Linear Regression"].first_valid_index() == 13
    expected = {
        "Linear Regression": [84.221451, 84.076754, 84.329392],
        "Slope": [-0.142173, -0.151512, -0.118621],
        "Intercept": [86.069704, 86.046412, 85.871461],
        "Angle": [-8.091702, -8.615479, -6.764852],
        "Time Series Forecast": [84.079278, 83.925242, 84.210772],
    }
    for column, values in expected.items():
        assert last_three(result[column]) == pytest.approx(values, abs=1e-6)


def test_linear_regression_of_a_straight_line_is_the_line():
    prices = pd.Series(np.arange(20, dtype=float) * 2 + 5)
    result = statistics_model.get_linear_regression(prices, 5)

    assert result["Linear Regression"].iloc[4:].tolist() == pytest.approx(
        prices.iloc[4:].tolist()
    )
    assert result["Slope"].iloc[-1] == pytest.approx(2)
    assert result["Time Series Forecast"].iloc[-1] == pytest.approx(prices.iloc[-1] + 2)
    assert result["Angle"].iloc[-1] == pytest.approx(np.degrees(np.arctan(2)))


def test_linear_regression_needs_two_periods():
    with pytest.raises(ValueError, match="at least 2"):
        statistics_model.get_linear_regression(pd.Series([1.0, 2.0, 3.0]), 1)


def test_slope_matches_pandas_ta_classic():
    """pandas-ta-classic's slope(5, as_angle=True, to_degrees=True)."""
    closes = random_prices()["Close"]

    assert last_three(
        statistics_model.get_slope(closes, 5, as_angle=True)
    ) == pytest.approx([12.989139, 11.314716, 14.218889], abs=1e-6)
    assert statistics_model.get_slope(closes).iloc[-1] == pytest.approx(
        closes.iloc[-1] - closes.iloc[-2]
    )


def test_rolling_correlation_matches_talib():
    """TA-Lib's CORREL(30) of the high and the low."""
    prices = random_prices()
    result = statistics_model.get_rolling_correlation(prices["High"], prices["Low"], 30)

    assert last_three(result) == pytest.approx([0.94909, 0.950296, 0.947656], abs=1e-6)


def test_rolling_correlation_of_a_flat_series_is_missing():
    result = statistics_model.get_rolling_correlation(
        pd.Series([5.0] * 10), pd.Series(np.arange(10.0)), 5
    )

    assert result.isna().all()


def test_rolling_median_matches_talib():
    result = statistics_model.get_rolling_median(random_prices()["Close"], 30)

    assert last_three(result) == pytest.approx(
        [85.560623, 85.560623, 85.734428], abs=1e-6
    )


def test_rolling_percentile_matches_talib_and_pandas_ta_classic():
    """TA-Lib's PERCENTILE(30, 75) and pandas-ta-classic's quantile(30, 0.3)."""
    closes = random_prices()["Close"]

    assert last_three(
        statistics_model.get_rolling_percentile(closes, 30, 0.75)
    ) == pytest.approx([86.242352] * 3, abs=1e-6)
    assert last_three(
        statistics_model.get_rolling_percentile(closes, 30, 0.3, "linear")
    ) == pytest.approx([85.190015] * 3, abs=1e-6)


def test_rolling_percentile_rejects_an_unknown_method_and_a_percentage():
    closes = random_prices()["Close"]

    with pytest.raises(ValueError, match="method must be one of"):
        statistics_model.get_rolling_percentile(closes, 30, 0.5, "midpoint")
    with pytest.raises(ValueError, match="between 0 and 1"):
        statistics_model.get_rolling_percentile(closes, 30, 50)


def test_percent_rank_matches_talib():
    """TA-Lib's PERCENTRANK(30), as a decimal; equal prices do not count as lower."""
    result = statistics_model.get_percent_rank(random_prices()["Close"], 30)

    assert result.first_valid_index() == 30
    assert last_three(result * 100) == pytest.approx(
        [16.666667, 13.333333, 53.333333], abs=1e-6
    )
    assert statistics_model.get_percent_rank(
        pd.Series([1.0, 2.0, 2.0, 1.0, 2.0]), 4
    ).iloc[-1] == pytest.approx(0.5)


def test_rolling_z_score_matches_pandas_ta_classic():
    result = statistics_model.get_rolling_z_score(random_prices()["Close"], 30)

    assert last_three(result) == pytest.approx(
        [-0.845895, -1.116233, 0.130722], abs=1e-6
    )
    assert np.isnan(
        statistics_model.get_rolling_z_score(pd.Series([2.0] * 10), 5).iloc[-1]
    )


def test_rolling_mean_absolute_deviation_matches_talib():
    result = statistics_model.get_rolling_mean_absolute_deviation(
        random_prices()["Close"], 14
    )

    assert last_three(result) == pytest.approx([0.844041, 0.87172, 0.910563], abs=1e-6)


def test_rolling_standard_error_matches_pandas_ta_classic():
    result = statistics_model.get_rolling_standard_error(random_prices()["Close"], 14)

    assert last_three(result) == pytest.approx([0.296464, 0.299753, 0.305034], abs=1e-6)


def test_rolling_entropy_matches_pandas_ta_classic():
    result = statistics_model.get_rolling_entropy(random_prices()["Close"], 10, 2)

    assert last_three(result) == pytest.approx([3.321847, 3.321866, 3.321854], abs=1e-6)
    # Equal prices give the maximum, log2 of the window.
    assert statistics_model.get_rolling_entropy(pd.Series([4.0] * 8), 8).iloc[
        -1
    ] == pytest.approx(3)
    assert np.isnan(
        statistics_model.get_rolling_entropy(pd.Series([1.0, -1.0, 2.0]), 3).iloc[-1]
    )


def test_a_missing_price_leaves_its_windows_empty():
    closes = random_prices()["Close"].copy()
    closes.iloc[40] = np.nan

    for result in (
        statistics_model.get_linear_regression(closes, 5)["Linear Regression"],
        statistics_model.get_rolling_percentile(closes, 5, 0.5),
        statistics_model.get_percent_rank(closes, 5),
        statistics_model.get_rolling_mean_absolute_deviation(closes, 5),
        statistics_model.get_rolling_entropy(closes, 5),
    ):
        assert result.iloc[40:45].isna().all()
        assert result.iloc[46:].notna().all()
