"""Campbell, Hilscher and Szilagyi Model Tests"""

import numpy as np
import pandas as pd

from financetoolkit.models.distress import chs_model

# pylint: disable=missing-function-docstring

TICKERS = ["AAA", "BBB"]
COLUMNS = ["2023", "2024"]


def frame(values) -> pd.DataFrame:
    return pd.DataFrame(values, index=TICKERS, columns=COLUMNS, dtype=float)


def test_get_geometric_average():
    # Equal values average to themselves, as the weights add up to one.
    values = [frame([[0.01, 0.02], [0.03, -0.01]])] * 4
    average = chs_model.get_geometric_average(values, chs_model.MONTHLY_DECAY**3)
    pd.testing.assert_frame_equal(average, values[0])

    # A single non-zero lag gets the weight (1 - d) d^k / (1 - d^n).
    decay = chs_model.MONTHLY_DECAY
    lags = [frame([[0, 0], [0, 0]]) for _ in range(12)]
    lags[2] = frame([[1, 1], [1, 1]])
    average = chs_model.get_geometric_average(lags, decay)
    assert np.isclose(average.iloc[0, 0], (1 - decay) * decay**2 / (1 - decay**12))


def test_get_adjusted_book_equity():
    adjusted = chs_model.get_adjusted_book_equity(
        book_equity=frame([[100, -500], [50, 10]]),
        market_equity=frame([[300, 100], [50, 10]]),
    )
    # 100 + 0.1 * (300 - 100) = 120; -500 + 0.1 * 600 = -440 becomes 1.
    pd.testing.assert_frame_equal(adjusted, frame([[120, 1], [50, 10]]))


def test_get_winsorized_variable():
    clipped = chs_model.get_winsorized_variable(
        frame([[10.0, 0.1], [np.nan, -1.0]]), "MB"
    )
    pd.testing.assert_frame_equal(clipped, frame([[6.471, 0.358], [np.nan, 0.358]]))


def test_get_log_price():
    log_price = chs_model.get_log_price(frame([[30, 15], [1, 0.5]]))
    pd.testing.assert_frame_equal(
        log_price, frame([[np.log(15), np.log(15)], [0, np.log(0.5)]])
    )


def test_get_sigma():
    days = pd.bdate_range("2024-01-01", "2024-06-30")
    returns = pd.DataFrame(
        {"AAA": np.resize([0.01, -0.01], len(days)), "BBB": 0.0}, index=days
    )
    dates = pd.DataFrame(
        {"2023": pd.Timestamp("2024-03-31"), "2024": pd.Timestamp("2024-06-30")},
        index=TICKERS,
    )
    sigma = chs_model.get_sigma(returns, dates)
    window = returns.loc["2024-01-01":"2024-03-31", "AAA"]
    expected = np.sqrt(252 / (len(window) - 1) * (window**2).sum())
    assert np.isclose(sigma.loc["AAA", "2023"], expected)
    # Without five non-zero returns the volatility is missing.
    assert sigma.loc["BBB"].isna().all()


def test_get_monthly_log_excess_returns():
    days = pd.date_range("2023-01-01", "2024-12-31")
    prices = pd.DataFrame(
        {"AAA": np.exp(np.arange(len(days)) * 0.001), "BBB": 1.0}, index=days
    )
    index_levels = pd.Series(np.exp(np.arange(len(days)) * 0.0005), index=days)
    dates = pd.DataFrame(
        {"2023": pd.Timestamp("2023-12-31"), "2024": pd.Timestamp("2024-12-31")},
        index=TICKERS,
    )
    excess = chs_model.get_monthly_log_excess_returns(prices, index_levels, dates)
    assert len(excess) == 12
    # December 2024 has 31 days: AAA earns 0.031 and the index 0.0155 in logs.
    assert np.isclose(excess[0].loc["AAA", "2024"], 0.031 - 0.0155)
    assert np.isclose(excess[0].loc["BBB", "2024"], -0.0155)


def test_get_index_market_value():
    days = pd.date_range("2024-01-01", periods=4)
    price_index = pd.Series([100.0, 110.0, 120.0, 125.0], index=days)
    market_value = chs_model.get_index_market_value(
        price_index, pd.Series({days[2]: 60.0})
    )
    pd.testing.assert_series_equal(
        market_value, pd.Series([50.0, 55.0, 60.0, 62.5], index=days)
    )
    assert (
        chs_model.get_index_market_value(price_index, pd.Series(dtype=float))
        .isna()
        .all()
    )


def test_get_chs_score_and_probability():
    ones = frame([[1, 1], [1, 1]])
    zeros = frame([[0, 0], [0, 0]])
    # With every variable zero the score is the constant.
    score = chs_model.get_chs_score(*[zeros] * 8)
    assert np.allclose(score, -9.164)
    # With every variable one it is the sum of all coefficients.
    score = chs_model.get_chs_score(*[ones] * 8)
    assert np.allclose(score, sum(chs_model.COEFFICIENTS.values()))
    probability = chs_model.get_chs_failure_probability(score)
    assert np.allclose(probability, 1 / (1 + np.exp(-score)))
