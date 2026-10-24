"""Market Breadth, Seasonality and Relative Strength Tests"""

import copy

import numpy as np
import pandas as pd
import pytest

from financetoolkit.technicals import breadth_model, seasonality_model

# pylint: disable=missing-function-docstring


@pytest.fixture(name="prices")
def fixture_prices():
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2020-01-01", periods=600).to_period("D")
    return pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (600, 20)), axis=0)),
        index=index,
        columns=[f"T{number}" for number in range(20)],
    )


def test_market_breadth_counts_the_constituents(prices):
    breadth = breadth_model.get_market_breadth(prices)

    assert (
        breadth["Advancers"] + breadth["Decliners"] + breadth["Unchanged"] == 20
    ).all()
    assert breadth["Advance-Decline Line"].iloc[-1] == breadth["Net Advances"].sum()
    assert breadth["Percentage Above 200-Period Average"].dropna().between(0, 1).all()
    assert np.isnan(breadth["Percentage Above 200-Period Average"].iloc[100])


def test_seasonality_per_month_and_weekday(prices):
    monthly = prices["T0"].groupby(prices.index.asfreq("M")).last().pct_change()

    months = seasonality_model.get_seasonality(monthly)
    weekdays = seasonality_model.get_seasonality(prices["T0"].pct_change(), "weekday")

    assert list(months.index[:2]) == ["January", "February"]
    assert months["Observations"].sum() == monthly.notna().sum()
    assert list(weekdays.index) == [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
    ]
    with pytest.raises(ValueError, match="frequency"):
        seasonality_model.get_seasonality(monthly, "year")


def test_relative_strength_and_rating(prices):
    relative, mansfield = seasonality_model.get_relative_strength(
        prices, prices.mean(axis=1), window=50
    )
    rating = seasonality_model.get_relative_strength_rating(prices)

    assert relative.iloc[0].to_numpy() == pytest.approx(np.ones(20))
    assert mansfield.iloc[:49].isna().all().all()
    # The ratings of a date run from 1 to 99, the weakest to the strongest.
    assert rating.iloc[-1].min() == 1 and rating.iloc[-1].max() == 99  # noqa: PLR2004
    assert rating.iloc[200].isna().all()


def test_new_technicals_on_the_toolkit(technical_module):
    breadth = technical_module.get_market_breadth()
    assert {"Advancers", "McClellan Oscillator"} <= set(breadth.columns)

    seasonality = technical_module.get_seasonality()
    assert ("Average Return", "AAPL") in seasonality.columns

    relative_strength = technical_module.get_relative_strength()
    assert set(relative_strength.columns.get_level_values(0)) == {
        "Relative Strength",
        "Mansfield Relative Strength",
        "Relative Strength Rating",
    }


def seeded_ema(values: list[float], span: int) -> list[float]:
    """The exponential moving average seeded with the simple average of the first span values."""
    alpha = 2 / (span + 1)
    averages = [np.nan] * (span - 1) + [float(np.mean(values[:span]))]
    for value in values[span:]:
        averages.append(alpha * value + (1 - alpha) * averages[-1])
    return averages


def test_mcclellan_oscillator_is_ratio_adjusted_and_seeded():
    rng = np.random.default_rng(4)
    index = pd.bdate_range("2021-01-01", periods=121).to_period("D")
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (121, 6)), axis=0)),
        index=index,
        columns=[f"T{number}" for number in range(6)],
    )
    # Two tickers that never move: unchanged issues stay out of the denominator.
    prices["Flat"] = prices["Still"] = 50.0

    breadth = breadth_model.get_market_breadth(prices)

    advancers, decliners = breadth["Advancers"], breadth["Decliners"]
    ratio_adjusted = ((advancers - decliners) / (advancers + decliners) * 1000).tolist()
    expected = np.array(seeded_ema(ratio_adjusted, 19)) - np.array(
        seeded_ema(ratio_adjusted, 39)
    )

    assert (breadth["Unchanged"] == 2).all()
    assert breadth["McClellan Oscillator"].to_numpy() == pytest.approx(
        expected, nan_ok=True
    )
    # The oscillator and its running total start once the 39-period average exists.
    assert breadth["McClellan Oscillator"].iloc[:38].isna().all()
    assert breadth["McClellan Oscillator"].iloc[38:].notna().all()
    assert breadth["McClellan Summation Index"].iloc[-1] == pytest.approx(
        np.nansum(expected)
    )


def test_ratio_adjusted_net_advances_without_advancers_or_decliners():
    index = pd.bdate_range("2021-01-01", periods=60).to_period("D")
    rising = pd.Series(np.arange(60, dtype=float) + 100, index=index)
    prices = pd.DataFrame({"Up": rising, "Flat": 100.0})
    # Every ticker unchanged on one day: no net advance rather than a missing value.
    prices.iloc[30, 0] = prices.iloc[29, 0]

    breadth = breadth_model.get_market_breadth(prices)

    # 1,000 on every day but one, so the averages dip once instead of breaking off.
    assert breadth["McClellan Oscillator"].iloc[38:].notna().all()
    assert breadth["McClellan Oscillator"].iloc[-1] != 0


def test_new_highs_need_a_close_above_the_previous_high():
    index = pd.bdate_range("2021-01-01", periods=30).to_period("D")
    # A new high on day 15, which the close then only equals; a new low on day 25.
    path = [100.0] * 10 + [101, 102, 103, 104, 105, 106] + [106.0] * 9 + [90.0] * 5
    prices = pd.DataFrame({"A": path, "B": np.linspace(50, 60, 30)}, index=index)

    breadth = breadth_model.get_market_breadth(prices, high_low_window=10)

    new_highs = breadth["New Highs"]
    # NaN, not zero, until the first full window of 10 closes (the result starts a day in).
    assert new_highs.iloc[:8].isna().all()
    assert breadth["New Lows"].iloc[:8].isna().all()
    # B rises every day, A only makes a new high on the days its close goes up.
    assert new_highs.loc[index[15]] == 2
    assert (new_highs.loc[index[16] : index[24]] == 1).all()
    assert breadth["New Lows"].loc[index[25]] == 1
    assert breadth["New Lows"].loc[index[26]] == 0


def test_seasonality_leaves_out_a_month_that_is_not_complete():
    index = pd.period_range("2020-01", "2021-06", freq="M")
    returns = pd.Series(0.01, index=index)
    returns.iloc[-1] = 0.5

    partial = seasonality_model.get_seasonality(returns, end_date="2021-06-15")
    complete = seasonality_model.get_seasonality(returns, end_date="2021-06-30")
    without_end = seasonality_model.get_seasonality(returns)

    assert partial.loc["June", "Observations"] == 1
    assert partial.loc["June", "Average Return"] == pytest.approx(0.01)
    assert complete.loc["June", "Observations"] == 2
    assert without_end.loc["June", "Observations"] == 2
    # A month that ends on a weekend is complete from its last business day on.
    returns.index = pd.period_range("2020-02", "2021-07", freq="M")
    assert (
        seasonality_model.get_seasonality(returns, end_date="2021-07-30").loc[
            "July", "Observations"
        ]
        == 2
    )


def test_relative_strength_rating_gives_tied_leaders_99():
    index = pd.bdate_range("2020-01-01", periods=260).to_period("D")
    growth = np.linspace(0, 1, 260)
    prices = pd.DataFrame(
        {
            "Leader": 100 * np.exp(growth),
            "Twin": 100 * np.exp(growth),
            "Middle": 100 * np.exp(0.5 * growth),
            "Laggard": 100 * np.exp(-growth),
        },
        index=index,
    )

    rating = seasonality_model.get_relative_strength_rating(prices).iloc[-1]

    assert rating["Leader"] == rating["Twin"] == 99
    assert rating["Laggard"] == 1
    assert rating["Middle"] == round(1 + 1 / 3 * 98)


def test_windows_default_to_a_year_of_periods(technical_module):
    weekly = technical_module.get_relative_strength(period="weekly")
    explicit = technical_module.get_relative_strength(period="weekly", window=52)

    pd.testing.assert_frame_equal(weekly, explicit)
    assert weekly["Mansfield Relative Strength"].notna().any().all()

    breadth = technical_module.get_market_breadth(period="weekly")
    pd.testing.assert_frame_equal(
        breadth,
        technical_module.get_market_breadth(period="weekly", high_low_window=52),
    )
    assert breadth["New Highs"].notna().any()


def test_seasonality_on_the_toolkit_leaves_out_an_incomplete_month(technical_module):
    module = copy.copy(technical_module)
    module._end_date = "2022-12-15"

    complete = technical_module.get_seasonality()
    partial = module.get_seasonality()

    assert ("Average Return", "Benchmark") in complete.columns
    assert (
        partial.loc["December", ("Observations", "AAPL")]
        == complete.loc["December", ("Observations", "AAPL")] - 1
    )
    assert (
        partial.loc["November", ("Observations", "AAPL")]
        == complete.loc["November", ("Observations", "AAPL")]
    )
