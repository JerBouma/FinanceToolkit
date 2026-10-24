"""Market Breadth, Seasonality and Relative Strength Tests"""

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
