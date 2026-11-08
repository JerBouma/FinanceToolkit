"""Models Helpers Tests"""

import pandas as pd
import pytest

from financetoolkit.models import helpers


def test_rates_per_period_broadcast_given_rates_and_fill_the_rest_by_period():
    periods = pd.period_range("2021", "2023", freq="Y")
    default = pd.DataFrame(
        [[0.05, 0.06, 0.07], [0.08, 0.09, 0.10]], index=["A", "B"], columns=periods
    )

    rates = helpers.rates_per_period(
        {"A": 0.04}, ["A", "B"], periods, "rate", lambda: default
    )

    assert rates.loc["A"].tolist() == [0.04, 0.04, 0.04]
    assert rates.loc["B"].tolist() == [0.08, 0.09, 0.10]
    assert (
        helpers.rates_per_period(0.03, ["A"], periods, "rate").loc["A"].tolist()
        == [0.03] * 3
    )

    with pytest.raises(ValueError, match="No rate given"):
        helpers.rates_per_period(None, ["A"], periods, "rate")


def test_exceeds_growth_per_period_leaves_out_only_those_periods():
    rates = pd.DataFrame([[0.05, 0.02, float("nan")]], index=["A"])
    growth = pd.DataFrame([[0.02, 0.03, 0.02]], index=["A"])

    assert helpers.exceeds_growth_per_period(rates, growth).loc["A"].tolist() == [
        True,
        False,
        True,
    ]


def test_trailing_within_periods_covers_the_trailing_periods():
    days = pd.period_range("2024-01-01", "2024-12-31", freq="D")
    within = pd.Series(
        range(len(days)),
        index=pd.MultiIndex.from_arrays([days.asfreq("Q"), days]),
        dtype=float,
    )

    trailing = helpers.trailing_within_periods(within, 2)

    # The first quarter has no earlier quarter; the others hold two quarters of days.
    assert len(trailing.loc[pd.Period("2024Q1")]) == len(
        within.loc[pd.Period("2024Q1")]
    )
    assert len(trailing.loc[pd.Period("2024Q3")]) == len(
        within.loc[pd.Period("2024Q2")]
    ) + len(within.loc[pd.Period("2024Q3")])
