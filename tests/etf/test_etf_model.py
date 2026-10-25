"""ETF Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.etf import etf_model

# pylint: disable=missing-function-docstring


def test_concentration():
    result = etf_model.get_concentration(
        pd.Series({"A": 0.3, "B": 0.2, "C": 0.5, "D": -0.1})
    )

    assert result["Number of Holdings"] == 3  # noqa: PLR2004
    assert result["Largest Holding"] == "C"
    assert result["Herfindahl-Hirschman Index"] == pytest.approx(0.38)
    assert result["Effective Number of Holdings"] == pytest.approx(1 / 0.38)


def test_overlap_counts_the_smaller_shared_weight():
    overlap = etf_model.get_overlap(
        {"X": pd.Series({"A": 0.5, "B": 0.5}), "Y": pd.Series({"A": 0.2, "C": 0.8})}
    )

    assert overlap.loc["X", "Y"] == pytest.approx(0.2)
    assert overlap.loc["X", "X"] == pytest.approx(1)


def test_look_through_valuation_uses_harmonic_means():
    weights = pd.Series({"A": 0.3, "B": 0.2, "C": 0.5})
    ratios = pd.DataFrame(
        {"Price-to-Earnings": [10, 20, 40], "Dividend Yield": [0.03, 0.02, 0.0]},
        index=["A", "B", "C"],
    )

    result = etf_model.get_look_through_valuation(weights, ratios)

    assert result["Price-to-Earnings"] == pytest.approx(
        1 / (0.3 / 10 + 0.2 / 20 + 0.5 / 40)
    )
    # A holding without a dividend counts with a yield of zero.
    assert result["Dividend Yield"] == pytest.approx(0.3 * 0.03 + 0.2 * 0.02)
    assert result["Coverage"] == pytest.approx(1)


def test_price_impact_and_tracking():
    returns = pd.Series([0.01, -0.02, 0.01])
    dollar_volume = pd.Series([1e9, 2e9, 1e9])
    assert etf_model.get_price_impact(returns, dollar_volume) == pytest.approx(
        np.mean([0.01 / 1e9, 0.02 / 2e9, 0.01 / 1e9]) * 1e8
    )

    rng = np.random.default_rng(0)
    benchmark = pd.Series(rng.normal(0.0004, 0.01, 500))
    fund = benchmark - 0.0003 / 252
    tracking = etf_model.get_tracking(fund, benchmark)
    # The difference of the annualized returns: about the yearly cost.
    assert tracking["Tracking Difference"] == pytest.approx(-0.0003, rel=0.05)
    assert tracking["Tracking Error"] == pytest.approx(0, abs=1e-12)
    assert tracking["Beta"] == pytest.approx(1)


def test_the_tracking_difference_is_the_difference_of_annualized_returns():
    benchmark = pd.Series([0.01, -0.02, 0.015, 0.005] * 63)
    fund = benchmark - 0.0001
    tracking = etf_model.get_tracking(fund, benchmark, 252)

    expected = (1 + fund).prod() - (1 + benchmark).prod()
    assert tracking["Tracking Difference"] == pytest.approx(expected)


def test_a_fund_without_weights_has_no_concentration_or_overlap():
    empty = pd.Series(dtype=float)
    assert etf_model.get_concentration(empty).drop("Largest Holding").isna().all()

    weights = {"VOO": pd.Series({"AAPL": 0.6, "MSFT": 0.4}), "NEW": empty}
    overlap = etf_model.get_overlap(weights)
    assert overlap.loc["VOO", "VOO"] == pytest.approx(1.0)
    assert overlap["NEW"].isna().all() and overlap.loc["NEW"].isna().all()
