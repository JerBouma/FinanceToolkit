"""ETF Controller Tests, on synthetic funds."""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.etf.etf_controller import ETF

# pylint: disable=missing-function-docstring


@pytest.fixture(name="etf")
def fixture_etf():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2025-01-01", periods=300).to_period("D")
    benchmark = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 300)))
    closes = {
        "AAA": benchmark * np.exp(np.cumsum(rng.normal(0, 0.0005, 300))),
        "BBB": 50 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, 300))),
        "Benchmark": benchmark,
    }
    frames = {}
    for column in ("Open", "High", "Low", "Close", "Adj Close"):
        frames[column] = pd.DataFrame(closes, index=dates)
    frames["Volume"] = pd.DataFrame(1e6, index=dates, columns=list(closes))
    daily = pd.concat(frames, axis=1)

    information = pd.DataFrame(
        {
            "AAA": {
                "Name": "Fund A",
                "Expense Ratio": 0.0003,
                "NAV": float(closes["AAA"][-1]),
                "Assets Under Management": 1e10,
                "Updated At": str(dates[-1]),
                "Holdings Count": 3,
            },
            "BBB": {
                "Name": "Fund B",
                "Expense Ratio": 0.005,
                "NAV": float(closes["BBB"][-1]) / 1.01,
                "Assets Under Management": 1e8,
                "Updated At": str(dates[-1]),
                "Holdings Count": 2,
            },
        }
    )
    holdings = pd.DataFrame(
        {"Weight": [0.5, 0.3, 0.2, 0.6, 0.4], "ISIN": ["I1", "I2", "I3", "I4", "I5"]},
        index=pd.MultiIndex.from_tuples(
            [("AAA", "X"), ("AAA", "Y"), ("AAA", "Z"), ("BBB", "X"), ("BBB", "")],
            names=[None, "Asset"],
        ),
    )
    ratios = pd.DataFrame(
        {
            "X": {"Price-to-Earnings": 20.0, "Dividend Yield": 0.01},
            "Y": {"Price-to-Earnings": 10.0, "Dividend Yield": 0.03},
            "Z": {"Price-to-Earnings": 40.0, "Dividend Yield": 0.0},
        }
    )
    return ETF(
        tickers=["AAA", "BBB", "Benchmark"],
        daily_historical=daily,
        information_function=lambda: information,
        holdings_function=lambda: holdings,
        ratios_function=lambda tickers: ratios[[t for t in tickers if t in ratios]],
        start_date="2025-01-01",
        end_date="2026-12-31",
    )


def test_profile_and_premium(etf):
    profile = etf.get_profile()

    assert list(profile.columns) == ["AAA", "BBB"]
    assert profile.loc["Premium/Discount", "AAA"] == pytest.approx(0)
    assert profile.loc["Premium/Discount", "BBB"] == pytest.approx(0.01, abs=1e-4)


def test_tracking(etf):
    tracking = etf.get_tracking()
    yearly = etf.get_tracking(yearly=True)

    assert tracking.loc["Correlation", "AAA"] > 0.99  # noqa: PLR2004
    assert tracking.loc["Tracking Error", "AAA"] < tracking.loc["Tracking Error", "BBB"]
    assert list(yearly.columns) == ["AAA", "BBB"]


def test_concentration_overlap_and_look_through(etf):
    concentration = etf.get_concentration()
    overlap = etf.get_overlap()
    valuation = etf.get_look_through_valuation()

    assert concentration.loc["Largest Holding", "AAA"] == "X"
    # The holding of BBB without a symbol is identified by its ISIN.
    assert concentration.loc["Number of Holdings", "BBB"] == 2  # noqa: PLR2004
    assert overlap.loc["AAA", "BBB"] == pytest.approx(0.5)
    assert valuation.loc["Price-to-Earnings", "AAA"] == pytest.approx(
        1 / (0.5 / 20 + 0.3 / 10 + 0.2 / 40), abs=1e-4
    )
    assert valuation.loc["Coverage", "BBB"] == pytest.approx(0.6)


def test_liquidity(etf):
    liquidity = etf.get_liquidity()

    assert liquidity.loc["Average Volume", "AAA"] == pytest.approx(1e6)
    assert (
        liquidity.loc["Dollar Volume to Assets", "BBB"]
        > liquidity.loc["Dollar Volume to Assets", "AAA"]
    )


def test_a_single_fund(etf):
    holdings = etf._holdings_function().loc["AAA"]
    single = ETF(
        tickers=["AAA"],
        daily_historical=etf._daily_historical,
        information_function=etf._information_function,
        holdings_function=lambda: holdings,
        ratios_function=etf._ratios_function,
    )

    assert single.get_concentration().loc["Largest Holding", "AAA"] == "X"
