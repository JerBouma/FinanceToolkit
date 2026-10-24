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
    sectors = pd.DataFrame(
        {"AAA": [0.2, 0.8, None], "BBB": [0.5, 0.3, 0.2]},
        index=pd.Index(["Energy", "Technology", "Utilities"], name="Sector"),
    )
    datasets = {
        "information": information,
        "holdings": holdings,
        "sector_weightings": sectors,
        "country_weightings": pd.DataFrame(),
    }
    return ETF(
        tickers=["AAA", "BBB", "Benchmark"],
        historical_function=lambda: daily,
        fund_data_function=lambda dataset, overwrite=False: datasets[dataset],
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


def test_holdings_and_weightings(etf):
    largest = etf.get_holdings(limit=1)
    sectors = etf.get_sector_weightings()

    assert list(largest["Weight"]) == [0.5, 0.6]
    assert list(etf.get_holdings(limit=-1)["Weight"]) == [0.2, 0.4]
    # The largest weights of the first fund come first, and a sector a fund does not hold
    # weighs zero in it.
    assert list(sectors.index) == ["Technology", "Energy", "Utilities"]
    assert sectors.loc["Utilities", "AAA"] == 0
    assert etf.get_country_weightings().empty


def test_a_single_fund(etf):
    single = ETF(
        tickers=["AAA"],
        historical_function=etf._historical_function,
        fund_data_function=etf._fund_data_function,
        ratios_function=etf._ratios_function,
    )

    # The holdings of a single fund are indexed by the asset only.
    assert list(single.get_holdings().index) == ["X", "Y", "Z"]
    assert single.get_concentration().loc["Largest Holding", "AAA"] == "X"


def test_the_holdings_need_no_prices(etf):
    def no_prices():
        raise AssertionError("The prices were retrieved.")

    etf._historical_function = no_prices

    assert not etf.get_holdings().empty
    assert not etf.get_concentration().empty


def test_the_premium_compares_the_nav_with_the_close_of_its_date(etf):
    information = etf._fund_data_function("information").copy()
    close = etf._historical()["Close"]["AAA"]
    # A NAV published after midnight UTC is the previous New York day's, equal to that
    # day's close, three trading days before the last close.
    nav_day = close.index[-4].to_timestamp()
    information.loc["NAV", "AAA"] = float(close.iloc[-4])
    information.loc["Updated At", "AAA"] = (
        nav_day + pd.Timedelta(days=1, hours=2)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    datasets = {
        "information": information,
        "holdings": etf._fund_data_function("holdings"),
    }
    etf._fund_data_function = lambda dataset, overwrite=False: datasets[dataset]

    profile = etf.get_profile(rounding=8)

    assert profile.loc["Premium/Discount", "AAA"] == pytest.approx(0, abs=1e-8)
    assert profile.loc["NAV Date", "AAA"] == nav_day.strftime("%Y-%m-%d")


def with_information(etf, **fields):
    """The ETF with fields of fund AAA's information replaced."""
    information = etf._fund_data_function("information").copy()
    for field, value in fields.items():
        information.loc[field.replace("_", " "), "AAA"] = value
    datasets = {
        "information": information,
        "holdings": etf._fund_data_function("holdings"),
    }
    etf._fund_data_function = lambda dataset, overwrite=False: datasets[dataset]
    return etf


def test_a_nav_updated_before_the_open_is_of_the_previous_close(etf):
    close = etf._historical()["Close"]["AAA"]
    # An update at 09:00 New York time (13:00 UTC in summer, 14:00 in winter) comes before
    # that day's close, so the NAV is of the previous trading day's close.
    nav_day = close.index[-2].to_timestamp()
    update_day = close.index[-1].to_timestamp()
    updated = (
        (update_day + pd.Timedelta(hours=9))
        .tz_localize("America/New_York")
        .tz_convert("UTC")
    )
    with_information(
        etf,
        NAV=float(close.iloc[-2]),
        Updated_At=updated.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    profile = etf.get_profile(rounding=8)

    assert profile.loc["NAV Date", "AAA"] == nav_day.strftime("%Y-%m-%d")
    assert profile.loc["Premium/Discount", "AAA"] == pytest.approx(0, abs=1e-8)


def test_a_date_only_update_is_the_nav_date(etf):
    close = etf._historical()["Close"]["AAA"]
    day = close.index[-1].to_timestamp().strftime("%Y-%m-%d")
    with_information(etf, Updated_At=day)

    assert etf.get_profile().loc["NAV Date", "AAA"] == day


def test_no_premium_against_a_nav_in_another_currency(etf):
    etf._currency_function = lambda: pd.Series({"AAA": "USD", "BBB": "USD"})
    with_information(etf, NAV_Currency="EUR")

    profile = etf.get_profile()

    assert pd.isna(profile.loc["Premium/Discount", "AAA"])
    assert profile.loc["Premium/Discount", "BBB"] == pytest.approx(0.01, abs=1e-4)


def test_a_close_in_pence_is_compared_in_pounds(etf):
    close = etf._historical()["Close"]["AAA"]
    # The close is quoted in pence (GBp) and the NAV in pounds, a hundredth of it.
    etf._currency_function = lambda: pd.Series({"AAA": "GBp"})
    with_information(etf, NAV=float(close.iloc[-1]) / 100, NAV_Currency="GBP")

    assert etf.get_profile(rounding=8).loc["Premium/Discount", "AAA"] == pytest.approx(
        0, abs=1e-8
    )


def test_an_implausible_premium_is_left_out_with_a_warning(etf, caplog):
    close = etf._historical()["Close"]["AAA"]
    # Without a known listing currency a NAV in another unit shows as a premium of 99%.
    with_information(etf, NAV=float(close.iloc[-1]) / 100)

    with caplog.at_level("WARNING"):
        profile = etf.get_profile()

    assert pd.isna(profile.loc["Premium/Discount", "AAA"])
    assert "AAA" in caplog.text


def test_a_failing_currency_lookup_keeps_the_premium(etf):
    def fail():
        raise ValueError("no statistics")

    etf._currency_function = fail

    assert etf.get_profile().loc["Premium/Discount", "BBB"] == pytest.approx(
        0.01, abs=1e-4
    )
