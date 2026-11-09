"""Tests for the earnings surprise and the share price reaction to it."""

import numpy as np
import pandas as pd
import pytest

from financetoolkit import Toolkit, earnings_model, toolkit_controller


def announcements(dates, eps, estimates):
    return pd.DataFrame(
        {
            "EPS": eps,
            "Estimated EPS": estimates,
            "Revenue": [110.0] * len(dates),
            "Estimated Revenue": [100.0] * len(dates),
        },
        index=pd.DatetimeIndex(dates),
    )


def business_day_prices(start, values):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def test_the_surprises_are_fractions_of_the_absolute_estimate():
    earnings = announcements(["2024-01-10", "2024-04-10"], [1.1, -0.5], [1.0, -1.0])
    prices = business_day_prices("2024-01-01", np.linspace(100, 120, 100))

    result = earnings_model.get_earnings_surprise(earnings, prices)

    # A loss smaller than the expected loss is a positive surprise.
    assert result["EPS Surprise"].tolist() == pytest.approx([0.1, 0.5])
    assert result["Revenue Surprise"].tolist() == pytest.approx([0.1, 0.1])


def test_the_reaction_covers_the_day_before_to_the_day_after():
    # Wednesday 2024-01-10 is day 0: the reaction runs from Monday's close (before day -1,
    # Tuesday) to Thursday's close (day +1).
    prices = business_day_prices("2024-01-01", [100.0 + day for day in range(20)])
    benchmark = business_day_prices("2024-01-01", [50.0] * 20)
    earnings = announcements(["2024-01-10"], [1.1], [1.0])

    result = earnings_model.get_earnings_surprise(earnings, prices, benchmark)

    monday, thursday = prices.loc["2024-01-08"], prices.loc["2024-01-11"]
    assert result["Reaction"].iloc[0] == pytest.approx(thursday / monday - 1)
    assert result["Benchmark Reaction"].iloc[0] == 0
    assert result["Abnormal Reaction"].iloc[0] == pytest.approx(thursday / monday - 1)


def test_an_announcement_on_a_weekend_reacts_from_the_next_trading_day():
    prices = business_day_prices("2024-01-01", [100.0 + day**2 for day in range(20)])
    saturday = announcements(["2024-01-13"], [1.1], [1.0])
    monday = announcements(["2024-01-15"], [1.1], [1.0])

    assert (
        earnings_model.get_earnings_surprise(saturday, prices)["Reaction"].iloc[0]
        == earnings_model.get_earnings_surprise(monday, prices)["Reaction"].iloc[0]
    )


def test_an_announcement_without_enough_prices_around_it_has_no_reaction():
    prices = business_day_prices("2024-01-01", [100.0 + day for day in range(10)])
    earnings = announcements(["2024-01-01", "2024-01-12"], [1.1, 1.1], [1.0, 1.0])

    result = earnings_model.get_earnings_surprise(earnings, prices)

    assert result["Reaction"].isna().all()


def test_the_sue_is_scaled_by_the_earlier_surprises_only():
    dates = pd.date_range("2020-01-15", periods=6, freq="QS").shift(14, freq="D")
    differences = [0.1, -0.1, 0.2, 0.0, 0.3, 0.5]
    earnings = announcements(dates, [1.0 + d for d in differences], [1.0] * 6)
    prices = business_day_prices("2019-12-01", np.linspace(100, 150, 600))

    result = earnings_model.get_earnings_surprise(earnings, prices, sue_window=4)

    assert result["SUE"].iloc[:4].isna().all()
    assert result["SUE"].iloc[4] == pytest.approx(0.3 / np.std(differences[:4], ddof=1))
    assert result["SUE"].iloc[5] == pytest.approx(
        0.5 / np.std(differences[1:5], ddof=1)
    )


def test_announcements_that_are_not_reported_yet_are_left_out():
    earnings = announcements(["2024-01-10", "2024-04-10"], [1.1, np.nan], [1.0, 1.2])
    prices = business_day_prices("2024-01-01", np.linspace(100, 120, 100))

    result = earnings_model.get_earnings_surprise(earnings, prices)

    assert result.index.tolist() == [pd.Timestamp("2024-01-10")]


def test_the_toolkit_scales_the_sue_with_reports_before_its_start_date(
    monkeypatch, known_fmp_plan
):
    dates = pd.bdate_range("2023-01-02", "2024-12-31")
    columns = pd.MultiIndex.from_product(
        [["Adj Close", "Volume"], ["AAPL", "Benchmark"]]
    )
    historical = pd.DataFrame(1.0, index=dates, columns=columns)
    historical[("Adj Close", "AAPL")] = np.linspace(100, 200, len(dates))
    historical.index = historical.index.to_period("D")

    report_dates = pd.date_range("2022-02-01", periods=12, freq="QS").shift(
        14, freq="D"
    )
    calendar = announcements(
        report_dates,
        np.linspace(1.0, 2.0, 12) + [0.1, -0.1] * 6,
        np.linspace(1.0, 2.0, 12),
    )
    requested = []

    def fake_calendar(tickers, start_date=None, **kwargs):
        requested.append(start_date)
        return pd.concat({"AAPL": calendar}), []

    monkeypatch.setattr(toolkit_controller, "_get_earnings_calendar", fake_calendar)
    toolkit = Toolkit(
        tickers=["AAPL"],
        api_key="KEY",
        historical=historical,
        start_date="2024-01-01",
        end_date="2024-12-31",
        convert_currency=False,
        sleep_timer=False,
    )
    # A risk-free rate up front keeps the test offline: otherwise the Toolkit downloads the
    # Treasury rates, whose parallel yfinance requests can lock yfinance's SQLite cache.
    toolkit._daily_risk_free_rate = pd.Series(0.0, index=historical.index)

    result = toolkit.get_earnings_surprise(sue_window=4)

    assert requested == [None]
    assert result.index.min() >= pd.Timestamp("2024-01-01")
    assert result["SUE"].notna().all()
    assert result["Reaction"].notna().all()
