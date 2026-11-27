"""Historical Model Tests"""

# ruff: noqa


import pandas as pd
import pytest

from financetoolkit import historical_model


def _frame(start: str, end: str) -> pd.DataFrame:
    """A daily historical frame spanning the given dates, as the providers return."""
    index = pd.PeriodIndex(pd.date_range(start=start, end=end, freq="D"), freq="D")

    return pd.DataFrame({"Adj Close": 1.0}, index=index)


def test_covers_requested_range_accepts_a_full_response():
    data = _frame("2005-01-01", "2025-01-01")

    assert historical_model._covers_requested_range(data, "2005-01-01", "2025-01-01")


def test_covers_requested_range_rejects_a_drastically_short_span():
    # Three months returned where twenty years were asked for.
    data = _frame("2024-10-01", "2025-01-01")

    assert not historical_model._covers_requested_range(
        data, "2005-01-01", "2025-01-01"
    )


def test_covers_requested_range_rejects_a_row_capped_response():
    """The response runs to the requested end but is missing history at the start.

    FinancialModelingPrep caps at a fixed row count rather than a fixed span, so a
    capped response spans ~99% of the request and is only short at the beginning.
    Comparing span lengths alone does not catch it.
    """
    data = _frame("2006-02-16", "2025-01-01")

    assert not historical_model._covers_requested_range(
        data, "2005-01-01", "2025-01-01"
    )


def test_covers_requested_range_tolerates_a_start_on_a_non_trading_day():
    # 2022-01-01 was a Saturday; the first bar is the following Monday.
    data = _frame("2022-01-03", "2025-01-01")

    assert historical_model._covers_requested_range(data, "2022-01-01", "2025-01-01")


def test_covers_requested_range_skips_short_requests():
    data = _frame("2024-06-10", "2024-06-14")

    assert historical_model._covers_requested_range(data, "2024-06-01", "2024-06-14")


def test_covers_requested_range_handles_missing_dates_and_empty_data():
    data = _frame("2024-01-01", "2024-06-01")

    assert historical_model._covers_requested_range(data, None, "2025-01-01")
    assert historical_model._covers_requested_range(data, "2005-01-01", None)
    assert historical_model._covers_requested_range(
        pd.DataFrame(), "2005-01-01", "2025-01-01"
    )


def test_reaches_further_back_prefers_the_longer_history():
    yahoo = _frame("2005-01-03", "2025-01-01")
    truncated_fmp = _frame("2006-02-16", "2025-01-01")

    assert historical_model._reaches_further_back(yahoo, truncated_fmp)


def test_reaches_further_back_rejects_an_equally_short_history():
    """A young ticker yields the same short history from both providers.

    In that case there is nothing to gain by switching provider, so the original
    response is kept rather than replaced.
    """
    yahoo = _frame("2024-03-21", "2025-01-01")
    truncated_fmp = _frame("2024-03-21", "2025-01-01")

    assert not historical_model._reaches_further_back(yahoo, truncated_fmp)


def test_reaches_further_back_rejects_an_empty_candidate():
    truncated_fmp = _frame("2006-02-16", "2025-01-01")

    assert not historical_model._reaches_further_back(pd.DataFrame(), truncated_fmp)


def test_reaches_further_back_accepts_anything_over_nothing():
    yahoo = _frame("2024-03-21", "2025-01-01")

    assert historical_model._reaches_further_back(yahoo, pd.DataFrame())


def test_reaches_further_back_compares_across_index_types():
    """PeriodIndex and DatetimeIndex frames must stay comparable."""
    yahoo = _frame("2005-01-03", "2025-01-01")
    truncated_fmp = _frame("2006-02-16", "2025-01-01")
    truncated_fmp.index = truncated_fmp.index.to_timestamp()

    assert historical_model._reaches_further_back(yahoo, truncated_fmp)


def test_fill_non_trading_days_keeps_each_tickers_own_trading_days():
    """A holiday on one exchange gets the last price as a flat bar, no volume and no return."""
    index = pd.period_range("2024-01-01", periods=5, freq="D")
    a_close = [100.0, 101.0, 103.0, 104.0]
    frames = {
        # A has no row for 3 January, B traded that day.
        "A": pd.DataFrame(
            {
                "Open": [99.0, 100.5, 102.0, 103.5],
                "High": [101.0, 102.0, 104.0, 105.0],
                "Low": [98.0, 100.0, 101.0, 103.0],
                "Close": a_close,
                "Adj Close": a_close,
                "Volume": 10.0,
            },
            index=index.delete(2),
        ),
        "B": pd.DataFrame(
            {
                column: [50.0, 51, 52, 53, 54]
                for column in ("Open", "High", "Low", "Close", "Adj Close")
            }
            | {"Volume": 5.0},
            index=index,
        ),
    }

    for frame in frames.values():
        frame["Return"] = frame["Adj Close"].pct_change()

    historical_data = pd.concat(frames).unstack(level=0)

    filled = historical_model.fill_non_trading_days(historical_data)

    # The last price, not one interpolated from the next trading day (look-ahead).
    holiday = filled.loc[index[2]].xs("A", level=1)
    assert holiday[["Open", "High", "Low", "Close", "Adj Close"]].eq(101.0).all()
    assert holiday["Volume"] == 0
    assert pd.isna(holiday["Return"])
    # The next trading day keeps the move since A's previous trading day, 101 to 103, so A
    # has the same returns as without B, and B is untouched.
    pd.testing.assert_series_equal(
        filled[("Return", "A")].dropna(),
        frames["A"]["Return"].dropna(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        filled[("Return", "B")], historical_data[("Return", "B")]
    )


def test_a_ticker_without_data_does_not_break_the_others(monkeypatch):
    # The ticker without data is filled with placeholder rows that are no dates, which
    # must be gone before the history is put in date order.
    def fmp(ticker, start, end, interval, api_key, **kwargs):
        if ticker == "NOTATICKER":
            return pd.DataFrame()
        index = pd.PeriodIndex(
            pd.date_range("2024-01-01", periods=3, freq="D"), freq="D"
        )
        return pd.DataFrame(
            {
                "Open": 1.0,
                "High": 1.0,
                "Low": 1.0,
                "Close": [1.0, 2.0, 3.0],
                "Adj Close": [1.0, 2.0, 3.0],
                "Volume": 1,
                "Dividends": 0.0,
            },
            index=index,
        )

    monkeypatch.setattr(historical_model.fmp_model, "get_historical_data", fmp)
    # Yahoo Finance answers an unknown ticker with a frame of numbered rows, not dates.
    monkeypatch.setattr(
        historical_model.yfinance_model,
        "get_historical_data",
        lambda *args, **kwargs: pd.DataFrame(
            {
                column: [float("nan")]
                for column in ("Open", "High", "Low", "Close", "Adj Close", "Volume")
            }
        ),
    )

    data, no_data = historical_model.get_historical_data(
        ["AAPL", "NOTATICKER"],
        api_key="KEY",
        start="2024-01-01",
        end="2024-01-10",
        sleep_timer=False,
    )

    assert data.index.is_monotonic_increasing
    assert data["Close"]["AAPL"].tolist()[:3] == [1.0, 2.0, 3.0]


def test_calculate_period_average_averages_the_observed_days():
    """Each period is the mean of its own dates; days without data carry no weight."""
    dates = pd.PeriodIndex(
        ["2023-01-30", "2023-01-31", "2023-02-01", "2023-02-28"], freq="D"
    )
    daily = pd.DataFrame(
        {
            ("Adj Close", "^TNX"): [0.03, 0.05, 0.04, 0.02],
            ("Return", "^TNX"): [0.0, 0.1, 0.2, 0.3],
        },
        index=dates,
    )

    average = historical_model.calculate_period_average(
        "monthly", daily, columns=["Adj Close"]
    )

    assert list(average.columns) == [("Adj Close", "^TNX")]
    assert average[("Adj Close", "^TNX")].tolist() == pytest.approx([0.04, 0.03])
    assert str(average.index[0]) == "2023-01"

    with pytest.raises(ValueError):
        historical_model.calculate_period_average("daily", daily)


def test_period_aggregation_matches_an_aggregation_per_column():
    historical = pd.read_pickle("tests/datasets/historical_dataset.pickle")
    aggregations = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Volume": "sum",
        "Dividends": "sum",
    }

    for period in ["weekly", "monthly", "quarterly", "yearly"]:
        dates = historical_model.to_period_index(historical.index).asfreq(
            historical_model.PERIOD_TRANSLATION[period]
        )
        expected = historical.groupby(dates.rename("Date")).agg(
            {column: aggregations.get(column[0], "last") for column in historical}
        )
        converted = historical_model.convert_daily_to_other_period(period, historical)

        assert list(converted.columns) == list(historical.columns)

        for column in ["Open", "High", "Low", "Close", "Volume", "Dividends"]:
            pd.testing.assert_frame_equal(converted[column], expected[column])


def test_period_conversion_keeps_the_column_level_names():
    columns = pd.MultiIndex.from_product(
        [
            ["Open", "Close", "Adj Close", "Volume", "Dividends", "Return"],
            ["AAPL", "MSFT"],
        ],
        names=["Field", "Ticker"],
    )
    daily = pd.DataFrame(
        1.0, index=pd.period_range("2024-01-01", periods=70, freq="D"), columns=columns
    )

    monthly = historical_model.convert_daily_to_other_period("monthly", daily)

    assert list(monthly.columns.names) == ["Field", "Ticker"]


def test_a_single_ticker_string_and_default_dates_are_accepted(monkeypatch):
    requested = []

    def fmp(ticker, start, end, interval, api_key, **kwargs):  # noqa: ARG001
        requested.append((start, end))
        return pd.DataFrame(
            {
                "Open": 1.0,
                "High": 1.0,
                "Low": 1.0,
                "Close": 2.0,
                "Adj Close": 2.0,
                "Volume": 1,
                "Dividends": 0.0,
            },
            index=pd.period_range(start, end, freq="D"),
        )

    monkeypatch.setattr(historical_model.fmp_model, "get_historical_data", fmp)
    monkeypatch.setattr(
        historical_model.yfinance_model,
        "get_historical_data",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    data, no_data = historical_model.get_historical_data(
        "AAPL", api_key="KEY", sleep_timer=False
    )

    start, end = requested[0]
    # Ten years up to today unless dates are given.
    assert pd.Timestamp(end) - pd.Timestamp(start) == pd.Timedelta(days=10 * 365)
    assert not data["Close"]["AAPL"].dropna().empty
    assert (data["Close"]["AAPL"].dropna() == 2.0).all()
    assert no_data == []


def test_a_ticker_without_data_adds_no_calendar_days(monkeypatch):
    # A Friday and the Monday after it: the ticker without data must not add the weekend,
    # over which the other ticker's prices would then be carried.
    trading_days = pd.PeriodIndex(["2024-01-05", "2024-01-08"], freq="D")

    def fmp(ticker, start, end, interval, api_key, **kwargs):
        if ticker == "NOTATICKER":
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "Open": 1.0,
                "High": 1.0,
                "Low": 1.0,
                "Close": [1.0, 2.0],
                "Adj Close": [1.0, 2.0],
                "Volume": 1,
                "Dividends": 0.0,
            },
            index=trading_days,
        )

    monkeypatch.setattr(historical_model.fmp_model, "get_historical_data", fmp)
    monkeypatch.setattr(
        historical_model.yfinance_model,
        "get_historical_data",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    data, no_data = historical_model.get_historical_data(
        ["AAPL", "NOTATICKER"],
        api_key="KEY",
        start="2024-01-05",
        end="2024-01-08",
        sleep_timer=False,
    )

    assert no_data == ["NOTATICKER"]
    assert data.index.tolist() == trading_days.tolist()
    assert data["Close"]["NOTATICKER"].isna().all()


def test_period_open_high_low_leave_out_the_dates_a_ticker_did_not_trade():
    """A stock's week that starts on a holiday, filled with a flat bar at the previous close
    because another ticker traded that day, opens on its first trading day, and its High and
    Low come from its trading days."""
    index = pd.period_range("2024-01-01", "2024-01-21", freq="D")
    crypto_close = [float(10 + day) for day in range(len(index))]
    # The stock trades on weekdays except Monday 8 January, and not at all in the third week.
    stock_days = [
        day for day in index if day.dayofweek < 5 and day.day != 8 and day.day < 15
    ]
    stock_close = [100.0 + 2 * position for position in range(len(stock_days))]
    frames = {
        "STOCK": pd.DataFrame(
            {
                "Open": [close - 1 for close in stock_close],
                "High": [close + 3 for close in stock_close],
                "Low": [close - 3 for close in stock_close],
                "Close": stock_close,
                "Adj Close": stock_close,
                "Volume": 10.0,
            },
            index=pd.PeriodIndex(stock_days, freq="D"),
        ),
        "BTC": pd.DataFrame(
            {
                column: crypto_close
                for column in ("Open", "High", "Low", "Close", "Adj Close")
            }
            | {"Volume": 5.0},
            index=index,
        ),
    }
    for frame in frames.values():
        frame["Return"] = frame["Adj Close"].pct_change()

    daily = historical_model.fill_non_trading_days(
        pd.concat(frames).unstack(level=0).reindex(index)
    )
    weekly = historical_model.convert_daily_to_other_period("weekly", daily)

    second_week = pd.Period("2024-01-08", freq="W")
    traded = frames["STOCK"].loc["2024-01-09":"2024-01-12"]
    assert weekly.loc[second_week, ("Open", "STOCK")] == traded["Open"].iloc[0]
    assert weekly.loc[second_week, ("High", "STOCK")] == traded["High"].max()
    assert weekly.loc[second_week, ("Low", "STOCK")] == traded["Low"].min()
    assert weekly.loc[second_week, ("Close", "STOCK")] == traded["Close"].iloc[-1]

    # Past the stock's last price nothing is filled, so the third week has no bar.
    assert (
        weekly.loc[pd.Period("2024-01-15", freq="W"), ["Open", "High", "Low"]]
        .xs("STOCK", level=1)
        .isna()
        .all()
    )
    # The cryptocurrency traded every day and is aggregated as it is.
    assert weekly.loc[second_week, ("Open", "BTC")] == crypto_close[7]
    assert weekly.loc[second_week, ("Low", "BTC")] == crypto_close[7]


def test_the_portfolio_column_is_not_requested_for_statistics(monkeypatch):
    requested = []

    def yahoo(ticker):
        requested.append(("yahoo", ticker))
        return pd.Series({"Currency": "USD"})

    def fmp(ticker, api_key):
        requested.append(("fmp", ticker))
        return pd.Series()

    monkeypatch.setattr(
        historical_model.yfinance_model, "get_historical_statistics", yahoo
    )
    monkeypatch.setattr(historical_model.fmp_model, "get_historical_statistics", fmp)

    statistics, _ = historical_model.get_historical_statistics(
        tickers=["AAPL", "Portfolio"], api_key="KEY"
    )

    assert requested == [("yahoo", "AAPL")]
    assert statistics.loc["Currency", "AAPL"] == "USD"
