"""Portfolio Controller Tests""" ""
import warnings

import numpy as np
import pandas as pd
import pytest

from financetoolkit import Portfolio, Toolkit, historical_model
from financetoolkit.utilities.error_model import InvalidArgumentError

historical = pd.read_pickle("tests/datasets/portfolio_historical_dataset.pickle")
benchmark = pd.read_pickle("tests/datasets/portfolio_benchmark_dataset.pickle")
treasury_data = pd.read_pickle("tests/datasets/treasury_data.pickle")
risk_free_rate = pd.read_pickle("tests/datasets/risk_free_rate.pickle")

portfolio = Portfolio(portfolio_dataset="tests/datasets/portfolio_test.xlsx")

portfolio._daily_historical_data = historical
portfolio._daily_benchmark_data = benchmark
portfolio._currency_column = None  # type: ignore # noqa


if not portfolio._benchmark_toolkit:
    portfolio._benchmark_tickers = {}
    for ticker in portfolio._original_tickers:
        portfolio._benchmark_tickers[ticker] = portfolio._cfg["general"][
            "benchmark_ticker"
        ]

    portfolio._benchmark_toolkit = Toolkit(
        tickers=list(set(portfolio._benchmark_tickers.values())),
        historical=portfolio._daily_benchmark_data,
        benchmark_ticker=None,
        start_date=portfolio._start_date,
        sleep_timer=False,
    )

if not portfolio._toolkit:
    portfolio._toolkit = Toolkit(
        tickers=portfolio._tickers,
        benchmark_ticker=None,
        start_date=portfolio._start_date,
        historical=portfolio._daily_historical_data,
        sleep_timer=False,
    )

portfolio._toolkit._daily_treasury_data = treasury_data
# The benchmark's periods need the risk-free rate too; without it the test would retrieve
# Treasury yields from Yahoo Finance, whose parallel requests lock its SQLite cache in CI.
portfolio._benchmark_toolkit._daily_treasury_data = treasury_data
portfolio._benchmark_toolkit._daily_risk_free_rate = risk_free_rate
portfolio._toolkit._daily_risk_free_rate = risk_free_rate


@pytest.fixture(autouse=True)
def offline_rates(monkeypatch):
    """
    Give every Toolkit built from supplied prices the supplied Treasury yields too. Without
    them, valuing a portfolio retrieves the yields from Yahoo Finance, whose parallel requests
    lock its SQLite cache when tests run in parallel.
    """
    initialise = Toolkit.__init__

    def initialise_offline(self, *args, **kwargs):
        initialise(self, *args, **kwargs)
        if kwargs.get("historical") is not None:
            self._daily_treasury_data = treasury_data
            self._daily_risk_free_rate = risk_free_rate

    monkeypatch.setattr(Toolkit, "__init__", initialise_offline)


def test_read_portfolio_dataset(recorder):
    # A separate Portfolio: reading the dataset again restores its currency column, which the
    # shared portfolio above leaves out so its tests never ask for the tickers' currencies
    # online (and, run in parallel, never touch yfinance's shared timezone cache).
    reader = Portfolio(portfolio_dataset="tests/datasets/portfolio_test.xlsx")
    recorder.capture(reader.read_portfolio_dataset(adjust_duplicates=True))
    recorder.capture(reader.read_portfolio_dataset(adjust_duplicates=False))


def test_collect_benchmark_historical_data(recorder):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in accumulate",
            category=RuntimeWarning,
            module="numpy._core.fromnumeric",
        )
        recorder.capture(portfolio.collect_benchmark_historical_data())


def test_collect_historical_data(recorder):
    recorder.capture(portfolio.collect_historical_data())


def test_get_transactions_performance(recorder):
    recorder.capture(portfolio.get_transactions_performance(period="daily"))
    recorder.capture(portfolio.get_transactions_performance(period="weekly"))
    recorder.capture(portfolio.get_transactions_performance(period="monthly"))
    recorder.capture(portfolio.get_transactions_performance(period="quarterly"))
    recorder.capture(portfolio.get_transactions_performance(period="yearly"))


def test_get_money_weighted_return(recorder):
    recorder.capture(portfolio.get_money_weighted_return())


def test_get_turnover(recorder):
    daily = portfolio.get_turnover(rounding=8)
    recorder.capture(daily[daily["Traded Value"] > 0])
    # The first purchase is a turnover of 2: the gross value of the empty day before
    # halves the average (up to the difference between the trade price and the close).
    assert daily["Turnover"].iloc[0] == pytest.approx(2, rel=1e-4)
    recorder.capture(portfolio.get_turnover(denominator="net_value", period="yearly"))


def test_get_gross_leverage(recorder):
    table = portfolio.get_gross_leverage(period="yearly")
    recorder.capture(table)
    # The portfolio holds no short positions.
    assert np.allclose(table["Gross Leverage"].dropna(), 1)


def test_get_trades(recorder):
    trades = portfolio.get_trades(rounding=8)
    recorder.capture(trades)
    recorder.capture(portfolio.get_trades(method="LIFO"))
    # The gross PnL of the closed trades of a ticker is its realized PnL with the same method.
    overview = portfolio.get_transactions_overview(exclude_sold_positions=False)
    realized = overview.groupby(level=1)["PnL"].sum()
    closed = trades[trades["Status"] == "Closed"].groupby("Ticker")["Gross PnL"].sum()
    assert np.allclose(closed, realized.reindex(closed.index), atol=1e-4)


def test_get_trade_statistics(recorder):
    recorder.capture(portfolio.get_trade_statistics())
    recorder.capture(portfolio.get_trade_statistics(by="ticker"))
    recorder.capture(portfolio.get_trade_statistics(method="LIFO", by="direction"))


def test_get_position_timing(recorder):
    recorder.capture(portfolio.get_position_timing())


def test_get_return_contributions(recorder):
    table = portfolio.get_return_contributions(period="yearly", rounding=8)
    recorder.capture(table)
    assert np.allclose(
        table.drop(columns="Portfolio").sum(axis=1), table["Portfolio"], atol=1e-7
    )


def _offline_price_toolkit(self, tickers, start_date):  # noqa: ARG001
    self._toolkit = Toolkit(
        tickers=tickers,
        historical=pd.read_pickle("tests/datasets/historical_dataset.pickle"),
        benchmark_ticker=None,
        start_date="2019-12-31",
        sleep_timer=False,
    )
    return self._toolkit


def test_a_portfolio_from_weights(monkeypatch):
    monkeypatch.setattr(Portfolio, "_price_toolkit", _offline_price_toolkit)
    monkeypatch.setattr(Portfolio, "_ticker_currencies", lambda self, toolkit: {})

    weighted = Portfolio(weights="AAPL:0.6, MSFT:0.4", rebalancing="yearly")

    dataset = weighted._portfolio_dataset
    first_day = dataset.index.get_level_values(0).min()
    invested = (
        dataset.loc[first_day, "Price"] * dataset.loc[first_day, "Volume"]
    ).sum()
    assert invested == pytest.approx(10000, rel=1e-3)
    assert weighted.definition.weights == pytest.approx({"AAPL": 0.6, "MSFT": 0.4})
    assert weighted.definition.rebalancing == "yearly"


def test_a_portfolio_from_a_preset_definition(monkeypatch):
    from financetoolkit.portfolio.definition_model import (  # noqa: PLC0415
        PortfolioDefinition,
    )

    monkeypatch.setattr(Portfolio, "_price_toolkit", _offline_price_toolkit)
    monkeypatch.setattr(Portfolio, "_ticker_currencies", lambda self, toolkit: {})
    definition = PortfolioDefinition.from_weights(
        {"AAPL": 0.5, "MSFT": 0.5}, name="Even"
    )

    even = Portfolio(definition=definition, initial_value=5000)

    assert even.definition.name == "Even"
    assert set(even._tickers) == {"AAPL", "MSFT"}


def test_transactions_without_prices_are_completed(monkeypatch):
    monkeypatch.setattr(Portfolio, "_price_toolkit", _offline_price_toolkit)
    monkeypatch.setattr(Portfolio, "_ticker_currencies", lambda self, toolkit: {})
    trades = pd.DataFrame(
        {
            "Trade Date": ["2021-01-05", "2021-06-01", "2022-02-01"],
            "Symbol": ["AAPL", "MSFT", "AAPL"],
            "Shares": [10, 5, -4],
            "Costs": [-1.0, -1.0, -1.0],
        }
    )

    completed = Portfolio(transactions=trades)._portfolio_dataset

    assert completed["Price"].notna().all()
    assert completed["Volume"].tolist() == [10.0, 5.0, -4.0]
    assert set(completed["Currency"]) == {"USD"}


def test_holdings_are_bought_at_the_traded_close(monkeypatch):
    monkeypatch.setattr(Portfolio, "_price_toolkit", _offline_price_toolkit)
    monkeypatch.setattr(
        Portfolio, "_ticker_currencies", lambda self, toolkit: {"AAPL": "USD"}
    )

    held = Portfolio(holdings={"AAPL": 10}, start_date="2019-12-31")

    prices = pd.read_pickle("tests/datasets/historical_dataset.pickle")
    first_day = held._portfolio_dataset.index.get_level_values(0).min()
    close = float(prices.loc[first_day, ("Close", "AAPL")])
    assert close != pytest.approx(float(prices.loc[first_day, ("Adj Close", "AAPL")]))
    assert held._portfolio_dataset["Price"].iloc[0] == pytest.approx(close)


def test_a_portfolio_in_two_currencies_is_valued_in_one(monkeypatch):
    prices = pd.read_pickle("tests/datasets/historical_dataset.pickle")
    pounds_per_dollar = pd.DataFrame(
        {"USDGBP=X": [0.8] * 400 + [0.75] * (len(prices) - 400)}, index=prices.index
    )
    monkeypatch.setattr(Portfolio, "_price_toolkit", _offline_price_toolkit)
    # MSFT is taken to be a London listing quoted in pence.
    monkeypatch.setattr(
        Portfolio,
        "_ticker_currencies",
        lambda self, toolkit: {"AAPL": "USD", "MSFT": "GBp"},
    )
    monkeypatch.setattr(
        Portfolio,
        "_get_exchange_rates",
        lambda self, rate_tickers, start_date: pounds_per_dollar[sorted(rate_tickers)],
    )
    monkeypatch.setattr(
        Toolkit,
        "get_historical_statistics",
        lambda self, *args, **kwargs: pd.DataFrame(),
    )

    weighted = Portfolio(weights="AAPL:0.5, MSFT:0.5", rebalancing="yearly")
    dataset = weighted._portfolio_dataset

    assert set(dataset["Currency"]) == {"USD"}
    first_day = dataset.index.get_level_values(0).min()
    first = dataset.loc[first_day]
    value = first["Price"] * first["Volume"]
    assert (value / value.sum()).to_dict() == pytest.approx({"AAPL": 0.5, "MSFT": 0.5})
    assert first.loc["MSFT", "Price"] == pytest.approx(
        float(prices.loc[first_day, ("Close", "MSFT")]) / 100 / 0.8
    )

    # The historical data is converted with the same rates, so the price of every
    # transaction is the converted price of its day and the returns are in dollars.
    historical = weighted.collect_historical_data()
    for (day, ticker), price in dataset["Price"].items():
        assert float(historical.loc[day, ("Close", ticker)]) == pytest.approx(
            price, rel=1e-4
        )
    rate_change = historical.index[400]
    assert float(historical.loc[rate_change, ("Return", "MSFT")]) == pytest.approx(
        (
            float(prices.loc[rate_change, ("Adj Close", "MSFT")]) / 0.75
            - float(prices.iloc[399][("Adj Close", "MSFT")]) / 0.8
        )
        / (float(prices.iloc[399][("Adj Close", "MSFT")]) / 0.8),
        rel=1e-3,
    )
    assert float(
        weighted._weekly_historical_data[("Adj Close", "MSFT")].iloc[-1]
    ) == pytest.approx(float(prices[("Adj Close", "MSFT")].iloc[-1]) / 100 / 0.75)


def test_money_weighted_return_treats_costs_as_paid_whatever_their_sign():
    costs_column = portfolio._costs_column
    original = portfolio._portfolio_dataset[costs_column].copy()
    expected = portfolio.get_money_weighted_return()
    try:
        portfolio._portfolio_dataset[costs_column] = original.abs()
        positive = portfolio.get_money_weighted_return()
    finally:
        portfolio._portfolio_dataset[costs_column] = original

    assert (original <= 0).all()
    pd.testing.assert_frame_equal(positive, expected)


def test_the_benchmark_is_converted_to_the_currency_of_the_transactions(monkeypatch):
    prices = pd.read_pickle("tests/datasets/historical_dataset.pickle")
    rates = pd.Series([0.8] * 400 + [0.75] * (len(prices) - 400), index=prices.index)
    monkeypatch.setattr(Portfolio, "_price_toolkit", _offline_price_toolkit)
    monkeypatch.setattr(
        Portfolio,
        "_ticker_currencies",
        lambda self, toolkit: {"AAPL": "USD", "MSFT": "USD", "^GSPC": "USD"},
    )
    monkeypatch.setattr(
        Portfolio,
        "_get_exchange_rates",
        lambda self, rate_tickers, start_date: pd.DataFrame(
            {rate_ticker: rates for rate_ticker in rate_tickers}
        ),
    )
    monkeypatch.setattr(
        Toolkit,
        "get_historical_statistics",
        lambda self, *args, **kwargs: pd.DataFrame(),
    )

    weighted = Portfolio(
        weights="AAPL:0.5, MSFT:0.5", rebalancing="yearly", currency="GBP"
    )
    weighted._benchmark_tickers = dict.fromkeys(weighted._original_tickers, "^GSPC")
    weighted._benchmark_toolkit = Toolkit(
        tickers=["^GSPC"],
        historical=prices.xs("Benchmark", level=1, axis=1, drop_level=False).rename(
            columns={"Benchmark": "^GSPC"}, level=1
        ),
        benchmark_ticker=None,
        start_date="2019-12-31",
        sleep_timer=False,
    )

    weighted.collect_historical_data()
    benchmark = weighted.collect_benchmark_historical_data()

    close = prices[("Close", "Benchmark")]
    converted = weighted._benchmark_prices_per_ticker["AAPL"]
    assert float(converted.iloc[0]) == pytest.approx(float(close.iloc[0]) / 0.8)
    assert float(converted.iloc[-1]) == pytest.approx(float(close.iloc[-1]) / 0.75)
    assert float(benchmark["Close"].iloc[-1]) == pytest.approx(
        float(close.iloc[-1]) / 0.75
    )
    # The return on the day the rate moves includes the currency move.
    adjusted = prices[("Adj Close", "Benchmark")]
    assert float(benchmark["Return"].iloc[400]) == pytest.approx(
        (float(adjusted.iloc[400]) / 0.75) / (float(adjusted.iloc[399]) / 0.8) - 1,
        rel=1e-6,
    )
    assert float(weighted._weekly_benchmark_data["Close"].iloc[-1]) == pytest.approx(
        float(close.iloc[-1]) / 0.75
    )


def test_the_portfolio_overview_adds_up_with_dividends():
    overview = portfolio.get_portfolio_overview(rounding=None)

    assert "Dividends" in overview.columns
    assert (overview["Dividends"] >= 0).all()
    assert overview.loc["Portfolio", "Dividends"] > 0
    return_value = (
        overview["Latest Value"] + overview["Dividends"] - overview["Invested"]
    )
    # Return Value also holds the sale proceeds, which are never negative.
    assert (overview["Return Value"] >= return_value - 1e-6).all()


def _with_a_cryptocurrency(holdings):
    """
    The holdings next to "BTC", a ticker trading every calendar day, as the historical data
    holds them: the stocks keep their last close on the weekends, without a return.
    """
    days = pd.period_range(holdings.index[0], holdings.index[-1], freq="D")
    cryptocurrency = (
        holdings.xs("AAPL", level=1, axis=1)
        .reindex(days)
        .interpolate()
        .assign(Dividends=0.0)
    )
    cryptocurrency["Return"] = cryptocurrency["Adj Close"].pct_change()
    cryptocurrency.columns = pd.MultiIndex.from_product(
        [cryptocurrency.columns, ["BTC"]]
    )
    combined = pd.concat([holdings.reindex(days), cryptocurrency], axis=1)
    stock_weekends = combined["Return"].isna() & combined["Close"].ffill().notna()
    combined = historical_model.fill_non_trading_days(combined)
    # A carried-forward bar pays no dividend again.
    combined["Dividends"] = combined["Dividends"].mask(stock_weekends, 0.0).to_numpy()

    return combined


def _self_benchmarked_portfolio(monkeypatch, transactions, cryptocurrency=False):
    """A portfolio of transactions whose benchmark is AAPL for every ticker."""
    prices = pd.read_pickle("tests/datasets/historical_dataset.pickle")
    # The dataset's own benchmark column is left out, so it does not pose as a holding.
    holdings = prices.drop(columns="Benchmark", level=1)
    if cryptocurrency:
        holdings = _with_a_cryptocurrency(holdings)

    def price_toolkit(self, tickers, start_date):  # noqa: ARG001
        self._toolkit = Toolkit(
            tickers=tickers,
            historical=holdings,
            benchmark_ticker=None,
            start_date="2019-12-31",
            sleep_timer=False,
        )
        return self._toolkit

    monkeypatch.setattr(Portfolio, "_price_toolkit", price_toolkit)
    monkeypatch.setattr(Portfolio, "_ticker_currencies", lambda self, toolkit: {})
    monkeypatch.setattr(
        Toolkit,
        "get_historical_statistics",
        lambda self, *args, **kwargs: pd.DataFrame(),
    )

    self_benchmarked = Portfolio(transactions=transactions)
    self_benchmarked._benchmark_tickers = dict.fromkeys(
        self_benchmarked._original_tickers, "AAPL"
    )
    self_benchmarked._benchmark_toolkit = Toolkit(
        tickers=["AAPL"],
        historical=prices.xs("AAPL", level=1, axis=1, drop_level=False),
        benchmark_ticker=None,
        start_date="2019-12-31",
        sleep_timer=False,
    )

    return self_benchmarked, prices


def test_a_weekend_transaction_is_priced_at_the_next_close(monkeypatch):
    # 2021-01-09 is a Saturday, so the position, benchmark and dividends start on Monday 2021-01-11.  # noqa: E501
    trades = pd.DataFrame(
        {
            "Date": ["2021-01-09", "2021-01-11"],
            "Ticker": ["AAPL", "MSFT"],
            "Volume": [10, 5],
            "Costs": [0.0, 0.0],
        }
    )
    self_benchmarked, prices = _self_benchmarked_portfolio(monkeypatch, trades)

    close = prices[("Close", "AAPL")]
    close.index = close.index.to_timestamp()
    assert float(self_benchmarked._portfolio_dataset["Price"].iloc[0]) == pytest.approx(
        float(close.loc["2021-01-11"])
    )

    overview = self_benchmarked.get_portfolio_overview(rounding=12)

    # An asset measured against itself over the same days has no alpha.
    assert overview.loc["AAPL", "Alpha"] == pytest.approx(0.0, abs=1e-9)


def test_a_transaction_after_the_last_close_is_priced_at_the_last_close(monkeypatch):
    # The data ends on Friday 2022-12-30, so a trade of Saturday 2022-12-31 has no close on or after its date and takes, as its position, benchmark and dividends do, the close of 2022-12-30.  # noqa: E501
    def overview_for(date):
        trades = pd.DataFrame(
            {
                "Date": ["2021-01-11", date],
                "Ticker": ["AAPL", "MSFT"],
                "Volume": [10, 5],
                "Costs": [0.0, 0.0],
            }
        )
        portfolio_after, prices = _self_benchmarked_portfolio(monkeypatch, trades)
        return (
            portfolio_after,
            prices,
            portfolio_after.get_portfolio_overview(rounding=12),
        )

    after, prices, overview = overview_for("2022-12-31")
    _, _, on_last_close = overview_for("2022-12-30")

    assert float(after._portfolio_dataset["Price"].iloc[-1]) == pytest.approx(
        float(prices[("Close", "MSFT")].iloc[-1])
    )
    # The benchmark is bought on the same day as the position: no return over no time, not -100%.  # noqa: E501
    assert overview.loc["MSFT", "Benchmark Return"] == pytest.approx(0.0, abs=1e-12)
    pd.testing.assert_frame_equal(overview, on_last_close)
    assert after.get_positions_overview()[("Volume", "MSFT")].iloc[-1] == pytest.approx(
        5.0
    )


def test_a_tiny_sale_changes_the_portfolio_return_only_marginally(monkeypatch):
    buys = {
        "Date": ["2021-01-11", "2021-06-01", "2021-06-01"],
        "Ticker": ["AAPL", "AAPL", "MSFT"],
        "Volume": [10.0, 5.0, 3.0],
        "Costs": [-1.0, -1.0, -1.0],
    }
    with_sale = {
        column: values + [value]
        for (column, values), value in zip(
            buys.items(), ["2022-06-01", "AAPL", -0.01, 0.0], strict=True
        )
    }
    without, _ = _self_benchmarked_portfolio(monkeypatch, pd.DataFrame(buys))
    sold, _ = _self_benchmarked_portfolio(monkeypatch, pd.DataFrame(with_sale))

    plain = without.get_portfolio_overview(rounding=12).loc["Portfolio"]
    tiny = sold.get_portfolio_overview(rounding=12).loc["Portfolio"]
    proceeds = 0.01 * float(sold._portfolio_dataset["Price"].iloc[-1])

    # The sale lowers the capital by at most its proceeds rather than switching to a different measure.  # noqa: E501
    capital = plain["Return Value"] / plain["Return"]
    capital_with_sale = tiny["Return Value"] / tiny["Return"]
    assert capital - proceeds - 1e-6 <= capital_with_sale <= capital + 1e-6
    assert tiny["Return"] == pytest.approx(plain["Return"], abs=1e-4)


def test_the_capital_of_buys_is_invested_less_the_dividends_between_them(monkeypatch):
    def portfolio_row(dates):
        trades = pd.DataFrame(
            {
                "Date": dates,
                "Ticker": ["AAPL", "AAPL", "MSFT"],
                "Volume": [10, 5, 3],
                "Costs": [-1.0, -1.0, -1.0],
            }
        )
        buys, _ = _self_benchmarked_portfolio(monkeypatch, trades)
        row = buys.get_portfolio_overview(rounding=12).loc["Portfolio"]
        return buys, row, row["Return Value"] / row["Return"]

    # Without a dividend between the buys, every buy is new capital.
    together, row, capital = portfolio_row(["2021-01-11", "2021-01-12", "2021-01-12"])
    assert capital == pytest.approx(row["Invested"])
    assert together._positions_overview["Dividends"].iloc[-1].sum() > 0

    # The AAPL dividends received between the buys pay for part of the later buys.
    apart, row, capital = portfolio_row(["2021-01-11", "2021-06-01", "2021-06-01"])
    dividends = apart._positions_overview["Dividends"].sum(axis=1)
    dividends.index = dividends.index.to_timestamp()
    before_second_buy = float(dividends.loc[:"2021-05-31"].iloc[-1])
    assert before_second_buy > 0
    assert capital == pytest.approx(row["Invested"] - before_second_buy)


def test_a_portfolio_of_one_ticker(monkeypatch):
    trades = pd.DataFrame(
        {
            "Date": ["2021-01-11", "2021-06-01", "2022-02-01"],
            "Ticker": ["AAPL", "AAPL", "AAPL"],
            "Volume": [10, 5, -4],
            "Costs": [-1.0, -1.0, -1.0],
        }
    )
    single, _ = _self_benchmarked_portfolio(monkeypatch, trades)

    overview = single.get_portfolio_overview(rounding=12)

    assert list(overview.index) == ["AAPL", "Portfolio"]
    assert overview.loc["AAPL", "Volume"] == pytest.approx(11.0)
    # The asset is its own benchmark, so it has no alpha.
    assert overview.loc["AAPL", "Alpha"] == pytest.approx(0.0, abs=1e-9)

    positions = single.get_positions_overview()
    assert positions["Volume"].columns.tolist() == ["AAPL"]
    assert positions[("Volume", "AAPL")].iloc[-1] == pytest.approx(11.0)

    performance = single.get_portfolio_performance(period="yearly")
    assert not performance.empty

    transactions = single.get_transactions_overview(exclude_sold_positions=False)
    assert len(transactions) == 3

    money_weighted = single.get_money_weighted_return()
    assert {"AAPL", "Portfolio"} <= set(money_weighted.index)
    assert money_weighted["Money-Weighted Return"].notna().all()


def test_a_portfolio_of_one_ticker_reads_its_statistics_as_a_series(monkeypatch):
    # A Toolkit of one ticker returns its historical statistics as a Series, not a table.
    trades = pd.DataFrame(
        {
            "Date": ["2021-01-11", "2021-06-01"],
            "Ticker": ["AAPL", "AAPL"],
            "Volume": [10, 5],
            "Costs": [0, 0],
        }
    )
    single, _ = _self_benchmarked_portfolio(monkeypatch, trades)
    monkeypatch.setattr(
        Toolkit,
        "get_historical_statistics",
        lambda self, *args, **kwargs: pd.Series(
            {"Currency": "USD", "Exchange Name": "NMS"}, name="AAPL"
        ),
    )

    positions = single.get_positions_overview()

    assert positions[("Volume", "AAPL")].iloc[-1] == pytest.approx(15.0)
    assert single._historical_statistics.loc["Currency", "AAPL"] == "USD"
    assert set(single._portfolio_dataset["Currency"]) == {"USD"}


class _EmptyToolkit:
    """A Toolkit that records the tickers it is given and finds no historical data."""

    tickers: list = []

    def __init__(self, tickers, **kwargs):  # noqa: ARG002
        _EmptyToolkit.tickers = list(tickers)

    def get_historical_data(self, period="daily"):  # noqa: ARG002
        return pd.DataFrame()

    def get_historical_statistics(self):
        return pd.DataFrame()


def _portfolio_with_historical_data(**kwargs):
    held = Portfolio(portfolio_dataset="tests/datasets/portfolio_test.xlsx", **kwargs)
    held._daily_historical_data = historical
    held._weekly_historical_data = historical
    return held


def test_the_benchmark_given_to_the_portfolio_is_collected(monkeypatch):
    from financetoolkit.portfolio import portfolio_controller  # noqa: PLC0415

    monkeypatch.setattr(portfolio_controller, "Toolkit", _EmptyToolkit)

    _portfolio_with_historical_data(
        benchmark_ticker="QQQ"
    ).collect_benchmark_historical_data()
    assert _EmptyToolkit.tickers == ["QQQ"]

    # The argument of the method comes first, the configuration's benchmark last.
    _portfolio_with_historical_data(
        benchmark_ticker="QQQ"
    ).collect_benchmark_historical_data(benchmark_ticker="VT")
    assert _EmptyToolkit.tickers == ["VT"]

    default = _portfolio_with_historical_data()
    default.collect_benchmark_historical_data()
    assert _EmptyToolkit.tickers == [default._cfg["general"]["benchmark_ticker"]]


def test_a_benchmark_without_data_gives_an_empty_toolkit(monkeypatch):
    from financetoolkit.portfolio import portfolio_controller  # noqa: PLC0415

    monkeypatch.setattr(portfolio_controller, "Toolkit", _EmptyToolkit)
    held = _portfolio_with_historical_data(benchmark_ticker="NOTATICKER")

    assert held.collect_benchmark_historical_data().empty
    toolkit = held.toolkit
    assert isinstance(toolkit, pd.DataFrame)
    assert toolkit.empty


def test_an_overview_with_a_benchmark_without_data_is_empty(monkeypatch):
    from financetoolkit.portfolio import portfolio_controller  # noqa: PLC0415

    monkeypatch.setattr(portfolio_controller, "Toolkit", _EmptyToolkit)
    held = _portfolio_with_historical_data(benchmark_ticker="NOTATICKER")

    assert held.get_portfolio_overview().empty
    assert held.get_portfolio_performance().empty
    assert held.get_transactions_performance().empty


def test_the_portfolio_benchmark_replicates_sales_as_well_as_buys(monkeypatch):
    trades = pd.DataFrame(
        {
            "Date": [
                "2020-01-10",
                "2020-06-01",
                "2021-03-01",
                "2021-09-01",
                "2022-06-01",
            ],
            "Ticker": ["AAPL"] * 5,
            "Volume": [20, 10, -15, 5, -8],
            "Costs": [-1.0] * 5,
        }
    )
    traded, _ = _self_benchmarked_portfolio(monkeypatch, trades)

    overview = traded.get_portfolio_overview(rounding=12)

    # A portfolio of the benchmark alone matches it on every buy and sale, so it has no alpha.
    assert overview.loc["Portfolio", "Benchmark Return"] == pytest.approx(
        overview.loc["Portfolio", "Return"], abs=1e-9
    )
    assert overview.loc["Portfolio", "Alpha"] == pytest.approx(0.0, abs=1e-9)


def test_a_rebalancing_does_not_buy_the_benchmark_again(monkeypatch):
    first, rebalance = pd.Timestamp("2020-01-10"), pd.Timestamp("2021-06-01")
    prices = pd.read_pickle("tests/datasets/historical_dataset.pickle")
    close = prices["Close"].copy()
    close.index = close.index.to_timestamp()
    # The AAPL bought on the rebalancing date costs exactly what the MSFT sold on it raises.
    aapl_bought = 5 * close.loc[rebalance, "MSFT"] / close.loc[rebalance, "AAPL"]
    trades = pd.DataFrame(
        {
            "Date": [first, first, rebalance, rebalance],
            "Ticker": ["AAPL", "MSFT", "MSFT", "AAPL"],
            "Volume": [10.0, 10.0, -5.0, aapl_bought],
            "Costs": [0.0] * 4,
        }
    )
    rebalanced, _ = _self_benchmarked_portfolio(monkeypatch, trades)

    portfolio_row = rebalanced.get_portfolio_overview(rounding=12).loc["Portfolio"]

    # The rebalancing sells as much benchmark as it buys, so the benchmark is held from the first day alone, with its dividends paid as cash.  # noqa: E501
    dividends = prices[("Dividends", "AAPL")].fillna(0)
    dividends.index = dividends.index.to_timestamp()
    held = (close["AAPL"].iloc[-1] + dividends.loc[first:].iloc[1:].sum()) / close.loc[
        first, "AAPL"
    ] - 1
    assert portfolio_row["Benchmark Return"] == pytest.approx(held, rel=1e-6)


def test_tickers_are_matched_to_their_own_data_whatever_their_case(monkeypatch):
    trades = pd.DataFrame(
        {
            "Date": ["2021-01-05", "2021-01-06", "2021-01-07"],
            "Symbol": ["msft", "AAPL", "MSFT"],
            "Shares": [1, 1, 1],
            "Costs": [0.0, 0.0, 0.0],
        }
    )
    lowercase, prices = _self_benchmarked_portfolio(monkeypatch, trades)

    assert sorted(lowercase._tickers) == ["AAPL", "MSFT"]

    historical = lowercase.collect_historical_data(rounding=None)

    assert lowercase._ticker_combinations == {"AAPL": "AAPL", "MSFT": "MSFT"}
    for ticker in ["AAPL", "MSFT"]:
        assert historical[("Close", ticker)].iloc[-1] == pytest.approx(
            float(prices[("Close", ticker)].iloc[-1])
        )


def test_transactions_in_two_currencies_are_expressed_in_one(monkeypatch):
    trades = pd.DataFrame(
        {
            "Date": ["2021-01-05", "2021-01-05"],
            "Ticker": ["AAPL", "MSFT"],
            "Price": [100.0, 100.0],
            "Volume": [20, 10],
            "Costs": [-1.0, -1.0],
            "Currency": ["USD", "EUR"],
        }
    )
    dates = pd.period_range("2021-01-01", "2021-01-10", freq="D")
    euros_per_dollar = pd.DataFrame({"USDEUR=X": 0.8}, index=dates)
    monkeypatch.setattr(
        Portfolio,
        "_get_exchange_rates",
        lambda self, rate_tickers, start_date: euros_per_dollar[
            sorted(rate_ticker for rate_ticker in rate_tickers if rate_ticker)
        ],
    )
    mixed = Portfolio(transactions=trades)

    mixed._convert_transactions_to_reporting_currency()

    # Most is invested in dollars, so the euro transaction is converted at 0.8 euro per dollar.
    dataset = mixed._portfolio_dataset.droplevel(0)
    assert set(dataset["Currency"]) == {"USD"}
    assert dataset.loc["MSFT", "Price"] == pytest.approx(125.0)
    assert dataset.loc["MSFT", "Costs"] == pytest.approx(-1.25)
    assert dataset.loc["AAPL", "Price"] == pytest.approx(100.0)


def test_a_period_with_purchases_and_larger_sales_keeps_its_performance(monkeypatch):
    trades = pd.DataFrame(
        {
            "Date": ["2021-01-05", "2021-03-02", "2021-03-03"],
            "Symbol": ["AAPL", "AAPL", "AAPL"],
            "Shares": [10, 5, -10],
            "Costs": [0.0, 0.0, 0.0],
        }
    )
    traded, _ = _self_benchmarked_portfolio(monkeypatch, trades)

    performance = traded.get_transactions_performance(period="monthly")

    march = performance.loc[(pd.Period("2021-03", freq="M"), "AAPL")]
    assert march["Volume"] == -5.0
    assert march["Invested Amount"] > 0


def _converted_portfolio(monkeypatch, trades, rates, **kwargs):
    monkeypatch.setattr(
        Portfolio,
        "_get_exchange_rates",
        lambda self, rate_tickers, start_date: rates.reindex(
            columns=sorted(rate_ticker for rate_ticker in rate_tickers if rate_ticker)
        ),
    )
    portfolio = Portfolio(transactions=trades, **kwargs)
    portfolio._convert_transactions_to_reporting_currency()
    return portfolio._portfolio_dataset.droplevel(0)


def test_the_reporting_currency_compares_amounts_in_one_currency(monkeypatch):
    trades = pd.DataFrame(
        {
            "Date": ["2021-01-05", "2021-01-05"],
            "Ticker": ["7203.T", "AAPL"],
            "Price": [1000.0, 200.0],
            "Volume": [1000, 250],
            "Costs": [0.0, 0.0],
            "Currency": ["JPY", "USD"],
        }
    )
    dates = pd.period_range("2021-01-01", "2021-01-10", freq="D")
    rates = pd.DataFrame({"USDJPY=X": 150.0}, index=dates)

    dataset = _converted_portfolio(monkeypatch, trades, rates)

    # 1,000,000 yen is about 6,667 dollars, less than the 50,000 dollars in AAPL.
    assert set(dataset["Currency"]) == {"USD"}
    assert dataset.loc["AAPL", "Price"] == pytest.approx(200.0)
    assert dataset.loc["7203.T", "Price"] == pytest.approx(1000.0 / 150.0)


def test_a_weekend_transaction_uses_the_last_rate_before_it(monkeypatch):
    trades = pd.DataFrame(
        {
            "Date": ["2024-01-06", "2024-01-08"],
            "Ticker": ["ASML.AS", "AAPL"],
            "Price": [100.0, 100.0],
            "Volume": [1, 10],
            "Costs": [0.0, 0.0],
            "Currency": ["EUR", "USD"],
        }
    )
    # Rates retrieved from before the first transaction, with Friday's rate the last before Saturday.  # noqa: E501
    rates = pd.DataFrame(
        {"USDEUR=X": [0.9, 0.8]},
        index=pd.PeriodIndex(["2024-01-05", "2024-01-08"], freq="D"),
    )

    dataset = _converted_portfolio(monkeypatch, trades, rates)

    assert dataset.loc["ASML.AS", "Price"] == pytest.approx(100.0 / 0.9)


def test_a_transaction_without_any_rate_before_it_raises(monkeypatch):
    trades = pd.DataFrame(
        {
            "Date": ["2024-01-06", "2024-01-08"],
            "Ticker": ["ASML.AS", "AAPL"],
            "Price": [100.0, 100.0],
            "Volume": [1, 10],
            "Costs": [0.0, 0.0],
            "Currency": ["EUR", "USD"],
        }
    )
    rates = pd.DataFrame(
        {"USDEUR=X": [0.8]}, index=pd.PeriodIndex(["2024-01-08"], freq="D")
    )

    with pytest.raises(ValueError, match="USDEUR=X on or before 2024-01-06"):
        _converted_portfolio(monkeypatch, trades, rates)


def test_pence_and_pounds_are_expressed_in_pounds(monkeypatch):
    trades = pd.DataFrame(
        {
            "Date": ["2021-01-05", "2021-01-05"],
            "Ticker": ["VOD.L", "BP.L"],
            "Price": [120.0, 3.0],
            "Volume": [100, 10],
            "Costs": [-50.0, -1.0],
            "Currency": ["GBp", "GBP"],
        }
    )

    dataset = _converted_portfolio(monkeypatch, trades, pd.DataFrame())

    assert set(dataset["Currency"]) == {"GBP"}
    assert dataset.loc["VOD.L", "Price"] == pytest.approx(1.2)
    assert dataset.loc["VOD.L", "Costs"] == pytest.approx(-0.5)
    assert dataset.loc["BP.L", "Price"] == pytest.approx(3.0)


def test_pence_are_expressed_in_pounds_when_the_currency_is_pounds(monkeypatch):
    trades = pd.DataFrame(
        {
            "Date": ["2021-01-05"],
            "Ticker": ["VOD.L"],
            "Price": [120.0],
            "Volume": [100],
            "Costs": [0.0],
            "Currency": ["GBp"],
        }
    )

    dataset = _converted_portfolio(monkeypatch, trades, pd.DataFrame(), currency="GBP")

    assert set(dataset["Currency"]) == {"GBP"}
    assert dataset.loc["VOD.L", "Price"] == pytest.approx(1.2)


def test_a_transaction_without_a_currency_keeps_its_price(monkeypatch, caplog):
    trades = pd.DataFrame(
        {
            "Date": ["2021-01-05", "2021-01-05", "2021-01-05"],
            "Ticker": ["AAPL", "MSFT", "NOVO"],
            "Price": [100.0, 100.0, 5.0],
            "Volume": [20, 10, 1],
            "Costs": [-1.0, -1.0, -1.0],
            "Currency": ["USD", "EUR", "USD"],
        }
    )
    dates = pd.period_range("2021-01-01", "2021-01-10", freq="D")
    rates = pd.DataFrame({"USDEUR=X": 0.8}, index=dates)
    monkeypatch.setattr(
        Portfolio,
        "_get_exchange_rates",
        lambda self, rate_tickers, start_date: rates.reindex(
            columns=sorted(rate_ticker for rate_ticker in rate_tickers if rate_ticker)
        ),
    )
    portfolio = Portfolio(transactions=trades)
    # A ticker whose currency the data source doesn't report either.
    portfolio._portfolio_dataset.loc[
        portfolio._portfolio_dataset.index.get_level_values(1) == "NOVO", "Currency"
    ] = None

    with caplog.at_level("WARNING"):
        portfolio._convert_transactions_to_reporting_currency()
    dataset = portfolio._portfolio_dataset.droplevel(0)

    # Taken to be in the reporting currency, and named in a warning, instead of becoming NaN.
    assert dataset.loc["NOVO", "Price"] == pytest.approx(5.0)
    assert dataset.loc["NOVO", "Costs"] == pytest.approx(-1.0)
    assert dataset.loc["MSFT", "Price"] == pytest.approx(125.0)
    assert "NOVO" in caplog.text


def test_a_weekend_transaction_belongs_to_the_period_of_its_trading_day(monkeypatch):
    # Saturday 2021-07-31 is priced at the close of Monday 2021-08-02, so it is an August purchase, valued at the August close and receiving the AAPL dividend of 2021-08-06.  # noqa: E501
    trades = pd.DataFrame(
        {
            "Date": ["2021-07-31", "2021-06-01"],
            "Ticker": ["AAPL", "MSFT"],
            "Volume": [10, 5],
            "Costs": [0.0, 0.0],
        }
    )
    weekend, prices = _self_benchmarked_portfolio(monkeypatch, trades)

    performance = weekend.get_transactions_performance(period="monthly", rounding=12)

    close = prices[("Close", "AAPL")]
    close.index = close.index.to_timestamp()
    dividends = prices[("Dividends", "AAPL")]
    dividends.index = dividends.index.to_timestamp()
    aapl = performance.xs("AAPL", level=1)
    assert aapl.index.tolist() == [pd.Period("2021-08", freq="M")]
    august = aapl.iloc[0]
    assert august["Price"] == pytest.approx(float(close.loc["2021-08-02"]))
    assert august["Current Value"] == pytest.approx(10 * float(close.loc["2021-08-31"]))
    assert august["Dividends"] == pytest.approx(
        10 * float(dividends.loc["2021-08-03":"2021-08-31"].sum())
    )
    assert august["Dividends"] > 0


def test_the_money_weighted_return_values_a_price_below_the_rounding(monkeypatch):
    # A token priced at 0.00002 rounds to 0 at four decimals, which would value the position at nothing.  # noqa: E501
    prices = pd.read_pickle("tests/datasets/historical_dataset.pickle")
    holdings = prices.drop(columns="Benchmark", level=1).copy()
    for column in ["Open", "High", "Low", "Close", "Adj Close", "Dividends"]:
        if (column, "MSFT") in holdings.columns:
            holdings[(column, "MSFT")] = holdings[(column, "MSFT")] / 1e7

    def price_toolkit(self, tickers, start_date):  # noqa: ARG001
        self._toolkit = Toolkit(
            tickers=tickers,
            historical=holdings,
            benchmark_ticker=None,
            start_date="2019-12-31",
            sleep_timer=False,
            rounding=None,
        )
        return self._toolkit

    monkeypatch.setattr(Portfolio, "_price_toolkit", price_toolkit)
    monkeypatch.setattr(Portfolio, "_ticker_currencies", lambda self, toolkit: {})
    monkeypatch.setattr(
        Toolkit,
        "get_historical_statistics",
        lambda self, *args, **kwargs: pd.DataFrame(),
    )
    tiny = Portfolio(
        transactions=pd.DataFrame(
            {
                "Date": ["2021-01-11"],
                "Ticker": ["MSFT"],
                "Volume": [1e9],
                "Costs": [0.0],
            }
        )
    )
    tiny._benchmark_tickers = dict.fromkeys(tiny._original_tickers, "AAPL")
    tiny._benchmark_toolkit = Toolkit(
        tickers=["AAPL"],
        historical=prices.xs("AAPL", level=1, axis=1, drop_level=False),
        benchmark_ticker=None,
        start_date="2019-12-31",
        sleep_timer=False,
    )

    result = tiny.get_money_weighted_return(rounding=12)

    latest_close = float(holdings[("Close", "MSFT")].iloc[-1])
    assert latest_close < 5e-5
    assert result.loc["MSFT", "Latest Value"] == pytest.approx(1e9 * latest_close)
    # Bought at 0.000021749 and worth 0.000023982 two years later, with dividends: a gain.
    assert result.loc["MSFT", "Money-Weighted Return"] > 0.05


def test_sold_positions_leave_the_overview_and_short_positions_stay(monkeypatch):
    # 1.1 + 2.2 - 3.3 leaves 4.4e-16 AAPL units; MSFT is sold short.
    trades = pd.DataFrame(
        {
            "Date": ["2021-01-11", "2021-02-01", "2021-03-01", "2021-01-11"],
            "Ticker": ["AAPL", "AAPL", "AAPL", "MSFT"],
            "Volume": [1.1, 2.2, -3.3, -5.0],
            "Costs": [0.0, 0.0, 0.0, 0.0],
        }
    )
    traded, _ = _self_benchmarked_portfolio(monkeypatch, trades)

    overview = traded.get_portfolio_overview(rounding=12)
    performance = traded.get_portfolio_performance(period="yearly", rounding=12)

    assert overview.index.tolist() == ["MSFT", "Portfolio"]
    assert overview.loc["MSFT", "Volume"] == -5.0
    assert performance.index.get_level_values(1).unique().tolist() == ["MSFT"]
    every_position = traded.get_portfolio_overview(
        rounding=12, exclude_sold_positions=False
    )
    assert every_position.index.tolist() == ["AAPL", "MSFT", "Portfolio"]


def test_a_weekend_cash_flow_is_dated_on_its_trading_day(monkeypatch):
    # Saturday 2021-07-31 is bought at Monday 2021-08-02's close, so it flows on that Monday.
    def money_weighted(date):
        trades = pd.DataFrame(
            {
                "Date": [date, "2021-06-01"],
                "Ticker": ["AAPL", "MSFT"],
                "Volume": [10, 5],
                "Costs": [0.0, 0.0],
            }
        )
        portfolio, _ = _self_benchmarked_portfolio(monkeypatch, trades)
        return portfolio.get_money_weighted_return(rounding=12)

    saturday = money_weighted("2021-07-31")
    monday = money_weighted("2021-08-02")

    assert saturday.loc["Portfolio"].tolist() == pytest.approx(
        monday.loc["Portfolio"].tolist()
    )


def test_a_portfolio_without_transactions_says_so(monkeypatch):
    with pytest.raises(ValueError, match="no transactions"):
        _self_benchmarked_portfolio(
            monkeypatch,
            pd.DataFrame(columns=["Date", "Ticker", "Volume", "Costs"]),
        )


@pytest.mark.parametrize("other", ["MSFT", "BTC"])
@pytest.mark.parametrize("date", ["2021-01-30", "2021-02-01"])
def test_a_weekend_transaction_next_to_a_cryptocurrency_waits_for_its_own_trading_day(
    monkeypatch, other, date
):
    # Saturday 2021-01-30 is bought at the close of Monday 2021-02-01 and is a February purchase, also when a cryptocurrency in the portfolio traded that Saturday and the data holds AAPL's Friday close on it.  # noqa: E501
    trades = pd.DataFrame(
        {
            "Date": ["2021-01-04", date],
            "Ticker": [other, "AAPL"],
            "Volume": [1, 10],
            "Costs": [0.0, 0.0],
        }
    )
    weekend, prices = _self_benchmarked_portfolio(
        monkeypatch, trades, cryptocurrency=other == "BTC"
    )
    monday_close = float(prices[("Close", "AAPL")].loc["2021-02-01"])
    assert monday_close != pytest.approx(
        float(prices[("Close", "AAPL")].loc["2021-01-29"])
    )

    assert weekend.get_portfolio_overview(rounding=12).loc[
        "AAPL", "Price"
    ] == pytest.approx(monday_close)
    performance = weekend.get_transactions_performance(period="monthly", rounding=12)
    assert performance.xs("AAPL", level=1).index[0] == pd.Period("2021-02", freq="M")

    # The position starts on Monday as well, so the Saturday shows no loss against the Friday close carried forward.  # noqa: E501
    positions = weekend.get_positions_overview(rounding=12)
    assert positions.loc[:"2021-01-31", ("Volume", "AAPL")].iloc[-1] == pytest.approx(
        0.0
    )
    assert positions.loc["2021-02-01", ("Volume", "AAPL")] == pytest.approx(10.0)
    assert positions.loc["2021-02-01", ("Cumulative Return", "AAPL")] == pytest.approx(
        0.0, abs=1e-12
    )


def test_a_weekend_cash_flow_next_to_a_cryptocurrency_is_dated_on_its_trading_day(
    monkeypatch,
):
    def money_weighted(date):
        trades = pd.DataFrame(
            {
                "Date": ["2021-01-04", date],
                "Ticker": ["BTC", "AAPL"],
                "Volume": [1, 10],
                "Costs": [0.0, 0.0],
            }
        )
        portfolio, _ = _self_benchmarked_portfolio(
            monkeypatch, trades, cryptocurrency=True
        )
        return portfolio.get_money_weighted_return(rounding=12)

    saturday = money_weighted("2021-01-30")
    monday = money_weighted("2021-02-01")

    numeric = ["Invested", "Latest Value", "Money-Weighted Return"]
    pd.testing.assert_frame_equal(saturday[numeric], monday[numeric])


def test_the_weekly_data_converts_each_day_at_its_own_rate(monkeypatch):
    prices = pd.read_pickle("tests/datasets/historical_dataset.pickle")
    # A rate that changes every day, so a week's first day and last day differ.
    pounds_per_dollar = pd.DataFrame(
        {"USDGBP=X": np.linspace(0.7, 0.9, len(prices))}, index=prices.index
    )
    monkeypatch.setattr(Portfolio, "_price_toolkit", _offline_price_toolkit)
    monkeypatch.setattr(
        Portfolio,
        "_ticker_currencies",
        lambda self, toolkit: {"AAPL": "USD", "MSFT": "GBp"},
    )
    monkeypatch.setattr(
        Portfolio,
        "_get_exchange_rates",
        lambda self, rate_tickers, start_date: pounds_per_dollar[sorted(rate_tickers)],
    )
    monkeypatch.setattr(
        Toolkit,
        "get_historical_statistics",
        lambda self, *args, **kwargs: pd.DataFrame(),
    )

    weighted = Portfolio(weights="AAPL:0.5, MSFT:0.5", rebalancing="yearly")
    weighted.collect_historical_data()
    daily = weighted._daily_historical_data
    weekly = weighted._weekly_historical_data

    weeks = pd.PeriodIndex(daily.index, freq="W")
    for column, aggregation in (
        ("Open", "first"),
        ("High", "max"),
        ("Low", "min"),
        ("Close", "last"),
        ("Dividends", "sum"),
    ):
        expected = daily[(column, "MSFT")].groupby(weeks).agg(aggregation)
        pd.testing.assert_series_equal(
            weekly[(column, "MSFT")],
            expected.reindex(weekly.index),
            check_names=False,
            check_freq=False,
        )
    # The Open of a week is converted at the rate of its first day, not its last.
    week = weekly.index[10]
    first_day = daily.index[weeks == week][0]
    assert float(weekly.loc[week, ("Open", "MSFT")]) == pytest.approx(
        float(prices.loc[first_day.to_timestamp(), ("Open", "MSFT")])
        / 100
        / float(pounds_per_dollar.loc[first_day.to_timestamp(), "USDGBP=X"])
    )


def _offline_portfolio():
    """A Portfolio of the test dataset on copies of the offline prices, set up as the shared one above."""
    held = Portfolio(portfolio_dataset="tests/datasets/portfolio_test.xlsx")
    held._daily_historical_data = historical.copy()
    held._daily_benchmark_data = benchmark.copy()
    held._currency_column = None  # type: ignore # noqa
    held._benchmark_tickers = dict.fromkeys(
        held._original_tickers, held._cfg["general"]["benchmark_ticker"]
    )
    held._benchmark_toolkit = Toolkit(
        tickers=list(set(held._benchmark_tickers.values())),
        historical=held._daily_benchmark_data,
        benchmark_ticker=None,
        start_date=held._start_date,
        sleep_timer=False,
    )
    held._toolkit = Toolkit(
        tickers=held._tickers,
        benchmark_ticker=None,
        start_date=held._start_date,
        historical=held._daily_historical_data,
        sleep_timer=False,
    )
    held.collect_historical_data()
    return held


def test_the_toolkit_leaves_the_portfolio_data_untouched():
    untouched = _offline_portfolio()
    touched = _offline_portfolio()
    daily_historical_data = touched._daily_historical_data.copy()

    assert isinstance(touched.toolkit, Toolkit)
    # The Toolkit holds the benchmark next to the tickers; the Portfolio's own data does not.
    pd.testing.assert_frame_equal(touched._daily_historical_data, daily_historical_data)

    for method, kwargs in [
        ("get_portfolio_overview", {}),
        ("get_positions_overview", {}),
        ("get_portfolio_performance", {"period": "yearly"}),
        ("get_transactions_overview", {}),
        ("get_transactions_performance", {"period": "yearly"}),
        ("get_money_weighted_return", {}),
        ("get_turnover", {"period": "yearly"}),
        ("get_gross_leverage", {"period": "yearly"}),
        ("get_return_contributions", {"period": "yearly"}),
        ("get_trades", {}),
        ("get_trade_statistics", {}),
        ("get_position_timing", {}),
    ]:
        pd.testing.assert_frame_equal(
            getattr(touched, method)(**kwargs),
            getattr(untouched, method)(**kwargs),
            obj=method,
        )


def test_the_portfolio_in_its_toolkit_has_its_own_return_series():
    held = _offline_portfolio()
    toolkit = held.toolkit
    returns = toolkit.get_historical_data()["Return"]

    # Every day each position earns its return with its weight at the previous close.
    weights = held._positions_overview["Current Weight"].astype(float)
    expected = (
        (
            weights.shift(1)
            * held._daily_historical_data["Return"][weights.columns].fillna(0.0)
        )
        .sum(axis=1, min_count=1)
        .where(weights.shift(1).fillna(0.0).sum(axis=1) > 0)
    )
    # The Toolkit rounds to the four decimals of the Portfolio.
    pd.testing.assert_series_equal(
        returns["Portfolio"], expected, check_names=False, check_freq=False, atol=5e-5
    )

    # The volatility of the Portfolio is that of its returns, not a weighted average of the holdings' volatilities.
    volatility = toolkit.risk.get_volatility(period="yearly")
    last_year = volatility.index[-1]
    assert volatility.loc[last_year, "Portfolio"] == pytest.approx(
        returns["Portfolio"].loc[str(last_year)].std() * np.sqrt(252), abs=1e-3
    )
    holdings = volatility.loc[last_year].drop(["Portfolio", "Benchmark"])
    assert volatility.loc[last_year, "Portfolio"] < holdings.max()


def test_another_benchmark_replaces_the_one_collected(monkeypatch):
    from financetoolkit.portfolio import portfolio_controller  # noqa: PLC0415

    held = _offline_portfolio()
    held.get_portfolio_overview()
    assert not held._portfolio_volatilities.empty

    monkeypatch.setattr(portfolio_controller, "Toolkit", _EmptyToolkit)
    held.collect_benchmark_historical_data(benchmark_ticker="TLT")

    # The benchmark collected before and what was calculated from it are gone.
    assert _EmptyToolkit.tickers == ["TLT"]
    assert held._benchmark_ticker == "TLT"
    assert held._portfolio_volatilities.empty
    assert held._daily_benchmark_data.empty
    assert held._toolkit_instance is None

    # Asking for the current benchmark, by name or by leaving it out, keeps what was collected.
    _EmptyToolkit.tickers = []
    held.collect_benchmark_historical_data()
    held.collect_benchmark_historical_data(benchmark_ticker="TLT")
    assert held._benchmark_ticker == "TLT"
    assert _EmptyToolkit.tickers == []


def test_reading_the_transactions_again_starts_over():
    held = _offline_portfolio()
    held.get_portfolio_overview()
    transactions = held.read_portfolio_dataset()

    assert held._daily_historical_data.empty
    assert held._weekly_benchmark_data.empty
    assert held._positions_overview.empty
    assert held._toolkit is None
    pd.testing.assert_frame_equal(
        transactions,
        Portfolio(
            portfolio_dataset="tests/datasets/portfolio_test.xlsx"
        )._portfolio_dataset,
    )


def test_the_money_weighted_return_does_not_take_the_rounding_of_the_portfolio():
    rounded = _offline_portfolio()
    rounded._rounding = 0
    unrounded = _offline_portfolio()

    pd.testing.assert_frame_equal(
        rounded.get_money_weighted_return(rounding=6),
        unrounded.get_money_weighted_return(rounding=6),
    )


def test_a_purchase_from_before_a_split_is_restated(monkeypatch):
    from financetoolkit.portfolio import portfolio_controller  # noqa: PLC0415

    held = _offline_portfolio()
    ticker = held._tickers[0]
    first = held._portfolio_dataset.index[
        held._portfolio_dataset.index.get_level_values(1) == ticker
    ][0]
    volume = held._portfolio_dataset.loc[first, held._volume_column]
    price = held._portfolio_dataset.loc[first, held._price_column]
    # The purchase as confirmed before a four for one split the day after it.
    held._portfolio_dataset.loc[first, held._volume_column] = volume / 4
    held._portfolio_dataset.loc[first, held._price_column] = price * 4
    split_date = first[0].to_timestamp() + pd.Timedelta(days=1)

    class _SplitToolkit:
        def __init__(self, tickers, **kwargs):  # noqa: ARG002
            self.tickers = list(tickers)

        def get_stock_splits(self):
            assert self.tickers == [ticker]
            return pd.DataFrame(
                {"Numerator": [4], "Denominator": [1]},
                index=pd.Index([split_date], name="Date"),
            )

    monkeypatch.setattr(portfolio_controller, "Toolkit", _SplitToolkit)
    held._api_key = "KEY"
    held._restate_split_transactions()

    assert held._portfolio_dataset.loc[first, held._volume_column] == pytest.approx(
        volume
    )
    assert held._portfolio_dataset.loc[first, held._price_column] == pytest.approx(
        price
    )


def test_the_performance_of_the_portfolio_in_its_toolkit_comes_from_its_own_returns():
    toolkit = _offline_portfolio().toolkit
    performance = toolkit.performance
    performance._rounding = None
    returns = performance._historical_data["daily"]["Return"]

    # The Portfolio is listed once, next to the tickers whose returns it combines.
    assert performance._tickers.count("Portfolio") == 1

    # The excess returns over a trading day of the annualized risk free rate quoted on each date.
    rate = risk_free_rate["Adj Close"].copy()
    rate.index = pd.PeriodIndex(rate.index, freq="D")
    rate = rate.reindex(rate.index.union(returns.index)).ffill().reindex(returns.index)
    last_year = str(returns.index[-1].year)
    portfolio = returns["Portfolio"].loc[last_year].dropna()
    benchmark = returns["Benchmark"].loc[last_year].reindex(portfolio.index)
    excess_return = portfolio - ((1 + rate.loc[portfolio.index]) ** (1 / 252) - 1)

    sharpe_ratio = performance.get_sharpe_ratio(period="yearly")
    assert sharpe_ratio.iloc[-1]["Portfolio"] == pytest.approx(
        excess_return.mean() / excess_return.std()
    )
    assert performance.get_beta(period="yearly").iloc[-1]["Portfolio"] == pytest.approx(
        np.cov(portfolio, benchmark)[0, 1] / benchmark.var()
    )


def test_a_fully_sold_portfolio_says_why_its_overview_is_empty(monkeypatch, caplog):
    trades = pd.DataFrame(
        {
            "Date": ["2021-01-11", "2021-06-01"],
            "Ticker": ["AAPL", "AAPL"],
            "Volume": [10, -10],
            "Costs": [0.0, 0.0],
        }
    )
    sold, _ = _self_benchmarked_portfolio(monkeypatch, trades)

    with caplog.at_level("WARNING"):
        assert sold.get_portfolio_overview().empty
        assert sold.get_portfolio_performance(period="yearly").empty

    assert "the overview without sold positions is empty" in caplog.text
    assert "the performance without sold positions is empty" in caplog.text
    assert not sold.get_portfolio_overview(exclude_sold_positions=False).empty


@pytest.mark.parametrize(
    "method", ["get_portfolio_performance", "get_transactions_performance"]
)
def test_an_unknown_portfolio_period_raises(method):
    with pytest.raises(InvalidArgumentError, match="period must be one of"):
        getattr(portfolio, method)(period="hourly")
