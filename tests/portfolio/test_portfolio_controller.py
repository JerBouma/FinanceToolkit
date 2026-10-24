"""Portfolio Controller Tests""" ""
import warnings

import pandas as pd
import pytest

from financetoolkit import Portfolio, Toolkit

# pylint: disable=protected-access

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
portfolio._toolkit._daily_risk_free_rate = risk_free_rate

# pylint: disable=missing-function-docstring,protected-access


def test_read_portfolio_dataset(recorder):
    recorder.capture(portfolio.read_portfolio_dataset(adjust_duplicates=True))
    recorder.capture(portfolio.read_portfolio_dataset(adjust_duplicates=False))


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
        weighted._monthly_historical_data[("Adj Close", "MSFT")].iloc[-1]
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
    assert float(weighted._yearly_benchmark_data["Close"].iloc[-1]) == pytest.approx(
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


def _self_benchmarked_portfolio(monkeypatch, transactions):
    """A portfolio of transactions whose benchmark is AAPL for every ticker."""
    prices = pd.read_pickle("tests/datasets/historical_dataset.pickle")
    # The dataset's own benchmark column is left out, so it does not pose as a holding.
    holdings = prices.drop(columns="Benchmark", level=1)

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


def test_the_portfolio_return_of_buys_alone_is_on_the_amount_invested(monkeypatch):
    trades = pd.DataFrame(
        {
            "Date": ["2021-01-11", "2021-06-01", "2021-06-01"],
            "Ticker": ["AAPL", "AAPL", "MSFT"],
            "Volume": [10, 5, 3],
            "Costs": [-1.0, -1.0, -1.0],
        }
    )
    self_benchmarked, _ = _self_benchmarked_portfolio(monkeypatch, trades)

    overview = self_benchmarked.get_portfolio_overview(rounding=12)

    assert overview.loc["Portfolio", "Return"] == pytest.approx(
        overview.loc["Portfolio", "Return Value"]
        / overview.loc["Portfolio", "Invested"]
    )
    assert self_benchmarked._positions_overview["Dividends"].iloc[-1].sum() > 0


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
