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
