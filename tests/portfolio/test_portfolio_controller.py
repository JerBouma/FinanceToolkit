"""Portfolio Controller Tests""" ""
import warnings

import pandas as pd

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
