"""Portfolio Controller Tests""" ""
import pandas as pd

from financetoolkit import Portfolio

historical = pd.read_pickle("tests/datasets/portfolio_historical_dataset.pickle")
benchmark = pd.read_pickle("tests/datasets/portfolio_benchmark_dataset.pickle")

portfolio = Portfolio(portfolio_dataset="tests/datasets/portfolio_test.xlsx")

portfolio._daily_historical_data = historical
portfolio._daily_benchmark_data = benchmark
portfolio._currency_column = None

# pylint: disable=missing-function-docstring,protected-access


def test_read_portfolio_dataset(recorder):
    recorder.capture(portfolio.read_portfolio_dataset(adjust_duplicates=True))
    recorder.capture(portfolio.read_portfolio_dataset(adjust_duplicates=False))


def test_collect_benchmark_historical_data(recorder):
    recorder.capture(portfolio.collect_benchmark_historical_data())


def test_collect_historical_data(recorder):
    recorder.capture(portfolio.collect_historical_data())


def test_get_positions_overview(recorder):
    recorder.capture(portfolio.get_positions_overview())


def test_get_portfolio_overview(recorder):
    recorder.capture(portfolio.get_portfolio_overview(include_portfolio=True))
    recorder.capture(portfolio.get_portfolio_overview(include_portfolio=False))


def test_get_portfolio_performance(recorder):
    recorder.capture(portfolio.get_portfolio_performance(period="daily"))
    recorder.capture(portfolio.get_portfolio_performance(period="weekly"))
    recorder.capture(portfolio.get_portfolio_performance(period="monthly"))
    recorder.capture(portfolio.get_portfolio_performance(period="quarterly"))
    recorder.capture(portfolio.get_portfolio_performance(period="yearly"))


def test_get_transactions_overview(recorder):
    recorder.capture(portfolio.get_transactions_overview(pnl_method="FIFO"))
    recorder.capture(portfolio.get_transactions_overview(pnl_method="LIFO"))
    recorder.capture(portfolio.get_transactions_overview(pnl_method="average"))


def test_get_transactions_performance(recorder):
    recorder.capture(portfolio.get_transactions_performance(period="daily"))
    recorder.capture(portfolio.get_transactions_performance(period="weekly"))
    recorder.capture(portfolio.get_transactions_performance(period="monthly"))
    recorder.capture(portfolio.get_transactions_performance(period="quarterly"))
    recorder.capture(portfolio.get_transactions_performance(period="yearly"))
