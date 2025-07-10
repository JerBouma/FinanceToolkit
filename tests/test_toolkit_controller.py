# ruff: noqa
"""Toolkit Controller Tests""" ""
import pandas as pd

from financetoolkit import Toolkit

balance_dataset = pd.read_pickle("tests/datasets/balance_dataset.pickle")
income_dataset = pd.read_pickle("tests/datasets/income_dataset.pickle")
cash_dataset = pd.read_pickle("tests/datasets/cash_dataset.pickle")
historical_dataset = pd.read_pickle("tests/datasets/historical_dataset.pickle")
risk_free_rate = pd.read_pickle("tests/datasets/risk_free_rate.pickle")
treasury_data = pd.read_pickle("tests/datasets/treasury_data.pickle")


# pylint: disable=missing-function-docstring


def test_toolkit_balance(recorder):
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        balance=balance_dataset,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    toolkit._daily_risk_free_rate = risk_free_rate
    toolkit._daily_treasury_data = treasury_data

    recorder.capture(toolkit.get_balance_sheet_statement())
    recorder.capture(toolkit.get_balance_sheet_statement(growth=True))
    recorder.capture(toolkit.get_balance_sheet_statement(growth=True, lag=[1, 2, 3]))


def test_toolkit_income(recorder):
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        income=income_dataset,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    toolkit._daily_risk_free_rate = risk_free_rate
    toolkit._daily_treasury_data = treasury_data

    recorder.capture(toolkit.get_income_statement())
    recorder.capture(toolkit.get_income_statement(growth=True))
    recorder.capture(toolkit.get_income_statement(growth=True, lag=[1, 2, 3]))


def test_toolkit_cash(recorder):
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        cash=cash_dataset,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    toolkit._daily_risk_free_rate = risk_free_rate
    toolkit._daily_treasury_data = treasury_data

    recorder.capture(toolkit.get_cash_flow_statement())
    recorder.capture(toolkit.get_cash_flow_statement(growth=True))
    recorder.capture(toolkit.get_cash_flow_statement(growth=True, lag=[1, 2, 3]))


def test_toolkit_historical(recorder):
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=historical_dataset,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    toolkit._daily_risk_free_rate = risk_free_rate
    toolkit._daily_treasury_data = treasury_data

    recorder.capture(toolkit.get_historical_data(period="yearly").round(0))


def test_toolkit_ratios(recorder):
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        balance=balance_dataset,
        income=income_dataset,
        cash=cash_dataset,
        historical=historical_dataset,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    toolkit._daily_risk_free_rate = risk_free_rate
    toolkit._daily_treasury_data = treasury_data

    recorder.capture(toolkit.ratios.get_debt_to_assets_ratio())
    recorder.capture(toolkit.ratios.get_debt_to_assets_ratio(growth=True))
    recorder.capture(
        toolkit.ratios.get_debt_to_assets_ratio(growth=True, lag=[1, 2, 3])
    )


def test_toolkit_models(recorder):
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=historical_dataset,
        balance=balance_dataset,
        income=income_dataset,
        cash=cash_dataset,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    toolkit._daily_risk_free_rate = risk_free_rate
    toolkit._daily_treasury_data = treasury_data

    recorder.capture(toolkit.models.get_dupont_analysis())
    recorder.capture(toolkit.models.get_dupont_analysis(growth=True))
    recorder.capture(toolkit.models.get_dupont_analysis(growth=True, lag=[1, 2, 3]))


def test_toolkit_performance(recorder):
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        balance=balance_dataset,
        income=income_dataset,
        cash=cash_dataset,
        historical=historical_dataset,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    toolkit._daily_risk_free_rate = risk_free_rate
    toolkit._daily_treasury_data = treasury_data

    recorder.capture(toolkit.performance.get_beta())
    recorder.capture(toolkit.performance.get_beta(growth=True))
    recorder.capture(toolkit.performance.get_beta(growth=True, lag=[1, 2, 3]))


def test_toolkit_risk(recorder):
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        balance=balance_dataset,
        income=income_dataset,
        cash=cash_dataset,
        historical=historical_dataset,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    toolkit._daily_risk_free_rate = risk_free_rate
    toolkit._daily_treasury_data = treasury_data

    recorder.capture(toolkit.risk.get_conditional_value_at_risk())
    recorder.capture(toolkit.risk.get_conditional_value_at_risk(growth=True))
    recorder.capture(
        toolkit.risk.get_conditional_value_at_risk(growth=True, lag=[1, 2, 3])
    )


def test_toolkit_technicals(recorder):
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        balance=balance_dataset,
        income=income_dataset,
        cash=cash_dataset,
        historical=historical_dataset,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    toolkit._daily_risk_free_rate = risk_free_rate
    toolkit._daily_treasury_data = treasury_data

    recorder.capture(toolkit.technicals.collect_all_indicators().round(0))
    recorder.capture(toolkit.technicals.collect_all_indicators(growth=True).round(0))
    recorder.capture(
        toolkit.technicals.collect_all_indicators(growth=True, lag=[1, 2, 3]).round(0)
    )
