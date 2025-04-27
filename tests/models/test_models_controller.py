"""Models Controller Tests""" ""
import pandas as pd

from financetoolkit import Toolkit

# pylint: disable=missing-function-docstring

balance_dataset = pd.read_pickle("tests/datasets/balance_dataset.pickle")
historical = pd.read_pickle("tests/datasets/historical_dataset.pickle")
income_dataset = pd.read_pickle("tests/datasets/income_dataset.pickle")
cash_dataset = pd.read_pickle("tests/datasets/cash_dataset.pickle")
risk_free_rate = pd.read_pickle("tests/datasets/risk_free_rate.pickle")
treasury_data = pd.read_pickle("tests/datasets/treasury_data.pickle")

toolkit = Toolkit(
    tickers=["AAPL", "MSFT"],
    historical=historical,
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

models_module = toolkit.models


def test_get_dupont_analysis(recorder):
    recorder.capture(models_module.get_dupont_analysis())
    recorder.capture(models_module.get_dupont_analysis(growth=True))
    recorder.capture(models_module.get_dupont_analysis(growth=True, lag=[1, 2, 3]))


def test_get_extended_dupont_analysis(recorder):
    recorder.capture(models_module.get_extended_dupont_analysis())
    recorder.capture(models_module.get_extended_dupont_analysis(growth=True))
    recorder.capture(
        models_module.get_extended_dupont_analysis(growth=True, lag=[1, 2, 3])
    )


def test_get_enterprise_value_breakdown(recorder):
    recorder.capture(models_module.get_enterprise_value_breakdown())
    recorder.capture(models_module.get_enterprise_value_breakdown(growth=True))
    recorder.capture(
        models_module.get_enterprise_value_breakdown(growth=True, lag=[1, 2, 3])
    )


def test_get_weighted_average_cost_of_capital(recorder):
    recorder.capture(models_module.get_weighted_average_cost_of_capital().round(2))
    recorder.capture(
        models_module.get_weighted_average_cost_of_capital(growth=True).round(2)
    )
    recorder.capture(
        models_module.get_weighted_average_cost_of_capital(growth=True, lag=[1, 2, 3])
    )


def test_get_gorden_growth_model(recorder):
    recorder.capture(
        models_module.get_gorden_growth_model(rate_of_return=0.1, growth_rate=0.09)
    )
    recorder.capture(
        models_module.get_gorden_growth_model(
            rate_of_return=0.2, growth_rate=0.1, project_periods=10
        )
    )


def test_get_altman_z_score(recorder):
    recorder.capture(models_module.get_altman_z_score())
    recorder.capture(models_module.get_altman_z_score(growth=True))
    recorder.capture(models_module.get_altman_z_score(growth=True, lag=[1, 2, 3]))


def test_get_present_value_of_growth_opportunities(recorder):
    recorder.capture(models_module.get_present_value_of_growth_opportunities())
    recorder.capture(
        models_module.get_present_value_of_growth_opportunities(calculate_daily=True)
    )
