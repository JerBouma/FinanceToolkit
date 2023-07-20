"""Models Controller Tests""" ""
import pandas as pd

from financetoolkit import Toolkit

# pylint: disable=missing-function-docstring

balance_dataset = pd.read_pickle("tests/base/datasets/balance_dataset.pickle")

income_dataset = pd.read_pickle("tests/base/datasets/income_dataset.pickle")
cash_dataset = pd.read_pickle("tests/base/datasets/cash_dataset.pickle")

toolkit = Toolkit(
    tickers=["AAPL", "MSFT"],
    balance=balance_dataset,
    income=income_dataset,
    cash=cash_dataset,
)

models_module = toolkit.models


def test_get_dupont_analysis(recorder):
    recorder.capture(models_module.get_dupont_analysis())


def test_get_extended_dupont_analysis(recorder):
    recorder.capture(models_module.get_extended_dupont_analysis())
