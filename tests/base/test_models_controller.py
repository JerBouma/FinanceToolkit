"""Models Controller Tests""" ""
import pandas as pd

from financetoolkit import Toolkit

# pylint: disable=missing-function-docstring

balance_dataset = pd.read_csv(
    "tests/base/datasets/balance_dataset.csv", index_col=[0, 1]
)
income_dataset = pd.read_csv("tests/base/datasets/income_dataset.csv", index_col=[0, 1])
cash_dataset = pd.read_csv("tests/base/datasets/cash_dataset.csv", index_col=[0, 1])

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
