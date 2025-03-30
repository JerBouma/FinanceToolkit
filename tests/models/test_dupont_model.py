"""Dupont Model Tests""" ""
import pandas as pd

from financetoolkit.models import dupont_model

# pylint: disable=missing-function-docstring


def test_get_dupont_analysis(recorder):
    recorder.capture(
        dupont_model.get_dupont_analysis(
            net_income=pd.Series([100, 110, 120, 130, 80]),
            total_revenue=pd.Series([200, 150, 130, 200, 110]),
            average_total_assets=pd.Series([500, 400, 300, 200, 100]),
            average_total_equity=pd.Series([400, 300, 200, 150, 80]),
        )
    )


def test_get_extended_dupont_analysis(recorder):
    recorder.capture(
        dupont_model.get_extended_dupont_analysis(
            income_before_tax=pd.Series([130, 120, 110, 150, 100]),
            operating_income=pd.Series([150, 140, 120, 189, 100]),
            net_income=pd.Series([100, 110, 120, 130, 80]),
            total_revenue=pd.Series([200, 150, 130, 200, 110]),
            average_total_assets=pd.Series([500, 400, 300, 200, 100]),
            average_total_equity=pd.Series([400, 300, 200, 150, 80]),
        )
    )
