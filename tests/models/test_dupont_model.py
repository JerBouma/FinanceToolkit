"""Dupont Model Tests""" ""
import pandas as pd

from financetoolkit.models.profitability import dupont_model


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


def test_get_nissim_penman_decomposition(recorder):
    recorder.capture(
        dupont_model.get_nissim_penman_decomposition(
            net_income=pd.Series([100, 110, 120, 130, 80]),
            net_financial_expense=pd.Series([10, 12, -5, 8, 6]),
            average_net_financial_obligations=pd.Series([200, 220, -150, 160, 120]),
            average_common_equity=pd.Series([400, 420, 450, 380, 300]),
        )
    )


def test_nissim_penman_decomposition_adds_up_to_the_return_on_common_equity():
    net_income = pd.Series([100.0, 110.0, 120.0, -30.0])
    common_equity = pd.Series([400.0, 420.0, 450.0, 380.0])
    decomposition = dupont_model.get_nissim_penman_decomposition(
        net_income=net_income,
        net_financial_expense=pd.Series([10.0, 12.0, -5.0, 8.0]),
        average_net_financial_obligations=pd.Series([200.0, 220.0, -150.0, 160.0]),
        average_common_equity=common_equity,
    )

    pd.testing.assert_series_equal(
        decomposition.loc["Return on Common Equity"],
        net_income / common_equity,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        decomposition.loc["Leverage Effect"],
        decomposition.loc["Financial Leverage"] * decomposition.loc["Spread"],
        check_names=False,
    )
