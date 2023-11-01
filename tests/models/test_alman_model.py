"""Altman Model Tests""" ""
import pandas as pd

from financetoolkit.models import altman_model

# pylint: disable=missing-function-docstring


def test_get_working_capital_to_total_assets_ratio(recorder):
    recorder.capture(
        altman_model.get_working_capital_to_total_assets_ratio(
            working_capital=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_retained_earnings_to_total_assets_ratio(recorder):
    recorder.capture(
        altman_model.get_retained_earnings_to_total_assets_ratio(
            retained_earnings=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_earnings_before_interest_and_taxes_to_total_assets_ratio(recorder):
    recorder.capture(
        altman_model.get_earnings_before_interest_and_taxes_to_total_assets_ratio(
            ebit=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_market_value_of_equity_to_book_value_of_total_liabilities_ratio(recorder):
    recorder.capture(
        altman_model.get_market_value_of_equity_to_book_value_of_total_liabilities_ratio(
            market_value_of_equity=pd.Series([100, 110, 120, 130, 80]),
            total_liabilities=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_sales_to_total_assets_ratio(recorder):
    recorder.capture(
        altman_model.get_sales_to_total_assets_ratio(
            sales=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_altman_z_score(recorder):
    recorder.capture(
        altman_model.get_altman_z_score(
            working_capital_to_total_assets_ratio=pd.Series([0.2, 0.2, 0.4, 0.5, 0.6]),
            retained_earnings_to_total_assets_ratio=pd.Series(
                [0.2, 0.3, 0.3, 0.5, 0.6]
            ),
            earnings_before_interest_and_taxes_to_total_assets_ratio=pd.Series(
                [0.2, 0.3, 0.3, 0.5, 0.6]
            ),
            market_value_of_equity_to_book_value_of_total_liabilities_ratio=pd.Series(
                [0.2, 0.3, 0.4, 0.3, 0.6]
            ),
            sales_to_total_assets_ratio=pd.Series([0.2, 0.3, 0.4, 0.4, 0.6]),
        )
    )
