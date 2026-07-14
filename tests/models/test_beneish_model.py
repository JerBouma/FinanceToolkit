"""Beneish Model Tests""" ""
import pandas as pd

from financetoolkit.models import beneish_model

# pylint: disable=missing-function-docstring

COLUMNS = ["2020", "2021", "2022", "2023", "2024"]


def test_get_days_sales_in_receivables_index(recorder):
    recorder.capture(
        beneish_model.get_days_sales_in_receivables_index(
            net_receivables=pd.DataFrame(
                [[80, 90, 105, 130, 140]], index=["AAPL"], columns=COLUMNS
            ),
            revenue=pd.DataFrame(
                [[800, 850, 900, 950, 1000]], index=["AAPL"], columns=COLUMNS
            ),
        )
    )


def test_get_gross_margin_index(recorder):
    recorder.capture(
        beneish_model.get_gross_margin_index(
            revenue=pd.DataFrame(
                [[800, 850, 900, 950, 1000]], index=["AAPL"], columns=COLUMNS
            ),
            cost_of_goods_sold=pd.DataFrame(
                [[500, 520, 560, 570, 610]], index=["AAPL"], columns=COLUMNS
            ),
        )
    )


def test_get_asset_quality_index(recorder):
    recorder.capture(
        beneish_model.get_asset_quality_index(
            total_current_assets=pd.DataFrame(
                [[300, 320, 340, 360, 380]], index=["AAPL"], columns=COLUMNS
            ),
            property_plant_and_equipment=pd.DataFrame(
                [[400, 410, 420, 430, 440]], index=["AAPL"], columns=COLUMNS
            ),
            total_assets=pd.DataFrame(
                [[1000, 1100, 1200, 1300, 1400]], index=["AAPL"], columns=COLUMNS
            ),
        )
    )


def test_get_sales_growth_index(recorder):
    recorder.capture(
        beneish_model.get_sales_growth_index(
            revenue=pd.DataFrame(
                [[800, 850, 900, 950, 1000]], index=["AAPL"], columns=COLUMNS
            ),
        )
    )


def test_get_depreciation_index(recorder):
    recorder.capture(
        beneish_model.get_depreciation_index(
            depreciation_and_amortization=pd.DataFrame(
                [[40, 42, 41, 39, 36]], index=["AAPL"], columns=COLUMNS
            ),
            property_plant_and_equipment=pd.DataFrame(
                [[400, 410, 420, 430, 440]], index=["AAPL"], columns=COLUMNS
            ),
        )
    )


def test_get_selling_general_and_administrative_expenses_index(recorder):
    recorder.capture(
        beneish_model.get_selling_general_and_administrative_expenses_index(
            selling_general_and_administrative_expenses=pd.DataFrame(
                [[90, 95, 105, 108, 120]], index=["AAPL"], columns=COLUMNS
            ),
            revenue=pd.DataFrame(
                [[800, 850, 900, 950, 1000]], index=["AAPL"], columns=COLUMNS
            ),
        )
    )


def test_get_leverage_index(recorder):
    recorder.capture(
        beneish_model.get_leverage_index(
            total_current_liabilities=pd.DataFrame(
                [[200, 210, 220, 230, 240]], index=["AAPL"], columns=COLUMNS
            ),
            long_term_debt=pd.DataFrame(
                [[300, 310, 330, 350, 400]], index=["AAPL"], columns=COLUMNS
            ),
            total_assets=pd.DataFrame(
                [[1000, 1100, 1200, 1300, 1400]], index=["AAPL"], columns=COLUMNS
            ),
        )
    )


def test_get_total_accruals_to_total_assets(recorder):
    recorder.capture(
        beneish_model.get_total_accruals_to_total_assets(
            net_income=pd.Series([100, 110, 120, 130, 140]),
            cash_flow_from_operations=pd.Series([120, 115, 118, 160, 150]),
            total_assets=pd.Series([1000, 1100, 1200, 1300, 1400]),
        )
    )


def test_get_beneish_m_score(recorder):
    recorder.capture(
        beneish_model.get_beneish_m_score(
            days_sales_in_receivables_index=pd.Series([1.1, 1.05, 0.95, 1.2]),
            gross_margin_index=pd.Series([1.02, 0.98, 1.05, 1.1]),
            asset_quality_index=pd.Series([1.0, 1.1, 0.9, 1.15]),
            sales_growth_index=pd.Series([1.05, 1.1, 1.08, 1.2]),
            depreciation_index=pd.Series([1.0, 0.95, 1.05, 0.9]),
            selling_general_and_administrative_expenses_index=pd.Series(
                [1.0, 1.05, 0.98, 1.1]
            ),
            leverage_index=pd.Series([1.0, 1.02, 0.99, 1.05]),
            total_accruals_to_total_assets=pd.Series([0.02, -0.01, 0.03, 0.05]),
        )
    )
