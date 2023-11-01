"""Enterprise Model Tests""" ""
import pandas as pd

from financetoolkit.models import enterprise_model

# pylint: disable=missing-function-docstring


def test_get_enterprise_value_breakdown(recorder):
    recorder.capture(
        enterprise_model.get_enterprise_value_breakdown(
            share_price=pd.Series([100, 110, 120, 130, 80]),
            shares_outstanding=pd.Series([200, 150, 130, 200, 110]),
            total_debt=pd.Series([500, 400, 300, 200, 100]),
            minority_interest=pd.Series([500, 430, 340, 240, 140]),
            preferred_equity=pd.Series([400, 300, 200, 150, 80]),
            cash_and_cash_equivalents=pd.Series([430, 340, 240, 150, 80]),
        )
    )
