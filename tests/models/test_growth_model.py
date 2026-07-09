"""Growth Model Tests""" ""

import pandas as pd

from financetoolkit.models import growth_model

# pylint: disable=missing-function-docstring


def test_get_present_value_of_growth_opportunities(recorder):
    recorder.capture(
        growth_model.get_present_value_of_growth_opportunities(
            weighted_average_cost_of_capital=pd.Series([0.30, 0.05, 0.10]),
            earnings_per_share=pd.Series([0.10, 0.02, 0.03]),
            close_prices=pd.Series([1, 2, 3]),
        )
    )


def test_get_sustainable_growth_rate(recorder):
    recorder.capture(
        growth_model.get_sustainable_growth_rate(
            return_on_equity=pd.Series([0.15, 0.18, 0.12, 0.2]),
            retention_ratio=pd.Series([0.6, 0.55, 0.7, 0.5]),
        )
    )


def test_get_internal_growth_rate(recorder):
    recorder.capture(
        growth_model.get_internal_growth_rate(
            return_on_assets=pd.Series([0.08, 0.09, 0.07, 0.1]),
            retention_ratio=pd.Series([0.6, 0.55, 0.7, 0.5]),
        )
    )
