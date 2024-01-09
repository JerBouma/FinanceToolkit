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
