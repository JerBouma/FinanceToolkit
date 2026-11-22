"""CFROI Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.models.valuation import cfroi_model


def frame(value: float) -> pd.DataFrame:
    return pd.DataFrame([[value]])


def test_cfroi_is_the_internal_rate_of_return_of_the_assets():
    # A gross investment of 1,000 with 200 of non-depreciating assets, a gross cash flow
    # of 180 for 8 years and the 200 released at the end.
    result = cfroi_model.get_cash_flow_return_on_investment(
        gross_investment=frame(1000.0),
        gross_cash_flow=frame(180.0),
        non_depreciating_assets=frame(200.0),
        asset_life=frame(8.0),
    )
    rate = result.iloc[0, 0]
    cash_flows = [-1000.0] + [180.0] * 7 + [380.0]
    net_present_value = sum(
        flow / (1 + rate) ** year for year, flow in enumerate(cash_flows)
    )
    assert net_present_value == pytest.approx(0, abs=1e-6)
    # The root of the net present value of these cash flows (scipy's brentq): about 11.34% a year.
    assert rate == pytest.approx(0.113365, abs=1e-6)


def test_cfroi_without_a_solution_is_nan():
    result = cfroi_model.get_cash_flow_return_on_investment(
        gross_investment=pd.DataFrame([[1000.0, -5.0, 1000.0]]),
        gross_cash_flow=pd.DataFrame([[-100.0, 100.0, np.nan]]),
        non_depreciating_assets=pd.DataFrame([[0.0, 0.0, 0.0]]),
        asset_life=pd.DataFrame([[5.0, 5.0, 5.0]]),
    )
    assert result.isna().all().all()
