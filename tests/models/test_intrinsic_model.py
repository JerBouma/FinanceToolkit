"""Intrinsic Model Tests""" ""

from financetoolkit.models import intrinsic_model

# pylint: disable=missing-function-docstring


def test_get_intrinsic_value(recorder):
    recorder.capture(
        intrinsic_model.get_intrinsic_value(
            cash_flow=500,
            growth_rate=0.05,
            perpetual_growth_rate=0.02,
            weighted_average_cost_of_capital=0.1,
            cash_and_cash_equivalents=100,
            total_debt=100,
            shares_outstanding=100,
            periods=5,
        )
    )
