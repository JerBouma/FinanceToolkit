# ruff: noqa

"""Fixed Income Controller Tests"""


import pandas as pd

from financetoolkit.fixedincome import fixedincome_controller

# pylint: disable=missing-function-docstring


def test_fixedincome_controller_initialization(recorder):
    """Test that Fixed Income controller initializes correctly."""
    fixedincome = fixedincome_controller.FixedIncome(
        start_date="2020-01-01", end_date="2023-12-31", quarterly=True, rounding=4
    )

    recorder.capture(fixedincome._start_date == "2020-01-01")
    recorder.capture(fixedincome._end_date == "2023-12-31")
    recorder.capture(fixedincome._quarterly == True)
    recorder.capture(fixedincome._rounding == 4)


def test_get_present_value(recorder):
    """Test get_present_value method."""
    fixedincome = fixedincome_controller.FixedIncome()

    # Test present value calculation with proper parameters
    result = fixedincome.get_present_value(
        par_value=1000,
        coupon_rate=0.05,
        years_to_maturity=5,
        yield_to_maturity=0.06,
        frequency=1,
    )

    recorder.capture(isinstance(result, pd.DataFrame))
    recorder.capture(result.shape[0] > 0)
    recorder.capture(result.shape[1] > 0)
