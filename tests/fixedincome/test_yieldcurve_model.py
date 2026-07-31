"""Yield Curve Model Tests"""

import pandas as pd
import pytest

from financetoolkit.fixedincome import yieldcurve_model

# pylint: disable=missing-function-docstring

FLAT_CURVE_RATE = 0.05


def test_get_forward_rate(recorder):
    recorder.capture(
        yieldcurve_model.get_forward_rate(
            near_rate=0.03, far_rate=0.05, near_maturity=1, far_maturity=3
        )
    )


def test_get_forward_rate_flat_curve_equals_spot_rate(recorder):
    forward_rate = yieldcurve_model.get_forward_rate(
        near_rate=0.05, far_rate=0.05, near_maturity=1, far_maturity=3
    )
    recorder.capture(bool(round(forward_rate, 8) == FLAT_CURVE_RATE))


def test_get_forward_rate_with_series(recorder):
    near_rate = pd.Series([0.03, 0.032])
    far_rate = pd.Series([0.05, 0.052])
    near_maturity = pd.Series([1, 2])
    far_maturity = pd.Series([3, 4])
    recorder.capture(
        yieldcurve_model.get_forward_rate(
            near_rate=near_rate,
            far_rate=far_rate,
            near_maturity=near_maturity,
            far_maturity=far_maturity,
        )
    )


def test_get_forward_rate_invalid_maturity_order():
    with pytest.raises(ValueError):
        yieldcurve_model.get_forward_rate(
            near_rate=0.03, far_rate=0.05, near_maturity=5, far_maturity=1
        )


def test_get_forward_rate_type_error():
    with pytest.raises(TypeError):
        yieldcurve_model.get_forward_rate(
            near_rate="0.03", far_rate=0.05, near_maturity=1, far_maturity=3
        )


def test_get_par_yield(recorder):
    spot_rates = pd.Series({1: 0.03, 2: 0.04, 3: 0.05})
    recorder.capture(
        yieldcurve_model.get_par_yield(
            spot_rates=spot_rates, years_to_maturity=3, frequency=1
        )
    )


def test_get_par_yield_flat_curve_equals_spot_rate(recorder):
    spot_rates = pd.Series({1: 0.05, 2: 0.05, 3: 0.05, 5: 0.05, 10: 0.05})
    par_yield = yieldcurve_model.get_par_yield(
        spot_rates=spot_rates, years_to_maturity=5, frequency=1
    )
    recorder.capture(bool(round(par_yield, 8) == FLAT_CURVE_RATE))


def test_get_par_yield_type_error():
    with pytest.raises(TypeError):
        yieldcurve_model.get_par_yield(
            spot_rates=[0.03, 0.04, 0.05], years_to_maturity=3
        )


def test_get_par_yield_invalid_maturity():
    spot_rates = pd.Series({1: 0.03, 2: 0.04, 3: 0.05})
    with pytest.raises(ValueError):
        yieldcurve_model.get_par_yield(spot_rates=spot_rates, years_to_maturity=0)


def test_get_yield_curve_spread(recorder):
    recorder.capture(
        yieldcurve_model.get_yield_curve_spread(long_yield=0.045, short_yield=0.02)
    )


def test_get_yield_curve_spread_with_series(recorder):
    long_yield = pd.Series([0.045, 0.05])
    short_yield = pd.Series([0.02, 0.025])
    recorder.capture(
        yieldcurve_model.get_yield_curve_spread(
            long_yield=long_yield, short_yield=short_yield
        )
    )


def test_get_yield_curve_spread_type_error():
    with pytest.raises(TypeError):
        yieldcurve_model.get_yield_curve_spread(long_yield="0.045", short_yield=0.02)


def test_get_breakeven_inflation_rate(recorder):
    recorder.capture(
        yieldcurve_model.get_breakeven_inflation_rate(
            nominal_yield=0.045, real_yield=0.018
        )
    )


def test_get_breakeven_inflation_rate_with_series(recorder):
    nominal_yield = pd.Series([0.045, 0.05])
    real_yield = pd.Series([0.018, 0.02])
    recorder.capture(
        yieldcurve_model.get_breakeven_inflation_rate(
            nominal_yield=nominal_yield, real_yield=real_yield
        )
    )


def test_get_breakeven_inflation_rate_type_error():
    with pytest.raises(TypeError):
        yieldcurve_model.get_breakeven_inflation_rate(
            nominal_yield="0.045", real_yield=0.018
        )
