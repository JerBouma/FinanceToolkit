# ruff: noqa

"""Fixed Income Controller Tests"""

import pytest

from financetoolkit.fixedincome import fixedincome_controller

# pylint: disable=missing-function-docstring


def test_fixedincome_controller_initialization(recorder):
    fixedincome = fixedincome_controller.FixedIncome(
        start_date="2020-01-01", end_date="2023-12-31", quarterly=True, rounding=4
    )

    recorder.capture(fixedincome._start_date == "2020-01-01")
    recorder.capture(fixedincome._end_date == "2023-12-31")
    recorder.capture(fixedincome._quarterly == True)
    recorder.capture(fixedincome._rounding == 4)


def test_collect_bond_statistics(recorder, fixedincome_module):
    recorder.capture(fixedincome_module.collect_bond_statistics())
    recorder.capture(
        fixedincome_module.collect_bond_statistics(
            par_value=1000,
            coupon_rate=0.06,
            years_to_maturity=10,
            yield_to_maturity=0.07,
        )
    )


def test_get_present_value(recorder, fixedincome_module):
    recorder.capture(
        fixedincome_module.get_present_value(
            par_value=1000,
            coupon_rate=0.05,
            years_to_maturity=5,
            yield_to_maturity=0.06,
            frequency=1,
        )
    )
    recorder.capture(
        fixedincome_module.get_present_value(
            par_value=1000,
            coupon_rate=0.05,
            years_to_maturity=5,
            yield_to_maturity=0.06,
            frequency=2,
        )
    )


def test_get_duration(recorder, fixedincome_module):
    recorder.capture(fixedincome_module.get_duration())
    recorder.capture(fixedincome_module.get_duration(duration_type="macaulay"))
    recorder.capture(fixedincome_module.get_duration(duration_type="effective"))
    recorder.capture(fixedincome_module.get_duration(duration_type="dollar"))


def test_get_yield_to_maturity(recorder, fixedincome_module):
    recorder.capture(fixedincome_module.get_yield_to_maturity())
    recorder.capture(
        fixedincome_module.get_yield_to_maturity(
            par_value=1000, coupon_rate=0.05, years_to_maturity=5, bond_price=950
        )
    )


def test_get_forward_rate(recorder, fixedincome_module):
    recorder.capture(
        fixedincome_module.get_forward_rate(
            near_maturity=[1, 2, 3], far_maturity=[5, 10]
        )
    )


def test_get_par_yield(recorder, fixedincome_module):
    recorder.capture(
        fixedincome_module.get_par_yield(years_to_maturity=[1, 2, 3, 5, 10])
    )


def test_get_yield_curve_spread(recorder, fixedincome_module):
    recorder.capture(
        fixedincome_module.get_yield_curve_spread(
            long_maturity=[10, 30], short_maturity=[1, 2]
        )
    )


def test_get_breakeven_inflation_rate(recorder, fixedincome_module):
    recorder.capture(
        fixedincome_module.get_breakeven_inflation_rate(maturity=[1, 5, 10, 30])
    )


def test_get_z_spread(recorder, fixedincome_module):
    recorder.capture(
        fixedincome_module.get_z_spread(
            coupon_rate=0.05,
            years_to_maturity=[5, 10, 15],
            bond_price=[95, 100, 105],
        )
    )


def test_get_bond_equivalent_yield(recorder, fixedincome_module):
    recorder.capture(
        fixedincome_module.get_bond_equivalent_yield(
            discount_yield=[0.03, 0.05, 0.07], days_to_maturity=[90, 180, 360]
        )
    )


def test_get_key_rate_duration(recorder, fixedincome_module):
    recorder.capture(
        fixedincome_module.get_key_rate_duration(
            coupon_rate=0.05,
            years_to_maturity=[5, 10],
            key_rate_maturity=[2, 5, 10],
        )
    )


def test_get_taylor_price_change(recorder, fixedincome_module):
    recorder.capture(
        fixedincome_module.get_taylor_price_change(
            coupon_rate=[0.03, 0.05, 0.07],
            years_to_maturity=[5, 10, 15],
            yield_to_maturity=0.08,
            yield_change=0.01,
        )
    )


def test_get_derivative_price(recorder, fixedincome_module):
    recorder.capture(fixedincome_module.get_derivative_price())
    recorder.capture(fixedincome_module.get_derivative_price(model="bachelier"))
    recorder.capture(fixedincome_module.get_derivative_price(is_receiver=False))


def test_get_government_bond_yield(recorder, fixedincome_module, live_mode):
    if not live_mode:
        pytest.skip("Requires --live flag for live OECD data")
    recorder.capture(fixedincome_module.get_government_bond_yield())
    recorder.capture(fixedincome_module.get_government_bond_yield(short_term=True))
    recorder.capture(fixedincome_module.get_government_bond_yield(growth=True))


def test_get_ice_bofa_option_adjusted_spread(recorder, fixedincome_module, live_mode):
    if not live_mode:
        pytest.skip("Requires --live flag for live FRED data")
    recorder.capture(fixedincome_module.get_ice_bofa_option_adjusted_spread())
    recorder.capture(
        fixedincome_module.get_ice_bofa_option_adjusted_spread(maturity=False)
    )


def test_get_ice_bofa_effective_yield(recorder, fixedincome_module, live_mode):
    if not live_mode:
        pytest.skip("Requires --live flag for live FRED data")
    recorder.capture(fixedincome_module.get_ice_bofa_effective_yield())
    recorder.capture(fixedincome_module.get_ice_bofa_effective_yield(maturity=False))


def test_get_ice_bofa_total_return(recorder, fixedincome_module, live_mode):
    if not live_mode:
        pytest.skip("Requires --live flag for live FRED data")
    recorder.capture(fixedincome_module.get_ice_bofa_total_return())
    recorder.capture(fixedincome_module.get_ice_bofa_total_return(maturity=False))


def test_get_ice_bofa_yield_to_worst(recorder, fixedincome_module, live_mode):
    if not live_mode:
        pytest.skip("Requires --live flag for live FRED data")
    recorder.capture(fixedincome_module.get_ice_bofa_yield_to_worst())
    recorder.capture(fixedincome_module.get_ice_bofa_yield_to_worst(maturity=False))


def test_get_euribor_rates(recorder, fixedincome_module, live_mode):
    if not live_mode:
        pytest.skip("Requires --live flag for live ECB data")
    recorder.capture(fixedincome_module.get_euribor_rates())
    recorder.capture(fixedincome_module.get_euribor_rates(maturities=["1M", "3M"]))


def test_get_european_central_bank_rates(recorder, fixedincome_module, live_mode):
    if not live_mode:
        pytest.skip("Requires --live flag for live ECB data")
    recorder.capture(fixedincome_module.get_european_central_bank_rates())


def test_get_federal_reserve_rates(recorder, fixedincome_module, live_mode):
    if not live_mode:
        pytest.skip("Requires --live flag for live Federal Reserve data")
    recorder.capture(fixedincome_module.get_federal_reserve_rates())
    recorder.capture(fixedincome_module.get_federal_reserve_rates(rate="SOFR"))
