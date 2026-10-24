# ruff: noqa

"""Fixed Income Controller Tests"""

import pandas as pd
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


def test_curves_with_string_keys_are_sorted_by_maturity(fixedincome_module):
    """'10' and '10Y' sort after '2' as maturities, not as text."""
    numeric = {1: 0.042, 2: 0.04, 5: 0.041, 10: 0.043}
    text = {"1": 0.042, "2": 0.04, "5": 0.041, "10": 0.043}
    labels = {"1Y": 0.042, "2Y": 0.04, "5Y": 0.041, "10Y": 0.043}

    expected = fixedincome_module.get_forward_rate(
        spot_rates=numeric, near_maturity=1, far_maturity=5, show_input_info=False
    )
    for curve in (text, labels):
        pd.testing.assert_frame_equal(
            fixedincome_module.get_forward_rate(
                spot_rates=curve, near_maturity=1, far_maturity=5, show_input_info=False
            ),
            expected,
        )
        pd.testing.assert_series_equal(
            fixedincome_module.get_par_yield(
                spot_rates=curve, years_to_maturity=[3, 7], show_input_info=False
            ),
            fixedincome_module.get_par_yield(
                spot_rates=numeric, years_to_maturity=[3, 7], show_input_info=False
            ),
        )
        pd.testing.assert_frame_equal(
            fixedincome_module.get_z_spread(
                spot_rates=curve,
                years_to_maturity=[3, 7],
                bond_price=[95],
                show_input_info=False,
            ),
            fixedincome_module.get_z_spread(
                spot_rates=numeric,
                years_to_maturity=[3, 7],
                bond_price=[95],
                show_input_info=False,
            ),
        )

    pd.testing.assert_frame_equal(
        fixedincome_module.get_key_rate_duration(
            spot_rates=labels,
            years_to_maturity=[7],
            key_rate_maturity=["5Y", 10],
            show_input_info=False,
        ),
        fixedincome_module.get_key_rate_duration(
            spot_rates=numeric,
            years_to_maturity=[7],
            key_rate_maturity=[5, 10],
            show_input_info=False,
        ),
    )

    # Months are converted to years.
    assert fixedincome_controller._maturity_in_years("6M") == 0.5
    with pytest.raises(ValueError):
        fixedincome_controller._maturity_in_years("ten years")


def test_ice_bofa_option_adjusted_spread_is_decimal(monkeypatch):
    """FRED quotes the OAS in percent (0.77); it is returned as a decimal (0.0077)."""
    from financetoolkit.fixedincome import fred_model

    def fake_fred_data(series, start_date, end_date, api_key):
        return pd.DataFrame(
            [[0.77] * len(series)], index=pd.to_datetime(["2024-01-02"])
        )

    monkeypatch.setattr(fred_model, "get_fred_data", fake_fred_data)

    for oas in (
        fred_model.get_maturity_option_adjusted_spread(
            "2024-01-01", "2024-01-15", "key"
        ),
        fred_model.get_rating_option_adjusted_spread("2024-01-01", "2024-01-15", "key"),
    ):
        assert oas.to_numpy() == pytest.approx(0.0077)
