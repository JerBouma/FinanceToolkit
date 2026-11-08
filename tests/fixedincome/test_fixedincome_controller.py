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


def test_maturity_in_years_reads_days_and_weeks():
    """Day and week labels, which the MCP server accepts, count calendar days of a 365-day year."""
    assert fixedincome_controller._maturity_in_years("1D") == 1 / 365
    assert fixedincome_controller._maturity_in_years("1W") == 7 / 365
    assert fixedincome_controller._maturity_in_years("2w") == 14 / 365
    assert fixedincome_controller._maturity_in_years("36M") == 3.0
    assert fixedincome_controller._maturity_in_years("10Y") == 10.0
    assert fixedincome_controller._maturity_in_years("6M") == 0.5
    with pytest.raises(ValueError):
        fixedincome_controller._maturity_in_years("ten years")
