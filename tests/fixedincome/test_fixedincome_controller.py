# ruff: noqa

"""Fixed Income Controller Tests"""

import pandas as pd
import pytest

from financetoolkit.fixedincome import fixedincome_controller


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


@pytest.fixture
def weekly_mortgage_rates(monkeypatch):
    """The 30-year mortgage rate of three survey dates, read without a FRED API key."""
    from financetoolkit.economics import freddie_mac_model

    monkeypatch.delenv("FRED_API_KEY", raising=False)
    dates = pd.PeriodIndex(["2021-01-07", "2023-10-26", "2026-11-05"], freq="D")
    monkeypatch.setattr(
        freddie_mac_model,
        "get_mortgage_rate_30_year",
        lambda: pd.DataFrame({"United States": [0.0265, 0.0779, 0.0718]}, index=dates),
    )
    return fixedincome_controller.FixedIncome(
        start_date="2021-01-01",
        end_date="2026-11-06",
        rounding=None,
        use_cached_data=False,
    )


def test_get_loan_payment_follows_the_mortgage_rate(weekly_mortgage_rates):
    payments = weekly_mortgage_rates.get_loan_payment()

    assert list(payments.columns) == [
        "Rate",
        "Payment",
        "Total Interest",
        "First Payment Interest",
    ]
    assert payments["Payment"].iloc[0] == pytest.approx(1611.8552, abs=1e-4)
    assert payments["Payment"].iloc[-1] == pytest.approx(2709.7393, abs=1e-4)
    assert payments["Total Interest"].iloc[-1] == pytest.approx(575_506.16, abs=0.01)
    assert payments["First Payment Interest"].iloc[0] == pytest.approx(
        883.3333, abs=1e-4
    )


def test_get_affordable_loan_keeps_the_first_payment(weekly_mortgage_rates):
    affordable = weekly_mortgage_rates.get_affordable_loan()

    assert affordable["Affordable Loan"].iloc[0] == pytest.approx(400_000)
    assert affordable["Affordable Loan"].iloc[-1] == pytest.approx(237_935.09, abs=0.01)
    assert affordable["Change"].iloc[-1] == pytest.approx(237_935.09 / 400_000 - 1)

    # A payment of its own is carried at every rate.
    own = weekly_mortgage_rates.get_affordable_loan(payment=2000)
    assert (own["Payment"] == 2000).all()  # noqa: PLR2004


def test_get_amortization_schedule_reads_the_rate_of_a_date(weekly_mortgage_rates):
    latest = weekly_mortgage_rates.get_amortization_schedule()
    assert latest["Rate"].iloc[0] == 0.0718  # noqa: PLR2004
    assert latest["Interest"].iloc[0] == pytest.approx(2393.3333, abs=1e-4)

    # A date between two surveys reads the last survey before it.
    earlier = weekly_mortgage_rates.get_amortization_schedule(date="2023-11-01")
    assert earlier["Rate"].iloc[0] == 0.0779  # noqa: PLR2004

    # Before the first survey there is no rate, which comes back empty.
    assert weekly_mortgage_rates.get_amortization_schedule(date="2020-12-31").empty


def test_european_central_bank_rates_take_the_rate_at_the_end_of_each_period(
    monkeypatch,
):
    from financetoolkit.fixedincome import fixedincome_controller

    days = pd.period_range("2026-08-01", "2026-10-31", freq="D")
    deposit = pd.Series(0.025, index=days)
    deposit[deposit.index >= pd.Period("2026-09-17", "D")] = 0.0225
    monkeypatch.setattr(
        fixedincome_controller.ecb_model, "get_deposit_facility", lambda: deposit
    )

    fixedincome = fixedincome_controller.FixedIncome(
        start_date="2026-08-01", end_date="2026-10-31"
    )
    monthly = fixedincome.get_european_central_bank_rates(
        rate="deposit", period="monthly"
    )

    assert list(monthly["Deposit"]) == pytest.approx([0.025, 0.0225, 0.0225])
    assert (
        len(
            fixedincome.get_european_central_bank_rates(
                rate="deposit", period="quarterly"
            )
        )
        == 2
    )

    assert fixedincome.get_european_central_bank_rates(
        rate="deposit", period="hourly"
    ).empty
