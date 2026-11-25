"""Treasury market model tests: floating rate notes, TIPS, bond forwards and futures.

The reference values are the Treasury's own (31 CFR 356, Appendix B, its auction results and
TreasuryDirect's security data) and those of QuantLib 1.44, FinancePy 1.1.2 and rateslib
3.0.0 on the same inputs, computed outside the test suite and kept as constants.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from financetoolkit.fixedincome.markets import (
    bond_data_model,
    bond_futures_model,
    frn_model,
    inflation_bond_model,
)

# The 13-week bill auctions of July to November 2026: auction date, high rate and the days
# from issue to maturity, as TreasuryDirect publishes them.
BILL_AUCTIONS = [
    ("2026-07-13", 0.03760, 91),
    ("2026-07-20", 0.03730, 91),
    ("2026-07-27", 0.03815, 91),
    ("2026-08-03", 0.03750, 91),
    ("2026-08-10", 0.03735, 91),
    ("2026-08-17", 0.03715, 91),
    ("2026-08-24", 0.03715, 92),
    ("2026-08-31", 0.03770, 91),
    ("2026-09-08", 0.03800, 91),
    ("2026-09-14", 0.03970, 91),
    ("2026-09-21", 0.04015, 91),
    ("2026-09-28", 0.04110, 91),
    ("2026-10-05", 0.04050, 91),
    ("2026-10-13", 0.04085, 91),
    ("2026-10-19", 0.04095, 91),
    ("2026-10-26", 0.04050, 91),
    ("2026-11-02", 0.04040, 91),
    ("2026-11-09", 0.04060, 91),
    ("2026-11-16", 0.04155, 91),
]


def index_rates() -> pd.Series:
    return pd.Series(
        [frn_model.get_index_rate(rate, days) for _, rate, days in BILL_AUCTIONS],
        index=[date.fromisoformat(day) for day, _, _ in BILL_AUCTIONS],
    )


def test_index_rate_follows_the_treasury_example():
    # 31 CFR 356, Appendix B, IV.A: a high rate of 0.095% on a 91-day bill.
    assert frn_model.get_index_rate(0.00095, 91) == pytest.approx(
        0.00095022819, abs=1e-13
    )


def test_daily_index_rates_apply_from_the_day_after_the_auction_and_lock_out():
    rates = frn_model.get_daily_index_rates(
        index_rates(), date(2026, 9, 14), date(2026, 9, 18), []
    )
    # Monday's auction applies from Tuesday.
    assert rates[date(2026, 9, 14)] == pytest.approx(
        frn_model.get_index_rate(0.038, 91)
    )
    assert rates[date(2026, 9, 15)] == pytest.approx(
        frn_model.get_index_rate(0.0397, 91)
    )

    # A settlement on Wednesday 16 September locks out the auction of Monday 14 September
    # up to and including the settlement date.
    locked = frn_model.get_daily_index_rates(
        index_rates(), date(2026, 9, 14), date(2026, 9, 18), [date(2026, 9, 16)]
    )
    assert locked[date(2026, 9, 15)] == pytest.approx(
        frn_model.get_index_rate(0.038, 91)
    )
    assert locked[date(2026, 9, 16)] == pytest.approx(
        frn_model.get_index_rate(0.038, 91)
    )
    assert locked[date(2026, 9, 17)] == pytest.approx(
        frn_model.get_index_rate(0.0397, 91)
    )

    with pytest.raises(ValueError, match="No 13-week bill auction"):
        frn_model.get_daily_index_rates(
            index_rates(), date(2026, 7, 1), date(2026, 7, 2)
        )


def test_frn_reopening_reproduces_the_treasury_auction():
    # 91282CRD5 (spread 0.050%, dated 31 July 2026) reopened on 25 September 2026 at a
    # discount margin of 0.040%: the Treasury announced accrued interest of 0.6036702480
    # and a price of 100.015499.
    settlement, dated = date(2026, 9, 25), date(2026, 7, 31)
    accrued = frn_model.get_floating_rate_note_accrued_interest(
        frn_model.get_daily_index_rates(index_rates(), dated, settlement), 0.0005
    )
    assert accrued == pytest.approx(0.6036702480, abs=1e-9)

    index_rate = frn_model.get_daily_index_rates(
        index_rates(), settlement, date(2026, 9, 26), [settlement]
    ).iloc[0]
    assert index_rate == pytest.approx(frn_model.get_index_rate(0.04015, 91))

    _, payments = bond_data_model.get_coupon_dates(
        settlement, date(2028, 7, 31), frequency=4
    )
    days = np.diff([settlement, *payments]).astype("timedelta64[D]").astype(int)
    assert days.tolist() == [36, 92, 89, 92, 92, 92, 90, 92]

    _, clean = frn_model.get_frn_price(index_rate, 0.0005, 0.0004, days, accrued)
    assert round(clean, 6) == pytest.approx(100.015499)
    assert frn_model.get_discount_margin(
        100.015499, index_rate, 0.0005, days, accrued
    ) == pytest.approx(0.0004, abs=1e-7)


def test_frn_price_and_durations_against_financepy():
    # FinancePy's BondFRN (31 July 2026 to 31 October 2028, quarterly, ACT/360) settling on
    # 20 November 2026 at a flat index of 4.20%, a spread of 0.038% and a margin of 0.010%.
    settlement = date(2026, 11, 20)
    _, payments = bond_data_model.get_coupon_dates(
        settlement, date(2028, 10, 31), frequency=4
    )
    days = np.diff([settlement, *payments]).astype("timedelta64[D]").astype(int)
    accrued = 100 * (0.042 + 0.00038) * 20 / 360

    dirty, clean = frn_model.get_frn_price(0.042, 0.00038, 0.0001, days, accrued)
    assert dirty == pytest.approx(100.28627788, abs=1e-8)
    assert accrued == pytest.approx(0.23544444, abs=1e-8)
    assert frn_model.get_discount_margin(
        clean, 0.042, 0.00038, days, accrued
    ) == pytest.approx(0.0001, abs=1e-10)

    index_duration, spread_duration = frn_model.get_frn_durations(
        0.042, 0.00038, 0.0001, days, accrued
    )
    # FinancePy's modified credit duration is 1.881137; it takes a one-sided change.
    assert spread_duration == pytest.approx(1.881137, abs=5e-4)
    # The coupons follow the index, so the index duration is close to zero.
    assert 0 < index_duration < 0.01


def test_discount_margin_out_of_range_is_nan():
    assert np.isnan(frn_model.get_discount_margin(10.0, 0.04, 0.001, [90, 92]))
    assert np.isnan(frn_model.get_frn_durations(0.04, 0.001, np.nan, [90, 92])[0])


def test_reference_cpi_and_index_ratio_follow_the_treasury_example():
    # 31 CFR 356, Appendix B, I.B.3: CPI of January (154.40) and February 1996 (154.90).
    cpi = pd.Series(
        {pd.Period("1996-01", "M"): 154.40, pd.Period("1996-02", "M"): 154.90}
    )
    april_15 = inflation_bond_model.get_reference_cpi(cpi, date(1996, 4, 15))
    april_16 = inflation_bond_model.get_reference_cpi(cpi, date(1996, 4, 16))

    # QuantLib's CPI.laggedFixing (linear, three months) gives 154.6333333.
    assert april_15 == pytest.approx(154.63333, abs=1e-9)
    assert april_16 == pytest.approx(154.65000, abs=1e-9)
    assert inflation_bond_model.get_index_ratio(april_16, april_15) == 1.00011
    assert np.isnan(inflation_bond_model.get_reference_cpi(cpi, date(1996, 6, 1)))


def test_reference_cpi_matches_treasurydirect():
    # The BLS's CPI-U, not seasonally adjusted, of August and September 2026; TreasuryDirect
    # gives 91282CRE3 a reference CPI of 336.87467 and an index ratio of 1.00870 on its
    # reopening of 30 November 2026 (dated date reference CPI 333.96974).
    cpi = pd.Series(
        {pd.Period("2026-08", "M"): 334.980, pd.Period("2026-09", "M"): 336.940}
    )
    reference = inflation_bond_model.get_reference_cpi(cpi, date(2026, 11, 30))

    assert reference == pytest.approx(336.87467, abs=1e-9)
    assert inflation_bond_model.get_index_ratio(reference, 333.96974) == 1.0087


def test_unreported_cpi_takes_the_index_number_the_treasury_announced():
    # The BLS did not publish October 2025; the Treasury used September 2025 times its
    # twelve-month change to the twelfth root, 325.604, which TreasuryDirect's reference
    # CPI of 30 January 2026 (324.21761) and of 31 December 2025 (325.57806) reflect.
    months = pd.period_range("2024-09", "2025-11", freq="M")
    cpi = pd.Series(np.nan, index=months)
    cpi[pd.Period("2024-09", "M")] = 315.301
    cpi[pd.Period("2025-09", "M")] = 324.800
    cpi[pd.Period("2025-11", "M")] = 324.122
    filled = inflation_bond_model.fill_unreported_months(cpi)

    assert filled[pd.Period("2025-10", "M")] == pytest.approx(325.604, abs=5e-4)
    assert inflation_bond_model.get_reference_cpi(
        filled, date(2026, 1, 30)
    ) == pytest.approx(324.21761, abs=1e-5)
    assert inflation_bond_model.get_reference_cpi(
        filled, date(2025, 12, 31)
    ) == pytest.approx(325.57806, abs=1e-5)


def test_inflation_adjusted_price():
    adjusted = inflation_bond_model.get_inflation_adjusted_price(95.5, 0.4, 1.0087)

    assert adjusted == pytest.approx((96.33085, 0.40348, 96.73433))


# A 4.625% note maturing on 15 August 2036 at a clean price of 98.4375 on 20 November
# 2026: QuantLib's BondForward clean forward price on a flat simple ACT/360 repo curve.
FORWARD_CASES = [
    (date(2027, 2, 20), 0.039, 98.27221156, 2e-5),
    (date(2026, 12, 30), 0.040, 98.37770079, 1e-8),
    (date(2027, 3, 31), 0.0375, 98.13132842, 1e-4),
]


@pytest.mark.parametrize(
    ("forward_date", "repo", "quantlib", "tolerance"), FORWARD_CASES
)
def test_bond_forward_price_against_quantlib(forward_date, repo, quantlib, tolerance):
    settlement, maturity, coupon = date(2026, 11, 20), date(2036, 8, 15), 0.04625
    accrued = bond_data_model.get_treasury_accrued_interest(
        coupon, settlement, maturity
    )
    forward_accrued = bond_data_model.get_treasury_accrued_interest(
        coupon, forward_date, maturity
    )
    _, coupon_dates = bond_data_model.get_coupon_dates(settlement, maturity)
    coupons = [
        (100 * coupon / 2, (paid - settlement).days)
        for paid in coupon_dates
        if paid <= forward_date
    ]
    days = (forward_date - settlement).days

    forward = bond_futures_model.get_bond_forward_price(
        98.4375, accrued, forward_accrued, repo, days, coupons
    )
    # Equal without a coupon in between; with one, QuantLib discounts the coupon on the
    # repo curve where the formula reinvests it at simple interest.
    assert forward == pytest.approx(quantlib, abs=tolerance)
    assert bond_futures_model.get_implied_repo_rate(
        98.4375, accrued, forward, forward_accrued, days, coupons
    ) == pytest.approx(repo, abs=1e-12)


def test_conversion_factor_follows_hull():
    # Hull (2022), ch. 6.2: 10% with 20 years and 2 months, 8% with 18 years and 4 months.
    assert bond_futures_model.get_conversion_factor(0.10, 20, 2) == 1.4623
    assert bond_futures_model.get_conversion_factor(0.08, 18, 4) == 1.2199

    with pytest.raises(ValueError, match="rounding"):
        bond_futures_model.get_conversion_factor(0.04, 10, 0, "year")


# Deliverables for a delivery month starting 1 December 2026: coupon, maturity and the
# conversion factor of rateslib's BondFuture (CMELong for quarters, CMEShort for months).
# FinancePy's BondFuture gives the same quarterly factors except 0.9611 for 2028-10-31.
CONVERSION_FACTORS = [
    (0.04625, date(2034, 11, 15), 0.9157, 0.9143),
    (0.04125, date(2035, 2, 15), 0.8822, 0.8803),
    (0.04250, date(2034, 5, 15), 0.8983, 0.8964),
    (0.03750, date(2033, 8, 15), 0.8804, 0.8778),
    (0.03875, date(2033, 6, 30), 0.8870, 0.8870),
    (0.04500, date(2046, 2, 15), 0.8313, 0.8304),
    (0.04750, date(2052, 11, 15), 0.8370, 0.8366),
    (0.04000, date(2031, 11, 30), 0.9183, 0.9159),
    (0.03625, date(2028, 10, 31), 0.9610, 0.9593),
    (0.02500, date(2050, 2, 15), 0.5664, 0.5649),
]


@pytest.mark.parametrize(("coupon", "maturity", "quarter", "month"), CONVERSION_FACTORS)
def test_conversion_factor_against_rateslib(coupon, maturity, quarter, month):
    years = maturity.year - 2026 - (maturity.month < 12)
    months = (maturity.month - 12) % 12

    assert bond_futures_model.get_conversion_factor(coupon, years, months) == quarter
    assert (
        bond_futures_model.get_conversion_factor(coupon, years, months, "month")
        == month
    )


def test_futures_basis_is_consistent():
    coupons = [(2.25, 30)]
    basis = bond_futures_model.get_futures_basis(
        97.5, 1.0, 106.0, 0.9202, 1.5, 0.039, 41, coupons
    )

    assert basis["Invoice Price"] == pytest.approx(106.0 * 0.9202 + 1.5)
    assert basis["Gross Basis"] == pytest.approx(97.5 - 106.0 * 0.9202)
    assert basis["Net Basis"] == pytest.approx(basis["Gross Basis"] - basis["Carry"])

    # Financed at the implied repo rate, the net basis is zero.
    at_implied = bond_futures_model.get_futures_basis(
        97.5, 1.0, 106.0, 0.9202, 1.5, basis["Implied Repo Rate"], 41, coupons
    )
    assert at_implied["Net Basis"] == pytest.approx(0, abs=1e-12)


def test_coupon_dates_follow_the_end_of_month_rule():
    previous, coupons = bond_data_model.get_coupon_dates(
        date(2026, 11, 20), date(2028, 4, 30), frequency=4
    )

    assert previous == date(2026, 10, 31)
    assert coupons[:3] == [date(2027, 1, 31), date(2027, 4, 30), date(2027, 7, 31)]
    assert coupons[-1] == date(2028, 4, 30)

    # On a coupon date the coupon belongs to the seller.
    previous, coupons = bond_data_model.get_coupon_dates(
        date(2027, 1, 31), date(2028, 4, 30), frequency=4
    )
    assert previous == date(2027, 1, 31)
    assert coupons[0] == date(2027, 4, 30)
