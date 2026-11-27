"""Offline tests of the Treasury market analytics: floating rate notes, TIPS, forward prices and
the futures basis, on recorded FedInvest prices and TreasuryDirect, BLS, New York Fed and
Yahoo Finance data of November 2026."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from financetoolkit import historical_model
from financetoolkit.cache.cache_controller import clear_active_cache
from financetoolkit.economics.sources import bls_model
from financetoolkit.fixedincome.fixedincome_controller import FixedIncome
from financetoolkit.fixedincome.markets import (
    bond_data_model,
    bond_futures_model,
    frn_model,
)
from financetoolkit.fixedincome.sources import fed_model, treasurydirect_model

PRICE_DATE = date(2026, 11, 19)

# FedInvest prices of 19 November 2026: CUSIP, type, coupon, maturity, price.
PRICES = [
    ("912797TV9", "Bill", 0.0, "2027-02-18", 98.9653),
    ("91282CMJ7", "FRN", 0.04357, "2027-01-31", 100.0283),
    ("91282CRS2", "FRN", 0.04296, "2028-10-31", 100.0229),
    ("91282CRP8", "Note", 0.0475, "2028-09-30", 100.0938),
    ("91282CJA0", "Note", 0.04625, "2028-09-30", 99.875),
    ("91282CQD6", "Note", 0.035, "2031-02-28", 94.9688),
    ("91282CQW4", "Note", 0.0425, "2033-06-30", 96.2188),
    ("91282CHT1", "Note", 0.03875, "2033-08-15", 93.9688),
    ("91282CRJ2", "Note", 0.045, "2033-08-31", 97.5),
    ("91282CJJ1", "Note", 0.045, "2033-11-15", 97.4062),
    ("91282CLF6", "Note", 0.03875, "2034-08-15", 93.0),
    ("91282CRF0", "Note", 0.04625, "2036-08-15", 96.8125),
    ("912810UJ5", "Bond", 0.0475, "2045-02-15", 92.25),
    ("912810UN6", "Bond", 0.04875, "2045-08-15", 93.5312),
    ("912810TL2", "Bond", 0.04, "2052-11-15", 79.9375),
    ("91282CLY5", "TIPS", 0.01625, "2029-10-15", 99.9688),
    ("912810US5", "TIPS", 0.02375, "2056-02-15", 83.4375),
]

# The 13-week bill auctions of July to November 2026 (auction date, high rate, days).
BILL_AUCTIONS = [
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

# The CPI-U, not seasonally adjusted, of 2026 as the BLS publishes it.
CPI = {
    "2025-09": 324.800,
    "2025-11": 324.122,
    "2025-12": 324.054,
    "2026-01": 325.252,
    "2026-02": 326.785,
    "2026-03": 330.213,
    "2026-04": 333.020,
    "2026-05": 335.123,
    "2026-06": 333.952,
    "2026-07": 333.918,
    "2026-08": 334.980,
    "2026-09": 336.940,
    "2026-10": 337.511,
}

# The Yahoo Finance settlement prices of the December 2026 contracts.
FUTURES = {
    "ZNZ26.CBT": 105.875,
    "ZBZ26.CBT": 106.28125,
    "ZTZ26.CBT": 101.945312,
    "ZFZ26.CBT": 104.320312,
    "TNZ26.CBT": 107.203125,
    "UBZ26.CBT": 107.75,
}


def treasury_prices(price_date):
    prices = pd.DataFrame(
        PRICES, columns=["CUSIP", "Security Type", "Coupon", "Maturity", "Price"]
    )
    prices["Maturity"] = pd.to_datetime(prices["Maturity"])
    prices["Call Date"] = pd.NaT
    return prices, PRICE_DATE


def auction(cusip, **terms):
    row = {
        "CUSIP": cusip,
        "Security Term": terms.get("term", ""),
        "Original Security Term": terms.get("term", ""),
        "Reopening": terms.get("reopening", False),
        "Auction Date": terms.get("auction"),
        "Issue Date": terms.get("auction"),
        "Dated Date": terms.get("dated"),
        "Maturity Date": None,
        "Days to Maturity": terms.get("days", np.nan),
        "Interest Rate": np.nan,
        "Spread": terms.get("spread", np.nan),
        "High Rate": terms.get("high_rate", np.nan),
        "Reference CPI on Dated Date": terms.get("reference_cpi", np.nan),
        "Cash Management Bill": False,
    }
    return row


def auctioned_securities(security_type, security_term=None):
    if security_type == "FRN":
        rows = [
            auction("91282CMJ7", dated=date(2025, 1, 31), spread=0.00098),
            auction("91282CRS2", dated=date(2026, 10, 31), spread=0.00038),
            # A reopening repeats the terms of the original auction.
            auction(
                "91282CRS2", dated=date(2026, 10, 31), spread=0.00038, reopening=True
            ),
        ]
    elif security_type == "TIPS":
        rows = [
            auction("91282CLY5", dated=date(2024, 10, 15), reference_cpi=313.89197),
            auction("912810US5", dated=date(2026, 2, 15), reference_cpi=324.08800),
        ]
    elif security_type == "Bill":
        rows = [
            auction(
                f"BILL{number}",
                auction=date.fromisoformat(day),
                high_rate=rate,
                days=days,
            )
            for number, (day, rate, days) in enumerate(BILL_AUCTIONS)
        ]
    else:
        original_terms = {
            "2-Year": ["91282CRP8"],
            "3-Year": ["91282CJA0"],
            "5-Year": ["91282CQD6"],
            "10-Year": ["91282CRF0"],
        }
        rows = [auction(cusip) for cusip in original_terms.get(security_term, [])]
    return pd.DataFrame(rows)


def historical_data(tickers, **kwargs):
    dates = pd.period_range("2026-11-12", "2026-11-19", freq="D")
    closes = pd.DataFrame(
        {ticker: FUTURES[ticker] for ticker in tickers}, index=dates, dtype=float
    )
    closes.columns = pd.MultiIndex.from_product([["Close"], closes.columns])
    return closes, []


def government_curves() -> pd.DataFrame:
    labels = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    years = np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    dates = pd.bdate_range("2026-11-02", "2026-11-19").to_period("D")
    shape = 0.039 + 0.016 * (1 - np.exp(-years / 8))
    return pd.DataFrame(np.tile(shape, (len(dates), 1)), index=dates, columns=labels)


@pytest.fixture
def fixedincome(monkeypatch):
    clear_active_cache()
    curves = government_curves()
    monkeypatch.setattr(treasurydirect_model, "get_treasury_prices", treasury_prices)
    monkeypatch.setattr(
        treasurydirect_model, "get_auctioned_securities", auctioned_securities
    )
    monkeypatch.setattr(
        bls_model,
        "get_consumer_price_index",
        lambda: pd.DataFrame(
            {"all": list(CPI.values())},
            index=pd.PeriodIndex(list(CPI), freq="M"),
        ),
    )
    monkeypatch.setattr(
        fed_model,
        "get_tri_party_general_collateral_rate",
        lambda: pd.DataFrame(
            {"Rate": [0.0389, 0.0388, 0.0390]},
            index=pd.PeriodIndex(["2026-11-18", "2026-11-19", "2026-11-20"], freq="D"),
        ),
    )
    monkeypatch.setattr(historical_model, "get_historical_data", historical_data)
    monkeypatch.setattr(
        FixedIncome,
        "_get_country_yield_curves",
        lambda self, start_date: {"United States": lambda: curves},
    )
    yield FixedIncome(fred_api_key="", use_cached_data=False)
    clear_active_cache()


def test_get_treasury_floating_rate_notes(recorder, fixedincome):
    recorder.capture(fixedincome.get_treasury_floating_rate_notes())
    recorder.capture(
        fixedincome.get_treasury_floating_rate_notes(bonds="91282CRS2", rounding=8)
    )


def test_floating_rate_notes_follow_the_treasury_pricing(fixedincome):
    notes = fixedincome.get_treasury_floating_rate_notes(rounding=10)
    note = notes.loc["91282CRS2"]

    # The 16 November auction applies from 17 November and is not locked out on the 19th.
    index_rate = frn_model.get_index_rate(0.04155, 91)
    assert note["Index Rate"] == pytest.approx(index_rate)
    _, payments = bond_data_model.get_coupon_dates(
        PRICE_DATE, date(2028, 10, 31), frequency=4
    )
    days = np.diff([PRICE_DATE, *payments]).astype("timedelta64[D]").astype(int)
    _, clean = frn_model.get_frn_price(
        index_rate,
        0.00038,
        note["Discount Margin"],
        days,
        note["Accrued Interest"],
    )
    assert clean == pytest.approx(100.0229)
    assert note["Yield"] == pytest.approx(index_rate + note["Discount Margin"])
    # The coupons follow the index: an index duration near zero, a spread duration close
    # to the two years to maturity.
    assert abs(note["Index Duration"]) < 0.01
    assert 1.8 < note["Spread Duration"] < 1.95


def test_treasury_securities_fill_the_floating_rate_notes(fixedincome):
    securities = fixedincome.get_treasury_securities(rounding=10)
    notes = fixedincome.get_treasury_floating_rate_notes(rounding=10)

    assert securities.loc["91282CRS2", "Yield to Maturity"] == pytest.approx(
        notes.loc["91282CRS2", "Yield"]
    )
    assert securities.loc["91282CRS2", "Modified Duration"] == pytest.approx(
        notes.loc["91282CRS2", "Index Duration"]
    )


def test_get_treasury_inflation_protected_securities(recorder, fixedincome):
    recorder.capture(fixedincome.get_treasury_inflation_protected_securities())
    recorder.capture(
        fixedincome.get_treasury_inflation_protected_securities(bonds="912810US5")
    )


def test_inflation_protected_securities_index_ratio(fixedincome):
    tips = fixedincome.get_treasury_inflation_protected_securities(rounding=10)

    # Ref CPI of 19 November 2026: August 334.980 plus 18/30 of the step to September.
    assert tips["Reference CPI"].iloc[0] == pytest.approx(336.156, abs=1e-9)
    assert tips.loc["912810US5", "Index Ratio"] == pytest.approx(1.03724)
    assert tips.loc["912810US5", "Adjusted Price"] == pytest.approx(83.4375 * 1.03724)
    assert tips.loc["912810US5", "Breakeven Inflation Rate"] == pytest.approx(
        tips.loc["912810US5", "Nominal Yield"] - tips.loc["912810US5", "Real Yield"]
    )


def test_inflation_protected_securities_need_the_cpi(fixedincome, monkeypatch):
    monkeypatch.setattr(
        bls_model,
        "get_consumer_price_index",
        lambda: pd.DataFrame(
            {"all": [337.511]}, index=pd.PeriodIndex(["2026-10"], freq="M")
        ),
    )

    # Without a FRED API key the CPI of August 2026 is missing.
    assert fixedincome.get_treasury_inflation_protected_securities().empty


def test_get_treasury_forward_prices(recorder, fixedincome):
    recorder.capture(fixedincome.get_treasury_forward_prices())
    recorder.capture(
        fixedincome.get_treasury_forward_prices(
            forward_date="2027-02-19", repo_rate=0.035, security_type="Note,Bond"
        )
    )


def test_forward_prices_use_the_repo_rate_of_the_price_date(fixedincome):
    forwards = fixedincome.get_treasury_forward_prices(
        forward_date="2026-12-31", security_type="Bill", rounding=10
    )
    bill = forwards.loc["912797TV9"]

    # The repo rate of 19 November, not of the 20th.
    assert bill["Repo Rate"] == pytest.approx(0.0388)
    assert bill["Forward Price"] == pytest.approx(98.9653 * (1 + 0.0388 * 42 / 360))
    assert bill["Carry"] == pytest.approx(bill["Price"] - bill["Forward Price"])

    assert fixedincome.get_treasury_forward_prices(forward_date="2026-11-01").empty
    assert fixedincome.get_treasury_forward_prices(security_type="TIPS").empty


def test_get_treasury_futures_basis(recorder, fixedincome):
    recorder.capture(fixedincome.get_treasury_futures_basis())
    recorder.capture(fixedincome.get_treasury_futures_basis(contract="ZT"))
    recorder.capture(
        fixedincome.get_treasury_futures_basis(contract="ZB", repo_rate=0.04)
    )


def test_futures_basket_follows_the_contract_rules(fixedincome):
    notes = fixedincome.get_treasury_futures_basis(rounding=10)

    # 6.5 to 8 years from 1 December 2026; the 2036 note is beyond, the 2028 and 2031
    # notes before.
    assert sorted(notes.index) == [
        "91282CHT1",
        "91282CJJ1",
        "91282CLF6",
        "91282CQW4",
        "91282CRJ2",
    ]
    assert notes["Futures Symbol"].iloc[0] == "ZNZ26.CBT"
    assert notes["Delivery Date"].iloc[0] == "2026-12-31"
    assert notes["Cheapest to Deliver"].tolist() == [True, False, False, False, False]
    assert notes["Implied Repo Rate"].is_monotonic_decreasing

    ctd = notes.iloc[0]
    remaining = relativedelta(pd.Timestamp(ctd["Maturity"]).date(), date(2026, 12, 1))
    assert ctd["Conversion Factor"] == bond_futures_model.get_conversion_factor(
        ctd["Coupon"], remaining.years, remaining.months
    )

    # The 2- and 5-year notes must have been auctioned with a term of up to 5 years 3
    # months, and deliver on the third business day of the next month.
    two_year = fixedincome.get_treasury_futures_basis(contract="ZT")
    assert sorted(two_year.index) == ["91282CJA0", "91282CRP8"]
    assert two_year["Delivery Date"].iloc[0] == "2027-01-05"
    assert fixedincome.get_treasury_futures_basis(contract="ZF").index.tolist() == [
        "91282CQD6"
    ]
    assert fixedincome.get_treasury_futures_basis(contract="TN").index.tolist() == [
        "91282CRF0"
    ]
    # Bonds with 15 to 25 years go into ZB, from 25 years into UB.
    assert sorted(fixedincome.get_treasury_futures_basis(contract="ZB").index) == [
        "912810UJ5",
        "912810UN6",
    ]
    assert fixedincome.get_treasury_futures_basis(contract="UB").index.tolist() == [
        "912810TL2"
    ]


def test_futures_basis_invalid(fixedincome):
    assert fixedincome.get_treasury_futures_basis(contract="ZQ").empty
    assert fixedincome.get_treasury_futures_basis(contract_month="2026-11").empty
    # The September contract has delivered by 19 November.
    assert fixedincome.get_treasury_futures_basis(contract_month="2026-09").empty


def test_floating_rate_note_index_is_not_locked_out_on_a_price_date(
    fixedincome, monkeypatch
):
    # A Tuesday price date, the day after the 16 November auction. The lockout applies only to
    # the issue date and interest payment dates (31 CFR 356, Appendix B, IV), so that auction
    # sets the index rate rather than the one of 9 November.
    monkeypatch.setattr(
        treasurydirect_model,
        "get_treasury_prices",
        lambda price_date: (treasury_prices(price_date)[0], date(2026, 11, 17)),
    )
    notes = fixedincome.get_treasury_floating_rate_notes(rounding=10)

    assert notes.loc["91282CRS2", "Index Rate"] == pytest.approx(
        frn_model.get_index_rate(0.04155, 91)
    )


def test_forward_prices_without_a_repo_rate_say_why(fixedincome, monkeypatch, caplog):
    # The New York Fed could not be reached, so no TGCR came back.
    monkeypatch.setattr(
        fed_model, "get_tri_party_general_collateral_rate", pd.DataFrame
    )

    with caplog.at_level("WARNING"):
        assert fixedincome.get_treasury_forward_prices(
            forward_date="2026-12-31", security_type="Bill"
        ).empty
    assert "pass repo_rate" in caplog.text
    # A repo rate of your own still prices the forwards.
    assert not fixedincome.get_treasury_forward_prices(
        forward_date="2026-12-31", security_type="Bill", repo_rate=0.039
    ).empty


def test_unreachable_auctions_leave_the_notes_empty_without_a_second_message(
    fixedincome, monkeypatch, caplog
):
    # TreasuryDirect unreachable: the source reports it and answers with no auctions.
    monkeypatch.setattr(
        treasurydirect_model,
        "get_auctioned_securities",
        lambda *args, **kwargs: pd.DataFrame(),  # noqa: ARG005
    )

    assert fixedincome.get_treasury_floating_rate_notes().empty
    assert fixedincome.get_treasury_inflation_protected_securities().empty
    assert "is missing" not in caplog.text
    assert "could not be calculated" not in caplog.text
