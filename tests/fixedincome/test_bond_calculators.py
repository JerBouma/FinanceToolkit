"""Bond and Rate Calculator Tests, on fixed Treasury prices and curves"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from financetoolkit.cache.cache_controller import clear_active_cache
from financetoolkit.economics import treasury_model
from financetoolkit.fixedincome import (
    bond_data_model,
    bond_model,
    fixedincome_controller,
    helpers,
    treasurydirect_model,
    yieldcurve_model,
)

PRICE_DATE = date(2026, 11, 5)

# A FedInvest day: a bill, notes and bonds around the auctioned maturities, a TIPS and an FRN.
PRICES = pd.DataFrame(
    [
        ("912797VW4", "Bill", 0.0, "2026-11-10", 99.957556),
        ("912797WN3", "Bill", 0.0, "2027-01-05", 99.3367),
        ("91282CRU7", "Note", 0.04625, "2028-10-31", 100.3125),
        ("91282CLR0", "Note", 0.04125, "2029-10-31", 98.875),
        ("91282CRT0", "Note", 0.0475, "2031-10-31", 100.6875),
        ("91282CRV5", "Note", 0.04875, "2033-10-31", 101.0625),
        ("91282CPZ8", "Note", 0.04125, "2036-02-15", 94.90625),
        ("91282CRF0", "Note", 0.04625, "2036-08-15", 98.4375),
        ("912810FT0", "Bond", 0.045, "2036-02-15", 98.875),
        ("912810RU4", "Bond", 0.02875, "2046-11-15", 70.03125),
        ("912810UW6", "Bond", 0.05125, "2056-08-15", 98.125),
        ("91282CLY5", "TIPS", 0.01625, "2029-10-15", 101.28125),
        ("91282CMH1", "FRN", 0.00205, "2027-01-31", 100.03125),
    ],
    columns=["CUSIP", "Security Type", "Coupon", "Maturity", "Price"],
).assign(Maturity=lambda prices: pd.to_datetime(prices["Maturity"]))

# A Treasury par curve, as the yield curve of the price date gives it.
PAR_CURVE = pd.Series(
    {
        "1M": 0.039,
        "3M": 0.0395,
        "6M": 0.0405,
        "1Y": 0.042,
        "2Y": 0.0445,
        "3Y": 0.0452,
        "5Y": 0.046,
        "7Y": 0.047,
        "10Y": 0.0484,
        "20Y": 0.0527,
        "30Y": 0.0525,
    }
)

BENCHMARKS = [
    "91282CRU7",
    "91282CLR0",
    "91282CRT0",
    "91282CRV5",
    "91282CRF0",
    "912810RU4",
    "912810UW6",
]


@pytest.fixture
def fixedincome(monkeypatch):
    clear_active_cache()
    monkeypatch.setattr(
        treasurydirect_model,
        "get_treasury_prices",
        lambda price_date: (PRICES.copy(), PRICE_DATE),
    )
    monkeypatch.setattr(
        fixedincome_controller.FixedIncome,
        "_latest_yield_curve",
        lambda self, country, date=None: PAR_CURVE,
    )
    yield fixedincome_controller.FixedIncome(rounding=None)
    clear_active_cache()


def spot_curve(fixedincome):
    """The zero-coupon curve the bond calculators discount with, compounded twice a year."""
    zero_curve = fixedincome._latest_zero_curve("United States")
    return 2 * ((1 + zero_curve) ** 0.5 - 1)


def test_treasury_cusip():
    assert helpers.treasury_cusip("912810FT0") == "912810FT0"
    assert helpers.treasury_cusip("US912810FT08") == "912810FT0"
    assert helpers.treasury_cusip("US037833ET32") is None
    assert helpers.treasury_cusip("037833ET3") is None


def test_maturity_label():
    assert helpers.maturity_label(0.25) == "3M"
    assert helpers.maturity_label(1.5 / 12) == "1.5M"
    assert helpers.maturity_label(2) == "2Y"
    assert helpers.maturity_label(10.0) == "10Y"


def test_benchmark_treasuries_without_arguments(fixedincome):
    statistics = fixedincome.collect_bond_statistics()

    # The note (up to 10 years) or bond maturing closest to 2, 3, 5, 7, 10, 20 and 30 years.
    assert statistics.index.tolist() == BENCHMARKS
    assert set(statistics["Issuer"]) == {"United States Treasury"}
    assert set(statistics["Price Date"]) == {"2026-11-05"}

    bond = statistics.loc["91282CRF0"]
    years = bond_data_model.get_years_to_maturity(PRICE_DATE, date(2036, 8, 15))
    yield_to_maturity, duration = bond_data_model.get_bond_analytics(
        0.04625, years, 98.4375
    )
    assert bond["Years to Maturity"] == pytest.approx(years)
    assert bond["Yield to Maturity"] == pytest.approx(yield_to_maturity)
    assert bond["Modified Duration"] == pytest.approx(duration)
    assert bond["Accrued Interest"] == pytest.approx(
        bond_model.get_accrued_interest(100, 0.04625, years, 2)
    )
    assert bond["Current Yield"] == pytest.approx(4.625 / 98.4375)
    assert bond["Convexity"] == pytest.approx(
        bond_model.get_convexity(100, 0.04625, years, yield_to_maturity, 2)
    )

    curve = spot_curve(fixedincome)
    assert bond["Present Value"] == pytest.approx(
        bond_model.get_bond_price_from_curve(100, 0.04625, years, curve, 2)
    )
    assert bond["Z-Spread"] == pytest.approx(
        bond_model.get_z_spread(100, 0.04625, years, 98.4375, curve, 2)
    )


def test_every_bond_calculator_works_without_arguments(fixedincome):
    for method, column in [
        ("get_present_value", "Present Value"),
        ("get_duration", "Modified Duration"),
        ("get_yield_to_maturity", "Yield to Maturity"),
        ("get_z_spread", "Z-Spread"),
        ("get_taylor_price_change", "Price Change"),
        ("get_key_rate_duration", "10Y"),
    ]:
        result = getattr(fixedincome, method)()
        assert result.index.tolist() == BENCHMARKS, method
        assert result[column].notna().all(), method
        assert result.columns[:6].tolist() == fixedincome_controller.BOND_COLUMNS


def test_maturity_range_selects_every_fixed_coupon_treasury(fixedincome):
    result = fixedincome.get_yield_to_maturity(
        maturity_start="2026-11-01", maturity_end="2030-01-01"
    )

    # The TIPS and the FRN are left out unless asked for by CUSIP.
    assert result.index.tolist() == [
        "912797VW4",
        "912797WN3",
        "91282CRU7",
        "91282CLR0",
    ]
    assert result.loc["912797VW4", "Yield to Maturity"] == pytest.approx(
        bond_data_model.get_bill_analytics(5, 99.957556)[0]
    )


def test_bonds_by_cusip_and_isin(fixedincome):
    result = fixedincome.collect_bond_statistics(
        bonds="US912810FT08, 91282CLY5, 91282CMH1"
    )

    # The FRN is left out; the TIPS has a real yield but no measures on the nominal curve.
    assert result.index.tolist() == ["91282CLY5", "912810FT0"]
    assert np.isnan(result.loc["91282CLY5", "Z-Spread"])
    assert np.isnan(result.loc["91282CLY5", "Present Value"])
    assert result.loc["91282CLY5", "Yield to Maturity"] > 0

    key_rates = fixedincome.get_key_rate_duration(bonds="91282CLY5")
    assert key_rates.loc["91282CLY5", "10Y"] != key_rates.loc["91282CLY5", "10Y"]


def test_duration_types(fixedincome):
    bond = PRICES.set_index("CUSIP").loc["91282CPZ8"]
    years = bond_data_model.get_years_to_maturity(PRICE_DATE, date(2036, 2, 15))
    yield_to_maturity = bond_model.get_yield_to_maturity(
        100, bond["Coupon"], years, bond["Price"], 2
    )
    for duration_type, function in [
        ("macaulay", bond_model.get_macaulays_duration),
        ("effective", bond_model.get_effective_duration),
        ("dollar", bond_model.get_dollar_duration),
    ]:
        result = fixedincome.get_duration(duration_type, bonds="91282CPZ8")
        assert result.iloc[0, -1] == pytest.approx(
            function(100, bond["Coupon"], years, yield_to_maturity, 2), rel=1e-4
        )

    assert fixedincome.get_duration("average").empty


def test_key_rate_durations_add_up_to_the_curve_duration(fixedincome):
    key_rates = fixedincome.get_key_rate_duration(bonds="91282CRF0")

    assert key_rates.columns[6:].tolist() == PAR_CURVE.index.tolist()
    # Summed over every maturity of the curve, the key rate durations are the duration of
    # a parallel shift of the curve.
    curve = spot_curve(fixedincome)
    years = bond_data_model.get_years_to_maturity(PRICE_DATE, date(2036, 8, 15))
    up = bond_model.get_bond_price_from_curve(
        100, 0.04625, years, curve + 0.0001, 2, clean=False
    )
    down = bond_model.get_bond_price_from_curve(
        100, 0.04625, years, curve - 0.0001, 2, clean=False
    )
    base = bond_model.get_bond_price_from_curve(
        100, 0.04625, years, curve, 2, clean=False
    )
    assert key_rates.iloc[0, 6:].sum() == pytest.approx(
        -(up - down) / (2 * base * 0.0001), rel=1e-4
    )

    chosen = fixedincome.get_key_rate_duration(
        key_rate_maturity="5Y, 10", bonds="91282CRF0"
    )
    assert chosen.columns[6:].tolist() == ["5Y", "10Y"]
    assert chosen.loc["91282CRF0", "10Y"] == pytest.approx(
        key_rates.loc["91282CRF0", "10Y"]
    )


def test_taylor_price_change(fixedincome):
    fall = fixedincome.get_taylor_price_change(yield_change=-0.01)
    rise = fixedincome.get_taylor_price_change(yield_change=0.01)

    # Convexity makes a fall in yields raise the price more than a rise lowers it.
    assert (fall["Price Change"] > -rise["Price Change"]).all()


def test_bond_equivalent_yield(fixedincome):
    bills = fixedincome.get_bond_equivalent_yield()

    assert bills.index.tolist() == ["912797VW4", "912797WN3"]
    assert bills.loc["912797WN3", "Days to Maturity"] == 61
    discount_yield = (100 - 99.3367) / 100 * 360 / 61
    assert bills.loc["912797WN3", "Discount Yield"] == pytest.approx(discount_yield)
    assert bills.loc["912797WN3", "Bond-Equivalent Yield"] == pytest.approx(
        bond_model.get_bond_equivalent_yield(discount_yield, 61)
    )

    one = fixedincome.get_bond_equivalent_yield(bonds="US912797VW43")
    assert one.index.tolist() == ["912797VW4"]


def test_corporate_bonds_by_issuer(fixedincome, monkeypatch):
    universe = pd.DataFrame(
        {
            "Issuer": ["APPLE INC"],
            "CUSIP": ["037833ET3"],
            "Coupon": [0.04],
            "Maturity": ["2033-05-10"],
            "Rule 144A": [False],
            "Price": [98.0],
            "Yield to Maturity": [0.0436],
            "Modified Duration": [5.4],
            "G-Spread": [0.0004],
            "Price Date": ["2026-11-05"],
            "Funds": ["LQD"],
        },
        index=pd.Index(["US037833ET32"], name="ISIN"),
    )
    requested = {}

    def get_bond_universe(self, **kwargs):
        requested.update(kwargs)
        return universe

    monkeypatch.setattr(
        fixedincome_controller.FixedIncome, "get_bond_universe", get_bond_universe
    )

    result = fixedincome.get_z_spread(issuer="Apple")

    assert requested["issuer"] == "Apple"
    assert result.index.tolist() == ["US037833ET32"]
    assert result.loc["US037833ET32", "Security Type"] == "Corporate"
    years = bond_data_model.get_years_to_maturity(PRICE_DATE, date(2033, 5, 10))
    assert result.loc["US037833ET32", "Z-Spread"] == pytest.approx(
        bond_model.get_z_spread(100, 0.04, years, 98.0, spot_curve(fixedincome), 2)
    )

    # Treasuries and corporate bonds together, by identifier.
    mixed = fixedincome.get_yield_to_maturity(bonds="US037833ET32,912810FT0")
    assert requested["bonds"] == ["US037833ET32"]
    assert mixed.index.tolist() == ["US037833ET32", "912810FT0"]


def test_forward_rate(fixedincome):
    forward_rates = fixedincome.get_forward_rate(
        near_maturity=[1, 2], far_maturity="5Y, 10"
    )
    zero_curve = fixedincome._latest_zero_curve("United States")

    assert forward_rates.columns.tolist() == [5, 10]
    rate = yieldcurve_model.get_forward_rate(zero_curve[2], zero_curve[5], 2, 5)
    assert forward_rates.loc[2, 5] == pytest.approx(rate)

    # Without arguments: 1 to 5 years against 5 to 10 years.
    default = fixedincome.get_forward_rate()
    assert default.shape == (5, 6)
    assert np.isnan(default.loc[5, 5])


def test_par_yield_returns_the_par_curve(fixedincome):
    par_yields = fixedincome.get_par_yield(years_to_maturity=[2, 5, 10])

    # The zero-coupon curve is bootstrapped from the par curve, so at its maturities the
    # par yields come back.
    assert par_yields.tolist() == pytest.approx(
        PAR_CURVE[["2Y", "5Y", "10Y"]].tolist(), abs=1e-5
    )
    assert len(fixedincome.get_par_yield()) == 30


def test_yield_curve_spread(fixedincome):
    spreads = fixedincome.get_yield_curve_spread()

    assert spreads.index.tolist() == ["5Y", "10Y", "30Y"]
    assert spreads.columns.tolist() == ["3M", "2Y"]
    assert spreads.loc["10Y", "2Y"] == pytest.approx(0.0484 - 0.0445)
    assert spreads.loc["30Y", "3M"] == pytest.approx(0.0525 - 0.0395)

    between = fixedincome.get_yield_curve_spread(long_maturity=15, short_maturity="1Y")
    assert between.loc["15Y", "1Y"] == pytest.approx((0.0484 + 0.0527) / 2 - 0.042)


def test_breakeven_inflation_rate(fixedincome, monkeypatch):
    days = pd.period_range("2026-11-04", "2026-11-06", freq="D")
    curves = {
        "nominal": pd.DataFrame(
            {"5 Yr": [0.045, 0.046, 0.047], "10 Yr": [0.048, 0.0484, 0.049]},
            index=days,
        ),
        "real": pd.DataFrame(
            {"5 YR": [0.023, 0.0233, 0.024], "10 YR": [0.026, 0.026, 0.027]},
            index=days,
        ),
    }
    monkeypatch.setattr(
        treasury_model,
        "get_yield_curve",
        lambda kind, start_date, end_date: curves[kind],
    )

    rates = fixedincome.get_breakeven_inflation_rate(date="2026-11-05")

    # The Treasury serves the whole month: the day after the date is not used.
    assert set(rates["Date"]) == {"2026-11-05"}
    assert rates.loc["10 Year", "Breakeven Inflation Rate"] == pytest.approx(0.0224)
    assert rates.loc["5 Year", "Nominal Yield"] == pytest.approx(0.046)
    assert np.isnan(rates.loc["30 Year", "Breakeven Inflation Rate"])
