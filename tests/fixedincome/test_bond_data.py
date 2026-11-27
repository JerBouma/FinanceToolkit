"""Bond Universe and Treasury Securities Tests"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from financetoolkit import fmp_model as toolkit_fmp_model
from financetoolkit.cache.cache_controller import clear_active_cache
from financetoolkit.fixedincome import fixedincome_controller
from financetoolkit.fixedincome.bonds import bond_model
from financetoolkit.fixedincome.markets import bond_data_model
from financetoolkit.fixedincome.sources import treasurydirect_model
from financetoolkit.utilities.error_model import InvalidArgumentError


@pytest.fixture(autouse=True)
def isolate_from_the_active_cache():
    """Start every test without an active cache, so the fake responses are used."""
    clear_active_cache()
    yield
    clear_active_cache()


# The names as bond funds print them, with what they should read as.
@pytest.mark.parametrize(
    ("name", "issuer", "coupon", "maturity", "rule_144a"),
    [
        (
            "ANHEUSER-BUSCH COMPANIES LLC 4.90% 02/01/2046",
            "ANHEUSER-BUSCH COMPANIES LLC",
            0.049,
            date(2046, 2, 1),
            False,
        ),
        (
            "SPACE EXPLORATION TECHNOLOGIES COR 144A 5.35% 07/15/2031",
            "SPACE EXPLORATION TECHNOLOGIES COR",
            0.0535,
            date(2031, 7, 15),
            True,
        ),
        (
            "TREASURY NOTE 4.63% 08/15/2036 (BBG02444V9S2)",
            "TREASURY NOTE",
            0.0463,
            date(2036, 8, 15),
            False,
        ),
        (
            "JPMORGAN CHASE & CO 0% 1/5/2030",
            "JPMORGAN CHASE & CO",
            0.0,
            date(2030, 1, 5),
            False,
        ),
    ],
)
def test_parse_bond_name(name, issuer, coupon, maturity, rule_144a):
    parsed = bond_data_model.parse_bond_name(name)

    assert parsed["Issuer"] == issuer
    assert parsed["Coupon"] == pytest.approx(coupon)
    assert parsed["Maturity"] == maturity
    assert parsed["Rule 144A"] is rule_144a


@pytest.mark.parametrize(
    "name",
    [
        "BLK CSH FND TREASURY SL AGENCY",
        "USD CASH",
        "SSC GOVERNMENT MM GVMXX",
        "BOND 5% 13/45/2030",
    ],
)
def test_parse_bond_name_rejects_other_holdings(name):
    assert bond_data_model.parse_bond_name(name) is None


@pytest.mark.parametrize(
    ("name", "maturity", "reason"),
    [
        (
            "GOLDMAN SACHS GROUP INC (FXD-FRN) 4.15% 01/21/2029 (BBG01ZNGFRN9)",
            None,
            "floating rate",
        ),
        ("BANK OF AMERICA CORP MTN FRN 5.00% 01/01/2030", None, "floating rate"),
        ("TREASURY (CPI) NOTE 1.88% 07/15/2035", None, "inflation-linked"),
        (
            "FHLMC 30YR UMBS SUPER 5.50% 06/01/2053 (BBG01FRNH9H9)",
            None,
            "mortgage pool",
        ),
        ("Fannie Mae Pool 4.50% 01/01/2050", None, "mortgage pool"),
        ("BARCLAYS PLC 8.00% 12/31/2999", date(2999, 12, 31), "perpetual"),
        ("NORFOLK SOUTHERN CORP 4.10% 05/15/2121", date(2121, 5, 15), None),
        ("APPLE INC 4.00% 05/10/2033", date(2033, 5, 10), None),
    ],
)
def test_exclusion_reason(name, maturity, reason):
    assert bond_data_model.exclusion_reason(name, maturity) == reason


def test_get_years_to_maturity_counts_actual_coupon_periods():
    # On a coupon date the years are whole half-years.
    assert (
        bond_data_model.get_years_to_maturity(date(2026, 2, 15), date(2036, 2, 15))
        == 10
    )
    # Halfway (92 of 184 days) through the period from 15 August to 15 February.
    assert bond_data_model.get_years_to_maturity(
        date(2026, 11, 15), date(2036, 2, 15)
    ) == pytest.approx(9.25)
    # A bond maturing at the end of August pays at the end of February.
    assert (
        bond_data_model.get_years_to_maturity(date(2027, 2, 28), date(2030, 8, 31))
        == 3.5
    )
    with pytest.raises(ValueError, match="on or after the maturity date"):
        bond_data_model.get_years_to_maturity(date(2030, 1, 1), date(2030, 1, 1))


@pytest.mark.parametrize(
    ("maturity", "coupon", "price", "accrued", "yield_to_maturity", "duration"),
    [
        # Notes maturing on a month's last day pay on the last day of every coupon month
        # (30 April and 31 October); QuantLib's FixedRateBond on ActualActual(ISMA) with an
        # end-of-month schedule, settled on 5 November 2026.
        (date(2028, 4, 30), 0.0375, 99.5, 0.05180, 0.0409986, 1.42937),
        (date(2028, 2, 29), 0.04, 100.1, 0.72928, 0.0391786, 1.26372),
        (date(2028, 6, 30), 0.04, 100.1, 1.39130, 0.0393429, 1.56340),
        (date(2031, 10, 31), 0.04125, 100.6875, 0.05698, 0.0397154, 4.46823),
        (date(2036, 8, 15), 0.04625, 98.4375, 1.03057, 0.0482648, 7.69144),
    ],
)
def test_get_years_to_maturity_end_of_month_against_quantlib(
    maturity, coupon, price, accrued, yield_to_maturity, duration
):
    years = bond_data_model.get_years_to_maturity(date(2026, 11, 5), maturity)

    assert bond_model.get_accrued_interest(100, coupon, years, 2) == pytest.approx(
        accrued, abs=1e-5
    )
    assert bond_data_model.get_bond_analytics(coupon, years, price) == pytest.approx(
        (yield_to_maturity, duration), abs=1e-5
    )


@pytest.mark.parametrize(
    ("settlement", "maturity", "coupon", "price", "accrued", "yield_to_maturity"),
    [
        # QuantLib's FixedRateBond on Thirty360(BondBasis), end-of-month where the maturity
        # is a month's last day.
        (date(2026, 11, 5), date(2033, 5, 10), 0.04, 98.0, 1.944444, 0.04355857),
        (date(2026, 11, 5), date(2028, 4, 30), 0.0375, 99.5, 0.052083, 0.04099879),
        (date(2026, 11, 5), date(2046, 2, 1), 0.049, 92.5, 1.279444, 0.05537891),
        (date(2026, 11, 5), date(2036, 8, 15), 0.04625, 98.4375, 1.027778, 0.04826467),
        (date(2026, 11, 30), date(2030, 5, 31), 0.045, 100.4, 0.0, 0.04375497),
        # 31 August to 28 February counts 178 days on 30/360, so QuantLib pays a coupon of
        # 178/360 of the rate; the accrued interest is the same, 65 days.
        (date(2026, 11, 5), date(2031, 2, 28), 0.05, 101.2, 0.902778, None),
    ],
)
def test_get_years_to_maturity_30_360_against_quantlib(
    settlement, maturity, coupon, price, accrued, yield_to_maturity
):
    years = bond_data_model.get_years_to_maturity(
        settlement, maturity, day_count="30/360"
    )

    assert bond_model.get_accrued_interest(100, coupon, years, 2) == pytest.approx(
        accrued, abs=1e-6
    )
    if yield_to_maturity is not None:
        assert bond_model.get_yield_to_maturity(
            100, coupon, years, price, 2, guess=0.04, tolerance=1e-12
        ) == pytest.approx(yield_to_maturity, abs=1e-8)


def test_get_years_to_maturity_rejects_unknown_day_count():
    with pytest.raises(InvalidArgumentError, match="day_count must be one of"):
        bond_data_model.get_years_to_maturity(
            date(2026, 11, 5), date(2030, 5, 31), day_count="act/360"
        )


def test_get_bond_analytics():
    # A bond priced at par on a coupon date yields its coupon.
    yield_to_maturity, duration = bond_data_model.get_bond_analytics(0.05, 10, 100)
    assert yield_to_maturity == pytest.approx(0.05, abs=1e-8)
    assert duration == pytest.approx(
        bond_model.get_modified_duration(100, 0.05, 10, 0.05, 2)
    )

    # Between coupon dates the clean price plus accrued interest is discounted.
    yield_to_maturity, _ = bond_data_model.get_bond_analytics(0.04, 7.25, 95)
    assert bond_model.get_bond_price(
        100, 0.04, 7.25, yield_to_maturity, 2
    ) == pytest.approx(95, abs=1e-6)

    assert all(
        np.isnan(value)
        for value in bond_data_model.get_bond_analytics(0.05, np.nan, 100)
    )


def test_get_bill_analytics():
    # A 182-day bill at 98: a discount yield of 3.956% and its bond-equivalent yield.
    yield_to_maturity, duration = bond_data_model.get_bill_analytics(182, 98)
    discount_yield = 0.02 * 360 / 182
    assert yield_to_maturity == pytest.approx(
        bond_model.get_bond_equivalent_yield(discount_yield, 182)
    )
    assert 0 < duration < 182 / 365

    assert all(np.isnan(value) for value in bond_data_model.get_bill_analytics(0, 99))


@pytest.mark.parametrize(
    ("days", "bond_equivalent_yield", "price", "duration", "dv01", "convexity"),
    [
        # QuantLib's ZeroCouponBond at a simple Actual365Fixed rate: the price and the
        # modified duration, DV01 and convexity from its prices around the yield.
        (28, 0.039, 99.7017143232, 0.0764835068, 0.0007625537, 0.011699),
        (91, 0.04, 99.0125868056, 0.2468532985, 0.0024441584, 0.121873),
        (182, 0.0405, 98.0205225434, 0.4887598658, 0.0047908498, 0.477772),
    ],
)
def test_get_bill_measures_against_quantlib(
    days, bond_equivalent_yield, price, duration, dv01, convexity
):
    years = days / 365
    measures = bond_data_model.get_bill_measures(years, bond_equivalent_yield)

    assert bond_data_model.get_bill_price(
        years, bond_equivalent_yield
    ) == pytest.approx(price, abs=1e-9)
    assert measures["Macaulay's Duration"] == pytest.approx(years)
    assert measures["Modified Duration"] == pytest.approx(duration, abs=1e-9)
    assert measures["DV01"] == pytest.approx(dv01, abs=1e-9)
    assert measures["Convexity"] == pytest.approx(convexity, abs=1e-5)


def test_get_bill_measures_beyond_half_a_year():
    # Beyond half a year the price compounds once, (1 + y / 2) * (1 + y (t - 0.5)), the
    # pricing get_bond_equivalent_yield inverts, and the duration is its log derivative.
    years, bond_equivalent_yield = 300 / 365, 0.042
    discount_yield = 0.04
    price = 100 * (1 - discount_yield * 300 / 360)
    assert bond_data_model.get_bill_price(
        years, bond_model.get_bond_equivalent_yield(discount_yield, 300)
    ) == pytest.approx(price)

    measures = bond_data_model.get_bill_measures(years, bond_equivalent_yield)
    step = 1e-7
    numerical = (
        bond_data_model.get_bill_price(years, bond_equivalent_yield - step)
        - bond_data_model.get_bill_price(years, bond_equivalent_yield + step)
    ) / (2 * step * bond_data_model.get_bill_price(years, bond_equivalent_yield))
    assert measures["Modified Duration"] == pytest.approx(numerical, abs=1e-8)
    assert measures["Modified Duration"] == pytest.approx(0.807339331, abs=1e-9)

    # get_bill_analytics takes its duration from the same pricing.
    yield_to_maturity, duration = bond_data_model.get_bill_analytics(91, 98.99)
    assert duration == pytest.approx((91 / 365) / (1 + yield_to_maturity * 91 / 365))


def test_interpolate_curve():
    curve = pd.Series({"1M": 0.04, "1Y": 0.042, "10Y": 0.045, "30Y": 0.05})
    years = pd.Series([0.01, 1, 5.5, 20, 40], index=list("abcde"))

    interpolated = bond_data_model.interpolate_curve(curve, years)

    assert list(interpolated.index) == list("abcde")
    assert interpolated.tolist() == pytest.approx([0.04, 0.042, 0.0435, 0.0475, 0.05])


PRICE_PAGE = """
<table>
<tr><th>CUSIP</th><th>SECURITY TYPE</th><th>RATE</th><th>MATURITY DATE</th>
<th>CALL DATE</th><th>BUY</th><th>SELL</th><th>END OF DAY</th></tr>
<tr>
    <td>912797VW4</td>
    <td style="white-space: nowrap; text-align: left">MARKET BASED BILL</td>
    <td>0.000%</td><td>11/10/2026</td><td></td>
    <td>0.000000</td><td>99.957556</td><td>0.000000</td>
</tr>
<tr>
    <td>91282CPZ8</td>
    <td style="white-space: nowrap; text-align: left">MARKET BASED NOTE</td>
    <td>4.125%</td><td>02/15/2036</td><td></td>
    <td>0.000000</td><td>94.875000</td><td>94.906250</td>
</tr>
<tr>
    <td>912810FT0</td>
    <td>MARKET BASED BOND</td>
    <td>4.500%</td><td>02/15/2036</td><td></td>
    <td>0.000000</td><td>98.843750</td><td>98.875000</td>
</tr>
<tr>
    <td>91282CLY5</td><td>TIPS</td><td>1.625%</td><td>10/15/2029</td><td></td>
    <td>0.000000</td><td>101.250000</td><td>101.281250</td>
</tr>
<tr>
    <td>91282CMH1</td><td>MARKET BASED FRN</td><td>0.205%</td><td>01/31/2027</td><td></td>
    <td>0.000000</td><td>100.031250</td><td>100.031250</td>
</tr>
</table>
"""


def test_parse_prices():
    prices = treasurydirect_model._parse_prices(PRICE_PAGE)

    assert prices["CUSIP"].tolist() == [
        "912797VW4",
        "91282CPZ8",
        "912810FT0",
        "91282CLY5",
        "91282CMH1",
    ]
    assert prices["Security Type"].tolist() == ["Bill", "Note", "Bond", "TIPS", "FRN"]
    assert prices["Coupon"].tolist() == pytest.approx(
        [0, 0.04125, 0.045, 0.01625, 0.00205]
    )
    assert prices["Maturity"].iloc[1] == pd.Timestamp("2036-02-15")
    # A price the page shows as 0 is not published yet.
    assert np.isnan(prices["End of Day"].iloc[0])
    assert prices["Sell"].iloc[0] == pytest.approx(99.957556)

    assert treasurydirect_model._parse_prices("<html>No prices</html>").empty


def test_get_treasury_prices_skips_days_without_prices(monkeypatch):
    requested = []

    def get_day(day):
        requested.append(day)
        if day == date(2026, 11, 6):  # a holiday without prices
            return treasurydirect_model._parse_prices("")
        return treasurydirect_model._parse_prices(PRICE_PAGE)

    monkeypatch.setattr(treasurydirect_model, "_get_day", get_day)

    # A Sunday: the weekend is skipped without a request, the empty Friday is passed.
    prices, price_date = treasurydirect_model.get_treasury_prices("2026-11-08")

    assert requested == [date(2026, 11, 6), date(2026, 11, 5)]
    assert price_date == date(2026, 11, 5)
    # The end-of-day price, or the bid before it is published.
    assert prices["Price"].tolist() == pytest.approx(
        [99.957556, 94.90625, 98.875, 101.28125, 100.03125]
    )
    assert "End of Day" not in prices.columns


def test_get_treasury_prices_without_any_prices(monkeypatch):
    monkeypatch.setattr(
        treasurydirect_model,
        "_get_day",
        lambda day: treasurydirect_model._parse_prices(""),
    )

    with pytest.raises(ValueError, match="no Treasury prices"):
        treasurydirect_model.get_treasury_prices("2026-11-05")


def frn_auctions(security_type, security_term=None):
    """The terms of the floating rate note of PRICE_PAGE and the 13-week bills before it."""
    if security_type == "FRN":
        return pd.DataFrame(
            {
                "CUSIP": ["91282CMH1"],
                "Reopening": [False],
                "Spread": [0.00205],
                "Dated Date": [date(2025, 1, 31)],
            }
        )
    auctions = [
        (date(2026, 10, 13), 0.04085),
        (date(2026, 10, 19), 0.04095),
        (date(2026, 10, 26), 0.04050),
        (date(2026, 11, 2), 0.04040),
    ]
    return pd.DataFrame(
        {
            "Auction Date": [day for day, _ in auctions],
            "High Rate": [rate for _, rate in auctions],
            "Days to Maturity": [91.0] * len(auctions),
            "Cash Management Bill": [False] * len(auctions),
        }
    )


def test_get_treasury_securities(monkeypatch):
    monkeypatch.setattr(
        treasurydirect_model,
        "get_treasury_prices",
        lambda price_date: (
            treasurydirect_model._parse_prices(PRICE_PAGE)
            .assign(Price=lambda prices: prices["End of Day"].fillna(prices["Sell"]))
            .drop(columns=["Buy", "Sell", "End of Day"]),
            date(2026, 11, 5),
        ),
    )
    monkeypatch.setattr(
        treasurydirect_model,
        "get_auctioned_securities",
        frn_auctions,
    )
    fixedincome = fixedincome_controller.FixedIncome(rounding=None)

    securities = fixedincome.get_treasury_securities(date="2026-11-05")

    assert securities.index.tolist() == [
        "912797VW4",
        "91282CMH1",
        "91282CLY5",
        "912810FT0",
        "91282CPZ8",
    ]
    assert securities.loc["91282CPZ8", "Price Date"] == "2026-11-05"
    years = bond_data_model.get_years_to_maturity(date(2026, 11, 5), date(2036, 2, 15))
    expected = bond_data_model.get_bond_analytics(0.04125, years, 94.90625)
    assert securities.loc[
        "91282CPZ8", ["Yield to Maturity", "Modified Duration"]
    ].tolist() == pytest.approx(expected)
    assert securities.loc["912797VW4", "Yield to Maturity"] == pytest.approx(
        bond_data_model.get_bill_analytics(5, 99.957556)[0]
    )
    # A floating rate note yields its index rate plus its discount margin, and its duration
    # is that to the index rate, close to zero (see test_treasury_markets_controller).
    assert securities.loc["91282CMH1", "Yield to Maturity"] == pytest.approx(
        0.0412, abs=0.0005
    )
    assert 0 <= securities.loc["91282CMH1", "Modified Duration"] < 0.01

    # Without the terms of the notes they keep a price but no yield or duration.
    monkeypatch.setattr(
        treasurydirect_model,
        "get_auctioned_securities",
        lambda security_type, security_term=None: pd.DataFrame(),
    )
    securities = fixedincome.get_treasury_securities(date="2026-11-05")
    assert np.isnan(securities.loc["91282CMH1", "Yield to Maturity"])
    assert securities.loc["91282CMH1", "Price"] == pytest.approx(100.03125)

    notes = fixedincome.get_treasury_securities(security_type="note,Bond")
    assert notes["Security Type"].tolist() == ["Bond", "Note"]

    # An unknown type is logged and gives an empty result.
    assert fixedincome.get_treasury_securities(security_type="Strip").empty


def holding(
    fund, name, isin, cusip, shares, market_value, updated="2026-11-06 14:32:06"
):
    return {
        "Fund": fund,
        "Asset": "",
        "Name": name,
        "ISIN": isin,
        "CUSIP": cusip,
        "Shares": shares,
        "Weight": 0.01,
        "Market Value": market_value,
        "Updated At": updated,
    }


HOLDINGS = pd.DataFrame(
    [
        # iShares count par in hundreds: market value / amount is the price per 100.
        holding(
            "LQD",
            "APPLE INC 4.00% 05/10/2033",
            "US037833ET32",
            "037833ET3",
            1000,
            98000,
        ),
        holding(
            "LQD",
            "ORACLE CORP 144A 5.50% 11/15/2045",
            "US68389XCE31",
            "68389XCE3",
            2000,
            180000,
        ),
        holding(
            "LQD",
            "BLK CSH FND TREASURY SL AGENCY",
            "US0669224778",
            "066922477",
            5e6,
            5e6,
        ),
        holding(
            "LQD",
            "BANK OF AMERICA CORP FXD-FRN 5.00% 01/01/2030",
            "US06051GLA57",
            "06051GLA5",
            10,
            1000,
        ),
        holding("LQD", "SIEMENS AG 3.00% 03/01/2035", "DE000A3E5MG8", "", 100, 9000),
        # A sterling note of a US company: a US CUSIP, but no ISIN to show its currency.
        holding("LQD", "APPLE INC 3.60% 07/31/2042", "", "037833BK5", 100, 12000),
        holding(
            "IGSB",
            "APPLE INC 4.00% 05/10/2033",
            "US037833ET32",
            "037833ET3",
            1000,
            99000,
        ),
        holding(
            "IGSB",
            "MICROSOFT CORP 2.00% 08/08/2026",
            "US594918BS21",
            "594918BS2",
            1000,
            99000,
        ),
        holding(
            "IGSB",
            "MICROSOFT CORP 3.00% 02/12/2045",
            "US594918BY98",
            "594918BY9",
            1000,
            450000,
        ),
        # A feed of a day earlier: valued at the close of 4 November.
        holding(
            "IGIB",
            "APPLE INC 4.00% 05/10/2033",
            "US037833ET32",
            "037833ET3",
            1000,
            90000,
            "2026-11-05 14:32:06",
        ),
        holding(
            "IGIB",
            "MICROSOFT CORP 3.50% 02/12/2035",
            "US594918BJ27",
            "594918BJ2",
            1000,
            95000,
            "2026-11-05 14:32:06",
        ),
        # Schwab gives no ISINs (and counts par in dollars), so its bonds are left out.
        holding("SCHZ", "APPLE INC 4.00% 05/10/2033", "", "037833ET3", 100000, 98000),
        holding("SCHZ", "MICROSOFT CORP 3.50% 02/12/2035", "", "", 100000, 95000),
    ]
).set_index(["Fund", "Asset"])

# The Treasury par curve of each price date.
CURVES = {
    "2026-11-05": pd.Series({"1Y": 0.04, "10Y": 0.045, "30Y": 0.05}),
    "2026-11-04": pd.Series({"1Y": 0.03, "10Y": 0.035, "30Y": 0.04}),
}


@pytest.fixture
def bond_universe(monkeypatch):
    funds = HOLDINGS.index.get_level_values(0)
    monkeypatch.setattr(
        toolkit_fmp_model,
        "get_etf_holdings",
        lambda requested, api_key: (
            HOLDINGS.loc[[fund for fund in requested if fund in funds]],
            [],
        ),
    )
    monkeypatch.setattr(
        fixedincome_controller.FixedIncome,
        "_latest_yield_curve",
        lambda self, country, date=None: CURVES[date],
    )
    return fixedincome_controller.FixedIncome(api_key="test", rounding=None)


def test_get_bond_universe(bond_universe):
    bonds = bond_universe.get_bond_universe(etfs="LQD,SCHZ")

    # The cash line, the fixed-to-floating note, the euro bond, the sterling note without an
    # ISIN and the Schwab bonds are left out.
    assert bonds.index.tolist() == ["US037833ET32", "US68389XCE31"]
    assert bonds.loc["US037833ET32", "Funds"] == "LQD"
    assert bonds.loc["US037833ET32", "Price"] == pytest.approx(98)
    assert bonds.loc["US68389XCE31", "Issuer"] == "ORACLE CORP"
    assert bool(bonds.loc["US68389XCE31", "Rule 144A"])
    # The holdings of 6 November are valued at the close of 5 November.
    assert set(bonds["Price Date"]) == {"2026-11-05"}

    # Corporate bonds count their days on the 30/360 Bond Basis; QuantLib's FixedRateBond on
    # Thirty360(BondBasis) gives a yield of 4.355857% and a modified duration of 5.556143.
    years = bond_data_model.get_years_to_maturity(
        date(2026, 11, 5), date(2033, 5, 10), day_count="30/360"
    )
    yield_to_maturity, duration = bond_data_model.get_bond_analytics(0.04, years, 98)
    assert [yield_to_maturity, duration] == pytest.approx(
        [0.04355857, 5.556143], abs=1e-6
    )
    treasury = bond_data_model.interpolate_curve(
        CURVES["2026-11-05"], pd.Series([years])
    ).iloc[0]
    assert bonds.loc[
        "US037833ET32", ["Yield to Maturity", "Modified Duration", "G-Spread"]
    ].tolist() == pytest.approx(
        [yield_to_maturity, duration, yield_to_maturity - treasury]
    )


def test_get_bond_universe_needs_isins(bond_universe, caplog):
    # A fund without ISINs gives no bonds: its currency and valuation date are unknown.
    assert bond_universe.get_bond_universe(etfs="SCHZ").empty
    assert "SCHZ give no ISINs" in caplog.text


def test_get_bond_universe_takes_the_median_price_whatever_the_order(bond_universe):
    one = bond_universe.get_bond_universe(etfs="LQD,IGSB")
    other = bond_universe.get_bond_universe(etfs="IGSB,LQD")

    # The matured bond and the bond priced at 450 are left out; Apple, held by both funds,
    # is one row at the median of their prices (98 and 99).
    assert one.index.tolist() == ["US037833ET32", "US68389XCE31"]
    assert one.loc["US037833ET32", "Funds"] == "LQD, IGSB"
    assert other.loc["US037833ET32", "Funds"] == "IGSB, LQD"
    assert one.loc["US037833ET32", "Price"] == pytest.approx(98.5)
    pd.testing.assert_frame_equal(
        one.drop(columns="Funds"), other.drop(columns="Funds").loc[one.index]
    )


def test_get_bond_universe_values_each_bond_on_its_own_price_date(bond_universe):
    bonds = bond_universe.get_bond_universe(etfs="LQD,IGIB")

    # Apple is priced by LQD, whose feed is the latest; IGIB's older price is not used.
    assert bonds.loc["US037833ET32", "Price"] == pytest.approx(98)
    assert bonds.loc["US037833ET32", "Price Date"] == "2026-11-05"
    assert bonds.loc["US037833ET32", "Funds"] == "LQD, IGIB"

    # Microsoft, held only by IGIB, counts its years from 4 November and takes the G-Spread
    # over the curve of that day, whatever the other funds add.
    microsoft = bonds.loc["US594918BJ27"]
    assert microsoft["Price Date"] == "2026-11-04"
    years = bond_data_model.get_years_to_maturity(
        date(2026, 11, 4), date(2035, 2, 12), day_count="30/360"
    )
    yield_to_maturity = bond_data_model.get_bond_analytics(0.035, years, 95)[0]
    treasury = bond_data_model.interpolate_curve(
        CURVES["2026-11-04"], pd.Series([years])
    ).iloc[0]
    assert [microsoft["Yield to Maturity"], microsoft["G-Spread"]] == pytest.approx(
        [yield_to_maturity, yield_to_maturity - treasury]
    )
    alone = bond_universe.get_bond_universe(etfs="IGIB", bonds="US594918BJ27")
    assert alone.loc["US594918BJ27", "G-Spread"] == pytest.approx(microsoft["G-Spread"])


def test_get_bond_universe_filters(bond_universe):
    microsoft = bond_universe.get_bond_universe(
        etfs="LQD,IGIB", issuer="microsoft,oracle"
    )
    assert microsoft["Issuer"].tolist() == ["MICROSOFT CORP", "ORACLE CORP"]

    short = bond_universe.get_bond_universe(
        etfs="LQD,IGIB", maturity_start="2030-01-01", maturity_end="2040-12-31"
    )
    assert short["Maturity"].tolist() == ["2033-05-10", "2035-02-12"]

    # Errors are logged and give an empty result.
    assert bond_universe.get_bond_universe(etfs="LQD", issuer="NVIDIA").empty


def test_get_bond_universe_requires_an_api_key():
    assert fixedincome_controller.FixedIncome().get_bond_universe().empty


def test_spreads_at_maturities_interpolate_each_issuer_and_never_extrapolate():
    bonds = pd.DataFrame(
        {
            "Issuer": ["APPLE INC", "APPLE INC", "APPLE INC", "BOEING CO", "BOEING CO"],
            "Years": [2.0, 6.0, 6.0, 3.0, 9.0],
            "G-Spread": [0.002, 0.004, 0.006, 0.008, 0.011],
        }
    )

    spreads = bond_data_model.get_spreads_at_maturities(bonds, [2, 4, 10])

    # Apple's two six-year bonds count with their average spread, 0.005.
    assert spreads.loc["APPLE INC", "4Y"] == pytest.approx(0.0035)
    assert spreads.loc["APPLE INC", "2Y"] == pytest.approx(0.002)
    assert pd.isna(spreads.loc["APPLE INC", "10Y"])
    assert pd.isna(spreads.loc["BOEING CO", "2Y"])
    assert spreads.loc["BOEING CO", "4Y"] == pytest.approx(0.0085)
    assert spreads.loc["APPLE INC", "Bonds"] == 3
    assert spreads.loc["BOEING CO", "Longest"] == 9.0


def test_programme_markers_are_not_part_of_the_issuer():
    for name in (
        "FORD MOTOR CREDIT COMPANY LLC MTN 4.5% 01/01/2031",
        "FORD MOTOR CREDIT COMPANY LLC (FXD 4.5% 01/01/2031",
        "FORD MOTOR CREDIT COMPANY LLC 4.5% 01/01/2031",
    ):
        assert (
            bond_data_model.parse_bond_name(name)["Issuer"]
            == "FORD MOTOR CREDIT COMPANY LLC"
        )


def test_a_matured_bond_or_impossible_frequency_is_refused():
    with pytest.raises(ValueError, match="on or after the maturity date"):
        bond_data_model.get_years_to_maturity(date(2031, 1, 1), date(2030, 1, 1))
    with pytest.raises(ValueError, match="on or after the maturity date"):
        bond_data_model.get_treasury_accrued_interest(
            0.04, date(2030, 1, 1), date(2030, 1, 1)
        )
    for frequency in (0, -1, 5):
        with pytest.raises(ValueError, match="frequency"):
            bond_data_model.get_years_to_maturity(
                date(2026, 1, 1), date(2030, 1, 1), frequency=frequency
            )
