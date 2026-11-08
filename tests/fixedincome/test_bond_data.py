"""Bond Universe and Treasury Securities Tests"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from financetoolkit.cache.cache_controller import clear_active_cache
from financetoolkit.fixedincome import (
    bond_data_model,
    bond_model,
    fixedincome_controller,
    treasurydirect_model,
)


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


@pytest.mark.parametrize(
    ("cusip", "isin"),
    [
        ("037833100", "US0378331005"),  # Apple shares
        ("594918104", "US5949181045"),  # Microsoft shares
        ("84615QAA1", "US84615QAA13"),  # a 144A bond, as LQD lists it
        ("03522AAJ9", "US03522AAJ97"),
        ("", ""),
        ("12345", ""),
    ],
)
def test_cusip_to_isin(cusip, isin):
    assert bond_data_model.cusip_to_isin(cusip) == isin


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
    assert np.isnan(
        bond_data_model.get_years_to_maturity(date(2030, 1, 1), date(2030, 1, 1))
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
    # A floating rate note has no yield to maturity from its price alone.
    assert np.isnan(securities.loc["91282CMH1", "Yield to Maturity"])

    notes = fixedincome.get_treasury_securities(security_type="note,Bond")
    assert notes["Security Type"].tolist() == ["Bond", "Note"]

    # An unknown type is logged and gives an empty result.
    assert fixedincome.get_treasury_securities(security_type="Strip").empty


def holding(fund, name, isin, cusip, shares, market_value):
    return {
        "Fund": fund,
        "Asset": "",
        "Name": name,
        "ISIN": isin,
        "CUSIP": cusip,
        "Shares": shares,
        "Weight": 0.01,
        "Market Value": market_value,
        "Updated At": "2026-11-06 14:32:06",
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
            "LQD", "ORACLE CORP 144A 5.50% 11/15/2045", "", "68389XCE3", 2000, 180000
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
        # Schwab counts par in dollars: market value / amount is the price per 1.
        holding("SCHZ", "APPLE INC 4.00% 05/10/2033", "", "037833ET3", 100000, 98000),
        holding("SCHZ", "MICROSOFT CORP 3.50% 02/12/2035", "", "", 100000, 95000),
        holding("SCHZ", "MICROSOFT CORP 2.00% 08/08/2026", "", "", 100000, 99000),
        holding("SCHZ", "MICROSOFT CORP 3.00% 02/12/2045", "", "", 100000, 450000),
    ]
).set_index(["Fund", "Asset"])


@pytest.fixture
def bond_universe(monkeypatch):
    monkeypatch.setattr(
        fixedincome_controller.toolkit_fmp_model,
        "get_etf_holdings",
        lambda funds, api_key: (
            HOLDINGS.loc[[fund for fund in funds if fund in ("LQD", "SCHZ")]],
            [],
        ),
    )
    monkeypatch.setattr(
        fixedincome_controller.FixedIncome,
        "_latest_yield_curve",
        lambda self, country, date=None: pd.Series(
            {"1Y": 0.04, "10Y": 0.045, "30Y": 0.05}
        ),
    )
    return fixedincome_controller.FixedIncome(api_key="test", rounding=None)


def test_get_bond_universe(bond_universe):
    bonds = bond_universe.get_bond_universe(etfs="LQD,SCHZ")

    # The cash line, the fixed-to-floating note, the euro bond, the matured bond and the
    # bond priced at 450 are left out; Apple, held by both funds, is one row.
    assert bonds.index.tolist() == [
        "US037833ET32",
        "MICROSOFT CORP 3.50% 02/12/2035",
        "US68389XCE31",
    ]
    assert bonds.loc["US037833ET32", "Funds"] == "LQD, SCHZ"
    assert bonds.loc["US037833ET32", "Price"] == pytest.approx(98)
    assert bonds.loc["MICROSOFT CORP 3.50% 02/12/2035", "Price"] == pytest.approx(95)
    assert bonds.loc["US68389XCE31", "Issuer"] == "ORACLE CORP"
    assert bool(bonds.loc["US68389XCE31", "Rule 144A"])
    # The holdings of 6 November are valued at the close of 5 November.
    assert set(bonds["Price Date"]) == {"2026-11-05"}

    years = bond_data_model.get_years_to_maturity(date(2026, 11, 5), date(2033, 5, 10))
    yield_to_maturity, duration = bond_data_model.get_bond_analytics(0.04, years, 98)
    curve = pd.Series({"1Y": 0.04, "10Y": 0.045, "30Y": 0.05})
    treasury = bond_data_model.interpolate_curve(curve, pd.Series([years])).iloc[0]
    assert bonds.loc[
        "US037833ET32", ["Yield to Maturity", "Modified Duration", "G-Spread"]
    ].tolist() == pytest.approx(
        [yield_to_maturity, duration, yield_to_maturity - treasury]
    )


def test_get_bond_universe_filters(bond_universe):
    microsoft = bond_universe.get_bond_universe(
        etfs="LQD,SCHZ", issuer="microsoft,oracle"
    )
    assert microsoft["Issuer"].tolist() == ["MICROSOFT CORP", "ORACLE CORP"]

    short = bond_universe.get_bond_universe(
        etfs="LQD,SCHZ", maturity_start="2030-01-01", maturity_end="2040-12-31"
    )
    assert short["Maturity"].tolist() == ["2033-05-10", "2035-02-12"]

    # Errors are logged and give an empty result.
    assert bond_universe.get_bond_universe(etfs="LQD", issuer="NVIDIA").empty


def test_get_bond_universe_requires_an_api_key():
    assert fixedincome_controller.FixedIncome().get_bond_universe().empty
