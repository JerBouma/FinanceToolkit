"""Offline tests for the volatility, stress test, corporate bond and carbon price sources."""

import io

import pandas as pd
import pytest
import requests

from financetoolkit.cache.cache_controller import clear_active_cache


@pytest.fixture(autouse=True)
def isolate_from_the_active_cache():
    """Start every test without an active cache another test may have left behind."""
    clear_active_cache()
    yield
    clear_active_cache()


class FakeResponse:
    def __init__(self, text="", content=None, status_code=200):
        self.text = text
        self.content = content if content is not None else text.encode()
        self.status_code = status_code
        self.headers = {}


def _workbook(sheets: dict[str, list[list]]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(
                writer, sheet_name=name, header=False, index=False
            )
    return buffer.getvalue()


def _not_found(url):
    response = requests.Response()
    response.status_code = 404
    return requests.exceptions.HTTPError(f"404 for {url}", response=response)


def test_implied_volatility_is_a_decimal_per_market_and_maturity(monkeypatch):
    from financetoolkit import Economics
    from financetoolkit.economics import cboe_model, fmp_model, stoxx_model

    days = ("2026-09-29", "2026-09-30")
    cboe = {
        "VIX9D": "DATE,OPEN,HIGH,LOW,CLOSE\n09/29/2026,1,1,1,13.0\n09/30/2026,1,1,1,14.2\n",
        "VIX": "DATE,OPEN,HIGH,LOW,CLOSE\n09/29/2026,1,1,1,15.5\n09/30/2026,1,1,1,16.3\n",
        "VVIX": "DATE,VVIX\n09/29/2026,85.4\n09/30/2026,86.0\n",
    }

    def fake_cboe(url, timeout):
        symbol = url.rsplit("/", 1)[-1].removesuffix("_History.csv")
        if symbol not in cboe:
            raise _not_found(url)
        return FakeResponse(text=cboe[symbol])

    vstoxx = "Date;Symbol;Indexvalue\n29.09.2026;V6I1;17.6\n30.09.2026;V6I1;18.1\n"
    monkeypatch.setattr(cboe_model, "get_request", fake_cboe)
    monkeypatch.setattr(
        stoxx_model,
        "get_request",
        lambda url, timeout: FakeResponse(
            text=vstoxx if url.endswith("v6i1.txt") else ""
        ),
    )
    monkeypatch.setattr(
        fmp_model,
        "get_index_history",
        lambda symbol, api_key, start, end: pd.Series(
            [105.2, 110.4] if symbol == "^MOVE" else [4.5, 4.6],
            index=pd.PeriodIndex(list(days), freq="D"),
        ),
    )

    economics = Economics(start_date="2026-09-01", end_date="2026-09-30", api_key="key")
    volatility = economics.get_implied_volatility(
        markets=[
            "US Equity",
            "Euro Area Equity",
            "US Treasury Rates",
            "US Equity Volatility",
        ],
        rounding=6,
    )
    last = volatility.loc[pd.Period("2026-09-30", "D")]

    assert last[("US Equity", "9D")] == pytest.approx(0.142)
    assert last[("US Equity", "1M")] == pytest.approx(0.163)
    assert last[("Euro Area Equity", "1M")] == pytest.approx(0.181)
    assert last[("US Equity Volatility", "1M")] == pytest.approx(0.86)
    # The MOVE is in basis points: 110.4 basis points is 0.01104.
    assert last[("US Treasury Rates", "1M")] == pytest.approx(0.01104)
    assert volatility.columns.names == ["Market", "Maturity"]

    # Without an API key the FMP markets are left out.
    without_key = Economics(start_date="2026-09-01", end_date="2026-09-30")
    assert without_key.get_implied_volatility(markets="US Treasury Rates").empty


def test_federal_reserve_scenario_is_split_by_region_in_decimals(monkeypatch):
    from financetoolkit.economics import frb_model

    domestic = (
        "Scenario Name,Date,Real GDP growth,10-year Treasury yield,"
        "House Price Index (Level),Market Volatility Index (Level)\n"
        "Supervisory Severely Adverse,2026 Q1,-5.4,3.1,303,59.7\n"
        "Supervisory Severely Adverse,2026 Q2,-4.9,2.7,283.2,72\n"
    )
    international = (
        "Scenario Name,Date,Euro area real GDP growth,"
        "Euro area bilateral dollar exchange rate (USD/euro),U.K. inflation\n"
        "Supervisory Severely Adverse,2026 Q1,-8.6,1.124,0.9\n"
        "Supervisory Severely Adverse,2026 Q2,-8.5,1.08,-0.1\n"
    )
    requested = []

    def fake(url, timeout):
        requested.append(url)
        if "2026_Final" not in url:
            raise _not_found(url)
        return FakeResponse(
            text=domestic if url.endswith("Domestic.csv") else international
        )

    monkeypatch.setattr(frb_model, "get_request", fake)

    scenario = frb_model.get_supervisory_scenario("adverse", year=2026)
    first = scenario.loc[pd.Period("2026Q1", "Q")]

    assert first[("United States", "Real GDP growth")] == pytest.approx(-0.054)
    assert first[("United States", "House Price Index (Level)")] == pytest.approx(303)
    assert first[("United States", "Market Volatility Index (VIX)")] == pytest.approx(
        0.597
    )
    assert first[("Euro Area", "Real GDP growth")] == pytest.approx(-0.086)
    assert first[
        ("Euro Area", "Bilateral dollar exchange rate (USD/euro)")
    ] == pytest.approx(1.124)
    assert first[("United Kingdom", "Inflation")] == pytest.approx(0.009)
    # A year whose scenarios are not published gives nothing rather than an error.
    assert frb_model.get_supervisory_scenario("adverse", year=2031).empty


def test_esrb_scenario_reads_the_country_panel(monkeypatch):
    from financetoolkit.economics import esrb_model

    gdp = [[None] * 9 for _ in range(3)]
    gdp[1] = [None, "Real GDP"] + [None] * 7
    gdp += [
        [
            None,
            None,
            None,
            "Historical growth (%)",
            "Baseline growth (%)",
            None,
            "Adverse growth (%)",
            None,
            None,
        ],
        [None, None, None, 2024, 2025, 2026, 2025, 2026, None],
        [None, "Germany", "DE", -0.16, 0.16, 0.84, -3.63, -4.24, None],
        [None, "Euro area", "EA", 0.8, 0.9, 1.2, -2.8, -3.5, None],
        [None, "Notes: annual averages.", None, None, None, None, None, None, None],
    ]
    itraxx = [[None] * 9 for _ in range(3)]
    itraxx += [
        [
            None,
            None,
            "Historical level",
            "Baseline level",
            None,
            "Adverse level",
            None,
            None,
            None,
        ],
        [None, None, 2024, 2025, 2026, 2025, 2026, None, None],
        [None, "iTraxx Overall 5y", 60, 62, 64, 240, 198, None, None],
    ]
    workbook = _workbook({"GDP": gdp, "Itraxx": itraxx})
    page = (
        '<a href="/mppa/stress/shared/pdf/esrb.stress_test230131.macrofinancialscenario~a.en.xlsx">2023</a>'
        '<a href="/mppa/stress/shared/pdf/esrb.stress_test250120.macrofinancialscenario~b.en.xlsx">2025</a>'
    )
    requested = []

    def fake(url, timeout):
        requested.append(url)
        return (
            FakeResponse(text=page)
            if url.endswith(".html")
            else FakeResponse(content=workbook)
        )

    monkeypatch.setattr(esrb_model, "get_request", fake)

    adverse = esrb_model.get_macro_financial_scenario("adverse")

    assert requested[-1].endswith("stress_test250120.macrofinancialscenario~b.en.xlsx")
    assert adverse.loc[
        pd.Period("2026", "Y"), ("Germany", "Real GDP growth")
    ] == pytest.approx(-0.0424)
    assert adverse.loc[
        pd.Period("2025", "Y"), ("Euro Area", "Real GDP growth")
    ] == pytest.approx(-0.028)
    # iTraxx spreads are in basis points.
    assert adverse.loc[
        pd.Period("2025", "Y"), ("iTraxx Overall 5y", "Credit spread")
    ] == pytest.approx(0.024)
    assert not any(
        label.startswith("Notes") for label in adverse.columns.get_level_values(0)
    )

    historic = esrb_model.get_macro_financial_scenario("historic")
    assert list(historic.index) == [pd.Period("2024", "Y")]


def test_corporate_bond_yields_and_spreads(monkeypatch):
    from financetoolkit import FixedIncome
    from financetoolkit.fixedincome import bundesbank_model, rba_model

    bundesbank = (
        "BBK_SEIS_ISSUER_CLASS;TIME_PERIOD;OBS_VALUE;OBS_STATUS\n"
        "X2000;2026-07;3.96;\nX2000;2026-08;4.06;\n"
        "S13;2026-07;3.11;\nS13;2026-08;3.21;\nS13;1955-01;0.00;N\n"
    )
    monkeypatch.setattr(
        bundesbank_model,
        "get_request",
        lambda url, timeout, extra_headers: FakeResponse(text=bundesbank),
    )
    f3 = (
        "F3 AGGREGATE MEASURES\nTitle,a,b\nSeries ID,FNFYA5M,FNFYBBB7M\n"
        "31/07/2026,5.41,5.70\n31/08/2026,5.55,5.82\n"
    )
    f2 = (
        "F2 CAPITAL MARKET YIELDS\nSeries ID,FCMYGBAG2D,FCMYGBAG3D,FCMYGBAG5D,FCMYGBAG10D\n"
        "31-Jul-2026,4.5,4.5,4.6,4.9\n31-Aug-2026,4.6,4.6,4.7,5.1\n"
    )
    monkeypatch.setattr(
        rba_model,
        "get_request",
        lambda url, timeout, extra_headers: FakeResponse(
            text=f3 if "f3-" in url else f2
        ),
    )

    fixedincome = FixedIncome(start_date="2026-07-01", end_date="2026-08-31")
    yields = fixedincome.get_corporate_bond_yields()
    spreads = fixedincome.get_corporate_bond_yields(spread=True)
    august = pd.Period("2026-08", "M")

    assert yields.loc[august, ("Germany", "All ratings")] == pytest.approx(0.0406)
    assert yields.loc[august, ("Australia", "BBB 7Y")] == pytest.approx(0.0582)
    assert spreads.loc[august, ("Germany", "All ratings")] == pytest.approx(0.0085)
    assert spreads.loc[august, ("Australia", "A 5Y")] == pytest.approx(0.0085)
    # 7 years lies between the 5 and 10-year government yields: 4.7% + 2/5 of 0.4%.
    assert spreads.loc[august, ("Australia", "BBB 7Y")] == pytest.approx(
        0.0582 - 0.0486
    )
    assert "United States" not in yields.columns.get_level_values(0)


def test_carbon_price_averages_the_days_general_allowance_auctions(monkeypatch):
    from financetoolkit import Economics
    from financetoolkit.economics import eex_model

    rows = [[None] * 7 for _ in range(5)]
    rows += [
        [
            None,
            "Date",
            "Time",
            "Auction Name",
            "Contract",
            "Status",
            "Auction Price €/tCO2",
        ],
        [
            None,
            pd.Timestamp("2026-10-06"),
            None,
            "Auction CAP3 EU",
            "T3PA",
            "successful",
            84.31,
        ],
        [
            None,
            pd.Timestamp("2026-10-06"),
            None,
            "Auction DE",
            "T3PA",
            "successful",
            84.11,
        ],
        [
            None,
            pd.Timestamp("2026-10-05"),
            None,
            "Auction CAP3 EU",
            "T3PA",
            "successful",
            82.6,
        ],
        [
            None,
            pd.Timestamp("2026-10-05"),
            None,
            "EUAA Auction",
            "EAA3",
            "successful",
            70.0,
        ],
        [
            None,
            pd.Timestamp("2026-10-02"),
            None,
            "Auction CAP3 EU",
            "T3PA",
            "cancelled",
            None,
        ],
    ]
    workbook = _workbook({"Primary Market Auction": rows})
    requested = []

    def fake(url, timeout):
        requested.append(url)
        if "2026" not in url:
            raise _not_found(url)
        return FakeResponse(content=workbook)

    monkeypatch.setattr(eex_model, "get_request", fake)

    prices = Economics(
        start_date="2026-10-01", end_date="2026-10-06"
    ).get_carbon_price()

    assert prices["European Union"].to_dict() == {
        pd.Period("2026-10-05", "D"): pytest.approx(82.6),
        pd.Period("2026-10-06", "D"): pytest.approx(84.21),
    }
    # Only the years from the buffered start date to the end date are requested.
    assert all("2025" in url or "2026" in url for url in requested)
