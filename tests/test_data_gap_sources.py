"""Offline tests for the EIOPA, Bank of England, Eurostat, BIS, IMF, Shiller and ECB sources."""

import io
import zipfile

import pandas as pd
import pytest

from financetoolkit.cache.cache_controller import clear_active_cache


@pytest.fixture(autouse=True)
def isolate_from_the_active_cache():
    """Start every test without an active cache another test may have left behind."""
    clear_active_cache()
    yield
    clear_active_cache()


class FakeResponse:
    def __init__(self, text="", content=None, payload=None, headers=None):
        self.text = text
        self.content = content if content is not None else text.encode()
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload


def _workbook(sheets: dict[str, list[list]]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(
                writer, sheet_name=name, header=False, index=False
            )
    return buffer.getvalue()


def test_eiopa_release_is_parsed_per_country_and_maturity(monkeypatch):
    from financetoolkit.fixedincome import eiopa_model

    rows = [[None] * 4 for _ in range(10)]
    rows[1] = [None, "Main menu", "Euro", "Czechia"]
    rows += [[None, 1, 0.0327, 0.035], [None, 2, 0.0344, 0.036]]
    workbook = _workbook({"RFR_spot_no_VA": rows})
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("EIOPA_RFR_20260930_Term_Structures.xlsx", workbook)
    page = '<a href="/document/download/abc_en?filename=EIOPA_RFR_20260930.zip">September</a>'

    def fake(url, timeout):
        return (
            FakeResponse(text=page)
            if url.endswith("_en")
            else FakeResponse(content=archive.getvalue())
        )

    monkeypatch.setattr(eiopa_model, "get_request", fake)

    curves = eiopa_model.get_risk_free_rate_term_structures(
        "spot_no_va", "2026-09-01", "2026-09-30"
    )

    assert list(curves.columns) == [
        ("Czech Republic", "1Y"),
        ("Czech Republic", "2Y"),
        ("Euro Area", "1Y"),
        ("Euro Area", "2Y"),
    ]
    assert curves.loc[pd.Period("2026-09", "M"), ("Euro Area", "2Y")] == pytest.approx(
        0.0344
    )


def test_bank_of_england_spot_curve_finds_its_worksheet_by_name():
    from financetoolkit.economics import boe_model

    rows = [
        [None, "UK real spot curve"],
        ["Maturity", None, None],
        ["years:", 2.5, 3],
        [None, None, None],
        [pd.Timestamp("1985-01-02"), 4.27, 4.21],
    ]
    workbook = _workbook({"info": [["x"]], "4.  real spot curve": rows})

    curve = boe_model._parse_spot_curve(workbook, "test")

    assert list(curve.columns) == ["2.5Y", "3Y"]
    assert curve.iloc[0].tolist() == [pytest.approx(0.0427), pytest.approx(0.0421)]


def test_eurostat_life_table_has_a_column_per_country_and_age(monkeypatch):
    from financetoolkit.economics import eurostat_model

    payload = {
        "id": ["freq", "indic_de", "sex", "age", "geo", "time"],
        "size": [1, 1, 1, 3, 1, 2],
        "dimension": {
            "freq": {"category": {"index": {"A": 0}}},
            "indic_de": {"category": {"index": {"PROBDEATH": 0}}},
            "sex": {"category": {"index": {"M": 0}}},
            "age": {"category": {"index": {"Y_LT1": 0, "Y65": 1, "Y_GE95": 2}}},
            "geo": {"category": {"index": {"DE": 0}}},
            "time": {"category": {"index": {"2023": 0, "2024": 1}}},
        },
        "value": {
            "0": 0.0033,
            "1": 0.0035,
            "2": 0.0153,
            "3": 0.0151,
            "4": 1.0,
            "5": 1.0,
        },
    }
    monkeypatch.setattr(
        eurostat_model,
        "get_request",
        lambda url, timeout: FakeResponse(payload=payload),
    )

    table = eurostat_model.get_life_table(
        "death_probability", "male", "2023-01-01", "2024-12-31", ["DE"]
    )

    assert list(table.columns) == [("Germany", 0), ("Germany", 65), ("Germany", 95)]
    assert table.loc[pd.Period("2024", "Y"), ("Germany", 65)] == pytest.approx(0.0151)


def test_bis_exchange_rates_and_property_series_selection(monkeypatch):
    from financetoolkit.economics import bis_model

    fx = "FREQ,REF_AREA,CURRENCY,COLLECTION,TIME_PERIOD,OBS_VALUE\nM,GB,GBP,E,2026-08,0.7386\nM,JP,JPY,E,2026-08,159.73\n"
    property_prices = (
        "FREQ,REF_AREA,COVERED_AREA,RE_TYPE,RE_VINTAGE,COMPILING_ORG,PRICED_UNIT,ADJUST_CODED,TIME_PERIOD,OBS_VALUE\n"
        "Q,JP,3,M,1,4,1,0,1955-Q1,10\nQ,JP,3,M,1,4,1,0,2026-Q1,99\nQ,JP,0,A,0,3,6,0,2026-Q1,120\n"
    )

    def fake(url, timeout):
        return FakeResponse(text=fx if "WS_XRU" in url else property_prices)

    monkeypatch.setattr(bis_model, "get_request", fake)

    rates = bis_model.get_exchange_rates("monthly", "2026-06-01", "2026-09-30")
    prices = bis_model.get_commercial_property_prices("2025-01-01", "2026-09-30")

    assert rates.loc[pd.Period("2026-08", "M"), "Japan"] == pytest.approx(159.73)
    # The whole-country index of all commercial property wins over a longer partial one.
    assert prices.loc[pd.Period("2026Q1", "Q"), "Japan"] == 120


def test_imf_observations_are_read_and_named(monkeypatch):
    from financetoolkit.economics import imf_model

    structure = (
        '<str:Codelist id="CL_COUNTRY"><str:Code id="PAK"><com:Name xml:lang="en">Pakistan</com:Name></str:Code>'
        '<str:Code id="BHS"><com:Name xml:lang="en">Bahamas, The</com:Name></str:Code></str:Codelist>'
    )
    data = (
        '<Series COUNTRY="PAK" FREQUENCY="M"><Obs TIME_PERIOD="2026-M09" OBS_VALUE="10.2591"/></Series>'
        '<Series COUNTRY="BHS" FREQUENCY="M"><Obs TIME_PERIOD="2026-M09" OBS_VALUE="2.5"/></Series>'
    )

    def fake(url, timeout):
        return FakeResponse(text=structure if "dataflow" in url else data)

    monkeypatch.setattr(imf_model, "get_request", fake)

    rates = imf_model.get_inflation_rate("2026-09-01", "2026-09-30")

    assert rates.loc[pd.Period("2026-09", "M")].to_dict() == {
        "Bahamas": pytest.approx(0.025),
        "Pakistan": pytest.approx(0.102591),
    }


def test_shiller_dates_read_october_correctly():
    from financetoolkit.economics import shiller_model

    assert shiller_model._to_month(1871.01) == pd.Period("1871-01", "M")
    assert shiller_model._to_month(2023.1) == pd.Period("2023-10", "M")
    assert shiller_model._to_month(2023.12) == pd.Period("2023-12", "M")


def test_ecb_series_by_area_name_the_euro_area(monkeypatch):
    from financetoolkit.economics import ecb_model

    text = (
        "KEY,TIME_PERIOD,OBS_VALUE\n"
        "MIR.M.U2.B.A2I.AM.R.A.2240.EUR.N,2026-08,3.77\nMIR.M.DE.B.A2I.AM.R.A.2240.EUR.N,2026-08,3.88\n"
    )
    monkeypatch.setattr(
        ecb_model, "get_request", lambda url, timeout: FakeResponse(text=text)
    )

    cost = ecb_model.get_corporate_borrowing_cost("2026-01-01", "2026-09-30")

    assert cost.loc[pd.Period("2026-08", "M")].to_dict() == {
        "Euro Area": pytest.approx(0.0377),
        "Germany": pytest.approx(0.0388),
    }


def test_new_methods_refuse_unknown_options():
    from financetoolkit import Economics, FixedIncome

    assert FixedIncome().get_eiopa_risk_free_rate(curve="forward").empty
    assert FixedIncome().get_bank_of_england_yield_curve(curve="corporate").empty
    assert (
        Economics(gmdb_source=False)
        .get_life_table(countries="Germany", measure="height")
        .empty
    )
    assert (
        Economics(gmdb_source=False)
        .get_life_table(countries="Germany", sex="other")
        .empty
    )
