"""Offline tests for the keyless government bond yield curve sources."""

import pandas as pd
import pytest
import requests

from financetoolkit.fixedincome import (
    boc_model,
    bundesbank_model,
    norgesbank_model,
    riksbank_model,
)


class FakeResponse:
    """The parts of a requests.Response the models use."""

    def __init__(self, payload=None, text=""):
        self._payload = payload
        self.text = text
        self.content = text.encode("utf-8") if text else b"x"
        self.headers = {}

    def json(self):
        return self._payload


def test_bundesbank_curve_handles_either_separator(monkeypatch):
    for separator in (",", ";"):
        text = separator.join(["BBK_SEIS_MATURITY", "TIME_PERIOD", "OBS_VALUE"]) + "\n"
        text += separator.join(["R10XX", "2026-10-01", "3.66"]) + "\n"
        text += separator.join(["R02XX", "2026-10-01", "3.21"]) + "\n"
        text += separator.join(["R02XX", "2026-10-03", "."]) + "\n"
        monkeypatch.setattr(
            bundesbank_model,
            "get_request",
            lambda url, timeout, extra_headers, t=text: FakeResponse(text=t),
        )

        curve = bundesbank_model.get_yield_curve("2026-09-28", "2026-10-03")

        # Weekends are listed without a value and dropped; maturities are in order.
        assert list(curve.columns) == ["2Y", "10Y"]
        assert list(curve.index) == [pd.Period("2026-10-01", "D")]
        assert curve.iloc[0].tolist() == [pytest.approx(0.0321), pytest.approx(0.0366)]


def test_bank_of_canada_curve(monkeypatch):
    payload = {
        "observations": [
            {
                "d": "2026-09-30",
                "V80691344": {"v": "2.39"},
                "BD.CDN.10YR.DQ.YLD": {"v": "3.94"},
            },
            {"d": "2026-10-01", "BD.CDN.10YR.DQ.YLD": {"v": "3.96"}},
        ]
    }
    monkeypatch.setattr(
        boc_model, "get_request", lambda url, timeout: FakeResponse(payload)
    )

    curve = boc_model.get_yield_curve("2026-09-28", "2026-10-02")

    assert curve.loc[pd.Period("2026-09-30", "D"), "3M"] == pytest.approx(0.0239)
    assert curve.loc[pd.Period("2026-10-01", "D"), "10Y"] == pytest.approx(0.0396)
    assert pd.isna(curve.loc[pd.Period("2026-10-01", "D"), "3M"])


def test_norges_bank_curve(monkeypatch):
    text = "TENOR;TIME_PERIOD;OBS_VALUE\n10Y;2026-10-01;4.636\n3Y;2026-10-01;4.85\n"
    monkeypatch.setattr(
        norgesbank_model, "get_request", lambda url, timeout: FakeResponse(text=text)
    )

    curve = norgesbank_model.get_yield_curve("2026-09-28", "2026-10-02")

    assert curve.loc[pd.Period("2026-10-01", "D"), "3Y"] == pytest.approx(0.0485)


def test_riksbank_uses_groups_for_a_year_and_series_beyond(monkeypatch):
    urls = []

    def fake(url, timeout, extra_headers):
        urls.append(url)
        if "ByGroup" in url:
            return FakeResponse(
                [{"seriesId": "SEGVB10YC", "date": "2026-10-01", "value": 3.14}]
            )
        return FakeResponse([{"date": "2026-10-01", "value": 3.14}])

    monkeypatch.setattr(riksbank_model, "get_request", fake)

    recent = riksbank_model.get_yield_curve("2026-09-01", "2026-10-02")
    assert len(urls) == 2 and all("ByGroup" in url for url in urls)
    assert recent.loc[pd.Period("2026-10-01", "D"), "10Y"] == pytest.approx(0.0314)

    urls.clear()
    riksbank_model.get_yield_curve("2020-01-01", "2026-10-02")
    assert len(urls) == len(riksbank_model.YIELD_CURVE_SERIES)


def test_riksbank_waits_for_the_rate_limit(monkeypatch):
    attempts = []
    waits = []

    def fake(url, timeout, extra_headers):
        attempts.append(url)
        if len(attempts) == 1:
            response = requests.Response()
            response.status_code = 429
            response.headers["Retry-After"] = "40"
            raise requests.exceptions.HTTPError(response=response)
        return FakeResponse(
            [{"seriesId": "SEGVB10YC", "date": "2026-10-01", "value": 3.14}]
        )

    monkeypatch.setattr(riksbank_model, "get_request", fake)
    monkeypatch.setattr(riksbank_model.time, "sleep", waits.append)

    curve = riksbank_model.get_yield_curve("2026-09-01", "2026-10-02")

    assert waits == [40.0]
    assert not curve.empty


def test_curve_combines_countries_sorted_by_maturity(monkeypatch):
    from financetoolkit import FixedIncome
    from financetoolkit.fixedincome import fixedincome_controller

    days = pd.period_range("2026-09-30", "2026-10-01", freq="D")
    norway = pd.DataFrame({"10Y": [0.046, 0.0464], "3Y": [0.048, 0.0485]}, index=days)
    monkeypatch.setattr(
        fixedincome_controller.norgesbank_model,
        "get_yield_curve",
        lambda start, end: norway,
    )

    fixedincome = FixedIncome(start_date="2026-09-01", end_date="2026-10-02")
    curve = fixedincome.get_government_bond_yield_curve(countries=["Norway", "China"])

    assert list(curve.columns) == [("Norway", "3Y"), ("Norway", "10Y")]
    assert curve.loc[pd.Period("2026-10-01", "D"), ("Norway", "10Y")] == pytest.approx(
        0.0464
    )


def test_treasury_rates_fall_back_without_a_working_key(monkeypatch):
    from financetoolkit import FixedIncome
    from financetoolkit.fixedincome import fixedincome_controller

    days = pd.period_range("2026-10-01", periods=1, freq="D")
    curve = pd.DataFrame(
        {"1 Mo": [0.0417], "1.5 Month": [0.0418], "10 Yr": [0.0528]}, index=days
    )
    monkeypatch.setattr(
        fixedincome_controller.treasury_model, "get_yield_curve", lambda *args: curve
    )

    def refused(**kwargs):
        raise ValueError("Your plan does not include this endpoint")

    monkeypatch.setattr(fixedincome_controller.fmp_model, "get_treasury_rates", refused)

    for api_key in ("", "key"):
        rates = FixedIncome(
            start_date="2026-09-01", end_date="2026-10-02", api_key=api_key
        ).get_treasury_rates()

        # FinancialModelingPrep's names, without the maturities it does not publish.
        assert list(rates.columns) == ["1 Month", "10 Year"]
        assert rates.index.name == "Date"
        assert rates.iloc[0, 1] == pytest.approx(0.0528)
