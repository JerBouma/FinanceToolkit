"""SEC Fund Prospectus Fee Model Tests"""

import io
import zipfile
from datetime import datetime

import pandas as pd
import pytest
import requests

from financetoolkit.cache.cache_controller import clear_active_cache
from financetoolkit.etf import sec_model

# pylint: disable=missing-function-docstring

HEADER = "adsh\ttag\tversion\tddate\tuom\tseries\tclass\tmeasure\tdocument\totherdims\tiprx\tvalue\n"


def quarter_file(rows: list[tuple]) -> bytes:
    text = HEADER + "".join(
        f"x\t{tag}\tv\t{date}\tpure\t{series}\t\t\tdoc\t{dims}\t0\t{value}\n"
        for tag, date, series, dims, value in rows
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("num.tsv", text)
    return buffer.getvalue()


class Response:
    def __init__(self, content=None, data=None, status=200):
        self.content, self._data, self.status_code = content, data, status

    def json(self):
        return self._data


@pytest.fixture(autouse=True)
def isolate_from_the_active_cache():
    clear_active_cache()
    yield
    clear_active_cache()


def test_the_fee_table_is_read_per_share_class():
    fees = sec_model.read_risk_return_fees(
        quarter_file(
            [
                ("ManagementFeesOverAssets", "20251231", "S1", "Class=C1;", "0.0003"),
                ("ExpensesOverAssets", "20251231", "S1", "Class=C1;", "0.0004"),
                ("ExpensesOverAssets", "20241231", "S1", "Class=C1;", "0.0009"),
                ("PortfolioTurnoverRate", "20251231", "S1", "", "0.0300"),
                ("AvgAnnlRtrPct", "20251231", "S1", "Class=C1;", "0.1"),
            ]
        )
    )

    assert fees.loc["C1", "Management Fee"] == pytest.approx(0.0003)
    # The latest value of a tag is kept.
    assert fees.loc["C1", "Gross Expense Ratio"] == pytest.approx(0.0004)
    assert fees.loc["C1", "Portfolio Turnover"] == pytest.approx(0.03)
    assert fees.loc["C1", "Prospectus Date"] == "2025-12-31"


def test_a_user_agent_with_an_email_is_required(monkeypatch):
    monkeypatch.delenv(sec_model.USER_AGENT_ENVIRONMENT_VARIABLE, raising=False)
    with pytest.raises(ValueError, match="email"):
        sec_model.resolve_user_agent()
    assert (
        sec_model.resolve_user_agent("Jane Doe jane@example.com")
        == "Jane Doe jane@example.com"
    )


def test_the_quarters_are_read_back_until_every_fund_is_found(monkeypatch):
    requested = []

    def fake(url, timeout=60, extra_headers=None):
        requested.append(url)
        assert "@" in extra_headers["User-Agent"]
        if url.endswith("company_tickers_mf.json"):
            return Response(
                data={
                    "fields": ["cik", "seriesId", "classId", "symbol"],
                    "data": [[1, "S1", "C1", "AAA"], [2, "S2", "C2", "BBB"]],
                }
            )
        if "2026q2" in url:
            return Response(
                content=quarter_file(
                    [("ExpensesOverAssets", "20260331", "S1", "Class=C1;", "0.0005")]
                )
            )
        if "2026q1" in url:
            return Response(
                content=quarter_file(
                    [("ExpensesOverAssets", "20251231", "S2", "Class=C2;", "0.0010")]
                )
            )
        error = requests.exceptions.HTTPError("404")
        error.response = Response(status=404)
        raise error

    monkeypatch.setattr(sec_model, "get_request", fake)

    fees = sec_model.get_fee_breakdown(
        ["AAA", "BBB", "SPY"], "Jane Doe jane@example.com", today=datetime(2026, 8, 1)
    )

    assert list(fees.index) == ["AAA", "BBB"]
    assert fees.loc["BBB", "Gross Expense Ratio"] == pytest.approx(0.001)
    # Without a waiver the net expense ratio is the gross one.
    assert fees.loc["AAA", "Net Expense Ratio"] == pytest.approx(0.0005)
    assert any("2026q1" in url for url in requested)
    assert isinstance(fees, pd.DataFrame)
