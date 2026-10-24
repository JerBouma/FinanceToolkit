"""SEC Fund Prospectus Fee Model Tests"""

import io
import zipfile
from datetime import datetime

import numpy as np
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
    # Without the filings (sub.tsv) there is no prospectus date, and the period end date of
    # the facts is not reported as one.
    assert pd.isna(fees.loc["C1", "Prospectus Date"])
    assert "Fiscal Year End" not in fees.columns


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


def filings_file(rows: list[tuple], filings: dict[str, tuple]) -> bytes:
    """
    A quarterly file with several filings: num.tsv rows with their adsh, and sub.tsv with
    the filing date, acceptance time and, optionally, the prospectus date of each filing.
    """
    text = HEADER + "".join(
        f"{adsh}\t{tag}\tv\t{date}\tpure\tS1\t\t\tdoc\tClass=C1;\t0\t{value}\n"
        for adsh, tag, date, value in rows
    )
    submissions = "adsh\tform\tpdate\tfiled\taccepted\n" + "".join(
        f"{adsh}\t485BPOS\t{(rest or [''])[0]}\t{filed}\t{accepted}\n"
        for adsh, (filed, accepted, *rest) in filings.items()
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("num.tsv", text)
        archive.writestr("sub.tsv", submissions)
    return buffer.getvalue()


def test_the_latest_filing_of_a_class_is_kept():
    # A supplement filed after the annual prospectus cuts the fees of the same fiscal year.
    fees = sec_model.read_risk_return_fees(
        filings_file(
            [
                ("late", "ManagementFeesOverAssets", "20251231", "0.0055"),
                ("early", "ManagementFeesOverAssets", "20251231", "0.0074"),
                ("early", "OtherExpensesOverAssets", "20251231", "0.0010"),
            ],
            {
                "early": ("20260428", "2026-04-28 16:00:00.0"),
                "late": ("20260630", "2026-06-30 09:00:00.0"),
            },
        )
    )

    assert fees.loc["C1", "Management Fee"] == pytest.approx(0.0055)
    # Tags of an earlier filing are not mixed in.
    assert pd.isna(fees.loc["C1"].get("Other Expenses"))
    assert fees.loc["C1", "Filing Date"] == "2026-06-30"


def test_the_fee_waiver_is_negative_and_completes_the_net_ratio(monkeypatch):
    table = pd.DataFrame(
        {
            "Gross Expense Ratio": [0.0195, 0.0100],
            "Fee Waiver": [-0.002, 0.001],
            "Net Expense Ratio": [np.nan, 0.009],
            "Series": ["S1", "S2"],
            "Prospectus Date": ["2026-04-28"] * 2,
            "Filing Date": ["2026-04-28"] * 2,
        },
        index=["C1", "C2"],
    )
    monkeypatch.setattr(
        sec_model,
        "get_fund_classes",
        lambda user_agent: pd.DataFrame({"Class": ["C1", "C2"]}, index=["AAA", "BBB"]),
    )
    monkeypatch.setattr(sec_model, "_quarter_fees", lambda *args: table)

    breakdown = sec_model.get_fee_breakdown(["AAA", "BBB"], "Jane Doe jane@example.com")

    assert breakdown.loc["AAA", "Net Expense Ratio"] == pytest.approx(0.0175)
    # A waiver entered as a positive reduction is the difference of the two ratios.
    assert breakdown.loc["BBB", "Fee Waiver"] == pytest.approx(-0.001)


def test_the_prospectus_date_comes_from_the_filing():
    # The period end date of the facts (ddate) is the date of the fee table's fiscal period
    # at best; the prospectus date is the one on the prospectus cover (sub.tsv's pdate).
    fees = sec_model.read_risk_return_fees(
        filings_file(
            [("only", "ManagementFeesOverAssets", "20260630", "0.0010")],
            {"only": ("20260428", "2026-04-28 16:00:00.0", "20260501")},
        )
    )

    assert fees.loc["C1", "Prospectus Date"] == "2026-05-01"
    assert fees.loc["C1", "Filing Date"] == "2026-04-28"
    assert "Fiscal Year End" not in fees.columns


def test_a_reported_waiver_is_kept_without_a_gross_ratio(monkeypatch):
    table = pd.DataFrame(
        {
            "Gross Expense Ratio": [np.nan, 0.0100],
            "Fee Waiver": [0.002, np.nan],
            "Net Expense Ratio": [0.0050, 0.0100],
            "Series": ["S1", "S2"],
            "Prospectus Date": ["2026-04-28"] * 2,
            "Filing Date": ["2026-04-28"] * 2,
        },
        index=["C1", "C2"],
    )
    monkeypatch.setattr(
        sec_model,
        "get_fund_classes",
        lambda user_agent: pd.DataFrame({"Class": ["C1", "C2"]}, index=["AAA", "BBB"]),
    )
    monkeypatch.setattr(sec_model, "_quarter_fees", lambda *args: table)

    breakdown = sec_model.get_fee_breakdown(["AAA", "BBB"], "Jane Doe jane@example.com")

    # The waiver stays (negative) and the gross ratio is the net one minus the waiver.
    assert breakdown.loc["AAA", "Fee Waiver"] == pytest.approx(-0.002)
    assert breakdown.loc["AAA", "Gross Expense Ratio"] == pytest.approx(0.007)
    # Without a reported waiver none is made up from two equal ratios.
    assert pd.isna(breakdown.loc["BBB", "Fee Waiver"])
    assert list(breakdown.columns).index("Prospectus Date") < list(
        breakdown.columns
    ).index("Filing Date")
