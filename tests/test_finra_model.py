"""Tests for the FINRA short sale volume model and the Toolkit method built on it."""

import json
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests

from financetoolkit import finra_model
from financetoolkit.toolkit_controller import Toolkit

# Two trading days in FINRA's format: pipe delimited, fractional volumes and a closing line
# with the record count.
DAY_FILES = {
    "20260302": (
        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
        "20260302|AAPL|400.5|10|1000.25|B,Q,N\n"
        "20260302|BRK/B|50|0|200|B,Q,N\n"
        "20260302|GME|90|5|100|B,Q,N\n"
        "3\n"
    ),
    "20260303": (
        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
        "20260303|AAPL|600|20|1000|B,Q,N\n"
        "20260303|BRK/B|75|1|300|B,Q,N\n"
        "2\n"
    ),
}


class FakeResponse:
    def __init__(self, text):
        self.text = text


def fake_finra(requested, failing=()):
    """A get_request that serves DAY_FILES, answers 403 for other days and fails for some."""

    def get_request(url, timeout=60):  # noqa: ARG001
        day = url.rsplit("CNMSshvol", 1)[1].removesuffix(".txt")
        requested.append(day)

        if day in failing:
            raise requests.exceptions.ConnectionError("offline")

        if day not in DAY_FILES:
            response = requests.Response()
            response.status_code = 403
            raise requests.exceptions.HTTPError(response=response)

        return FakeResponse(DAY_FILES[day])

    return get_request


@pytest.fixture(autouse=True)
def no_cache(monkeypatch):
    monkeypatch.setattr(finra_model, "get_active_cache", lambda: None)


def test_a_daily_file_is_read_without_its_closing_record_count():
    volumes = finra_model.read_daily_file(DAY_FILES["20260302"])

    assert volumes.index.tolist() == ["AAPL", "BRK/B", "GME"]
    assert volumes.loc["AAPL"].tolist() == [400.5, 10.0, 1000.25]
    assert volumes.columns.tolist() == [
        "Short Volume",
        "Short Exempt Volume",
        "Total Volume",
    ]


def test_a_file_without_the_expected_columns_is_a_format_change():
    with pytest.raises(ValueError, match="TotalVolume"):
        finra_model.read_daily_file("Date|Symbol|ShortVolume\n20260302|AAPL|1\n")


@pytest.mark.parametrize(
    "ticker, symbol",
    [("AAPL", "AAPL"), ("BRK-B", "BRK/B"), ("ASML.AS", None), ("^GSPC", None)],
)
def test_tickers_are_written_as_finra_reports_them(ticker, symbol):
    assert finra_model.to_finra_symbol(ticker) == symbol


def test_the_z_score_compares_a_day_with_the_days_before_it():
    volumes = pd.DataFrame(
        {"Short Volume": [40.0, 50.0, 60.0, 90.0], "Total Volume": 100.0},
        index=pd.period_range("2026-03-02", periods=4, freq="D"),
    )

    statistics = finra_model.get_short_volume_ratio_statistics(volumes, window=3)

    assert statistics["Short Volume Ratio"].tolist() == [0.4, 0.5, 0.6, 0.9]
    # The first three days have fewer than three earlier days to compare with.
    assert statistics["Short Volume Ratio Z-Score"].iloc[:3].isna().all()
    # The fourth day against 0.4, 0.5 and 0.6: mean 0.5, standard deviation 0.1.
    assert statistics["Short Volume Ratio Z-Score"].iloc[3] == pytest.approx(4.0)


def test_each_day_is_downloaded_once_for_all_symbols(monkeypatch):
    requested = []
    monkeypatch.setattr(finra_model, "get_request", fake_finra(requested))

    volumes = finra_model.collect_short_volume(
        ["AAPL", "BRK/B", "NOTATICKER"], "2026-03-02", "2026-03-04"
    )

    # Three weekdays, the last without a file (as on a holiday).
    assert sorted(requested) == ["20260302", "20260303", "20260304"]
    assert volumes["AAPL"]["Short Volume"].tolist() == [400.5, 600.0]
    assert volumes["BRK/B"].index.astype(str).tolist() == ["2026-03-02", "2026-03-03"]
    assert volumes["NOTATICKER"].empty


def test_weekends_and_dates_before_the_first_file_are_not_requested(monkeypatch):
    requested = []
    monkeypatch.setattr(finra_model, "get_request", fake_finra(requested))

    finra_model.collect_short_volume(["AAPL"], "2018-07-28", "2018-08-05")

    assert sorted(requested) == ["20180801", "20180802", "20180803"]


def test_a_day_that_cannot_be_retrieved_is_left_out_and_not_cached(monkeypatch, caplog):
    stored = []
    cache = SimpleNamespace(
        plan=lambda **kwargs: SimpleNamespace(
            cached_frame=lambda symbol: None,
            get_fetch_span=lambda symbol: (
                pd.Timestamp("2026-03-02"),
                pd.Timestamp("2026-03-03"),
            ),
        ),
        store=lambda **kwargs: stored.append(kwargs),
    )
    monkeypatch.setattr(finra_model, "get_active_cache", lambda: cache)
    monkeypatch.setattr(
        finra_model, "get_request", fake_finra([], failing=("20260303",))
    )

    with caplog.at_level(logging.WARNING):
        volumes = finra_model.collect_short_volume(["AAPL"], "2026-03-02", "2026-03-03")

    assert volumes["AAPL"]["Short Volume"].tolist() == [400.5]
    assert stored == []
    assert "1 of 2 days" in caplog.text


def test_the_fetched_days_are_cached_per_symbol(monkeypatch):
    stored = []
    cache = SimpleNamespace(
        plan=lambda **kwargs: SimpleNamespace(
            cached_frame=lambda symbol: None,
            get_fetch_span=lambda symbol: (
                pd.Timestamp("2026-03-02"),
                pd.Timestamp("2026-03-03"),
            ),
        ),
        store=lambda **kwargs: stored.append(kwargs),
    )
    monkeypatch.setattr(finra_model, "get_active_cache", lambda: cache)
    monkeypatch.setattr(finra_model, "get_request", fake_finra([]))

    finra_model.collect_short_volume(["AAPL", "NOTATICKER"], "2026-03-02", "2026-03-03")

    assert [entry["entity"] for entry in stored] == ["AAPL", "NOTATICKER"]
    # A symbol without short sales is cached as answered, so it is not asked again.
    assert stored[1]["data"].empty
    assert (stored[0]["start"], stored[0]["end"]) == ("2026-03-02", "2026-03-03")


def toolkit(tickers, start_date="2026-01-01", end_date="2026-03-03"):
    return SimpleNamespace(
        _tickers=tickers, _start_date=start_date, _end_date=end_date, _rounding=4
    )


def test_the_toolkit_gives_each_ticker_its_short_volume(monkeypatch, caplog):
    requested = []
    monkeypatch.setattr(finra_model, "get_request", fake_finra(requested))

    with caplog.at_level(logging.WARNING):
        short_volume = Toolkit.get_short_volume(
            toolkit(["AAPL", "BRK-B", "ASML.AS"]), history_days=2, window=1
        )

    # Two calendar days before the end date: the 1st (a Sunday) to the 3rd.
    assert sorted(requested) == ["20260302", "20260303"]
    assert short_volume.columns.get_level_values(1).unique().tolist() == [
        "AAPL",
        "BRK-B",
    ]
    assert short_volume[("Short Volume Ratio", "BRK-B")].tolist() == [0.25, 0.25]
    assert short_volume[("Short Volume Ratio", "AAPL")].iloc[0] == pytest.approx(0.4004)
    assert "ASML.AS" in caplog.text


def test_the_history_never_starts_before_the_start_date(monkeypatch):
    requested = []
    monkeypatch.setattr(finra_model, "get_request", fake_finra(requested))

    short_volume = Toolkit.get_short_volume(
        toolkit(["AAPL"], start_date="2026-03-03"), show_columns=["Short Volume"]
    )

    assert requested == ["20260303"]
    assert short_volume.columns.tolist() == ["Short Volume"]
    assert short_volume["Short Volume"].tolist() == [600.0]


def test_a_toolkit_without_covered_tickers_returns_nothing(monkeypatch):
    monkeypatch.setattr(finra_model, "get_request", fake_finra([]))

    assert Toolkit.get_short_volume(toolkit(["ASML.AS"])).empty


# Two settlement dates of FINRA's short interest as its Query API returns them in JSON; the
# GME figures and FINRA's own days to cover (6.87 and 4.29) are those it published.
SHORT_INTEREST_RECORDS = [
    {
        "symbolCode": "GME",
        "settlementDate": "2026-11-13",
        "currentShortPositionQuantity": 41054446,
        "averageDailyVolumeQuantity": 5977827,
        "daysToCoverQuantity": 6.87,
    },
    {
        "symbolCode": "GME",
        "settlementDate": "2026-09-15",
        "currentShortPositionQuantity": 39069126,
        "averageDailyVolumeQuantity": 9105968,
        "daysToCoverQuantity": 4.29,
    },
    {
        "symbolCode": "BRKB",
        "settlementDate": "2026-11-13",
        "currentShortPositionQuantity": 16584564,
        "averageDailyVolumeQuantity": 0,
        "daysToCoverQuantity": 0,
    },
]


class FakeJsonResponse:
    def __init__(self, records, total):
        self.records = records
        self.text = "[]" if records else ""
        self.headers = {"record-total": str(total)}

    def json(self):
        return self.records


def fake_short_interest(requests_made, records=SHORT_INTEREST_RECORDS, page=None):
    """A get_request that answers the short interest query with records, page by page."""

    def get_request(url, timeout=60, extra_headers=None, data=None):  # noqa: ARG001
        payload = json.loads(data)
        requests_made.append(payload)
        size = page or payload["limit"]
        start = payload["offset"] // payload["limit"] * size
        return FakeJsonResponse(records[start : start + size], len(records))

    return get_request


@pytest.mark.parametrize(
    "ticker, symbol", [("AAPL", "AAPL"), ("BRK-B", "BRKB"), ("ASML.AS", None)]
)
def test_short_interest_symbols_have_no_share_class_separator(ticker, symbol):
    assert finra_model.to_finra_symbol(ticker, separator="") == symbol


def test_the_short_interest_records_are_read_per_symbol():
    positions = finra_model.read_short_interest(SHORT_INTEREST_RECORDS)

    assert sorted(positions) == ["BRKB", "GME"]
    assert positions["GME"].index.astype(str).tolist() == ["2026-09-15", "2026-11-13"]
    assert positions["GME"]["Short Interest"].tolist() == [39069126.0, 41054446.0]
    assert finra_model.read_short_interest([]) == {}

    with pytest.raises(ValueError, match="averageDailyVolumeQuantity"):
        finra_model.read_short_interest([{"symbolCode": "GME"}])


def test_the_days_to_cover_match_finras_own_figures():
    positions = finra_model.read_short_interest(SHORT_INTEREST_RECORDS)["GME"]

    days_to_cover = finra_model.get_days_to_cover(
        positions["Short Interest"], positions["Average Daily Volume"]
    )

    # FINRA rounds its days to cover to two decimals.
    assert days_to_cover.round(2).tolist() == [4.29, 6.87]
    assert np.isnan(finra_model.get_days_to_cover(100.0, 0.0))
    assert finra_model.get_short_interest_to_float(25.0, 100.0) == 0.25
    assert np.isnan(finra_model.get_short_interest_to_float(25.0, 0.0))


def test_the_short_interest_is_requested_once_for_all_symbols(monkeypatch):
    requests_made = []
    monkeypatch.setattr(finra_model, "get_request", fake_short_interest(requests_made))

    positions = finra_model.collect_short_interest(
        ["GME", "BRKB", "NOTATICKER"], "2026-09-01", "2026-11-27"
    )

    assert len(requests_made) == 1
    assert requests_made[0]["domainFilters"][0]["values"] == [
        "GME",
        "BRKB",
        "NOTATICKER",
    ]
    assert requests_made[0]["dateRangeFilters"][0]["startDate"] == "2026-09-01"
    assert positions["NOTATICKER"].empty
    assert positions["GME"]["Short Interest"].tolist() == [39069126.0, 41054446.0]


def test_the_short_interest_is_read_page_by_page(monkeypatch):
    requests_made = []
    monkeypatch.setattr(
        finra_model, "get_request", fake_short_interest(requests_made, page=2)
    )
    monkeypatch.setattr(finra_model, "SHORT_INTEREST_PAGE", 2)

    positions = finra_model.collect_short_interest(
        ["GME", "BRKB"], "2026-09-01", "2026-11-27"
    )

    assert [payload["offset"] for payload in requests_made] == [0, 2]
    assert len(positions["GME"]) == 2
    assert len(positions["BRKB"]) == 1


def test_short_interest_that_cannot_be_retrieved_is_not_cached(monkeypatch, caplog):
    stored = []
    cache = SimpleNamespace(
        plan=lambda **kwargs: SimpleNamespace(
            cached_frame=lambda symbol: None,
            get_fetch_span=lambda symbol: (
                pd.Timestamp("2026-09-01"),
                pd.Timestamp("2026-11-27"),
            ),
        ),
        store=lambda **kwargs: stored.append(kwargs),
    )

    def offline(*args, **kwargs):
        raise requests.exceptions.ConnectionError("offline")

    monkeypatch.setattr(finra_model, "get_active_cache", lambda: cache)
    monkeypatch.setattr(finra_model, "get_request", offline)

    with caplog.at_level(logging.WARNING):
        positions = finra_model.collect_short_interest(
            ["GME"], "2026-09-01", "2026-11-27"
        )

    assert positions["GME"].empty
    assert stored == []
    assert "short interest" in caplog.text


def test_the_toolkit_gives_each_ticker_its_short_interest(monkeypatch, caplog):
    monkeypatch.setattr(finra_model, "get_request", fake_short_interest([]))
    instance = SimpleNamespace(
        _tickers=["GME", "BRK-B", "ASML.AS"],
        _start_date="2026-09-01",
        _end_date="2026-11-27",
        _rounding=4,
        _api_key="KEY",
        get_shares_float=lambda: pd.DataFrame(
            {"GME": [400_000_000.0], "BRK-B": [1_300_000_000.0]},
            index=["Float Shares"],
        ),
    )

    with caplog.at_level(logging.WARNING):
        short_interest = Toolkit.get_short_interest(instance)

    assert short_interest.columns.get_level_values(0).unique().tolist() == [
        "Short Interest",
        "Average Daily Volume",
        "Days to Cover",
        "Short Interest to Float",
    ]
    november = short_interest.loc[pd.Period("2026-11-13", freq="D")]
    assert november[("Days to Cover", "GME")] == pytest.approx(6.8678)
    assert november[("Short Interest to Float", "GME")] == pytest.approx(0.1026)
    # A cycle without volume has no days to cover.
    assert np.isnan(november[("Days to Cover", "BRK-B")])
    assert "ASML.AS" in caplog.text


def test_the_short_interest_to_float_is_empty_without_an_api_key(monkeypatch):
    monkeypatch.setattr(finra_model, "get_request", fake_short_interest([]))
    instance = SimpleNamespace(
        _tickers=["GME"],
        _start_date="2026-09-01",
        _end_date="2026-11-27",
        _rounding=4,
        _api_key="",
    )

    short_interest = Toolkit.get_short_interest(instance)

    assert short_interest["Short Interest to Float"].isna().all()
    assert short_interest["Days to Cover"].tolist() == [4.2905, 6.8678]
