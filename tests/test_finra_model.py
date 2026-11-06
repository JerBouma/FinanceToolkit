"""Tests for the FINRA short sale volume model and the Toolkit method built on it."""

import logging
from types import SimpleNamespace

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
