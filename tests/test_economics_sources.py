"""Offline tests for the keyless high-frequency Economics sources."""

import importlib
import json

import pandas as pd
import pytest
import requests

from financetoolkit.cache.cache_controller import (
    Cache,
    clear_active_cache,
    set_active_cache,
)
from financetoolkit.economics import helpers
from financetoolkit.economics.sources import (
    bis_model,
    boe_model,
    boj_model,
    eurostat_model,
    gmdb_model,
    mof_model,
    oecd_model,
    ons_model,
    sbj_model,
    treasury_model,
)
from financetoolkit.utilities.error_model import InvalidArgumentError


@pytest.fixture(autouse=True)
def isolate_from_the_active_cache():
    """Start every test without an active cache, as another test (e.g. a Toolkit) may
    have left one behind in the same worker, which would answer these fetches instead.
    """
    clear_active_cache()
    yield
    clear_active_cache()


class FakeResponse:
    """The parts of a requests.Response the models use."""

    def __init__(self, payload=None, text="", content_type="text/csv"):
        self._payload = payload
        self.text = text
        self.content = text.encode("cp932" if content_type == "cp932" else "utf-8")
        self.headers = {"Content-Type": content_type}

    def json(self):
        return self._payload


@pytest.fixture(name="cache")
def fixture_cache(tmp_path):
    cache = Cache(tmp_path / "cache")
    set_active_cache(cache)
    yield cache
    clear_active_cache()


def json_stat(geo_codes, months, values, extra_size=1):
    """Builds a JSON-stat response with a unit dimension, geo and time."""
    return {
        "id": ["unit", "geo", "time"],
        "size": [extra_size, len(geo_codes), len(months)],
        "dimension": {
            "unit": {"category": {"index": {f"U{i}": i for i in range(extra_size)}}},
            "geo": {
                "category": {"index": {code: i for i, code in enumerate(geo_codes)}}
            },
            "time": {
                "category": {"index": {month: i for i, month in enumerate(months)}}
            },
        },
        "value": {str(i): value for i, value in enumerate(values) if value is not None},
    }


def test_json_stat_names_countries_and_uses_the_current_euro_area():
    response = json_stat(
        ["DE", "EA19", "EA21", "EU", "EU27_2020", "UK"],
        ["2026-08", "2026-09"],
        [2.9, 3.3, 3.0, None, 3.2, 3.8, 9.9, 9.9, 3.1, 3.5, 2.0, None],
    )

    data = eurostat_model.parse_json_stat(response, "test")

    # EA21 is the larger, current composition; "EU" is skipped in favour of EU27_2020.
    assert data.loc[pd.Period("2026-09", "M"), "Euro Area"] == 3.8
    assert sorted(data.columns) == [
        "Euro Area",
        "European Union",
        "Germany",
        "United Kingdom",
    ]
    assert data.index.freqstr == "M"


def test_json_stat_refuses_more_than_one_series_per_country():
    response = json_stat(["DE"], ["2026-09"], [1.0, 2.0], extra_size=2)

    with pytest.raises(ValueError, match="more than one series"):
        eurostat_model.parse_json_stat(response, "test")


def test_json_stat_reports_a_changed_format():
    with pytest.raises(ValueError, match="JSON-stat"):
        eurostat_model.parse_json_stat({"label": "something else"}, "test")


def test_freshest_source_wins_and_ties_follow_the_order():
    months = pd.period_range("2026-06", "2026-09", freq="M")
    eurostat = pd.DataFrame(
        {"Germany": [1.0, 1.0, 1.0, 1.0], "United Kingdom": [5.0, None, None, None]},
        index=months,
    )
    ons = pd.DataFrame({"United Kingdom": [2.0, 2.0, 2.0, None]}, index=months)
    oecd = pd.DataFrame(
        {"Germany": [3.0, 3.0, 3.0, 3.0], "Japan": [4.0] * 4}, index=months
    )

    combined = helpers.combine_sources([eurostat, ons, oecd])

    # Germany is equally recent in both, so the first source wins; the UK column Eurostat
    # stopped updating loses to the ONS.
    assert combined["Germany"].tolist() == [1.0] * 4
    assert combined["United Kingdom"].tolist()[:3] == [2.0] * 3
    assert combined["Japan"].tolist() == [4.0] * 4


def test_resampling_takes_the_last_value_of_each_period():
    days = pd.period_range("2026-09-28", "2026-10-06", freq="D")
    daily = pd.DataFrame({"Euro Area": range(len(days))}, index=days, dtype=float)

    weekly = helpers.resample_to_period(daily, "weekly")
    monthly = helpers.resample_to_period(daily, "monthly")

    # Weeks end on Friday: 2026-10-02 is the fifth day.
    assert weekly.iloc[0, 0] == 4
    assert monthly.loc[pd.Period("2026-09", "M"), "Euro Area"] == 2
    assert monthly.loc[pd.Period("2026-10", "M"), "Euro Area"] == 8


def test_periods_are_validated_with_annual_as_an_alias():
    assert (
        helpers.validate_period("Annual", ["monthly", "yearly"], "inflation rate")
        == "yearly"
    )

    with pytest.raises(ValueError, match="period='weekly'"):
        helpers.validate_period("weekly", ["monthly", "yearly"], "inflation rate")


def test_an_unreachable_source_serves_the_cached_copy(cache, monkeypatch):
    data = pd.DataFrame(
        {"Japan": [1.0]}, index=pd.period_range("2026-09", periods=1, freq="M")
    )
    helpers.collect_cached_data("ONS", "series", "x", lambda: data, "test")

    # Expire the entry, then make the source unreachable.
    monkeypatch.setattr(helpers, "STALE_TTL_SECONDS", 10**9)
    monkeypatch.setattr(cache, "get", _expire_fresh_reads(cache.get))

    def unreachable():
        raise requests.exceptions.ConnectionError("down")

    served = helpers.collect_cached_data("ONS", "series", "x", unreachable, "test")

    assert served.equals(data)


def _expire_fresh_reads(get):
    """Treats every read with the default time-to-live as expired."""

    def wrapper(*args, ttl=None, **kwargs):
        return None if ttl is None else get(*args, ttl=ttl, **kwargs)

    return wrapper


def test_a_changed_format_is_not_answered_with_cached_data(cache):
    def changed():
        raise ValueError("format changed")

    with pytest.raises(ValueError, match="format changed"):
        helpers.collect_cached_data("ONS", "series", "y", changed, "test")


def test_ranged_requests_only_ask_for_what_is_missing(cache, monkeypatch):
    requested = []

    def fetch(start, end):
        requested.append((start, end))
        days = pd.period_range(start, end, freq="D")
        return pd.DataFrame({"Japan": 1.0}, index=days)

    helpers.collect_ranged_data(
        "BankOfJapan", "series", "x", fetch, "2026-09-01", "2026-09-30", "test"
    )
    again = helpers.collect_ranged_data(
        "BankOfJapan", "series", "x", fetch, "2026-09-10", "2026-09-20", "test"
    )

    # The second, narrower range is fully cached, so it makes no request.
    assert requested == [("2026-09-01", "2026-09-30")]
    assert len(again) == 11


def test_a_gateway_timeout_is_retried(monkeypatch):
    monkeypatch.setattr(helpers, "RETRY_DELAY_SECONDS", 0)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            response = requests.Response()
            response.status_code = 504
            raise requests.exceptions.HTTPError(response=response)
        return pd.DataFrame({"Japan": [1.0]})

    assert not helpers.collect_cached_data("BoJ", "series", "z", flaky, "test").empty
    assert len(attempts) == 2


def test_a_rate_limit_is_not_retried(monkeypatch):
    monkeypatch.setattr(helpers, "RETRY_DELAY_SECONDS", 0)
    attempts = []

    def limited():
        attempts.append(1)
        response = requests.Response()
        response.status_code = 429
        raise requests.exceptions.HTTPError(response=response)

    assert helpers.collect_cached_data("BoJ", "series", "w", limited, "test").empty
    assert len(attempts) == 1


def test_ons_months_are_parsed_into_decimals(monkeypatch):
    payload = {
        "months": [
            {"date": "2026 JUL", "value": "2.9"},
            {"date": "2026 AUG", "value": "3.1"},
        ]
    }
    monkeypatch.setattr(
        ons_model, "get_request", lambda url, timeout: FakeResponse(payload)
    )

    inflation = ons_model.get_inflation_rate()

    assert inflation.loc[pd.Period("2026-08", "M"), "United Kingdom"] == pytest.approx(
        0.031
    )


def test_ons_without_monthly_observations_is_a_changed_format(monkeypatch):
    monkeypatch.setattr(
        ons_model, "get_request", lambda url, timeout: FakeResponse({"years": []})
    )

    with pytest.raises(ValueError, match="changed its website"):
        ons_model.get_unemployment_rate()


def test_bank_of_england_web_page_is_a_rejected_request(monkeypatch):
    monkeypatch.setattr(
        boe_model,
        "get_request",
        lambda url, timeout, extra_headers: FakeResponse(
            text="<html>", content_type="text/html"
        ),
    )

    with pytest.raises(ValueError, match="web page"):
        boe_model.get_overnight_rate("2026-09-01", "2026-09-30")


def test_bank_of_england_requests_the_range_from_1990_at_the_earliest(monkeypatch):
    urls = []

    def fake(url, timeout, extra_headers):
        urls.append(url)
        return FakeResponse(
            text="DATE,IUDSOIA\n01 Sep 2026,3.73\n", content_type="application/csv"
        )

    monkeypatch.setattr(boe_model, "get_request", fake)

    rate = boe_model.get_overnight_rate("1950-01-01", "2026-09-30")

    assert "Datefrom=01/Jan/1990&Dateto=30/Sep/2026" in urls[0]
    assert rate.iloc[0, 0] == pytest.approx(0.0373)


def test_bank_of_japan_errors_are_reported(monkeypatch):
    monkeypatch.setattr(
        boj_model,
        "get_request",
        lambda url, timeout: FakeResponse({"STATUS": 400, "MESSAGE": "Invalid code"}),
    )

    with pytest.raises(ValueError, match="Invalid code"):
        boj_model.get_overnight_rate("2026-09-01", "2026-09-30")


def test_bank_of_japan_skips_days_without_a_value(monkeypatch):
    payload = {
        "STATUS": 200,
        "RESULTSET": [
            {"VALUES": {"SURVEY_DATES": [20260926, 20260929], "VALUES": [None, 1.226]}}
        ],
    }
    monkeypatch.setattr(
        boj_model, "get_request", lambda url, timeout: FakeResponse(payload)
    )

    rate = boj_model.get_overnight_rate("2026-09-01", "2026-09-30")

    assert rate["Japan"].tolist() == [pytest.approx(0.01226)]


def test_bis_rates_are_named_and_in_decimals(monkeypatch):
    text = "FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE\nD,US,2026-09-29,3.875\nD,XM,2026-09-29,2.5\nD,1X,2026-09-29,9\n"
    monkeypatch.setattr(
        bis_model, "get_request", lambda url, timeout: FakeResponse(text=text)
    )

    rates = bis_model.get_central_bank_policy_rate("2026-09-01", "2026-09-30")

    assert rates.iloc[0].to_dict() == {"Euro Area": 0.025, "United States": 0.03875}


def test_jgb_files_are_combined_and_dashes_are_missing(monkeypatch):
    history = "Interest Rate,,(Unit : %)\nDate,1Y,10Y,40Y\n1974/9/24,10.327,8.127,-\n2026/9/30,1.684,3.057,4.099\n"
    current = (
        "Interest Rate (October 2026),,(Unit : %)\nDate,1Y,10Y,40Y\n2026/10/1,1.668,3.092,4.125\n,,,\n"
        '"If you cannot download the latest csv data, please clear the browser cache"\n'
    )
    monkeypatch.setattr(
        mof_model,
        "get_request",
        lambda url, timeout: FakeResponse(
            text=history if "historical" in url else current
        ),
    )

    yields = mof_model.get_government_bond_yields()

    assert yields.index[-1] == pd.Period("2026-10-01", "D")
    assert yields.loc[pd.Period("2026-10-01", "D"), "10Y"] == pytest.approx(0.03092)
    assert pd.isna(yields.loc[pd.Period("1974-09-24", "D"), "40Y"])


def test_japanese_cpi_bases_are_chained(monkeypatch):
    def base_file(base_year, rows):
        header = "類・品目,総合\nGroup/Item,All items\n"
        return header + "".join(f"{month},{value}\n" for month, value in rows)

    files = {
        2020: base_file(
            2020, [("202412", 110.0), ("202501", 111.0), ("202601", 113.22)]
        ),
        2025: base_file(2025, [("202501", 100.0), ("202601", 102.0)]),
    }

    def fake(url, timeout):
        if url.endswith("1.html"):
            return FakeResponse(text='<a href="/data/cpi/2025/csv/zmi2025aa.csv">')
        for year, text in files.items():
            if f"zmi{year}aa.csv" in url:
                return FakeResponse(text=text, content_type="cp932")
        response = requests.Response()
        response.status_code = 404
        raise requests.exceptions.HTTPError(response=response)

    monkeypatch.setattr(sbj_model, "get_request", fake)

    index = sbj_model.get_consumer_price_index()["Japan"]

    # December 2024 is rescaled onto the 2025 base: 110 x (100 / 111) on average.
    assert index.loc[pd.Period("2025-01", "M")] == 100.0
    assert index.loc[pd.Period("2024-12", "M")] == pytest.approx(99.1, abs=0.1)


def test_japanese_cpi_without_all_items_is_a_changed_layout(monkeypatch):
    monkeypatch.setattr(
        sbj_model,
        "get_request",
        lambda url, timeout: FakeResponse(
            text="Group/Item,Food\n202501,1\n", content_type="cp932"
        ),
    )

    with pytest.raises(ValueError, match="All items"):
        sbj_model.get_consumer_price_index()


def test_treasury_requests_only_the_years_in_the_range(monkeypatch):
    years = []

    def fake(url, timeout):
        year = url.split("daily-treasury-rates.csv/")[1][:4]
        years.append(year)
        return FakeResponse(text=f'Date,"10 Yr"\n01/02/{year},4.00\n')

    monkeypatch.setattr(treasury_model, "get_request", fake)

    yields = treasury_model.get_long_term_interest_rate("2024-03-01", "2025-06-30")

    assert sorted(years) == ["2024", "2025"]
    assert yields["United States"].tolist() == [0.04, 0.04]


def test_a_faster_publisher_extends_matching_history():
    months = pd.period_range("2026-05", "2026-08", freq="M")
    oecd = pd.DataFrame(
        {"United States": [0.042, 0.041, 0.041, 0.041], "Japan": 0.025}, index=months
    )
    bls = pd.DataFrame(
        {"United States": [0.041, 0.041, 0.042]},
        index=pd.period_range("2026-07", "2026-09", freq="M"),
    )

    extended = helpers.extend_with_recent(oecd, bls, "United States")

    assert extended.loc[pd.Period("2026-09", "M"), "United States"] == 0.042
    assert pd.isna(extended.loc[pd.Period("2026-09", "M"), "Japan"])


def test_a_different_series_does_not_extend_the_history():
    months = pd.period_range("2026-07", "2026-08", freq="M")
    oecd = pd.DataFrame({"United States": [0.041, 0.041]}, index=months)
    other = pd.DataFrame(
        {"United States": [0.05, 0.05, 0.05]},
        index=pd.period_range("2026-07", "2026-09", freq="M"),
    )

    assert helpers.extend_with_recent(oecd, other, "United States").equals(oecd)


def test_bls_unemployment_is_parsed_into_decimals(monkeypatch):
    from financetoolkit.economics.sources import bls_model

    payload = {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [
                {
                    "seriesID": "LNS14000000",
                    "data": [
                        {"year": "2026", "period": "M09", "value": "4.2"},
                        {"year": "2026", "period": "M13", "value": "4.1"},
                        {"year": "2026", "period": "M08", "value": "4.1"},
                    ],
                }
            ]
        },
    }
    monkeypatch.setattr(
        bls_model, "get_request", lambda url, **kwargs: FakeResponse(payload)
    )

    rate = bls_model.get_unemployment_rate()

    # M13 is an annual average, not a month.
    assert rate["United States"].tolist() == [
        pytest.approx(0.041),
        pytest.approx(0.042),
    ]


def test_a_refused_bls_request_is_not_a_format_change(monkeypatch):
    from financetoolkit.economics.sources import bls_model

    payload = {
        "status": "REQUEST_NOT_PROCESSED",
        "message": ["Daily threshold reached."],
    }
    monkeypatch.setattr(
        bls_model, "get_request", lambda url, **kwargs: FakeResponse(payload)
    )

    # Treated like an unreachable source: no data (or the cached copy), not an error.
    assert bls_model.get_unemployment_rate().empty


def test_an_unreachable_bls_is_logged_as_the_oecd_fallback(monkeypatch, caplog):
    import logging  # noqa: PLC0415

    from financetoolkit.economics import helpers  # noqa: PLC0415
    from financetoolkit.economics.sources import bls_model  # noqa: PLC0415

    def busy(url, **kwargs):
        response = requests.Response()
        response.status_code = 503
        raise requests.HTTPError("503 Server Error", response=response)

    monkeypatch.setattr(bls_model, "get_request", busy)
    monkeypatch.setattr(helpers, "RETRY_DELAY_SECONDS", 0)

    with caplog.at_level(logging.INFO, logger="financetoolkit"):
        assert bls_model.get_unemployment_rate().empty

    messages = [record for record in caplog.records if "BLS" in record.getMessage()]
    assert messages and all(record.levelno == logging.INFO for record in messages)
    assert "the OECD's" in messages[-1].getMessage()


def test_ons_quarters_are_parsed(monkeypatch):
    payload = {
        "quarters": [
            {"date": "2026 Q1", "value": "0.6"},
            {"date": "2026 Q2", "value": "0.5"},
        ]
    }
    monkeypatch.setattr(
        ons_model, "get_request", lambda url, timeout: FakeResponse(payload)
    )

    growth = ons_model.get_gross_domestic_product_growth()

    assert growth.index.freqstr.startswith("Q")
    assert growth.index.name is None
    assert growth.loc[pd.Period("2026Q2", "Q"), "United Kingdom"] == pytest.approx(
        0.005
    )


def test_eurostat_quarters_are_requested_and_parsed(monkeypatch):
    urls = []

    def fake(url, timeout):
        urls.append(url)
        return FakeResponse(json_stat(["EA"], ["2026-Q1", "2026-Q2"], [0.0, 0.6]))

    monkeypatch.setattr(eurostat_model, "get_request", fake)

    growth = eurostat_model.get_gross_domestic_product_growth(
        "2025-11-15", "2026-09-30"
    )

    assert "sinceTimePeriod=2025-Q4&untilTimePeriod=2026-Q3" in urls[0]
    assert growth.loc[pd.Period("2026Q2", "Q"), "Euro Area"] == pytest.approx(0.006)


def test_freddie_mac_mortgage_rates_match_freds_layout(monkeypatch):
    from financetoolkit.economics.sources import freddie_mac_model

    text = "date,pmms30,pmms30p,pmms15\n9/24/2026,7.03, ,6.1\n10/1/2026,7.28, ,6.3\n"
    monkeypatch.setattr(
        freddie_mac_model, "get_request", lambda url, timeout: FakeResponse(text=text)
    )

    rate = freddie_mac_model.get_mortgage_rate_30_year()

    assert rate.index.name == "Date"
    assert rate.loc[pd.Period("2026-10-01", "D"), "United States"] == pytest.approx(
        0.0728
    )


def test_industrial_production_reads_the_total_index_only(monkeypatch):
    from financetoolkit.economics.sources import frb_model

    text = (
        '"B50001: Total index" "B50001"  2026  101.926  101.617  0  0  0  0  0  0  0  0  0  0\n'
        '"B50002" 2026 99 99 99\n'
    )
    monkeypatch.setattr(
        frb_model, "get_request", lambda url, timeout: FakeResponse(text=text)
    )

    index = frb_model.get_industrial_production_index()["United States"]

    # Months not published yet are zero and left out; dated on the first day like FRED.
    assert index.to_dict() == {
        pd.Period("2026-01-01", "D"): 101.926,
        pd.Period("2026-02-01", "D"): 101.617,
    }


def test_recession_indicator_follows_freds_usrec_rules(monkeypatch):
    from financetoolkit.economics.sources import nber_model

    cycles = [
        {"peak": "", "trough": "1854-12-01"},
        {"peak": "2020-02-01", "trough": "2020-04-01"},
    ]
    monkeypatch.setattr(
        nber_model, "get_request", lambda url, timeout: FakeResponse(cycles)
    )

    indicator = nber_model.get_recession_indicator()["United States"]
    months = {str(period)[:7]: value for period, value in indicator.items()}

    assert months["1854-12"] == 1
    assert (
        months["2020-02"],
        months["2020-03"],
        months["2020-04"],
        months["2020-05"],
    ) == (0, 1, 1, 0)


def test_breakeven_inflation_and_the_five_year_forward(monkeypatch):
    def fake(url, timeout):
        if "real_yield_curve" in url:
            return FakeResponse(
                text='Date,"5 YR","7 YR","10 YR","20 YR","30 YR"\n10/02/2026,2.69,2.80,2.92,3.19,3.34\n'
            )
        return FakeResponse(
            text='Date,"5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"\n10/02/2026,5.06,5.17,5.28,5.67,5.63\n'
        )

    monkeypatch.setattr(treasury_model, "get_request", fake)

    breakeven = treasury_model.get_breakeven_inflation_expectations(
        "2026-10-01", "2026-10-02"
    ).iloc[0]

    assert breakeven["10 Year"] == pytest.approx(0.0236)
    assert breakeven["5 Year"] == pytest.approx(0.0237)
    # ((1 + 10y)^10 / (1 + 5y)^5)^(1/5) - 1, as FRED computes T5YIFR.
    assert breakeven["5 Year, 5 Year Forward"] == pytest.approx(
        ((1.0236**10) / (1.0237**5)) ** 0.2 - 1
    )


def test_a_fred_key_selects_fred_and_none_selects_the_keyless_source(monkeypatch):
    from financetoolkit import Economics
    from financetoolkit.economics.sources import fred_model, freddie_mac_model

    days = pd.period_range("2026-09-24", periods=1, freq="D", name="Date")
    monkeypatch.setattr(
        fred_model,
        "get_mortgage_rate_30_year",
        lambda *args: pd.DataFrame({"United States": [0.01]}, index=days),
    )
    monkeypatch.setattr(
        freddie_mac_model,
        "get_mortgage_rate_30_year",
        lambda: pd.DataFrame({"United States": [0.02]}, index=days),
    )
    monkeypatch.setattr(
        gmdb_model,
        "collect_global_macro_database_dataset",
        lambda **_: pd.DataFrame(),
    )

    with_key = Economics(start_date="2026-01-01", gmdb_source=False, fred_api_key="key")
    without_key = Economics(start_date="2026-01-01", gmdb_source=False, fred_api_key="")

    assert with_key.get_mortgage_rate_30_year().iloc[0, 0] == 0.01
    assert without_key.get_mortgage_rate_30_year().iloc[0, 0] == 0.02


def test_rows_empty_for_the_selected_countries_are_dropped():
    from financetoolkit.utilities.statistics_model import finalize_dataset

    months = pd.period_range("2026-06", "2026-09", freq="M")
    data = pd.DataFrame(
        {
            "United Kingdom": [0.049, None, None, None],
            "United States": [0.041, 0.041, 0.041, 0.042],
        },
        index=months,
    )

    result = finalize_dataset(
        dataset=data,
        start_date="2026-01-01",
        end_date="2026-12-31",
        default_rounding=4,
        countries=["United Kingdom"],
        axis="rows",
        row_slice=True,
        dropna=True,
    )

    assert list(result.index) == [pd.Period("2026-06", "M")]


def test_bis_consumer_prices_are_named_and_in_decimals(monkeypatch):
    text = "FREQ,REF_AREA,UNIT_MEASURE,TIME_PERIOD,OBS_VALUE\nM,US,771,2026-08,3.4\nM,IN,771,2026-08,4.8\n"
    urls = []

    def fake(url, timeout):
        urls.append(url)
        return FakeResponse(text=text)

    monkeypatch.setattr(bis_model, "get_request", fake)

    rates = bis_model.get_consumer_prices("inflation_rate", "2025-06-15", "2026-09-30")

    assert (
        "/api/v2/data/dataflow/BIS/WS_LONG_CPI/~/M..771?" in urls[0]
        and "startPeriod=2025-06&endPeriod=2026-09" in urls[0]
    )
    assert rates.iloc[0].to_dict() == {
        "India": pytest.approx(0.048),
        "United States": pytest.approx(0.034),
    }


def test_ibge_unemployment_asks_for_json_and_is_in_decimals(monkeypatch):
    from financetoolkit.economics.sources import ibge_model

    headers = []
    rows = [
        {"D3C": "Mês (Código)", "V": "Valor"},
        {"D3C": "202607", "V": "5.3"},
        {"D3C": "202608", "V": "5.3"},
    ]

    def fake(url, timeout, extra_headers):
        headers.append(extra_headers)
        return FakeResponse(rows)

    monkeypatch.setattr(ibge_model, "get_request", fake)

    rate = ibge_model.get_unemployment_rate()

    assert headers[0] == {"Accept": "application/json"}
    assert rate.loc[pd.Period("2026-08", "M"), "Brazil"] == pytest.approx(0.053)


def test_ecb_requests_start_on_the_first_of_the_month(monkeypatch):
    from financetoolkit.economics.sources import ecb_model

    urls = []
    text = (
        "KEY,TIME_PERIOD,OBS_VALUE\n"
        "YC.B.U2.EUR.4F.G_N_C.SV_C_YM.SR_3M,2026-09-15,2.62\n"
        "YC.B.U2.EUR.4F.G_N_C.SV_C_YM.SR_10Y,2026-09-15,4.19\n"
        "YC.B.U2.EUR.4F.G_N_C.SV_C_YM.SR_10Y,2026-09-02,4.10\n"
    )

    def fake(url, timeout):
        urls.append(url)
        return FakeResponse(text=text)

    monkeypatch.setattr(ecb_model, "get_request", fake)

    curve = ecb_model.get_yield_curve("2026-09-10", "2026-09-30")

    # The same URL all month, so the portal can serve it from its cache; the range is
    # applied afterwards.
    assert urls[0].endswith("startPeriod=2026-09-01")
    assert "SR_3M+SR_6M" in urls[0]
    assert list(curve.index) == [pd.Period("2026-09-15", "D")]
    assert curve.iloc[0].to_dict() == {
        "3M": pytest.approx(0.0262),
        "10Y": pytest.approx(0.0419),
    }


def test_ecb_ten_year_rate_is_annually_compounded(monkeypatch):
    from financetoolkit.economics.sources import ecb_model

    curve = pd.DataFrame(
        {"3M": [0.0262], "10Y": [0.0419]},
        index=pd.PeriodIndex(["2026-09-15"], freq="D"),
    )
    monkeypatch.setattr(ecb_model, "get_yield_curve", lambda start, end: curve)

    rate = ecb_model.get_long_term_interest_rate("2026-09-10", "2026-09-30")

    # The Svensson spot rate is continuously compounded: exp(0.0419) - 1 per annum.
    assert list(rate.columns) == ["Euro Area"]
    assert rate.iloc[0, 0] == pytest.approx(0.042790, abs=1e-6)


def test_calendar_events_filter_and_survey_levels(monkeypatch):
    from financetoolkit.economics.sources import fmp_model as economics_fmp_model

    releases = pd.DataFrame(
        [
            {
                "date": "2026-09-30 01:45:00",
                "country": "CN",
                "event": "PMI (Sep)",
                "actual": 52.4,
                "unit": "%",
            },
            {
                "date": "2026-10-01 14:00:00",
                "country": "US",
                "event": "ISM Manufacturing PMI (Sep)",
                "actual": 54.5,
                "unit": "Points",
            },
            {
                "date": "2026-09-26 12:30:00",
                "country": "US",
                "event": "Core PCE Price Index YoY (Aug)",
                "actual": 3.0,
                "unit": "%",
            },
            {
                "date": "2026-09-01 14:00:00",
                "country": "US",
                "event": "Unemployment Rate (Aug)",
                "actual": 4.1,
                "unit": "%",
            },
        ]
    ).assign(currency="USD", previous=None, estimate=None, impact="High")
    monkeypatch.setattr(
        economics_fmp_model,
        "get_cached_financial_data",
        lambda url, user_subscription="Free": releases.copy(),
    )

    pmis = economics_fmp_model.get_economic_calendar(
        "key", "2026-09-01", "2026-10-03", events="pmi"
    )
    everything = economics_fmp_model.get_economic_calendar(
        "key", "2026-09-01", "2026-10-03"
    ).set_index("Event")

    assert sorted(pmis["Event"]) == ["ISM Manufacturing PMI (Sep)", "PMI (Sep)"]
    # A PMI labelled "%" stays a level; a rate of change of an index is a percentage.
    assert everything.loc["PMI (Sep)", "Actual"] == 52.4
    assert everything.loc["Core PCE Price Index YoY (Aug)", "Actual"] == pytest.approx(
        0.03
    )
    assert everything.loc["Unemployment Rate (Aug)", "Actual"] == pytest.approx(0.041)


def test_calendar_events_match_without_spaces_or_hyphens(monkeypatch):
    from financetoolkit.economics.sources import fmp_model as economics_fmp_model

    releases = pd.DataFrame(
        [
            {
                "date": "2026-10-02 12:30:00",
                "country": "US",
                "event": "Non Farm Payrolls (Sep)",
            },
            {
                "date": "2026-10-02 12:30:00",
                "country": "US",
                "event": "Nonfarm Payrolls Private (Sep)",
            },
            {
                "date": "2026-10-03 12:30:00",
                "country": "US",
                "event": "Unemployment Rate (Sep)",
            },
        ]
    ).assign(
        currency="USD",
        previous=None,
        estimate=None,
        actual=1.0,
        unit=None,
        impact="High",
    )
    monkeypatch.setattr(
        economics_fmp_model,
        "get_cached_financial_data",
        lambda url, user_subscription="Free": releases.copy(),
    )

    payrolls = economics_fmp_model.get_economic_calendar(
        "key", "2026-09-01", "2026-10-03", events="Nonfarm Payrolls"
    )

    assert sorted(payrolls["Event"]) == [
        "Non Farm Payrolls (Sep)",
        "Nonfarm Payrolls Private (Sep)",
    ]


def test_a_period_defaults_to_the_highest_frequency_available():
    from financetoolkit import Economics  # noqa: PLC0415

    economics = Economics(use_cached_data=False)
    quarterly = Economics(quarterly=True, use_cached_data=False)

    # Without a period, the highest frequency the indicator is published at.
    assert economics._default_period(None, "daily") == "daily"
    assert economics._default_period(None, "monthly") == "monthly"
    # quarterly=True asks for quarterly data, where the indicator has it.
    assert quarterly._default_period(None, "monthly") == "quarterly"
    assert (
        quarterly._default_period(None, "daily", quarterly_available=False) == "daily"
    )
    # An explicit period always wins.
    assert quarterly._default_period("yearly", "daily") == "yearly"


def test_quarterly_oecd_indicators_default_to_quarterly_data(monkeypatch):
    from financetoolkit import Economics  # noqa: PLC0415

    requested = {}

    def fake(name):
        def fetch(quarterly=None, **kwargs):
            requested[name] = quarterly
            index = pd.period_range("2023Q1", periods=2, freq="Q")
            return pd.DataFrame({"Netherlands": [100.0, 101.0]}, index=index)

        return fetch

    for name in (
        "get_house_prices",
        "get_rent_prices",
        "get_household_savings_rate",
        "get_household_debt_to_income_ratio",
    ):
        monkeypatch.setattr(oecd_model, name, fake(name))

    # China's house prices come from the BIS.
    from financetoolkit.economics.sources import bis_model  # noqa: PLC0415

    monkeypatch.setattr(
        bis_model,
        "get_residential_property_prices",
        lambda *args, **kwargs: pd.Series(name="China", dtype=float),
    )

    economics = Economics(start_date="2023-01-01", use_cached_data=False)

    # Without arguments, the quarterly OECD data, also for house prices whose default
    # source, the Global Macro Database, is yearly only.
    economics.get_house_prices()
    economics.get_rent_prices()
    economics.get_household_savings_rate()
    economics.get_household_debt_to_income_ratio()
    assert requested == {
        "get_house_prices": True,
        "get_rent_prices": True,
        "get_household_savings_rate": True,
        "get_household_debt_to_income_ratio": True,
    }

    # An explicit quarterly=False still asks for the annual data.
    economics.get_rent_prices(quarterly=False)
    assert requested["get_rent_prices"] is False


def test_the_government_bond_yield_defaults_to_monthly(monkeypatch):
    from financetoolkit import FixedIncome  # noqa: PLC0415

    requested = []

    def fetch(period="", **kwargs):
        requested.append(period)
        index = pd.period_range("2023-01", periods=2, freq="M")
        return pd.DataFrame({"Netherlands": [0.03, 0.031]}, index=index)

    monkeypatch.setattr(oecd_model, "get_long_term_interest_rate", fetch)

    FixedIncome(start_date="2023-01-01").get_government_bond_yield()
    # The MCP server creates the class with quarterly=False, which must not mean yearly.
    FixedIncome(start_date="2023-01-01", quarterly=False).get_government_bond_yield()
    FixedIncome(start_date="2023-01-01", quarterly=True).get_government_bond_yield()

    assert requested == ["monthly", "monthly", "quarterly"]


def test_bis_commercial_property_leaves_out_dwellings_land_and_real_prices(monkeypatch):
    text = (
        "FREQ,REF_AREA,COVERED_AREA,RE_TYPE,RE_VINTAGE,COMPILING_ORG,PRICED_UNIT,ADJUST_CODED,TIME_PERIOD,OBS_VALUE\n"
        # Switzerland only has rented dwellings and the Czech Republic only commercial land.
        "Q,CH,0,O,0,2,6,0,2025-Q4,100\nQ,CH,0,O,0,2,6,0,2026-Q1,101\n"
        "Q,CZ,0,M,0,2,6,0,2026-Q1,150\n"
        # Iceland's real index is as long as its nominal one and is not taken.
        "Q,IS,3,A,0,2,8,0,2025-Q4,80\nQ,IS,3,A,0,2,8,0,2026-Q1,81\n"
        "Q,IS,3,A,0,2,1,0,2025-Q4,90\nQ,IS,3,A,0,2,1,0,2026-Q1,92\n"
        # Turkey's offices: the whole country wins over the capital of the same length.
        "Q,TR,2,B,0,2,6,0,2026-Q1,300\nQ,TR,0,B,0,2,6,0,2026-Q1,200\n"
    )
    monkeypatch.setattr(
        bis_model, "get_request", lambda url, timeout: FakeResponse(text=text)
    )

    prices = bis_model.get_commercial_property_prices("2025-10-01", "2026-03-31")

    assert sorted(prices.columns) == ["Iceland", "Turkey"]
    assert prices.loc[pd.Period("2026Q1", "Q"), "Iceland"] == 92  # noqa: PLR2004
    assert prices.loc[pd.Period("2026Q1", "Q"), "Turkey"] == 200  # noqa: PLR2004


def test_real_interest_rate_and_misery_index_line_up_oecd_and_gmdb_names(monkeypatch):
    from financetoolkit.economics import economics_controller

    years = pd.PeriodIndex(["2022", "2023"], freq="Y")
    oecd = pd.DataFrame({"Russia": [0.09, 0.11], "Germany": [0.02, 0.025]}, index=years)
    # The GMDB's own "Russian Federation", renamed when the dataset is read.
    inflation = pd.DataFrame(
        {"Russia": [0.14, 0.06], "Germany": [0.07, 0.06]}, index=years
    )

    monkeypatch.setattr(
        economics_controller.Economics, "_get_gmdb_dataset", lambda self, **kw: None
    )
    monkeypatch.setattr(
        oecd_model,
        "get_long_term_interest_rate",
        lambda **kw: oecd,
    )
    monkeypatch.setattr(oecd_model, "get_unemployment_rate", lambda **kw: oecd)
    monkeypatch.setattr(gmdb_model, "get_inflation_rate", lambda **kw: inflation)

    economics = economics_controller.Economics(
        start_date="2022-01-01", end_date="2023-12-31"
    )

    real_rate = economics.get_real_interest_rate(gmdb_source=False, period="yearly")
    misery = economics.get_misery_index(gmdb_source=False)

    assert sorted(real_rate.columns) == ["Germany", "Russia"]
    assert real_rate.loc["2023", "Russia"] == pytest.approx(0.05)
    assert misery.loc["2022", "Russia"] == pytest.approx(0.23)


def test_an_unpublished_bis_period_is_no_data(monkeypatch):
    def not_published(url, timeout):
        response = requests.Response()
        response.status_code = 404
        raise requests.HTTPError(response=response)

    monkeypatch.setattr(bis_model, "get_request", not_published)
    assert bis_model.request_csv("https://stats.bis.org/x", timeout=1) == ""

    def unavailable(url, timeout):
        response = requests.Response()
        response.status_code = 503
        raise requests.HTTPError(response=response)

    monkeypatch.setattr(bis_model, "get_request", unavailable)
    with pytest.raises(requests.HTTPError):
        bis_model.request_csv("https://stats.bis.org/x", timeout=1)


class WorkbookResponse:
    """A response carrying an Excel workbook, as the New York Fed serves its data."""

    def __init__(self, sheets: dict[str, pd.DataFrame]):
        import io

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            for name, sheet in sheets.items():
                sheet.to_excel(writer, sheet_name=name, index=False)
        self.content = buffer.getvalue()


def test_the_term_premium_splits_each_yield_into_decimals(monkeypatch):
    from financetoolkit.economics.sources import nyfed_model

    sheet = pd.DataFrame({"DATE": ["02-Nov-2026", "03-Nov-2026"]})
    for maturity in range(1, 11):
        # In percent, as the workbook publishes them: the yield is the sum of the parts.
        sheet[f"ACMY{maturity:02d}"] = [4.0 + maturity / 10, 4.1]
        sheet[f"ACMTP{maturity:02d}"] = [maturity / 10, 0.1]
        sheet[f"ACMRNY{maturity:02d}"] = [4.0, 4.0]

    monkeypatch.setattr(
        nyfed_model,
        "get_request",
        lambda url, timeout: WorkbookResponse({"ACM Daily": sheet}),
    )

    term_premium = nyfed_model.get_term_premium()

    assert list(term_premium.columns.get_level_values(0).unique()) == [
        "Term Premium",
        "Expected Short Rate",
        "Fitted Yield",
    ]
    assert term_premium.index.tolist() == [
        pd.Period("2026-11-02", freq="D"),
        pd.Period("2026-11-03", freq="D"),
    ]
    first = term_premium.iloc[0]
    assert first[("Term Premium", "10Y")] == pytest.approx(0.01)
    assert first[("Fitted Yield", "10Y")] == pytest.approx(
        first[("Term Premium", "10Y")] + first[("Expected Short Rate", "10Y")]
    )


def test_a_term_premium_workbook_without_its_columns_is_reported(monkeypatch):
    from financetoolkit.economics.sources import nyfed_model

    sheet = pd.DataFrame({"DATE": ["03-Nov-2026"], "ACMY01": [4.0]})
    monkeypatch.setattr(
        nyfed_model,
        "get_request",
        lambda url, timeout: WorkbookResponse({"ACM Daily": sheet}),
    )

    with pytest.raises(ValueError, match="changed its format"):
        nyfed_model.get_term_premium()


def test_the_supply_chain_pressure_index_skips_the_empty_rows(monkeypatch):
    from financetoolkit.economics.sources import nyfed_model

    # The workbook leaves empty rows between the header and the first month.
    sheet = pd.DataFrame(
        {
            "Date": [None, None, "31-Jan-1998", "28-Feb-1998"],
            "GSCPI": [None, None, -1.18, -0.49],
        }
    )
    monkeypatch.setattr(
        nyfed_model,
        "get_request",
        lambda url, timeout: WorkbookResponse({"GSCPI Monthly Data": sheet}),
    )

    index = nyfed_model.get_supply_chain_pressure_index()

    assert index.index.tolist() == [
        pd.Period("1998-01", freq="M"),
        pd.Period("1998-02", freq="M"),
    ]
    assert index["World"].tolist() == [-1.18, -0.49]


class SheetResponse:
    """A response carrying a workbook of one sheet laid out cell by cell, without a
    header row, as the Census Bureau and the Treasury lay out their files."""

    def __init__(self, rows: list[list], sheet_name: str = "Sheet1"):
        import io

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame(rows).to_excel(
                writer, sheet_name=sheet_name, index=False, header=False
            )
        self.content = buffer.getvalue()


def test_department_of_labor_claims_are_posted_for_and_seasonally_adjusted(monkeypatch):
    from financetoolkit.economics.sources import dol_model

    text = (
        "<r539cyState><week><weekEnded>10/18/2026</weekEnded><InitialClaims><NSA>201,000"
        "</NSA><SA>232,000</SA></InitialClaims></week><week><weekEnded>10/25/2026"
        "</weekEnded><InitialClaims><NSA></NSA><SA></SA></InitialClaims></week>"
    )
    requests_made = []

    def post(url, **kwargs):
        requests_made.append(kwargs)
        return FakeResponse(text=text)

    monkeypatch.setattr(dol_model, "get_request", post)

    claims = dol_model.get_initial_jobless_claims()

    # The report only answers its form, so the request is a POST of its fields.
    assert requests_made[0]["data"]["filetype"] == "xml"
    # A week not published yet carries no number and is left out.
    assert claims.index.tolist() == [pd.Period("2026-10-18", "D")]
    assert claims.index.name == "Date"
    assert claims["United States"].iloc[0] == 232000


def test_department_of_labor_outage_serves_the_cached_claims(cache, monkeypatch):
    from financetoolkit.economics.sources import dol_model

    report = (
        "<r539cyNational><week><weekEnded>10/18/2026</weekEnded><InitialClaims><NSA>"
        "201,000</NSA><SA>232,000</SA></InitialClaims></week></r539cyNational>"
    )
    # What the report answers when its database cannot be reached.
    outage = (
        "Unable to connect!<br/>SQLSTATE=HY000, SQLDriverConnect: -25546 [Informix]"
        "[Informix ODBC Driver][Informix]Invalid ASF assoc_id."
    )
    answers = [report]
    monkeypatch.setattr(
        dol_model, "get_request", lambda url, **kwargs: FakeResponse(text=answers[-1])
    )
    claims = dol_model.get_initial_jobless_claims()

    # Expire the cached claims, then let the database be unreachable: the outage is
    # retried and answered with the cached claims, not reported as a changed layout.
    answers.append(outage)
    monkeypatch.setattr(dol_model, "OUTAGE_DELAY_SECONDS", 0)
    monkeypatch.setattr(helpers, "RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(helpers, "STALE_TTL_SECONDS", 10**9)
    monkeypatch.setattr(cache, "get", _expire_fresh_reads(cache.get))

    assert dol_model.get_initial_jobless_claims().equals(claims)


CLAIMS_REPORT = (
    '<r539cyNational rundate="11/19/2026">'
    "<week><weekEnded>10/17/2026</weekEnded><InitialClaims><NSA>190,000</NSA><SF>97.1</SF>"
    "<SA>195,000</SA><SA4WK>201,500</SA4WK></InitialClaims><ContinuedClaims><NSA>1,500,000"
    "</NSA><SF>88.0</SF><SA>1,695,000</SA><SA4WK>1,700,000</SA4WK></ContinuedClaims><IUR>"
    "<NSA>1.0</NSA><SA>1.1</SA></IUR><CoveredEmployment>153,911,765</CoveredEmployment>"
    "</week>"
    "<week><weekEnded>10/24/2026</weekEnded><InitialClaims><NSA>192,000</NSA><SF>98.0</SF>"
    "<SA>196,000</SA><SA4WK>&#160;</SA4WK></InitialClaims><ContinuedClaims><NSA>&#160;</NSA>"
    "<SF>87.5</SF><SA>&#160;</SA><SA4WK>&#160;</SA4WK></ContinuedClaims><IUR><NSA>&#160;"
    "</NSA><SA>&#160;</SA></IUR><CoveredEmployment>&#160;</CoveredEmployment></week>"
    "<week><weekEnded>10/31/2026</weekEnded><InitialClaims><NSA>&#160;</NSA><SF>103.6</SF>"
    "<SA>&#160;</SA><SA4WK>&#160;</SA4WK></InitialClaims><ContinuedClaims><NSA>&#160;</NSA>"
    "<SF>87.5</SF><SA>&#160;</SA><SA4WK>&#160;</SA4WK></ContinuedClaims><IUR><NSA>&#160;"
    "</NSA><SA>&#160;</SA></IUR><CoveredEmployment>&#160;</CoveredEmployment></week>"
    "</r539cyNational>"
)


def test_department_of_labor_claims_report_gives_every_measure_from_one_request(
    cache, monkeypatch
):
    from financetoolkit.economics.sources import dol_model

    requests_made = []

    def post(url, **kwargs):
        requests_made.append(kwargs)
        return FakeResponse(text=CLAIMS_REPORT)

    monkeypatch.setattr(dol_model, "get_request", post)

    initial = dol_model.get_initial_jobless_claims()
    continued = dol_model.get_continuing_jobless_claims()
    rate = dol_model.get_insured_unemployment_rate()

    # The three series come from the one cached report.
    assert len(requests_made) == 1
    # The seasonally adjusted figures, not the unadjusted ones or the four-week averages;
    # a week without a figure is left out of that series only.
    assert initial["United States"].tolist() == [195000, 196000]
    assert continued["United States"].tolist() == [1695000]
    assert continued.index.tolist() == [pd.Period("2026-10-17", "D")]
    assert continued.index.name == "Date"
    # The rate is published in percent and returned as a decimal.
    assert rate["United States"].tolist() == [pytest.approx(0.011)]


def test_department_of_labor_claims_report_without_claims_is_a_changed_layout():
    from financetoolkit.economics.sources import dol_model

    with pytest.raises(ValueError, match="layout changed"):
        dol_model.parse_claims_report("<html>A different page</html>")


def test_weekly_series_aggregate_into_complete_periods_by_the_day_the_week_ends():
    # Saturdays from 2026-08-29 to 2026-10-24: August's last week, all of September (four
    # Saturdays) and October without its last Saturday (October 31).
    saturdays = pd.PeriodIndex(
        pd.date_range("2026-08-29", "2026-10-24", freq="W-SAT"), freq="D", name="Date"
    )
    claims = pd.DataFrame(
        {"United States": [10.0, 20.0, 50.0, 30.0, 40.0, 60.0, 70.0, 80.0, 90.0]},
        index=saturdays,
    )

    mean = helpers.aggregate_weekly_series(claims, "monthly")
    highest = helpers.aggregate_weekly_series(claims, "monthly", "max")
    lowest = helpers.aggregate_weekly_series(claims, "monthly", "min")
    last = helpers.aggregate_weekly_series(claims, "monthly", "last")

    # The week ending Saturday August 29 belongs to August, which starts before the data
    # and so is incomplete, and October still misses its fifth week: only September is
    # returned, the weeks ending 5, 12, 19 and 26 September.
    assert mean.index.tolist() == [pd.Period("2026-09", "M")]
    assert mean.index.name == "Date"
    assert mean["United States"].iloc[0] == 35.0
    assert highest["United States"].iloc[0] == 50.0
    assert lowest["United States"].iloc[0] == 20.0
    assert last["United States"].iloc[0] == 40.0

    # The week ending on the last day of a month belongs to that month.
    october = pd.PeriodIndex(
        pd.date_range("2026-10-03", "2026-10-31", freq="W-SAT"), freq="D", name="Date"
    )
    full_october = helpers.aggregate_weekly_series(
        pd.DataFrame({"United States": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=october),
        "monthly",
    )
    assert full_october.index.tolist() == [pd.Period("2026-10", "M")]
    assert full_october["United States"].iloc[0] == 3.0

    # Weekly data is returned as it is, and the quarter and year are incomplete here.
    assert helpers.aggregate_weekly_series(claims, "weekly") is claims
    assert helpers.aggregate_weekly_series(claims, "quarterly").empty
    with pytest.raises(ValueError, match="aggregation 'median'"):
        helpers.aggregate_weekly_series(claims, "monthly", "median")


def test_claims_are_combined_before_growth_and_cut_to_complete_months(monkeypatch):
    from financetoolkit import Economics
    from financetoolkit.economics.sources import dol_model

    saturdays = pd.PeriodIndex(
        pd.date_range("2025-01-04", "2026-10-24", freq="W-SAT"), freq="D", name="Date"
    )
    claims = pd.DataFrame(
        {"United States": [float(day.month) for day in saturdays]}, index=saturdays
    )
    monkeypatch.setattr(dol_model, "get_initial_jobless_claims", lambda: claims)

    economics = Economics(
        start_date="2026-01-01",
        end_date="2026-12-31",
        gmdb_source=False,
        fred_api_key="",
    )
    monthly = economics.get_initial_jobless_claims(period="monthly")
    growth = economics.get_initial_jobless_claims(period="monthly", growth=True, lag=12)

    # January to September 2026; October misses the week ending 31 October.
    assert monthly.index.tolist() == list(
        pd.period_range("2026-01", "2026-09", freq="M")
    )
    assert monthly["United States"].tolist() == [float(month) for month in range(1, 10)]
    # The growth compares a month with the same month a year earlier, from the history
    # before the start date.
    assert growth["United States"].eq(0).all()
    with pytest.raises(InvalidArgumentError, match="period must be one of"):
        economics.get_initial_jobless_claims(period="daily")


def test_census_retail_sales_leave_out_the_seasonal_factors(monkeypatch):
    from financetoolkit.economics.sources import census_model

    text = (
        "ADVANCE MONTHLY SALES\n"
        "YEAR  JAN  FEB  MAR  APR  MAY  JUN  JUL  AUG  SEP  OCT  NOV  DEC\n"
        "2026  700  701  702  703  704  705  706  707  708  (NA)\n"
        "SEASONAL FACTORS\n"
        "2026  0.9  0.9  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.1\n"
    )
    monkeypatch.setattr(
        census_model, "get_request", lambda url, **kwargs: FakeResponse(text=text)
    )

    sales = census_model.get_retail_sales()

    assert len(sales) == 9
    assert sales.index[0] == pd.Period("2026-01-01", "D")
    assert sales.loc[pd.Period("2026-09-01", "D"), "United States"] == 708


def test_census_housing_starts_read_the_dated_rows(monkeypatch):
    from financetoolkit.economics.sources import census_model

    rows = [
        ["New Privately-Owned Housing Units Started", None],
        ["Period", "Total"],
        ["August 2026", 1307],
        ["Sep 2026", 1352],
        ["Notes: (S) Does not meet publication standards.", None],
    ]
    response = SheetResponse(rows, sheet_name="Seasonally Adjusted")
    monkeypatch.setattr(census_model, "get_request", lambda url, **kwargs: response)

    starts = census_model.get_housing_starts()

    assert starts.index.tolist() == [
        pd.Period("2026-08-01", "D"),
        pd.Period("2026-09-01", "D"),
    ]
    assert starts["United States"].tolist() == [1307, 1352]


def test_bea_real_personal_income_is_in_billions_rounded_half_up(monkeypatch):
    from financetoolkit.economics.sources import bea_model

    text = (
        "%SeriesCode,Period,Value\n"
        'W875RX,2026M08,"17,015,150"\n'
        'W875RX,2026M09,"17,015,149"\n'
        'W875RX1,2026M09,"99"\n'
        'A065RC,2026M09,"25,000,000"\n'
    )
    monkeypatch.setattr(
        bea_model, "get_request", lambda url, **kwargs: FakeResponse(text=text)
    )

    income = bea_model.get_real_personal_income()

    # A half is rounded up, as the BEA rounds its published tables.
    assert income["United States"].tolist() == [17015.2, 17015.1]
    assert income.index.tolist() == [
        pd.Period("2026-08-01", "D"),
        pd.Period("2026-09-01", "D"),
    ]


def test_bea_gross_national_product_deflator_is_yearly(monkeypatch):
    from financetoolkit.economics.sources import bea_model

    text = (
        "%SeriesCode,Period,Value\n"
        'A001RD,1968,"18.246"\n'
        'A001RD,2025,"128.801"\n'
        'A191RD,2025,"128.893"\n'
        'A001RC,2025,"30,000"\n'
    )
    requested = []
    monkeypatch.setattr(
        bea_model,
        "get_request",
        lambda url, **kwargs: requested.append(url) or FakeResponse(text=text),
    )

    deflator = bea_model.get_gross_national_product_deflator()

    # Only the GNP deflator (A001RD), not the GDP deflator (A191RD) or GNP itself.
    assert requested == [bea_model.ANNUAL_URL]
    assert deflator["United States"].tolist() == [18.246, 128.801]
    assert deflator.index.tolist() == [
        pd.Period("1968", "Y"),
        pd.Period("2025", "Y"),
    ]


def test_bls_nonfarm_payrolls_are_dated_on_the_first_of_the_month(monkeypatch):
    from financetoolkit.economics.sources import bls_model

    payload = {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [
                {
                    "seriesID": "CES0000000001",
                    "data": [{"year": "2026", "period": "M09", "value": "159540"}],
                }
            ]
        },
    }
    monkeypatch.setattr(
        bls_model, "get_request", lambda url, **kwargs: FakeResponse(payload)
    )

    payrolls = bls_model.get_nonfarm_payrolls()

    assert payrolls.index.tolist() == [pd.Period("2026-09-01", "D")]
    assert payrolls.index.name == "Date"
    assert payrolls["United States"].iloc[0] == 159540


@pytest.mark.parametrize(
    ("method", "keyless_model"),
    [
        ("get_nonfarm_payrolls", "bls_model"),
        ("get_initial_jobless_claims", "dol_model"),
        ("get_continuing_jobless_claims", "dol_model"),
        ("get_insured_unemployment_rate", "dol_model"),
        ("get_retail_sales", "census_model"),
        ("get_housing_starts", "census_model"),
        ("get_real_personal_income", "bea_model"),
    ],
)
def test_a_fred_key_selects_fred_for_the_us_activity_indicators(
    monkeypatch, method, keyless_model
):
    from financetoolkit import Economics
    from financetoolkit.economics import sources
    from financetoolkit.economics.sources import fred_model

    days = pd.period_range("2026-09-01", periods=1, freq="D", name="Date")
    monkeypatch.setattr(
        fred_model,
        method,
        lambda *args: pd.DataFrame({"United States": [1.0]}, index=days),
    )
    monkeypatch.setattr(
        importlib.import_module(f"{sources.__name__}.{keyless_model}"),
        method,
        lambda: pd.DataFrame({"United States": [2.0]}, index=days),
    )

    with_key = Economics(start_date="2026-09-01", gmdb_source=False, fred_api_key="key")
    without_key = Economics(start_date="2026-09-01", gmdb_source=False, fred_api_key="")

    assert getattr(with_key, method)().iloc[0, 0] == 1.0
    assert getattr(without_key, method)().iloc[0, 0] == 2.0


def hqm_spot_sheet() -> list[list]:
    """The layout of a Treasury HQM spot rate file: the year above its first month, the
    month names below it, and from the seventh row a row per maturity."""
    rows = [[None, None, None] for _ in range(6)]
    rows[0][0] = "HQM Corporate Bond Yield Curve Spot Rates"
    rows[3][1:] = [2026, None]
    rows[4][1:] = ["Sep", "Oct"]
    rows[5][0] = "Maturity"
    for maturity in (0.5, 1, 1.5, 2, 100):
        rows.append([maturity, 4.0 + maturity / 100, None])

    return rows


def test_hqm_spot_rates_keep_the_named_maturities_in_decimals(monkeypatch):
    from financetoolkit.fixedincome.sources import hqm_model

    requested = []

    def answer(url, **kwargs):
        requested.append(url.rsplit("/", 1)[-1])
        return SheetResponse(hqm_spot_sheet())

    monkeypatch.setattr(hqm_model, "get_request", answer)

    rates = hqm_model.get_hqm_spot_rates("2026-01-01", "2026-12-31")

    # Only the five-year file holding 2026 is requested.
    assert requested == ["hqm_24_28.xls"]
    # The month without rates yet is left out, and so are the unnamed maturities.
    assert rates.index.tolist() == [pd.Period("2026-09", "M")]
    assert rates.columns.tolist() == ["6M", "1Y", "2Y", "100Y"]
    assert rates.index.name is None
    assert rates.loc[pd.Period("2026-09", "M"), "100Y"] == pytest.approx(0.05)


def test_hqm_par_yields_and_the_spread_over_treasury_averages(monkeypatch):
    from financetoolkit.fixedincome.sources import hqm_model

    rows = [
        ["HQM Par Yields", None, None, None, None, None],
        ["Date", None, "2-year", "5-year", "10-year", "30-year"],
        ["Sep 2026", None, 4.5, 4.9, 5.4, 6.2],
    ]
    monkeypatch.setattr(
        hqm_model, "get_request", lambda url, **kwargs: SheetResponse(rows)
    )
    days = pd.period_range("2026-09-01", periods=3, freq="D")
    # A 2-year average of 3.505 percent, which H.15 rounds up to 3.51.
    daily = pd.DataFrame(
        {
            "1 Mo": [4.0, 4.0, 4.0],
            "2 Yr": [0.03500, 0.03505, 0.03510],
            "5 Yr": [0.04, 0.04, 0.04],
            "10 Yr": [0.045, 0.045, 0.045],
            "30 Yr": [None, None, None],
        },
        index=days,
    )
    monkeypatch.setattr(
        hqm_model.treasury_model, "get_yield_curve", lambda *args: daily
    )

    par = hqm_model.get_hqm_par_yields()
    spread = hqm_model.get_hqm_corporate_bond_spread("2026-09-01", "2026-09-30")

    assert par.loc[pd.Period("2026-09", "M"), "30Y"] == pytest.approx(0.062)
    september = spread.loc[pd.Period("2026-09", "M")]
    assert september["2Y"] == pytest.approx(0.045 - 0.0351)
    assert september["10Y"] == pytest.approx(0.009)
    # Months without a 30-year Treasury yield have no 30-year spread.
    assert pd.isna(september["30Y"])


def test_hqm_curve_and_spread_need_no_fred_key(monkeypatch):
    from financetoolkit import FixedIncome
    from financetoolkit.fixedincome.sources import fred_model, hqm_model

    months = pd.period_range("2026-08", periods=2, freq="M")
    curve = pd.DataFrame({"10Y": [0.05, 0.051]}, index=months)
    monkeypatch.setattr(hqm_model, "get_hqm_spot_rates", lambda *args: curve)
    monkeypatch.setattr(hqm_model, "get_hqm_par_yields", lambda: curve + 0.001)
    monkeypatch.setattr(
        hqm_model, "get_hqm_corporate_bond_spread", lambda *args: curve / 10
    )
    monkeypatch.setattr(
        fred_model,
        "get_hqm_corporate_bond_yield_curve",
        lambda *args: curve * 2,
    )

    without_key = FixedIncome(
        start_date="2026-08-01", end_date="2026-09-30", fred_api_key=""
    )
    with_key = FixedIncome(
        start_date="2026-08-01", end_date="2026-09-30", fred_api_key="key"
    )

    assert without_key.get_hqm_corporate_bond_yield_curve().iloc[-1, 0] == 0.051
    assert without_key.get_hqm_corporate_bond_yield_curve(rate="par").iloc[
        -1, 0
    ] == pytest.approx(0.052)
    assert without_key.get_hqm_corporate_bond_spread().iloc[-1, 0] == pytest.approx(
        0.0051
    )
    assert with_key.get_hqm_corporate_bond_yield_curve().iloc[-1, 0] == 0.102


def test_bank_of_japan_price_indices_are_monthly_index_levels(monkeypatch):
    payload = {
        "STATUS": 200,
        "RESULTSET": [
            {
                "VALUES": {
                    "SURVEY_DATES": [202608, 202609, 202610],
                    "VALUES": [136.1, 135.7, None],
                }
            }
        ],
    }
    urls = []

    def fake(url, timeout):
        urls.append(url)
        return FakeResponse(payload)

    monkeypatch.setattr(boj_model, "get_request", fake)

    index = boj_model.collect_price_index("producer")

    assert "db=PR01&code=PRCG20_2200000000&startDate=196001" in urls[0]
    assert list(index.index) == list(pd.period_range("2026-08", "2026-09", freq="M"))
    assert index["Japan"].tolist() == [136.1, 135.7]


def test_monthly_indices_average_into_complete_periods_only():
    months = pd.period_range("2026-01", "2026-08", freq="M")
    data = pd.DataFrame(
        {"Japan": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}, index=months
    )

    quarterly = helpers.average_monthly_index(data, "quarterly")
    yearly = helpers.average_monthly_index(data, "yearly")

    assert quarterly["Japan"].tolist()[:2] == [2.0, 5.0]
    assert pd.isna(quarterly.loc[pd.Period("2026Q3", "Q"), "Japan"])
    assert pd.isna(yearly.iloc[0, 0])
    assert helpers.average_monthly_index(data, "monthly") is data


def test_eurostat_industrial_prices_request_the_headline_scope(monkeypatch):
    urls = []

    def fake(url, timeout):
        urls.append(url)
        return FakeResponse(json_stat(["EA21", "DE"], ["2026-09"], [137.8, 134.9]))

    monkeypatch.setattr(eurostat_model, "get_request", fake)

    producer = eurostat_model.get_industrial_price_index(
        "producer", "2026-09-01", "2026-09-30"
    )
    eurostat_model.get_industrial_price_index("import", "2026-09-01", "2026-09-30")

    assert "sts_inppd_m?" in urls[0] and "nace_r2=B-E36" in urls[0]
    assert "s_adj=NSA" in urls[0] and "unit=I21" in urls[0]
    assert "sts_inpi_m?" in urls[1] and "cpa2_1=CPA_B-D" in urls[1]
    assert producer.loc[pd.Period("2026-09", "M"), "Euro Area"] == 137.8


def test_price_indices_combine_their_publishers_and_select_fred_with_a_key(
    monkeypatch, caplog
):
    from financetoolkit import Economics
    from financetoolkit.economics.sources import bls_model, fred_model

    months = pd.period_range("2025-01", "2026-09", freq="M")
    rising = [100 + month for month in range(len(months))]
    monkeypatch.setattr(
        eurostat_model,
        "get_industrial_price_index",
        lambda measure, *args: pd.DataFrame(
            {"Euro Area": rising, "Germany": rising}, index=months
        ),
    )
    monkeypatch.setattr(
        boj_model,
        "collect_price_index",
        lambda measure: pd.DataFrame({"Japan": rising}, index=months),
    )
    monkeypatch.setattr(
        fred_model,
        "get_price_index",
        lambda *args: pd.DataFrame(
            {"United States": [1.0] * len(months)}, index=months
        ),
    )
    monkeypatch.setattr(
        bls_model,
        "get_price_index",
        lambda measure: pd.DataFrame({"United States": [2.0] * 9}, index=months[-9:]),
    )

    with_key = Economics(
        start_date="2025-01-01", end_date="2026-09-30", fred_api_key="key"
    )
    without_key = Economics(
        start_date="2025-01-01", end_date="2026-09-30", fred_api_key=""
    )

    producer = with_key.get_producer_price_index(
        countries=["United States", "Euro Area", "Japan", "China"]
    )
    assert list(producer.columns) == ["United States", "Euro Area", "Japan"]
    assert producer["United States"].iloc[-1] == 1.0

    growth = with_key.get_import_price_index(countries=["Japan"], growth=True, lag=12)
    assert growth.loc[pd.Period("2026-09", "M"), "Japan"] == pytest.approx(
        120 / 108 - 1, abs=1e-4
    )

    with caplog.at_level("WARNING"):
        keyless = without_key.get_export_price_index(countries=["United States"])
    assert keyless["United States"].dropna().iloc[-1] == 2.0
    assert "Without a FRED API key" in caplog.text

    quarterly = with_key.get_producer_price_index(
        countries=["Japan"], period="quarterly"
    )
    assert quarterly.loc[pd.Period("2026Q3", "Q"), "Japan"] == pytest.approx(119.0)


def test_overnight_rates_compound_over_calendar_days_with_their_day_count():
    # A Friday and the Monday after: Friday's rate applies over three days.
    index = pd.PeriodIndex(["2026-09-03", "2026-09-04", "2026-09-07"], freq="D")
    rates = pd.DataFrame(
        {"United States": [0.036, 0.0365, 0.037], "Japan": [0.01, None, 0.01]},
        index=index,
    )

    growth = helpers.compound_overnight_rate(rates, helpers.OVERNIGHT_RATE_DAY_COUNTS)

    assert growth["United States"].tolist() == pytest.approx(
        [1.0, 1 + 0.036 / 360, (1 + 0.036 / 360) * (1 + 0.0365 * 3 / 360)]
    )
    # A day without a publication stays empty and the rate before it covers the gap.
    assert growth["Japan"].iloc[0] == 1.0
    assert pd.isna(growth["Japan"].iloc[1])
    assert growth["Japan"].iloc[2] == pytest.approx(1 + 0.01 * 4 / 365)


def test_rebasing_starts_every_country_at_its_own_first_value():
    data = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0], "B": [None, 50.0, 40.0]},
        index=pd.period_range("2020-01", periods=3, freq="M"),
    )

    rebased = helpers.rebase_to_first(data)

    assert rebased["A"].tolist() == pytest.approx([1.0, 1.1, 1.21])
    assert pd.isna(rebased["B"].iloc[0])
    assert rebased["B"].iloc[1:].tolist() == pytest.approx([1.0, 0.8])


def _purchasing_power_economics(monkeypatch):
    from financetoolkit.economics.economics_controller import Economics

    economics = Economics(start_date="2020-01-01", end_date="2020-03-31")
    prices = pd.DataFrame(
        {
            "United States": [99.0, 100.0, 102.0, 104.0],
            "Netherlands": [50.0, 50.0, 51.0, None],
            "Brazil": [10.0, 10.0, 11.0, 12.0],
        },
        index=pd.period_range("2019-12", periods=4, freq="M"),
    )
    days = pd.period_range("2019-12-31", "2020-04-01", freq="D")
    rates = pd.DataFrame(
        {"United States": 0.036, "Euro Area": -0.0036}, index=days, dtype=float
    )
    monkeypatch.setattr(
        economics, "_get_consumer_price_index", lambda *args, **kwargs: prices
    )
    monkeypatch.setattr(economics, "_get_overnight_rate", lambda *args: rates)

    return economics, rates


def test_purchasing_power_is_100_times_the_first_price_over_the_price(monkeypatch):
    economics, _ = _purchasing_power_economics(monkeypatch)

    purchasing_power = economics.get_purchasing_power(rounding=6)

    assert purchasing_power["United States"].tolist() == pytest.approx(
        [100.0, 100 * 100 / 102, 100 * 100 / 104]
    )
    assert purchasing_power["Netherlands"].iloc[:2].tolist() == pytest.approx(
        [100.0, 100 * 50 / 51]
    )


def test_purchasing_power_of_cash_pairs_each_month_with_its_own_prices(
    monkeypatch,
):
    economics, rates = _purchasing_power_economics(monkeypatch)

    cash = economics.get_purchasing_power(
        countries=["United States", "Netherlands", "Brazil"],
        overnight_rate=True,
        rounding=8,
    )

    # Brazil has no overnight rate here: it is left out rather than an error.
    assert list(cash.columns) == ["United States", "Netherlands"]

    growth = helpers.compound_overnight_rate(
        rates.loc["2020-01-01":"2020-03-31"], helpers.OVERNIGHT_RATE_DAY_COUNTS
    )
    assert cash.loc["2020-02", "United States"] == pytest.approx(
        100 * growth.loc["2020-02-29", "United States"] * 100 / 102
    )
    # A euro member earns €STR, and a month without its prices is left empty.
    assert cash.loc["2020-02", "Netherlands"] == pytest.approx(
        100 * growth.loc["2020-02-29", "Euro Area"] * 50 / 51
    )
    assert pd.isna(cash.loc["2020-03", "Netherlands"])


def test_purchasing_power_of_cash_is_monthly_only(monkeypatch):
    economics, _ = _purchasing_power_economics(monkeypatch)

    with pytest.raises(InvalidArgumentError, match="monthly"):
        economics.get_purchasing_power(overnight_rate=True, period="yearly")


def test_world_bank_commodity_prices_are_read_from_the_linked_workbook(
    cache, monkeypatch
):
    import io

    from financetoolkit.economics.sources import world_bank_model

    rows = [
        ["World Bank Commodity Price Data (The Pink Sheet)", None, None],
        [None, None, None],
        [None, None, None],
        [None, None, None],
        [None, "Crude oil, Brent", "Beef **"],
        [None, "($/bbl)", "($/kg)"],
        ["1960M01", 1.63, "…"],
        ["1960M02", 1.64, 0.5],
    ]
    workbook = io.BytesIO()
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame(rows).to_excel(
            writer, sheet_name="Monthly Prices", header=False, index=False
        )
    link = "https://thedocs.worldbank.org/en/doc/x-0050012026/related/CMO-Historical-Data-Monthly.xlsx"
    requested = []

    def fake_get(url, timeout=60):
        requested.append(url)
        response = FakeResponse(text=f'<a href="{link}">Monthly prices</a>')
        if url == link:
            response.content = workbook.getvalue()
        return response

    monkeypatch.setattr(world_bank_model, "get_request", fake_get)

    prices = world_bank_model.get_commodity_prices()

    # The link is read from the page, as it changes every year.
    assert requested == [world_bank_model.PAGE_URL, link]
    # Names without commas and footnote marks, a missing price as NaN.
    assert prices.columns.tolist() == ["Crude oil Brent", "Beef"]
    assert prices.index.tolist() == [
        pd.Period("1960-01", "M"),
        pd.Period("1960-02", "M"),
    ]
    assert prices.loc["1960-02", "Crude oil Brent"] == 1.64
    assert pd.isna(prices.loc["1960-01", "Beef"])


def test_commodity_prices_are_selected_and_deflated(monkeypatch):
    from financetoolkit import Economics
    from financetoolkit.economics.sources import world_bank_model

    months = pd.period_range("2024-01", "2024-04", freq="M")
    monkeypatch.setattr(
        world_bank_model,
        "get_commodity_prices",
        lambda: pd.DataFrame(
            {"Gold": [100.0, 110.0, 120.0, 130.0], "Copper": [1.0, 2.0, 3.0, 4.0]},
            index=months,
        ),
    )
    economics = Economics(start_date="2024-01-01", end_date="2024-04-30", rounding=None)
    monkeypatch.setattr(
        economics,
        "get_consumer_price_index",
        lambda **_: pd.DataFrame(
            {"United States": [100.0, 105.0, 110.0]}, index=months[:3]
        ),
    )

    nominal = economics.get_commodity_prices(commodity="gold")
    real = economics.get_commodity_prices(
        commodity="Gold", deflate_with="United States"
    )

    assert nominal.columns.tolist() == ["Gold"]
    # In the money of the latest CPI month (March); April, after it, keeps its price.
    assert real["Gold"].tolist() == pytest.approx(
        [110.0, 110.0 * 110 / 105, 120.0, 130.0]
    )


def test_bis_monthly_average_rate_takes_the_euro_area_for_the_euro(monkeypatch):
    requested = []
    csv = {
        "EUR": (
            "FREQ,REF_AREA,CURRENCY,COLLECTION,TIME_PERIOD,OBS_VALUE\n"
            "M,DE,EUR,A,1998-12,0.8538\n"
            "M,XM,EUR,A,1998-12,0.8532\n"
            "M,DE,EUR,A,1999-01,0.8616\n"
            "M,XM,EUR,A,1999-01,0.8616\n"
        ),
        # A shared currency: the same rate in each economy, one of them a month short.
        "XAF": (
            "FREQ,REF_AREA,CURRENCY,COLLECTION,TIME_PERIOD,OBS_VALUE\n"
            "M,CM,XAF,A,2026-01,558.8\n"
            "M,CM,XAF,A,2026-02,554.8\n"
            "M,GA,XAF,A,2026-02,554.8\n"
            "M,GA,XAF,A,2026-03,567.5\n"
        ),
    }

    def fake(url, timeout):
        requested.append(url)
        currency = url.split("/M..")[1][:3]
        if currency not in csv:
            response = requests.Response()
            response.status_code = 404
            raise requests.HTTPError(response=response)
        return FakeResponse(text=csv[currency])

    monkeypatch.setattr(bis_model, "get_request", fake)

    euro = bis_model.get_monthly_average_exchange_rate(
        "eur", "1998-12-01", "1999-01-31"
    )
    # The average (A) of each month, keyed by currency rather than by economy.
    assert "WS_XRU/~/M..EUR.A?" in requested[0]
    assert "startPeriod=1998-12&endPeriod=1999-01" in requested[0]
    # The members' converted former currencies are not the euro's history.
    assert euro.tolist() == pytest.approx([0.8532, 0.8616])
    assert euro.name == "EUR"

    franc = bis_model.get_monthly_average_exchange_rate(
        "XAF", "2026-01-01", "2026-03-31"
    )
    assert franc.index.tolist() == list(pd.period_range("2026-01", "2026-03", freq="M"))
    assert franc.tolist() == pytest.approx([558.8, 554.8, 567.5])

    assert bis_model.get_monthly_average_exchange_rate(
        "XYZ", "2026-01-01", "2026-03-31"
    ).empty


def _commodity_economics(monkeypatch, rates):
    from financetoolkit import Economics
    from financetoolkit.economics.sources import world_bank_model

    months = pd.period_range("2026-07", "2026-10", freq="M")
    monkeypatch.setattr(
        world_bank_model,
        "get_commodity_prices",
        lambda: pd.DataFrame(
            {
                "Crude oil Brent": [70.0, 80.0, 100.0, 125.0],
                "Natural gas index": [100.0, 110.0, 150.0, 180.0],
            },
            index=months,
        ),
    )
    requested = []

    def fake_rate(currency, start_date, end_date):
        requested.append((currency, start_date, end_date))
        return pd.Series(rates, index=months[: len(rates)], name=currency)

    monkeypatch.setattr(bis_model, "get_monthly_average_exchange_rate", fake_rate)
    economics = Economics(start_date="2026-07-01", end_date="2026-10-31", rounding=None)

    return economics, months, requested


def test_commodity_prices_are_converted_at_the_monthly_average_rate(monkeypatch):
    economics, _, requested = _commodity_economics(
        monkeypatch, [0.876, 0.863, 0.869, 0.893]
    )

    euros = economics.get_commodity_prices(currency="eur")

    assert requested[0][0] == "EUR"
    # Each month's price at the same month's rate (euros per US dollar).
    assert euros["Crude oil Brent"].tolist() == pytest.approx(
        [70.0 * 0.876, 80.0 * 0.863, 100.0 * 0.869, 125.0 * 0.893]
    )
    # The index is not a dollar amount, so it is not converted.
    assert euros["Natural gas index"].tolist() == [100.0, 110.0, 150.0, 180.0]
    # US dollars are the prices themselves, without asking for a rate.
    dollars = economics.get_commodity_prices(currency="USD")
    assert dollars["Crude oil Brent"].tolist() == [70.0, 80.0, 100.0, 125.0]
    assert len(requested) == 1


def test_commodity_prices_without_a_rate_are_nan_with_a_warning(monkeypatch, caplog):
    import logging

    economics, _, _ = _commodity_economics(monkeypatch, [0.876, 0.863])

    with caplog.at_level(logging.WARNING, logger="financetoolkit"):
        euros = economics.get_commodity_prices(
            commodity="Crude oil Brent", currency="EUR"
        )

    # No rate is carried forward into the months the BIS has not published.
    assert euros["Crude oil Brent"].iloc[:2].tolist() == pytest.approx(
        [70.0 * 0.876, 80.0 * 0.863]
    )
    assert euros["Crude oil Brent"].iloc[2:].isna().all()
    warnings = [
        r.getMessage() for r in caplog.records if "EUR exchange rate" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "2026-09, 2026-10" in warnings[0]


def test_commodity_prices_are_converted_before_they_are_deflated(monkeypatch):
    economics, months, _ = _commodity_economics(
        monkeypatch, [0.876, 0.863, 0.869, 0.893]
    )
    monkeypatch.setattr(
        economics,
        "get_consumer_price_index",
        lambda **_: pd.DataFrame(
            {"Euro Area": [100.0, 101.0, 102.0, 104.0]}, index=months
        ),
    )

    real = economics.get_commodity_prices(
        commodity="Crude oil Brent", currency="EUR", deflate_with="Euro Area"
    )

    # In euros of October, the latest month of the euro area consumer price index.
    assert real["Crude oil Brent"].tolist() == pytest.approx(
        [
            70.0 * 0.876 * 104 / 100,
            80.0 * 0.863 * 104 / 101,
            100.0 * 0.869 * 104 / 102,
            125.0 * 0.893,
        ]
    )


def test_commodity_prices_in_an_unknown_currency_are_an_error(monkeypatch):
    economics, _, _ = _commodity_economics(monkeypatch, [])

    assert economics.get_commodity_prices(currency="XYZ").empty


def test_bls_consumer_prices_come_in_one_request(monkeypatch):
    from financetoolkit.economics.sources import bls_model

    requests_sent = []

    def respond(url, **kwargs):
        requests_sent.append(json.loads(kwargs["data"]))
        return FakeResponse(
            {
                "status": "REQUEST_SUCCEEDED",
                "Results": {
                    "series": [
                        {
                            "seriesID": series,
                            "data": [
                                {"year": "2026", "period": "M10", "value": "337.511"},
                                {"year": "2026", "period": "M09", "value": "336.940"},
                            ],
                        }
                        for series in requests_sent[-1]["seriesid"]
                    ]
                },
            }
        )

    monkeypatch.setattr(bls_model, "get_request", respond)

    prices = bls_model.get_consumer_price_index()

    assert len(requests_sent) == 1
    assert list(prices.columns) == list(bls_model.CONSUMER_PRICE_SERIES)
    assert prices.loc[pd.Period("2026-10", "M"), "core"] == pytest.approx(337.511)


def test_millennium_series_in_real_terms(monkeypatch):
    from financetoolkit import Economics
    from financetoolkit.economics.sources import boe_model

    years = pd.period_range("1861", periods=4, freq="Y")
    monkeypatch.setattr(
        boe_model,
        "get_millennium_data",
        lambda: pd.DataFrame({"$ Oil prices": [0.49, 1.0, 2.0, 8.06]}, index=years),
    )
    monkeypatch.setattr(
        Economics,
        "get_consumer_price_index",
        lambda self, **kwargs: pd.DataFrame(
            {"United States": [3.7146, 4.2402, 5.2911, 6.623]}, index=years
        ),
    )

    real = Economics(
        start_date="1861-01-01", end_date="1864-12-31"
    ).get_millennium_of_macroeconomic_data(
        series="$ Oil prices", deflate_with="United States", rounding=6
    )

    # In the money of the latest year of the price index, 1864.
    assert real.loc["1864", "$ Oil prices"] == pytest.approx(8.06)
    assert real.loc["1861", "$ Oil prices"] == pytest.approx(
        0.49 * 6.623 / 3.7146, abs=1e-6
    )


def test_eurostat_leaves_out_its_experimental_us_index(monkeypatch):
    months = pd.period_range("2022-06", periods=2, freq="M")
    hicp = pd.DataFrame(
        {"Germany": [8.2, 8.5], "United States": [10.1, 9.8]}, index=months
    )
    monkeypatch.setattr(
        eurostat_model, "collect_eurostat_data", lambda *args, **kwargs: hicp.copy()
    )

    # The BIS (the CPI-U) is then the only source of the United States, also when both
    # end in the same month.
    for prices in (
        eurostat_model.get_inflation_rate("2022-06-01", "2022-07-31"),
        eurostat_model.get_consumer_price_index("2022-06-01", "2022-07-31"),
    ):
        assert list(prices.columns) == ["Germany"]


def test_yield_decomposition_adds_the_parts_and_skips_missing_days(monkeypatch):
    from financetoolkit import Economics

    days = pd.to_datetime(["2026-11-06", "2026-11-09", "2026-11-10"])
    columns = pd.MultiIndex.from_tuples(
        [("United States", "5Y"), ("United States", "10Y")],
        names=["Country", "Maturity"],
    )
    real = pd.DataFrame(
        [[0.0235, 0.027], [0.0242, 0.0266], [0.0256, 0.0278]],
        index=days,
        columns=columns,
    )
    # The breakeven has no value on the 9th: the change of the 10th is over two days.
    breakeven = pd.DataFrame(
        [[0.0229, 0.0222], [float("nan"), float("nan")], [0.0225, 0.022]],
        index=days,
        columns=columns,
    )
    monkeypatch.setattr(Economics, "get_real_yield_curve", lambda self, **kwargs: real)
    monkeypatch.setattr(
        Economics,
        "get_breakeven_inflation_expectations",
        lambda self, **kwargs: breakeven,
    )

    decomposition = Economics(
        start_date="2026-11-06", end_date="2026-11-10"
    ).get_yield_decomposition(maturities="5y", rounding=6)
    five_year = decomposition[("United States", "5Y")]

    assert list(five_year.index) == [days[0], days[2]]
    assert five_year.loc[days[2], "Nominal Yield"] == pytest.approx(0.0481)
    assert five_year.loc[days[2], "Real Change"] == pytest.approx(0.0021)
    assert five_year.loc[days[2], "Breakeven Change"] == pytest.approx(-0.0004)
    assert five_year.loc[days[2], "Nominal Change"] == pytest.approx(0.0017)
    assert ("United States", "10Y") not in decomposition.columns


def test_long_run_prices_warn_about_repeated_years_and_hyperinflations(caplog):
    from financetoolkit.mcp_server.registry.diagnostics_model import missing_data_notes

    years = pd.period_range("1911", "1926", freq="Y")
    germany = [
        9.25,
        9.74,
        9.74,
        9.74,
        13.1,
        17.5,
        21.9,
        30.2,
        47.7,
        101.7,
        130.2,
        1464.2,
        1.5e12,
        1.27e13,
        1.38e13,
        1.39e13,
    ]
    index = pd.DataFrame(
        {"Germany": germany, "France": [float(n) for n in range(10, 26)]}, index=years
    )

    with caplog.at_level("WARNING"):
        helpers.warn_about_long_run_prices(index, "Germany", "1912", "1930")

    messages = [record.getMessage() for record in caplog.records]
    # 1913 and 1914 repeat 1912; 1922 and 1923 rose more than tenfold; France is not asked.
    assert len(messages) == 2
    assert "1913 and 1914" in messages[0]
    assert "Prices in Germany rose more than tenfold in 1922 and 1923" in messages[1]
    # The MCP server passes both on as notes.
    assert missing_data_notes(messages) == messages

    caplog.clear()
    with caplog.at_level("WARNING"):
        helpers.warn_about_long_run_prices(index, None, "1916", "1921")
    assert not caplog.records


def test_portwatch_transits_are_read_in_pages(cache, monkeypatch):
    from urllib.parse import parse_qs, urlsplit

    from financetoolkit.economics.sources import portwatch_model

    monkeypatch.setattr(portwatch_model, "PAGE_SIZE", 2)
    days = ["2026-02-27", "2026-02-28", "2026-03-01"]
    queries = []

    def row(day, tankers):
        fields = {f"n_{category}": 0 for category in portwatch_model.VESSEL_TYPES}
        fields |= {
            f"capacity_{category}": 0.0
            for category in portwatch_model.VESSEL_TYPES
            if category != "total"
        }
        return fields | {
            "date": day,
            "n_tanker": tankers,
            "n_total": tankers + 1,
            "capacity_tanker": tankers * 1000.0,
            "capacity": tankers * 1000.0 + 500.0,
        }

    def fake_get(url, timeout=60):
        query = {key: value[0] for key, value in parse_qs(urlsplit(url).query).items()}
        queries.append(query)
        offset = int(query["resultOffset"])
        page = [row(day, tankers) for day, tankers in zip(days, [37, 36, 5])][
            offset : offset + 2
        ]
        return FakeResponse(
            payload={
                "features": [{"attributes": attributes} for attributes in page],
                "exceededTransferLimit": offset + 2 < len(days),
            }
        )

    monkeypatch.setattr(portwatch_model, "get_request", fake_get)

    transits = portwatch_model.get_chokepoint_transits("Strait of Hormuz")

    # Two pages of at most two days, for the chokepoint by name.
    assert [query["resultOffset"] for query in queries] == ["0", "2"]
    assert queries[0]["where"] == "portname='Strait of Hormuz'"
    assert transits.index.tolist() == [pd.Period(day, "D") for day in days]
    assert transits["Transits Tanker"].tolist() == [37.0, 36.0, 5.0]
    assert transits["Transits Total"].tolist() == [38.0, 37.0, 6.0]
    # capacity is the volume of all ships, named after the Total category.
    assert transits.loc["2026-03-01", "Trade Volume Total"] == 5500.0


def test_portwatch_error_answers_raise(cache, monkeypatch):
    from financetoolkit.economics.sources import portwatch_model

    monkeypatch.setattr(
        portwatch_model,
        "get_request",
        lambda url, timeout=60: FakeResponse(
            payload={"error": {"code": 400, "message": "Invalid query"}}
        ),
    )

    with pytest.raises(ValueError, match="Invalid query"):
        portwatch_model.get_chokepoint_transits("Suez Canal")


def test_portwatch_rate_limit_serves_the_cached_copy(cache, monkeypatch):
    from financetoolkit.economics.sources import portwatch_model

    attributes = {f"n_{category}": 3 for category in portwatch_model.VESSEL_TYPES}
    attributes |= {
        f"capacity_{category}": 100.0
        for category in portwatch_model.VESSEL_TYPES
        if category != "total"
    }
    answers = [
        {
            "features": [
                {"attributes": attributes | {"date": "2026-03-01", "capacity": 600.0}}
            ]
        }
    ]
    requests_made = []

    def fake_get(url, timeout=60):
        requests_made.append(url)
        return FakeResponse(payload=answers[-1])

    monkeypatch.setattr(portwatch_model, "get_request", fake_get)
    transits = portwatch_model.get_chokepoint_transits("Suez Canal")

    # Expire the cached transits, then let PortWatch answer with its rate limit, which
    # arrives with status 200 and the 429 in the body: the cached transits are served,
    # and the source is not asked again.
    answers.append(
        {
            "error": {
                "code": 429,
                "message": "Unable to perform query. Too many requests.",
                "details": [],
            }
        }
    )
    monkeypatch.setattr(helpers, "STALE_TTL_SECONDS", 10**9)
    monkeypatch.setattr(cache, "get", _expire_fresh_reads(cache.get))
    requests_made.clear()

    assert portwatch_model.get_chokepoint_transits("Suez Canal").equals(transits)
    assert len(requests_made) == 1


def test_chokepoint_transits_select_measure_and_complete_periods(monkeypatch):
    from financetoolkit import Economics
    from financetoolkit.economics.sources import portwatch_model

    # Two full months and two days of March: March is not complete.
    days = pd.period_range("2026-01-01", "2026-03-02", freq="D")

    def transits(chokepoint):
        scale = 1.0 if chokepoint == "Strait of Hormuz" else 10.0
        return pd.DataFrame(
            {
                "Transits Tanker": scale,
                "Transits Total": 2 * scale,
                "Trade Volume Tanker": 1000 * scale,
                "Trade Volume Total": 3000 * scale,
            },
            index=days,
        )

    monkeypatch.setattr(
        portwatch_model,
        "get_chokepoint_names",
        lambda: ["Panama Canal", "Strait of Hormuz", "Suez Canal"],
    )
    monkeypatch.setattr(portwatch_model, "get_chokepoint_transits", transits)
    economics = Economics(start_date="2026-01-01", end_date="2026-03-31", rounding=None)

    # A part of a name selects the chokepoint; one chokepoint has a column per category.
    daily = economics.get_chokepoint_transits(chokepoints="hormuz")
    assert daily.columns.tolist() == ["Tanker", "Total"]
    assert len(daily) == len(days)

    monthly = economics.get_chokepoint_transits(
        chokepoints=["Strait of Hormuz", "suez canal"],
        measure="trade_volume",
        period="monthly",
    )
    assert monthly.columns.tolist() == [
        ("Strait of Hormuz", "Tanker"),
        ("Strait of Hormuz", "Total"),
        ("Suez Canal", "Tanker"),
        ("Suez Canal", "Total"),
    ]
    # January and February summed; the incomplete March is left out.
    assert monthly.index.tolist() == [
        pd.Period("2026-01", "M"),
        pd.Period("2026-02", "M"),
    ]
    assert monthly[("Suez Canal", "Tanker")].tolist() == [310000.0, 280000.0]

    # A name that matches several chokepoints, or none, is left out.
    assert economics.get_chokepoint_transits(chokepoints="Canal,Atlantis").empty

    # An unknown measure is refused, naming the measures.
    with pytest.raises(InvalidArgumentError, match="measure must be one of"):
        economics.get_chokepoint_transits(measure="volume")


def _bis_policy_rates(monkeypatch, frame):
    monkeypatch.setattr(
        bis_model, "get_central_bank_policy_rate", lambda start, end: frame
    )


def test_the_euro_area_policy_rate_is_extended_on_the_recent_month_only(monkeypatch):
    from financetoolkit import Economics
    from financetoolkit.fixedincome.sources import ecb_model as ecb_rates_model

    days = pd.PeriodIndex(pd.bdate_range("2026-10-01", "2026-11-10"), freq="D")
    _bis_policy_rates(monkeypatch, pd.DataFrame({"Euro Area": 0.02}, index=days))
    # The ECB publishes the deposit facility rate since 1999, which differs from the
    # main refinancing rate the BIS reports before 18 September 2024.
    history = pd.Series(
        0.04, index=pd.period_range("2023-09-20", "2026-11-17", freq="D")
    )
    history[pd.Period("2025-06-11", "D") :] = 0.02
    monkeypatch.setattr(ecb_rates_model, "get_deposit_facility", lambda: history)

    economics = Economics(start_date="2026-10-01", end_date="2026-11-17")
    rate = economics.get_central_bank_policy_rate(countries="Euro Area")

    assert rate["Euro Area"].last_valid_index() == pd.Period("2026-11-17", "D")


def test_a_refused_policy_rate_extension_is_logged(monkeypatch, caplog):
    days = pd.PeriodIndex(pd.bdate_range("2026-10-01", "2026-11-03"), freq="D")
    history = pd.DataFrame({"Euro Area": 0.02}, index=days)
    recent = pd.DataFrame(
        {"Euro Area": 0.03},
        index=pd.PeriodIndex(pd.bdate_range("2026-10-01", "2026-11-10"), freq="D"),
    )

    with caplog.at_level("INFO", logger="financetoolkit"):
        extended = helpers.extend_with_recent(history, recent, "Euro Area")

    assert extended.equals(history)
    assert "do not match the history" in caplog.text


def test_daily_policy_rates_leave_out_weekends_for_every_central_bank(monkeypatch):
    from financetoolkit import Economics

    # The BIS lists every calendar day for the Federal Reserve and the ECB.
    days = pd.period_range("2026-11-02", "2026-11-10", freq="D")
    _bis_policy_rates(
        monkeypatch,
        pd.DataFrame({"United States": 0.03875, "Japan": 0.0125}, index=days),
    )

    economics = Economics(start_date="2026-11-02", end_date="2026-11-10")
    rate = economics.get_central_bank_policy_rate()

    assert pd.Period("2026-11-07", "D") not in rate.index
    assert pd.Period("2026-11-08", "D") not in rate.index
    assert len(rate) == 7  # noqa: PLR2004


def test_policy_rates_of_a_short_form_country_are_extended(monkeypatch):
    from financetoolkit import Economics

    days = pd.PeriodIndex(pd.bdate_range("2026-10-26", "2026-11-03"), freq="D")
    _bis_policy_rates(monkeypatch, pd.DataFrame({"United Kingdom": 0.0375}, index=days))
    bank_rate_days = pd.PeriodIndex(
        pd.bdate_range("2026-10-26", "2026-11-09"), freq="D"
    )
    monkeypatch.setattr(
        boe_model,
        "get_bank_rate",
        lambda start, end: pd.DataFrame(
            {"United Kingdom": [0.0375] * 8 + [0.04] * 3}, index=bank_rate_days
        ),
    )

    economics = Economics(start_date="2026-10-26", end_date="2026-11-09")
    rate = economics.get_central_bank_policy_rate(countries="uk")

    assert (
        rate.loc[pd.Period("2026-11-09", "D"), "United Kingdom"] == 0.04
    )  # noqa: PLR2004


def test_policy_rates_can_be_quarterly(monkeypatch):
    from financetoolkit import Economics

    days = pd.PeriodIndex(pd.bdate_range("2026-01-01", "2026-09-30"), freq="D")
    rates = pd.Series(0.0425, index=days)
    rates[pd.Period("2026-06-18", "D") :] = 0.04
    _bis_policy_rates(monkeypatch, rates.to_frame("United States"))

    economics = Economics(
        start_date="2026-01-01", end_date="2026-09-30", quarterly=True
    )
    rate = economics.get_central_bank_policy_rate()

    # The rate on the last business day of each quarter.
    assert rate["United States"].to_dict() == {
        pd.Period("2026Q1", "Q"): 0.0425,
        pd.Period("2026Q2", "Q"): 0.04,
        pd.Period("2026Q3", "Q"): 0.04,
    }


def test_oecd_euro_area_rates_are_named_euro_area(monkeypatch):
    class Response:
        text = (
            "REF_AREA,TIME_PERIOD,OBS_VALUE\n"
            "EA20,2026-08,2.5\nEA20,2026-09,2.6\n"
            "DEU,2026-08,2.5\nDEU,2026-09,2.6\n"
        )

    monkeypatch.setattr(oecd_model, "get_request", lambda *args, **kwargs: Response())

    for rate in (
        oecd_model.get_short_term_interest_rate("monthly"),
        oecd_model.get_long_term_interest_rate("monthly"),
    ):
        assert list(rate.columns) == ["Euro Area", "Germany"]
        assert rate.loc[pd.Period("2026-09", "M"), "Euro Area"] == pytest.approx(0.026)


def test_single_country_sources_are_chosen_for_short_forms_and_any_casing(monkeypatch):
    from financetoolkit import Economics

    fetched = []

    def source(country):
        def fetch():
            fetched.append(country)
            return pd.DataFrame(
                {country: [0.04]}, index=pd.PeriodIndex(["2026-11-02"], freq="D")
            )

        return fetch

    sources = [
        (source("United States"), "United States"),
        (source("United Kingdom"), "United Kingdom"),
        (source("Japan"), "Japan"),
    ]

    combined = Economics._combine_sources_for(sources, "usa, united kingdom")

    assert sorted(fetched) == ["United Kingdom", "United States"]
    assert sorted(combined.columns) == ["United Kingdom", "United States"]


def test_long_term_rates_of_a_short_form_country_default_to_daily(monkeypatch):
    from financetoolkit import Economics

    requested = []
    monkeypatch.setattr(
        Economics,
        "_get_daily_long_term_interest_rate",
        lambda self, countries: requested.append(countries)
        or pd.DataFrame(
            {"United States": [0.041]}, index=pd.PeriodIndex(["2026-11-02"], freq="D")
        ),
    )

    economics = Economics(start_date="2026-11-02", end_date="2026-11-02")
    rate = economics.get_long_term_interest_rate(countries=["usa"])

    assert requested == [["usa"]]
    assert (
        rate.loc[pd.Period("2026-11-02", "D"), "United States"] == 0.041
    )  # noqa: PLR2004


def test_annual_short_and_long_term_rates_come_from_the_oecd_yearly(monkeypatch):
    from financetoolkit import Economics

    periods = []

    def fetch(period, start_date=None, end_date=None):
        periods.append(period)
        return pd.DataFrame(
            {"United States": [0.05]}, index=pd.PeriodIndex(["2025"], freq="Y")
        )

    monkeypatch.setattr(oecd_model, "get_short_term_interest_rate", fetch)
    monkeypatch.setattr(oecd_model, "get_long_term_interest_rate", fetch)

    economics = Economics(start_date="2025-01-01", end_date="2025-12-31")

    for method in (
        economics.get_short_term_interest_rate,
        economics.get_long_term_interest_rate,
        economics.get_yield_curve_slope,
    ):
        assert not method(
            countries=["United States"], period="annual", gmdb_source=False
        ).empty

    assert periods == ["yearly"] * 4


def test_inflation_expectations_follow_the_quarterly_setting(monkeypatch):
    from financetoolkit import Economics
    from financetoolkit.fixedincome.sources import bundesbank_model

    months = pd.period_range("2026-01", "2026-06", freq="M")
    monkeypatch.setattr(
        bundesbank_model,
        "get_survey_inflation_expectations",
        lambda start, end: pd.DataFrame(
            {"5Y": [0.021, 0.022, 0.023, 0.024, 0.025, 0.026]}, index=months
        ),
    )

    economics = Economics(
        start_date="2026-01-01", end_date="2026-06-30", quarterly=True
    )
    expectations = economics.get_inflation_expectations(countries="germany")

    assert list(expectations.index) == [
        pd.Period("2026Q1", "Q"),
        pd.Period("2026Q2", "Q"),
    ]
    assert (
        expectations.loc[pd.Period("2026Q2", "Q"), ("Germany", "5Y")] == 0.026
    )  # noqa: PLR2004


def test_commercial_property_series_do_not_depend_on_the_date_range(monkeypatch):
    header = "FREQ,REF_AREA,COVERED_AREA,RE_TYPE,RE_VINTAGE,COMPILING_ORG,PRICED_UNIT,ADJUST_CODED"
    # Saudi Arabia: a region from 2015 and the whole country from 2021; Slovenia's
    # offices, retail and office and retail indices are equally long.
    keys = (
        f"{header}\nQ,SA,2,C,0,1,6,0\nQ,SA,0,C,0,1,6,0\n"
        "Q,SI,0,D,0,1,6,0\nQ,SI,0,C,0,1,6,0\nQ,SI,0,B,0,1,6,0\n"
    )
    observations = (
        f"{header},TIME_PERIOD,OBS_VALUE\n"
        "Q,SA,2,C,0,1,6,0,2019-Q4,90\nQ,SA,2,C,0,1,6,0,2020-Q1,91\n"
        "Q,SI,0,D,0,1,6,0,2020-Q1,103\nQ,SI,0,C,0,1,6,0,2020-Q1,102\n"
        "Q,SI,0,B,0,1,6,0,2020-Q1,101\n"
    )
    monkeypatch.setattr(
        bis_model,
        "get_request",
        lambda url, timeout: FakeResponse(
            text=keys if "serieskeysonly" in url else observations
        ),
    )

    prices = bis_model.get_commercial_property_prices("2019-10-01", "2020-03-31")

    # The whole country is chosen even before its index starts, so a range that only
    # has the region does not return the region; offices come before retail.
    assert list(prices.columns) == ["Slovenia"]
    assert prices.loc[pd.Period("2020Q1", "Q"), "Slovenia"] == 101  # noqa: PLR2004


def test_quarterly_consumer_prices_keep_the_last_month_of_each_quarter(monkeypatch):
    months = pd.period_range("2025-01", "2025-12", freq="M")
    rows = [
        f"M,NZ,771,{month},{2.0 + (month.quarter - 1) * 0.5}\n"
        f"M,PL,771,{month},{4.0 if month.month < 4 else 3.0 + month.month / 10}\n"
        for month in months
    ]
    text = "FREQ,REF_AREA,UNIT_MEASURE,TIME_PERIOD,OBS_VALUE\n" + "".join(rows)
    monkeypatch.setattr(
        bis_model, "get_request", lambda url, timeout: FakeResponse(text=text)
    )

    rates = bis_model.get_consumer_prices("inflation_rate", "2025-01-01", "2025-12-31")

    # New Zealand repeats each quarter over its months; Poland repeats only its first
    # quarter, as a monthly series can, and keeps every month.
    assert rates["New Zealand"].dropna().to_dict() == {
        pd.Period("2025-03", "M"): pytest.approx(0.02),
        pd.Period("2025-06", "M"): pytest.approx(0.025),
        pd.Period("2025-09", "M"): pytest.approx(0.03),
        pd.Period("2025-12", "M"): pytest.approx(0.035),
    }
    assert rates["Poland"].notna().all()


def test_quarterly_only_unemployment_rates_are_named(monkeypatch, caplog):
    from financetoolkit import Economics

    months = pd.period_range("2026-01", "2026-06", freq="M")
    monkeypatch.setattr(
        Economics,
        "_combine_sources_for",
        lambda self, sources, countries: pd.DataFrame({"Germany": 0.063}, index=months),
    )
    monkeypatch.setattr(
        oecd_model,
        "get_unemployment_rate",
        lambda period, start_date, end_date: pd.DataFrame(
            {"New Zealand": [0.052, 0.054], "Germany": [0.063, 0.063]},
            index=pd.period_range("2026Q1", "2026Q2", freq="Q"),
        ),
    )

    economics = Economics(start_date="2026-01-01", end_date="2026-06-30")

    with caplog.at_level("WARNING", logger="financetoolkit"):
        rate = economics.get_unemployment_rate(countries=["Germany", "new zealand"])

    assert list(rate.columns) == ["Germany"]
    assert (
        "The unemployment rate of New Zealand is published quarterly, not monthly"
        in caplog.text
    )


def test_house_prices_of_china_come_from_the_bis(monkeypatch):
    from financetoolkit import Economics
    from financetoolkit.economics.sources import bis_model, oecd_model

    quarters = pd.period_range("2015Q1", "2016Q4", freq="Q")
    oecd = pd.DataFrame(
        {"China": [200.0] * 8, "Japan": [100.0 + i for i in range(8)]}, index=quarters
    )
    # The BIS index has 2010 = 100: 2015 averages 125, so it is rebased by 0.8.
    bis = pd.Series(
        [120.0, 124.0, 126.0, 130.0, 128.0, 126.0, 124.0, 122.0],
        index=quarters,
        name="China",
    )
    requested = {}

    def fake_oecd(quarterly, **kwargs):
        return oecd.copy() if quarterly else oecd.groupby(oecd.index.asfreq("Y")).mean()

    monkeypatch.setattr(oecd_model, "get_house_prices", fake_oecd)

    def fake_bis(country_code, inflation_adjusted, start_date, end_date):
        requested.update(
            country=country_code, real=inflation_adjusted, start=start_date
        )
        return bis.copy()

    monkeypatch.setattr(bis_model, "get_residential_property_prices", fake_bis)
    economics = Economics(
        start_date="2016-01-01",
        end_date="2016-12-31",
        rounding=None,
        use_cached_data=False,
    )

    quarterly = economics.get_house_prices(quarterly=True, inflation_adjusted=True)

    assert requested == {"country": "CN", "real": True, "start": "2015-01-01"}
    assert quarterly["China"].tolist() == pytest.approx([102.4, 100.8, 99.2, 97.6])
    assert quarterly["Japan"].tolist() == [104.0, 105.0, 106.0, 107.0]

    yearly = economics.get_house_prices(quarterly=False, gmdb_source=False)

    assert yearly.loc["2016", "China"] == pytest.approx(100.0)

    # When the BIS can't be reached, China is left out rather than taken from the OECD.
    monkeypatch.setattr(
        bis_model,
        "get_residential_property_prices",
        lambda *args, **kwargs: pd.Series(name="China", dtype=float),
    )

    assert "China" not in economics.get_house_prices(quarterly=True).columns


@pytest.mark.parametrize("end_date", ["2014-12-31", "2015-06-30", "2016-12-31"])
def test_house_prices_of_china_rebase_on_the_whole_base_year(monkeypatch, end_date):
    from financetoolkit import Economics
    from financetoolkit.economics.sources import bis_model, oecd_model

    quarters = pd.period_range("2014Q1", "2016Q4", freq="Q")
    oecd = pd.DataFrame({"Germany": [100.0 + i for i in range(12)]}, index=quarters)
    # The BIS index has 2010 = 100 and 2015 averages 125 (120 to 130), so 2014Q1 is 80.
    bis = pd.Series(
        [
            100.0,
            104.0,
            108.0,
            112.0,
            120.0,
            124.0,
            126.0,
            130.0,
            128.0,
            126.0,
            124.0,
            122.0,
        ],
        index=quarters,
        name="China",
    )

    def fake_oecd(quarterly, start_date, end_date, **kwargs):  # noqa: ARG001
        data = oecd.loc[: pd.Period(end_date, "Q")]
        return data.copy() if quarterly else data.groupby(data.index.asfreq("Y")).mean()

    def fake_bis(
        country_code, inflation_adjusted, start_date, end_date
    ):  # noqa: ARG001
        # The BIS returns the quarters up to the end date it is asked for.
        return bis.loc[pd.Period(start_date, "Q") : pd.Period(end_date, "Q")].copy()

    monkeypatch.setattr(oecd_model, "get_house_prices", fake_oecd)
    monkeypatch.setattr(bis_model, "get_residential_property_prices", fake_bis)
    economics = Economics(
        start_date="2014-01-01", end_date=end_date, rounding=None, use_cached_data=False
    )

    quarterly = economics.get_house_prices(quarterly=True)

    # Whichever the end date, the index is rebased on the four quarters of 2015 and cut to the range.  # noqa: E501
    assert quarterly["China"].iloc[0] == pytest.approx(80.0)
    assert quarterly.index[-1] == pd.Period(end_date, "Q")
    assert quarterly["China"].notna().all()

    yearly = economics.get_house_prices(quarterly=False, gmdb_source=False)

    # A year the range ends in part of stays out, as it does for the OECD indices.
    last_year = "2016" if end_date == "2016-12-31" else "2014"
    assert yearly["China"].dropna().index[-1] == pd.Period(last_year, "Y")
    assert yearly.loc["2014", "China"] == pytest.approx(84.8)


def test_requested_countries_skip_blanks_and_repeated_spellings():
    known = ["United States", "Germany"]

    assert helpers.normalize_requested_countries("Germany,", known) == ["Germany"]
    assert helpers.normalize_requested_countries(
        ["USA", "United States", " "], known
    ) == ["United States"]


def test_oecd_datasets_name_the_current_euro_area_euro_area(monkeypatch):
    csv = (
        "REF_AREA,TIME_PERIOD,OBS_VALUE\n"
        "EA20,2026-08,101.0\nEA20,2026-09,102.0\n"
        "EA19,2026-08,99.0\nEA19,2026-09,98.0\n"
        "DEU,2026-08,100.0\nDEU,2026-09,100.5\n"
    )

    class Response:
        text = csv

    monkeypatch.setattr(oecd_model, "get_request", lambda *args, **kwargs: Response())

    # Share prices, consumer confidence and exchange rates go through the same parsing
    # as the interest rates.
    data = oecd_model.collect_oecd_data("OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/", "M")

    assert sorted(data.columns) == ["Euro Area", "Euro Area 19", "Germany"]
    assert data.loc[pd.Period("2026-09", "M"), "Euro Area"] == 102.0  # noqa: PLR2004
    assert data.loc[pd.Period("2026-09", "M"), "Euro Area 19"] == 98.0  # noqa: PLR2004


def test_an_unreachable_policy_rate_extension_keeps_the_bis_series(monkeypatch, caplog):
    from financetoolkit import Economics
    from financetoolkit.fixedincome.sources import ecb_model as ecb_rates_model

    days = pd.PeriodIndex(pd.bdate_range("2026-10-01", "2026-11-10"), freq="D")
    _bis_policy_rates(
        monkeypatch,
        pd.DataFrame({"Euro Area": 0.02, "United Kingdom": 0.04}, index=days),
    )

    def unreachable(*args, **kwargs):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(ecb_rates_model, "get_deposit_facility", unreachable)
    monkeypatch.setattr(boe_model, "get_bank_rate", lambda start, end: pd.DataFrame())

    economics = Economics(start_date="2026-10-01", end_date="2026-11-17")

    with caplog.at_level("INFO", logger="financetoolkit"):
        rate = economics.get_central_bank_policy_rate()

    assert rate["Euro Area"].last_valid_index() == pd.Period("2026-11-10", "D")
    assert rate["United Kingdom"].last_valid_index() == pd.Period("2026-11-10", "D")
    assert "recent Euro Area policy rates could not be retrieved" in caplog.text


def test_repeated_quarters_are_thinned_per_quarter():
    months = pd.period_range("2025-01", "2026-09", freq="M")
    # Quarterly until 2025Q4, monthly from 2026; the monthly series repeats one value
    # over a single quarter (2025Q3) only.
    quarterly_then_monthly = [1.0] * 3 + [2.0] * 3 + [3.0] * 3 + [4.0] * 3
    quarterly_then_monthly += [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8]
    monthly = [float(month) for month in range(6)] + [6.0] * 3
    monthly += [float(month) for month in range(9, 21)]
    data = pd.DataFrame(
        {"Australia": quarterly_then_monthly, "Hong Kong": monthly}, index=months
    )

    thinned = helpers.keep_quarter_end_of_repeated_quarters(data)

    assert thinned["Australia"].loc[:"2025-12"].dropna().index.month.tolist() == [
        3,
        6,
        9,
        12,
    ]
    assert (
        thinned["Australia"].loc["2026-01":].equals(data["Australia"].loc["2026-01":])
    )
    assert thinned["Hong Kong"].equals(data["Hong Kong"])


def test_repeated_quarters_cut_off_by_the_range_keep_their_last_month():
    quarters = [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0]
    data = pd.DataFrame(
        {"New Zealand": quarters},
        index=pd.period_range("2025-11", "2026-08", freq="M"),
    )

    thinned = helpers.keep_quarter_end_of_repeated_quarters(data)

    assert thinned["New Zealand"].dropna().to_dict() == {
        pd.Period("2025-12", "M"): 1.0,
        pd.Period("2026-03", "M"): 2.0,
        pd.Period("2026-06", "M"): 3.0,
        pd.Period("2026-08", "M"): 4.0,
    }

    # A single complete quarter cannot be told apart from a monthly series.
    short = data.loc["2026-02":"2026-06"]
    assert helpers.keep_quarter_end_of_repeated_quarters(short).equals(short)


def test_a_refused_bls_request_leaves_nonfarm_payrolls_empty(monkeypatch, caplog):
    import logging  # noqa: PLC0415

    from financetoolkit import Economics  # noqa: PLC0415
    from financetoolkit.economics.sources import bls_model  # noqa: PLC0415

    payload = {
        "status": "REQUEST_NOT_PROCESSED",
        "message": ["Daily threshold reached."],
    }
    monkeypatch.setattr(
        bls_model, "get_request", lambda url, **kwargs: FakeResponse(payload)
    )

    economics = Economics(
        start_date="2024-01-01", end_date="2026-10-31", use_cached_data=False
    )
    clear_active_cache()

    # The refusal is reported with the BLS message rather than crashing on the empty frame.
    with caplog.at_level(logging.INFO, logger="financetoolkit"):
        assert economics.get_nonfarm_payrolls().empty

    assert any(
        "Daily threshold reached." in record.getMessage() for record in caplog.records
    )


def nmdb_archive(rows):
    """Zips an NMDB outstanding mortgage statistics CSV of (geography, market, period,
    series, VALUE1, VALUE2, suppressed) rows, as the FHFA publishes it."""
    import io
    import zipfile

    header = (
        "SOURCE,FREQUENCY,GEOLEVEL,GEOID,GEONAME,MARKET,PERIOD,YEAR,QUARTER,MONTH,"
        "SUPPRESSED,SERIESID,VALUE1,VALUE2\n"
    )
    lines = [
        f"NMDB,Quarterly,National,USA,{geography},{market},{period},{period[:4]},"
        f"{period[-1]},{int(period[-1]) * 3},{suppressed},{series},{value1},{value2}\n"
        for geography, market, period, series, value1, value2, suppressed in rows
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("outstanding.csv", header + "".join(lines))
    return buffer.getvalue()


def fhfa_lock_in_rows(geography="United States", market="All Mortgages"):
    """The interest rate rows of the 2026 Q1 and Q2 NMDB release, plus a series the
    model leaves out."""
    published = {
        "2026Q1": [4.4, 19.3, 30.1, 16.6, 11.8, 22.2],
        "2026Q2": [4.4, 19.2, 29.9, 16.5, 12.0, 22.5],
    }
    series = [
        "AVE_INTRATE",
        "PCT_INTRATE_LT_3",
        "PCT_INTRATE_3_4",
        "PCT_INTRATE_4_5",
        "PCT_INTRATE_5_6",
        "PCT_INTRATE_GE_6",
    ]
    rows = [
        (geography, market, period, name, value, value + 1, 0)
        for period, values in published.items()
        for name, value in zip(series, values, strict=True)
    ]
    rows.append((geography, market, "2026Q2", "TOT_LOANS", 54000, "", 0))
    return rows


def test_fhfa_outstanding_mortgage_rates_are_decimals_per_weighting(monkeypatch):
    from financetoolkit.economics.sources import fhfa_model

    content = nmdb_archive(
        fhfa_lock_in_rows()
        + [("Pacific", "All Mortgages", "2026Q2", "PCT_INTRATE_LT_3", 25.0, 26.0, 1)]
    )
    monkeypatch.setattr(
        fhfa_model,
        "get_request",
        lambda url, timeout: type("Response", (), {"content": content})(),
    )

    rates = fhfa_model.get_outstanding_mortgage_interest_rates()
    latest = rates.loc[pd.Period("2026Q2", "Q")]

    # VALUE1 is weighted by the number of mortgages, VALUE2 by their unpaid balance.
    assert latest[
        ("United States", "All Mortgages", "Loans", "Below 3%")
    ] == pytest.approx(0.192)
    assert latest[
        ("United States", "All Mortgages", "Balance", "Below 3%")
    ] == pytest.approx(0.202)
    assert latest[
        ("United States", "All Mortgages", "Loans", "Average Rate")
    ] == pytest.approx(0.044)
    # A suppressed statistic has no value, and other series are left out.
    assert pd.isna(latest[("Pacific", "All Mortgages", "Loans", "Below 3%")])
    assert "TOT_LOANS" not in rates.columns.get_level_values("Series")


def test_fhfa_archive_without_the_rate_series_is_a_changed_format(monkeypatch):
    from financetoolkit.economics.sources import fhfa_model

    content = nmdb_archive(
        [("United States", "All Mortgages", "2026Q2", "TOT_LOANS", 1, "", 0)]
    )
    monkeypatch.setattr(
        fhfa_model,
        "get_request",
        lambda url, timeout: type("Response", (), {"content": content})(),
    )

    with pytest.raises(ValueError, match="layout changed"):
        fhfa_model.get_outstanding_mortgage_interest_rates()


def test_outstanding_mortgage_rates_select_and_add_up_the_buckets(monkeypatch):
    from financetoolkit import Economics
    from financetoolkit.economics.sources import fhfa_model

    content = nmdb_archive(
        fhfa_lock_in_rows() + fhfa_lock_in_rows(market="Enterprise Acquisitions")
    )
    monkeypatch.setattr(
        fhfa_model,
        "get_request",
        lambda url, timeout: type("Response", (), {"content": content})(),
    )
    # The weekly 30-year rate on a new mortgage: only the surveys dated in 2026 Q2 make its
    # quarterly average, 6.5%.
    from financetoolkit.economics.sources import freddie_mac_model

    monkeypatch.delenv("FRED_API_KEY", raising=False)
    surveys = pd.PeriodIndex(
        ["2026-03-26", "2026-04-02", "2026-05-14", "2026-06-25", "2026-07-02"], freq="D"
    )
    monkeypatch.setattr(
        freddie_mac_model,
        "get_mortgage_rate_30_year",
        lambda: pd.DataFrame(
            {"United States": [0.07, 0.064, 0.065, 0.066, 0.06]}, index=surveys
        ),
    )
    economics = Economics(
        start_date="2026-04-01",
        end_date="2026-06-30",
        gmdb_source=False,
        rounding=4,
        use_cached_data=False,
    )

    buckets = economics.get_outstanding_mortgage_rates()
    lock_in = economics.get_outstanding_mortgage_rates(cumulative=True)
    by_balance = economics.get_outstanding_mortgage_rates(
        market="Enterprise", weighting="balance"
    )

    # The start date leaves out 2026 Q1.
    assert buckets.index.tolist() == [pd.Period("2026Q2", "Q")]
    assert buckets.columns.tolist() == [
        "Below 3%",
        "3% to 4%",
        "4% to 5%",
        "5% to 6%",
        "6% and Above",
        "Average Rate",
        "Market Rate",
        "Market Rate Gap",
    ]
    assert buckets.iloc[0].tolist() == pytest.approx(
        [0.192, 0.299, 0.165, 0.12, 0.225, 0.044, 0.065, 0.021]
    )
    # The shares below 4% and 6% the lock-in is reported with: 49.1% and 77.6%.
    assert lock_in.iloc[0].to_dict() == {
        "Below 3%": 0.192,
        "Below 4%": 0.491,
        "Below 5%": 0.656,
        "Below 6%": 0.776,
        "Average Rate": 0.044,
        "Market Rate": 0.065,
        "Market Rate Gap": 0.021,
    }
    assert by_balance.iloc[0]["Below 3%"] == pytest.approx(0.202)
    with pytest.raises(
        InvalidArgumentError, match="no outstanding mortgage statistics"
    ):
        economics.get_outstanding_mortgage_rates(region="Atlantis")


def test_eurostat_job_vacancies_and_capacity_utilization_pin_their_series(
    monkeypatch,
):
    urls = []

    # Excerpts of the jvs_q_r21 and ei_bsin_q_r2 answers for the euro area.
    def fake(url, timeout):
        urls.append(url)
        values = [2.2, 2.1] if "jvs_q_r21" in url else [78.5, 78.2]
        return FakeResponse(json_stat(["EA21"], ["2026-Q2", "2026-Q3"], values))

    monkeypatch.setattr(eurostat_model, "get_request", fake)

    vacancies = eurostat_model.get_job_vacancy_rate("2026-04-01", "2026-09-30")
    utilization = eurostat_model.get_capacity_utilization("2026-04-01", "2026-09-30")

    assert (
        "jvs_q_r21?freq=Q&indic_em=JVR&nace_r2_1=B-T&s_adj=SA&sizeclas=TOTAL" in urls[0]
    )
    assert "ei_bsin_q_r2?freq=Q&indic=BS-ICU-PC&s_adj=SA" in urls[1]
    # Published in percent, returned as decimals.
    assert vacancies.loc[pd.Period("2026Q2", "Q"), "Euro Area"] == pytest.approx(0.022)
    assert utilization.loc[pd.Period("2026Q3", "Q"), "Euro Area"] == pytest.approx(
        0.782
    )


def test_bls_job_openings_rate_is_in_decimals(monkeypatch):
    from financetoolkit.economics.sources import bls_model

    payload = {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [
                {
                    "seriesID": "JTS000000000000000JOR",
                    "data": [
                        {"year": "2026", "period": "M09", "value": "4.5"},
                        {"year": "2026", "period": "M08", "value": "4.3"},
                    ],
                }
            ]
        },
    }
    requested = []

    def respond(url, **kwargs):
        requested.append(json.loads(kwargs["data"])["seriesid"])
        return FakeResponse(payload)

    monkeypatch.setattr(bls_model, "get_request", respond)

    rate = bls_model.get_job_openings_rate()["United States"]

    assert requested == [["JTS000000000000000JOR"]]
    assert rate.to_dict() == {
        pd.Period("2026-08", "M"): pytest.approx(0.043),
        pd.Period("2026-09", "M"): pytest.approx(0.045),
    }


def test_capacity_utilization_reads_total_industry_in_decimals(monkeypatch):
    from financetoolkit.economics.sources import frb_model

    # An excerpt of utl_sa.txt: total industry, then manufacturing.
    text = (
        '"B50001: Total index"\n'
        '"B50001"        2026      75.2420   75.8142   75.6178   76.1792   76.1997   '
        "76.3367   76.4164   76.2918   76.7729   76.3633    0.0000    0.0000\n"
        '"B00004"        2026      74.0000   74.0000\n'
    )
    urls = []

    def fake(url, timeout):
        urls.append(url)
        return FakeResponse(text=text)

    monkeypatch.setattr(frb_model, "get_request", fake)

    utilization = frb_model.get_capacity_utilization()["United States"]

    assert urls == [frb_model.UTILIZATION_URL]
    # October 2026 is the 76.4 percent of the release; November and December are not
    # published yet.
    assert len(utilization) == 10  # noqa: PLR2004
    assert utilization[pd.Period("2026-10", "M")] == pytest.approx(0.763633)


def _labour_capacity_economics(monkeypatch, quarterly=False):
    from financetoolkit import Economics
    from financetoolkit.economics.sources import bls_model, frb_model

    months = pd.period_range("2026-01", "2026-08", freq="M")
    us = pd.DataFrame({"United States": [0.04] * 6 + [0.05, 0.06]}, index=months)
    quarters = pd.period_range("2026Q1", "2026Q2", freq="Q")
    europe = pd.DataFrame(
        {"Euro Area": [0.022, 0.021], "Germany": [0.026, 0.025]}, index=quarters
    )

    monkeypatch.setattr(bls_model, "get_job_openings_rate", lambda: us)
    monkeypatch.setattr(frb_model, "get_capacity_utilization", lambda: us)
    monkeypatch.setattr(eurostat_model, "get_job_vacancy_rate", lambda *a: europe)
    monkeypatch.setattr(eurostat_model, "get_capacity_utilization", lambda *a: europe)

    return Economics(
        start_date="2026-01-01",
        end_date="2026-09-30",
        quarterly=quarterly,
        fred_api_key="",
    )


@pytest.mark.parametrize("method", ["get_job_vacancy_rate", "get_capacity_utilization"])
def test_vacancies_and_utilization_are_monthly_for_the_us_alone(
    monkeypatch, caplog, method
):
    economics = _labour_capacity_economics(monkeypatch)

    # The United States alone is published monthly.
    monthly = getattr(economics, method)(countries="USA")
    assert monthly.index.freqstr.startswith("M")
    assert (
        monthly.loc[pd.Period("2026-08", "M"), "United States"] == 0.06
    )  # noqa: PLR2004

    # With other countries, the quarters, the US as the average of complete quarters.
    quarterly = getattr(economics, method)(countries=["United States", "Germany"])
    assert quarterly.index.freqstr.startswith("Q")
    assert (
        quarterly.loc[pd.Period("2026Q2", "Q"), "United States"] == 0.04
    )  # noqa: PLR2004
    assert quarterly.loc[pd.Period("2026Q2", "Q"), "Germany"] == 0.025  # noqa: PLR2004
    assert pd.Period("2026Q3", "Q") not in quarterly.index

    # Asked for monthly, the quarterly countries are named rather than left out silently.
    with caplog.at_level("WARNING", logger="financetoolkit"):
        getattr(economics, method)(
            countries=["United States", "Germany"], period="monthly"
        )
    assert "Germany is published quarterly" in caplog.text

    # Every country asked for monthly: the quarterly ones are mentioned as well.
    caplog.clear()
    with caplog.at_level("WARNING", logger="financetoolkit"):
        everything = getattr(economics, method)(period="monthly")
    assert list(everything.columns) == ["United States"]
    assert "the other countries quarterly" in caplog.text

    # quarterly=True on the class asks for quarters, also for the United States.
    quarterly_class = _labour_capacity_economics(monkeypatch, quarterly=True)
    assert getattr(quarterly_class, method)(
        countries="United States"
    ).index.freqstr.startswith("Q")

    # Yearly data is not published.
    with pytest.raises(InvalidArgumentError, match="period must be one of"):
        getattr(economics, method)(period="yearly")


def test_a_fred_key_gives_the_full_history_of_the_job_openings_rate(monkeypatch):
    from financetoolkit import Economics
    from financetoolkit.economics.sources import bls_model, fred_model

    months = pd.period_range("2026-01", "2026-03", freq="M")
    monkeypatch.setattr(
        fred_model,
        "get_job_openings_rate",
        lambda *args: pd.DataFrame({"United States": [0.01] * 3}, index=months),
    )
    monkeypatch.setattr(
        bls_model,
        "get_job_openings_rate",
        lambda: pd.DataFrame({"United States": [0.02] * 3}, index=months),
    )

    def rate(key):
        return Economics(
            start_date="2026-01-01", end_date="2026-03-31", fred_api_key=key
        ).get_job_vacancy_rate(countries="United States")

    assert rate("key").iloc[0, 0] == 0.01  # noqa: PLR2004
    assert rate("").iloc[0, 0] == 0.02  # noqa: PLR2004


def test_unit_labour_costs_are_the_oecd_quarterly_index(monkeypatch):
    requested = []

    def fake(oecd_data_string, period_code, start_date=None, end_date=None):
        requested.append((oecd_data_string, period_code))
        return pd.DataFrame(
            {"United States": [128.2185, 128.8209]},
            index=pd.period_range("2026Q1", periods=2, freq="Q"),
        )

    monkeypatch.setattr(oecd_model, "collect_oecd_data", fake)

    from financetoolkit import Economics

    costs = Economics(
        start_date="2026-01-01", end_date="2026-06-30"
    ).get_unit_labour_costs()

    # Employment based, total economy, index 2015 = 100, seasonally adjusted.
    assert requested == [
        ("OECD.SDD.TPS,DSD_PDB@DF_PDB_ULC_Q,1.0/.Q.ULCE._T.IX.V._Z.S.NC", "Q")
    ]
    assert (
        costs.loc[pd.Period("2026Q2", "Q"), "United States"] == 128.8209
    )  # noqa: PLR2004


def world_bank_page(records, page=1, pages=1):
    """A page of a World Bank API answer: the page information, then the records."""
    return [{"page": page, "pages": pages, "per_page": 20000}, records]


# Excerpts of the World Bank list of economies: an economy the Finance Toolkit names
# differently, one it does not know, the euro area and two aggregates, one of them an
# income group with the two letter code the economic calendar uses for the euro area.
WORLD_BANK_ECONOMIES = [
    {"id": "KOR", "iso2Code": "KR", "name": "Korea, Rep.", "region": {"value": "East Asia & Pacific"}},
    {"id": "BHS", "iso2Code": "BS", "name": "Bahamas, The", "region": {"value": "Latin America & Caribbean "}},
    {"id": "CPV", "iso2Code": "CV", "name": "Cabo Verde", "region": {"value": "Sub-Saharan Africa "}},
    {"id": "EMU", "iso2Code": "XC", "name": "Euro area", "region": {"value": "Aggregates"}},
    {"id": "LIC", "iso2Code": "XM", "name": "Low income", "region": {"value": "Aggregates"}},
    {"id": "SAS", "iso2Code": "8S", "name": "South Asia", "region": {"value": "Aggregates"}},
]  # fmt: skip


def wdi_record(code, year, value):
    return {
        "indicator": {"id": "FI.RES.TOTL.MO"},
        "country": {"id": "", "value": ""},
        "countryiso3code": code,
        "date": year,
        "value": value,
    }


def test_world_bank_indicators_name_economies_and_leave_out_aggregates(
    cache, monkeypatch
):
    from financetoolkit.economics.sources import world_bank_model

    # The indicator comes in two pages; values from FI.RES.TOTL.MO.
    indicator_pages = [
        world_bank_page(
            [
                wdi_record("KOR", "2025", 6.95),
                wdi_record("KOR", "2024", None),
                wdi_record("EMU", "2025", 3.49632888594924),
                # Income groups have no three letter code, several per year.
                wdi_record("", "2025", 7.91),
                wdi_record("", "2025", 9.94),
            ],
            page=1,
            pages=2,
        ),
        world_bank_page(
            [
                wdi_record("BHS", "2025", 4.1),
                wdi_record("CPV", "2025", 5.2),
                wdi_record("SAS", "2025", 8.0),
            ],
            page=2,
            pages=2,
        ),
    ]
    urls = []

    def fake(url, timeout):
        urls.append(url)
        if "/indicator/" in url:
            return FakeResponse(indicator_pages[int(url.split("page=")[-1]) - 1])
        return FakeResponse(world_bank_page(WORLD_BANK_ECONOMIES))

    monkeypatch.setattr(world_bank_model, "get_request", fake)

    cover = world_bank_model.get_indicator("FI.RES.TOTL.MO", "import cover")

    assert sum("FI.RES.TOTL.MO" in url for url in urls) == 2  # noqa: PLR2004
    assert sorted(cover.columns) == [
        "Bahamas",
        "Cape Verde",
        "Euro Area",
        "South Korea",
    ]
    assert cover.index.tolist() == [pd.Period("2025", "Y")]
    assert cover.loc["2025", "Euro Area"] == pytest.approx(3.49632888594924)


def test_world_bank_economies_are_named_as_the_global_macro_database_names_them(
    cache, monkeypatch
):
    from financetoolkit.economics.sources import world_bank_model

    # Economies outside the OECD and two letter code lists, as the World Bank names them.
    economies = [
        {"id": code, "iso2Code": iso2, "name": name, "region": {"value": "Latin America & Caribbean "}}
        for code, iso2, name in [
            ("KNA", "KN", "St. Kitts and Nevis"),
            ("LCA", "LC", "St. Lucia"),
            ("VCT", "VC", "St. Vincent and the Grenadines"),
            ("NRU", "NR", "Naoero"),
            ("PRI", "PR", "Puerto Rico (US)"),
            ("VIR", "VI", "Virgin Islands (U.S.)"),
            ("ABW", "AW", "Aruba"),
        ]
    ]  # fmt: skip
    monkeypatch.setattr(
        world_bank_model,
        "get_request",
        lambda url, timeout: FakeResponse(world_bank_page(economies)),
    )

    names = world_bank_model.get_economy_names()["Country"].to_dict()

    # The names of the Global Macro Database data, and the World Bank's name otherwise.
    assert names == {
        "KNA": "Saint Kitts and Nevis",
        "LCA": "Saint Lucia",
        "VCT": "St-Vincent",
        "NRU": "Nauru",
        "PRI": "Puerto Rico",
        "VIR": "US Virgin Islands",
        "ABW": "Aruba",
    }


def test_a_world_bank_error_message_is_a_changed_format(monkeypatch):
    from financetoolkit.economics.sources import world_bank_model

    message = [{"message": [{"id": "120", "key": "Invalid value"}]}]
    monkeypatch.setattr(
        world_bank_model, "get_request", lambda url, timeout: FakeResponse(message)
    )

    with pytest.raises(ValueError, match="not a page of records"):
        world_bank_model.get_indicator("FI.RES.TOTL.XX", "unknown indicator")


def test_foreign_exchange_reserves_are_in_millions_and_import_cover_in_months(
    monkeypatch,
):
    from financetoolkit import Economics
    from financetoolkit.economics.sources import world_bank_model

    years = pd.PeriodIndex(["2024", "2025"], freq="Y")
    published = {
        "FI.RES.TOTL.CD": pd.DataFrame(
            {"China": [3456024808739.12, 3748744952817.94]}, index=years
        ),
        "FI.RES.TOTL.MO": pd.DataFrame(
            {"China": [11.0155252612111, 11.9430464176221]}, index=years
        ),
    }
    monkeypatch.setattr(
        world_bank_model,
        "get_indicator",
        lambda indicator, description: published[indicator],
    )

    economics = Economics(start_date="2024-01-01", end_date="2025-12-31")

    reserves = economics.get_foreign_exchange_reserves(countries="China")
    cover = economics.get_reserves_import_cover(countries=["China", "Atlantis"])

    assert reserves.loc["2025", "China"] == pytest.approx(3748744.9528)
    assert cover.loc["2025", "China"] == pytest.approx(11.943)
    assert "Atlantis" not in cover.columns


def test_an_empty_policy_rate_answer_is_empty_rather_than_a_defect(monkeypatch):
    from financetoolkit import Economics

    _bis_policy_rates(monkeypatch, pd.DataFrame())

    economics = Economics(start_date="2026-10-01", end_date="2026-11-17")

    assert economics.get_central_bank_policy_rate().empty


@pytest.mark.parametrize("status", [500, 503, 403])
def test_an_oecd_server_error_is_reported_as_no_data(monkeypatch, caplog, status):
    def failing(url, timeout=300):  # noqa: ARG001
        response = requests.Response()
        response.status_code = status
        raise requests.exceptions.HTTPError(response=response)

    monkeypatch.setattr(oecd_model, "get_request", failing)

    with caplog.at_level("ERROR"):
        data = oecd_model.collect_oecd_data(
            "SOME/QUERY", "Y", "2020-01-01", "2020-12-31"
        )

    assert data.empty
    assert f"OECD API returned HTTP {status} for 'SOME/QUERY'" in caplog.text


def test_an_empty_oecd_body_is_an_empty_response(monkeypatch, caplog):
    monkeypatch.setattr(
        oecd_model, "get_request", lambda *args, **kwargs: FakeResponse(text="")
    )

    with caplog.at_level("WARNING"):
        data = oecd_model.collect_oecd_data(
            "SOME/QUERY", "Y", "2020-01-01", "2020-12-31"
        )

    assert data.empty
    assert "returned an empty response" in caplog.text


def test_an_unreachable_dnb_listing_is_reported_as_no_data(monkeypatch, caplog):
    from financetoolkit.economics.sources import dnb_model

    def unreachable(*args, **kwargs):
        raise requests.exceptions.ConnectionError("offline")

    monkeypatch.setattr(dnb_model, "get_request", unreachable)

    with caplog.at_level("ERROR"):
        assert dnb_model.get_scenario_set("real_world").empty

    assert "Could not reach DNB" in caplog.text


def test_an_answer_that_is_not_a_zip_archive_is_named(monkeypatch):
    from financetoolkit.economics.sources import fhfa_model

    page = FakeResponse(
        text="<html>Service Unavailable</html>", content_type="text/html"
    )

    with pytest.raises(ValueError, match="is not a zip archive.*text/html"):
        helpers.open_zip_archive(page, "Bank of England nominal yield curve archive")

    monkeypatch.setattr(fhfa_model, "get_request", lambda *args, **kwargs: page)

    with pytest.raises(ValueError, match="FHFA archive .* is not a zip archive"):
        fhfa_model.get_outstanding_mortgage_interest_rates()


@pytest.mark.parametrize(
    ("module", "call"),
    [
        ("bls_model", lambda model: model.get_unemployment_rate()),
        ("portwatch_model", lambda model: model.get_chokepoint_names()),
        (
            "boj_model",
            lambda model: model.request_boj_series(
                "FM01", "STRDCLUCON", "202601", "202609", "overnight rate"
            ),
        ),
    ],
)
def test_a_json_array_instead_of_an_object_is_an_unexpected_format(
    monkeypatch, module, call
):
    model = importlib.import_module(f"financetoolkit.economics.sources.{module}")
    monkeypatch.setattr(
        model, "get_request", lambda *args, **kwargs: FakeResponse(payload=[])
    )

    with pytest.raises(ValueError, match="unexpected format"):
        call(model)


def test_an_empty_business_cycle_chronology_is_refused_and_not_cached(
    cache, monkeypatch
):
    from financetoolkit.economics.sources import nber_model

    monkeypatch.setattr(
        nber_model, "get_request", lambda *args, **kwargs: FakeResponse(payload=[])
    )

    with pytest.raises(ValueError, match="not a non-empty list|not a list"):
        nber_model.get_recession_indicator()

    assert (
        cache.get(
            source="NationalBureauOfEconomicResearch",
            dataset="business_cycle_dates",
            entity="usrec",
        )
        is None
    )


def test_an_answer_that_is_not_json_is_named(monkeypatch):
    def html_page():
        raise requests.exceptions.JSONDecodeError("Expecting value", "<html>", 0)

    with pytest.raises(ValueError, match="BLS answered with something other than JSON"):
        helpers.collect_cached_data("BLS", "series", "json", html_page, "test")


def test_a_page_instead_of_a_workbook_is_named():
    from financetoolkit.utilities.excel_model import excel_file, read_excel

    page = b"<!DOCTYPE html><html><body>Service Unavailable</body></html>"

    with pytest.raises(
        ValueError, match="Shiller stock market data file is not an Excel"
    ):
        read_excel(page, "Shiller stock market data", sheet_name="Data")
    with pytest.raises(ValueError, match="downloaded file is not an Excel workbook"):
        excel_file(page)


def test_unreachable_outstanding_mortgages_are_empty(monkeypatch, caplog):
    from financetoolkit import Economics
    from financetoolkit.economics.sources import fhfa_model

    monkeypatch.setattr(
        fhfa_model, "get_outstanding_mortgage_interest_rates", pd.DataFrame
    )

    economics = Economics(start_date="2026-01-01", end_date="2026-06-30")

    assert economics.get_outstanding_mortgage_rates().empty
    # Without a misleading follow-up error about a missing region.
    assert "could not be calculated" not in caplog.text


def test_the_treasury_years_without_data_are_reported_in_one_line(monkeypatch, caplog):
    def unreachable(*args, **kwargs):
        raise requests.exceptions.ConnectionError("offline")

    monkeypatch.setattr(helpers, "RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(treasury_model, "get_request", unreachable)

    with caplog.at_level("DEBUG"):
        curve = treasury_model.get_yield_curve("nominal", "2021-01-01", "2023-12-31")

    errors = [record for record in caplog.records if record.levelname == "ERROR"]
    assert curve.empty
    assert len(errors) == 1
    assert "2021, 2022, 2023" in errors[0].getMessage()


def test_an_unreachable_yahoo_finance_stops_the_futures_scan(monkeypatch, caplog):
    from financetoolkit import yfinance_model

    attempts = []

    class Ticker:
        def __init__(self, ticker):
            attempts.append(ticker)

        def history(self, **kwargs):
            raise OSError("Could not resolve host")

    monkeypatch.setattr(yfinance_model.yf, "Ticker", Ticker)

    with caplog.at_level("ERROR"):
        curve = yfinance_model.get_commodity_forward_curve(
            "Crude Oil", "2026-01-01", "2026-06-30"
        )

    assert curve.empty
    assert len(attempts) == 1
    assert "Could not reach Yahoo Finance for the Crude Oil futures" in caplog.text


def test_commodity_curves_without_any_prices_are_empty(monkeypatch, caplog):
    from financetoolkit import Economics, yfinance_model

    monkeypatch.setattr(
        yfinance_model, "get_commodity_forward_curve", lambda *args: pd.DataFrame()
    )
    economics = Economics(start_date="2026-01-01", end_date="2026-06-30")

    assert economics.get_commodity_forward_curve().empty
    assert economics.get_commodity_term_structure().empty
    assert "could not be calculated" not in caplog.text


def test_a_failing_first_calendar_window_stops_the_others(monkeypatch):
    from financetoolkit.economics.sources import fmp_model as economics_fmp_model

    requested = []

    def unreachable(url, user_subscription="Free"):  # noqa: ARG001
        requested.append(url)
        economics_fmp_model.error_model.report_request_failure()
        return pd.DataFrame()

    monkeypatch.setattr(economics_fmp_model, "get_cached_financial_data", unreachable)

    calendar = economics_fmp_model.get_economic_calendar(
        api_key="KEY", start_date="2020-01-01", end_date="2025-12-31"
    )

    assert calendar.empty
    assert len(requested) == 1


@pytest.mark.parametrize(
    ("method", "arguments"),
    [
        ("get_unemployment_rate", {"period": "weekly"}),
        ("get_exchange_rates", {"period": "hourly"}),
        ("get_short_term_interest_rate", {"period": "daily"}),
        ("get_long_term_interest_rate", {"period": "biweekly"}),
        ("get_yield_curve_slope", {"period": "weekly"}),
        ("get_inflation_rate", {"period": "hourly"}),
        ("get_real_interest_rate", {"rate_type": "medium_term"}),
        ("get_government_debt", {"level": "local"}),
        ("get_life_table", {"measure": "nonsense"}),
        ("get_life_table", {"sex": "other"}),
        ("get_debt_service_ratio", {"borrowers": "banks"}),
        ("get_stress_test_scenario", {"scenario": "severe"}),
        ("get_stress_test_scenario", {"authority": "ecb"}),
        ("get_climate_scenario", {"variable": "nonsense"}),
        ("get_climate_scenario", {"model": "nonsense"}),
        ("get_long_run_asset_returns", {}),
        ("get_long_run_asset_returns", {"asset_class": "gold", "accept_licence": True}),
        ("get_scenario_set", {"authority": "ecb"}),
        ("get_scenario_set", {"measure": "nonsense"}),
        ("get_chokepoint_transits", {"measure": "volume"}),
        ("get_outstanding_mortgage_rates", {"market": "nonsense"}),
        ("get_total_consumption", {"component": "nonsense"}),
        ("get_total_consumption", {"inflation_adjusted": True, "usd": True}),
        ("get_purchasing_power", {"overnight_rate": True, "period": "yearly"}),
        ("get_asset_class_proxies", {"asset_classes": "Gold Bars"}),
    ],
)
def test_an_invalid_economics_argument_names_the_valid_values(
    monkeypatch, method, arguments
):
    from financetoolkit import Economics
    from financetoolkit.economics.sources import gmdb_model as gmdb
    from financetoolkit.utilities.error_model import InvalidArgumentError

    # Nothing is retrieved: the call is refused before any source is chosen.
    def no_data(*args, **kwargs):
        raise AssertionError("A source was asked for data.")

    monkeypatch.setattr(gmdb, "get_shared_dataset", no_data)

    economics = Economics(start_date="2026-01-01", end_date="2026-06-30")

    with pytest.raises(InvalidArgumentError):
        getattr(economics, method)(**arguments)


def test_an_invalid_economics_date_is_an_invalid_argument():
    from financetoolkit import Economics
    from financetoolkit.utilities.error_model import InvalidArgumentError

    with pytest.raises(InvalidArgumentError, match="valid start date"):
        Economics(start_date="2026-13-01")


def test_a_failing_yield_decomposition_part_gives_an_empty_result(monkeypatch):
    from financetoolkit import Economics

    economics = Economics(start_date="2026-01-01", end_date="2026-06-30")
    # handle_errors answers a failed part with an empty Series.
    monkeypatch.setattr(
        economics, "get_real_yield_curve", lambda **kwargs: pd.Series(dtype=object)
    )
    monkeypatch.setattr(
        economics,
        "get_breakeven_inflation_expectations",
        lambda **kwargs: pd.Series(dtype=object),
    )

    assert economics.get_yield_decomposition().empty


def test_proxies_without_prices_are_reported_in_one_line(monkeypatch, caplog):
    import logging

    from financetoolkit import Economics
    from financetoolkit.economics.history import history_controller

    def no_prices(tickers, **kwargs):
        for ticker in tickers:
            logging.getLogger("yfinance").error("$%s: possibly delisted", ticker)
        return pd.DataFrame(), None

    monkeypatch.setattr(
        history_controller.historical_model, "get_historical_data", no_prices
    )
    economics = Economics(start_date="2026-01-01", end_date="2026-06-30")

    with caplog.at_level("ERROR"):
        assert economics.get_asset_class_proxies().empty

    assert "possibly delisted" not in caplog.text
    assert caplog.text.count("No prices could be retrieved for the proxies of") == 1


def test_an_unreachable_source_is_asked_once_within_a_call(monkeypatch, caplog):
    from financetoolkit import Economics
    from financetoolkit.economics.sources import cboe_model

    requested = []

    def unreachable(url, **kwargs):  # noqa: ARG001
        requested.append(url)
        raise requests.exceptions.ConnectionError(f"cannot reach {url}")

    monkeypatch.setattr(cboe_model, "get_request", unreachable)
    monkeypatch.setattr(helpers.time, "sleep", lambda seconds: None)
    economics = Economics(start_date="2026-01-01", end_date="2026-06-30")

    assert economics.get_implied_volatility(markets="US Equity").empty
    # The first index is tried with its retries; the other four are not requested.
    assert len(requested) == helpers.RETRY_ATTEMPTS
    assert caplog.text.count("Could not reach Cboe") == 1

    # The next call asks again.
    economics.get_implied_volatility(markets="US Equity")
    assert len(requested) == 2 * helpers.RETRY_ATTEMPTS


def test_a_missing_series_does_not_stop_the_source(monkeypatch):
    def missing(*args):  # noqa: ARG001
        response = requests.Response()
        response.status_code = 404
        raise requests.exceptions.HTTPError("404 Not Found", response=response)

    with helpers.stop_at_unreachable_sources():
        for entity in ("first", "second"):
            assert helpers.collect_cached_data(
                source="Cboe",
                dataset="index",
                entity=entity,
                fetch=missing,
                description=f"Cboe {entity} index",
            ).empty
        assert "Cboe" not in helpers._UNREACHABLE_SOURCES.get()
