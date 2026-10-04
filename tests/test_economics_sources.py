"""Offline tests for the keyless high-frequency Economics sources."""

import pandas as pd
import pytest
import requests

from financetoolkit.cache.cache_controller import (
    Cache,
    clear_active_cache,
    set_active_cache,
)
from financetoolkit.economics import (
    bis_model,
    boe_model,
    boj_model,
    eurostat_model,
    helpers,
    mof_model,
    ons_model,
    sbj_model,
    treasury_model,
)


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
    from financetoolkit.economics import bls_model

    payload = {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [
                {
                    "data": [
                        {"year": "2026", "period": "M09", "value": "4.2"},
                        {"year": "2026", "period": "M13", "value": "4.1"},
                        {"year": "2026", "period": "M08", "value": "4.1"},
                    ]
                }
            ]
        },
    }
    monkeypatch.setattr(
        bls_model, "get_request", lambda url, timeout: FakeResponse(payload)
    )

    rate = bls_model.get_unemployment_rate()

    # M13 is an annual average, not a month.
    assert rate["United States"].tolist() == [
        pytest.approx(0.041),
        pytest.approx(0.042),
    ]


def test_a_refused_bls_request_is_not_a_format_change(monkeypatch):
    from financetoolkit.economics import bls_model

    payload = {
        "status": "REQUEST_NOT_PROCESSED",
        "message": ["Daily threshold reached."],
    }
    monkeypatch.setattr(
        bls_model, "get_request", lambda url, timeout: FakeResponse(payload)
    )

    # Treated like an unreachable source: no data (or the cached copy), not an error.
    assert bls_model.get_unemployment_rate().empty


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
    from financetoolkit.economics import freddie_mac_model

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
    from financetoolkit.economics import frb_model

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
    from financetoolkit.economics import nber_model

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
    from financetoolkit.economics import (
        economics_controller,
        fred_model,
        freddie_mac_model,
    )

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
        economics_controller.gmdb_model,
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
