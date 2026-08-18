"""Cache Controller Tests"""

from datetime import date, timedelta

import pandas as pd
import pytest

from financetoolkit.cache import cache_controller, policy_model


@pytest.fixture(name="cache")
def fixture_cache(tmp_path):
    """Provide a cache backed by an isolated temporary database."""
    cache_controller.reset_cache_registry()

    yield cache_controller.Cache(location=tmp_path / "cache.db")

    cache_controller.reset_cache_registry()


def create_frame(start: str, end: str, value: float = 1.0) -> pd.DataFrame:
    """Build a daily indexed frame spanning the given inclusive range."""
    index = pd.period_range(start=start, end=end, freq="D")

    return pd.DataFrame({"Close": value}, index=index)


def test_plan_on_empty_cache_requests_everything(cache):
    """Test that an empty cache reports the full range as missing."""
    plan = cache.plan(
        source="FinancialModelingPrep",
        dataset="historical",
        entities=["AAPL", "MSFT"],
        start="2020-01-01",
        end="2020-12-31",
    )

    assert plan.entities_to_fetch == ["AAPL", "MSFT"]
    assert plan.missing["AAPL"] == [(date(2020, 1, 1), date(2020, 12, 31))]
    assert not plan.cached


def test_widening_the_date_range_only_fetches_the_new_years(cache):
    """Test the core behaviour: an earlier start date fetches only the new period."""
    cache.store(
        source="FinancialModelingPrep",
        dataset="historical",
        entity="AAPL",
        data=create_frame("2020-01-01", "2020-12-31"),
        start="2020-01-01",
        end="2020-12-31",
    )

    plan = cache.plan(
        source="FinancialModelingPrep",
        dataset="historical",
        entities=["AAPL"],
        start="2018-01-01",
        end="2020-12-31",
        reference_date=date(2021, 6, 1),
    )

    assert plan.missing["AAPL"] == [(date(2018, 1, 1), date(2019, 12, 31))]
    assert not plan.cached["AAPL"].empty


def test_adding_a_ticker_leaves_the_others_cached(cache):
    """Test that a new ticker does not invalidate the tickers already stored."""
    cache.store(
        source="FinancialModelingPrep",
        dataset="historical",
        entity="AAPL",
        data=create_frame("2020-01-01", "2020-12-31"),
        start="2020-01-01",
        end="2020-12-31",
    )

    plan = cache.plan(
        source="FinancialModelingPrep",
        dataset="historical",
        entities=["AAPL", "MSFT"],
        start="2020-01-01",
        end="2020-12-31",
        reference_date=date(2021, 6, 1),
    )

    assert plan.entities_to_fetch == ["MSFT"]
    assert "AAPL" not in plan.missing


def test_unrelated_parameters_do_not_share_a_key(cache):
    """Test that a parameter changing the data shape produces a separate entry."""
    cache.store(
        source="FinancialModelingPrep",
        dataset="historical",
        entity="AAPL",
        data=create_frame("2020-01-01", "2020-12-31"),
        start="2020-01-01",
        end="2020-12-31",
        parameters={"interval": "1d"},
    )

    plan = cache.plan(
        source="FinancialModelingPrep",
        dataset="historical",
        entities=["AAPL"],
        start="2020-01-01",
        end="2020-12-31",
        parameters={"interval": "1wk"},
        reference_date=date(2021, 6, 1),
    )

    assert plan.entities_to_fetch == ["AAPL"]


def test_store_merges_and_lets_the_newest_values_win(cache):
    """Test that overlapping ranges merge with the fresher values taking priority."""
    cache.store(
        source="FinancialModelingPrep",
        dataset="historical",
        entity="AAPL",
        data=create_frame("2020-01-01", "2020-06-30", value=1.0),
        start="2020-01-01",
        end="2020-06-30",
    )
    cache.store(
        source="FinancialModelingPrep",
        dataset="historical",
        entity="AAPL",
        data=create_frame("2020-06-01", "2020-12-31", value=2.0),
        start="2020-06-01",
        end="2020-12-31",
    )

    plan = cache.plan(
        source="FinancialModelingPrep",
        dataset="historical",
        entities=["AAPL"],
        start="2020-01-01",
        end="2020-12-31",
        reference_date=date(2021, 6, 1),
    )

    stored = plan.cached["AAPL"]

    assert plan.fully_cached
    assert len(stored) == 366
    assert stored.loc[pd.Period("2020-01-02", freq="D"), "Close"] == 1.0
    assert stored.loc[pd.Period("2020-06-15", freq="D"), "Close"] == 2.0


def test_empty_response_is_remembered_as_answered(cache):
    """Test that a range the source has no data for is not requested repeatedly."""
    cache.store(
        source="FinancialModelingPrep",
        dataset="historical",
        entity="NEWCO",
        data=pd.DataFrame(),
        start="1990-01-01",
        end="1999-12-31",
    )

    plan = cache.plan(
        source="FinancialModelingPrep",
        dataset="historical",
        entities=["NEWCO"],
        start="1990-01-01",
        end="1999-12-31",
        reference_date=date(2021, 6, 1),
    )

    assert plan.fully_cached


def test_rerun_inside_the_time_to_live_makes_no_request(cache):
    """Test that repeating a request within the TTL is served entirely from cache."""
    today = date.today()
    start = today - timedelta(days=30)

    cache.store(
        source="FinancialModelingPrep",
        dataset="historical",
        entity="AAPL",
        data=create_frame(start.isoformat(), today.isoformat()),
        start=start.isoformat(),
        end=today.isoformat(),
    )

    plan = cache.plan(
        source="FinancialModelingPrep",
        dataset="historical",
        entities=["AAPL"],
        start=start.isoformat(),
        end=today.isoformat(),
    )

    assert plan.fully_cached


def test_stale_coverage_only_refetches_the_revision_window(cache, monkeypatch):
    """Test that refreshing a stale range asks for the tail rather than everything."""
    monkeypatch.setitem(
        policy_model.POLICIES,
        f"{policy_model.FINANCIAL_MODELING_PREP}.historical",
        policy_model.CachePolicy(ttl_seconds=0, revision_days=7),
    )

    today = date.today()
    start = today - timedelta(days=365)

    cache.store(
        source="FinancialModelingPrep",
        dataset="historical",
        entity="AAPL",
        data=create_frame(start.isoformat(), today.isoformat()),
        start=start.isoformat(),
        end=today.isoformat(),
    )

    plan = cache.plan(
        source="FinancialModelingPrep",
        dataset="historical",
        entities=["AAPL"],
        start=start.isoformat(),
        end=today.isoformat(),
    )

    assert plan.missing["AAPL"] == [(today - timedelta(days=7), today)]


def test_stale_coverage_without_revision_window_refetches_everything(
    cache, monkeypatch
):
    """Test that a zero revision window asks for the whole stale range again."""
    monkeypatch.setitem(
        policy_model.POLICIES,
        f"{policy_model.FINANCIAL_MODELING_PREP}.statements",
        policy_model.CachePolicy(ttl_seconds=0),
    )

    cache.store(
        source="FinancialModelingPrep",
        dataset="statements",
        entity="AAPL",
        data=create_frame("2020-01-01", "2020-12-31"),
        start="2020-01-01",
        end="2020-12-31",
    )

    plan = cache.plan(
        source="FinancialModelingPrep",
        dataset="statements",
        entities=["AAPL"],
        start="2020-01-01",
        end="2020-12-31",
    )

    assert plan.missing["AAPL"] == [(date(2020, 1, 1), date(2020, 12, 31))]


def test_get_fetch_span_covers_every_gap(cache):
    """Test that disjoint gaps collapse into a single requestable range."""
    cache.store(
        source="FinancialModelingPrep",
        dataset="historical",
        entity="AAPL",
        data=create_frame("2020-01-01", "2020-03-31"),
        start="2020-01-01",
        end="2020-03-31",
    )

    plan = cache.plan(
        source="FinancialModelingPrep",
        dataset="historical",
        entities=["AAPL"],
        start="2019-01-01",
        end="2020-12-31",
        reference_date=date(2021, 6, 1),
    )

    assert plan.get_fetch_span("AAPL") == (date(2019, 1, 1), date(2020, 12, 31))
    assert plan.get_fetch_span("MSFT") is None


def test_blob_round_trip_and_expiry(cache, monkeypatch):
    """Test that point-in-time data round-trips and expires with its policy."""
    profile = pd.DataFrame({"Sector": ["Technology"]}, index=["AAPL"])

    cache.set(
        source="FinancialModelingPrep", dataset="profile", entity="AAPL", data=profile
    )

    pd.testing.assert_frame_equal(
        cache.get(source="FinancialModelingPrep", dataset="profile", entity="AAPL"),
        profile,
    )

    monkeypatch.setitem(
        policy_model.POLICIES,
        f"{policy_model.FINANCIAL_MODELING_PREP}.profile",
        policy_model.CachePolicy(ttl_seconds=0),
    )

    assert (
        cache.get(source="FinancialModelingPrep", dataset="profile", entity="AAPL")
        is None
    )


def test_invalidate_removes_a_single_entity(cache):
    """Test that invalidating one entity leaves the others intact."""
    for ticker in ("AAPL", "MSFT"):
        cache.store(
            source="FinancialModelingPrep",
            dataset="historical",
            entity=ticker,
            data=create_frame("2020-01-01", "2020-12-31"),
            start="2020-01-01",
            end="2020-12-31",
        )

    cache.invalidate(
        source="FinancialModelingPrep", dataset="historical", entity="AAPL"
    )

    plan = cache.plan(
        source="FinancialModelingPrep",
        dataset="historical",
        entities=["AAPL", "MSFT"],
        start="2020-01-01",
        end="2020-12-31",
        reference_date=date(2021, 6, 1),
    )

    assert plan.entities_to_fetch == ["AAPL"]


def test_statement_layout_caches_along_the_column_axis(cache):
    """Test that frames with reporting periods as columns are cached correctly."""
    statement = pd.DataFrame(
        {"2020-12-31": [100.0, 50.0], "2021-12-31": [120.0, 60.0]},
        index=["Revenue", "Net Income"],
    )

    cache.store(
        source="FinancialModelingPrep",
        dataset="statements",
        entity="AAPL",
        data=statement,
        start="2020-01-01",
        end="2021-12-31",
        date_axis=1,
    )

    plan = cache.plan(
        source="FinancialModelingPrep",
        dataset="statements",
        entities=["AAPL"],
        start="2021-01-01",
        end="2021-12-31",
        date_axis=1,
        reference_date=date(2030, 1, 1),
    )

    assert list(plan.cached["AAPL"].columns) == ["2021-12-31"]


def test_disabled_cache_never_stores_anything(tmp_path):
    """Test that a disabled cache misses on every read and ignores every write."""
    cache = cache_controller.Cache(location=tmp_path / "cache.db", enabled=False)

    cache.store(
        source="FinancialModelingPrep",
        dataset="historical",
        entity="AAPL",
        data=create_frame("2020-01-01", "2020-12-31"),
        start="2020-01-01",
        end="2020-12-31",
    )

    plan = cache.plan(
        source="FinancialModelingPrep",
        dataset="historical",
        entities=["AAPL"],
        start="2020-01-01",
        end="2020-12-31",
    )

    assert plan.entities_to_fetch == ["AAPL"]
    assert not plan.cached


def test_get_cache_reuses_instances_per_location(tmp_path):
    """Test that the registry hands out one instance per database path."""
    cache_controller.reset_cache_registry()

    first = cache_controller.get_cache(tmp_path / "cache.db")
    second = cache_controller.get_cache(tmp_path / "cache.db")

    assert first is second

    cache_controller.reset_cache_registry()


def test_resolve_cache_location_accepts_directories_and_files(tmp_path):
    """Test that both a directory and an explicit database file are accepted."""
    assert cache_controller.resolve_cache_location(tmp_path).name == (
        cache_controller.DATABASE_FILE_NAME
    )
    assert cache_controller.resolve_cache_location(tmp_path / "own.db").name == "own.db"


def test_parse_use_cached_data():
    """Test the interpretation of the Toolkit's use_cached_data argument."""
    assert cache_controller.parse_use_cached_data(False) == (False, None)
    assert cache_controller.parse_use_cached_data(True) == (True, None)
    assert cache_controller.parse_use_cached_data("folder") == (True, "folder")
