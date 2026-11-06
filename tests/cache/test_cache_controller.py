"""Cache Controller Tests"""

import contextlib
import sqlite3
import threading
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from financetoolkit.cache import cache_controller, policy_model, sqlite_model

# Imported before the autouse fixture of conftest.py replaces it for every test.
from financetoolkit.cache.policy_model import get_last_closed_session


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


def test_a_range_into_the_future_is_covered_up_to_today(cache, monkeypatch):
    """Test that a future end neither claims unpublished days nor moves the revision window."""
    monkeypatch.setitem(
        policy_model.POLICIES,
        f"{policy_model.FINANCIAL_MODELING_PREP}.historical",
        policy_model.CachePolicy(ttl_seconds=0, revision_days=7),
    )

    today = date.today()
    start = today - timedelta(days=365)
    future = today + timedelta(days=60)

    cache.store(
        source="FinancialModelingPrep",
        dataset="historical",
        entity="AAPL",
        data=create_frame(start.isoformat(), today.isoformat()),
        start=start.isoformat(),
        end=future.isoformat(),
    )

    later = today + timedelta(days=10)
    plan = cache.plan(
        source="FinancialModelingPrep",
        dataset="historical",
        entities=["AAPL"],
        start=start.isoformat(),
        end=future.isoformat(),
        reference_date=later,
    )

    # The days published since the store are asked for, not the 60 days that ran ahead.
    assert plan.missing["AAPL"] == [(today + timedelta(days=1), later)]


def test_a_relative_cache_location_is_made_absolute(tmp_path, monkeypatch):
    """Test that a relative folder keeps meaning the same database after a chdir."""
    monkeypatch.chdir(tmp_path)

    location = cache_controller.resolve_cache_location("my_cache")

    assert location.is_absolute()
    assert location == tmp_path / "my_cache" / cache_controller.DATABASE_FILE_NAME


def test_stale_entries_of_every_source_are_removed_once_a_day(tmp_path, monkeypatch):
    """Test that point-in-time entries nobody refreshes are evicted, the MCP's excepted."""
    cache_controller.reset_cache_registry()
    location = tmp_path / "cache.db"
    cache = cache_controller.Cache(location=location)
    for source in ("FinancialModelingPrep", "OECD", "MCP"):
        cache.set(source=source, dataset="profile", entity="A", data={"a": 1})
    cache.store(
        source="FinancialModelingPrep",
        dataset="historical",
        entity="A",
        data=create_frame("2020-01-01", "2020-01-31"),
        start="2020-01-01",
        end="2020-01-31",
    )

    # Entries written a year ago are not evicted again within a day of the last run.
    with sqlite3.connect(location) as connection:
        connection.execute(
            "UPDATE blobs SET updated_at = updated_at - ?", (365 * policy_model.DAY,)
        )
    cache_controller.Cache(location=location)

    contents = {record["source"] for record in cache.get_contents()}
    assert contents == {"FinancialModelingPrep", "OECD", "MCP"}

    with sqlite3.connect(location) as connection:
        connection.execute("DELETE FROM meta WHERE name = 'last_blob_eviction'")
    cache_controller.Cache(location=location)

    remaining = {
        (record["source"], record["dataset"]) for record in cache.get_contents()
    }
    # The price history stays, and so do the MCP server's own responses.
    assert remaining == {("FinancialModelingPrep", "historical"), ("MCP", "profile")}

    cache_controller.reset_cache_registry()


def test_reads_do_not_wait_for_a_writer_of_the_same_process(cache):
    """Test that a writer queueing for the database does not hold up the readers."""
    cache.set(source="OECD", dataset="profile", entity="A", data={"a": 1})
    answers: list = []

    # The instance lock is what a writer holds while it waits for the database.
    with cache._backend._lock:  # noqa: SLF001
        reader = threading.Thread(
            target=lambda: answers.append(
                cache.get(source="OECD", dataset="profile", entity="A")
            )
        )
        reader.start()
        reader.join(timeout=5)

    assert answers == [{"a": 1}]


def test_opening_a_cache_does_not_hold_the_registry_lock(tmp_path, monkeypatch):
    """Test that a slow eviction at one location does not block the registry for others."""
    cache_controller.reset_cache_registry()
    seen = []

    def remove_stale_entries(self):
        # Another thread could not take the registry lock if it were held here.
        seen.append(cache_controller._REGISTRY_LOCK.acquire(blocking=False))
        if seen[-1]:
            cache_controller._REGISTRY_LOCK.release()

    monkeypatch.setattr(
        cache_controller.Cache, "_remove_stale_entries", remove_stale_entries
    )

    first = cache_controller.get_cache(tmp_path / "a")

    assert seen == [True]
    assert cache_controller.get_cache(tmp_path / "a") is first
    cache_controller.reset_cache_registry()


def test_coverage_and_payload_are_read_in_one_transaction(tmp_path, monkeypatch):
    """Test that both reads run inside one BEGIN/COMMIT, so they share a snapshot."""
    backend = sqlite_model.SQLiteBackend(tmp_path / "cache.db")
    statements = []
    connect = backend._connect

    @contextlib.contextmanager
    def tracing_connect(write=False):
        with connect(write) as connection:
            connection.set_trace_callback(statements.append)
            yield connection

    monkeypatch.setattr(backend, "_connect", tracing_connect)

    backend.read_entity_state("key", "entity", 0.0)

    queries = [statement.split()[0] for statement in statements]
    assert queries == ["BEGIN", "SELECT", "SELECT", "COMMIT"]


@pytest.mark.parametrize(
    ("moment", "last_closed"),
    [
        # 18:00 in Amsterdam, during the New York session.
        ("2026-11-05T18:00:00+01:00", date(2026, 11, 4)),
        # 22:50 in Amsterdam, after the New York close.
        ("2026-11-05T22:50:00+01:00", date(2026, 11, 5)),
        # Just after midnight in Tokyo, still the evening before in New York.
        ("2026-11-06T08:00:00+09:00", date(2026, 11, 5)),
    ],
)
def test_a_session_counts_as_closed_from_half_past_four_new_york_time(
    moment, last_closed
):
    assert get_last_closed_session(datetime.fromisoformat(moment)) == last_closed


def test_a_bar_of_an_unclosed_session_is_requested_again(cache, monkeypatch):
    """A price fetched during today's session is a snapshot, not the close."""
    today = date.today()
    start = today - timedelta(days=30)
    monkeypatch.setattr(
        policy_model, "get_last_closed_session", lambda now=None: today - timedelta(1)
    )

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

    # Only today is asked for again; the closed sessions before it stay cached.
    assert plan.missing == {"AAPL": [(today, today)]}
