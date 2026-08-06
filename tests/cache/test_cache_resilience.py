"""Cache Resilience Tests"""

import os
import sqlite3
import threading

import pandas as pd
import pytest

from financetoolkit.cache import cache_controller


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Keep the shared instance registry from leaking between tests."""
    cache_controller.reset_cache_registry()

    yield

    cache_controller.clear_active_cache()
    cache_controller.reset_cache_registry()


def create_frame(value: float = 1.0) -> pd.DataFrame:
    """Build a small daily frame to store."""
    return pd.DataFrame(
        {"Close": value}, index=pd.period_range("2020-01-01", "2020-12-31", freq="D")
    )


def test_a_corrupt_database_does_not_break_the_caller(tmp_path, caplog):
    """Test that an unopenable cache degrades to no caching instead of raising.

    A cache is an optimization. If the file is corrupt, the disk is full or the
    path is occupied by something else, the toolkit must still run.
    """
    location = tmp_path / "corrupt.db"
    location.write_bytes(b"this is definitely not a sqlite database")

    with caplog.at_level("WARNING"):
        cache = cache_controller.Cache(location=location)

    assert not cache.enabled
    assert "Continuing without caching" in caplog.text

    # Every operation still has to be safe to call.
    cache.store(
        source="FinancialModelingPrep",
        dataset="historical",
        entity="AAPL",
        data=create_frame(),
        start="2020-01-01",
        end="2020-12-31",
    )
    cache.set(
        source="FinancialModelingPrep",
        dataset="profile",
        entity="AAPL",
        data=create_frame(),
    )

    assert (
        cache.get(source="FinancialModelingPrep", dataset="profile", entity="AAPL")
        is None
    )
    assert cache.get_contents() == []
    assert cache.plan(
        source="FinancialModelingPrep",
        dataset="historical",
        entities=["AAPL"],
        start="2020-01-01",
        end="2020-12-31",
    ).entities_to_fetch == ["AAPL"]


def test_a_read_only_location_does_not_break_the_caller(tmp_path, caplog):
    """Test that a directory that cannot be written to degrades gracefully."""
    directory = tmp_path / "readonly"
    directory.mkdir()
    os.chmod(directory, 0o500)

    try:
        with caplog.at_level("WARNING"):
            cache = cache_controller.Cache(location=directory / "cache.db")

        assert not cache.enabled
        assert "Continuing without caching" in caplog.text
    finally:
        os.chmod(directory, 0o700)


def test_an_unreadable_payload_is_a_miss_not_a_crash(tmp_path):
    """Test that a corrupted stored value is reported as absent."""
    cache = cache_controller.Cache(location=tmp_path / "cache.db")
    cache.set(
        source="FinancialModelingPrep",
        dataset="profile",
        entity="AAPL",
        data=create_frame(),
    )

    with sqlite3.connect(cache.location) as connection:
        connection.execute("UPDATE blobs SET payload = ?", (b"garbage",))

    assert (
        cache.get(source="FinancialModelingPrep", dataset="profile", entity="AAPL")
        is None
    )


def test_an_unreadable_series_payload_is_planned_as_missing(tmp_path):
    """Test that a corrupted time series is re-requested rather than raising."""
    cache = cache_controller.Cache(location=tmp_path / "cache.db")
    cache.store(
        source="FinancialModelingPrep",
        dataset="historical",
        entity="AAPL",
        data=create_frame(),
        start="2020-01-01",
        end="2020-12-31",
    )

    with sqlite3.connect(cache.location) as connection:
        connection.execute("UPDATE series SET payload = ?", (b"garbage",))

    plan = cache.plan(
        source="FinancialModelingPrep",
        dataset="historical",
        entities=["AAPL"],
        start="2020-01-01",
        end="2020-12-31",
    )

    assert plan.entities_to_fetch == ["AAPL"]


def test_concurrent_writers_do_not_corrupt_the_cache(tmp_path):
    """Test that many threads writing at once leave every entity readable."""
    cache = cache_controller.Cache(location=tmp_path / "cache.db")
    errors: list[str] = []

    def worker(index: int) -> None:
        try:
            for _ in range(5):
                cache.store(
                    source="FinancialModelingPrep",
                    dataset="historical",
                    entity=f"T{index}",
                    data=create_frame(float(index)),
                    start="2020-01-01",
                    end="2020-12-31",
                )
        except Exception as error:  # noqa: BLE001
            errors.append(f"{type(error).__name__}: {error}")

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors

    plan = cache.plan(
        source="FinancialModelingPrep",
        dataset="historical",
        entities=[f"T{index}" for index in range(8)],
        start="2020-01-01",
        end="2020-12-31",
    )

    assert plan.fully_cached


def test_a_disabled_cache_never_creates_a_database(tmp_path):
    """Test that opting out leaves no file behind."""
    location = tmp_path / "never.db"
    cache = cache_controller.Cache(location=location, enabled=False)

    cache.store(
        source="FinancialModelingPrep",
        dataset="historical",
        entity="AAPL",
        data=create_frame(),
        start="2020-01-01",
        end="2020-12-31",
    )

    assert not location.exists()


def test_coverage_rows_are_compacted(tmp_path):
    """Test that repeated writes do not grow the coverage table without bound."""
    cache = cache_controller.Cache(location=tmp_path / "cache.db")

    for end in pd.date_range("2020-02-01", periods=40, freq="D").strftime("%Y-%m-%d"):
        cache.store(
            source="FinancialModelingPrep",
            dataset="historical",
            entity="AAPL",
            data=create_frame(),
            start="2020-01-01",
            end=end,
        )

    assert cache.get_statistics()["coverage_entries"] < 40
