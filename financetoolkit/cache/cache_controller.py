"""Cache Module"""

__docformat__ = "google"

import os
import platform
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd

from financetoolkit.cache import (
    coverage_model,
    frame_model,
    policy_model,
    serialization_model,
)
from financetoolkit.cache.sqlite_model import SCHEMA_VERSION, SQLiteBackend
from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()

DATABASE_FILE_NAME = "financetoolkit_cache.db"

# Compacted into merged intervals past this many rows, to bound table growth.
COVERAGE_COMPACTION_THRESHOLD = 16

_CACHE_REGISTRY: dict[str, "Cache"] = {}
_REGISTRY_LOCK = Lock()

# Published once per process: OECD/FRED/ECB/Fed reach the cache via free functions.
_ACTIVE_CACHE: "Cache | None" = None


def set_active_cache(cache: "Cache | None") -> None:
    """
    Publish the cache that free functions across the toolkit should use.

    A disabled cache never replaces an enabled one. Controllers are constructed
    freely and often, including inside a long running server that set up caching
    once at startup, and one of them opting out must not silently switch caching
    off for everything else in the process. Use ``clear_active_cache`` to actually
    withdraw one.

    Args:
        cache (Cache | None): The cache to make active. None, or a disabled cache,
            leaves any previously published cache in place.
    """
    global _ACTIVE_CACHE  # noqa: PLW0603

    if cache is None or not cache.enabled:
        return

    _ACTIVE_CACHE = cache


def clear_active_cache() -> None:
    """
    Withdraw the published cache so free functions stop caching.

    Separate from ``set_active_cache`` so that withdrawing is always deliberate.
    """
    global _ACTIVE_CACHE  # noqa: PLW0603
    _ACTIVE_CACHE = None


def get_active_cache() -> "Cache | None":
    """
    Return the cache published by the most recently initialized controller.

    Returns:
        Cache | None: The active cache, or None when caching is disabled or when no
            controller has published one yet.
    """
    if _ACTIVE_CACHE is not None and _ACTIVE_CACHE.enabled:
        return _ACTIVE_CACHE

    return None


def get_default_cache_location() -> Path:
    """
    Return the platform-specific path of the shared cache database.

    The database lives in the same user configuration directory the MCP server
    already uses for its global ``.env`` file, so a Toolkit run and a running MCP
    server warm the same cache instead of each maintaining their own copy:

    * **Windows** — ``%APPDATA%\\financetoolkit\\financetoolkit_cache.db``
    * **macOS / Linux** — ``$XDG_CONFIG_HOME/financetoolkit/financetoolkit_cache.db``
      (falling back to ``~/.config/financetoolkit/`` when the variable is unset)

    The location can be overridden with the ``FINANCE_TOOLKIT_CACHE_DB``
    environment variable.

    Returns:
        Path: Absolute path to the cache database file.
    """
    override = os.environ.get("FINANCE_TOOLKIT_CACHE_DB")

    if override:
        return Path(override).expanduser()

    if platform.system() == "Windows":
        base = Path(
            os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        )
    else:
        xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
        base = Path(xdg_config_home) if xdg_config_home else Path.home() / ".config"

    return base / "financetoolkit" / DATABASE_FILE_NAME


def resolve_cache_location(location: str | Path | None) -> Path:
    """
    Turn a user supplied cache location into a database file path.

    Accepts None for the shared default, a directory (the database is placed
    inside it, which is what ``use_cached_data="my_folder"`` means), or a direct
    path to a ``.db`` file.

    Args:
        location (str | Path | None): The requested location.

    Returns:
        Path: Absolute path to the cache database file.
    """
    if location is None:
        return get_default_cache_location()

    path = Path(location).expanduser()

    if path.suffix in (".db", ".sqlite", ".sqlite3"):
        return path

    return path / DATABASE_FILE_NAME


@dataclass
class CachePlan:
    """
    The result of asking the cache what still has to be fetched.

    Attributes:
        key (str): The dataset key the plan was computed for.
        cached (dict[str, pd.DataFrame | pd.Series]): Per entity, the data already
            held by the cache, restricted to the requested window.
        missing (dict[str, list[coverage_model.Interval]]): Per entity, the date
            ranges that still have to be requested from the source. An entity
            absent from this mapping is fully cached.
    """

    key: str
    cached: dict[str, pd.DataFrame | pd.Series] = field(default_factory=dict)
    missing: dict[str, list[coverage_model.Interval]] = field(default_factory=dict)

    @property
    def entities_to_fetch(self) -> list[str]:
        """
        The entities that still need at least one call to the external source.

        Returns:
            list[str]: Entity identifiers with outstanding gaps.
        """
        return [entity for entity, gaps in self.missing.items() if gaps]

    @property
    def fully_cached(self) -> bool:
        """
        Whether the request can be served entirely from the cache.

        Returns:
            bool: True when no entity has an outstanding gap.
        """
        return not self.entities_to_fetch

    def get_fetch_span(self, entity: str) -> coverage_model.Interval | None:
        """
        Return a single interval covering every gap for an entity.

        Most financial APIs accept one date range per request, so fetching the
        hull of the outstanding gaps in one call is cheaper than issuing a
        separate request per gap. Any already cached data inside that hull is
        simply overwritten by the fresher response when it is merged back in.

        Args:
            entity (str): The entity to compute the span for.

        Returns:
            coverage_model.Interval | None: The covering interval, or None when
                the entity is fully cached.
        """
        gaps = self.missing.get(entity, [])

        if not gaps:
            return None

        return (min(start for start, _ in gaps), max(end for _, end in gaps))


class Cache:
    """
    Incremental, per entity cache for data retrieved from external sources.

    Every dataset is identified by its source, its name and the parameters that
    change the shape of the response. Within a dataset, data is stored per entity
    (a ticker, a country, a series identifier) together with the date ranges that
    have actually been requested. Widening a date range, adding a ticker or
    changing an unrelated parameter therefore reuses everything already stored
    instead of triggering a full refetch.

    The cache only ever holds data obtained from an external source. Anything the
    toolkit computes itself is derived from that data on demand and is never
    written here.
    """

    def __init__(self, location: str | Path | None = None, enabled: bool = True):
        """
        Initialize the cache against a database location.

        Args:
            location (str | Path | None): Directory or database file to use. None
                selects the shared, platform-specific default location.
            enabled (bool): When False every read misses and every write is a
                no-op, so callers do not need to branch on whether caching is on.
        """
        self._enabled = enabled
        self._location = resolve_cache_location(location)
        self._backend = None

        if enabled:
            try:
                self._backend = SQLiteBackend(self._location)
            except Exception as error:  # pylint: disable=broad-except
                # A cache is an optimization, so a database that cannot be opened just disables it.
                self._enabled = False

                logger.warning(
                    "Could not open the cache at %s (%s). Continuing without caching. "
                    "Remove or relocate the file to restore it.",
                    self._location,
                    error,
                )

        self._warn_on_schema_mismatch()

    def _warn_on_schema_mismatch(self) -> None:
        """
        Report a cache written by a different schema version without touching it.

        Structural changes must never silently discard a cache: a user may have
        spent a long time and a large part of an API quota filling it. The
        mismatch is therefore only reported, and clearing stays an explicit act.
        """
        if self._backend is None:
            return

        stored_version = self._backend.get_schema_version()

        if stored_version is not None and stored_version != SCHEMA_VERSION:
            logger.warning(
                "The cache at %s was written by schema version %s while this version "
                "of the Finance Toolkit expects %s. Nothing has been removed. If you "
                "run into unexpected results, clear it with "
                "Toolkit.clear_cache(confirm=True).",
                self._location,
                stored_version,
                SCHEMA_VERSION,
            )

    @property
    def enabled(self) -> bool:
        """
        Whether this cache actually reads from and writes to storage.

        Returns:
            bool: True when caching is active.
        """
        return self._enabled

    @property
    def location(self) -> Path:
        """
        The database file this cache is backed by.

        Returns:
            Path: Absolute path to the cache database.
        """
        return self._location

    def create_key(
        self, source: str, dataset: str, parameters: dict[str, Any] | None = None
    ) -> str:
        """
        Build the key identifying a dataset variant.

        Args:
            source (str): The external data source, e.g. "fmp".
            dataset (str): The dataset within that source, e.g. "historical".
            parameters (dict[str, Any] | None): Parameters that change the returned
                data. Entities and date ranges must not be included here.

        Returns:
            str: The dataset key.
        """
        return serialization_model.create_cache_key(source, dataset, parameters or {})

    def plan(
        self,
        source: str,
        dataset: str,
        entities: list[str] | str,
        start: str | date | None = None,
        end: str | date | None = None,
        parameters: dict[str, Any] | None = None,
        date_axis: int = 0,
        reference_date: date | None = None,
    ) -> CachePlan:
        """
        Work out what is already cached and what still has to be fetched.

        Two things make a range worth requesting. It was never fetched at all, or
        it was fetched so long ago that the dataset's time-to-live has lapsed. The
        first always has to be requested; the second is narrowed to the dataset's
        revision window, so a daily rerun asks for the volatile tail rather than
        the whole history. A rerun inside the time-to-live makes no request at all.

        Args:
            source (str): The external data source.
            dataset (str): The dataset within that source.
            entities (list[str] | str): The entities being requested.
            start (str | date | None): Inclusive start of the requested range.
            end (str | date | None): Inclusive end of the requested range.
            parameters (dict[str, Any] | None): Parameters that change the data.
            date_axis (int): 0 when the index holds the dates, 1 when the columns do.
            reference_date (date | None): The date the revision window is measured
                back from when the requested range has no end. Defaults to today;
                mainly useful for testing.

        Returns:
            CachePlan: The cached data per entity and the gaps still to be fetched.
        """
        entity_list = [entities] if isinstance(entities, str) else list(entities)
        key = self.create_key(source, dataset, parameters)
        plan = CachePlan(key=key)

        reference = reference_date or date.today()  # noqa: DTZ011
        requested_start = (
            coverage_model.normalize_date(start) if start else date(1900, 1, 1)
        )
        requested_end = coverage_model.normalize_date(end) if end else reference

        if not self._enabled or self._backend is None:
            plan.missing = {
                entity: [(requested_start, requested_end)] for entity in entity_list
            }

            return plan

        policy = policy_model.get_policy(source, dataset)
        minimum_fetched_at = time.time() - policy.ttl_seconds
        request = (requested_start, requested_end)

        # A revision window of 0 means the source takes no sub-range, so refetch it whole.
        revision_tail = (
            (
                max(requested_start, requested_end - timedelta(policy.revision_days)),
                requested_end,
            )
            if policy.revision_days > 0
            else request
        )

        for entity in entity_list:
            try:
                ever_covered = self._backend.read_coverage(key, entity)
                fresh_covered = self._backend.read_coverage(
                    key, entity, minimum_fetched_at
                )
                payload = self._backend.read_series(key, entity)
            except Exception as error:  # pylint: disable=broad-except
                # A read that fails mid-session is a cache miss, not a failed call.
                logger.debug(
                    "Could not read cached %s.%s for %s: %s",
                    source,
                    dataset,
                    entity,
                    error,
                )
                plan.missing[entity] = [request]

                continue

            gaps = coverage_model.subtract_intervals(request, ever_covered)

            for stale_gap in coverage_model.subtract_intervals(request, fresh_covered):
                overlap = coverage_model.intersect_intervals(stale_gap, revision_tail)

                if overlap is not None:
                    gaps.append(overlap)

            gaps = coverage_model.merge_intervals(gaps)

            if gaps:
                plan.missing[entity] = gaps

            if payload is not None:
                try:
                    stored = serialization_model.decode_dataframe(payload)
                    plan.cached[entity] = frame_model.slice_frame(
                        stored, start, end, date_axis
                    )
                except Exception as error:  # pylint: disable=broad-except
                    # An unreadable payload is a miss: ask the source for the lot.
                    logger.debug(
                        "Could not decode cached %s.%s for %s: %s",
                        source,
                        dataset,
                        entity,
                        error,
                    )
                    plan.missing[entity] = [request]

        return plan

    def store(
        self,
        source: str,
        dataset: str,
        entity: str,
        data: pd.DataFrame | pd.Series,
        start: str | date | None = None,
        end: str | date | None = None,
        parameters: dict[str, Any] | None = None,
        date_axis: int = 0,
    ) -> None:
        """
        Merge a freshly fetched frame into the cache and record its coverage.

        The requested range rather than the returned range is recorded, so that a
        window in which the source genuinely has no observations (before a listing,
        after a delisting, or a reporting gap) is remembered as answered and is not
        requested over and over again.

        Args:
            source (str): The external data source.
            dataset (str): The dataset within that source.
            entity (str): The entity the data belongs to.
            data (pd.DataFrame | pd.Series): The freshly fetched data.
            start (str | date | None): Inclusive start of the range that was requested.
            end (str | date | None): Inclusive end of the range that was requested.
            parameters (dict[str, Any] | None): Parameters that change the data.
            date_axis (int): 0 when the index holds the dates, 1 when the columns do.
        """
        if not self._enabled or self._backend is None:
            return

        key = self.create_key(source, dataset, parameters)

        try:
            existing_payload = self._backend.read_series(key, entity)
            existing = (
                serialization_model.decode_dataframe(existing_payload)
                if existing_payload is not None
                else None
            )

            merged = frame_model.merge_frames(existing, data, date_axis)

            self._backend.write_series(
                key,
                entity,
                serialization_model.encode_dataframe(merged),
                source=source,
                dataset=dataset,
            )
        except Exception as error:  # pylint: disable=broad-except
            # A cache write must never break the call that produced the data.
            logger.debug(
                "Could not cache %s.%s for %s: %s", source, dataset, entity, error
            )

            return

        bounds = frame_model.get_date_bounds(data, date_axis)

        coverage_start = (
            coverage_model.normalize_date(start)
            if start
            else (bounds[0] if bounds else None)
        )
        coverage_end = (
            coverage_model.normalize_date(end)
            if end
            else (bounds[1] if bounds else None)
        )

        if coverage_start is None or coverage_end is None:
            return

        # Sources over-fetch (prices by a year either side); claim what is genuinely held.
        if bounds is not None:
            coverage_start = min(coverage_start, bounds[0])
            coverage_end = max(coverage_end, bounds[1])

        try:
            self._backend.write_coverage(
                key,
                entity,
                coverage_start,
                coverage_end,
                source=source,
                dataset=dataset,
            )
            self._compact_coverage(key, entity, source, dataset)
        except Exception as error:  # pylint: disable=broad-except
            logger.debug(
                "Could not record coverage for %s.%s of %s: %s",
                source,
                dataset,
                entity,
                error,
            )

    def _compact_coverage(
        self, key: str, entity: str, source: str, dataset: str
    ) -> None:
        """
        Collapse an entity's coverage rows once they have accumulated.

        Args:
            key (str): The dataset key.
            entity (str): The entity within the dataset.
            source (str): The external data source, carried onto the rewritten rows.
            dataset (str): The dataset within that source, carried onto the rewritten rows.
        """
        if self._backend is None:
            return

        intervals = self._backend.read_coverage(key, entity)

        if len(intervals) < COVERAGE_COMPACTION_THRESHOLD:
            return

        merged = coverage_model.merge_intervals(intervals)

        # Merged rows inherit the oldest timestamp so old data never looks freshly fetched.
        oldest = self._backend.get_oldest_coverage_timestamp(key, entity)

        self._backend.replace_coverage(
            key,
            entity,
            merged,
            oldest or time.time(),
            source=source,
            dataset=dataset,
        )

    def get(
        self,
        source: str,
        dataset: str,
        entity: str,
        parameters: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> Any | None:
        """
        Read a cached value that has no date dimension.

        Used for point-in-time data such as company profiles, quotes and ratings,
        and for the small pieces of metadata a fetch produces as a side effect,
        where there is nothing to append to and only the time-to-live applies.

        Args:
            source (str): The external data source.
            dataset (str): The dataset within that source.
            entity (str): The entity to read.
            parameters (dict[str, Any] | None): Parameters that change the data.
            ttl (int | None): Overrides the dataset's registered time-to-live for
                this read. Used where the lifetime is configured by the caller
                rather than by the nature of the data, such as the MCP server's
                configurable response cache. Defaults to None (use the policy).

        Returns:
            Any | None: The cached value, or None when absent or expired.
        """
        if not self._enabled or self._backend is None:
            return None

        key = self.create_key(source, dataset, parameters)

        try:
            record = self._backend.read_blob(key, entity)

            if record is None:
                return None

            payload, updated_at = record
            effective_ttl = (
                ttl
                if ttl is not None
                else policy_model.get_policy(source, dataset).ttl_seconds
            )

            if time.time() - updated_at > effective_ttl:
                return None

            return serialization_model.decode_object(payload)
        except Exception as error:  # pylint: disable=broad-except
            # Any read failure is reported as a miss, so the caller falls back to the source.
            logger.debug(
                "Could not read cached %s.%s for %s: %s", source, dataset, entity, error
            )

            return None

    def set(
        self,
        source: str,
        dataset: str,
        entity: str,
        data: Any,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """
        Write a cached value that has no date dimension.

        Args:
            source (str): The external data source.
            dataset (str): The dataset within that source.
            entity (str): The entity the data belongs to.
            data (Any): The value to store. Any picklable object is accepted.
            parameters (dict[str, Any] | None): Parameters that change the data.
        """
        if not self._enabled or self._backend is None:
            return

        key = self.create_key(source, dataset, parameters)

        try:
            self._backend.write_blob(
                key,
                entity,
                serialization_model.encode_object(data),
                source=source,
                dataset=dataset,
            )
        except Exception as error:  # pylint: disable=broad-except
            logger.debug(
                "Could not cache %s.%s for %s: %s", source, dataset, entity, error
            )

    def invalidate(
        self,
        source: str,
        dataset: str,
        entity: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> int:
        """
        Drop cached data for a dataset, optionally for a single entity.

        Args:
            source (str): The external data source.
            dataset (str): The dataset within that source.
            entity (str | None): The entity to drop, or None for the whole dataset.
            parameters (dict[str, Any] | None): Parameters that change the data.

        Returns:
            int: The number of stored payloads removed.
        """
        if not self._enabled or self._backend is None:
            return 0

        return self._backend.delete_entity(
            self.create_key(source, dataset, parameters), entity
        )

    def remove(
        self,
        source: str | None = None,
        dataset: str | None = None,
        entity: str | None = None,
    ) -> int:
        """
        Remove cached data by source, dataset and entity.

        Every argument left as None widens the scope, so calling this with no
        arguments at all removes everything. Nothing here is ever triggered
        automatically; removal is always something the caller asked for.

        Args:
            source (str | None): Only remove entries from this source, e.g. "oecd".
            dataset (str | None): Only remove entries from this dataset, e.g. "historical".
            entity (str | None): Only remove entries for this entity, e.g. "AAPL".

        Returns:
            int: The number of stored payloads removed.
        """
        if not self._enabled or self._backend is None:
            return 0

        return self._backend.delete_scope(source=source, dataset=dataset, entity=entity)

    def clear(self) -> int:
        """
        Remove every entry from the cache.

        Returns:
            int: The number of rows removed across all tables.
        """
        if not self._enabled or self._backend is None:
            return 0

        return self._backend.clear_all()

    def get_contents(self) -> list[dict]:
        """
        List what the cache holds, grouped by source and dataset.

        Returns:
            list[dict]: One entry per source and dataset combination.
        """
        if not self._enabled or self._backend is None:
            return []

        try:
            return self._backend.get_contents()
        except Exception as error:  # pylint: disable=broad-except
            logger.warning("Could not read the cache contents: %s", error)

            return []

    def remove_expired_entries(
        self, ttl: int, source: str | None = None, dataset: str | None = None
    ) -> int:
        """
        Delete entries older than a given age, regardless of their dataset policy.

        Args:
            ttl (int): Maximum age in seconds.
            source (str | None): Only evict entries from this source. None evicts
                across every source, which is rarely what a caller that owns just
                one part of a shared cache wants.
            dataset (str | None): Only evict entries from this dataset.

        Returns:
            int: The number of rows removed.
        """
        if not self._enabled or self._backend is None:
            return 0

        try:
            return self._backend.remove_expired_entries(
                ttl, source=source, dataset=dataset
            )
        except Exception as error:  # pylint: disable=broad-except
            # Eviction is housekeeping on a timer, so a failure here is not surfaced.
            logger.debug("Could not evict expired cache entries: %s", error)

            return 0

    def get_statistics(self) -> dict[str, Any]:
        """
        Summarize the contents of the cache.

        Returns:
            dict[str, Any]: Row counts, database size in bytes and the location of
                the database file.
        """
        if not self._enabled or self._backend is None:
            return {"enabled": False, "location": str(self._location)}

        statistics = self._backend.get_statistics()
        statistics["enabled"] = True
        statistics["location"] = str(self._location)

        return statistics


def get_cache(location: str | Path | None = None, enabled: bool = True) -> Cache:
    """
    Return the shared cache instance for a location.

    Instances are reused per resolved database path so that every part of the
    toolkit, and a co-located MCP server, talk to the same cache rather than each
    opening their own.

    Args:
        location (str | Path | None): Directory or database file to use. None
            selects the shared default location.
        enabled (bool): When False a disabled cache is returned, which misses on
            every read and ignores every write.

    Returns:
        Cache: The cache instance for this location.
    """
    if not enabled:
        return Cache(location=location, enabled=False)

    resolved = str(resolve_cache_location(location))

    with _REGISTRY_LOCK:
        if resolved not in _CACHE_REGISTRY:
            _CACHE_REGISTRY[resolved] = Cache(location=location, enabled=True)

        return _CACHE_REGISTRY[resolved]


def reset_cache_registry() -> None:
    """
    Forget every cached instance, forcing the next call to reopen the database.

    Mainly useful in tests, where each test points the cache at its own temporary
    database file.
    """
    with _REGISTRY_LOCK:
        _CACHE_REGISTRY.clear()


def parse_use_cached_data(
    use_cached_data: bool | str,
) -> tuple[bool, str | None]:
    """
    Interpret the Toolkit's ``use_cached_data`` argument.

    Preserves the existing convention where ``True`` means "cache in the default
    location" and a string means "cache in this directory", while routing both to
    the new database backed cache.

    Args:
        use_cached_data (bool | str): The value passed to the Toolkit.

    Returns:
        tuple[bool, str | None]: Whether caching is enabled, and the location to
            use (None for the shared default).
    """
    if isinstance(use_cached_data, str):
        return True, use_cached_data

    return bool(use_cached_data), None


def format_timestamp(timestamp: float) -> str:
    """
    Render a Unix timestamp as a readable local time.

    Args:
        timestamp (float): The Unix timestamp to format.

    Returns:
        str: The timestamp formatted as "YYYY-MM-DD HH:MM:SS".
    """
    return datetime.fromtimestamp(timestamp).strftime(  # noqa: DTZ006
        "%Y-%m-%d %H:%M:%S"
    )
