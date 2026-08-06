"""SQLite Module"""

__docformat__ = "google"

import contextlib
import sqlite3
import time
from datetime import date
from pathlib import Path
from threading import Lock

from financetoolkit.cache.coverage_model import Interval, normalize_date

SCHEMA_VERSION = 1

# Three tables; source and dataset stay plain columns so clearing can be scoped.
SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS series (
        key        TEXT NOT NULL,
        entity     TEXT NOT NULL,
        source     TEXT NOT NULL DEFAULT '',
        dataset    TEXT NOT NULL DEFAULT '',
        payload    BLOB NOT NULL,
        updated_at REAL NOT NULL,
        PRIMARY KEY (key, entity)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS coverage (
        key        TEXT NOT NULL,
        entity     TEXT NOT NULL,
        source     TEXT NOT NULL DEFAULT '',
        dataset    TEXT NOT NULL DEFAULT '',
        start_date TEXT NOT NULL,
        end_date   TEXT NOT NULL,
        fetched_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS blobs (
        key        TEXT NOT NULL,
        entity     TEXT NOT NULL,
        source     TEXT NOT NULL DEFAULT '',
        dataset    TEXT NOT NULL DEFAULT '',
        payload    BLOB NOT NULL,
        updated_at REAL NOT NULL,
        PRIMARY KEY (key, entity)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS meta (
        name  TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_coverage_key ON coverage(key, entity)",
    "CREATE INDEX IF NOT EXISTS idx_coverage_fetched ON coverage(fetched_at)",
    "CREATE INDEX IF NOT EXISTS idx_series_updated ON series(updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_blobs_updated ON blobs(updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_series_scope ON series(source, dataset, entity)",
    "CREATE INDEX IF NOT EXISTS idx_blobs_scope ON blobs(source, dataset, entity)",
    "CREATE INDEX IF NOT EXISTS idx_coverage_scope ON coverage(source, dataset, entity)",
)

PAYLOAD_TABLES = ("series", "blobs")
ALL_TABLES = ("series", "blobs", "coverage")


class SQLiteBackend:
    """
    Thread-safe SQLite storage for cached external data.

    SQLite is used rather than a directory of pickle files because the cache has
    to answer range questions ("which parts of 2015-2025 do I already have for
    this ticker?"), upsert single entities atomically, and be shared safely
    between a library process and a long running MCP server. Write-ahead logging
    is enabled so concurrent readers are never blocked by a writer.
    """

    def __init__(self, database_location: str | Path) -> None:
        """
        Initialize the backend and ensure the schema exists.

        Args:
            database_location (str | Path): Path to the SQLite database file. Parent
                directories are created automatically.
        """
        self._database_location = str(database_location)
        self._lock = Lock()

        self.initialize_database()

    @property
    def database_location(self) -> str:
        """
        The path of the underlying SQLite database file.

        Returns:
            str: The absolute or relative path the backend writes to.
        """
        return self._database_location

    @contextlib.contextmanager
    def _connect(self):
        """
        Yield a SQLite connection with write-ahead logging enabled.

        Connections are opened per operation and closed afterwards. SQLite's own
        context manager only commits, it does not close, so the connection is
        wrapped explicitly to avoid leaking file handles in long running servers.

        Yields:
            sqlite3.Connection: An open connection guarded by the instance lock.
        """
        with self._lock:
            connection = sqlite3.connect(self._database_location, timeout=30)
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=NORMAL")
                yield connection
                connection.commit()
            finally:
                connection.close()

    def initialize_database(self) -> None:
        """
        Create the tables and indices used by the cache if they do not exist yet.

        An existing database written by a different schema version is left exactly
        as it is. Dropping it automatically would silently discard hours of
        accumulated downloads, so the mismatch is surfaced to the caller instead
        and the decision to clear is left to the user.
        """
        Path(self._database_location).parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as connection:
            for statement in SCHEMA_STATEMENTS:
                connection.execute(statement)

            connection.execute(
                "INSERT OR IGNORE INTO meta (name, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )

    def get_schema_version(self) -> int | None:
        """
        Return the schema version recorded in the database.

        Returns:
            int | None: The stored version, or None when it is absent or unreadable.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM meta WHERE name = ?", ("schema_version",)
            ).fetchone()

        if row is None:
            return None

        try:
            return int(row[0])
        except (TypeError, ValueError):
            return None

    def read_series(self, key: str, entity: str) -> bytes | None:
        """
        Read the stored payload for a single entity of a dataset.

        Args:
            key (str): The dataset key.
            entity (str): The entity within the dataset, e.g. a ticker.

        Returns:
            bytes | None: The stored payload, or None when nothing is cached.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM series WHERE key = ? AND entity = ?",
                (key, entity),
            ).fetchone()

        return row[0] if row else None

    def write_series(
        self,
        key: str,
        entity: str,
        payload: bytes,
        source: str = "",
        dataset: str = "",
    ) -> None:
        """
        Insert or replace the payload for a single entity of a dataset.

        Args:
            key (str): The dataset key.
            entity (str): The entity within the dataset.
            payload (bytes): The serialized payload to store.
            source (str): The external data source, stored so the entry can be
                cleared by source later on.
            dataset (str): The dataset within that source, stored for the same reason.
        """
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO series "
                "(key, entity, source, dataset, payload, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key, entity, source, dataset, payload, time.time()),
            )

    def read_coverage(
        self, key: str, entity: str, minimum_fetched_at: float = 0.0
    ) -> list[Interval]:
        """
        Read the date ranges already fetched for an entity.

        Args:
            key (str): The dataset key.
            entity (str): The entity within the dataset.
            minimum_fetched_at (float): Only return ranges fetched at or after this
                Unix timestamp. Ranges fetched earlier are considered expired and
                are therefore omitted, which makes them show up as missing again.

        Returns:
            list[Interval]: The stored intervals, unmerged and in insertion order.
        """
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT start_date, end_date FROM coverage "
                "WHERE key = ? AND entity = ? AND fetched_at >= ?",
                (key, entity, minimum_fetched_at),
            ).fetchall()

        return [(normalize_date(start), normalize_date(end)) for start, end in rows]

    def get_oldest_coverage_timestamp(self, key: str, entity: str) -> float | None:
        """
        Return the oldest fetch timestamp recorded for an entity.

        Used when compacting coverage rows: merging several ranges into one has to
        adopt the oldest of their timestamps, otherwise compaction would silently
        reset the time-to-live and make stale ranges look freshly fetched.

        Args:
            key (str): The dataset key.
            entity (str): The entity within the dataset.

        Returns:
            float | None: The oldest Unix timestamp, or None when no coverage exists.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT MIN(fetched_at) FROM coverage WHERE key = ? AND entity = ?",
                (key, entity),
            ).fetchone()

        return row[0] if row and row[0] is not None else None

    def write_coverage(
        self,
        key: str,
        entity: str,
        start: date,
        end: date,
        source: str = "",
        dataset: str = "",
    ) -> None:
        """
        Record that a date range has been fetched for an entity.

        Args:
            key (str): The dataset key.
            entity (str): The entity within the dataset.
            start (date): The inclusive start of the fetched range.
            end (date): The inclusive end of the fetched range.
            source (str): The external data source, stored so the entry can be
                cleared by source later on.
            dataset (str): The dataset within that source, stored for the same reason.
        """
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO coverage "
                "(key, entity, source, dataset, start_date, end_date, fetched_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    entity,
                    source,
                    dataset,
                    start.isoformat(),
                    end.isoformat(),
                    time.time(),
                ),
            )

    def replace_coverage(
        self,
        key: str,
        entity: str,
        intervals: list[Interval],
        fetched_at: float,
        source: str = "",
        dataset: str = "",
    ) -> None:
        """
        Overwrite all coverage rows for an entity with a compacted interval list.

        Coverage rows accumulate one per fetch, so an entity refreshed daily would
        grow an unbounded number of overlapping rows. Periodically collapsing them
        into merged intervals keeps the table small without losing information.

        Args:
            key (str): The dataset key.
            entity (str): The entity within the dataset.
            intervals (list[Interval]): The merged intervals to store.
            fetched_at (float): The Unix timestamp to record for every interval.
                The oldest original timestamp is normally used so that compaction
                never makes stale data look fresh.
            source (str): The external data source, stored so the entry can be
                cleared by source later on.
            dataset (str): The dataset within that source, stored for the same reason.
        """
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM coverage WHERE key = ? AND entity = ?", (key, entity)
            )
            connection.executemany(
                "INSERT INTO coverage "
                "(key, entity, source, dataset, start_date, end_date, fetched_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        key,
                        entity,
                        source,
                        dataset,
                        start.isoformat(),
                        end.isoformat(),
                        fetched_at,
                    )
                    for start, end in intervals
                ],
            )

    def read_blob(self, key: str, entity: str) -> tuple[bytes, float] | None:
        """
        Read a cached dataset that has no date dimension.

        Args:
            key (str): The dataset key.
            entity (str): The entity within the dataset.

        Returns:
            tuple[bytes, float] | None: The payload and the Unix timestamp it was
                written at, or None when nothing is cached.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload, updated_at FROM blobs WHERE key = ? AND entity = ?",
                (key, entity),
            ).fetchone()

        return (row[0], row[1]) if row else None

    def write_blob(
        self,
        key: str,
        entity: str,
        payload: bytes,
        source: str = "",
        dataset: str = "",
    ) -> None:
        """
        Insert or replace a cached dataset that has no date dimension.

        Args:
            key (str): The dataset key.
            entity (str): The entity within the dataset.
            payload (bytes): The serialized payload to store.
            source (str): The external data source, stored so the entry can be
                cleared by source later on.
            dataset (str): The dataset within that source, stored for the same reason.
        """
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO blobs "
                "(key, entity, source, dataset, payload, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key, entity, source, dataset, payload, time.time()),
            )

    def delete_entity(self, key: str, entity: str | None = None) -> int:
        """
        Remove cached data for a dataset, optionally limited to one entity.

        Args:
            key (str): The dataset key.
            entity (str | None): The entity to remove. When None the whole dataset
                is removed.

        Returns:
            int: The number of payload rows deleted across the series and blob tables.
        """
        conditions = ["key = ?"]
        values: list[str] = [key]

        if entity is not None:
            conditions.append("entity = ?")
            values.append(entity)

        return self._delete_where(conditions, values)

    def delete_scope(
        self,
        source: str | None = None,
        dataset: str | None = None,
        entity: str | None = None,
    ) -> int:
        """
        Remove cached data by source, dataset and entity rather than by key.

        This is the deletion path meant for people rather than for code. The key is
        an opaque hash, so clearing "everything from OECD" or "everything for AAPL"
        is only expressible against the stored source, dataset and entity columns.

        Args:
            source (str | None): Only remove entries from this source. None matches
                every source.
            dataset (str | None): Only remove entries from this dataset. None matches
                every dataset.
            entity (str | None): Only remove entries for this entity. None matches
                every entity.

        Returns:
            int: The number of payload rows deleted across the series and blob tables.
        """
        conditions: list[str] = []
        values: list[str] = []

        for column, value in (
            ("source", source),
            ("dataset", dataset),
            ("entity", entity),
        ):
            if value is not None:
                conditions.append(f"{column} = ?")
                values.append(value)

        return self._delete_where(conditions, values)

    def _delete_where(self, conditions: list[str], values: list[str]) -> int:
        """
        Delete matching rows from every table, counting only the payload rows.

        Args:
            conditions (list[str]): SQL conditions to apply, combined with AND. An
                empty list matches every row.
            values (list[str]): The parameter values for those conditions.

        Returns:
            int: The number of payload rows deleted across the series and blob tables.
        """
        clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        deleted = 0

        with self._connect() as connection:
            for table in ALL_TABLES:
                rowcount = connection.execute(
                    f"DELETE FROM {table}{clause}",  # noqa: S608
                    values,
                ).rowcount

                if table in PAYLOAD_TABLES:
                    deleted += rowcount

        return deleted

    def remove_expired_entries(
        self, ttl: int, source: str | None = None, dataset: str | None = None
    ) -> int:
        """
        Delete entries older than the given time-to-live.

        The scope matters now that one database is shared between the library and a
        long running MCP server. Evicting unscoped with a short server-side TTL would
        also throw away the price history and filings the library accumulated, so
        callers that own only part of the cache pass their own source.

        Args:
            ttl (int): Maximum age in seconds. Rows written before ``now - ttl``
                are removed.
            source (str | None): Only evict entries from this source. None evicts
                across every source.
            dataset (str | None): Only evict entries from this dataset. None evicts
                across every dataset.

        Returns:
            int: The number of rows deleted across all tables.
        """
        cutoff = time.time() - ttl
        scope_conditions: list[str] = []
        scope_values: list[str] = []

        for column, value in (("source", source), ("dataset", dataset)):
            if value is not None:
                scope_conditions.append(f"{column} = ?")
                scope_values.append(value)

        clause = "".join(f" AND {condition}" for condition in scope_conditions)
        deleted = 0

        with self._connect() as connection:
            for table in ALL_TABLES:
                timestamp_column = "fetched_at" if table == "coverage" else "updated_at"
                deleted += connection.execute(
                    f"DELETE FROM {table} WHERE {timestamp_column} < ?{clause}",  # noqa: S608
                    [cutoff, *scope_values],
                ).rowcount

        return deleted

    def clear_all(self) -> int:
        """
        Remove every cached entry regardless of age.

        Returns:
            int: The number of rows deleted across all tables.
        """
        with self._connect() as connection:
            deleted = connection.execute("DELETE FROM series").rowcount
            deleted += connection.execute("DELETE FROM blobs").rowcount
            deleted += connection.execute("DELETE FROM coverage").rowcount

        return deleted

    def get_statistics(self) -> dict[str, int]:
        """
        Summarize what the cache currently holds.

        Returns:
            dict[str, int]: Row counts per table plus the size of the database file
                in bytes, useful for a user facing cache overview.
        """
        with self._connect() as connection:
            series_rows = connection.execute("SELECT COUNT(*) FROM series").fetchone()[
                0
            ]
            blob_rows = connection.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
            coverage_rows = connection.execute(
                "SELECT COUNT(*) FROM coverage"
            ).fetchone()[0]

        database_path = Path(self._database_location)
        size_in_bytes = database_path.stat().st_size if database_path.exists() else 0

        return {
            "series_entries": series_rows,
            "blob_entries": blob_rows,
            "coverage_entries": coverage_rows,
            "size_in_bytes": size_in_bytes,
        }

    def get_contents(self) -> list[dict]:
        """
        List what the cache holds, grouped by source and dataset.

        Intended to be shown to the user before they decide what to clear, so that
        removing part of the cache is an informed choice rather than a guess.

        Returns:
            list[dict]: One entry per source and dataset combination, holding the
                number of entities, the number of stored payloads, the oldest and
                newest write timestamps and the entities themselves.
        """
        rows: dict[tuple[str, str], dict] = {}

        with self._connect() as connection:
            for table in PAYLOAD_TABLES:
                for source, dataset, entity, updated_at in connection.execute(
                    f"SELECT source, dataset, entity, updated_at FROM {table}"  # noqa: S608
                ):
                    record = rows.setdefault(
                        (source, dataset),
                        {
                            "source": source,
                            "dataset": dataset,
                            "entities": set(),
                            "entries": 0,
                            "oldest_write": updated_at,
                            "newest_write": updated_at,
                        },
                    )

                    record["entities"].add(entity)
                    record["entries"] += 1
                    record["oldest_write"] = min(record["oldest_write"], updated_at)
                    record["newest_write"] = max(record["newest_write"], updated_at)

        return [
            {**record, "entities": sorted(record["entities"])}
            for record in sorted(
                rows.values(), key=lambda r: (r["source"], r["dataset"])
            )
        ]
