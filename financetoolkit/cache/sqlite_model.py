"""SQLite Module"""

__docformat__ = "google"

import contextlib
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from threading import Lock

from financetoolkit.cache.coverage_model import (
    Interval,
    merge_intervals,
    normalize_date,
)

SCHEMA_VERSION = 1

# How long a writer waits for another process to release the write lock.
BUSY_TIMEOUT_SECONDS = 30

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
    def _connect(self, write: bool = False):
        """
        Yield a SQLite connection with write-ahead logging enabled.

        Connections are opened per operation and closed afterwards. SQLite's own
        context manager only commits, it does not close, so the connection is
        wrapped explicitly to avoid leaking file handles in long running servers.

        Writers open an explicit ``BEGIN IMMEDIATE`` transaction. The database is
        shared between the library and a long running MCP server, so a Python
        lock cannot serialize anything: the write lock has to be taken in SQLite
        itself, and it has to be taken *up front*. A deferred transaction that
        only escalates to a write lock at its first ``INSERT`` cannot be retried
        under write-ahead logging once another process has committed in the
        meantime, so it fails with "database is locked" rather than waiting.

        Args:
            write (bool): True to take the database write lock for the duration of
                the block, so that every statement inside it commits or rolls back
                as one unit. False opens a plain read connection.

        Yields:
            sqlite3.Connection: An open connection guarded by the instance lock.
        """
        with self._lock:
            connection = sqlite3.connect(
                self._database_location,
                timeout=BUSY_TIMEOUT_SECONDS,
                isolation_level=None,
            )
            try:
                connection.execute(
                    f"PRAGMA busy_timeout={int(BUSY_TIMEOUT_SECONDS * 1000)}"
                )
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=NORMAL")

                if not write:
                    yield connection
                else:
                    connection.execute("BEGIN IMMEDIATE")

                    try:
                        yield connection
                    except BaseException:
                        connection.execute("ROLLBACK")

                        raise

                    connection.execute("COMMIT")
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

        with self._connect(write=True) as connection:
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

    def read_entity_state(
        self, key: str, entity: str, minimum_fetched_at: float = 0.0
    ) -> tuple[list[Interval], list[Interval], bytes | None]:
        """
        Read everything the planner needs about an entity from one snapshot.

        Reading the coverage and the payload over three separate connections lets
        a concurrent writer commit in between, so the planner could see coverage
        that does not describe the payload it read. A single connection gives one
        consistent view of both.

        Args:
            key (str): The dataset key.
            entity (str): The entity within the dataset.
            minimum_fetched_at (float): Coverage fetched before this Unix timestamp
                counts as stale and is excluded from the second list.

        Returns:
            tuple[list[Interval], list[Interval], bytes | None]: Every recorded
                interval, the intervals still within their time-to-live, and the
                stored payload (None when nothing is cached).
        """
        with self._connect() as connection:
            ever_rows = connection.execute(
                "SELECT start_date, end_date, fetched_at FROM coverage "
                "WHERE key = ? AND entity = ?",
                (key, entity),
            ).fetchall()
            payload_row = connection.execute(
                "SELECT payload FROM series WHERE key = ? AND entity = ?",
                (key, entity),
            ).fetchone()

        ever_covered = [
            (normalize_date(start), normalize_date(end)) for start, end, _ in ever_rows
        ]
        fresh_covered = [
            (normalize_date(start), normalize_date(end))
            for start, end, fetched_at in ever_rows
            if fetched_at >= minimum_fetched_at
        ]

        return ever_covered, fresh_covered, payload_row[0] if payload_row else None

    def store_series_and_coverage(
        self,
        key: str,
        entity: str,
        merge_payload: Callable[[bytes | None], bytes],
        coverage: Interval | None = None,
        source: str = "",
        dataset: str = "",
        compaction_threshold: int = 0,
    ) -> None:
        """
        Write a payload and the range it covers as one indivisible unit.

        Reading the stored payload, merging the new rows into it and recording the
        coverage are three steps that only mean anything together. Spread over
        separate connections, two processes storing different ranges of the same
        entity both read the same starting payload, and the second write silently
        discards the first one's rows while both coverage rows survive: the cache
        then reports a range as fully held that it never stored. Taking the write
        lock up front and committing once makes the two writers queue instead, so
        the second merges on top of what the first actually wrote.

        Args:
            key (str): The dataset key.
            entity (str): The entity within the dataset.
            merge_payload (Callable[[bytes | None], bytes]): Receives the payload
                currently stored (None when absent) and returns the payload to
                store. Called with the write lock held, so it must not touch the
                database itself.
            coverage (Interval | None): The inclusive range the payload now covers.
                None records no coverage, which is what a dataset without a date
                dimension needs.
            source (str): The external data source, stored so the entry can be
                cleared by source later on.
            dataset (str): The dataset within that source, stored for the same reason.
            compaction_threshold (int): Collapse the entity's coverage rows into
                merged intervals once there are at least this many. Zero disables
                compaction.
        """
        with self._connect(write=True) as connection:
            existing = connection.execute(
                "SELECT payload FROM series WHERE key = ? AND entity = ?",
                (key, entity),
            ).fetchone()

            payload = merge_payload(existing[0] if existing else None)
            now = time.time()

            connection.execute(
                "INSERT OR REPLACE INTO series "
                "(key, entity, source, dataset, payload, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key, entity, source, dataset, payload, now),
            )

            if coverage is None:
                return

            self._upsert_coverage(
                connection, key, entity, coverage, source, dataset, now
            )

            if compaction_threshold > 0:
                self._compact_coverage(
                    connection, key, entity, source, dataset, compaction_threshold
                )

    @staticmethod
    def _upsert_coverage(
        connection: sqlite3.Connection,
        key: str,
        entity: str,
        interval: Interval,
        source: str,
        dataset: str,
        fetched_at: float,
    ) -> None:
        """
        Record a fetched range, refreshing it rather than duplicating it.

        Storing the identical range twice would say nothing the first row does not
        already say, so a retried or repeated write updates the existing row's
        timestamp instead of inserting beside it. Without that, every rerun inside
        the revision window would add another row to compact away.

        Args:
            connection (sqlite3.Connection): An open connection inside a write
                transaction.
            key (str): The dataset key.
            entity (str): The entity within the dataset.
            interval (Interval): The inclusive range that was fetched.
            source (str): The external data source, carried onto the row.
            dataset (str): The dataset within that source, carried onto the row.
            fetched_at (float): The Unix timestamp to record.
        """
        start, end = interval
        parameters = (key, entity, start.isoformat(), end.isoformat())

        updated = connection.execute(
            "UPDATE coverage SET fetched_at = ?, source = ?, dataset = ? "
            "WHERE key = ? AND entity = ? AND start_date = ? AND end_date = ?",
            (fetched_at, source, dataset, *parameters),
        ).rowcount

        if updated == 0:
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
                    fetched_at,
                ),
            )

    @staticmethod
    def _compact_coverage(
        connection: sqlite3.Connection,
        key: str,
        entity: str,
        source: str,
        dataset: str,
        threshold: int,
    ) -> None:
        """
        Collapse an entity's coverage rows once they have accumulated.

        Runs inside the caller's write transaction so that the rows are never
        momentarily absent to another process. Merged rows inherit the oldest of
        the timestamps they replace, so compaction can never make a stale range
        look freshly fetched.

        Args:
            connection (sqlite3.Connection): An open connection inside a write
                transaction.
            key (str): The dataset key.
            entity (str): The entity within the dataset.
            source (str): The external data source, carried onto the rewritten rows.
            dataset (str): The dataset within that source, carried onto the rewritten rows.
            threshold (int): The number of rows at which compaction kicks in.
        """
        rows = connection.execute(
            "SELECT start_date, end_date, fetched_at FROM coverage "
            "WHERE key = ? AND entity = ?",
            (key, entity),
        ).fetchall()

        if len(rows) < threshold:
            return

        merged = merge_intervals(
            [(normalize_date(start), normalize_date(end)) for start, end, _ in rows]
        )
        oldest = min(fetched_at for _, _, fetched_at in rows)

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
                    oldest,
                )
                for start, end in merged
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
        with self._connect(write=True) as connection:
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

        with self._connect(write=True) as connection:
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

        with self._connect(write=True) as connection:
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
        with self._connect(write=True) as connection:
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
