"""Game state persistence using SQLite.

This module provides persistent storage for game states, allowing
the agent to save and restore state across sessions.

Features:
- SQLite backend for reliable storage
- State snapshots with timestamps
- Query historical states
- Graceful handling of corrupted data
- Automatic database initialization

Example:
    >>> from src.memory.persistence import SQLiteStatePersistence
    >>>
    >>> # Create persistence (uses default path if not specified)
    >>> persistence = SQLiteStatePersistence("data/game_state.db")
    >>>
    >>> # Save current state
    >>> state = {"resources": {"gold": 1000}, "level": 5}
    >>> persistence.save_state(state)
    >>>
    >>> # Load most recent state
    >>> loaded = persistence.load_state()
    >>> print(loaded["resources"]["gold"])  # 1000
    >>>
    >>> # Query history
    >>> history = persistence.get_state_history(limit=10)
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = Path("data/game_state.db")

# Schema version for migrations
SCHEMA_VERSION = 1


class PersistenceError(Exception):
    """Error raised when persistence operations fail."""

    pass


class CorruptedDataError(PersistenceError):
    """Error raised when data corruption is detected."""

    pass


class SQLiteStatePersistence:
    """SQLite-based game state persistence.

    This class provides persistent storage for game states using SQLite.
    It supports:
    - Saving game state snapshots with timestamps
    - Loading the most recent state
    - Querying historical states
    - Checksum verification for data integrity
    - Graceful handling of corrupted data

    The class is thread-safe and uses connection pooling.

    Attributes:
        db_path: Path to the SQLite database file.

    Example:
        >>> persistence = SQLiteStatePersistence("game.db")
        >>> persistence.save_state({"gold": 100})
        >>> state = persistence.load_state()
        >>> print(state)  # {"gold": 100}
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        auto_init: bool = True,
    ) -> None:
        """Initialize the persistence layer.

        Args:
            db_path: Path to the SQLite database file.
                    Uses DEFAULT_DB_PATH if not specified.
            auto_init: Whether to automatically initialize the database.
        """
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._lock = threading.Lock()
        self._local = threading.local()

        if auto_init:
            self._ensure_db_exists()
            self._init_schema()

        logger.debug(f"SQLiteStatePersistence initialized: {self._db_path}")

    @property
    def db_path(self) -> Path:
        """Get the database path."""
        return self._db_path

    def _ensure_db_exists(self) -> None:
        """Ensure the database directory exists."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection.

        Returns:
            SQLite connection for the current thread.
        """
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self._db_path),
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            )
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            # Use WAL mode for better concurrency
            self._local.connection.execute("PRAGMA journal_mode = WAL")

        conn: sqlite3.Connection = self._local.connection
        return conn

    def _init_schema(self) -> None:
        """Initialize the database schema."""
        conn = self._get_connection()

        with conn:
            # Create schema version table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)

            # Check current version
            cursor = conn.execute("SELECT version FROM schema_version LIMIT 1")
            row = cursor.fetchone()
            current_version = row[0] if row else 0

            if current_version < SCHEMA_VERSION:
                self._migrate_schema(conn, current_version)

    def _migrate_schema(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Migrate database schema to current version.

        Args:
            conn: Database connection.
            from_version: Current schema version in database.
        """
        if from_version < 1:
            # Initial schema
            conn.execute("""
                CREATE TABLE IF NOT EXISTS game_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    metadata_json TEXT
                )
            """)

            # Index for timestamp queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_game_states_timestamp
                ON game_states(timestamp DESC)
            """)

            # Update schema version
            conn.execute("DELETE FROM schema_version")
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )

            logger.info(f"Database schema migrated to version {SCHEMA_VERSION}")

    @staticmethod
    def _compute_checksum(data: str) -> str:
        """Compute checksum for data integrity verification.

        Args:
            data: String data to checksum.

        Returns:
            SHA-256 checksum as hex string.
        """
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    @staticmethod
    def _serialize_state(state: dict[str, Any]) -> str:
        """Serialize state to JSON string.

        Args:
            state: State dictionary to serialize.

        Returns:
            JSON string representation.

        Raises:
            PersistenceError: If state cannot be serialized.
        """
        try:
            return json.dumps(state, sort_keys=True, default=str)
        except (TypeError, ValueError) as e:
            raise PersistenceError(f"Failed to serialize state: {e}") from e

    @staticmethod
    def _deserialize_state(json_str: str) -> dict[str, Any]:
        """Deserialize state from JSON string.

        Args:
            json_str: JSON string to deserialize.

        Returns:
            State dictionary.

        Raises:
            CorruptedDataError: If JSON is invalid.
        """
        try:
            result: dict[str, Any] = json.loads(json_str)
            return result
        except json.JSONDecodeError as e:
            raise CorruptedDataError(f"Invalid JSON in state: {e}") from e

    def save_state(
        self,
        state: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Save the current game state.

        Args:
            state: Game state dictionary to persist.
            metadata: Optional metadata about the state.

        Returns:
            ID of the saved state record.

        Raises:
            PersistenceError: If save fails.
        """
        timestamp = datetime.now().isoformat()
        state_json = self._serialize_state(state)
        checksum = self._compute_checksum(state_json)
        metadata_json = json.dumps(metadata) if metadata else None

        with self._lock:
            try:
                conn = self._get_connection()
                with conn:
                    cursor = conn.execute(
                        """
                        INSERT INTO game_states (timestamp, state_json, checksum, metadata_json)
                        VALUES (?, ?, ?, ?)
                        """,
                        (timestamp, state_json, checksum, metadata_json),
                    )
                    state_id = cursor.lastrowid or 0

                logger.debug(f"Saved state #{state_id} at {timestamp}")
                return state_id

            except sqlite3.Error as e:
                raise PersistenceError(f"Failed to save state: {e}") from e

    def load_state(self, *, verify_checksum: bool = True) -> dict[str, Any] | None:
        """Load the most recent game state.

        Args:
            verify_checksum: Whether to verify data integrity.

        Returns:
            Most recent game state, or None if no state saved.

        Raises:
            CorruptedDataError: If checksum verification fails.
            PersistenceError: If load fails.
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute(
                    """
                    SELECT state_json, checksum FROM game_states
                    ORDER BY timestamp DESC LIMIT 1
                    """
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                state_json, stored_checksum = row

                if verify_checksum:
                    computed_checksum = self._compute_checksum(state_json)
                    if computed_checksum != stored_checksum:
                        raise CorruptedDataError(
                            "State checksum mismatch - data may be corrupted"
                        )

                return self._deserialize_state(state_json)

            except sqlite3.Error as e:
                raise PersistenceError(f"Failed to load state: {e}") from e

    def load_state_by_id(
        self,
        state_id: int,
        *,
        verify_checksum: bool = True,
    ) -> dict[str, Any] | None:
        """Load a specific game state by ID.

        Args:
            state_id: ID of the state to load.
            verify_checksum: Whether to verify data integrity.

        Returns:
            Game state, or None if not found.

        Raises:
            CorruptedDataError: If checksum verification fails.
            PersistenceError: If load fails.
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute(
                    """
                    SELECT state_json, checksum FROM game_states
                    WHERE id = ?
                    """,
                    (state_id,),
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                state_json, stored_checksum = row

                if verify_checksum:
                    computed_checksum = self._compute_checksum(state_json)
                    if computed_checksum != stored_checksum:
                        raise CorruptedDataError(
                            f"State #{state_id} checksum mismatch - data may be corrupted"
                        )

                return self._deserialize_state(state_json)

            except sqlite3.Error as e:
                raise PersistenceError(f"Failed to load state #{state_id}: {e}") from e

    def get_state_history(
        self,
        limit: int = 10,
        since: datetime | None = None,
        *,
        verify_checksums: bool = False,
    ) -> list[dict[str, Any]]:
        """Get historical game states.

        Args:
            limit: Maximum number of states to return.
            since: Only return states after this time.
            verify_checksums: Whether to verify data integrity.

        Returns:
            List of game states, most recent first.

        Raises:
            PersistenceError: If query fails.
        """
        with self._lock:
            try:
                conn = self._get_connection()

                if since:
                    cursor = conn.execute(
                        """
                        SELECT state_json, checksum FROM game_states
                        WHERE timestamp > ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (since.isoformat(), limit),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT state_json, checksum FROM game_states
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (limit,),
                    )

                states: list[dict[str, Any]] = []
                for state_json, stored_checksum in cursor.fetchall():
                    if verify_checksums:
                        computed = self._compute_checksum(state_json)
                        if computed != stored_checksum:
                            logger.warning("Skipping corrupted state in history")
                            continue

                    with contextlib.suppress(CorruptedDataError):
                        states.append(self._deserialize_state(state_json))

                return states

            except sqlite3.Error as e:
                raise PersistenceError(f"Failed to get state history: {e}") from e

    def get_state_count(self) -> int:
        """Get the total number of saved states.

        Returns:
            Number of states in the database.
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute("SELECT COUNT(*) FROM game_states")
                result = cursor.fetchone()
                return result[0] if result else 0
            except sqlite3.Error as e:
                raise PersistenceError(f"Failed to count states: {e}") from e

    def delete_state(self, state_id: int) -> bool:
        """Delete a specific state by ID.

        Args:
            state_id: ID of the state to delete.

        Returns:
            True if state was deleted, False if not found.
        """
        with self._lock:
            try:
                conn = self._get_connection()
                with conn:
                    cursor = conn.execute(
                        "DELETE FROM game_states WHERE id = ?",
                        (state_id,),
                    )
                    return cursor.rowcount > 0
            except sqlite3.Error as e:
                raise PersistenceError(f"Failed to delete state: {e}") from e

    def prune_old_states(self, keep_count: int = 1000) -> int:
        """Prune old states to manage storage.

        Keeps the most recent `keep_count` states and deletes the rest.

        Args:
            keep_count: Number of recent states to keep.

        Returns:
            Number of states deleted.
        """
        with self._lock:
            try:
                conn = self._get_connection()
                with conn:
                    # Get ID threshold
                    cursor = conn.execute(
                        """
                        SELECT id FROM game_states
                        ORDER BY timestamp DESC
                        LIMIT 1 OFFSET ?
                        """,
                        (keep_count,),
                    )
                    row = cursor.fetchone()

                    if row is None:
                        return 0  # Nothing to prune

                    threshold_id = row[0]

                    # Delete older states
                    cursor = conn.execute(
                        "DELETE FROM game_states WHERE id <= ?",
                        (threshold_id,),
                    )
                    deleted = cursor.rowcount

                    if deleted > 0:
                        logger.info(f"Pruned {deleted} old states")

                    return deleted

            except sqlite3.Error as e:
                raise PersistenceError(f"Failed to prune states: {e}") from e

    def clear_all(self) -> int:
        """Clear all saved states.

        Returns:
            Number of states deleted.

        Warning:
            This is a destructive operation!
        """
        with self._lock:
            try:
                conn = self._get_connection()
                with conn:
                    cursor = conn.execute("DELETE FROM game_states")
                    deleted = cursor.rowcount
                    logger.warning(f"Cleared all {deleted} states")
                    return deleted
            except sqlite3.Error as e:
                raise PersistenceError(f"Failed to clear states: {e}") from e

    def verify_integrity(self) -> tuple[int, int]:
        """Verify integrity of all stored states.

        Returns:
            Tuple of (valid_count, corrupted_count).
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute(
                    "SELECT id, state_json, checksum FROM game_states"
                )

                valid = 0
                corrupted = 0

                for state_id, state_json, stored_checksum in cursor.fetchall():
                    computed = self._compute_checksum(state_json)
                    if computed == stored_checksum:
                        valid += 1
                    else:
                        corrupted += 1
                        logger.warning(f"Corrupted state detected: #{state_id}")

                return valid, corrupted

            except sqlite3.Error as e:
                raise PersistenceError(f"Failed to verify integrity: {e}") from e

    def repair_database(self) -> int:
        """Attempt to repair the database by removing corrupted entries.

        Returns:
            Number of corrupted entries removed.
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute(
                    "SELECT id, state_json, checksum FROM game_states"
                )

                corrupted_ids: list[int] = []

                for state_id, state_json, stored_checksum in cursor.fetchall():
                    computed = self._compute_checksum(state_json)
                    if computed != stored_checksum:
                        corrupted_ids.append(state_id)

                if corrupted_ids:
                    with conn:
                        conn.executemany(
                            "DELETE FROM game_states WHERE id = ?",
                            [(id_,) for id_ in corrupted_ids],
                        )
                    logger.warning(f"Removed {len(corrupted_ids)} corrupted states")

                return len(corrupted_ids)

            except sqlite3.Error as e:
                raise PersistenceError(f"Failed to repair database: {e}") from e

    def export_to_json(self, output_path: str | Path) -> int:
        """Export all states to a JSON file.

        Args:
            output_path: Path to output file.

        Returns:
            Number of states exported.
        """
        states = self.get_state_history(limit=100000, verify_checksums=True)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(states, f, indent=2, default=str)

        logger.info(f"Exported {len(states)} states to {output_path}")
        return len(states)

    def import_from_json(self, input_path: str | Path) -> int:
        """Import states from a JSON file.

        Args:
            input_path: Path to input file.

        Returns:
            Number of states imported.
        """
        input_path = Path(input_path)

        with open(input_path) as f:
            states = json.load(f)

        if not isinstance(states, list):
            raise PersistenceError("Invalid import file format - expected list")

        for state in states:
            if isinstance(state, dict):
                self.save_state(state)

        logger.info(f"Imported {len(states)} states from {input_path}")
        return len(states)

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
            logger.debug("Database connection closed")

    def __enter__(self) -> SQLiteStatePersistence:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Destructor - close connection."""
        with contextlib.suppress(Exception):
            self.close()
