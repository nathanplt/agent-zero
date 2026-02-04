"""Strategy memory for tracking named strategies and their effectiveness.

This module provides strategy memory functionality, allowing the agent to:
- Store named strategies with descriptions
- Track success/failure and effectiveness per strategy
- Recommend strategies for situations (game state)
- Identify patterns from episodes

Implements the strategy-related operations of the MemoryStore interface
(see src/interfaces/memory.py). Can be composed with StatePersistence
and EpisodicMemory into a full MemoryStore.

Example:
    >>> from src.memory.strategy import SQLiteStrategyMemory
    >>> from src.interfaces.memory import Strategy
    >>>
    >>> memory = SQLiteStrategyMemory("data/strategies.db")
    >>> memory.save_strategy(Strategy("upgrade_first", "Upgrade when affordable"))
    >>> memory.update_strategy_outcome("upgrade_first", success=True)
    >>> best = memory.get_best_strategies(limit=5)
    >>> recommended = memory.recommend_for_situation({"gold": 100, "level": 5})
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.interfaces.memory import Episode, Strategy

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = Path("data/strategies.db")

# Schema version for migrations
SCHEMA_VERSION = 1


class StrategyMemoryError(Exception):
    """Error raised when strategy memory operations fail."""

    pass


class SQLiteStrategyMemory:
    """SQLite-based strategy memory for named strategies and effectiveness.

    This class provides persistent storage for strategies using SQLite.
    It supports:
    - Saving and retrieving named strategies with descriptions
    - Updating success/failure counts and effectiveness
    - Getting best strategies by effectiveness
    - Recommending a strategy for a game state (highest-rated)
    - Identifying patterns from episodes (e.g. action type + outcome)

    Attributes:
        db_path: Path to the SQLite database file.

    Example:
        >>> memory = SQLiteStrategyMemory("strategies.db")
        >>> memory.save_strategy(Strategy("s1", "First strategy"))
        >>> memory.update_strategy_outcome("s1", success=True)
        >>> memory.get_best_strategies(limit=5)
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        auto_init: bool = True,
    ) -> None:
        """Initialize the strategy memory.

        Args:
            db_path: Path to the SQLite database file.
                    Uses DEFAULT_DB_PATH if not specified.
            auto_init: Whether to automatically initialize the database.
        """
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._lock = threading.RLock()
        self._local = threading.local()

        if auto_init:
            self._ensure_db_exists()
            self._init_schema()

        logger.debug(f"SQLiteStrategyMemory initialized: {self._db_path}")

    @property
    def db_path(self) -> Path:
        """Get the database path."""
        return self._db_path

    def _ensure_db_exists(self) -> None:
        """Ensure the database directory exists."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self._db_path),
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            )
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            self._local.connection.execute("PRAGMA journal_mode = WAL")

        conn: sqlite3.Connection = self._local.connection
        return conn

    def _init_schema(self) -> None:
        """Initialize the database schema."""
        conn = self._get_connection()

        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)

            cursor = conn.execute("SELECT version FROM schema_version LIMIT 1")
            row = cursor.fetchone()
            current_version = row[0] if row else 0

            if current_version < SCHEMA_VERSION:
                self._migrate_schema(conn, current_version)

    def _migrate_schema(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Migrate database schema to current version."""
        if from_version < 1:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    name TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    last_used TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_strategies_effectiveness
                ON strategies(success_count, failure_count)
            """)

            conn.execute("DELETE FROM schema_version")
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )

            logger.info(f"Strategy memory schema migrated to version {SCHEMA_VERSION}")

    @staticmethod
    def _row_to_strategy(row: tuple[Any, ...]) -> Strategy:
        """Convert a database row to a Strategy object."""
        name, description, success_count, failure_count, last_used_str = row
        last_used = datetime.fromisoformat(last_used_str) if last_used_str else None
        return Strategy(
            name=name,
            description=description,
            success_count=int(success_count),
            failure_count=int(failure_count),
            last_used=last_used,
        )

    def save_strategy(self, strategy: Strategy) -> None:
        """Save or update a strategy.

        Args:
            strategy: The strategy to save.

        Raises:
            StrategyMemoryError: If save fails.
        """
        with self._lock:
            try:
                conn = self._get_connection()
                last_used_str = (
                    strategy.last_used.isoformat() if strategy.last_used is not None else None
                )
                with conn:
                    conn.execute(
                        """
                        INSERT INTO strategies
                        (name, description, success_count, failure_count, last_used)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(name) DO UPDATE SET
                            description = excluded.description,
                            success_count = excluded.success_count,
                            failure_count = excluded.failure_count,
                            last_used = excluded.last_used
                        """,
                        (
                            strategy.name,
                            strategy.description,
                            strategy.success_count,
                            strategy.failure_count,
                            last_used_str,
                        ),
                    )
                logger.debug(f"Saved strategy: {strategy.name}")
            except sqlite3.Error as e:
                raise StrategyMemoryError(f"Failed to save strategy: {e}") from e

    def get_strategy(self, name: str) -> Strategy | None:
        """Get a strategy by name.

        Args:
            name: Name of the strategy.

        Returns:
            The strategy, or None if not found.
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute(
                    """
                    SELECT name, description, success_count, failure_count, last_used
                    FROM strategies WHERE name = ?
                    """,
                    (name,),
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_strategy(row)
            except sqlite3.Error as e:
                raise StrategyMemoryError(f"Failed to get strategy: {e}") from e

    def get_best_strategies(self, limit: int = 5) -> list[Strategy]:
        """Get the most effective strategies.

        Orders by effectiveness (success_count / (success_count + failure_count)),
        then by total uses (more data preferred).

        Args:
            limit: Maximum number of strategies to return.

        Returns:
            List of strategies, most effective first.
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute(
                    """
                    SELECT name, description, success_count, failure_count, last_used
                    FROM strategies
                    ORDER BY
                        CAST(success_count AS REAL) / NULLIF(success_count + failure_count, 0) DESC,
                        (success_count + failure_count) DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                return [self._row_to_strategy(row) for row in cursor.fetchall()]
            except sqlite3.Error as e:
                raise StrategyMemoryError(f"Failed to get best strategies: {e}") from e

    def update_strategy_outcome(self, name: str, success: bool) -> None:
        """Update a strategy's success/failure count.

        Args:
            name: Name of the strategy.
            success: Whether the strategy succeeded.

        Raises:
            StrategyMemoryError: If strategy not found or update fails.
        """
        with self._lock:
            try:
                existing = self.get_strategy(name)
                if existing is None:
                    raise StrategyMemoryError(f"Strategy not found: {name}")

                if success:
                    new_success = existing.success_count + 1
                    new_failure = existing.failure_count
                else:
                    new_success = existing.success_count
                    new_failure = existing.failure_count + 1

                conn = self._get_connection()
                now = datetime.now().isoformat()
                with conn:
                    conn.execute(
                        """
                        UPDATE strategies
                        SET success_count = ?, failure_count = ?, last_used = ?
                        WHERE name = ?
                        """,
                        (new_success, new_failure, now, name),
                    )
                logger.debug(f"Updated outcome for {name}: success={success}")
            except sqlite3.Error as e:
                raise StrategyMemoryError(f"Failed to update strategy outcome: {e}") from e

    def recommend_for_situation(
        self,
        game_state: dict[str, Any],  # noqa: ARG002
    ) -> Strategy | None:
        """Recommend the highest-rated strategy for a game state.

        Currently returns the single best strategy by effectiveness.
        Callers can use this when no situation-specific filtering is needed.

        Args:
            game_state: Current game state (reserved for future situation-aware
                filtering).

        Returns:
            The best strategy, or None if no strategies exist.
        """
        best = self.get_best_strategies(limit=1)
        if not best:
            return None
        return best[0]

    def identify_patterns(
        self,
        episodes: list[Episode],
    ) -> list[dict[str, Any]]:
        """Identify patterns from episodes (e.g. by action type and outcome).

        Groups episodes by strategy name (from action_taken) or action type,
        computes success rate, and returns named patterns.

        Args:
            episodes: List of episodes to analyze.

        Returns:
            List of pattern dicts with keys such as name, description,
            success_rate, success_count, total_count.
        """
        if not episodes:
            return []

        # Group by strategy name if present, else by action type
        groups: dict[str, list[bool]] = defaultdict(list)

        for ep in episodes:
            action = ep.action_taken or {}
            strategy_name = action.get("strategy")
            if strategy_name:
                key = f"strategy:{strategy_name}"
            else:
                key = f"action:{action.get('type', 'unknown')}"
            groups[key].append(ep.success)

        patterns: list[dict[str, Any]] = []
        for key, outcomes in groups.items():
            total = len(outcomes)
            successes = sum(1 for o in outcomes if o)
            rate = successes / total if total else 0.0
            kind, name = key.split(":", 1) if ":" in key else ("action", key)
            patterns.append(
                {
                    "name": name,
                    "description": f"{kind} '{name}' success rate over {total} episodes",
                    "success_rate": rate,
                    "success_count": successes,
                    "total_count": total,
                }
            )

        # Sort by success rate descending
        patterns.sort(key=lambda p: (p["success_rate"], p["total_count"]), reverse=True)
        return patterns

    def get_strategy_count(self) -> int:
        """Get the total number of stored strategies.

        Returns:
            Number of strategies in the database.
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute("SELECT COUNT(*) FROM strategies")
                row = cursor.fetchone()
                return row[0] if row else 0
            except sqlite3.Error as e:
                raise StrategyMemoryError(f"Failed to count strategies: {e}") from e

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
            logger.debug("Strategy memory connection closed")

    def __enter__(self) -> SQLiteStrategyMemory:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()
