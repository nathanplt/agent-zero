"""Episodic memory for recording and retrieving action-outcome pairs.

This module provides episodic memory functionality, allowing the agent to:
- Record episodes (action-outcome pairs with game state context)
- Find similar past episodes for decision making
- Annotate success/failure outcomes
- Prune old/irrelevant memories

The similarity search uses a simple but effective approach:
- Extract key features from game states
- Compare feature vectors using cosine similarity
- Weight recent episodes higher

Example:
    >>> from src.memory.episodic import SQLiteEpisodicMemory
    >>> from src.interfaces.memory import Episode
    >>> from datetime import datetime
    >>>
    >>> memory = SQLiteEpisodicMemory("data/episodes.db")
    >>>
    >>> # Record an episode
    >>> episode = Episode(
    ...     episode_id="ep_001",
    ...     timestamp=datetime.now(),
    ...     game_state_before={"gold": 100, "level": 1},
    ...     action_taken={"type": "click", "target": "upgrade_button"},
    ...     game_state_after={"gold": 50, "level": 2},
    ...     success=True,
    ...     notes="Upgrade successful",
    ... )
    >>> memory.record_episode(episode)
    >>>
    >>> # Find similar episodes
    >>> similar = memory.find_similar_episodes({"gold": 90, "level": 1})
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from src.interfaces.memory import Episode

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = Path("data/episodes.db")

# Schema version for migrations
SCHEMA_VERSION = 1


class EpisodicMemoryError(Exception):
    """Error raised when episodic memory operations fail."""

    pass


class SQLiteEpisodicMemory:
    """SQLite-based episodic memory for action-outcome pairs.

    This class provides persistent storage for episodes using SQLite.
    It supports:
    - Recording episodes with game state context
    - Finding similar past episodes using feature-based similarity
    - Querying recent episodes
    - Success/failure annotation
    - Automatic pruning of old episodes

    The similarity search extracts numeric features from game states
    and uses cosine similarity to find matching episodes.

    Attributes:
        db_path: Path to the SQLite database file.

    Example:
        >>> memory = SQLiteEpisodicMemory("episodes.db")
        >>> memory.record_episode(episode)
        >>> similar = memory.find_similar_episodes(current_state, limit=5)
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        auto_init: bool = True,
        max_episodes: int = 10000,
    ) -> None:
        """Initialize the episodic memory.

        Args:
            db_path: Path to the SQLite database file.
                    Uses DEFAULT_DB_PATH if not specified.
            auto_init: Whether to automatically initialize the database.
            max_episodes: Maximum number of episodes to keep (for auto-pruning).
        """
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._lock = threading.RLock()  # Reentrant lock to allow nested calls
        self._local = threading.local()
        self._max_episodes = max_episodes

        if auto_init:
            self._ensure_db_exists()
            self._init_schema()

        logger.debug(f"SQLiteEpisodicMemory initialized: {self._db_path}")

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
            # Episodes table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    state_before_json TEXT NOT NULL,
                    action_json TEXT NOT NULL,
                    state_after_json TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    notes TEXT,
                    features_json TEXT
                )
            """)

            # Indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_episodes_timestamp
                ON episodes(timestamp DESC)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_episodes_success
                ON episodes(success)
            """)

            conn.execute("DELETE FROM schema_version")
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )

            logger.info(f"Episodic memory schema migrated to version {SCHEMA_VERSION}")

    @staticmethod
    def _extract_features(state: dict[str, Any]) -> dict[str, float]:
        """Extract numeric features from a game state for similarity comparison.

        This method recursively extracts all numeric values from the state,
        creating a flat feature vector.

        Args:
            state: Game state dictionary.

        Returns:
            Dictionary mapping feature names to numeric values.
        """
        features: dict[str, float] = {}

        def extract_recursive(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    extract_recursive(value, new_prefix)
            elif isinstance(obj, (list, tuple)):
                # Extract list length as a feature
                features[f"{prefix}._length"] = float(len(obj))
                for i, item in enumerate(obj[:10]):  # Limit to first 10 items
                    extract_recursive(item, f"{prefix}[{i}]")
            elif isinstance(obj, bool):
                features[prefix] = 1.0 if obj else 0.0
            elif isinstance(obj, (int, float)):
                features[prefix] = float(obj)
            elif isinstance(obj, str):
                # Hash strings to numeric values for comparison
                features[f"{prefix}._hash"] = float(
                    int(hashlib.md5(obj.encode()).hexdigest()[:8], 16) % 1000000
                )

        extract_recursive(state)
        return features

    @staticmethod
    def _compute_similarity(
        features1: dict[str, float],
        features2: dict[str, float],
    ) -> float:
        """Compute cosine similarity between two feature vectors.

        Args:
            features1: First feature vector.
            features2: Second feature vector.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        # Get common keys
        common_keys = set(features1.keys()) & set(features2.keys())

        if not common_keys:
            return 0.0

        # Compute cosine similarity
        dot_product = sum(features1[k] * features2[k] for k in common_keys)
        norm1 = math.sqrt(sum(features1[k] ** 2 for k in common_keys))
        norm2 = math.sqrt(sum(features2[k] ** 2 for k in common_keys))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # Bonus for having more common keys (feature overlap)
        all_keys = set(features1.keys()) | set(features2.keys())
        overlap_bonus = len(common_keys) / len(all_keys) if all_keys else 0

        # Combined score (weighted)
        return 0.7 * similarity + 0.3 * overlap_bonus

    def generate_episode_id(self) -> str:
        """Generate a unique episode ID.

        Returns:
            Unique episode ID string.
        """
        return f"ep_{uuid.uuid4().hex[:12]}"

    def record_episode(self, episode: Episode) -> None:
        """Record an episode of action and outcome.

        Args:
            episode: The episode to record.

        Raises:
            EpisodicMemoryError: If recording fails.
        """
        # Extract features for similarity search
        features = self._extract_features(episode.game_state_before)

        with self._lock:
            try:
                conn = self._get_connection()
                with conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO episodes
                        (id, timestamp, state_before_json, action_json, state_after_json,
                         success, notes, features_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            episode.id,
                            episode.timestamp.isoformat(),
                            json.dumps(episode.game_state_before, default=str),
                            json.dumps(episode.action_taken, default=str),
                            json.dumps(episode.game_state_after, default=str),
                            1 if episode.success else 0,
                            episode.notes,
                            json.dumps(features),
                        ),
                    )

                logger.debug(f"Recorded episode {episode.id}")

                # Auto-prune if needed
                self._auto_prune()

            except sqlite3.Error as e:
                raise EpisodicMemoryError(f"Failed to record episode: {e}") from e

    def get_episode(self, episode_id: str) -> Episode | None:
        """Get a specific episode by ID.

        Args:
            episode_id: The episode ID.

        Returns:
            The episode, or None if not found.
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute(
                    """
                    SELECT id, timestamp, state_before_json, action_json,
                           state_after_json, success, notes
                    FROM episodes WHERE id = ?
                    """,
                    (episode_id,),
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                return self._row_to_episode(row)

            except sqlite3.Error as e:
                raise EpisodicMemoryError(f"Failed to get episode: {e}") from e

    def _row_to_episode(self, row: tuple[Any, ...]) -> Episode:
        """Convert a database row to an Episode object."""
        (
            episode_id,
            timestamp_str,
            state_before_json,
            action_json,
            state_after_json,
            success,
            notes,
        ) = row

        return Episode(
            episode_id=episode_id,
            timestamp=datetime.fromisoformat(timestamp_str),
            game_state_before=json.loads(state_before_json),
            action_taken=json.loads(action_json),
            game_state_after=json.loads(state_after_json),
            success=bool(success),
            notes=notes,
        )

    def find_similar_episodes(
        self,
        current_state: dict[str, Any],
        limit: int = 10,
        *,
        success_only: bool = False,
        min_similarity: float = 0.1,
    ) -> list[Episode]:
        """Find episodes with similar game states.

        Args:
            current_state: The current game state to match against.
            limit: Maximum number of episodes to return.
            success_only: If True, only return successful episodes.
            min_similarity: Minimum similarity threshold (0.0 to 1.0).

        Returns:
            List of similar episodes, most similar first.
        """
        current_features = self._extract_features(current_state)

        with self._lock:
            try:
                conn = self._get_connection()

                # Build query
                query = """
                    SELECT id, timestamp, state_before_json, action_json,
                           state_after_json, success, notes, features_json
                    FROM episodes
                """
                params: list[Any] = []

                if success_only:
                    query += " WHERE success = 1"

                cursor = conn.execute(query, params)

                # Score and sort episodes
                scored_episodes: list[tuple[float, Episode]] = []

                for row in cursor.fetchall():
                    features_json = row[7]
                    if features_json:
                        stored_features: dict[str, float] = json.loads(features_json)
                        similarity = self._compute_similarity(
                            current_features, stored_features
                        )

                        if similarity >= min_similarity:
                            episode = self._row_to_episode(row[:7])
                            scored_episodes.append((similarity, episode))

                # Sort by similarity (descending)
                scored_episodes.sort(key=lambda x: x[0], reverse=True)

                return [ep for _, ep in scored_episodes[:limit]]

            except sqlite3.Error as e:
                raise EpisodicMemoryError(f"Failed to find similar episodes: {e}") from e

    def get_recent_episodes(self, limit: int = 10) -> list[Episode]:
        """Get the most recent episodes.

        Args:
            limit: Maximum number of episodes to return.

        Returns:
            List of episodes, most recent first.
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute(
                    """
                    SELECT id, timestamp, state_before_json, action_json,
                           state_after_json, success, notes
                    FROM episodes
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

                return [self._row_to_episode(row) for row in cursor.fetchall()]

            except sqlite3.Error as e:
                raise EpisodicMemoryError(f"Failed to get recent episodes: {e}") from e

    def get_episodes_by_action_type(
        self,
        action_type: str,
        limit: int = 10,
    ) -> list[Episode]:
        """Get episodes by action type.

        Args:
            action_type: The action type to filter by.
            limit: Maximum number of episodes to return.

        Returns:
            List of episodes with the specified action type.
        """
        with self._lock:
            try:
                conn = self._get_connection()
                # Use JSON extraction for SQLite
                cursor = conn.execute(
                    """
                    SELECT id, timestamp, state_before_json, action_json,
                           state_after_json, success, notes
                    FROM episodes
                    WHERE json_extract(action_json, '$.type') = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (action_type, limit),
                )

                return [self._row_to_episode(row) for row in cursor.fetchall()]

            except sqlite3.Error as e:
                raise EpisodicMemoryError(
                    f"Failed to get episodes by action type: {e}"
                ) from e

    def get_success_rate(
        self,
        action_type: str | None = None,
        limit: int | None = None,
    ) -> tuple[int, int, float]:
        """Get success rate statistics.

        Args:
            action_type: Optional action type to filter by.
            limit: Optional limit on episodes to consider.

        Returns:
            Tuple of (success_count, total_count, success_rate).
        """
        with self._lock:
            try:
                conn = self._get_connection()

                if action_type:
                    query = """
                        SELECT success FROM episodes
                        WHERE json_extract(action_json, '$.type') = ?
                        ORDER BY timestamp DESC
                    """
                    params: list[Any] = [action_type]
                else:
                    query = "SELECT success FROM episodes ORDER BY timestamp DESC"
                    params = []

                if limit:
                    query += f" LIMIT {limit}"

                cursor = conn.execute(query, params)
                results = cursor.fetchall()

                total = len(results)
                successes = sum(1 for (s,) in results if s)

                rate = successes / total if total > 0 else 0.0

                return successes, total, rate

            except sqlite3.Error as e:
                raise EpisodicMemoryError(f"Failed to get success rate: {e}") from e

    def get_episode_count(self) -> int:
        """Get the total number of episodes.

        Returns:
            Number of episodes in the database.
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute("SELECT COUNT(*) FROM episodes")
                result = cursor.fetchone()
                return result[0] if result else 0
            except sqlite3.Error as e:
                raise EpisodicMemoryError(f"Failed to count episodes: {e}") from e

    def delete_episode(self, episode_id: str) -> bool:
        """Delete a specific episode.

        Args:
            episode_id: ID of the episode to delete.

        Returns:
            True if episode was deleted, False if not found.
        """
        with self._lock:
            try:
                conn = self._get_connection()
                with conn:
                    cursor = conn.execute(
                        "DELETE FROM episodes WHERE id = ?",
                        (episode_id,),
                    )
                    return cursor.rowcount > 0
            except sqlite3.Error as e:
                raise EpisodicMemoryError(f"Failed to delete episode: {e}") from e

    def prune_old_episodes(self, keep_count: int | None = None) -> int:
        """Prune old episodes to manage storage.

        Keeps the most recent episodes and deletes the rest.

        Args:
            keep_count: Number of episodes to keep. Uses max_episodes if None.

        Returns:
            Number of episodes deleted.
        """
        keep = keep_count if keep_count is not None else self._max_episodes

        with self._lock:
            try:
                conn = self._get_connection()
                with conn:
                    # Get ID of the Nth most recent episode
                    cursor = conn.execute(
                        """
                        SELECT id FROM episodes
                        ORDER BY timestamp DESC
                        LIMIT 1 OFFSET ?
                        """,
                        (keep,),
                    )
                    row = cursor.fetchone()

                    if row is None:
                        return 0

                    # Get timestamp threshold
                    cursor = conn.execute(
                        "SELECT timestamp FROM episodes WHERE id = ?",
                        (row[0],),
                    )
                    threshold_row = cursor.fetchone()

                    if threshold_row is None:
                        return 0

                    threshold_timestamp = threshold_row[0]

                    # Delete older episodes
                    cursor = conn.execute(
                        "DELETE FROM episodes WHERE timestamp < ?",
                        (threshold_timestamp,),
                    )
                    deleted = cursor.rowcount

                    if deleted > 0:
                        logger.info(f"Pruned {deleted} old episodes")

                    return deleted

            except sqlite3.Error as e:
                raise EpisodicMemoryError(f"Failed to prune episodes: {e}") from e

    def _auto_prune(self) -> None:
        """Automatically prune if episode count exceeds max."""
        count = self.get_episode_count()
        if count > self._max_episodes * 1.1:  # 10% buffer
            self.prune_old_episodes()

    def clear_all(self) -> int:
        """Clear all episodes.

        Returns:
            Number of episodes deleted.
        """
        with self._lock:
            try:
                conn = self._get_connection()
                with conn:
                    cursor = conn.execute("DELETE FROM episodes")
                    deleted = cursor.rowcount
                    logger.warning(f"Cleared all {deleted} episodes")
                    return deleted
            except sqlite3.Error as e:
                raise EpisodicMemoryError(f"Failed to clear episodes: {e}") from e

    def get_action_type_stats(self) -> dict[str, tuple[int, int, float]]:
        """Get statistics grouped by action type.

        Returns:
            Dictionary mapping action type to (successes, total, rate).
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute(
                    """
                    SELECT json_extract(action_json, '$.type') as action_type,
                           SUM(success) as successes,
                           COUNT(*) as total
                    FROM episodes
                    GROUP BY action_type
                    """
                )

                stats: dict[str, tuple[int, int, float]] = {}
                for action_type, successes, total in cursor.fetchall():
                    if action_type:
                        rate = successes / total if total > 0 else 0.0
                        stats[action_type] = (int(successes), int(total), rate)

                return stats

            except sqlite3.Error as e:
                raise EpisodicMemoryError(f"Failed to get action stats: {e}") from e

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
            logger.debug("Episodic memory connection closed")

    def __enter__(self) -> SQLiteEpisodicMemory:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()
