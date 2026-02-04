"""Memory system interface for persistent state and learning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class Episode:
    """A recorded episode of agent action and outcome."""

    __slots__ = (
        "id",
        "timestamp",
        "game_state_before",
        "action_taken",
        "game_state_after",
        "success",
        "notes",
    )

    def __init__(
        self,
        episode_id: str,
        timestamp: datetime,
        game_state_before: dict[str, Any],
        action_taken: dict[str, Any],
        game_state_after: dict[str, Any],
        success: bool,
        notes: str | None = None,
    ) -> None:
        """Initialize an episode.

        Args:
            episode_id: Unique identifier for this episode.
            timestamp: When this episode occurred.
            game_state_before: Game state before the action.
            action_taken: Description of the action performed.
            game_state_after: Game state after the action.
            success: Whether the action achieved its goal.
            notes: Optional notes about the episode.
        """
        self.id = episode_id
        self.timestamp = timestamp
        self.game_state_before = game_state_before
        self.action_taken = action_taken
        self.game_state_after = game_state_after
        self.success = success
        self.notes = notes


class Strategy:
    """A named strategy with effectiveness tracking."""

    __slots__ = ("name", "description", "success_count", "failure_count", "last_used")

    def __init__(
        self,
        name: str,
        description: str,
        success_count: int = 0,
        failure_count: int = 0,
        last_used: datetime | None = None,
    ) -> None:
        """Initialize a strategy.

        Args:
            name: Unique name for this strategy.
            description: What this strategy does.
            success_count: Number of successful uses.
            failure_count: Number of failed uses.
            last_used: When this strategy was last used.
        """
        self.name = name
        self.description = description
        self.success_count = success_count
        self.failure_count = failure_count
        self.last_used = last_used

    @property
    def effectiveness(self) -> float:
        """Calculate the effectiveness ratio (0.0 to 1.0)."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Default for untested strategies
        return self.success_count / total


class MemoryStore(ABC):
    """Abstract interface for the memory system.

    The memory system is responsible for:
    - Persisting game state across sessions
    - Recording episodes of actions and outcomes
    - Tracking strategy effectiveness
    - Finding similar past situations
    """

    # Game State Persistence

    @abstractmethod
    def save_state(self, state: dict[str, Any]) -> None:
        """Save the current game state.

        Args:
            state: Game state dictionary to persist.
        """
        ...

    @abstractmethod
    def load_state(self) -> dict[str, Any] | None:
        """Load the most recent game state.

        Returns:
            Most recent game state, or None if no state saved.
        """
        ...

    @abstractmethod
    def get_state_history(
        self,
        limit: int = 10,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Get historical game states.

        Args:
            limit: Maximum number of states to return.
            since: Only return states after this time.

        Returns:
            List of game states, most recent first.
        """
        ...

    # Episodic Memory

    @abstractmethod
    def record_episode(self, episode: Episode) -> None:
        """Record an episode of action and outcome.

        Args:
            episode: The episode to record.
        """
        ...

    @abstractmethod
    def find_similar_episodes(
        self,
        current_state: dict[str, Any],
        limit: int = 10,
    ) -> list[Episode]:
        """Find episodes with similar game states.

        Args:
            current_state: The current game state to match against.
            limit: Maximum number of episodes to return.

        Returns:
            List of similar episodes, most similar first.
        """
        ...

    @abstractmethod
    def get_recent_episodes(self, limit: int = 10) -> list[Episode]:
        """Get the most recent episodes.

        Args:
            limit: Maximum number of episodes to return.

        Returns:
            List of episodes, most recent first.
        """
        ...

    # Strategy Memory

    @abstractmethod
    def save_strategy(self, strategy: Strategy) -> None:
        """Save or update a strategy.

        Args:
            strategy: The strategy to save.
        """
        ...

    @abstractmethod
    def get_strategy(self, name: str) -> Strategy | None:
        """Get a strategy by name.

        Args:
            name: Name of the strategy.

        Returns:
            The strategy, or None if not found.
        """
        ...

    @abstractmethod
    def get_best_strategies(self, limit: int = 5) -> list[Strategy]:
        """Get the most effective strategies.

        Args:
            limit: Maximum number of strategies to return.

        Returns:
            List of strategies, most effective first.
        """
        ...

    @abstractmethod
    def update_strategy_outcome(self, name: str, success: bool) -> None:
        """Update a strategy's success/failure count.

        Args:
            name: Name of the strategy.
            success: Whether the strategy succeeded.
        """
        ...

    # Maintenance

    @abstractmethod
    def prune_old_data(self, keep_episodes: int = 10000) -> int:
        """Prune old data to manage storage.

        Args:
            keep_episodes: Number of episodes to keep.

        Returns:
            Number of records deleted.
        """
        ...


class MemoryError(Exception):
    """Error raised when memory operations fail."""

    pass
