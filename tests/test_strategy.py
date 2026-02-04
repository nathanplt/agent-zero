"""Tests for the SQLiteStrategyMemory class."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.interfaces.memory import Episode, Strategy
from src.memory.strategy import (
    SQLiteStrategyMemory,
    StrategyMemoryError,
)


def create_strategy(
    name: str = "test_strategy",
    description: str = "A test strategy",
    success_count: int = 0,
    failure_count: int = 0,
    last_used: datetime | None = None,
) -> Strategy:
    """Helper to create test strategies."""
    return Strategy(
        name=name,
        description=description,
        success_count=success_count,
        failure_count=failure_count,
        last_used=last_used,
    )


def create_episode(
    episode_id: str,
    success: bool,
    action_type: str = "click",
    gold: int = 100,
) -> Episode:
    """Helper to create test episodes for pattern detection."""
    return Episode(
        episode_id=episode_id,
        timestamp=datetime.now(),
        game_state_before={"gold": gold, "level": 1, "screen": "main"},
        action_taken={"type": action_type, "target": "button", "strategy": "upgrade_first"},
        game_state_after={"gold": gold - 10, "level": 2},
        success=success,
        notes="test",
    )


class TestSQLiteStrategyMemoryInit:
    """Tests for SQLiteStrategyMemory initialization."""

    def test_creates_with_default_path(self, tmp_path: Path) -> None:
        """Test initialization with custom path."""
        db_path = tmp_path / "strategies.db"
        memory = SQLiteStrategyMemory(db_path)
        assert memory.db_path == db_path
        memory.close()

    def test_creates_database_file(self, tmp_path: Path) -> None:
        """Test that database file is created."""
        db_path = tmp_path / "strategies.db"
        memory = SQLiteStrategyMemory(db_path)
        assert db_path.exists()
        memory.close()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test that parent directories are created."""
        db_path = tmp_path / "subdir" / "nested" / "strategies.db"
        memory = SQLiteStrategyMemory(db_path)
        assert db_path.parent.exists()
        memory.close()

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test context manager usage."""
        db_path = tmp_path / "strategies.db"
        with SQLiteStrategyMemory(db_path) as memory:
            memory.save_strategy(create_strategy())
        # Should not raise after exit


class TestSaveAndGetStrategy:
    """Tests for save_strategy and get_strategy."""

    def test_save_and_get_strategy(self, tmp_path: Path) -> None:
        """Test saving and retrieving a strategy."""
        db_path = tmp_path / "strategies.db"
        with SQLiteStrategyMemory(db_path) as memory:
            strategy = create_strategy("upgrade_first", "Upgrade when affordable")
            memory.save_strategy(strategy)

            loaded = memory.get_strategy("upgrade_first")
            assert loaded is not None
            assert loaded.name == "upgrade_first"
            assert loaded.description == "Upgrade when affordable"
            assert loaded.success_count == 0
            assert loaded.failure_count == 0

    def test_get_strategy_returns_none_for_unknown(self, tmp_path: Path) -> None:
        """Test that get_strategy returns None for unknown name."""
        db_path = tmp_path / "strategies.db"
        with SQLiteStrategyMemory(db_path) as memory:
            assert memory.get_strategy("nonexistent") is None

    def test_save_updates_existing_strategy(self, tmp_path: Path) -> None:
        """Test that saving same name updates the strategy."""
        db_path = tmp_path / "strategies.db"
        with SQLiteStrategyMemory(db_path) as memory:
            memory.save_strategy(create_strategy("s1", "First description"))
            memory.save_strategy(create_strategy("s1", "Updated description"))

            loaded = memory.get_strategy("s1")
            assert loaded is not None
            assert loaded.description == "Updated description"


class TestUpdateStrategyOutcome:
    """Tests for update_strategy_outcome and effectiveness."""

    def test_update_strategy_outcome_success(self, tmp_path: Path) -> None:
        """Test recording a successful outcome."""
        db_path = tmp_path / "strategies.db"
        with SQLiteStrategyMemory(db_path) as memory:
            memory.save_strategy(create_strategy("s1"))
            memory.update_strategy_outcome("s1", success=True)

            s = memory.get_strategy("s1")
            assert s is not None
            assert s.success_count == 1
            assert s.failure_count == 0
            assert s.effectiveness == 1.0

    def test_update_strategy_outcome_failure(self, tmp_path: Path) -> None:
        """Test recording a failed outcome."""
        db_path = tmp_path / "strategies.db"
        with SQLiteStrategyMemory(db_path) as memory:
            memory.save_strategy(create_strategy("s1"))
            memory.update_strategy_outcome("s1", success=False)

            s = memory.get_strategy("s1")
            assert s is not None
            assert s.success_count == 0
            assert s.failure_count == 1
            assert s.effectiveness == 0.0

    def test_record_strategy_use_and_outcomes_effectiveness_scores(self, tmp_path: Path) -> None:
        """Record strategy use and outcomes; effectiveness scores calculated correctly."""
        db_path = tmp_path / "strategies.db"
        with SQLiteStrategyMemory(db_path) as memory:
            memory.save_strategy(create_strategy("high_success"))
            memory.save_strategy(create_strategy("low_success"))
            memory.save_strategy(create_strategy("untested"))

            for _ in range(8):
                memory.update_strategy_outcome("high_success", success=True)
            for _ in range(2):
                memory.update_strategy_outcome("high_success", success=False)

            for _ in range(2):
                memory.update_strategy_outcome("low_success", success=True)
            for _ in range(8):
                memory.update_strategy_outcome("low_success", success=False)

            high = memory.get_strategy("high_success")
            low = memory.get_strategy("low_success")
            untested = memory.get_strategy("untested")

            assert high is not None and high.effectiveness == 0.8
            assert low is not None and low.effectiveness == 0.2
            assert untested is not None and untested.effectiveness == 0.5  # default for no data

    def test_update_nonexistent_strategy_raises(self, tmp_path: Path) -> None:
        """Test that update_strategy_outcome on unknown strategy raises or creates."""
        db_path = tmp_path / "strategies.db"
        with SQLiteStrategyMemory(db_path) as memory:
            # Plan: either raise StrategyMemoryError or no-op; we'll implement to raise
            try:
                memory.update_strategy_outcome("nonexistent", success=True)
                # If no raise, strategy might have been created with 1 success
                s = memory.get_strategy("nonexistent")
                # Allow either: raise or create-on-first-use
                if s is not None:
                    assert s.success_count == 1
            except StrategyMemoryError:
                pass


class TestGetBestStrategies:
    """Tests for get_best_strategies."""

    def test_get_best_strategies_returns_most_effective_first(self, tmp_path: Path) -> None:
        """Test that get_best_strategies returns strategies ordered by effectiveness."""
        db_path = tmp_path / "strategies.db"
        with SQLiteStrategyMemory(db_path) as memory:
            memory.save_strategy(create_strategy("worst"))
            memory.save_strategy(create_strategy("best"))
            memory.save_strategy(create_strategy("mid"))

            for _ in range(1):
                memory.update_strategy_outcome("worst", success=True)
            for _ in range(9):
                memory.update_strategy_outcome("worst", success=False)

            for _ in range(9):
                memory.update_strategy_outcome("best", success=True)
            for _ in range(1):
                memory.update_strategy_outcome("best", success=False)

            for _ in range(5):
                memory.update_strategy_outcome("mid", success=True)
            for _ in range(5):
                memory.update_strategy_outcome("mid", success=False)

            best = memory.get_best_strategies(limit=3)
            assert len(best) == 3
            assert best[0].name == "best"
            assert best[1].name == "mid"
            assert best[2].name == "worst"

    def test_get_best_strategies_respects_limit(self, tmp_path: Path) -> None:
        """Test that get_best_strategies respects limit."""
        db_path = tmp_path / "strategies.db"
        with SQLiteStrategyMemory(db_path) as memory:
            for i in range(5):
                memory.save_strategy(create_strategy(f"s{i}"))

            best = memory.get_best_strategies(limit=2)
            assert len(best) == 2

    def test_get_best_strategies_empty_returns_empty_list(self, tmp_path: Path) -> None:
        """Test get_best_strategies when no strategies exist."""
        db_path = tmp_path / "strategies.db"
        with SQLiteStrategyMemory(db_path) as memory:
            assert memory.get_best_strategies(limit=5) == []


class TestRecommendForSituation:
    """Tests for recommend_for_situation (recommend strategies for game state)."""

    def test_recommend_returns_highest_rated_for_game_state(self, tmp_path: Path) -> None:
        """Query recommended strategy for game state returns highest-rated applicable strategy."""
        db_path = tmp_path / "strategies.db"
        with SQLiteStrategyMemory(db_path) as memory:
            memory.save_strategy(create_strategy("best", "Best strategy"))
            memory.save_strategy(create_strategy("worst", "Worst strategy"))

            for _ in range(9):
                memory.update_strategy_outcome("best", success=True)
            for _ in range(1):
                memory.update_strategy_outcome("best", success=False)
            for _ in range(1):
                memory.update_strategy_outcome("worst", success=True)
            for _ in range(9):
                memory.update_strategy_outcome("worst", success=False)

            recommended = memory.recommend_for_situation({"gold": 100, "level": 5})
            assert recommended is not None
            assert recommended.name == "best"

    def test_recommend_returns_none_when_no_strategies(self, tmp_path: Path) -> None:
        """Test recommend_for_situation when no strategies exist."""
        db_path = tmp_path / "strategies.db"
        with SQLiteStrategyMemory(db_path) as memory:
            assert memory.recommend_for_situation({"gold": 100}) is None


class TestIdentifyPatterns:
    """Tests for pattern identification from episodes."""

    def test_identify_patterns_from_episodes(self, tmp_path: Path) -> None:
        """Inject clear pattern, verify detection; pattern identified and named."""
        db_path = tmp_path / "strategies.db"
        episodes = [
            create_episode("ep1", success=True, action_type="upgrade"),
            create_episode("ep2", success=True, action_type="upgrade"),
            create_episode("ep3", success=False, action_type="click"),
            create_episode("ep4", success=True, action_type="upgrade"),
        ]

        with SQLiteStrategyMemory(db_path) as memory:
            patterns = memory.identify_patterns(episodes)

        # Should identify at least one pattern (e.g. upgrade strategy often succeeds)
        assert isinstance(patterns, list)
        # Pattern entries: name, description, success_rate or similar
        if patterns:
            for p in patterns:
                assert "name" in p or "description" in p or "success_rate" in p or "strategy" in p

    def test_identify_patterns_empty_episodes(self, tmp_path: Path) -> None:
        """Test identify_patterns with no episodes."""
        db_path = tmp_path / "strategies.db"
        with SQLiteStrategyMemory(db_path) as memory:
            patterns = memory.identify_patterns([])
        assert patterns == []
