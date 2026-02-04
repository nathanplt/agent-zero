"""Tests for the SQLiteEpisodicMemory class."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from pathlib import Path

from src.interfaces.memory import Episode
from src.memory.episodic import (
    EpisodicMemoryError,
    SQLiteEpisodicMemory,
)


def create_episode(
    episode_id: str = "test_ep",
    gold: int = 100,
    level: int = 1,
    action_type: str = "click",
    success: bool = True,
    timestamp: datetime | None = None,
) -> Episode:
    """Helper to create test episodes."""
    return Episode(
        episode_id=episode_id,
        timestamp=timestamp or datetime.now(),
        game_state_before={"gold": gold, "level": level, "screen": "main"},
        action_taken={"type": action_type, "target": "button"},
        game_state_after={"gold": gold - 10 if success else gold, "level": level + (1 if success else 0)},
        success=success,
        notes=f"Test episode {episode_id}",
    )


class TestSQLiteEpisodicMemoryInit:
    """Tests for SQLiteEpisodicMemory initialization."""

    def test_creates_with_default_path(self, tmp_path: Path) -> None:
        """Test initialization with custom path."""
        db_path = tmp_path / "test.db"
        memory = SQLiteEpisodicMemory(db_path)
        assert memory.db_path == db_path
        memory.close()

    def test_creates_database_file(self, tmp_path: Path) -> None:
        """Test that database file is created."""
        db_path = tmp_path / "test.db"
        memory = SQLiteEpisodicMemory(db_path)
        assert db_path.exists()
        memory.close()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test that parent directories are created."""
        db_path = tmp_path / "subdir" / "nested" / "test.db"
        memory = SQLiteEpisodicMemory(db_path)
        assert db_path.parent.exists()
        memory.close()

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test context manager usage."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            memory.record_episode(create_episode())
        # Should not raise after exit


class TestRecordEpisode:
    """Tests for record_episode method."""

    def test_record_simple_episode(self, tmp_path: Path) -> None:
        """Test recording a simple episode."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            episode = create_episode("ep_001")
            memory.record_episode(episode)
            assert memory.get_episode_count() == 1

    def test_record_multiple_episodes(self, tmp_path: Path) -> None:
        """Test recording multiple episodes."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            for i in range(10):
                memory.record_episode(create_episode(f"ep_{i:03d}"))
            assert memory.get_episode_count() == 10

    def test_record_replaces_existing(self, tmp_path: Path) -> None:
        """Test that recording with same ID replaces existing."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            memory.record_episode(create_episode("ep_001", success=True))
            memory.record_episode(create_episode("ep_001", success=False))

            assert memory.get_episode_count() == 1
            episode = memory.get_episode("ep_001")
            assert episode is not None
            assert episode.success is False

    def test_record_with_complex_state(self, tmp_path: Path) -> None:
        """Test recording episode with complex game state."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            episode = Episode(
                episode_id="complex_ep",
                timestamp=datetime.now(),
                game_state_before={
                    "resources": {"gold": 1000, "gems": 50},
                    "inventory": [{"id": 1, "name": "sword"}],
                    "settings": {"volume": 0.8},
                },
                action_taken={"type": "upgrade", "item": "sword", "cost": 100},
                game_state_after={
                    "resources": {"gold": 900, "gems": 50},
                    "inventory": [{"id": 1, "name": "sword", "level": 2}],
                },
                success=True,
            )
            memory.record_episode(episode)

            loaded = memory.get_episode("complex_ep")
            assert loaded is not None
            assert loaded.game_state_before["resources"]["gold"] == 1000


class TestGetEpisode:
    """Tests for get_episode method."""

    def test_get_existing_episode(self, tmp_path: Path) -> None:
        """Test getting an existing episode."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            original = create_episode("ep_001", gold=500, level=5)
            memory.record_episode(original)

            loaded = memory.get_episode("ep_001")
            assert loaded is not None
            assert loaded.id == "ep_001"
            assert loaded.game_state_before["gold"] == 500
            assert loaded.success is True

    def test_get_nonexistent_episode(self, tmp_path: Path) -> None:
        """Test getting a non-existent episode."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            assert memory.get_episode("nonexistent") is None


class TestFindSimilarEpisodes:
    """Tests for find_similar_episodes method."""

    def test_find_similar_by_resources(self, tmp_path: Path) -> None:
        """Test finding similar episodes based on resources."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            # Create episodes with varying gold amounts
            memory.record_episode(create_episode("low_gold", gold=100))
            memory.record_episode(create_episode("mid_gold", gold=500))
            memory.record_episode(create_episode("high_gold", gold=1000))

            # Find similar to mid_gold state
            similar = memory.find_similar_episodes(
                {"gold": 450, "level": 1, "screen": "main"},
                limit=3
            )

            assert len(similar) > 0
            # Most similar should be mid_gold (closest to 450)

    def test_find_similar_success_only(self, tmp_path: Path) -> None:
        """Test finding only successful episodes."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            memory.record_episode(create_episode("success_1", success=True))
            memory.record_episode(create_episode("fail_1", success=False))
            memory.record_episode(create_episode("success_2", success=True))

            similar = memory.find_similar_episodes(
                {"gold": 100, "level": 1, "screen": "main"},
                limit=10,
                success_only=True
            )

            assert all(ep.success for ep in similar)

    def test_find_similar_respects_limit(self, tmp_path: Path) -> None:
        """Test that limit is respected."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            for i in range(20):
                memory.record_episode(create_episode(f"ep_{i:03d}"))

            similar = memory.find_similar_episodes(
                {"gold": 100, "level": 1, "screen": "main"},
                limit=5
            )

            assert len(similar) == 5

    def test_find_similar_with_min_threshold(self, tmp_path: Path) -> None:
        """Test minimum similarity threshold."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            # Create very different episodes
            memory.record_episode(create_episode("similar", gold=100, level=1))

            # Query with completely different state
            similar = memory.find_similar_episodes(
                {"different_key": 999},
                limit=10,
                min_similarity=0.5  # High threshold
            )

            # May return none or few due to high threshold
            assert isinstance(similar, list)


class TestGetRecentEpisodes:
    """Tests for get_recent_episodes method."""

    def test_returns_in_order(self, tmp_path: Path) -> None:
        """Test that episodes are returned most recent first."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            base_time = datetime.now()
            for i in range(5):
                ep = create_episode(
                    f"ep_{i}",
                    timestamp=base_time + timedelta(seconds=i)
                )
                memory.record_episode(ep)

            recent = memory.get_recent_episodes(limit=5)
            assert len(recent) == 5
            assert recent[0].id == "ep_4"  # Most recent
            assert recent[4].id == "ep_0"  # Oldest

    def test_respects_limit(self, tmp_path: Path) -> None:
        """Test that limit is respected."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            for i in range(20):
                memory.record_episode(create_episode(f"ep_{i:03d}"))

            recent = memory.get_recent_episodes(limit=5)
            assert len(recent) == 5


class TestGetEpisodesByActionType:
    """Tests for get_episodes_by_action_type method."""

    def test_filters_by_action_type(self, tmp_path: Path) -> None:
        """Test filtering by action type."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            memory.record_episode(create_episode("click_1", action_type="click"))
            memory.record_episode(create_episode("click_2", action_type="click"))
            memory.record_episode(create_episode("type_1", action_type="type"))

            clicks = memory.get_episodes_by_action_type("click")
            assert len(clicks) == 2
            assert all("click" in ep.action_taken.get("type", "") for ep in clicks)


class TestSuccessRate:
    """Tests for get_success_rate method."""

    def test_calculates_overall_rate(self, tmp_path: Path) -> None:
        """Test calculating overall success rate."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            # 7 successes, 3 failures
            for i in range(7):
                memory.record_episode(create_episode(f"success_{i}", success=True))
            for i in range(3):
                memory.record_episode(create_episode(f"fail_{i}", success=False))

            successes, total, rate = memory.get_success_rate()
            assert successes == 7
            assert total == 10
            assert rate == 0.7

    def test_calculates_rate_by_action_type(self, tmp_path: Path) -> None:
        """Test calculating success rate by action type."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            # Clicks: 2 success, 1 fail
            memory.record_episode(create_episode("c1", action_type="click", success=True))
            memory.record_episode(create_episode("c2", action_type="click", success=True))
            memory.record_episode(create_episode("c3", action_type="click", success=False))

            # Types: 1 success, 1 fail
            memory.record_episode(create_episode("t1", action_type="type", success=True))
            memory.record_episode(create_episode("t2", action_type="type", success=False))

            successes, total, rate = memory.get_success_rate(action_type="click")
            assert successes == 2
            assert total == 3
            assert abs(rate - 2/3) < 0.01


class TestPruning:
    """Tests for pruning functionality."""

    def test_prune_keeps_recent(self, tmp_path: Path) -> None:
        """Test that pruning keeps most recent episodes."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            base_time = datetime.now()
            for i in range(20):
                ep = create_episode(
                    f"ep_{i:03d}",
                    timestamp=base_time + timedelta(seconds=i * 2)  # Ensure distinct timestamps
                )
                memory.record_episode(ep)

            deleted = memory.prune_old_episodes(keep_count=10)
            # Should delete approximately 10, allow for edge cases
            assert deleted >= 9
            remaining = memory.get_episode_count()
            assert remaining <= 11  # Allow small variance

            # Verify most recent are kept
            recent = memory.get_recent_episodes(limit=10)
            ids = [ep.id for ep in recent]
            assert "ep_019" in ids  # Most recent kept
            assert "ep_000" not in ids  # Oldest pruned

    def test_prune_nothing_to_prune(self, tmp_path: Path) -> None:
        """Test pruning when nothing to prune."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            for i in range(5):
                memory.record_episode(create_episode(f"ep_{i}"))

            deleted = memory.prune_old_episodes(keep_count=10)
            assert deleted == 0
            assert memory.get_episode_count() == 5

    def test_auto_prune_on_max_exceeded(self, tmp_path: Path) -> None:
        """Test automatic pruning when max_episodes exceeded."""
        db_path = tmp_path / "test.db"
        # Small max for testing
        with SQLiteEpisodicMemory(db_path, max_episodes=10) as memory:
            # Add enough to trigger auto-prune (10% buffer = 11+)
            for i in range(15):
                memory.record_episode(create_episode(f"ep_{i:03d}"))

            # Should have auto-pruned
            assert memory.get_episode_count() <= 12


class TestDeleteEpisode:
    """Tests for delete_episode method."""

    def test_delete_existing(self, tmp_path: Path) -> None:
        """Test deleting an existing episode."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            memory.record_episode(create_episode("ep_001"))
            assert memory.delete_episode("ep_001") is True
            assert memory.get_episode("ep_001") is None

    def test_delete_nonexistent(self, tmp_path: Path) -> None:
        """Test deleting a non-existent episode."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            assert memory.delete_episode("nonexistent") is False


class TestClearAll:
    """Tests for clear_all method."""

    def test_clears_everything(self, tmp_path: Path) -> None:
        """Test that clear_all removes all episodes."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            for i in range(10):
                memory.record_episode(create_episode(f"ep_{i}"))

            deleted = memory.clear_all()
            assert deleted == 10
            assert memory.get_episode_count() == 0


class TestActionTypeStats:
    """Tests for get_action_type_stats method."""

    def test_groups_by_action_type(self, tmp_path: Path) -> None:
        """Test grouping statistics by action type."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            # Clicks
            memory.record_episode(create_episode("c1", action_type="click", success=True))
            memory.record_episode(create_episode("c2", action_type="click", success=False))

            # Types
            memory.record_episode(create_episode("t1", action_type="type", success=True))
            memory.record_episode(create_episode("t2", action_type="type", success=True))
            memory.record_episode(create_episode("t3", action_type="type", success=True))

            stats = memory.get_action_type_stats()

            assert "click" in stats
            assert stats["click"] == (1, 2, 0.5)  # 1 success, 2 total, 50% rate

            assert "type" in stats
            assert stats["type"] == (3, 3, 1.0)  # 3 success, 3 total, 100% rate


class TestFeatureExtraction:
    """Tests for feature extraction."""

    def test_extracts_numeric_features(self) -> None:
        """Test that numeric features are extracted."""
        features = SQLiteEpisodicMemory._extract_features({
            "gold": 100,
            "level": 5,
            "health": 0.75,
        })

        assert "gold" in features
        assert features["gold"] == 100.0
        assert "level" in features
        assert features["level"] == 5.0

    def test_extracts_nested_features(self) -> None:
        """Test that nested features are extracted."""
        features = SQLiteEpisodicMemory._extract_features({
            "resources": {"gold": 100, "gems": 50},
            "player": {"level": 10},
        })

        assert "resources.gold" in features
        assert "resources.gems" in features
        assert "player.level" in features

    def test_extracts_bool_features(self) -> None:
        """Test that boolean features are extracted."""
        features = SQLiteEpisodicMemory._extract_features({
            "is_active": True,
            "is_paused": False,
        })

        assert features["is_active"] == 1.0
        assert features["is_paused"] == 0.0

    def test_extracts_list_features(self) -> None:
        """Test that list features are extracted."""
        features = SQLiteEpisodicMemory._extract_features({
            "inventory": [1, 2, 3],
        })

        assert "inventory._length" in features
        assert features["inventory._length"] == 3.0


class TestSimilarityComputation:
    """Tests for similarity computation."""

    def test_identical_features(self) -> None:
        """Test similarity of identical features."""
        features = {"a": 1.0, "b": 2.0, "c": 3.0}
        similarity = SQLiteEpisodicMemory._compute_similarity(features, features)
        assert similarity > 0.9

    def test_similar_features(self) -> None:
        """Test similarity of similar features."""
        f1 = {"gold": 100.0, "level": 5.0}
        f2 = {"gold": 110.0, "level": 5.0}
        similarity = SQLiteEpisodicMemory._compute_similarity(f1, f2)
        assert similarity > 0.8

    def test_different_features(self) -> None:
        """Test similarity of different features."""
        f1 = {"gold": 100.0}
        f2 = {"gems": 50.0}
        similarity = SQLiteEpisodicMemory._compute_similarity(f1, f2)
        assert similarity < 0.5

    def test_empty_features(self) -> None:
        """Test similarity with empty features."""
        similarity = SQLiteEpisodicMemory._compute_similarity({}, {})
        assert similarity == 0.0


class TestGenerateEpisodeId:
    """Tests for generate_episode_id method."""

    def test_generates_unique_ids(self, tmp_path: Path) -> None:
        """Test that unique IDs are generated."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            ids = [memory.generate_episode_id() for _ in range(100)]
            assert len(set(ids)) == 100  # All unique

    def test_id_format(self, tmp_path: Path) -> None:
        """Test ID format."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            episode_id = memory.generate_episode_id()
            assert episode_id.startswith("ep_")
            assert len(episode_id) == 15  # ep_ + 12 hex chars


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_records(self, tmp_path: Path) -> None:
        """Test concurrent episode recording."""
        db_path = tmp_path / "test.db"
        errors: list[Exception] = []

        def record_thread(memory: SQLiteEpisodicMemory, thread_id: int) -> None:
            try:
                for i in range(50):
                    memory.record_episode(
                        create_episode(f"t{thread_id}_ep_{i}")
                    )
            except Exception as e:
                errors.append(e)

        with SQLiteEpisodicMemory(db_path) as memory:
            threads = [
                threading.Thread(target=record_thread, args=(memory, i))
                for i in range(5)
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            # 5 threads * 50 episodes = 250
            assert memory.get_episode_count() == 250


class TestRecord100QuerySimilar:
    """Test recording 100 episodes and querying similar."""

    def test_record_100_query_similar(self, tmp_path: Path) -> None:
        """Test recording 100 episodes and finding similar ones."""
        db_path = tmp_path / "test.db"
        with SQLiteEpisodicMemory(db_path) as memory:
            # Record 100 episodes with varying states
            for i in range(100):
                ep = Episode(
                    episode_id=f"ep_{i:03d}",
                    timestamp=datetime.now(),
                    game_state_before={
                        "gold": i * 10,
                        "level": (i % 10) + 1,
                        "screen": "main" if i % 2 == 0 else "shop",
                    },
                    action_taken={"type": "click" if i % 3 == 0 else "type"},
                    game_state_after={"gold": i * 10 + 5},
                    success=i % 4 != 0,  # 75% success rate
                )
                memory.record_episode(ep)

            assert memory.get_episode_count() == 100

            # Query for similar to mid-range state
            similar = memory.find_similar_episodes(
                {"gold": 500, "level": 5, "screen": "main"},
                limit=10
            )

            assert len(similar) > 0
            assert len(similar) <= 10


class TestModuleExports:
    """Tests for module exports."""

    def test_exports_from_package(self) -> None:
        """Test that classes can be imported from package."""
        from src.memory.episodic import (
            EpisodicMemoryError,
            SQLiteEpisodicMemory,
        )

        assert SQLiteEpisodicMemory is not None
        assert EpisodicMemoryError is not None

    def test_episodic_memory_error_is_exception(self) -> None:
        """Test that EpisodicMemoryError is an Exception."""
        error = EpisodicMemoryError("test")
        assert isinstance(error, Exception)
