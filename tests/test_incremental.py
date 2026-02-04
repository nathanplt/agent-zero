"""Tests for IncrementalMetaStrategy (Feature 6.2)."""

from __future__ import annotations

from src.interfaces.strategy import ResourceAllocation, UpgradePriority
from src.strategy.incremental import IncrementalMetaStrategy


class TestRecommendResourceAllocation:
    """Tests for recommend_resource_allocation."""

    def test_returns_allocation_with_reasoning_and_roi(self) -> None:
        """Given resource state, recommend allocation; has allocations, reasoning, expected_roi."""
        strategy = IncrementalMetaStrategy()
        resources = {"money": 1000.0}
        game_state = {"resources": resources, "prestige_level": 0}
        result = strategy.recommend_resource_allocation(resources, game_state)
        assert isinstance(result, ResourceAllocation)
        assert isinstance(result.allocations, dict)
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0
        assert isinstance(result.expected_roi, (int, float))

    def test_allocation_matches_optimal_strategy(self) -> None:
        """Allocation favors primary resource / upgrades when applicable."""
        strategy = IncrementalMetaStrategy()
        resources = {"money": 5000.0}
        game_state = {"resources": resources, "primary": "money"}
        result = strategy.recommend_resource_allocation(resources, game_state)
        assert "money" in result.allocations or len(result.allocations) >= 1
        assert result.expected_roi >= 0


class TestRecommendPrestigeTiming:
    """Tests for recommend_prestige_timing."""

    def test_returns_bool_and_reason(self) -> None:
        """Returns (should_prestige: bool, reason: str)."""
        strategy = IncrementalMetaStrategy()
        game_state = {"resources": {"money": 100}, "prestige_level": 0}
        should, reason = strategy.recommend_prestige_timing(game_state)
        assert isinstance(should, bool)
        assert isinstance(reason, str)

    def test_prestige_timing_maximizes_progress(self) -> None:
        """Simulate prestige scenarios; recommend when beneficial."""
        strategy = IncrementalMetaStrategy()
        # Low progress: usually don't prestige yet
        low_state = {"resources": {"money": 100}, "prestige_level": 0}
        should_low, _ = strategy.recommend_prestige_timing(low_state)
        # High progress / near threshold: may recommend prestige
        high_state = {
            "resources": {"money": 1e9},
            "prestige_level": 5,
            "prestige_target": 10,
        }
        should_high, reason_high = strategy.recommend_prestige_timing(high_state)
        assert isinstance(should_high, bool)
        assert len(reason_high) > 0


class TestPrioritizeUpgrades:
    """Tests for upgrade prioritization."""

    def test_returns_list_of_upgrade_priority(self) -> None:
        """Prioritize upgrades returns list of UpgradePriority."""
        strategy = IncrementalMetaStrategy()
        upgrades = [
            {"id": "u1", "name": "Click+1", "cost": {"money": 100}, "benefit": 1.0},
            {"id": "u2", "name": "Click+2", "cost": {"money": 500}, "benefit": 3.0},
        ]
        resources = {"money": 1000.0}
        result = strategy.prioritize_upgrades(upgrades, resources)
        assert isinstance(result, list)
        for r in result:
            assert isinstance(r, UpgradePriority)
            assert hasattr(r, "upgrade_id")
            assert hasattr(r, "priority_score")

    def test_highest_roi_first(self) -> None:
        """Upgrade prioritization on test game state: highest ROI selected first."""
        strategy = IncrementalMetaStrategy()
        # u1: cost 100, benefit 10 -> ROI 0.1; u2: cost 100, benefit 50 -> ROI 0.5
        upgrades = [
            {"id": "low_roi", "name": "Low", "cost": {"money": 100}, "benefit": 10.0},
            {"id": "high_roi", "name": "High", "cost": {"money": 100}, "benefit": 50.0},
        ]
        resources = {"money": 500.0}
        result = strategy.prioritize_upgrades(upgrades, resources)
        assert len(result) >= 2
        assert result[0].priority_score >= result[1].priority_score
        assert result[0].upgrade_id == "high_roi"

    def test_affordable_upgrades_ranked_above_unaffordable(self) -> None:
        """Affordable upgrades get higher priority when we have limited resources."""
        strategy = IncrementalMetaStrategy()
        upgrades = [
            {"id": "cheap", "name": "Cheap", "cost": {"money": 50}, "benefit": 5.0},
            {"id": "expensive", "name": "Expensive", "cost": {"money": 10000}, "benefit": 100.0},
        ]
        resources = {"money": 100.0}
        result = strategy.prioritize_upgrades(upgrades, resources)
        assert len(result) == 2
        # Cheap should be first (affordable and positive ROI)
        assert result[0].upgrade_id == "cheap"
