"""Tests for the AgentMetrics and MetricsCollector classes."""

from __future__ import annotations

import threading
import time
from datetime import datetime

import pytest

from src.core.metrics import AgentMetrics, MetricsCollector, _TimingStats


class TestTimingStats:
    """Tests for _TimingStats internal class."""

    def test_initial_state(self) -> None:
        """Test initial state of timing stats."""
        stats = _TimingStats()
        assert stats.total_ms == 0.0
        assert stats.count == 0
        assert stats.average_ms == 0.0

    def test_record_single(self) -> None:
        """Test recording a single timing."""
        stats = _TimingStats()
        stats.record(100.0)
        assert stats.total_ms == 100.0
        assert stats.count == 1
        assert stats.average_ms == 100.0

    def test_record_multiple(self) -> None:
        """Test recording multiple timings."""
        stats = _TimingStats()
        stats.record(100.0)
        stats.record(200.0)
        stats.record(300.0)
        assert stats.total_ms == 600.0
        assert stats.count == 3
        assert stats.average_ms == 200.0

    def test_average_with_zero_count(self) -> None:
        """Test average when no recordings."""
        stats = _TimingStats()
        assert stats.average_ms == 0.0


class TestAgentMetrics:
    """Tests for AgentMetrics Pydantic model."""

    def test_default_values(self) -> None:
        """Test default metric values."""
        metrics = AgentMetrics()
        assert metrics.loop_count == 0
        assert metrics.loop_rate_hz == 0.0
        assert metrics.llm_calls_total == 0
        assert metrics.actions_total == 0
        assert metrics.errors_total == 0
        assert metrics.started_at is None
        assert metrics.uptime_seconds == 0.0

    def test_action_success_rate_no_actions(self) -> None:
        """Test action success rate with no actions."""
        metrics = AgentMetrics()
        assert metrics.action_success_rate == 0.0

    def test_action_success_rate_all_success(self) -> None:
        """Test action success rate with all successful."""
        metrics = AgentMetrics(
            actions_total=10,
            actions_successful=10,
            actions_failed=0,
        )
        assert metrics.action_success_rate == 1.0

    def test_action_success_rate_mixed(self) -> None:
        """Test action success rate with mixed results."""
        metrics = AgentMetrics(
            actions_total=10,
            actions_successful=7,
            actions_failed=3,
        )
        assert metrics.action_success_rate == 0.7

    def test_error_recovery_rate_no_errors(self) -> None:
        """Test error recovery rate with no errors."""
        metrics = AgentMetrics()
        assert metrics.error_recovery_rate == 1.0

    def test_error_recovery_rate_all_recovered(self) -> None:
        """Test error recovery rate with all recovered."""
        metrics = AgentMetrics(
            errors_total=5,
            errors_recovered=5,
        )
        assert metrics.error_recovery_rate == 1.0

    def test_error_recovery_rate_partial(self) -> None:
        """Test error recovery rate with partial recovery."""
        metrics = AgentMetrics(
            errors_total=10,
            errors_recovered=8,
        )
        assert metrics.error_recovery_rate == 0.8

    def test_metrics_immutable(self) -> None:
        """Test that metrics model is immutable."""
        from pydantic import ValidationError

        metrics = AgentMetrics()
        with pytest.raises(ValidationError):
            metrics.loop_count = 10  # type: ignore[misc]

    def test_metrics_serializable(self) -> None:
        """Test that metrics can be serialized."""
        metrics = AgentMetrics(
            loop_count=100,
            started_at=datetime.now(),
            errors_by_type={"ValueError": 5},
        )
        json_str = metrics.model_dump_json()
        assert "loop_count" in json_str
        assert "100" in json_str


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_initialization(self) -> None:
        """Test collector initialization."""
        collector = MetricsCollector()
        metrics = collector.get_metrics()
        assert metrics.loop_count == 0
        assert metrics.started_at is None

    def test_start_sets_time(self) -> None:
        """Test that start() sets started_at."""
        collector = MetricsCollector()
        collector.start()
        metrics = collector.get_metrics()
        assert metrics.started_at is not None

    def test_reset_clears_all(self) -> None:
        """Test that reset() clears all metrics."""
        collector = MetricsCollector()
        collector.start()
        collector.record_loop_iteration(100.0)
        collector.record_action(True, 50.0)
        collector.reset()
        metrics = collector.get_metrics()
        assert metrics.loop_count == 0
        assert metrics.actions_total == 0
        assert metrics.started_at is None

    def test_record_loop_iteration(self) -> None:
        """Test recording loop iterations."""
        collector = MetricsCollector()
        collector.start()
        collector.record_loop_iteration(100.0)
        collector.record_loop_iteration(150.0)
        metrics = collector.get_metrics()
        assert metrics.loop_count == 2

    def test_record_observation(self) -> None:
        """Test recording observations."""
        collector = MetricsCollector()
        collector.record_observation(50.0, llm_used=False)
        collector.record_observation(100.0, llm_used=True)
        metrics = collector.get_metrics()
        assert metrics.avg_observation_time_ms == 75.0
        assert metrics.llm_calls_vision == 1

    def test_record_decision(self) -> None:
        """Test recording decisions."""
        collector = MetricsCollector()
        collector.record_decision(200.0, llm_used=True, tokens=100, cost=0.01)
        metrics = collector.get_metrics()
        assert metrics.avg_decision_time_ms == 200.0
        assert metrics.llm_calls_decision == 1
        assert metrics.llm_tokens_total == 100
        assert metrics.llm_cost_usd == 0.01

    def test_record_action_success(self) -> None:
        """Test recording successful action."""
        collector = MetricsCollector()
        collector.record_action(success=True, duration_ms=25.0)
        metrics = collector.get_metrics()
        assert metrics.actions_total == 1
        assert metrics.actions_successful == 1
        assert metrics.actions_failed == 0

    def test_record_action_failure(self) -> None:
        """Test recording failed action."""
        collector = MetricsCollector()
        collector.record_action(success=False, duration_ms=10.0)
        metrics = collector.get_metrics()
        assert metrics.actions_total == 1
        assert metrics.actions_successful == 0
        assert metrics.actions_failed == 1

    def test_record_error_recovered(self) -> None:
        """Test recording recovered error."""
        collector = MetricsCollector()
        collector.record_error("ValueError", recovered=True)
        metrics = collector.get_metrics()
        assert metrics.errors_total == 1
        assert metrics.errors_recovered == 1
        assert metrics.errors_by_type == {"ValueError": 1}

    def test_record_error_fatal(self) -> None:
        """Test recording fatal error."""
        collector = MetricsCollector()
        collector.record_error("FatalError", recovered=False)
        metrics = collector.get_metrics()
        assert metrics.errors_total == 1
        assert metrics.errors_recovered == 0

    def test_record_error_accumulates_by_type(self) -> None:
        """Test that errors accumulate by type."""
        collector = MetricsCollector()
        collector.record_error("ValueError", recovered=True)
        collector.record_error("ValueError", recovered=True)
        collector.record_error("KeyError", recovered=True)
        metrics = collector.get_metrics()
        assert metrics.errors_by_type == {"ValueError": 2, "KeyError": 1}

    def test_record_llm_call_vision(self) -> None:
        """Test recording LLM call for vision."""
        collector = MetricsCollector()
        collector.record_llm_call("vision", tokens=500, cost=0.05)
        metrics = collector.get_metrics()
        assert metrics.llm_calls_vision == 1
        assert metrics.llm_calls_decision == 0
        assert metrics.llm_tokens_total == 500
        assert metrics.llm_cost_usd == 0.05

    def test_record_llm_call_decision(self) -> None:
        """Test recording LLM call for decision."""
        collector = MetricsCollector()
        collector.record_llm_call("decision", tokens=200, cost=0.02)
        metrics = collector.get_metrics()
        assert metrics.llm_calls_vision == 0
        assert metrics.llm_calls_decision == 1
        assert metrics.llm_tokens_total == 200

    def test_set_goal(self) -> None:
        """Test setting goal and progress."""
        collector = MetricsCollector()
        collector.set_goal("Complete tutorial", progress=50.0)
        metrics = collector.get_metrics()
        assert metrics.current_goal == "Complete tutorial"
        assert metrics.goal_progress_percent == 50.0

    def test_set_goal_clamps_progress(self) -> None:
        """Test that progress is clamped to 0-100."""
        collector = MetricsCollector()
        collector.set_goal("Test", progress=-10.0)
        metrics = collector.get_metrics()
        assert metrics.goal_progress_percent == 0.0

        collector.set_goal("Test", progress=150.0)
        metrics = collector.get_metrics()
        assert metrics.goal_progress_percent == 100.0

    def test_update_resources(self) -> None:
        """Test updating resource snapshot."""
        collector = MetricsCollector()
        collector.update_resources({"gold": 1000.0, "gems": 50.0})
        metrics = collector.get_metrics()
        assert metrics.resources_snapshot == {"gold": 1000.0, "gems": 50.0}

    def test_time_observation_context_manager(self) -> None:
        """Test time_observation context manager."""
        collector = MetricsCollector()
        with collector.time_observation():
            time.sleep(0.05)  # 50ms
        metrics = collector.get_metrics()
        assert metrics.avg_observation_time_ms >= 40.0  # Allow some variance

    def test_time_decision_context_manager(self) -> None:
        """Test time_decision context manager."""
        collector = MetricsCollector()
        with collector.time_decision():
            time.sleep(0.05)  # 50ms
        metrics = collector.get_metrics()
        assert metrics.avg_decision_time_ms >= 40.0

    def test_loop_rate_calculation(self) -> None:
        """Test loop rate calculation."""
        collector = MetricsCollector()
        collector.start()
        # Record 10 iterations over ~1 second
        for _ in range(10):
            collector.record_loop_iteration(100.0)
            time.sleep(0.1)
        metrics = collector.get_metrics()
        # Should be approximately 10 Hz (10 iterations per second)
        assert 5.0 <= metrics.loop_rate_hz <= 15.0

    def test_actions_per_minute_calculation(self) -> None:
        """Test actions per minute calculation."""
        collector = MetricsCollector()
        collector.start()
        # Record 6 actions
        for _ in range(6):
            collector.record_action(True, 10.0)
        # Wait a bit
        time.sleep(0.1)
        metrics = collector.get_metrics()
        # Should have positive actions per minute
        assert metrics.actions_per_minute > 0

    def test_uptime_calculation(self) -> None:
        """Test uptime calculation."""
        collector = MetricsCollector()
        collector.start()
        time.sleep(0.1)  # 100ms
        metrics = collector.get_metrics()
        assert metrics.uptime_seconds >= 0.08  # Allow some variance

    def test_thread_safety(self) -> None:
        """Test that collector is thread-safe."""
        collector = MetricsCollector()
        collector.start()

        errors: list[Exception] = []

        def record_thread() -> None:
            try:
                for _ in range(100):
                    collector.record_loop_iteration(10.0)
                    collector.record_action(True, 5.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        metrics = collector.get_metrics()
        assert metrics.loop_count == 500
        assert metrics.actions_total == 500

    def test_llm_calls_total(self) -> None:
        """Test that llm_calls_total sums vision and decision."""
        collector = MetricsCollector()
        collector.record_llm_call("vision")
        collector.record_llm_call("vision")
        collector.record_llm_call("decision")
        metrics = collector.get_metrics()
        assert metrics.llm_calls_total == 3
        assert metrics.llm_calls_vision == 2
        assert metrics.llm_calls_decision == 1


class TestModuleExports:
    """Tests for module exports."""

    def test_imports_from_package(self) -> None:
        """Test that classes can be imported from package."""
        from src.core.metrics import AgentMetrics, MetricsCollector

        assert AgentMetrics is not None
        assert MetricsCollector is not None
