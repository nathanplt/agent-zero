"""Tests for interface definitions.

These tests verify that:
1. All interfaces can be imported
2. Mock implementations can be created
3. Data classes work correctly
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.interfaces.actions import (
    Action,
    ActionError,
    ActionResult,
    ActionType,
    Point,
)
from src.interfaces.communication import (
    AgentState,
    CommunicationError,
    LogEntry,
    LogLevel,
)
from src.interfaces.environment import (
    EnvironmentError,
    EnvironmentHealth,
    EnvironmentStatus,
)
from src.interfaces.memory import Episode, MemoryError, Strategy
from src.interfaces.strategy import (
    Goal,
    GoalStatus,
    Plan,
    StrategyError,
)
from src.interfaces.vision import (
    UIElement,
    VisionError,
)


class TestPointDataClass:
    """Tests for Point class."""

    def test_create_point(self) -> None:
        """Test creating a point."""
        p = Point(100, 200)
        assert p.x == 100
        assert p.y == 200

    def test_point_repr(self) -> None:
        """Test point string representation."""
        p = Point(10, 20)
        assert repr(p) == "Point(10, 20)"

    def test_point_equality(self) -> None:
        """Test point equality comparison."""
        p1 = Point(10, 20)
        p2 = Point(10, 20)
        p3 = Point(10, 30)
        assert p1 == p2
        assert p1 != p3


class TestActionDataClasses:
    """Tests for Action-related data classes."""

    def test_create_action(self) -> None:
        """Test creating an action."""
        action = Action(
            action_type=ActionType.CLICK,
            target=Point(100, 200),
            parameters={"button": "left"},
            description="Click the button",
        )
        assert action.action_type == ActionType.CLICK
        assert action.target == Point(100, 200)
        assert action.parameters == {"button": "left"}
        assert action.description == "Click the button"

    def test_action_result(self) -> None:
        """Test creating an action result."""
        action = Action(ActionType.CLICK, Point(0, 0))
        result = ActionResult(
            success=True,
            action=action,
            duration_ms=50.0,
        )
        assert result.success is True
        assert result.action == action
        assert result.error is None
        assert result.duration_ms == 50.0


class TestUIElementDataClass:
    """Tests for UIElement class."""

    def test_create_ui_element(self) -> None:
        """Test creating a UI element."""
        elem = UIElement(
            element_type="button",
            x=100,
            y=200,
            width=50,
            height=30,
            confidence=0.95,
            label="Click Me",
        )
        assert elem.element_type == "button"
        assert elem.x == 100
        assert elem.y == 200
        assert elem.width == 50
        assert elem.height == 30
        assert elem.confidence == 0.95
        assert elem.label == "Click Me"

    def test_ui_element_center(self) -> None:
        """Test UI element center calculation."""
        elem = UIElement(
            element_type="button",
            x=100,
            y=200,
            width=50,
            height=30,
            confidence=0.9,
        )
        assert elem.center == (125, 215)


class TestGoalDataClass:
    """Tests for Goal class."""

    def test_create_goal(self) -> None:
        """Test creating a goal."""
        goal = Goal(
            goal_id="goal-1",
            name="Complete Tutorial",
            description="Finish the game tutorial",
            status=GoalStatus.IN_PROGRESS,
            priority=10,
            progress=0.5,
        )
        assert goal.id == "goal-1"
        assert goal.name == "Complete Tutorial"
        assert goal.status == GoalStatus.IN_PROGRESS
        assert goal.priority == 10
        assert goal.progress == 0.5


class TestPlanDataClass:
    """Tests for Plan class."""

    def test_create_plan(self) -> None:
        """Test creating a plan."""
        plan = Plan(
            plan_id="plan-1",
            goal_id="goal-1",
            steps=[{"action": "click", "target": "button1"}, {"action": "wait", "duration": 1}],
        )
        assert plan.id == "plan-1"
        assert len(plan.steps) == 2

    def test_plan_next_step(self) -> None:
        """Test getting next step from plan."""
        plan = Plan(
            plan_id="plan-1",
            goal_id="goal-1",
            steps=[{"action": "step1"}, {"action": "step2"}],
            current_step=0,
        )
        assert plan.next_step == {"action": "step1"}
        assert plan.is_complete is False

    def test_plan_complete(self) -> None:
        """Test plan completion detection."""
        plan = Plan(
            plan_id="plan-1",
            goal_id="goal-1",
            steps=[{"action": "step1"}],
            current_step=1,
        )
        assert plan.is_complete is True
        assert plan.next_step is None


class TestStrategyDataClass:
    """Tests for Strategy class."""

    def test_strategy_effectiveness(self) -> None:
        """Test strategy effectiveness calculation."""
        strategy = Strategy(
            name="rush_upgrade",
            description="Rush the first upgrade",
            success_count=8,
            failure_count=2,
        )
        assert strategy.effectiveness == 0.8

    def test_strategy_effectiveness_no_data(self) -> None:
        """Test strategy effectiveness with no data."""
        strategy = Strategy(
            name="new_strategy",
            description="A new untested strategy",
        )
        assert strategy.effectiveness == 0.5  # Default


class TestEpisodeDataClass:
    """Tests for Episode class."""

    def test_create_episode(self) -> None:
        """Test creating an episode."""
        episode = Episode(
            episode_id="ep-1",
            timestamp=datetime.now(),
            game_state_before={"gold": 100},
            action_taken={"type": "buy_upgrade", "name": "sword"},
            game_state_after={"gold": 50, "damage": 10},
            success=True,
        )
        assert episode.id == "ep-1"
        assert episode.success is True


class TestEnvironmentHealthDataClass:
    """Tests for EnvironmentHealth class."""

    def test_create_health(self) -> None:
        """Test creating environment health."""
        health = EnvironmentHealth(
            status=EnvironmentStatus.RUNNING,
            uptime_seconds=3600.0,
            cpu_percent=25.0,
            memory_mb=512.0,
            display_active=True,
            browser_active=True,
        )
        assert health.status == EnvironmentStatus.RUNNING
        assert health.uptime_seconds == 3600.0
        assert health.display_active is True


class TestLogEntryDataClass:
    """Tests for LogEntry class."""

    def test_create_log_entry(self) -> None:
        """Test creating a log entry."""
        entry = LogEntry(
            timestamp="2024-01-01T12:00:00Z",
            level=LogLevel.INFO,
            message="Agent started",
            data={"version": "1.0"},
        )
        assert entry.level == LogLevel.INFO
        assert entry.message == "Agent started"


class TestExceptionTypes:
    """Tests for exception types."""

    def test_vision_error(self) -> None:
        """Test VisionError can be raised."""
        with pytest.raises(VisionError, match="test error"):
            raise VisionError("test error")

    def test_action_error(self) -> None:
        """Test ActionError can be raised."""
        with pytest.raises(ActionError, match="test error"):
            raise ActionError("test error")

    def test_memory_error(self) -> None:
        """Test MemoryError can be raised."""
        with pytest.raises(MemoryError, match="test error"):
            raise MemoryError("test error")

    def test_strategy_error(self) -> None:
        """Test StrategyError can be raised."""
        with pytest.raises(StrategyError, match="test error"):
            raise StrategyError("test error")

    def test_environment_error(self) -> None:
        """Test EnvironmentError can be raised."""
        with pytest.raises(EnvironmentError, match="test error"):
            raise EnvironmentError("test error")

    def test_communication_error(self) -> None:
        """Test CommunicationError can be raised."""
        with pytest.raises(CommunicationError, match="test error"):
            raise CommunicationError("test error")


class TestInterfaceImports:
    """Tests that all interfaces can be imported from the package."""

    def test_import_from_package(self) -> None:
        """Test importing interfaces from the package."""
        import src.interfaces as interfaces

        # Verify they are abstract classes
        assert hasattr(interfaces.VisionSystem, "__abstractmethods__")
        assert hasattr(interfaces.ActionExecutor, "__abstractmethods__")
        assert hasattr(interfaces.MemoryStore, "__abstractmethods__")
        assert hasattr(interfaces.StrategyEngine, "__abstractmethods__")
        assert hasattr(interfaces.EnvironmentManager, "__abstractmethods__")
        assert hasattr(interfaces.CommunicationServer, "__abstractmethods__")


class TestEnumValues:
    """Tests for enum values."""

    def test_action_types(self) -> None:
        """Test ActionType enum values."""
        assert ActionType.CLICK.value == "click"
        assert ActionType.TYPE.value == "type"
        assert ActionType.SCROLL.value == "scroll"

    def test_goal_status(self) -> None:
        """Test GoalStatus enum values."""
        assert GoalStatus.PENDING.value == "pending"
        assert GoalStatus.COMPLETED.value == "completed"
        assert GoalStatus.BLOCKED.value == "blocked"

    def test_agent_state(self) -> None:
        """Test AgentState enum values."""
        assert AgentState.STOPPED.value == "stopped"
        assert AgentState.RUNNING.value == "running"
        assert AgentState.PAUSED.value == "paused"

    def test_log_level(self) -> None:
        """Test LogLevel enum values."""
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.DECISION.value == "decision"
