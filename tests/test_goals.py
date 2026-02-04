"""Tests for GoalManager (Feature 6.1 Goal Hierarchy)."""

from __future__ import annotations

from src.interfaces.strategy import Goal, GoalStatus
from src.strategy.goals import GoalManager


def make_goal(
    goal_id: str,
    name: str = "Goal",
    description: str = "",
    status: GoalStatus = GoalStatus.PENDING,
    priority: int = 0,
    parent_id: str | None = None,
    progress: float = 0.0,
) -> Goal:
    """Helper to create test goals."""
    return Goal(
        goal_id=goal_id,
        name=name,
        description=description or name,
        status=status,
        priority=priority,
        parent_id=parent_id,
        progress=progress,
    )


class TestGoalManagerInit:
    """Tests for GoalManager initialization."""

    def test_init_empty(self) -> None:
        """Initial state has no main goal and no goals."""
        mgr = GoalManager()
        assert mgr.get_current_goal() is None
        assert mgr.get_main_goal() is None


class TestHierarchicalGoals:
    """Tests for defining hierarchical goals."""

    def test_set_main_goal(self) -> None:
        """Setting main goal stores it and makes it current."""
        mgr = GoalManager()
        main = make_goal("main", "Main", priority=1)
        mgr.set_main_goal(main)
        assert mgr.get_main_goal() is not None
        assert mgr.get_main_goal().id == "main"
        assert mgr.get_current_goal() is not None
        assert mgr.get_current_goal().id == "main"

    def test_add_subgoal(self) -> None:
        """Adding subgoal requires parent_id and stores under parent."""
        mgr = GoalManager()
        main = make_goal("main", "Main")
        sub = make_goal("sub1", "Sub1", parent_id="main")
        mgr.set_main_goal(main)
        mgr.add_subgoal(sub)
        assert mgr.get_goal("sub1") is not None
        assert mgr.get_goal("sub1").parent_id == "main"
        assert mgr.get_subgoals("main")[0].id == "sub1"

    def test_goal_tree_structure(self) -> None:
        """Goal tree has main and multiple subgoals."""
        mgr = GoalManager()
        mgr.set_main_goal(make_goal("root", "Root"))
        mgr.add_subgoal(make_goal("a", "A", parent_id="root"))
        mgr.add_subgoal(make_goal("b", "B", parent_id="root"))
        mgr.add_subgoal(make_goal("a1", "A1", parent_id="a"))
        assert len(mgr.get_subgoals("root")) == 2
        assert len(mgr.get_subgoals("a")) == 1
        assert mgr.get_goal("a1").parent_id == "a"


class TestProgressTracking:
    """Tests for progress tracking."""

    def test_update_goal_progress(self) -> None:
        """Progress updates and status moves to in_progress."""
        mgr = GoalManager()
        mgr.set_main_goal(make_goal("g1", "G1", progress=0.0))
        mgr.update_goal_progress("g1", 0.5)
        g = mgr.get_goal("g1")
        assert g is not None
        assert g.progress == 0.5
        assert g.status == GoalStatus.IN_PROGRESS

    def test_progress_tracking_accuracy(self) -> None:
        """Progress percentages match actual state."""
        mgr = GoalManager()
        mgr.set_main_goal(make_goal("g1", "G1"))
        mgr.update_goal_progress("g1", 0.25)
        assert mgr.get_goal("g1").progress == 0.25
        mgr.update_goal_progress("g1", 0.75)
        assert mgr.get_goal("g1").progress == 0.75


class TestGoalCompletion:
    """Tests for detecting goal completion."""

    def test_mark_goal_complete(self) -> None:
        """Marking complete sets status and progress to 1.0."""
        mgr = GoalManager()
        mgr.set_main_goal(make_goal("g1", "G1"))
        mgr.mark_goal_complete("g1")
        g = mgr.get_goal("g1")
        assert g is not None
        assert g.status == GoalStatus.COMPLETED
        assert g.progress == 1.0

    def test_subgoals_complete_parent_completes(self) -> None:
        """When all subgoals complete, parent goal completes."""
        mgr = GoalManager()
        mgr.set_main_goal(make_goal("root", "Root"))
        mgr.add_subgoal(make_goal("sub1", "Sub1", parent_id="root"))
        mgr.add_subgoal(make_goal("sub2", "Sub2", parent_id="root"))
        mgr.mark_goal_complete("sub1")
        mgr.mark_goal_complete("sub2")
        # Manager should auto-complete parent when all children done
        root = mgr.get_goal("root")
        assert root is not None
        assert root.status == GoalStatus.COMPLETED


class TestReplanWhenBlocked:
    """Tests for replanning when a goal is blocked."""

    def test_mark_goal_blocked(self) -> None:
        """Marking blocked sets status and stores reason."""
        mgr = GoalManager()
        mgr.set_main_goal(make_goal("g1", "G1"))
        mgr.mark_goal_blocked("g1", "No resources")
        g = mgr.get_goal("g1")
        assert g is not None
        assert g.status == GoalStatus.BLOCKED
        assert g.metadata.get("blocked_reason") == "No resources"

    def test_block_subgoal_alternative_path_or_blocked(self) -> None:
        """When one subgoal is blocked, current goal can be another or parent marked blocked."""
        mgr = GoalManager()
        mgr.set_main_goal(make_goal("root", "Root", priority=0))
        mgr.add_subgoal(make_goal("sub_a", "SubA", parent_id="root", priority=1))
        mgr.add_subgoal(make_goal("sub_b", "SubB", parent_id="root", priority=2))
        mgr.mark_goal_blocked("sub_a", "blocked")
        # Current goal should skip blocked and return next: sub_b (higher priority) or root
        current = mgr.get_current_goal()
        assert current is not None
        assert current.id != "sub_a"
        # Either we get sub_b (in progress/pending) or root
        assert current.id in ("sub_b", "root")


class TestGetCurrentGoal:
    """Tests for get_current_goal behavior."""

    def test_returns_highest_priority_in_progress(self) -> None:
        """Current goal is highest priority in-progress or first pending."""
        mgr = GoalManager()
        mgr.set_main_goal(make_goal("main", "Main", priority=0))
        mgr.add_subgoal(make_goal("low", "Low", parent_id="main", priority=1))
        mgr.add_subgoal(make_goal("high", "High", parent_id="main", priority=2))
        mgr.update_goal_progress("high", 0.5)
        current = mgr.get_current_goal()
        assert current is not None
        assert current.id == "high"

    def test_returns_none_when_all_complete(self) -> None:
        """Current goal is None when all goals are completed."""
        mgr = GoalManager()
        mgr.set_main_goal(make_goal("main", "Main"))
        mgr.mark_goal_complete("main")
        assert mgr.get_current_goal() is None

    def test_skips_blocked_goals(self) -> None:
        """Blocked goals are skipped when choosing current goal."""
        mgr = GoalManager()
        mgr.set_main_goal(make_goal("main", "Main"))
        mgr.add_subgoal(make_goal("sub", "Sub", parent_id="main"))
        mgr.mark_goal_blocked("sub", "reason")
        current = mgr.get_current_goal()
        assert current is not None
        assert current.id == "main"
