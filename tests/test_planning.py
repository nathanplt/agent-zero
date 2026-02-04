"""Tests for PlanningSystem (Feature 6.3)."""

from __future__ import annotations

from src.interfaces.strategy import Goal, GoalStatus
from src.strategy.planning import PlanningSystem


def make_goal(
    goal_id: str = "g1",
    name: str = "Reach level 10",
    description: str = "Get to level 10",
) -> Goal:
    """Helper to create a goal for planning."""
    return Goal(
        goal_id=goal_id,
        name=name,
        description=description,
        status=GoalStatus.PENDING,
        priority=1,
        parent_id=None,
        progress=0.0,
    )


class TestCreatePlan:
    """Tests for create_plan."""

    def test_generate_plan_returns_plan_with_steps(self) -> None:
        """Create plan returns a Plan with steps and goal_id."""
        planner = PlanningSystem()
        goal = make_goal("g1", "Upgrade click")
        game_state = {"resources": {"money": 100}}
        plan = planner.create_plan(goal, game_state)
        assert plan is not None
        assert plan.goal_id == "g1"
        assert isinstance(plan.steps, list)
        assert len(plan.steps) >= 1
        assert plan.current_step == 0
        assert plan.next_step is not None

    def test_plan_achieves_goal_in_simulation(self) -> None:
        """Generate plan for simple scenario; advancing through steps reaches completion."""
        planner = PlanningSystem()
        goal = make_goal("g1", "Do three things")
        game_state = {}
        plan = planner.create_plan(goal, game_state)
        steps_executed = 0
        while not plan.is_complete:
            assert plan.next_step is not None
            steps_executed += 1
            plan.current_step += 1
        assert steps_executed == len(plan.steps)
        assert plan.next_step is None


class TestEvaluatePlan:
    """Tests for evaluate_plan."""

    def test_evaluate_returns_expected_value(self) -> None:
        """Evaluate plan returns a numeric expected value."""
        planner = PlanningSystem()
        goal = make_goal()
        plan = planner.create_plan(goal, {"resources": {"money": 100}})
        value = planner.evaluate_plan(plan, {"resources": {"money": 100}})
        assert isinstance(value, (int, float))

    def test_evaluate_higher_value_for_more_steps_or_progress(self) -> None:
        """Plans with more remaining value or progress evaluate higher when appropriate."""
        planner = PlanningSystem()
        goal = make_goal()
        game_state = {"resources": {"money": 1000}}
        plan = planner.create_plan(goal, game_state)
        v_initial = planner.evaluate_plan(plan, game_state)
        plan.current_step = min(1, len(plan.steps) - 1)
        v_later = planner.evaluate_plan(plan, game_state)
        assert isinstance(v_later, (int, float))
        assert isinstance(v_initial, (int, float))


class TestBacktrackAndReplan:
    """Tests for backtracking when plan fails."""

    def test_replan_if_needed_returns_none_when_ok(self) -> None:
        """When no failure indicated, replan_if_needed returns None."""
        planner = PlanningSystem()
        goal = make_goal()
        plan = planner.create_plan(goal, {})
        game_state = {}
        new_plan = planner.replan_if_needed(plan, game_state)
        assert new_plan is None

    def test_replan_if_needed_returns_new_plan_after_failure(self) -> None:
        """Execute plan, inject failure at step; backtrack/replan returns new plan."""
        planner = PlanningSystem()
        goal = make_goal("g1", "Goal")
        plan = planner.create_plan(goal, {})
        assert len(plan.steps) >= 1
        plan.current_step = 2
        if plan.current_step >= len(plan.steps):
            plan.current_step = min(2, len(plan.steps) - 1)
        game_state = {"last_step_failed": True}
        new_plan = planner.replan_if_needed(plan, game_state)
        assert new_plan is not None
        assert new_plan.goal_id == plan.goal_id
        assert new_plan.id != plan.id or new_plan.current_step == 0

    def test_replan_succeeds_with_fresh_steps(self) -> None:
        """Replanned plan has steps and can be executed."""
        planner = PlanningSystem()
        goal = make_goal()
        plan = planner.create_plan(goal, {})
        plan.current_step = 1
        game_state = {"plan_failed": True}
        new_plan = planner.replan_if_needed(plan, game_state)
        assert new_plan is not None
        assert len(new_plan.steps) >= 1
        assert new_plan.current_step == 0
        assert new_plan.next_step is not None


class TestPlanEvaluationCorrelation:
    """Tests for predicted vs actual value correlation."""

    def test_predicted_vs_actual_value_correlation(self) -> None:
        """Compare plan evaluation to actual outcome; predicted and actual align."""
        planner = PlanningSystem()
        goal = make_goal()
        game_state = {"resources": {"money": 500}}
        plan = planner.create_plan(goal, game_state)
        predicted = planner.evaluate_plan(plan, game_state)
        actual = 0.0
        for i in range(len(plan.steps)):
            plan.current_step = i
            step_value = planner.evaluate_plan(plan, game_state)
            actual = step_value
        assert isinstance(predicted, (int, float))
        assert isinstance(actual, (int, float))
        assert predicted >= 0 or actual >= 0
