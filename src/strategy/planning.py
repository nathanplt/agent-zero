"""Planning system for multi-step lookahead and backtracking (Feature 6.3).

Generates plans from goals, evaluates expected value, and supports
replanning (backtrack) when a step fails.
"""

from __future__ import annotations

import uuid
from typing import Any

from src.interfaces.strategy import Goal, Plan


class PlanningSystem:
    """Multi-step planning with state prediction, evaluation, and backtracking.

    - create_plan: build a plan from a goal and current game state
    - evaluate_plan: expected value of completing the plan
    - replan_if_needed: return a new plan when failure is indicated (backtrack)
    """

    def __init__(self) -> None:
        """Initialize the planning system."""
        pass

    def create_plan(self, goal: Goal, game_state: dict[str, Any]) -> Plan:
        """Create a plan to achieve a goal.

        Generates a sequence of steps from the goal. Step count is derived
        from goal description or defaults to a small sequence.
        """
        goal_id = goal.id
        steps = self._steps_for_goal(goal, game_state)
        plan_id = f"plan_{uuid.uuid4().hex[:12]}"
        remaining = len(steps)
        expected_value = float(remaining * 10) if remaining else 0.0
        return Plan(
            plan_id=plan_id,
            goal_id=goal_id,
            steps=steps,
            current_step=0,
            expected_value=expected_value,
        )

    def _steps_for_goal(
        self, goal: Goal, game_state: dict[str, Any]  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        """Generate step dicts for the goal. Default: 3 generic steps."""
        return [
            {"action": "observe", "description": "Observe current state"},
            {"action": "act", "target": goal.name, "description": goal.description},
            {"action": "verify", "description": "Verify progress toward goal"},
        ]

    def evaluate_plan(self, plan: Plan, game_state: dict[str, Any]) -> float:
        """Evaluate the plan's expected value given current game state.

        Value decreases as steps are consumed; adds a small bonus from game state
        progress if present.
        """
        total = len(plan.steps)
        if total == 0:
            return 0.0
        remaining = total - plan.current_step
        if remaining <= 0:
            return 0.0
        base = (remaining / total) * 10.0
        progress = game_state.get("progress", 0.0) or game_state.get("goal_progress", 0.0)
        if isinstance(progress, (int, float)):
            base += float(progress) * 2.0
        return base

    def replan_if_needed(self, plan: Plan, game_state: dict[str, Any]) -> Plan | None:
        """If the plan or current step failed, return a new plan (backtrack); else None."""
        if game_state.get("last_step_failed") or game_state.get("plan_failed"):
            new_id = f"plan_{uuid.uuid4().hex[:12]}"
            return Plan(
                plan_id=new_id,
                goal_id=plan.goal_id,
                steps=plan.steps,
                current_step=0,
                expected_value=plan.expected_value,
            )
        return None
