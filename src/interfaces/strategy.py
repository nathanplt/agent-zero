"""Strategy engine interface for game-specific intelligence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class GoalStatus(Enum):
    """Status of a goal."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


class Goal:
    """A goal in the goal hierarchy."""

    __slots__ = (
        "id",
        "name",
        "description",
        "status",
        "priority",
        "parent_id",
        "progress",
        "metadata",
    )

    def __init__(
        self,
        goal_id: str,
        name: str,
        description: str,
        status: GoalStatus = GoalStatus.PENDING,
        priority: int = 0,
        parent_id: str | None = None,
        progress: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a goal.

        Args:
            goal_id: Unique identifier for this goal.
            name: Short name for the goal.
            description: Detailed description of what to achieve.
            status: Current status of the goal.
            priority: Higher priority goals are pursued first.
            parent_id: ID of parent goal if this is a subgoal.
            progress: Completion progress (0.0 to 1.0).
            metadata: Additional goal-specific data.
        """
        self.id = goal_id
        self.name = name
        self.description = description
        self.status = status
        self.priority = priority
        self.parent_id = parent_id
        self.progress = progress
        self.metadata = metadata or {}


class Plan:
    """A multi-step plan to achieve a goal."""

    __slots__ = ("id", "goal_id", "steps", "current_step", "expected_value")

    def __init__(
        self,
        plan_id: str,
        goal_id: str,
        steps: list[dict[str, Any]],
        current_step: int = 0,
        expected_value: float = 0.0,
    ) -> None:
        """Initialize a plan.

        Args:
            plan_id: Unique identifier for this plan.
            goal_id: The goal this plan achieves.
            steps: List of action steps in order.
            current_step: Index of current step.
            expected_value: Expected value/utility of completing this plan.
        """
        self.id = plan_id
        self.goal_id = goal_id
        self.steps = steps
        self.current_step = current_step
        self.expected_value = expected_value

    @property
    def is_complete(self) -> bool:
        """Check if all steps have been executed."""
        return self.current_step >= len(self.steps)

    @property
    def next_step(self) -> dict[str, Any] | None:
        """Get the next step to execute, or None if complete."""
        if self.is_complete:
            return None
        return self.steps[self.current_step]


class ResourceAllocation:
    """A recommendation for resource allocation."""

    __slots__ = ("allocations", "reasoning", "expected_roi")

    def __init__(
        self,
        allocations: dict[str, float],
        reasoning: str,
        expected_roi: float,
    ) -> None:
        """Initialize a resource allocation.

        Args:
            allocations: Mapping of resource name to amount to allocate.
            reasoning: Explanation of why this allocation is recommended.
            expected_roi: Expected return on investment.
        """
        self.allocations = allocations
        self.reasoning = reasoning
        self.expected_roi = expected_roi


class UpgradePriority:
    """A prioritized upgrade recommendation."""

    __slots__ = ("upgrade_id", "name", "cost", "expected_benefit", "priority_score")

    def __init__(
        self,
        upgrade_id: str,
        name: str,
        cost: dict[str, float],
        expected_benefit: float,
        priority_score: float,
    ) -> None:
        """Initialize an upgrade priority.

        Args:
            upgrade_id: Unique identifier for the upgrade.
            name: Display name of the upgrade.
            cost: Resource costs for this upgrade.
            expected_benefit: Expected benefit from purchasing.
            priority_score: Overall priority score (higher = more important).
        """
        self.upgrade_id = upgrade_id
        self.name = name
        self.cost = cost
        self.expected_benefit = expected_benefit
        self.priority_score = priority_score


class StrategyEngine(ABC):
    """Abstract interface for the strategy engine.

    The strategy engine is responsible for:
    - Managing goal hierarchy
    - Planning multi-step actions
    - Optimizing resource allocation
    - Prioritizing upgrades
    - Game-specific strategy recommendations
    """

    # Goal Management

    @abstractmethod
    def set_main_goal(self, goal: Goal) -> None:
        """Set the main goal for the agent.

        Args:
            goal: The primary goal to achieve.
        """
        ...

    @abstractmethod
    def add_subgoal(self, goal: Goal) -> None:
        """Add a subgoal to the hierarchy.

        Args:
            goal: The subgoal to add. Must have parent_id set.
        """
        ...

    @abstractmethod
    def get_current_goal(self) -> Goal | None:
        """Get the current active goal.

        Returns:
            The highest priority in-progress goal, or None.
        """
        ...

    @abstractmethod
    def update_goal_progress(self, goal_id: str, progress: float) -> None:
        """Update progress on a goal.

        Args:
            goal_id: ID of the goal to update.
            progress: New progress value (0.0 to 1.0).
        """
        ...

    @abstractmethod
    def mark_goal_complete(self, goal_id: str) -> None:
        """Mark a goal as completed.

        Args:
            goal_id: ID of the goal to complete.
        """
        ...

    @abstractmethod
    def mark_goal_blocked(self, goal_id: str, reason: str) -> None:
        """Mark a goal as blocked.

        Args:
            goal_id: ID of the blocked goal.
            reason: Why the goal is blocked.
        """
        ...

    # Planning

    @abstractmethod
    def create_plan(self, goal: Goal, game_state: dict[str, Any]) -> Plan:
        """Create a plan to achieve a goal.

        Args:
            goal: The goal to plan for.
            game_state: Current game state.

        Returns:
            A plan with steps to achieve the goal.
        """
        ...

    @abstractmethod
    def evaluate_plan(self, plan: Plan, game_state: dict[str, Any]) -> float:
        """Evaluate a plan's expected value.

        Args:
            plan: The plan to evaluate.
            game_state: Current game state.

        Returns:
            Expected value of completing this plan.
        """
        ...

    @abstractmethod
    def replan_if_needed(self, plan: Plan, game_state: dict[str, Any]) -> Plan | None:
        """Check if replanning is needed and create new plan.

        Args:
            plan: The current plan.
            game_state: Current game state.

        Returns:
            New plan if replanning needed, None otherwise.
        """
        ...

    # Resource Optimization

    @abstractmethod
    def recommend_resource_allocation(
        self,
        resources: dict[str, float],
        game_state: dict[str, Any],
    ) -> ResourceAllocation:
        """Recommend how to allocate available resources.

        Args:
            resources: Currently available resources.
            game_state: Current game state.

        Returns:
            Recommended allocation with reasoning.
        """
        ...

    @abstractmethod
    def recommend_prestige_timing(self, game_state: dict[str, Any]) -> tuple[bool, str]:
        """Recommend whether to prestige now.

        Args:
            game_state: Current game state.

        Returns:
            Tuple of (should_prestige, reasoning).
        """
        ...

    # Upgrade Prioritization

    @abstractmethod
    def prioritize_upgrades(
        self,
        available_upgrades: list[dict[str, Any]],
        resources: dict[str, float],
    ) -> list[UpgradePriority]:
        """Prioritize available upgrades by expected value.

        Args:
            available_upgrades: List of available upgrade options.
            resources: Currently available resources.

        Returns:
            Prioritized list of upgrades.
        """
        ...


class StrategyError(Exception):
    """Error raised when strategy operations fail."""

    pass
