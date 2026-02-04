"""Goal hierarchy management for the strategy engine (Feature 6.1).

Provides hierarchical goal storage, progress tracking, completion detection,
and current-goal selection (skipping blocked/failed goals for replanning).
"""

from __future__ import annotations

from src.interfaces.strategy import Goal, GoalStatus


class GoalManager:
    """Manages a tree of goals with progress tracking and completion detection.

    - set_main_goal / add_subgoal: build the tree
    - update_goal_progress / mark_goal_complete / mark_goal_blocked: update state
    - get_current_goal: highest-priority in-progress or pending goal (skips blocked/failed)
    - When all subgoals of a parent are completed, the parent is auto-completed
    """

    def __init__(self) -> None:
        """Initialize an empty goal manager."""
        self._goals: dict[str, Goal] = {}
        self._main_goal_id: str | None = None

    def set_main_goal(self, goal: Goal) -> None:
        """Set the main goal for the agent."""
        self._goals[goal.id] = goal
        self._main_goal_id = goal.id

    def add_subgoal(self, goal: Goal) -> None:
        """Add a subgoal to the hierarchy (goal.parent_id must be set)."""
        if goal.parent_id is None:
            raise ValueError("Subgoal must have parent_id set")
        if goal.parent_id not in self._goals:
            raise ValueError(f"Parent goal not found: {goal.parent_id}")
        self._goals[goal.id] = goal

    def get_goal(self, goal_id: str) -> Goal | None:
        """Get a goal by ID."""
        return self._goals.get(goal_id)

    def get_main_goal(self) -> Goal | None:
        """Get the main goal, or None if not set."""
        if self._main_goal_id is None:
            return None
        return self._goals.get(self._main_goal_id)

    def get_subgoals(self, parent_id: str) -> list[Goal]:
        """Get direct subgoals of a parent, ordered by priority (desc) then id."""
        children = [g for g in self._goals.values() if g.parent_id == parent_id]
        children.sort(key=lambda g: (-g.priority, g.id))
        return children

    def get_current_goal(self) -> Goal | None:
        """Get the current active goal: highest priority in-progress or pending, skipping blocked/failed."""
        active_statuses = (GoalStatus.IN_PROGRESS, GoalStatus.PENDING)
        candidates = [g for g in self._goals.values() if g.status in active_statuses]
        if not candidates:
            return None
        candidates.sort(key=lambda g: (-g.priority, g.id))
        return candidates[0]

    def update_goal_progress(self, goal_id: str, progress: float) -> None:
        """Update progress on a goal; sets status to IN_PROGRESS if was PENDING."""
        g = self._goals.get(goal_id)
        if g is None:
            raise KeyError(f"Goal not found: {goal_id}")
        progress = max(0.0, min(1.0, progress))
        new_status = GoalStatus.IN_PROGRESS if g.status == GoalStatus.PENDING else g.status
        self._goals[goal_id] = Goal(
            goal_id=g.id,
            name=g.name,
            description=g.description,
            status=new_status,
            priority=g.priority,
            parent_id=g.parent_id,
            progress=progress,
            metadata=dict(g.metadata),
        )

    def mark_goal_complete(self, goal_id: str) -> None:
        """Mark a goal as completed (progress 1.0). If parent has all subgoals complete, complete parent."""
        g = self._goals.get(goal_id)
        if g is None:
            raise KeyError(f"Goal not found: {goal_id}")
        self._goals[goal_id] = Goal(
            goal_id=g.id,
            name=g.name,
            description=g.description,
            status=GoalStatus.COMPLETED,
            priority=g.priority,
            parent_id=g.parent_id,
            progress=1.0,
            metadata=dict(g.metadata),
        )
        parent_id = g.parent_id
        if parent_id is not None:
            self._maybe_complete_parent(parent_id)

    def _maybe_complete_parent(self, parent_id: str) -> None:
        """If all subgoals of parent are completed, mark parent completed and recurse."""
        parent = self._goals.get(parent_id)
        if parent is None or parent.status == GoalStatus.COMPLETED:
            return
        children = self.get_subgoals(parent_id)
        if not children:
            return
        if not all(c.status == GoalStatus.COMPLETED for c in children):
            return
        self._goals[parent_id] = Goal(
            goal_id=parent.id,
            name=parent.name,
            description=parent.description,
            status=GoalStatus.COMPLETED,
            priority=parent.priority,
            parent_id=parent.parent_id,
            progress=1.0,
            metadata=dict(parent.metadata),
        )
        if parent.parent_id is not None:
            self._maybe_complete_parent(parent.parent_id)

    def mark_goal_blocked(self, goal_id: str, reason: str) -> None:
        """Mark a goal as blocked and store the reason in metadata."""
        g = self._goals.get(goal_id)
        if g is None:
            raise KeyError(f"Goal not found: {goal_id}")
        meta = dict(g.metadata)
        meta["blocked_reason"] = reason
        self._goals[goal_id] = Goal(
            goal_id=g.id,
            name=g.name,
            description=g.description,
            status=GoalStatus.BLOCKED,
            priority=g.priority,
            parent_id=g.parent_id,
            progress=g.progress,
            metadata=meta,
        )
