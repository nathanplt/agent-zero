"""Metrics collection for the agent loop.

This module provides metrics tracking for:
- Loop timing and rate
- Observation, decision, and action timing
- LLM usage and cost
- Action success/failure rates
- Error tracking
- Game progress

Example:
    >>> from src.core.metrics import MetricsCollector
    >>>
    >>> metrics = MetricsCollector()
    >>> metrics.record_loop_iteration(100.5)
    >>> metrics.record_observation(50.0)
    >>> metrics.record_action(success=True, duration_ms=25.0)
    >>>
    >>> stats = metrics.get_metrics()
    >>> print(f"Loop rate: {stats.loop_rate_hz:.2f} Hz")
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentMetrics(BaseModel):
    """Metrics collected by the main loop.

    This class provides a snapshot of all agent metrics at a point in time.
    It is immutable and can be safely shared/serialized.

    Attributes:
        loop_count: Total number of loop iterations.
        loop_rate_hz: Current loop rate in Hz.
        avg_observation_time_ms: Average observation time.
        avg_decision_time_ms: Average decision time.
        avg_action_time_ms: Average action time.
        llm_calls_total: Total LLM API calls.
        llm_calls_vision: LLM calls for vision.
        llm_calls_decision: LLM calls for decisions.
        llm_tokens_total: Total tokens used.
        llm_cost_usd: Estimated cost in USD.
        actions_total: Total actions attempted.
        actions_successful: Successful actions.
        actions_failed: Failed actions.
        actions_per_minute: Action rate.
        errors_total: Total errors encountered.
        errors_recovered: Errors that were recovered from.
        errors_by_type: Count of errors by type.
        current_goal: Current goal being pursued.
        goal_progress_percent: Progress toward current goal.
        resources_snapshot: Latest resource values.
        started_at: When the agent started.
        uptime_seconds: Total uptime.
    """

    # Timing
    loop_count: int = Field(default=0, ge=0)
    loop_rate_hz: float = Field(default=0.0, ge=0.0)
    avg_observation_time_ms: float = Field(default=0.0, ge=0.0)
    avg_decision_time_ms: float = Field(default=0.0, ge=0.0)
    avg_action_time_ms: float = Field(default=0.0, ge=0.0)

    # LLM usage
    llm_calls_total: int = Field(default=0, ge=0)
    llm_calls_vision: int = Field(default=0, ge=0)
    llm_calls_decision: int = Field(default=0, ge=0)
    llm_tokens_total: int = Field(default=0, ge=0)
    llm_cost_usd: float = Field(default=0.0, ge=0.0)

    # Actions
    actions_total: int = Field(default=0, ge=0)
    actions_successful: int = Field(default=0, ge=0)
    actions_failed: int = Field(default=0, ge=0)
    actions_per_minute: float = Field(default=0.0, ge=0.0)

    # Errors
    errors_total: int = Field(default=0, ge=0)
    errors_recovered: int = Field(default=0, ge=0)
    errors_by_type: dict[str, int] = Field(default_factory=dict)

    # Game progress
    current_goal: str = Field(default="")
    goal_progress_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    resources_snapshot: dict[str, float] = Field(default_factory=dict)

    # Uptime
    started_at: datetime | None = Field(default=None)
    uptime_seconds: float = Field(default=0.0, ge=0.0)

    model_config = {"frozen": True}

    @property
    def action_success_rate(self) -> float:
        """Calculate action success rate (0.0 to 1.0)."""
        if self.actions_total == 0:
            return 0.0
        return self.actions_successful / self.actions_total

    @property
    def error_recovery_rate(self) -> float:
        """Calculate error recovery rate (0.0 to 1.0)."""
        if self.errors_total == 0:
            return 1.0  # No errors = 100% recovery
        return self.errors_recovered / self.errors_total


@dataclass
class _TimingStats:
    """Internal helper for tracking timing statistics."""

    total_ms: float = 0.0
    count: int = 0

    def record(self, duration_ms: float) -> None:
        """Record a timing measurement."""
        self.total_ms += duration_ms
        self.count += 1

    @property
    def average_ms(self) -> float:
        """Get average duration in milliseconds."""
        if self.count == 0:
            return 0.0
        return self.total_ms / self.count


class MetricsCollector:
    """Collects metrics during agent loop execution.

    This class is thread-safe and designed for concurrent access
    from the main loop and monitoring systems.

    All methods are lightweight and suitable for calling
    every loop iteration without significant overhead.

    Example:
        >>> collector = MetricsCollector()
        >>> collector.start()
        >>>
        >>> # In loop
        >>> with collector.time_observation() as timer:
        ...     observation = pipeline.observe()
        >>> # Timer automatically records duration
        >>>
        >>> metrics = collector.get_metrics()
    """

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self._lock = threading.Lock()

        # Timing stats
        self._loop_timing = _TimingStats()
        self._observation_timing = _TimingStats()
        self._decision_timing = _TimingStats()
        self._action_timing = _TimingStats()

        # LLM tracking
        self._llm_calls_vision = 0
        self._llm_calls_decision = 0
        self._llm_tokens_total = 0
        self._llm_cost_usd = 0.0

        # Action tracking
        self._actions_successful = 0
        self._actions_failed = 0

        # Error tracking
        self._errors_recovered = 0
        self._errors_fatal = 0
        self._errors_by_type: dict[str, int] = {}

        # Game progress
        self._current_goal = ""
        self._goal_progress = 0.0
        self._resources: dict[str, float] = {}

        # Timing
        self._started_at: datetime | None = None
        self._last_loop_time: float = 0.0
        self._loop_times: list[float] = []  # Last 100 loop times for rate calculation

        logger.debug("MetricsCollector initialized")

    def start(self) -> None:
        """Mark the start of metrics collection."""
        with self._lock:
            self._started_at = datetime.now()
            self._last_loop_time = time.time()
            logger.debug("Metrics collection started")

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        with self._lock:
            self._loop_timing = _TimingStats()
            self._observation_timing = _TimingStats()
            self._decision_timing = _TimingStats()
            self._action_timing = _TimingStats()

            self._llm_calls_vision = 0
            self._llm_calls_decision = 0
            self._llm_tokens_total = 0
            self._llm_cost_usd = 0.0

            self._actions_successful = 0
            self._actions_failed = 0

            self._errors_recovered = 0
            self._errors_fatal = 0
            self._errors_by_type.clear()

            self._current_goal = ""
            self._goal_progress = 0.0
            self._resources.clear()

            self._started_at = None
            self._last_loop_time = 0.0
            self._loop_times.clear()

            logger.debug("Metrics reset")

    def record_loop_iteration(self, duration_ms: float) -> None:
        """Record a completed loop iteration.

        Args:
            duration_ms: Duration of the iteration in milliseconds.
        """
        with self._lock:
            self._loop_timing.record(duration_ms)

            # Track for rate calculation
            now = time.time()
            self._loop_times.append(now)
            # Keep only last 100 timestamps
            if len(self._loop_times) > 100:
                self._loop_times = self._loop_times[-100:]

    def record_observation(self, duration_ms: float, llm_used: bool = False) -> None:
        """Record an observation.

        Args:
            duration_ms: Duration of the observation in milliseconds.
            llm_used: Whether LLM was used for this observation.
        """
        with self._lock:
            self._observation_timing.record(duration_ms)
            if llm_used:
                self._llm_calls_vision += 1

    def record_decision(
        self,
        duration_ms: float,
        llm_used: bool = True,
        tokens: int = 0,
        cost: float = 0.0,
    ) -> None:
        """Record a decision.

        Args:
            duration_ms: Duration of decision-making in milliseconds.
            llm_used: Whether LLM was used for this decision.
            tokens: Number of tokens used.
            cost: Cost in USD.
        """
        with self._lock:
            self._decision_timing.record(duration_ms)
            if llm_used:
                self._llm_calls_decision += 1
            self._llm_tokens_total += tokens
            self._llm_cost_usd += cost

    def record_action(self, success: bool, duration_ms: float) -> None:
        """Record an action execution.

        Args:
            success: Whether the action succeeded.
            duration_ms: Duration of action execution in milliseconds.
        """
        with self._lock:
            self._action_timing.record(duration_ms)
            if success:
                self._actions_successful += 1
            else:
                self._actions_failed += 1

    def record_error(self, error_type: str, recovered: bool) -> None:
        """Record an error.

        Args:
            error_type: Type/class name of the error.
            recovered: Whether the agent recovered from this error.
        """
        with self._lock:
            if recovered:
                self._errors_recovered += 1
            else:
                self._errors_fatal += 1

            self._errors_by_type[error_type] = (
                self._errors_by_type.get(error_type, 0) + 1
            )

    def record_llm_call(
        self,
        component: str,
        tokens: int = 0,
        cost: float = 0.0,
    ) -> None:
        """Record an LLM API call.

        Args:
            component: Component making the call ("vision" or "decision").
            tokens: Number of tokens used.
            cost: Cost in USD.
        """
        with self._lock:
            if component == "vision":
                self._llm_calls_vision += 1
            elif component == "decision":
                self._llm_calls_decision += 1

            self._llm_tokens_total += tokens
            self._llm_cost_usd += cost

    def set_goal(self, goal: str, progress: float = 0.0) -> None:
        """Set the current goal and progress.

        Args:
            goal: Description of the current goal.
            progress: Progress percentage (0-100).
        """
        with self._lock:
            self._current_goal = goal
            self._goal_progress = max(0.0, min(100.0, progress))

    def update_resources(self, resources: dict[str, float]) -> None:
        """Update the resource snapshot.

        Args:
            resources: Dictionary of resource name to amount.
        """
        with self._lock:
            self._resources = dict(resources)

    @contextmanager
    def time_observation(self) -> Iterator[None]:
        """Context manager to time an observation.

        Example:
            >>> with metrics.time_observation():
            ...     observation = pipeline.observe()
        """
        start = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start) * 1000
            self.record_observation(duration_ms)

    @contextmanager
    def time_decision(self) -> Iterator[None]:
        """Context manager to time a decision.

        Example:
            >>> with metrics.time_decision():
            ...     decision = engine.decide(observation)
        """
        start = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start) * 1000
            self.record_decision(duration_ms)

    @contextmanager
    def time_action(self) -> Iterator[None]:
        """Context manager to time an action.

        Note: You still need to call record_action() with success status.

        Example:
            >>> with metrics.time_action():
            ...     result = executor.execute(action)
            >>> metrics.record_action(result.success, result.duration_ms)
        """
        start = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start) * 1000
            self._action_timing.record(duration_ms)

    def _calculate_loop_rate(self) -> float:
        """Calculate current loop rate in Hz."""
        if len(self._loop_times) < 2:
            return 0.0

        # Calculate rate from last N timestamps
        duration = self._loop_times[-1] - self._loop_times[0]
        if duration <= 0:
            return 0.0

        return (len(self._loop_times) - 1) / duration

    def _calculate_actions_per_minute(self) -> float:
        """Calculate actions per minute."""
        if self._started_at is None:
            return 0.0

        uptime = (datetime.now() - self._started_at).total_seconds()
        if uptime <= 0:
            return 0.0

        total_actions = self._actions_successful + self._actions_failed
        return (total_actions / uptime) * 60

    def get_metrics(self) -> AgentMetrics:
        """Get a snapshot of all current metrics.

        Returns:
            AgentMetrics object with all current values.
        """
        with self._lock:
            uptime = 0.0
            if self._started_at is not None:
                uptime = (datetime.now() - self._started_at).total_seconds()

            return AgentMetrics(
                # Timing
                loop_count=self._loop_timing.count,
                loop_rate_hz=self._calculate_loop_rate(),
                avg_observation_time_ms=self._observation_timing.average_ms,
                avg_decision_time_ms=self._decision_timing.average_ms,
                avg_action_time_ms=self._action_timing.average_ms,
                # LLM
                llm_calls_total=self._llm_calls_vision + self._llm_calls_decision,
                llm_calls_vision=self._llm_calls_vision,
                llm_calls_decision=self._llm_calls_decision,
                llm_tokens_total=self._llm_tokens_total,
                llm_cost_usd=self._llm_cost_usd,
                # Actions
                actions_total=self._actions_successful + self._actions_failed,
                actions_successful=self._actions_successful,
                actions_failed=self._actions_failed,
                actions_per_minute=self._calculate_actions_per_minute(),
                # Errors
                errors_total=self._errors_recovered + self._errors_fatal,
                errors_recovered=self._errors_recovered,
                errors_by_type=dict(self._errors_by_type),
                # Game progress
                current_goal=self._current_goal,
                goal_progress_percent=self._goal_progress,
                resources_snapshot=dict(self._resources),
                # Uptime
                started_at=self._started_at,
                uptime_seconds=uptime,
            )
