"""Main agent loop implementation.

This module provides the AgentLoop class that orchestrates:
- Observation pipeline
- Decision engine
- Action executor

The loop follows the pattern: Observe → Decide → Act → Repeat

Features:
- Configurable loop rate
- Pause/resume/stop support
- Error recovery for transient errors
- Graceful shutdown on signals
- Comprehensive metrics collection

Example:
    >>> from src.core.loop import AgentLoop, LoopConfig
    >>> from src.core.metrics import MetricsCollector
    >>>
    >>> # Create components
    >>> metrics = MetricsCollector()
    >>> loop = AgentLoop(
    ...     observation_pipeline=pipeline,
    ...     decision_engine=engine,
    ...     action_executor=executor,
    ...     metrics=metrics,
    ... )
    >>>
    >>> # Run the loop
    >>> loop.start()
    >>> # ... later ...
    >>> loop.stop()
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from src.core.metrics import MetricsCollector

if TYPE_CHECKING:
    from src.actions.executor import GameActionExecutor
    from src.core.decision import DecisionEngine
    from src.core.observation import Observation, ObservationPipeline
    from src.interfaces.actions import Action, ActionResult
    from src.models.decisions import Decision

logger = logging.getLogger(__name__)


class LoopState(StrEnum):
    """Possible states of the agent loop."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPING = "stopping"


class RecoverableError(Exception):
    """Error that the loop can recover from.

    When this error is raised, the loop will log it and continue
    to the next iteration.
    """

    pass


class FatalError(Exception):
    """Error that requires stopping the loop.

    When this error is raised, the loop will stop and
    the error will be propagated.
    """

    pass


@dataclass
class LoopConfig:
    """Configuration for the agent loop.

    Attributes:
        target_rate_hz: Target loop rate in Hz (iterations per second).
        min_iteration_ms: Minimum time per iteration (rate limiting).
        max_consecutive_errors: Max errors before stopping.
        error_recovery_delay_ms: Delay after recoverable error.
        enable_signal_handlers: Whether to install signal handlers.
    """

    target_rate_hz: float = 1.0  # 1 iteration per second
    min_iteration_ms: float = 100.0  # At least 100ms per iteration
    max_consecutive_errors: int = 5
    error_recovery_delay_ms: float = 1000.0
    enable_signal_handlers: bool = True


class AgentLoop:
    """Main agent loop that orchestrates observe-decide-act cycle.

    This class runs the main agent loop:
    1. Observe: Capture and process the game state
    2. Decide: Use LLM to decide the next action
    3. Act: Execute the decided action
    4. Repeat

    The loop includes:
    - Rate limiting to control iteration speed
    - Error recovery for transient failures
    - Pause/resume functionality
    - Graceful shutdown on signals
    - Comprehensive metrics collection

    Attributes:
        state: Current loop state.
        metrics: Metrics collector instance.

    Example:
        >>> loop = AgentLoop(pipeline, engine, executor, metrics)
        >>> loop.start()  # Runs in background thread
        >>> time.sleep(10)
        >>> loop.pause()
        >>> # ... later ...
        >>> loop.resume()
        >>> loop.stop()
    """

    def __init__(
        self,
        observation_pipeline: ObservationPipeline,
        decision_engine: DecisionEngine,
        action_executor: GameActionExecutor,
        metrics: MetricsCollector | None = None,
        config: LoopConfig | None = None,
    ) -> None:
        """Initialize the agent loop.

        Args:
            observation_pipeline: Pipeline for capturing observations.
            decision_engine: Engine for making decisions.
            action_executor: Executor for performing actions.
            metrics: Metrics collector. Creates new one if None.
            config: Loop configuration. Uses defaults if None.
        """
        self._observation_pipeline = observation_pipeline
        self._decision_engine = decision_engine
        self._action_executor = action_executor
        self._metrics = metrics or MetricsCollector()
        self._config = config or LoopConfig()

        self._state = LoopState.STOPPED
        self._state_lock = threading.Lock()
        self._loop_thread: threading.Thread | None = None
        self._consecutive_errors = 0
        self._last_observation: Observation | None = None
        self._last_decision: Decision | None = None

        # Callbacks
        self._on_iteration_complete: Callable[[int], None] | None = None
        self._on_error: Callable[[Exception], None] | None = None
        self._on_state_change: Callable[[LoopState], None] | None = None

        logger.debug(
            f"AgentLoop initialized: target_rate={self._config.target_rate_hz}Hz"
        )

    @property
    def state(self) -> LoopState:
        """Get the current loop state."""
        with self._state_lock:
            return self._state

    @property
    def metrics(self) -> MetricsCollector:
        """Get the metrics collector."""
        return self._metrics

    @property
    def last_observation(self) -> Observation | None:
        """Get the last observation (for debugging)."""
        return self._last_observation

    @property
    def last_decision(self) -> Decision | None:
        """Get the last decision (for debugging)."""
        return self._last_decision

    def set_callbacks(
        self,
        on_iteration_complete: Callable[[int], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        on_state_change: Callable[[LoopState], None] | None = None,
    ) -> None:
        """Set optional callbacks.

        Args:
            on_iteration_complete: Called after each iteration with loop count.
            on_error: Called when an error occurs.
            on_state_change: Called when state changes.
        """
        self._on_iteration_complete = on_iteration_complete
        self._on_error = on_error
        self._on_state_change = on_state_change

    def _set_state(self, new_state: LoopState) -> None:
        """Set the loop state (thread-safe)."""
        with self._state_lock:
            old_state = self._state
            self._state = new_state

        if old_state != new_state:
            logger.info(f"Loop state: {old_state.value} -> {new_state.value}")
            if self._on_state_change:
                try:
                    self._on_state_change(new_state)
                except Exception as e:
                    logger.warning(f"State change callback error: {e}")

    def start(self, blocking: bool = False) -> None:
        """Start the agent loop.

        Args:
            blocking: If True, runs in current thread (blocks).
                     If False, runs in background thread.

        Raises:
            RuntimeError: If loop is already running.
        """
        if self.state in (LoopState.RUNNING, LoopState.PAUSED):
            raise RuntimeError(f"Loop is already {self.state.value}")

        # Install signal handlers if enabled
        if self._config.enable_signal_handlers:
            self._install_signal_handlers()

        self._metrics.start()
        self._set_state(LoopState.RUNNING)

        if blocking:
            self._run_loop()
        else:
            self._loop_thread = threading.Thread(
                target=self._run_loop,
                name="AgentLoop",
                daemon=True,
            )
            self._loop_thread.start()
            logger.info("Agent loop started in background thread")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the agent loop gracefully.

        Args:
            timeout: Maximum time to wait for loop to stop.
        """
        if self.state == LoopState.STOPPED:
            return

        self._set_state(LoopState.STOPPING)

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=timeout)
            if self._loop_thread.is_alive():
                logger.warning("Loop thread did not stop within timeout")

        self._set_state(LoopState.STOPPED)
        logger.info("Agent loop stopped")

    def pause(self) -> None:
        """Pause the agent loop.

        The loop will complete its current iteration and then wait.
        """
        if self.state == LoopState.RUNNING:
            self._set_state(LoopState.PAUSED)
            logger.info("Agent loop paused")

    def resume(self) -> None:
        """Resume the agent loop from paused state."""
        if self.state == LoopState.PAUSED:
            self._set_state(LoopState.RUNNING)
            logger.info("Agent loop resumed")

    def _install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""
        def signal_handler(signum: int, _frame: object) -> None:
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name}, stopping loop...")
            self.stop()

        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            logger.debug("Signal handlers installed")
        except ValueError:
            # Can only set handlers in main thread
            logger.debug("Could not install signal handlers (not main thread)")

    def _run_loop(self) -> None:
        """Main loop execution (runs in thread)."""
        logger.info("Agent loop running")

        while self.state in (LoopState.RUNNING, LoopState.PAUSED):
            iteration_start = time.time()

            # Handle paused state
            if self.state == LoopState.PAUSED:
                time.sleep(0.1)  # Check state periodically
                continue

            try:
                # Run one iteration
                self._run_iteration()
                self._consecutive_errors = 0

            except RecoverableError as e:
                self._handle_recoverable_error(e)

            except FatalError as e:
                self._handle_fatal_error(e)
                break

            except Exception as e:
                # Treat unknown exceptions as recoverable up to a limit
                logger.exception(f"Unexpected error in loop: {e}")
                self._metrics.record_error(type(e).__name__, recovered=True)
                self._consecutive_errors += 1

                if self._consecutive_errors >= self._config.max_consecutive_errors:
                    logger.error(
                        f"Max consecutive errors ({self._config.max_consecutive_errors}) reached"
                    )
                    self._set_state(LoopState.ERROR)
                    break

                time.sleep(self._config.error_recovery_delay_ms / 1000)

            # Rate limiting
            self._apply_rate_limit(iteration_start)

            # Call iteration callback
            if self._on_iteration_complete:
                try:
                    self._on_iteration_complete(self._metrics.get_metrics().loop_count)
                except Exception as e:
                    logger.warning(f"Iteration callback error: {e}")

        logger.info("Agent loop exited")

    def _run_iteration(self) -> None:
        """Run a single loop iteration."""
        iteration_start = time.time()

        # Step 1: Observe
        logger.debug("Observing...")
        obs_start = time.time()
        observation = self._observation_pipeline.observe()
        obs_duration = (time.time() - obs_start) * 1000
        self._metrics.record_observation(obs_duration, llm_used=observation.llm_used)
        self._last_observation = observation
        logger.debug(
            f"Observation: screen={observation.game_state.current_screen.value}, "
            f"elements={len(observation.ui_elements)}, {obs_duration:.1f}ms"
        )

        # Update resource metrics
        resources = {
            name: res.amount
            for name, res in observation.game_state.resources.items()
        }
        self._metrics.update_resources(resources)

        # Step 2: Decide
        logger.debug("Deciding...")
        dec_start = time.time()
        decision = self._decision_engine.decide(observation)
        dec_duration = (time.time() - dec_start) * 1000
        self._metrics.record_decision(dec_duration, llm_used=True)
        self._last_decision = decision
        logger.debug(
            f"Decision: {decision.action.type} "
            f"(confidence={decision.confidence:.2f}), {dec_duration:.1f}ms"
        )

        # Step 3: Act
        logger.debug(f"Acting: {decision.action.type}...")
        act_start = time.time()
        action, result = self._execute_action(decision)
        act_duration = (time.time() - act_start) * 1000
        self._metrics.record_action(result.success, act_duration)

        # Record action result for decision engine context
        self._decision_engine.record_action_result(
            action,  # Pass the converted interface Action
            result.success,
            outcome=result.error or "success",
        )

        logger.debug(
            f"Action result: {'success' if result.success else 'failed'}, {act_duration:.1f}ms"
        )

        # Record loop iteration
        iteration_duration = (time.time() - iteration_start) * 1000
        self._metrics.record_loop_iteration(iteration_duration)

        logger.info(
            f"Iteration #{self._metrics.get_metrics().loop_count}: "
            f"total={iteration_duration:.0f}ms "
            f"(obs={obs_duration:.0f}, dec={dec_duration:.0f}, act={act_duration:.0f})"
        )

    def _execute_action(self, decision: Decision) -> tuple[Action, ActionResult]:
        """Execute the decided action.

        Args:
            decision: The decision containing the action.

        Returns:
            Tuple of (Action, ActionResult) for the executed action.
        """
        from src.interfaces.actions import Action, ActionType, Point

        # Convert model action to interface action
        model_action = decision.action
        target = None
        if model_action.target:
            target = Point(model_action.target.x, model_action.target.y)

        action = Action(
            action_type=ActionType(model_action.type),
            target=target,
            parameters=model_action.parameters,
            description=model_action.description,
        )

        result = self._action_executor.execute(action)
        return action, result

    def _handle_recoverable_error(self, error: RecoverableError) -> None:
        """Handle a recoverable error."""
        logger.warning(f"Recoverable error: {error}")
        self._metrics.record_error(type(error).__name__, recovered=True)
        self._consecutive_errors += 1

        if self._on_error:
            try:
                self._on_error(error)
            except Exception as e:
                logger.warning(f"Error callback error: {e}")

        # Wait before retrying
        time.sleep(self._config.error_recovery_delay_ms / 1000)

    def _handle_fatal_error(self, error: FatalError) -> None:
        """Handle a fatal error."""
        logger.error(f"Fatal error: {error}")
        self._metrics.record_error(type(error).__name__, recovered=False)
        self._set_state(LoopState.ERROR)

        if self._on_error:
            try:
                self._on_error(error)
            except Exception as e:
                logger.warning(f"Error callback error: {e}")

    def _apply_rate_limit(self, iteration_start: float) -> None:
        """Apply rate limiting to maintain target rate."""
        iteration_duration = time.time() - iteration_start
        target_duration = 1.0 / self._config.target_rate_hz
        min_duration = self._config.min_iteration_ms / 1000

        required_sleep = max(target_duration, min_duration) - iteration_duration

        if required_sleep > 0:
            time.sleep(required_sleep)

    def run_once(self) -> tuple[Observation, Decision, ActionResult]:
        """Run a single iteration manually.

        Useful for testing or step-by-step execution.

        Returns:
            Tuple of (observation, decision, action_result).

        Raises:
            RuntimeError: If loop is currently running.
        """
        if self.state == LoopState.RUNNING:
            raise RuntimeError("Cannot run_once while loop is running")

        # Run one iteration
        observation = self._observation_pipeline.observe()
        self._last_observation = observation
        self._metrics.record_observation(observation.duration_ms, observation.llm_used)

        decision = self._decision_engine.decide(observation)
        self._last_decision = decision

        _action, result = self._execute_action(decision)
        self._metrics.record_action(result.success, result.duration_ms)

        return observation, decision, result
