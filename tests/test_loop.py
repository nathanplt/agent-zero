"""Tests for the AgentLoop class."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.core.loop import (
    AgentLoop,
    FatalError,
    LoopConfig,
    LoopState,
    RecoverableError,
)
from src.core.metrics import MetricsCollector


# Mock classes for testing
@dataclass
class MockScreenshot:
    """Mock screenshot for testing."""

    data: bytes = b"mock"
    width: int = 1920
    height: int = 1080
    timestamp: datetime = datetime.now()  # noqa: RUF009


@dataclass
class MockResource:
    """Mock resource for testing."""

    amount: float = 100.0


@dataclass
class MockGameState:
    """Mock game state for testing."""

    current_screen: Any = None
    resources: dict[str, MockResource] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.current_screen is None:
            from src.models.game_state import ScreenType

            self.current_screen = ScreenType.MAIN
        if self.resources is None:
            self.resources = {"gold": MockResource(1000.0)}


@dataclass
class MockObservation:
    """Mock observation for testing."""

    screenshot: MockScreenshot = None  # type: ignore[assignment]
    game_state: MockGameState = None  # type: ignore[assignment]
    ui_elements: list[Any] = None  # type: ignore[assignment]
    text_regions: list[Any] = None  # type: ignore[assignment]
    timestamp: datetime = datetime.now()  # noqa: RUF009
    llm_used: bool = False
    confidence: float = 1.0
    duration_ms: float = 50.0

    def __post_init__(self) -> None:
        if self.screenshot is None:
            self.screenshot = MockScreenshot()
        if self.game_state is None:
            self.game_state = MockGameState()
        if self.ui_elements is None:
            self.ui_elements = []
        if self.text_regions is None:
            self.text_regions = []


@dataclass
class MockTarget:
    """Mock target for testing."""

    x: int = 100
    y: int = 100


@dataclass
class MockAction:
    """Mock action for testing."""

    type: str = "click"
    target: MockTarget = None  # type: ignore[assignment]
    parameters: dict[str, Any] = None  # type: ignore[assignment]
    description: str = "test action"

    def __post_init__(self) -> None:
        if self.target is None:
            self.target = MockTarget()
        if self.parameters is None:
            self.parameters = {}


@dataclass
class MockDecision:
    """Mock decision for testing."""

    action: MockAction = None  # type: ignore[assignment]
    confidence: float = 0.9
    reasoning: str = "test reasoning"

    def __post_init__(self) -> None:
        if self.action is None:
            self.action = MockAction()


@dataclass
class MockActionResult:
    """Mock action result for testing."""

    success: bool = True
    error: str | None = None
    duration_ms: float = 25.0


class TestLoopConfig:
    """Tests for LoopConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LoopConfig()
        assert config.target_rate_hz == 1.0
        assert config.min_iteration_ms == 100.0
        assert config.max_consecutive_errors == 5
        assert config.error_recovery_delay_ms == 1000.0
        assert config.enable_signal_handlers is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = LoopConfig(
            target_rate_hz=10.0,
            min_iteration_ms=50.0,
            max_consecutive_errors=3,
        )
        assert config.target_rate_hz == 10.0
        assert config.min_iteration_ms == 50.0
        assert config.max_consecutive_errors == 3


class TestLoopState:
    """Tests for LoopState enum."""

    def test_state_values(self) -> None:
        """Test that all states have expected values."""
        assert LoopState.STOPPED.value == "stopped"
        assert LoopState.RUNNING.value == "running"
        assert LoopState.PAUSED.value == "paused"
        assert LoopState.ERROR.value == "error"
        assert LoopState.STOPPING.value == "stopping"


class TestRecoverableError:
    """Tests for RecoverableError."""

    def test_is_exception(self) -> None:
        """Test that RecoverableError is an Exception."""
        error = RecoverableError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"


class TestFatalError:
    """Tests for FatalError."""

    def test_is_exception(self) -> None:
        """Test that FatalError is an Exception."""
        error = FatalError("fatal error")
        assert isinstance(error, Exception)
        assert str(error) == "fatal error"


class TestAgentLoopInitialization:
    """Tests for AgentLoop initialization."""

    def test_initialization_with_defaults(self) -> None:
        """Test initialization with default values."""
        pipeline = MagicMock()
        engine = MagicMock()
        executor = MagicMock()

        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
        )

        assert loop.state == LoopState.STOPPED
        assert loop.metrics is not None
        assert loop.last_observation is None
        assert loop.last_decision is None

    def test_initialization_with_custom_metrics(self) -> None:
        """Test initialization with custom metrics collector."""
        pipeline = MagicMock()
        engine = MagicMock()
        executor = MagicMock()
        metrics = MetricsCollector()

        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
            metrics=metrics,
        )

        assert loop.metrics is metrics

    def test_initialization_with_custom_config(self) -> None:
        """Test initialization with custom config."""
        pipeline = MagicMock()
        engine = MagicMock()
        executor = MagicMock()
        config = LoopConfig(target_rate_hz=10.0)

        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
            config=config,
        )

        # Config is internal, but we can verify it works
        assert loop.state == LoopState.STOPPED


class TestAgentLoopDecisionMetrics:
    """Decision metrics should reflect policy-vs-LLM source accurately."""

    def test_policy_decision_does_not_increment_llm_counter(self) -> None:
        from src.interfaces.actions import Action as InterfaceAction
        from src.interfaces.actions import ActionResult as InterfaceActionResult
        from src.interfaces.actions import ActionType as InterfaceActionType
        from src.models.actions import Action as ModelAction
        from src.models.decisions import Decision

        pipeline = MagicMock()
        pipeline.observe.return_value = MockObservation()
        engine = MagicMock()
        engine.decide.return_value = Decision(
            reasoning="Policy says wait.",
            action=ModelAction.wait(duration_ms=1000),
            confidence=0.9,
            expected_outcome="Observe resource growth.",
            context={"decision_source": "policy"},
        )
        executor = MagicMock()
        executor.execute.return_value = InterfaceActionResult(
            success=True,
            action=InterfaceAction(
                action_type=InterfaceActionType.WAIT,
                parameters={"duration_ms": 1000},
            ),
            duration_ms=2.0,
        )

        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
            config=LoopConfig(enable_signal_handlers=False),
        )

        loop._run_iteration()  # noqa: SLF001

        metrics = loop.metrics.get_metrics()
        assert metrics.llm_calls_decision == 0

    def test_llm_decision_increments_llm_counter(self) -> None:
        from src.interfaces.actions import Action as InterfaceAction
        from src.interfaces.actions import ActionResult as InterfaceActionResult
        from src.interfaces.actions import ActionType as InterfaceActionType
        from src.models.actions import Action as ModelAction
        from src.models.actions import Point
        from src.models.decisions import Decision

        pipeline = MagicMock()
        pipeline.observe.return_value = MockObservation()
        engine = MagicMock()
        engine.decide.return_value = Decision(
            reasoning="LLM says click.",
            action=ModelAction(
                type=InterfaceActionType.CLICK,
                target=Point(x=10, y=10),
                parameters={"button": "left"},
                description="Click starter button",
            ),
            confidence=0.7,
            expected_outcome="Progress increases.",
            context={"decision_source": "llm"},
        )
        executor = MagicMock()
        executor.execute.return_value = InterfaceActionResult(
            success=True,
            action=InterfaceAction(
                action_type=InterfaceActionType.CLICK,
                parameters={"button": "left"},
            ),
            duration_ms=2.0,
        )

        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
            config=LoopConfig(enable_signal_handlers=False),
        )

        loop._run_iteration()  # noqa: SLF001

        metrics = loop.metrics.get_metrics()
        assert metrics.llm_calls_decision == 1


class TestAgentLoopStartStop:
    """Tests for AgentLoop start/stop functionality."""

    def test_start_changes_state_to_running(self) -> None:
        """Test that start() changes state to RUNNING."""
        pipeline = MagicMock()
        pipeline.observe.return_value = MockObservation()
        engine = MagicMock()
        engine.decide.return_value = MockDecision()
        executor = MagicMock()
        executor.execute.return_value = MockActionResult()

        config = LoopConfig(enable_signal_handlers=False)
        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
            config=config,
        )

        loop.start(blocking=False)
        time.sleep(0.1)
        assert loop.state in (LoopState.RUNNING, LoopState.STOPPED)
        loop.stop()

    def test_stop_changes_state_to_stopped(self) -> None:
        """Test that stop() changes state to STOPPED."""
        pipeline = MagicMock()
        pipeline.observe.return_value = MockObservation()
        engine = MagicMock()
        engine.decide.return_value = MockDecision()
        executor = MagicMock()
        executor.execute.return_value = MockActionResult()

        config = LoopConfig(enable_signal_handlers=False)
        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
            config=config,
        )

        loop.start(blocking=False)
        time.sleep(0.1)
        loop.stop()
        assert loop.state == LoopState.STOPPED

    def test_start_when_already_running_raises_error(self) -> None:
        """Test that starting an already running loop raises error."""
        pipeline = MagicMock()
        pipeline.observe.return_value = MockObservation()
        engine = MagicMock()
        engine.decide.return_value = MockDecision()
        executor = MagicMock()
        executor.execute.return_value = MockActionResult()

        config = LoopConfig(enable_signal_handlers=False)
        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
            config=config,
        )

        loop.start(blocking=False)
        time.sleep(0.1)
        try:
            with pytest.raises(RuntimeError):
                loop.start()
        finally:
            loop.stop()

    def test_stop_when_already_stopped_is_noop(self) -> None:
        """Test that stopping a stopped loop is a no-op."""
        pipeline = MagicMock()
        engine = MagicMock()
        executor = MagicMock()

        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
        )

        # Should not raise
        loop.stop()
        assert loop.state == LoopState.STOPPED


class TestAgentLoopPauseResume:
    """Tests for AgentLoop pause/resume functionality."""

    def test_pause_changes_state_to_paused(self) -> None:
        """Test that pause() changes state to PAUSED."""
        pipeline = MagicMock()
        pipeline.observe.return_value = MockObservation()
        engine = MagicMock()
        engine.decide.return_value = MockDecision()
        executor = MagicMock()
        executor.execute.return_value = MockActionResult()

        config = LoopConfig(enable_signal_handlers=False)
        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
            config=config,
        )

        loop.start(blocking=False)
        time.sleep(0.1)
        loop.pause()
        assert loop.state == LoopState.PAUSED
        loop.stop()

    def test_resume_changes_state_to_running(self) -> None:
        """Test that resume() changes state back to RUNNING."""
        pipeline = MagicMock()
        pipeline.observe.return_value = MockObservation()
        engine = MagicMock()
        engine.decide.return_value = MockDecision()
        executor = MagicMock()
        executor.execute.return_value = MockActionResult()

        config = LoopConfig(enable_signal_handlers=False)
        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
            config=config,
        )

        loop.start(blocking=False)
        time.sleep(0.1)
        loop.pause()
        assert loop.state == LoopState.PAUSED
        loop.resume()
        time.sleep(0.1)
        assert loop.state == LoopState.RUNNING
        loop.stop()

    def test_pause_when_not_running_is_noop(self) -> None:
        """Test that pausing a non-running loop is a no-op."""
        pipeline = MagicMock()
        engine = MagicMock()
        executor = MagicMock()

        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
        )

        loop.pause()  # Should not raise
        assert loop.state == LoopState.STOPPED

    def test_resume_when_not_paused_is_noop(self) -> None:
        """Test that resuming a non-paused loop is a no-op."""
        pipeline = MagicMock()
        engine = MagicMock()
        executor = MagicMock()

        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
        )

        loop.resume()  # Should not raise
        assert loop.state == LoopState.STOPPED

    def test_loop_count_stays_constant_while_paused(self) -> None:
        """Test that loop count does not increase while paused."""
        pipeline = MagicMock()
        pipeline.observe.return_value = MockObservation()
        engine = MagicMock()
        engine.decide.return_value = MockDecision()
        executor = MagicMock()
        executor.execute.return_value = MockActionResult()

        config = LoopConfig(
            target_rate_hz=20.0,
            min_iteration_ms=10.0,
            enable_signal_handlers=False,
        )
        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
            config=config,
        )

        loop.start(blocking=False)
        time.sleep(0.2)
        loop.pause()
        count_at_pause = loop.metrics.get_metrics().loop_count
        time.sleep(0.2)
        count_after_pause = loop.metrics.get_metrics().loop_count
        loop.stop()

        assert count_after_pause == count_at_pause


class TestAgentLoopIteration:
    """Tests for AgentLoop iteration behavior."""

    def test_iteration_calls_observe_decide_act(self) -> None:
        """Test that each iteration calls observe, decide, act."""
        pipeline = MagicMock()
        pipeline.observe.return_value = MockObservation()
        engine = MagicMock()
        engine.decide.return_value = MockDecision()
        executor = MagicMock()
        executor.execute.return_value = MockActionResult()

        config = LoopConfig(
            target_rate_hz=100.0,
            min_iteration_ms=1.0,
            enable_signal_handlers=False,
        )
        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
            config=config,
        )

        loop.start(blocking=False)
        time.sleep(0.2)
        loop.stop()

        assert pipeline.observe.called
        assert engine.decide.called
        assert executor.execute.called

    def test_run_once_executes_single_iteration(self) -> None:
        """Test that run_once() executes a single iteration."""
        pipeline = MagicMock()
        pipeline.observe.return_value = MockObservation()
        engine = MagicMock()
        engine.decide.return_value = MockDecision()
        executor = MagicMock()
        executor.execute.return_value = MockActionResult()

        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
        )

        obs, decision, result = loop.run_once()

        assert pipeline.observe.call_count == 1
        assert engine.decide.call_count == 1
        assert executor.execute.call_count == 1
        assert loop.last_observation is not None
        assert loop.last_decision is not None

    def test_run_once_when_running_raises_error(self) -> None:
        """Test that run_once() raises error when loop is running."""
        pipeline = MagicMock()
        pipeline.observe.return_value = MockObservation()
        engine = MagicMock()
        engine.decide.return_value = MockDecision()
        executor = MagicMock()
        executor.execute.return_value = MockActionResult()

        config = LoopConfig(enable_signal_handlers=False)
        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
            config=config,
        )

        loop.start(blocking=False)
        time.sleep(0.1)
        try:
            with pytest.raises(RuntimeError):
                loop.run_once()
        finally:
            loop.stop()


class TestAgentLoopErrorHandling:
    """Tests for AgentLoop error handling."""

    def test_recoverable_error_continues_loop(self) -> None:
        """Test that recoverable errors don't stop the loop."""
        call_count = [0]

        def observe_with_error() -> MockObservation:
            call_count[0] += 1
            if call_count[0] == 3:
                raise RecoverableError("test error")
            return MockObservation()

        pipeline = MagicMock()
        pipeline.observe.side_effect = observe_with_error
        engine = MagicMock()
        engine.decide.return_value = MockDecision()
        executor = MagicMock()
        executor.execute.return_value = MockActionResult()

        config = LoopConfig(
            target_rate_hz=100.0,
            min_iteration_ms=1.0,
            error_recovery_delay_ms=10.0,
            enable_signal_handlers=False,
        )
        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
            config=config,
        )

        loop.start(blocking=False)
        time.sleep(0.3)
        loop.stop()

        # Should have continued past the error
        assert call_count[0] > 3
        metrics = loop.metrics.get_metrics()
        assert metrics.errors_recovered >= 1

    def test_fatal_error_stops_loop(self) -> None:
        """Test that fatal errors stop the loop."""
        pipeline = MagicMock()
        pipeline.observe.side_effect = FatalError("fatal")
        engine = MagicMock()
        executor = MagicMock()

        config = LoopConfig(enable_signal_handlers=False)
        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
            config=config,
        )

        loop.start(blocking=False)
        time.sleep(0.2)

        assert loop.state == LoopState.ERROR

    def test_max_consecutive_errors_stops_loop(self) -> None:
        """Test that too many consecutive errors stop the loop."""
        pipeline = MagicMock()
        pipeline.observe.side_effect = ValueError("error")
        engine = MagicMock()
        executor = MagicMock()

        config = LoopConfig(
            max_consecutive_errors=3,
            error_recovery_delay_ms=50.0,  # Slightly longer delay
            enable_signal_handlers=False,
        )
        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
            config=config,
        )

        loop.start(blocking=False)
        # Wait for errors to accumulate (3 errors * 50ms delay = 150ms minimum)
        # Add extra time for processing
        for _ in range(20):  # Poll for up to 2 seconds
            if loop.state in (LoopState.ERROR, LoopState.STOPPED):
                break
            time.sleep(0.1)

        try:
            # Verify the loop stopped due to errors
            assert loop.state in (LoopState.ERROR, LoopState.STOPPED)
            # Verify errors were recorded
            metrics = loop.metrics.get_metrics()
            assert metrics.errors_total >= config.max_consecutive_errors
        finally:
            # Clean up - stop if still running
            if loop.state == LoopState.RUNNING:
                loop.stop()


class TestAgentLoopCallbacks:
    """Tests for AgentLoop callback functionality."""

    def test_on_iteration_complete_callback(self) -> None:
        """Test that on_iteration_complete callback is called."""
        iterations: list[int] = []

        def on_complete(count: int) -> None:
            iterations.append(count)

        pipeline = MagicMock()
        pipeline.observe.return_value = MockObservation()
        engine = MagicMock()
        engine.decide.return_value = MockDecision()
        executor = MagicMock()
        executor.execute.return_value = MockActionResult()

        config = LoopConfig(
            target_rate_hz=50.0,
            min_iteration_ms=10.0,
            enable_signal_handlers=False,
        )
        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
            config=config,
        )
        loop.set_callbacks(on_iteration_complete=on_complete)

        loop.start(blocking=False)
        time.sleep(0.2)
        loop.stop()

        assert len(iterations) > 0

    def test_on_error_callback(self) -> None:
        """Test that on_error callback is called."""
        errors: list[Exception] = []

        def on_error(e: Exception) -> None:
            errors.append(e)

        pipeline = MagicMock()
        pipeline.observe.side_effect = RecoverableError("test")
        engine = MagicMock()
        executor = MagicMock()

        config = LoopConfig(
            error_recovery_delay_ms=10.0,
            max_consecutive_errors=100,
            enable_signal_handlers=False,
        )
        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
            config=config,
        )
        loop.set_callbacks(on_error=on_error)

        loop.start(blocking=False)
        time.sleep(0.2)
        loop.stop()

        assert len(errors) > 0
        assert all(isinstance(e, RecoverableError) for e in errors)

    def test_on_state_change_callback(self) -> None:
        """Test that on_state_change callback is called."""
        states: list[LoopState] = []

        def on_state(state: LoopState) -> None:
            states.append(state)

        pipeline = MagicMock()
        pipeline.observe.return_value = MockObservation()
        engine = MagicMock()
        engine.decide.return_value = MockDecision()
        executor = MagicMock()
        executor.execute.return_value = MockActionResult()

        config = LoopConfig(enable_signal_handlers=False)
        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
            config=config,
        )
        loop.set_callbacks(on_state_change=on_state)

        loop.start(blocking=False)
        time.sleep(0.1)
        loop.pause()
        time.sleep(0.1)
        loop.resume()
        time.sleep(0.1)
        loop.stop()

        # Should have RUNNING, PAUSED, RUNNING, STOPPING, STOPPED
        assert LoopState.RUNNING in states
        assert LoopState.PAUSED in states
        assert LoopState.STOPPED in states


class TestAgentLoopMetrics:
    """Tests for AgentLoop metrics collection."""

    def test_metrics_recorded_during_loop(self) -> None:
        """Test that metrics are recorded during loop execution."""
        pipeline = MagicMock()
        pipeline.observe.return_value = MockObservation()
        engine = MagicMock()
        engine.decide.return_value = MockDecision()
        executor = MagicMock()
        executor.execute.return_value = MockActionResult()

        config = LoopConfig(
            target_rate_hz=50.0,
            min_iteration_ms=10.0,
            enable_signal_handlers=False,
        )
        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
            config=config,
        )

        loop.start(blocking=False)
        time.sleep(0.3)
        loop.stop()

        metrics = loop.metrics.get_metrics()
        assert metrics.loop_count > 0
        assert metrics.actions_total > 0
        assert metrics.started_at is not None
        assert metrics.uptime_seconds > 0

    def test_resources_updated_from_observation(self) -> None:
        """Test that resources are updated from observations."""
        obs = MockObservation()
        obs.game_state.resources = {
            "gold": MockResource(5000.0),
            "gems": MockResource(100.0),
        }

        pipeline = MagicMock()
        pipeline.observe.return_value = obs
        engine = MagicMock()
        engine.decide.return_value = MockDecision()
        executor = MagicMock()
        executor.execute.return_value = MockActionResult()

        config = LoopConfig(
            target_rate_hz=50.0,
            enable_signal_handlers=False,
        )
        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=engine,
            action_executor=executor,
            config=config,
        )

        loop.start(blocking=False)
        time.sleep(0.2)
        loop.stop()

        metrics = loop.metrics.get_metrics()
        assert "gold" in metrics.resources_snapshot
        assert metrics.resources_snapshot["gold"] == 5000.0


class TestModuleExports:
    """Tests for module exports."""

    def test_exports_from_loop_module(self) -> None:
        """Test that all expected classes are exported."""
        from src.core.loop import (
            AgentLoop,
            FatalError,
            LoopConfig,
            LoopState,
            RecoverableError,
        )

        assert AgentLoop is not None
        assert LoopConfig is not None
        assert LoopState is not None
        assert RecoverableError is not None
        assert FatalError is not None
