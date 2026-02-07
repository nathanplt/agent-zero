"""Tests for Feature 3.3: Action Executor.

These tests verify:
- Action validation (bounds, parameters)
- Action execution dispatch
- Rate limiting
- Post-action verification hooks
"""

import time
from unittest.mock import MagicMock

import pytest

from src.actions.backend import NullInputBackend
from src.actions.executor import (
    GameActionExecutor,
    RateLimitConfig,
    ValidationConfig,
)
from src.interfaces.actions import Action, ActionType, Point


class TestActionExecutorInitialization:
    """Tests for executor initialization."""

    def test_initialization_default(self):
        """Should initialize with default config."""
        executor = GameActionExecutor()

        assert executor._backend is not None
        assert executor._mouse is not None
        assert executor._keyboard is not None

    def test_initialization_with_backend(self):
        """Should accept custom backend."""
        backend = NullInputBackend()
        executor = GameActionExecutor(backend=backend)

        assert executor._backend is backend

    def test_initialization_with_configs(self):
        """Should accept custom configurations."""
        validation_config = ValidationConfig(screen_width=800, screen_height=600)
        rate_config = RateLimitConfig(min_action_interval_ms=100)

        executor = GameActionExecutor(
            validation_config=validation_config,
            rate_limit_config=rate_config,
        )

        assert executor._validation_config.screen_width == 800
        assert executor._rate_limit_config.min_action_interval_ms == 100


class TestActionValidation:
    """Tests for action validation."""

    @pytest.fixture
    def executor(self):
        """Create executor with custom screen bounds."""
        config = ValidationConfig(screen_width=1920, screen_height=1080)
        return GameActionExecutor(validation_config=config)

    def test_validate_click_valid(self, executor):
        """Valid click should pass validation."""
        action = Action(
            action_type=ActionType.CLICK,
            target=Point(100, 200),
            parameters={"button": "left"},
        )

        is_valid, error = executor.validate(action)

        assert is_valid is True
        assert error is None

    def test_validate_click_out_of_bounds_x(self, executor):
        """Click with X out of bounds should fail."""
        action = Action(
            action_type=ActionType.CLICK,
            target=Point(2000, 200),
            parameters={"button": "left"},
        )

        is_valid, error = executor.validate(action)

        assert is_valid is False
        assert "X coordinate" in error
        assert "out of bounds" in error

    def test_validate_click_out_of_bounds_y(self, executor):
        """Click with Y out of bounds should fail."""
        action = Action(
            action_type=ActionType.CLICK,
            target=Point(100, 2000),
            parameters={"button": "left"},
        )

        is_valid, error = executor.validate(action)

        assert is_valid is False
        assert "Y coordinate" in error

    def test_validate_click_negative_x(self, executor):
        """Click with negative X should fail."""
        action = Action(
            action_type=ActionType.CLICK,
            target=Point(-10, 200),
            parameters={},
        )

        is_valid, error = executor.validate(action)

        assert is_valid is False
        assert "X coordinate" in error

    def test_validate_click_no_target(self, executor):
        """Click without target should fail."""
        action = Action(action_type=ActionType.CLICK, parameters={})

        is_valid, error = executor.validate(action)

        assert is_valid is False
        assert "requires a target" in error

    def test_validate_click_invalid_button(self, executor):
        """Click with invalid button should fail."""
        action = Action(
            action_type=ActionType.CLICK,
            target=Point(100, 200),
            parameters={"button": "invalid"},
        )

        is_valid, error = executor.validate(action)

        assert is_valid is False
        assert "Invalid button" in error

    def test_validate_type_valid(self, executor):
        """Valid type action should pass."""
        action = Action(
            action_type=ActionType.TYPE,
            parameters={"text": "Hello"},
        )

        is_valid, error = executor.validate(action)

        assert is_valid is True

    def test_validate_type_no_text(self, executor):
        """Type without text should fail."""
        action = Action(action_type=ActionType.TYPE, parameters={})

        is_valid, error = executor.validate(action)

        assert is_valid is False
        assert "text" in error.lower()

    def test_validate_type_text_too_long(self, executor):
        """Type with text too long should fail."""
        executor._validation_config.max_text_length = 10
        action = Action(
            action_type=ActionType.TYPE,
            parameters={"text": "x" * 100},
        )

        is_valid, error = executor.validate(action)

        assert is_valid is False
        assert "too long" in error

    def test_validate_key_press_valid(self, executor):
        """Valid key press should pass."""
        action = Action(
            action_type=ActionType.KEY_PRESS,
            parameters={"key": "enter"},
        )

        is_valid, error = executor.validate(action)

        assert is_valid is True

    def test_validate_key_press_no_key(self, executor):
        """Key press without key should fail."""
        action = Action(action_type=ActionType.KEY_PRESS, parameters={})

        is_valid, error = executor.validate(action)

        assert is_valid is False
        assert "key" in error.lower()

    def test_validate_key_combo_valid(self, executor):
        """Valid key combo should pass."""
        action = Action(
            action_type=ActionType.KEY_COMBO,
            parameters={"keys": ["ctrl", "a"]},
        )

        is_valid, error = executor.validate(action)

        assert is_valid is True

    def test_validate_key_combo_single_key(self, executor):
        """Key combo with single key should fail."""
        action = Action(
            action_type=ActionType.KEY_COMBO,
            parameters={"keys": ["ctrl"]},
        )

        is_valid, error = executor.validate(action)

        assert is_valid is False
        assert "at least 2" in error

    def test_validate_scroll_valid(self, executor):
        """Valid scroll should pass."""
        action = Action(
            action_type=ActionType.SCROLL,
            parameters={"direction": "up"},
        )

        is_valid, error = executor.validate(action)

        assert is_valid is True

    def test_validate_scroll_invalid_direction(self, executor):
        """Scroll with invalid direction should fail."""
        action = Action(
            action_type=ActionType.SCROLL,
            parameters={"direction": "sideways"},
        )

        is_valid, error = executor.validate(action)

        assert is_valid is False
        assert "direction" in error.lower()

    def test_validate_wait_valid(self, executor):
        """Valid wait should pass."""
        action = Action(
            action_type=ActionType.WAIT,
            parameters={"duration_ms": 100},
        )

        is_valid, error = executor.validate(action)

        assert is_valid is True

    def test_validate_wait_negative_duration(self, executor):
        """Wait with negative duration should fail."""
        action = Action(
            action_type=ActionType.WAIT,
            parameters={"duration_ms": -100},
        )

        is_valid, error = executor.validate(action)

        assert is_valid is False
        assert "duration" in error.lower()

    def test_validate_wait_zero_duration(self, executor):
        """Wait with zero duration should fail."""
        action = Action(
            action_type=ActionType.WAIT,
            parameters={"duration_ms": 0},
        )

        is_valid, error = executor.validate(action)

        assert is_valid is False
        assert "duration" in error.lower()

    def test_validate_move_valid(self, executor):
        """Valid move should pass."""
        action = Action(
            action_type=ActionType.MOVE,
            target=Point(500, 500),
        )

        is_valid, error = executor.validate(action)

        assert is_valid is True

    def test_validate_move_no_target(self, executor):
        """Move without target should fail."""
        action = Action(action_type=ActionType.MOVE, parameters={})

        is_valid, error = executor.validate(action)

        assert is_valid is False
        assert "target" in error.lower()

    def test_validate_bounds_disabled(self):
        """Validation should skip bounds check when disabled."""
        config = ValidationConfig(validate_bounds=False)
        executor = GameActionExecutor(validation_config=config)

        action = Action(
            action_type=ActionType.CLICK,
            target=Point(99999, 99999),
            parameters={"button": "left"},
        )

        is_valid, error = executor.validate(action)

        assert is_valid is True

    def test_validate_double_click_no_target(self, executor):
        """Double-click without target should fail."""
        action = Action(action_type=ActionType.DOUBLE_CLICK, parameters={})

        is_valid, error = executor.validate(action)

        assert is_valid is False
        assert "target" in error.lower()

    def test_validate_right_click_no_target(self, executor):
        """Right-click without target should fail."""
        action = Action(action_type=ActionType.RIGHT_CLICK, parameters={})

        is_valid, error = executor.validate(action)

        assert is_valid is False
        assert "target" in error.lower()

    def test_validate_drag_no_end(self, executor):
        """Drag without end point should fail."""
        action = Action(
            action_type=ActionType.DRAG,
            target=Point(100, 100),
            parameters={},
        )

        is_valid, error = executor.validate(action)

        assert is_valid is False
        assert "end" in error.lower()

    def test_validate_drag_valid(self, executor):
        """Valid drag should pass validation."""
        action = Action(
            action_type=ActionType.DRAG,
            target=Point(100, 100),
            parameters={"end": {"x": 200, "y": 200}},
        )

        is_valid, error = executor.validate(action)

        assert is_valid is True


class TestActionExecution:
    """Tests for action execution."""

    @pytest.fixture
    def executor(self):
        """Create executor with NullInputBackend and no rate limiting."""
        rate_config = RateLimitConfig(min_action_interval_ms=0, max_actions_per_second=1000)
        return GameActionExecutor(rate_limit_config=rate_config)

    def test_execute_click(self, executor):
        """Should execute click action."""
        action = Action(
            action_type=ActionType.CLICK,
            target=Point(100, 200),
            parameters={"button": "left"},
        )

        result = executor.execute(action)

        assert result.success is True
        assert result.action is action
        assert result.duration_ms > 0

    def test_execute_double_click(self, executor):
        """Should execute double-click action."""
        action = Action(
            action_type=ActionType.DOUBLE_CLICK,
            target=Point(100, 200),
        )

        result = executor.execute(action)

        assert result.success is True

    def test_execute_right_click(self, executor):
        """Should execute right-click action."""
        action = Action(
            action_type=ActionType.RIGHT_CLICK,
            target=Point(100, 200),
        )

        result = executor.execute(action)

        assert result.success is True

    def test_execute_type(self, executor):
        """Should execute type action."""
        action = Action(
            action_type=ActionType.TYPE,
            parameters={"text": "Hello"},
        )

        result = executor.execute(action)

        assert result.success is True

    def test_execute_key_press(self, executor):
        """Should execute key press action."""
        action = Action(
            action_type=ActionType.KEY_PRESS,
            parameters={"key": "enter"},
        )

        result = executor.execute(action)

        assert result.success is True

    def test_execute_key_combo(self, executor):
        """Should execute key combo action."""
        action = Action(
            action_type=ActionType.KEY_COMBO,
            parameters={"keys": ["ctrl", "a"]},
        )

        result = executor.execute(action)

        assert result.success is True

    def test_execute_scroll(self, executor):
        """Should execute scroll action."""
        action = Action(
            action_type=ActionType.SCROLL,
            parameters={"direction": "down", "amount": 3},
        )

        result = executor.execute(action)

        assert result.success is True

    def test_execute_wait(self, executor):
        """Should execute wait action."""
        action = Action(
            action_type=ActionType.WAIT,
            parameters={"duration_ms": 10},
        )

        start = time.time()
        result = executor.execute(action)
        elapsed = (time.time() - start) * 1000

        assert result.success is True
        assert elapsed >= 10

    def test_execute_move(self, executor):
        """Should execute move action."""
        action = Action(
            action_type=ActionType.MOVE,
            target=Point(500, 500),
        )

        result = executor.execute(action)

        assert result.success is True

    def test_execute_drag(self, executor):
        """Should execute drag action."""
        action = Action(
            action_type=ActionType.DRAG,
            target=Point(100, 100),
            parameters={"end": {"x": 200, "y": 200}},
        )

        result = executor.execute(action)

        assert result.success is True

    def test_execute_invalid_action_returns_failure(self, executor):
        """Invalid action should return failure result, not raise."""
        action = Action(
            action_type=ActionType.CLICK,
            target=Point(99999, 99999),  # Out of bounds
            parameters={},
        )

        result = executor.execute(action)

        assert result.success is False
        assert result.error is not None

    def test_execute_sequence(self, executor):
        """Should execute sequence of actions."""
        actions = [
            Action(action_type=ActionType.MOVE, target=Point(100, 100)),
            Action(
                action_type=ActionType.CLICK,
                target=Point(100, 100),
                parameters={"button": "left"},
            ),
            Action(action_type=ActionType.TYPE, parameters={"text": "test"}),
        ]

        results = executor.execute_sequence(actions)

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_execute_sequence_stops_on_failure(self, executor):
        """Sequence should stop on first failure."""
        actions = [
            Action(action_type=ActionType.MOVE, target=Point(100, 100)),
            Action(action_type=ActionType.CLICK, target=Point(99999, 99999)),  # Fails
            Action(action_type=ActionType.TYPE, parameters={"text": "test"}),
        ]

        results = executor.execute_sequence(actions)

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False


class TestVerificationCallback:
    """Tests for post-action verification."""

    @pytest.fixture
    def executor(self):
        """Create executor with NullInputBackend."""
        rate_config = RateLimitConfig(min_action_interval_ms=0, max_actions_per_second=1000)
        return GameActionExecutor(rate_limit_config=rate_config)

    def test_verification_callback_success(self, executor):
        """Verification callback returning True should keep success."""
        callback = MagicMock(return_value=True)
        executor.set_verification_callback(callback)

        action = Action(action_type=ActionType.MOVE, target=Point(100, 100))
        result = executor.execute(action)

        assert result.success is True
        callback.assert_called_once()

    def test_verification_callback_failure(self, executor):
        """Verification callback returning False should mark failure."""
        callback = MagicMock(return_value=False)
        executor.set_verification_callback(callback)

        action = Action(action_type=ActionType.MOVE, target=Point(100, 100))
        result = executor.execute(action)

        assert result.success is False
        assert "verification failed" in result.error.lower()

    def test_verification_callback_cleared(self, executor):
        """Setting callback to None should disable verification."""
        callback = MagicMock(return_value=True)
        executor.set_verification_callback(callback)
        executor.set_verification_callback(None)

        action = Action(action_type=ActionType.MOVE, target=Point(100, 100))
        executor.execute(action)

        callback.assert_not_called()


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_min_interval_enforced(self):
        """Minimum interval between actions should be enforced."""
        rate_config = RateLimitConfig(
            min_action_interval_ms=50,
            max_actions_per_second=100,
        )
        executor = GameActionExecutor(rate_limit_config=rate_config)

        action = Action(action_type=ActionType.MOVE, target=Point(100, 100))

        # Execute two actions and measure time
        start = time.time()
        executor.execute(action)
        executor.execute(action)
        elapsed = (time.time() - start) * 1000

        # Should have waited at least min_interval_ms between actions
        assert elapsed >= 50


class TestDirectMethods:
    """Tests for direct convenience methods."""

    @pytest.fixture
    def executor(self):
        """Create executor with NullInputBackend."""
        return GameActionExecutor()

    def test_move_mouse(self, executor):
        """Should move mouse to target."""
        target = Point(500, 500)

        # Should not raise
        executor.move_mouse(target, human_like=True)

    def test_move_mouse_direct(self, executor):
        """Should move mouse directly (no curve)."""
        target = Point(500, 500)

        # Should not raise
        executor.move_mouse(target, human_like=False)

    def test_click_at_target(self, executor):
        """Should click at target."""
        target = Point(100, 200)

        # Should not raise
        executor.click(target, button="left")

    def test_click_no_target(self, executor):
        """Should click at current position."""
        # Should not raise
        executor.click(None, button="left")

    def test_type_text(self, executor):
        """Should type text."""
        # Should not raise
        executor.type_text("Hello World")

    def test_type_text_fixed_interval(self, executor):
        """Should type text with fixed interval."""
        # Should not raise
        executor.type_text("Hi", interval_ms=10)

    def test_press_key(self, executor):
        """Should press key."""
        # Should not raise
        executor.press_key("enter")

    def test_key_combo(self, executor):
        """Should press key combo."""
        # Should not raise
        executor.key_combo(["ctrl", "a"])

    def test_scroll_directions(self, executor):
        """Should scroll in all directions."""
        for direction in ["up", "down", "left", "right"]:
            # Should not raise
            executor.scroll(direction, amount=1)


class TestModuleExports:
    """Tests for module exports."""

    def test_game_action_executor_exported(self):
        """GameActionExecutor should be exported."""
        from src.actions import GameActionExecutor

        assert GameActionExecutor is not None

    def test_rate_limit_config_exported(self):
        """RateLimitConfig should be exported."""
        from src.actions import RateLimitConfig

        assert RateLimitConfig is not None

    def test_validation_config_exported(self):
        """ValidationConfig should be exported."""
        from src.actions import ValidationConfig

        assert ValidationConfig is not None
