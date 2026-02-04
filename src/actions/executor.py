"""Action executor implementation.

This module provides the concrete implementation of ActionExecutor that:
- Coordinates mouse and keyboard controllers
- Validates actions before execution
- Executes actions with proper timing
- Supports rate limiting to avoid detection

Example:
    >>> from src.actions.executor import GameActionExecutor
    >>> from src.actions.backend import PlaywrightInputBackend
    >>> from src.interfaces.actions import Action, ActionType, Point
    >>>
    >>> # With a Playwright page
    >>> backend = PlaywrightInputBackend(page)
    >>> executor = GameActionExecutor(backend)
    >>>
    >>> # Execute a click action
    >>> action = Action(ActionType.CLICK, target=Point(100, 200))
    >>> result = executor.execute(action)
    >>> print(f"Success: {result.success}")
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.actions.backend import InputBackend, NullInputBackend
from src.actions.keyboard import KeyboardController
from src.actions.mouse import MouseController
from src.interfaces.actions import (
    Action,
    ActionError,
    ActionExecutor,
    ActionResult,
    ActionType,
    Point,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for action rate limiting.

    Attributes:
        min_action_interval_ms: Minimum time between actions in milliseconds.
        max_actions_per_second: Maximum actions allowed per second.
    """

    min_action_interval_ms: float = 50.0
    max_actions_per_second: float = 20.0


@dataclass
class ValidationConfig:
    """Configuration for action validation.

    Attributes:
        screen_width: Screen width for bounds checking.
        screen_height: Screen height for bounds checking.
        validate_bounds: Whether to validate coordinate bounds.
        max_text_length: Maximum text length for type actions.
    """

    screen_width: int = 1920
    screen_height: int = 1080
    validate_bounds: bool = True
    max_text_length: int = 10000


class GameActionExecutor(ActionExecutor):
    """Concrete implementation of ActionExecutor for game control.

    This executor:
    - Uses MouseController and KeyboardController for human-like input
    - Validates actions before execution (bounds, parameters)
    - Implements rate limiting to prevent detection
    - Provides hooks for post-action verification

    Attributes:
        _backend: The input backend for actual input operations.
        _mouse: Mouse controller for movement and clicks.
        _keyboard: Keyboard controller for typing and key presses.
        _validation_config: Configuration for action validation.
        _rate_limit_config: Configuration for rate limiting.
        _last_action_time: Timestamp of the last executed action.

    Example:
        >>> executor = GameActionExecutor()
        >>> action = Action(ActionType.CLICK, target=Point(100, 200))
        >>> result = executor.execute(action)
    """

    def __init__(
        self,
        backend: InputBackend | None = None,
        validation_config: ValidationConfig | None = None,
        rate_limit_config: RateLimitConfig | None = None,
    ) -> None:
        """Initialize the action executor.

        Args:
            backend: Input backend for actual input. Defaults to NullInputBackend.
            validation_config: Configuration for validation. Uses defaults if None.
            rate_limit_config: Configuration for rate limiting. Uses defaults if None.
        """
        self._backend = backend if backend is not None else NullInputBackend()
        self._mouse = MouseController(backend=self._backend)
        self._keyboard = KeyboardController(backend=self._backend)

        self._validation_config = validation_config or ValidationConfig()
        self._rate_limit_config = rate_limit_config or RateLimitConfig()

        self._last_action_time: float = 0.0
        self._action_timestamps: list[float] = []

        # Verification callback (can be set externally)
        self._verify_callback: Callable[[Action, ActionResult], bool] | None = None

        logger.debug(
            f"GameActionExecutor initialized with {type(self._backend).__name__}"
        )

    @property
    def mouse(self) -> MouseController:
        """Get the mouse controller."""
        return self._mouse

    @property
    def keyboard(self) -> KeyboardController:
        """Get the keyboard controller."""
        return self._keyboard

    def set_verification_callback(
        self, callback: Callable[[Action, ActionResult], bool] | None
    ) -> None:
        """Set a callback for post-action verification.

        The callback receives the action and its result, and should return
        True if the action was verified successfully, False otherwise.

        Args:
            callback: Verification callback, or None to disable verification.
        """
        self._verify_callback = callback

    def validate(self, action: Action) -> tuple[bool, str | None]:
        """Validate an action before execution.

        Checks:
        - Target coordinates are within screen bounds (if applicable)
        - Required parameters are present
        - Parameter values are valid

        Args:
            action: The action to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        config = self._validation_config

        # Validate target coordinates if present and bounds checking enabled
        if action.target and config.validate_bounds:
            if action.target.x < 0 or action.target.x >= config.screen_width:
                return (
                    False,
                    f"X coordinate {action.target.x} out of bounds [0, {config.screen_width})",
                )
            if action.target.y < 0 or action.target.y >= config.screen_height:
                return (
                    False,
                    f"Y coordinate {action.target.y} out of bounds [0, {config.screen_height})",
                )

        # Validate action-specific parameters
        if action.action_type == ActionType.CLICK:
            if action.target is None:
                return False, "Click action requires a target"
            button = action.parameters.get("button", "left")
            if button not in ("left", "right", "middle"):
                return False, f"Invalid button: {button}"

        elif action.action_type == ActionType.DOUBLE_CLICK:
            if action.target is None:
                return False, "Double-click action requires a target"

        elif action.action_type == ActionType.RIGHT_CLICK:
            if action.target is None:
                return False, "Right-click action requires a target"

        elif action.action_type == ActionType.DRAG:
            if action.target is None:
                return False, "Drag action requires a target (start point)"
            end = action.parameters.get("end")
            if not end:
                return False, "Drag action requires 'end' parameter"

        elif action.action_type == ActionType.TYPE:
            text = action.parameters.get("text")
            if not text:
                return False, "Type action requires 'text' parameter"
            if len(text) > config.max_text_length:
                return False, f"Text too long: {len(text)} > {config.max_text_length}"

        elif action.action_type == ActionType.KEY_PRESS:
            key = action.parameters.get("key")
            if not key:
                return False, "Key press action requires 'key' parameter"

        elif action.action_type == ActionType.KEY_COMBO:
            keys = action.parameters.get("keys")
            if not keys or not isinstance(keys, list):
                return False, "Key combo action requires 'keys' parameter (list)"
            if len(keys) < 2:
                return False, "Key combo requires at least 2 keys"

        elif action.action_type == ActionType.SCROLL:
            direction = action.parameters.get("direction")
            if direction not in ("up", "down", "left", "right"):
                return False, f"Invalid scroll direction: {direction}"

        elif action.action_type == ActionType.WAIT:
            duration = action.parameters.get("duration_ms")
            if duration is None or duration < 0:
                return False, "Wait action requires positive 'duration_ms' parameter"

        elif action.action_type == ActionType.MOVE:
            if action.target is None:
                return False, "Move action requires a target"

        return True, None

    def _apply_rate_limit(self) -> None:
        """Apply rate limiting before executing an action.

        Waits if necessary to maintain the configured rate limits.
        """
        current_time = time.time()
        config = self._rate_limit_config

        # Enforce minimum interval between actions
        elapsed_ms = (current_time - self._last_action_time) * 1000
        if elapsed_ms < config.min_action_interval_ms:
            sleep_time = (config.min_action_interval_ms - elapsed_ms) / 1000
            time.sleep(sleep_time)
            current_time = time.time()

        # Track action timestamps for rate limiting (keep last second)
        cutoff = current_time - 1.0
        self._action_timestamps = [t for t in self._action_timestamps if t >= cutoff]
        self._action_timestamps.append(current_time)

        # If we've exceeded the rate limit, wait
        if len(self._action_timestamps) > config.max_actions_per_second:
            # Wait until the oldest action is more than 1 second old
            oldest = self._action_timestamps[0]
            wait_time = 1.0 - (current_time - oldest)
            if wait_time > 0:
                logger.debug(f"Rate limiting: waiting {wait_time:.3f}s")
                time.sleep(wait_time)

    def execute(self, action: Action) -> ActionResult:
        """Execute a single action.

        Args:
            action: The action to execute.

        Returns:
            Result indicating success/failure.

        Raises:
            ActionError: If execution fails unexpectedly.
        """
        start_time = time.time()

        # Validate action
        is_valid, error = self.validate(action)
        if not is_valid:
            logger.warning(f"Action validation failed: {error}")
            return ActionResult(
                success=False,
                action=action,
                error=error,
                duration_ms=0.0,
            )

        # Apply rate limiting
        self._apply_rate_limit()

        try:
            # Execute the action
            self._execute_action(action)
            self._last_action_time = time.time()

            duration_ms = (time.time() - start_time) * 1000

            result = ActionResult(
                success=True,
                action=action,
                duration_ms=duration_ms,
            )

            # Run verification if callback is set
            if self._verify_callback:
                verified = self._verify_callback(action, result)
                if not verified:
                    logger.warning(f"Action verification failed: {action}")
                    result = ActionResult(
                        success=False,
                        action=action,
                        error="Post-action verification failed",
                        duration_ms=duration_ms,
                    )

            logger.debug(f"Executed {action.action_type.value} in {duration_ms:.1f}ms")
            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Action execution failed: {e}")
            raise ActionError(
                f"Failed to execute {action.action_type.value}: {e}"
            ) from e

    def _execute_action(self, action: Action) -> None:
        """Execute the actual action using controllers.

        Args:
            action: The action to execute.
        """
        if action.action_type == ActionType.CLICK:
            assert action.target is not None
            button = action.parameters.get("button", "left")
            self._mouse.click(action.target, button=button)

        elif action.action_type == ActionType.DOUBLE_CLICK:
            assert action.target is not None
            self._mouse.double_click(action.target)

        elif action.action_type == ActionType.RIGHT_CLICK:
            assert action.target is not None
            self._mouse.right_click(action.target)

        elif action.action_type == ActionType.DRAG:
            assert action.target is not None
            end_data = action.parameters["end"]
            end_point = Point(end_data["x"], end_data["y"])
            self._mouse.drag(action.target, end_point)

        elif action.action_type == ActionType.TYPE:
            text = action.parameters["text"]
            self._keyboard.type_text(text)

        elif action.action_type == ActionType.KEY_PRESS:
            key = action.parameters["key"]
            self._keyboard.press_key(key)

        elif action.action_type == ActionType.KEY_COMBO:
            keys = action.parameters["keys"]
            self._keyboard.key_combo(keys)

        elif action.action_type == ActionType.SCROLL:
            direction = action.parameters["direction"]
            amount = action.parameters.get("amount", 1)
            self._mouse.scroll(direction, amount)

        elif action.action_type == ActionType.WAIT:
            duration_ms = action.parameters["duration_ms"]
            time.sleep(duration_ms / 1000)

        elif action.action_type == ActionType.MOVE:
            assert action.target is not None
            self._mouse.move(action.target)

        else:
            raise ActionError(f"Unknown action type: {action.action_type}")

    def execute_sequence(self, actions: list[Action]) -> list[ActionResult]:
        """Execute a sequence of actions.

        Stops on first failure.

        Args:
            actions: List of actions to execute in order.

        Returns:
            List of results for each action.
        """
        results: list[ActionResult] = []

        for action in actions:
            result = self.execute(action)
            results.append(result)

            if not result.success:
                logger.warning(f"Sequence stopped at failed action: {action}")
                break

        return results

    def move_mouse(self, target: Point, human_like: bool = True) -> None:
        """Move the mouse to a target position.

        Args:
            target: Destination point.
            human_like: If True, use curved path with natural timing.
        """
        if human_like:
            self._mouse.move(target)
        else:
            # Direct move via backend
            self._backend.mouse_move(target.x, target.y)

    def click(self, target: Point | None = None, button: str = "left") -> None:
        """Click at the current or specified position.

        Args:
            target: Optional target point. If None, clicks at current position.
            button: Mouse button ('left', 'right', 'middle').
        """
        self._mouse.click(target, button=button)

    def type_text(self, text: str, interval_ms: float | None = None) -> None:
        """Type text with natural timing.

        Args:
            text: Text to type.
            interval_ms: Optional fixed interval between keystrokes.
        """
        self._keyboard.type_text(text, interval_ms=interval_ms)

    def press_key(self, key: str) -> None:
        """Press a single key.

        Args:
            key: Key to press (e.g., 'enter', 'escape', 'tab').
        """
        self._keyboard.press_key(key)

    def key_combo(self, keys: list[str]) -> None:
        """Press a key combination.

        Args:
            keys: List of keys to press together (e.g., ['ctrl', 'a']).
        """
        self._keyboard.key_combo(keys)

    def scroll(self, direction: str, amount: int = 1) -> None:
        """Scroll in a direction.

        Args:
            direction: 'up', 'down', 'left', or 'right'.
            amount: Number of scroll units.
        """
        self._mouse.scroll(direction, amount)
