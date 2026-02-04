"""Action executor interface for controlling the game."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class ActionType(Enum):
    """Types of actions the agent can perform."""

    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    TYPE = "type"
    KEY_PRESS = "key_press"
    KEY_COMBO = "key_combo"
    SCROLL = "scroll"
    WAIT = "wait"
    MOVE = "move"


class Point:
    """A point in screen coordinates."""

    __slots__ = ("x", "y")

    def __init__(self, x: int, y: int) -> None:
        """Initialize a point.

        Args:
            x: X coordinate.
            y: Y coordinate.
        """
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y


class Action:
    """An action to be executed by the agent."""

    __slots__ = ("action_type", "target", "parameters", "description")

    def __init__(
        self,
        action_type: ActionType,
        target: Point | None = None,
        parameters: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize an action.

        Args:
            action_type: The type of action to perform.
            target: Target point for click/move actions.
            parameters: Additional parameters (text for typing, key for press, etc.).
            description: Human-readable description of the action.
        """
        self.action_type = action_type
        self.target = target
        self.parameters = parameters or {}
        self.description = description

    def __repr__(self) -> str:
        return f"Action({self.action_type.value}, target={self.target})"


class ActionResult:
    """Result of an executed action."""

    __slots__ = ("success", "action", "error", "duration_ms")

    def __init__(
        self,
        success: bool,
        action: Action,
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Initialize an action result.

        Args:
            success: Whether the action completed successfully.
            action: The action that was executed.
            error: Error message if action failed.
            duration_ms: How long the action took in milliseconds.
        """
        self.success = success
        self.action = action
        self.error = error
        self.duration_ms = duration_ms


class ActionExecutor(ABC):
    """Abstract interface for executing actions.

    The action executor is responsible for:
    - Moving the mouse with human-like movements
    - Clicking and typing
    - Validating actions before execution
    - Verifying action effects
    """

    @abstractmethod
    def execute(self, action: Action) -> ActionResult:
        """Execute a single action.

        Args:
            action: The action to execute.

        Returns:
            Result indicating success/failure.

        Raises:
            ActionError: If execution fails unexpectedly.
        """
        ...

    @abstractmethod
    def execute_sequence(self, actions: list[Action]) -> list[ActionResult]:
        """Execute a sequence of actions.

        Args:
            actions: List of actions to execute in order.

        Returns:
            List of results for each action.
        """
        ...

    @abstractmethod
    def validate(self, action: Action) -> tuple[bool, str | None]:
        """Validate an action before execution.

        Args:
            action: The action to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        ...

    @abstractmethod
    def move_mouse(self, target: Point, human_like: bool = True) -> None:
        """Move the mouse to a target position.

        Args:
            target: Destination point.
            human_like: If True, use curved path with natural timing.
        """
        ...

    @abstractmethod
    def click(self, target: Point | None = None, button: str = "left") -> None:
        """Click at the current or specified position.

        Args:
            target: Optional target point. If None, clicks at current position.
            button: Mouse button ('left', 'right', 'middle').
        """
        ...

    @abstractmethod
    def type_text(self, text: str, interval_ms: float | None = None) -> None:
        """Type text with natural timing.

        Args:
            text: Text to type.
            interval_ms: Optional fixed interval between keystrokes.
        """
        ...

    @abstractmethod
    def press_key(self, key: str) -> None:
        """Press a single key.

        Args:
            key: Key to press (e.g., 'enter', 'escape', 'tab').
        """
        ...

    @abstractmethod
    def key_combo(self, keys: list[str]) -> None:
        """Press a key combination.

        Args:
            keys: List of keys to press together (e.g., ['ctrl', 'a']).
        """
        ...

    @abstractmethod
    def scroll(self, direction: str, amount: int = 1) -> None:
        """Scroll in a direction.

        Args:
            direction: 'up', 'down', 'left', or 'right'.
            amount: Number of scroll units.
        """
        ...


class ActionError(Exception):
    """Error raised when action execution fails."""

    pass
