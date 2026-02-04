"""Mouse control module for human-like mouse movement and clicks.

This module provides:
- Human-like mouse movement using Bezier curves
- Natural timing variance
- Click variations (single, double, right-click, hold)
- Drag operations

Example:
    >>> from src.actions.mouse import MouseController
    >>> from src.interfaces.actions import Point
    >>>
    >>> controller = MouseController()
    >>> controller.move(Point(500, 300))
    >>> controller.click()
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import TYPE_CHECKING

from src.interfaces.actions import Point

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def generate_bezier_path(
    start: tuple[float, float],
    end: tuple[float, float],
    num_points: int | None = None,
    curvature: float = 0.3,
) -> list[tuple[float, float]]:
    """Generate a curved path using a cubic Bezier curve.

    Creates a human-like curved path between two points using
    randomized control points for natural movement.

    Args:
        start: Starting point (x, y).
        end: Ending point (x, y).
        num_points: Number of points in the path. If None, calculated from distance.
        curvature: How much the path curves (0.0 = straight, 1.0 = very curved).

    Returns:
        List of (x, y) points along the curve.
    """
    # Calculate distance for determining number of points
    distance = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)

    if num_points is None:
        # Scale points with distance, minimum 10, maximum 100
        num_points = max(10, min(100, int(distance / 5)))

    # Generate random control points for cubic Bezier curve
    # Control points are offset perpendicular to the line
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    # Perpendicular direction
    perp_x = -dy
    perp_y = dx

    # Normalize perpendicular
    perp_len = math.sqrt(perp_x**2 + perp_y**2)
    if perp_len > 0:
        perp_x /= perp_len
        perp_y /= perp_len

    # Random offsets for control points
    offset1 = random.gauss(0, curvature * distance * 0.3)
    offset2 = random.gauss(0, curvature * distance * 0.3)

    # Control point 1 (1/3 of the way)
    cp1_x = start[0] + dx * 0.33 + perp_x * offset1
    cp1_y = start[1] + dy * 0.33 + perp_y * offset1

    # Control point 2 (2/3 of the way)
    cp2_x = start[0] + dx * 0.67 + perp_x * offset2
    cp2_y = start[1] + dy * 0.67 + perp_y * offset2

    # Generate path points using cubic Bezier formula
    path = []
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0

        # Cubic Bezier formula: B(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt

        x = mt3 * start[0] + 3 * mt2 * t * cp1_x + 3 * mt * t2 * cp2_x + t3 * end[0]
        y = mt3 * start[1] + 3 * mt2 * t * cp1_y + 3 * mt * t2 * cp2_y + t3 * end[1]

        path.append((x, y))

    return path


def calculate_movement_duration(
    distance: float,
    base_speed: float = 500.0,
    variance: float = 0.2,
) -> float:
    """Calculate duration for mouse movement with natural variance.

    Uses Fitts's Law inspired calculation with random variance.

    Args:
        distance: Distance to travel in pixels.
        base_speed: Base speed in pixels per second.
        variance: Amount of random variance (0.0 to 1.0).

    Returns:
        Duration in seconds.
    """
    # Base duration from distance
    base_duration = distance / base_speed

    # Add logarithmic component for longer distances (Fitts's Law)
    if distance > 100:
        base_duration += math.log(distance / 100) * 0.1

    # Add random variance
    variance_amount = base_duration * variance * random.gauss(0, 1)
    duration = base_duration + variance_amount

    # Clamp to reasonable range
    return max(0.05, min(2.0, duration))


def calculate_click_duration(
    base_ms: float = 80.0,
    variance_ms: float = 30.0,
) -> float:
    """Calculate duration for a click with natural variance.

    Args:
        base_ms: Base click duration in milliseconds.
        variance_ms: Standard deviation of variance.

    Returns:
        Duration in seconds.
    """
    duration_ms = base_ms + random.gauss(0, variance_ms)
    return max(0.03, duration_ms / 1000.0)


class MouseController:
    """Controller for human-like mouse movement and clicks.

    Provides natural mouse movement using Bezier curves,
    click operations with timing variance, and drag support.

    Attributes:
        _screen_width: Screen width in pixels.
        _screen_height: Screen height in pixels.
        _current_position: Current mouse position.

    Example:
        >>> controller = MouseController(screen_width=1920, screen_height=1080)
        >>> controller.move(Point(500, 300))
        >>> controller.click()
    """

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> None:
        """Initialize the mouse controller.

        Args:
            screen_width: Screen width in pixels.
            screen_height: Screen height in pixels.
        """
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._current_position = (0, 0)

        logger.debug(f"MouseController initialized for {screen_width}x{screen_height}")

    def _clamp_to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Clamp coordinates to screen bounds.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            Clamped (x, y) as integers.
        """
        clamped_x = max(0, min(self._screen_width - 1, int(x)))
        clamped_y = max(0, min(self._screen_height - 1, int(y)))
        return (clamped_x, clamped_y)

    def _execute_move(self, path: list[tuple[float, float]]) -> None:
        """Execute mouse movement along a path.

        This is the internal method that performs the actual movement.
        Override or mock for testing.

        Args:
            path: List of (x, y) points to move through.
        """
        # Default implementation just updates position
        # Real implementation would use pyautogui or similar
        if path:
            final = path[-1]
            self._current_position = self._clamp_to_screen(final[0], final[1])

    def _execute_click(self, button: str = "left") -> None:
        """Execute a mouse click.

        Args:
            button: Mouse button ('left', 'right', 'middle').
        """
        # Default implementation does nothing
        # Real implementation would use pyautogui or similar
        logger.debug(f"Click: {button} at {self._current_position}")

    def _mouse_down(self, button: str = "left") -> None:
        """Press mouse button down.

        Args:
            button: Mouse button.
        """
        logger.debug(f"Mouse down: {button}")

    def _mouse_up(self, button: str = "left") -> None:
        """Release mouse button.

        Args:
            button: Mouse button.
        """
        logger.debug(f"Mouse up: {button}")

    def _get_current_position(self) -> tuple[int, int]:
        """Get current mouse position.

        Returns:
            Current (x, y) position.
        """
        return self._current_position

    def get_position(self) -> Point:
        """Get current mouse position as Point.

        Returns:
            Current position as Point object.
        """
        pos = self._get_current_position()
        return Point(pos[0], pos[1])

    def move(
        self,
        target: Point,
        human_like: bool = True,
    ) -> Point:
        """Move mouse to target position.

        Args:
            target: Target position.
            human_like: If True, use curved path with natural timing.

        Returns:
            Final position after movement.
        """
        current = self._get_current_position()

        # Clamp target to screen bounds
        target_x, target_y = self._clamp_to_screen(target.x, target.y)

        if human_like:
            # Generate curved path
            path = generate_bezier_path(
                current,
                (target_x, target_y),
            )

            # Calculate total duration
            distance = math.sqrt(
                (target_x - current[0]) ** 2 + (target_y - current[1]) ** 2
            )
            total_duration = calculate_movement_duration(distance)

            # Execute movement with timing
            if len(path) > 1:
                step_duration = total_duration / (len(path) - 1)
                for point in path:
                    self._execute_move([point])
                    time.sleep(step_duration)
            else:
                self._execute_move(path)
        else:
            # Direct movement
            self._execute_move([(target_x, target_y)])

        self._current_position = (target_x, target_y)
        logger.debug(f"Moved to ({target_x}, {target_y})")

        return Point(target_x, target_y)

    def click(
        self,
        target: Point | None = None,
        button: str = "left",
    ) -> None:
        """Click at target or current position.

        Args:
            target: Optional target position. If None, clicks at current position.
            button: Mouse button ('left', 'right', 'middle').
        """
        if target is not None:
            self.move(target)

        # Natural click timing
        click_duration = calculate_click_duration()

        self._mouse_down(button)
        time.sleep(click_duration)
        self._mouse_up(button)

        logger.debug(f"Clicked {button} at {self._current_position}")

    def double_click(
        self,
        target: Point | None = None,
    ) -> None:
        """Double-click at target or current position.

        Args:
            target: Optional target position.
        """
        if target is not None:
            self.move(target)

        # Two rapid clicks
        self._execute_click("left")
        time.sleep(random.uniform(0.05, 0.1))  # Natural delay between clicks
        self._execute_click("left")

        logger.debug(f"Double-clicked at {self._current_position}")

    def right_click(
        self,
        target: Point | None = None,
    ) -> None:
        """Right-click at target or current position.

        Args:
            target: Optional target position.
        """
        self.click(target, button="right")

    def click_and_hold(
        self,
        target: Point | None = None,
        duration_ms: float = 500,
    ) -> None:
        """Click and hold at target or current position.

        Args:
            target: Optional target position.
            duration_ms: How long to hold in milliseconds.
        """
        if target is not None:
            self.move(target)

        self._mouse_down("left")
        time.sleep(duration_ms / 1000.0)
        self._mouse_up("left")

        logger.debug(f"Click and hold at {self._current_position} for {duration_ms}ms")

    def drag(
        self,
        start: Point,
        end: Point,
        button: str = "left",
    ) -> None:
        """Drag from start to end position.

        Args:
            start: Starting position.
            end: Ending position.
            button: Mouse button to hold during drag.
        """
        # Move to start
        self.move(start)

        # Press, move, release
        self._mouse_down(button)
        self.move(end)
        self._mouse_up(button)

        logger.debug(f"Dragged from {start} to {end}")

    def scroll(
        self,
        direction: str = "down",
        amount: int = 3,
    ) -> None:
        """Scroll in a direction.

        Args:
            direction: 'up', 'down', 'left', or 'right'.
            amount: Number of scroll units.
        """
        # Default implementation does nothing
        # Real implementation would use pyautogui or similar
        logger.debug(f"Scroll {direction} by {amount}")
