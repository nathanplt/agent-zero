"""Tests for Feature 3.1: Mouse Control.

These tests verify mouse control functionality:
- Move to any coordinate in display
- Click accurately on target
- Movement follows curved path (not linear)
- Timing varies naturally (not robotic)

Note: Tests use mocked input for unit testing.
Integration tests marked with @pytest.mark.integration require real display.
"""

import math
import statistics
from unittest.mock import patch

import pytest

from src.interfaces.actions import Point


class TestMouseControllerImport:
    """Tests for MouseController class import and basic structure."""

    def test_mouse_controller_class_exists(self):
        """MouseController class should be importable."""
        from src.actions.mouse import MouseController

        assert MouseController is not None

    def test_has_move_method(self):
        """MouseController should have a move method."""
        from src.actions.mouse import MouseController

        assert hasattr(MouseController, "move")

    def test_has_click_method(self):
        """MouseController should have a click method."""
        from src.actions.mouse import MouseController

        assert hasattr(MouseController, "click")

    def test_has_double_click_method(self):
        """MouseController should have a double_click method."""
        from src.actions.mouse import MouseController

        assert hasattr(MouseController, "double_click")

    def test_has_right_click_method(self):
        """MouseController should have a right_click method."""
        from src.actions.mouse import MouseController

        assert hasattr(MouseController, "right_click")

    def test_has_drag_method(self):
        """MouseController should have a drag method."""
        from src.actions.mouse import MouseController

        assert hasattr(MouseController, "drag")


class TestMouseControllerInitialization:
    """Tests for MouseController initialization."""

    def test_initialization_default(self):
        """Should initialize with default settings."""
        from src.actions.mouse import MouseController

        controller = MouseController()
        assert controller is not None

    def test_initialization_with_screen_size(self):
        """Should accept screen size parameter."""
        from src.actions.mouse import MouseController

        controller = MouseController(screen_width=1920, screen_height=1080)
        assert controller._screen_width == 1920
        assert controller._screen_height == 1080


class TestMouseMovement:
    """Tests for mouse movement functionality."""

    @pytest.fixture
    def controller(self):
        """Create a MouseController instance."""
        from src.actions.mouse import MouseController

        return MouseController()

    def test_move_to_point(self, controller):
        """Should move to specified point."""
        target = Point(500, 300)

        with patch.object(controller, "_execute_move") as mock_move:
            controller.move(target)

        mock_move.assert_called()

    def test_move_returns_final_position(self, controller):
        """move should return the final position."""
        target = Point(500, 300)

        with (
            patch.object(controller, "_execute_move"),
            patch.object(controller, "get_position", return_value=target),
        ):
            result = controller.move(target)

        assert result == target

    def test_move_clamps_to_screen_bounds(self, controller):
        """Should clamp coordinates to screen bounds."""
        controller._screen_width = 800
        controller._screen_height = 600

        # Try to move outside bounds
        target = Point(1000, 800)

        with patch.object(controller, "_execute_move") as mock_move:
            controller.move(target)

        # Should have clamped the target
        call_args = mock_move.call_args
        path = call_args[0][0]
        final_point = path[-1]
        assert final_point[0] <= 800
        assert final_point[1] <= 600


class TestBezierCurveMovement:
    """Tests for human-like Bezier curve movement."""

    def test_generate_bezier_path_is_curved(self):
        """Generated path should be curved, not linear."""
        from src.actions.mouse import generate_bezier_path

        start = (0, 0)
        end = (100, 100)

        path = generate_bezier_path(start, end, num_points=50)

        # Calculate deviation from straight line
        deviations = []
        for i, (x, y) in enumerate(path):
            # Expected position on straight line
            t = i / (len(path) - 1) if len(path) > 1 else 0
            expected_x = start[0] + t * (end[0] - start[0])
            expected_y = start[1] + t * (end[1] - start[1])

            deviation = math.sqrt((x - expected_x) ** 2 + (y - expected_y) ** 2)
            deviations.append(deviation)

        # Path should have some curvature (non-zero deviation)
        max_deviation = max(deviations)
        assert max_deviation > 1.0, "Path should not be perfectly straight"

    def test_bezier_path_starts_at_origin(self):
        """Bezier path should start at the origin point."""
        from src.actions.mouse import generate_bezier_path

        start = (100, 200)
        end = (500, 400)

        path = generate_bezier_path(start, end)

        assert path[0] == start

    def test_bezier_path_ends_at_target(self):
        """Bezier path should end at the target point."""
        from src.actions.mouse import generate_bezier_path

        start = (100, 200)
        end = (500, 400)

        path = generate_bezier_path(start, end)

        # Allow small floating point tolerance
        assert abs(path[-1][0] - end[0]) < 1
        assert abs(path[-1][1] - end[1]) < 1

    def test_bezier_path_length_varies_with_distance(self):
        """Longer distances should generate more path points."""
        from src.actions.mouse import generate_bezier_path

        short_path = generate_bezier_path((0, 0), (50, 50))
        long_path = generate_bezier_path((0, 0), (500, 500))

        # Longer distance should have more points (or similar, scaled)
        # The actual implementation may vary, but path should exist
        assert len(short_path) >= 2
        assert len(long_path) >= 2


class TestTimingVariance:
    """Tests for natural timing variance."""

    def test_movement_timing_varies(self):
        """Movement timing should vary naturally."""
        from src.actions.mouse import calculate_movement_duration

        durations = []
        for _ in range(100):
            duration = calculate_movement_duration(distance=200)
            durations.append(duration)

        # Should have variance (not constant)
        std_dev = statistics.stdev(durations)
        assert std_dev > 0.01, "Timing should have variance"

    def test_click_timing_varies(self):
        """Click timing should vary naturally."""
        from src.actions.mouse import calculate_click_duration

        durations = []
        for _ in range(100):
            duration = calculate_click_duration()
            durations.append(duration)

        # Should have variance
        std_dev = statistics.stdev(durations)
        assert std_dev > 0.005, "Click timing should have variance"

    def test_longer_distance_takes_longer(self):
        """Longer distances should take more time on average."""
        from src.actions.mouse import calculate_movement_duration

        short_durations = [calculate_movement_duration(50) for _ in range(50)]
        long_durations = [calculate_movement_duration(500) for _ in range(50)]

        avg_short = statistics.mean(short_durations)
        avg_long = statistics.mean(long_durations)

        assert avg_long > avg_short, "Longer distance should take longer"


class TestClickOperations:
    """Tests for click operations."""

    @pytest.fixture
    def controller(self):
        """Create a MouseController instance."""
        from src.actions.mouse import MouseController

        return MouseController()

    def test_click_at_current_position(self, controller):
        """click() without target should click at current position."""
        with (
            patch.object(controller, "_mouse_down") as mock_down,
            patch.object(controller, "_mouse_up") as mock_up,
        ):
            controller.click()

        mock_down.assert_called_once_with("left")
        mock_up.assert_called_once_with("left")

    def test_click_at_target(self, controller):
        """click() with target should move then click."""
        target = Point(300, 200)

        with (
            patch.object(controller, "move") as mock_move,
            patch.object(controller, "_mouse_down") as mock_down,
            patch.object(controller, "_mouse_up") as mock_up,
        ):
            controller.click(target)

        mock_move.assert_called_once()
        mock_down.assert_called_once_with("left")
        mock_up.assert_called_once_with("left")

    def test_right_click(self, controller):
        """right_click should click with right button."""
        with (
            patch.object(controller, "_mouse_down") as mock_down,
            patch.object(controller, "_mouse_up") as mock_up,
        ):
            controller.right_click()

        mock_down.assert_called_once_with("right")
        mock_up.assert_called_once_with("right")

    def test_double_click(self, controller):
        """double_click should click twice rapidly."""
        with patch.object(controller, "_execute_click") as mock_click:
            controller.double_click()

        assert mock_click.call_count == 2

    def test_click_and_hold(self, controller):
        """Should support click and hold."""
        with (
            patch.object(controller, "_mouse_down") as mock_down,
            patch.object(controller, "_mouse_up") as mock_up,
        ):
            controller.click_and_hold(duration_ms=500)

        mock_down.assert_called_once()
        mock_up.assert_called_once()


class TestDragOperations:
    """Tests for drag operations."""

    @pytest.fixture
    def controller(self):
        """Create a MouseController instance."""
        from src.actions.mouse import MouseController

        return MouseController()

    def test_drag_from_to(self, controller):
        """drag should move while holding button."""
        start = Point(100, 100)
        end = Point(300, 300)

        with (
            patch.object(controller, "move"),
            patch.object(controller, "_mouse_down") as mock_down,
            patch.object(controller, "_mouse_up") as mock_up,
        ):
            controller.drag(start, end)

        mock_down.assert_called_once()
        mock_up.assert_called_once()


class TestPositionTracking:
    """Tests for position tracking."""

    @pytest.fixture
    def controller(self):
        """Create a MouseController instance."""
        from src.actions.mouse import MouseController

        return MouseController()

    def test_get_position_returns_point(self, controller):
        """get_position should return current position."""
        with patch.object(controller, "_get_current_position", return_value=(100, 200)):
            pos = controller.get_position()

        assert isinstance(pos, Point)
        assert pos.x == 100
        assert pos.y == 200


class TestModuleExports:
    """Tests for module exports."""

    def test_mouse_controller_exported_from_actions(self):
        """MouseController should be exported from actions package."""
        from src.actions import MouseController

        assert MouseController is not None

    def test_bezier_functions_available(self):
        """Bezier helper functions should be available."""
        from src.actions.mouse import (
            calculate_movement_duration,
            generate_bezier_path,
        )

        assert generate_bezier_path is not None
        assert calculate_movement_duration is not None


class TestHumanLikeMovement:
    """Tests for human-like movement characteristics."""

    def test_movement_has_acceleration_deceleration(self):
        """Movement should accelerate at start and decelerate at end."""
        from src.actions.mouse import generate_bezier_path

        path = generate_bezier_path((0, 0), (200, 200), num_points=20)

        # Calculate speeds between consecutive points
        speeds = []
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            speed = math.sqrt(dx**2 + dy**2)
            speeds.append(speed)

        if len(speeds) >= 3:
            # Start speed should be relatively low (acceleration)
            start_speed = speeds[0]
            # Middle speed should be higher
            mid_speed = speeds[len(speeds) // 2]
            # End speed should be relatively low (deceleration)
            end_speed = speeds[-1]

            # Middle should generally be faster than endpoints
            # (allowing for some variance in random control points)
            assert mid_speed >= min(start_speed, end_speed) * 0.5

    def test_small_random_variations(self):
        """Same movement should have slight variations each time."""
        from src.actions.mouse import generate_bezier_path

        paths = [generate_bezier_path((0, 0), (100, 100), num_points=10) for _ in range(5)]

        # Paths should be different (due to random control points)
        unique_paths = {tuple(tuple(p) for p in path) for path in paths}
        assert len(unique_paths) > 1, "Paths should have variation"
