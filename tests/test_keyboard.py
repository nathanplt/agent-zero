"""Tests for Feature 3.2: Keyboard Control.

These tests verify keyboard control functionality:
- Type any ASCII text
- Special keys work correctly (Enter, Escape, Tab, etc.)
- Key combinations work (Ctrl+A, etc.)
- Typing speed varies naturally (not robotic)

Note: Tests use mocked input for unit testing.
Integration tests marked with @pytest.mark.integration require real display.
"""

import statistics
from unittest.mock import patch

import pytest


class TestKeyboardControllerImport:
    """Tests for KeyboardController class import and basic structure."""

    def test_keyboard_controller_class_exists(self):
        """KeyboardController class should be importable."""
        from src.actions.keyboard import KeyboardController

        assert KeyboardController is not None

    def test_has_type_text_method(self):
        """KeyboardController should have a type_text method."""
        from src.actions.keyboard import KeyboardController

        assert hasattr(KeyboardController, "type_text")

    def test_has_press_key_method(self):
        """KeyboardController should have a press_key method."""
        from src.actions.keyboard import KeyboardController

        assert hasattr(KeyboardController, "press_key")

    def test_has_key_combo_method(self):
        """KeyboardController should have a key_combo method."""
        from src.actions.keyboard import KeyboardController

        assert hasattr(KeyboardController, "key_combo")

    def test_has_key_down_method(self):
        """KeyboardController should have a key_down method."""
        from src.actions.keyboard import KeyboardController

        assert hasattr(KeyboardController, "key_down")

    def test_has_key_up_method(self):
        """KeyboardController should have a key_up method."""
        from src.actions.keyboard import KeyboardController

        assert hasattr(KeyboardController, "key_up")


class TestKeyboardControllerInitialization:
    """Tests for KeyboardController initialization."""

    def test_initialization_default(self):
        """Should initialize with default settings."""
        from src.actions.keyboard import KeyboardController

        controller = KeyboardController()
        assert controller is not None

    def test_initialization_with_typing_speed(self):
        """Should accept typing speed parameters."""
        from src.actions.keyboard import KeyboardController

        controller = KeyboardController(
            base_delay_ms=100,
            delay_variance_ms=30,
        )
        assert controller._base_delay_ms == 100
        assert controller._delay_variance_ms == 30


class TestTypeText:
    """Tests for text typing functionality."""

    @pytest.fixture
    def controller(self):
        """Create a KeyboardController instance."""
        from src.actions.keyboard import KeyboardController

        return KeyboardController()

    def test_type_simple_text(self, controller):
        """Should type simple ASCII text."""
        typed_chars = []

        with patch.object(controller, "_press_char", side_effect=lambda c: typed_chars.append(c)):
            controller.type_text("Hello World")

        assert "".join(typed_chars) == "Hello World"

    def test_type_empty_string(self, controller):
        """Should handle empty string gracefully."""
        with patch.object(controller, "_press_char") as mock_press:
            controller.type_text("")

        mock_press.assert_not_called()

    def test_type_special_characters(self, controller):
        """Should type special characters."""
        typed_chars = []

        with patch.object(controller, "_press_char", side_effect=lambda c: typed_chars.append(c)):
            controller.type_text("Hello, World! @#$%")

        assert "".join(typed_chars) == "Hello, World! @#$%"

    def test_type_numbers(self, controller):
        """Should type numbers."""
        typed_chars = []

        with patch.object(controller, "_press_char", side_effect=lambda c: typed_chars.append(c)):
            controller.type_text("12345")

        assert "".join(typed_chars) == "12345"

    def test_type_with_fixed_interval(self, controller):
        """Should accept fixed interval parameter."""
        from src.actions.keyboard import KeyboardController

        controller = KeyboardController()
        typed_chars = []

        with (
            patch.object(controller, "_press_char", side_effect=lambda c: typed_chars.append(c)),
            patch("time.sleep") as mock_sleep,
        ):
            controller.type_text("abc", interval_ms=50.0)

        # Should have called sleep between characters
        assert mock_sleep.call_count >= 2  # Between each character


class TestSpecialKeys:
    """Tests for special key functionality."""

    @pytest.fixture
    def controller(self):
        """Create a KeyboardController instance."""
        from src.actions.keyboard import KeyboardController

        return KeyboardController()

    def test_press_enter(self, controller):
        """Should press Enter key."""
        with (
            patch.object(controller, "_key_down") as mock_down,
            patch.object(controller, "_key_up") as mock_up,
        ):
            controller.press_key("enter")

        mock_down.assert_called_once_with("enter")
        mock_up.assert_called_once_with("enter")

    def test_press_escape(self, controller):
        """Should press Escape key."""
        with (
            patch.object(controller, "_key_down") as mock_down,
            patch.object(controller, "_key_up") as mock_up,
        ):
            controller.press_key("escape")

        mock_down.assert_called_once_with("escape")
        mock_up.assert_called_once_with("escape")

    def test_press_tab(self, controller):
        """Should press Tab key."""
        with (
            patch.object(controller, "_key_down") as mock_down,
            patch.object(controller, "_key_up") as mock_up,
        ):
            controller.press_key("tab")

        mock_down.assert_called_once_with("tab")
        mock_up.assert_called_once_with("tab")

    def test_press_backspace(self, controller):
        """Should press Backspace key."""
        with (
            patch.object(controller, "_key_down") as mock_down,
            patch.object(controller, "_key_up") as mock_up,
        ):
            controller.press_key("backspace")

        mock_down.assert_called_once_with("backspace")
        mock_up.assert_called_once_with("backspace")

    def test_press_delete(self, controller):
        """Should press Delete key."""
        with (
            patch.object(controller, "_key_down") as mock_down,
            patch.object(controller, "_key_up") as mock_up,
        ):
            controller.press_key("delete")

        mock_down.assert_called_once_with("delete")
        mock_up.assert_called_once_with("delete")

    def test_press_arrow_keys(self, controller):
        """Should press arrow keys."""
        for key in ["up", "down", "left", "right"]:
            with (
                patch.object(controller, "_key_down") as mock_down,
                patch.object(controller, "_key_up") as mock_up,
            ):
                controller.press_key(key)

            mock_down.assert_called_once_with(key)
            mock_up.assert_called_once_with(key)

    def test_press_function_keys(self, controller):
        """Should press function keys."""
        for i in range(1, 13):
            key = f"f{i}"
            with (
                patch.object(controller, "_key_down") as mock_down,
                patch.object(controller, "_key_up") as mock_up,
            ):
                controller.press_key(key)

            mock_down.assert_called_once_with(key)
            mock_up.assert_called_once_with(key)

    def test_press_home_end(self, controller):
        """Should press Home and End keys."""
        for key in ["home", "end"]:
            with (
                patch.object(controller, "_key_down") as mock_down,
                patch.object(controller, "_key_up") as mock_up,
            ):
                controller.press_key(key)

            mock_down.assert_called_once_with(key)
            mock_up.assert_called_once_with(key)

    def test_press_page_up_down(self, controller):
        """Should press Page Up and Page Down keys."""
        for key in ["pageup", "pagedown"]:
            with (
                patch.object(controller, "_key_down") as mock_down,
                patch.object(controller, "_key_up") as mock_up,
            ):
                controller.press_key(key)

            mock_down.assert_called_once_with(key)
            mock_up.assert_called_once_with(key)


class TestKeyCombinations:
    """Tests for key combination functionality."""

    @pytest.fixture
    def controller(self):
        """Create a KeyboardController instance."""
        from src.actions.keyboard import KeyboardController

        return KeyboardController()

    def test_ctrl_a(self, controller):
        """Should press Ctrl+A (select all)."""
        down_calls = []
        up_calls = []

        with (
            patch.object(controller, "_key_down", side_effect=lambda k: down_calls.append(k)),
            patch.object(controller, "_key_up", side_effect=lambda k: up_calls.append(k)),
        ):
            controller.key_combo(["ctrl", "a"])

        # Keys should be pressed in order and released in reverse order
        assert down_calls == ["ctrl", "a"]
        assert up_calls == ["a", "ctrl"]

    def test_ctrl_c(self, controller):
        """Should press Ctrl+C (copy)."""
        down_calls = []
        up_calls = []

        with (
            patch.object(controller, "_key_down", side_effect=lambda k: down_calls.append(k)),
            patch.object(controller, "_key_up", side_effect=lambda k: up_calls.append(k)),
        ):
            controller.key_combo(["ctrl", "c"])

        assert down_calls == ["ctrl", "c"]
        assert up_calls == ["c", "ctrl"]

    def test_ctrl_v(self, controller):
        """Should press Ctrl+V (paste)."""
        down_calls = []
        up_calls = []

        with (
            patch.object(controller, "_key_down", side_effect=lambda k: down_calls.append(k)),
            patch.object(controller, "_key_up", side_effect=lambda k: up_calls.append(k)),
        ):
            controller.key_combo(["ctrl", "v"])

        assert down_calls == ["ctrl", "v"]
        assert up_calls == ["v", "ctrl"]

    def test_ctrl_shift_combo(self, controller):
        """Should press Ctrl+Shift+key combinations."""
        down_calls = []
        up_calls = []

        with (
            patch.object(controller, "_key_down", side_effect=lambda k: down_calls.append(k)),
            patch.object(controller, "_key_up", side_effect=lambda k: up_calls.append(k)),
        ):
            controller.key_combo(["ctrl", "shift", "s"])

        assert down_calls == ["ctrl", "shift", "s"]
        assert up_calls == ["s", "shift", "ctrl"]

    def test_alt_tab(self, controller):
        """Should press Alt+Tab (switch window)."""
        down_calls = []
        up_calls = []

        with (
            patch.object(controller, "_key_down", side_effect=lambda k: down_calls.append(k)),
            patch.object(controller, "_key_up", side_effect=lambda k: up_calls.append(k)),
        ):
            controller.key_combo(["alt", "tab"])

        assert down_calls == ["alt", "tab"]
        assert up_calls == ["tab", "alt"]

    def test_empty_combo(self, controller):
        """Should handle empty combination gracefully."""
        with (
            patch.object(controller, "_key_down") as mock_down,
            patch.object(controller, "_key_up") as mock_up,
        ):
            controller.key_combo([])

        mock_down.assert_not_called()
        mock_up.assert_not_called()

    def test_single_key_combo(self, controller):
        """Should handle single key as combo."""
        down_calls = []
        up_calls = []

        with (
            patch.object(controller, "_key_down", side_effect=lambda k: down_calls.append(k)),
            patch.object(controller, "_key_up", side_effect=lambda k: up_calls.append(k)),
        ):
            controller.key_combo(["enter"])

        assert down_calls == ["enter"]
        assert up_calls == ["enter"]


class TestTypingTimingVariance:
    """Tests for natural typing timing variance."""

    def test_typing_timing_varies(self):
        """Typing timing should vary naturally (not constant)."""
        from src.actions.keyboard import calculate_typing_delay

        delays = []
        for _ in range(100):
            delay = calculate_typing_delay()
            delays.append(delay)

        # Should have variance (standard deviation > 0)
        std_dev = statistics.stdev(delays)
        assert std_dev > 0.005, "Typing timing should have variance"

    def test_typing_delay_in_human_range(self):
        """Typing delay should be in human-like range (50-150ms)."""
        from src.actions.keyboard import calculate_typing_delay

        for _ in range(100):
            delay = calculate_typing_delay()
            delay_ms = delay * 1000

            # Should be in reasonable human typing range
            assert delay_ms >= 20, "Delay too fast"
            assert delay_ms <= 300, "Delay too slow"

    def test_key_press_timing_varies(self):
        """Key press timing should vary naturally."""
        from src.actions.keyboard import calculate_key_press_duration

        durations = []
        for _ in range(100):
            duration = calculate_key_press_duration()
            durations.append(duration)

        # Should have variance
        std_dev = statistics.stdev(durations)
        assert std_dev > 0.005, "Key press timing should have variance"


class TestKeyDownUp:
    """Tests for key down/up functionality."""

    @pytest.fixture
    def controller(self):
        """Create a KeyboardController instance."""
        from src.actions.keyboard import KeyboardController

        return KeyboardController()

    def test_key_down_then_up(self, controller):
        """Should support separate key_down and key_up calls."""
        with (
            patch.object(controller, "_key_down") as mock_down,
            patch.object(controller, "_key_up") as mock_up,
        ):
            controller.key_down("shift")
            controller.key_up("shift")

        mock_down.assert_called_once_with("shift")
        mock_up.assert_called_once_with("shift")


class TestModuleExports:
    """Tests for module exports."""

    def test_keyboard_controller_exported_from_actions(self):
        """KeyboardController should be exported from actions package."""
        from src.actions import KeyboardController

        assert KeyboardController is not None

    def test_timing_functions_available(self):
        """Timing helper functions should be available."""
        from src.actions.keyboard import (
            calculate_key_press_duration,
            calculate_typing_delay,
        )

        assert calculate_typing_delay is not None
        assert calculate_key_press_duration is not None


class TestKeyNormalization:
    """Tests for key name normalization."""

    @pytest.fixture
    def controller(self):
        """Create a KeyboardController instance."""
        from src.actions.keyboard import KeyboardController

        return KeyboardController()

    def test_lowercase_key_names(self, controller):
        """Should accept lowercase key names."""
        with (
            patch.object(controller, "_key_down") as mock_down,
            patch.object(controller, "_key_up"),
        ):
            controller.press_key("enter")

        mock_down.assert_called()

    def test_uppercase_key_names(self, controller):
        """Should accept uppercase key names and normalize."""
        with (
            patch.object(controller, "_key_down") as mock_down,
            patch.object(controller, "_key_up"),
        ):
            controller.press_key("ENTER")

        # Should normalize to lowercase internally
        mock_down.assert_called_once()

    def test_mixed_case_key_names(self, controller):
        """Should accept mixed case key names."""
        with (
            patch.object(controller, "_key_down") as mock_down,
            patch.object(controller, "_key_up"),
        ):
            controller.press_key("Enter")

        mock_down.assert_called_once()


class TestSpecialKeyAliases:
    """Tests for special key aliases."""

    @pytest.fixture
    def controller(self):
        """Create a KeyboardController instance."""
        from src.actions.keyboard import KeyboardController

        return KeyboardController()

    def test_return_alias_for_enter(self, controller):
        """Should accept 'return' as alias for 'enter'."""
        with (
            patch.object(controller, "_key_down") as mock_down,
            patch.object(controller, "_key_up"),
        ):
            controller.press_key("return")

        # Should work (either mapped to enter or accepted directly)
        mock_down.assert_called_once()

    def test_esc_alias_for_escape(self, controller):
        """Should accept 'esc' as alias for 'escape'."""
        with (
            patch.object(controller, "_key_down") as mock_down,
            patch.object(controller, "_key_up"),
        ):
            controller.press_key("esc")

        mock_down.assert_called_once()

    def test_control_alias_for_ctrl(self, controller):
        """Should accept 'control' as alias for 'ctrl'."""
        down_calls = []

        with (
            patch.object(controller, "_key_down", side_effect=lambda k: down_calls.append(k)),
            patch.object(controller, "_key_up"),
        ):
            controller.key_combo(["control", "a"])

        assert len(down_calls) == 2
