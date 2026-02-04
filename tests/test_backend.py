"""Tests for InputBackend interface and implementations.

These tests verify:
- InputBackend interface is properly defined
- NullInputBackend does nothing (for testing)
- PlaywrightInputBackend calls correct Playwright methods
- Key mapping works correctly
"""

from unittest.mock import MagicMock

import pytest


class TestInputBackendInterface:
    """Tests for InputBackend abstract interface."""

    def test_input_backend_is_abstract(self):
        """InputBackend should be abstract and not instantiable."""
        from src.actions.backend import InputBackend

        with pytest.raises(TypeError):
            InputBackend()  # type: ignore

    def test_input_backend_has_keyboard_methods(self):
        """InputBackend should define keyboard methods."""
        from src.actions.backend import InputBackend

        assert hasattr(InputBackend, "key_down")
        assert hasattr(InputBackend, "key_up")
        assert hasattr(InputBackend, "type_char")

    def test_input_backend_has_mouse_methods(self):
        """InputBackend should define mouse methods."""
        from src.actions.backend import InputBackend

        assert hasattr(InputBackend, "mouse_move")
        assert hasattr(InputBackend, "mouse_down")
        assert hasattr(InputBackend, "mouse_up")
        assert hasattr(InputBackend, "mouse_click")
        assert hasattr(InputBackend, "scroll")


class TestNullInputBackend:
    """Tests for NullInputBackend (no-op implementation)."""

    @pytest.fixture
    def backend(self):
        """Create a NullInputBackend instance."""
        from src.actions.backend import NullInputBackend

        return NullInputBackend()

    def test_key_down_does_nothing(self, backend):
        """key_down should not raise."""
        backend.key_down("a")
        backend.key_down("Enter")
        backend.key_down("Control")

    def test_key_up_does_nothing(self, backend):
        """key_up should not raise."""
        backend.key_up("a")
        backend.key_up("Enter")

    def test_type_char_does_nothing(self, backend):
        """type_char should not raise."""
        backend.type_char("a")
        backend.type_char("!")

    def test_mouse_move_does_nothing(self, backend):
        """mouse_move should not raise."""
        backend.mouse_move(100, 200)

    def test_mouse_down_does_nothing(self, backend):
        """mouse_down should not raise."""
        backend.mouse_down("left")
        backend.mouse_down("right")

    def test_mouse_up_does_nothing(self, backend):
        """mouse_up should not raise."""
        backend.mouse_up("left")

    def test_mouse_click_does_nothing(self, backend):
        """mouse_click should not raise."""
        backend.mouse_click(100, 200, "left")

    def test_scroll_does_nothing(self, backend):
        """scroll should not raise."""
        backend.scroll(100, 200, 0, -100)

    def test_implements_input_backend(self, backend):
        """NullInputBackend should implement InputBackend."""
        from src.actions.backend import InputBackend

        assert isinstance(backend, InputBackend)


class TestPlaywrightInputBackend:
    """Tests for PlaywrightInputBackend."""

    @pytest.fixture
    def mock_page(self):
        """Create a mock Playwright page."""
        page = MagicMock()
        page.keyboard = MagicMock()
        page.mouse = MagicMock()
        return page

    @pytest.fixture
    def backend(self, mock_page):
        """Create a PlaywrightInputBackend with mock page."""
        from src.actions.backend import PlaywrightInputBackend

        return PlaywrightInputBackend(mock_page)

    def test_key_down_calls_playwright(self, backend, mock_page):
        """key_down should call page.keyboard.down."""
        backend.key_down("a")
        mock_page.keyboard.down.assert_called_once_with("a")

    def test_key_up_calls_playwright(self, backend, mock_page):
        """key_up should call page.keyboard.up."""
        backend.key_up("a")
        mock_page.keyboard.up.assert_called_once_with("a")

    def test_type_char_calls_playwright(self, backend, mock_page):
        """type_char should call page.keyboard.type."""
        backend.type_char("a")
        mock_page.keyboard.type.assert_called_once_with("a")

    def test_mouse_move_calls_playwright(self, backend, mock_page):
        """mouse_move should call page.mouse.move."""
        backend.mouse_move(100, 200)
        mock_page.mouse.move.assert_called_once_with(100, 200)

    def test_mouse_down_calls_playwright(self, backend, mock_page):
        """mouse_down should call page.mouse.down."""
        backend.mouse_down("left")
        mock_page.mouse.down.assert_called_once_with(button="left")

    def test_mouse_up_calls_playwright(self, backend, mock_page):
        """mouse_up should call page.mouse.up."""
        backend.mouse_up("left")
        mock_page.mouse.up.assert_called_once_with(button="left")

    def test_mouse_click_calls_playwright(self, backend, mock_page):
        """mouse_click should call page.mouse.click."""
        backend.mouse_click(100, 200, "left")
        mock_page.mouse.click.assert_called_once_with(100, 200, button="left")

    def test_scroll_calls_playwright(self, backend, mock_page):
        """scroll should move then wheel."""
        backend.scroll(100, 200, 0, -100)
        mock_page.mouse.move.assert_called_once_with(100, 200)
        mock_page.mouse.wheel.assert_called_once_with(0, -100)

    def test_implements_input_backend(self, backend):
        """PlaywrightInputBackend should implement InputBackend."""
        from src.actions.backend import InputBackend

        assert isinstance(backend, InputBackend)


class TestKeyMapping:
    """Tests for key name mapping to Playwright format."""

    def test_ctrl_maps_to_control(self):
        """'ctrl' should map to 'Control'."""
        from src.actions.backend import _to_playwright_key

        assert _to_playwright_key("ctrl") == "Control"
        assert _to_playwright_key("CTRL") == "Control"
        assert _to_playwright_key("control") == "Control"

    def test_enter_maps_correctly(self):
        """'enter' and 'return' should map to 'Enter'."""
        from src.actions.backend import _to_playwright_key

        assert _to_playwright_key("enter") == "Enter"
        assert _to_playwright_key("return") == "Enter"
        assert _to_playwright_key("ENTER") == "Enter"

    def test_escape_maps_correctly(self):
        """'escape' and 'esc' should map to 'Escape'."""
        from src.actions.backend import _to_playwright_key

        assert _to_playwright_key("escape") == "Escape"
        assert _to_playwright_key("esc") == "Escape"

    def test_arrow_keys_map_correctly(self):
        """Arrow keys should map to Playwright format."""
        from src.actions.backend import _to_playwright_key

        assert _to_playwright_key("up") == "ArrowUp"
        assert _to_playwright_key("down") == "ArrowDown"
        assert _to_playwright_key("left") == "ArrowLeft"
        assert _to_playwright_key("right") == "ArrowRight"

    def test_function_keys_map_correctly(self):
        """Function keys should map to uppercase."""
        from src.actions.backend import _to_playwright_key

        assert _to_playwright_key("f1") == "F1"
        assert _to_playwright_key("f12") == "F12"

    def test_space_maps_correctly(self):
        """'space' should map to ' '."""
        from src.actions.backend import _to_playwright_key

        assert _to_playwright_key("space") == " "
        assert _to_playwright_key("spacebar") == " "

    def test_unknown_key_passes_through(self):
        """Unknown keys should pass through unchanged."""
        from src.actions.backend import _to_playwright_key

        assert _to_playwright_key("a") == "a"
        assert _to_playwright_key("Z") == "Z"
        assert _to_playwright_key("1") == "1"

    def test_playwright_backend_uses_key_mapping(self):
        """PlaywrightInputBackend should use key mapping."""
        from src.actions.backend import PlaywrightInputBackend

        mock_page = MagicMock()
        backend = PlaywrightInputBackend(mock_page)

        backend.key_down("ctrl")
        mock_page.keyboard.down.assert_called_with("Control")

        backend.key_down("enter")
        mock_page.keyboard.down.assert_called_with("Enter")


class TestModuleExports:
    """Tests for module exports."""

    def test_input_backend_exported(self):
        """InputBackend should be exported from actions package."""
        from src.actions import InputBackend

        assert InputBackend is not None

    def test_null_input_backend_exported(self):
        """NullInputBackend should be exported from actions package."""
        from src.actions import NullInputBackend

        assert NullInputBackend is not None

    def test_playwright_input_backend_exported(self):
        """PlaywrightInputBackend should be exported from actions package."""
        from src.actions import PlaywrightInputBackend

        assert PlaywrightInputBackend is not None


class TestControllerBackendIntegration:
    """Tests for controller integration with backends."""

    def test_mouse_controller_accepts_backend(self):
        """MouseController should accept a backend parameter."""
        from src.actions import MouseController, NullInputBackend

        backend = NullInputBackend()
        controller = MouseController(backend=backend)
        assert controller._backend is backend

    def test_keyboard_controller_accepts_backend(self):
        """KeyboardController should accept a backend parameter."""
        from src.actions import KeyboardController, NullInputBackend

        backend = NullInputBackend()
        controller = KeyboardController(backend=backend)
        assert controller._backend is backend

    def test_mouse_controller_uses_null_backend_by_default(self):
        """MouseController should use NullInputBackend by default."""
        from src.actions import MouseController
        from src.actions.backend import NullInputBackend

        controller = MouseController()
        assert isinstance(controller._backend, NullInputBackend)

    def test_keyboard_controller_uses_null_backend_by_default(self):
        """KeyboardController should use NullInputBackend by default."""
        from src.actions import KeyboardController
        from src.actions.backend import NullInputBackend

        controller = KeyboardController()
        assert isinstance(controller._backend, NullInputBackend)

    def test_mouse_controller_calls_backend_on_click(self):
        """MouseController should call backend methods on click."""
        from src.actions import MouseController

        mock_backend = MagicMock()
        controller = MouseController(backend=mock_backend)

        # Set position first
        controller._current_position = (100, 200)

        # Click should call backend
        controller._mouse_down("left")
        mock_backend.mouse_down.assert_called_with("left")

        controller._mouse_up("left")
        mock_backend.mouse_up.assert_called_with("left")

    def test_keyboard_controller_calls_backend_on_key(self):
        """KeyboardController should call backend methods on key press."""
        from src.actions import KeyboardController

        mock_backend = MagicMock()
        controller = KeyboardController(backend=mock_backend)

        controller._key_down("enter")
        mock_backend.key_down.assert_called_with("enter")

        controller._key_up("enter")
        mock_backend.key_up.assert_called_with("enter")

    def test_keyboard_controller_calls_backend_on_type(self):
        """KeyboardController should call backend on type_char."""
        from src.actions import KeyboardController

        mock_backend = MagicMock()
        controller = KeyboardController(backend=mock_backend)

        controller._press_char("a")
        mock_backend.type_char.assert_called_with("a")
