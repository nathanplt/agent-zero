"""Tests for Feature 1.3: Environment Manager.

These tests verify the Python API to start/stop/manage the container environment:
- EnvironmentManager.start() launches container
- EnvironmentManager.stop() cleanly shuts down
- EnvironmentManager.status() returns health info
- EnvironmentManager.screenshot() returns current frame
- Handles crashes gracefully with auto-restart
- Double start is no-op

Note: Some tests mock external dependencies to run without Docker.
Tests marked with @pytest.mark.docker require Docker to be running.
"""

import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from src.environment.browser import BrowserRuntime
from src.interfaces.environment import EnvironmentStatus


# Mark for tests requiring Docker
def _docker_available() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True)
        return result.returncode == 0
    except (FileNotFoundError, OSError):
        return False

docker = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker not available",
)

# Mark for tests requiring Playwright
try:
    import playwright  # noqa: F401
    _has_playwright = True
except ImportError:
    _has_playwright = False

requires_playwright = pytest.mark.skipif(
    not _has_playwright,
    reason="Playwright not installed",
)


class TestVirtualDisplay:
    """Tests for VirtualDisplay class."""

    def test_virtual_display_class_exists(self):
        """VirtualDisplay class should be importable."""
        from src.environment.display import VirtualDisplay

        assert VirtualDisplay is not None

    def test_virtual_display_has_start_method(self):
        """VirtualDisplay should have a start method."""
        from src.environment.display import VirtualDisplay

        assert hasattr(VirtualDisplay, "start")

    def test_virtual_display_has_stop_method(self):
        """VirtualDisplay should have a stop method."""
        from src.environment.display import VirtualDisplay

        assert hasattr(VirtualDisplay, "stop")

    def test_virtual_display_has_is_running_property(self):
        """VirtualDisplay should have is_running property."""
        from src.environment.display import VirtualDisplay

        assert hasattr(VirtualDisplay, "is_running")

    def test_virtual_display_has_screenshot_method(self):
        """VirtualDisplay should have screenshot method."""
        from src.environment.display import VirtualDisplay

        assert hasattr(VirtualDisplay, "screenshot")

    def test_virtual_display_initialization(self):
        """VirtualDisplay should initialize with configurable display number."""
        from src.environment.display import VirtualDisplay

        display = VirtualDisplay(display=":99", width=1920, height=1080)
        assert display.display == ":99"
        assert display.width == 1920
        assert display.height == 1080
        assert not display.is_running


class TestLocalEnvironmentManagerImport:
    """Tests for LocalEnvironmentManager class import and basic structure."""

    def test_local_environment_manager_exists(self):
        """LocalEnvironmentManager should be importable."""
        from src.environment.manager import LocalEnvironmentManager

        assert LocalEnvironmentManager is not None

    def test_implements_environment_manager_interface(self):
        """LocalEnvironmentManager should implement EnvironmentManager interface."""
        from src.environment.manager import LocalEnvironmentManager
        from src.interfaces.environment import EnvironmentManager

        assert issubclass(LocalEnvironmentManager, EnvironmentManager)

    def test_has_required_methods(self):
        """LocalEnvironmentManager should have all required methods."""
        from src.environment.manager import LocalEnvironmentManager

        required_methods = [
            "start",
            "stop",
            "restart",
            "status",
            "screenshot",
            "screenshot_pil",
            "navigate",
            "is_running",
            "wait_for_ready",
            "get_display_size",
        ]
        for method in required_methods:
            assert hasattr(LocalEnvironmentManager, method), f"Missing method: {method}"


class TestLocalEnvironmentManagerUnit:
    """Unit tests for LocalEnvironmentManager with mocked dependencies."""

    @pytest.fixture
    def mock_browser(self):
        """Create a mock browser runtime."""
        mock = MagicMock(spec=BrowserRuntime)
        mock.is_running = False
        mock.get_viewport_size.return_value = (1920, 1080)
        return mock

    @pytest.fixture
    def mock_display(self):
        """Create a mock virtual display."""
        from src.environment.display import VirtualDisplay

        mock = MagicMock(spec=VirtualDisplay)
        mock.is_running = False
        mock.width = 1920
        mock.height = 1080
        return mock

    def test_initial_status_is_stopped(self, mock_browser, mock_display):
        """Status should be STOPPED before start() is called."""
        from src.environment.manager import LocalEnvironmentManager

        with patch.object(
            LocalEnvironmentManager, "_create_display", return_value=mock_display
        ), patch.object(
            LocalEnvironmentManager, "_create_browser", return_value=mock_browser
        ):
            manager = LocalEnvironmentManager(headless=True)
            health = manager.status()
            assert health.status == EnvironmentStatus.STOPPED

    def test_is_running_false_initially(self, mock_browser, mock_display):
        """is_running() should return False before start()."""
        from src.environment.manager import LocalEnvironmentManager

        with patch.object(
            LocalEnvironmentManager, "_create_display", return_value=mock_display
        ), patch.object(
            LocalEnvironmentManager, "_create_browser", return_value=mock_browser
        ):
            manager = LocalEnvironmentManager(headless=True)
            assert not manager.is_running()

    def test_start_starts_display_and_browser(self, mock_browser, mock_display):
        """start() should start both display and browser."""
        from src.environment.manager import LocalEnvironmentManager

        with patch.object(
            LocalEnvironmentManager, "_create_display", return_value=mock_display
        ), patch.object(
            LocalEnvironmentManager, "_create_browser", return_value=mock_browser
        ):
            # Use headless=False to test display + browser coordination
            manager = LocalEnvironmentManager(headless=False)
            # Make is_running return True after start is called
            mock_display.is_running = True
            mock_browser.is_running = True

            manager.start()

            mock_display.start.assert_called_once()
            mock_browser.start.assert_called_once()

    def test_start_headed_without_virtual_display_skips_display_start(self, mock_browser, mock_display):
        """Headed native display mode should not attempt virtual-display startup."""
        from src.environment.manager import LocalEnvironmentManager

        mock_browser.is_running = True

        with patch.object(
            LocalEnvironmentManager, "_create_display", return_value=mock_display
        ) as create_display, patch.object(
            LocalEnvironmentManager, "_create_browser", return_value=mock_browser
        ):
            manager = LocalEnvironmentManager(headless=False, use_virtual_display=False)
            manager.start()

            create_display.assert_not_called()
            mock_display.start.assert_not_called()
            mock_browser.start.assert_called_once()
            assert manager.is_running()

    def test_status_running_after_start(self, mock_browser, mock_display):
        """status() should return RUNNING after successful start()."""
        from src.environment.manager import LocalEnvironmentManager

        mock_display.is_running = True
        mock_browser.is_running = True

        with patch.object(
            LocalEnvironmentManager, "_create_display", return_value=mock_display
        ), patch.object(
            LocalEnvironmentManager, "_create_browser", return_value=mock_browser
        ):
            manager = LocalEnvironmentManager(headless=True)
            manager.start()

            health = manager.status()
            assert health.status == EnvironmentStatus.RUNNING
            assert health.browser_active
            assert health.display_active

    def test_stop_stops_browser_and_display(self, mock_browser, mock_display):
        """stop() should stop browser and display."""
        from src.environment.manager import LocalEnvironmentManager

        mock_display.is_running = True
        mock_browser.is_running = True

        with patch.object(
            LocalEnvironmentManager, "_create_display", return_value=mock_display
        ), patch.object(
            LocalEnvironmentManager, "_create_browser", return_value=mock_browser
        ):
            manager = LocalEnvironmentManager(headless=True)
            manager.start()

            # Simulate stopping
            mock_display.is_running = False
            mock_browser.is_running = False

            manager.stop()

            mock_browser.stop.assert_called_once()
            mock_display.stop.assert_called_once()

    def test_double_start_is_noop(self, mock_browser, mock_display):
        """Calling start() twice should not restart components."""
        from src.environment.manager import LocalEnvironmentManager

        mock_display.is_running = True
        mock_browser.is_running = True

        with patch.object(
            LocalEnvironmentManager, "_create_display", return_value=mock_display
        ), patch.object(
            LocalEnvironmentManager, "_create_browser", return_value=mock_browser
        ):
            # Use headless=False to test display + browser coordination
            manager = LocalEnvironmentManager(headless=False)
            manager.start()
            manager.start()  # Second call

            # Should only be called once
            assert mock_display.start.call_count == 1
            assert mock_browser.start.call_count == 1

    def test_screenshot_returns_bytes(self, mock_browser, mock_display):
        """screenshot() should return PNG bytes."""
        from src.environment.manager import LocalEnvironmentManager

        mock_display.is_running = True
        mock_browser.is_running = True
        # Return fake PNG bytes
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mock_browser.screenshot.return_value = fake_png

        with patch.object(
            LocalEnvironmentManager, "_create_display", return_value=mock_display
        ), patch.object(
            LocalEnvironmentManager, "_create_browser", return_value=mock_browser
        ):
            manager = LocalEnvironmentManager(headless=True)
            manager.start()

            result = manager.screenshot()

            assert isinstance(result, bytes)
            assert result.startswith(b"\x89PNG")

    def test_screenshot_raises_when_not_running(self, mock_browser, mock_display):
        """screenshot() should raise when environment not running."""
        from src.environment.manager import LocalEnvironmentManager
        from src.interfaces.environment import EnvironmentSetupError

        mock_display.is_running = False
        mock_browser.is_running = False

        with patch.object(
            LocalEnvironmentManager, "_create_display", return_value=mock_display
        ), patch.object(
            LocalEnvironmentManager, "_create_browser", return_value=mock_browser
        ):
            manager = LocalEnvironmentManager(headless=True)

            with pytest.raises(EnvironmentSetupError):
                manager.screenshot()

    def test_navigate_calls_browser_navigate(self, mock_browser, mock_display):
        """navigate() should call browser's navigate method."""
        from src.environment.manager import LocalEnvironmentManager

        mock_display.is_running = True
        mock_browser.is_running = True

        with patch.object(
            LocalEnvironmentManager, "_create_display", return_value=mock_display
        ), patch.object(
            LocalEnvironmentManager, "_create_browser", return_value=mock_browser
        ):
            manager = LocalEnvironmentManager(headless=True)
            manager.start()

            manager.navigate("https://example.com")
            mock_browser.navigate.assert_called_once_with("https://example.com")

    def test_get_browser_runtime_returns_public_browser_reference(self, mock_browser, mock_display):
        """Public browser accessor should avoid private reach-through in CLI."""
        from src.environment.manager import LocalEnvironmentManager

        mock_display.is_running = True
        mock_browser.is_running = True

        with patch.object(
            LocalEnvironmentManager, "_create_display", return_value=mock_display
        ), patch.object(
            LocalEnvironmentManager, "_create_browser", return_value=mock_browser
        ):
            manager = LocalEnvironmentManager(headless=True)
            manager.start()

            assert manager.get_browser_runtime() is mock_browser

    def test_get_display_size_returns_tuple(self, mock_browser, mock_display):
        """get_display_size() should return (width, height) tuple."""
        from src.environment.manager import LocalEnvironmentManager

        mock_display.width = 1920
        mock_display.height = 1080

        with patch.object(
            LocalEnvironmentManager, "_create_display", return_value=mock_display
        ), patch.object(
            LocalEnvironmentManager, "_create_browser", return_value=mock_browser
        ):
            manager = LocalEnvironmentManager(headless=True)

            size = manager.get_display_size()

            assert size == (1920, 1080)

    def test_restart_stops_and_starts(self, mock_browser, mock_display):
        """restart() should stop and then start the environment."""
        from src.environment.manager import LocalEnvironmentManager

        mock_display.is_running = True
        mock_browser.is_running = True

        with patch.object(
            LocalEnvironmentManager, "_create_display", return_value=mock_display
        ), patch.object(
            LocalEnvironmentManager, "_create_browser", return_value=mock_browser
        ):
            # Use headless=False to test display + browser coordination
            manager = LocalEnvironmentManager(headless=False)
            manager.start()

            # Reset mocks for restart
            mock_browser.reset_mock()
            mock_display.reset_mock()
            mock_display.is_running = True
            mock_browser.is_running = True

            manager.restart()

            # Stop should be called before start
            mock_browser.stop.assert_called_once()
            mock_display.stop.assert_called_once()
            mock_display.start.assert_called_once()
            mock_browser.start.assert_called_once()

    def test_status_includes_uptime(self, mock_browser, mock_display):
        """status() should include uptime_seconds when running."""
        from src.environment.manager import LocalEnvironmentManager

        mock_display.is_running = True
        mock_browser.is_running = True

        with patch.object(
            LocalEnvironmentManager, "_create_display", return_value=mock_display
        ), patch.object(
            LocalEnvironmentManager, "_create_browser", return_value=mock_browser
        ):
            manager = LocalEnvironmentManager(headless=True)
            manager.start()

            # Wait a bit
            time.sleep(0.1)

            health = manager.status()

            assert health.uptime_seconds >= 0.1

    def test_wait_for_ready_returns_true_when_running(self, mock_browser, mock_display):
        """wait_for_ready() should return True when environment is running."""
        from src.environment.manager import LocalEnvironmentManager

        mock_display.is_running = True
        mock_browser.is_running = True

        with patch.object(
            LocalEnvironmentManager, "_create_display", return_value=mock_display
        ), patch.object(
            LocalEnvironmentManager, "_create_browser", return_value=mock_browser
        ):
            manager = LocalEnvironmentManager(headless=True)
            manager.start()

            result = manager.wait_for_ready(timeout_seconds=1.0)

            assert result is True

    def test_wait_for_ready_returns_false_on_timeout(self, mock_browser, mock_display):
        """wait_for_ready() should return False if not ready within timeout."""
        from src.environment.manager import LocalEnvironmentManager

        mock_display.is_running = False
        mock_browser.is_running = False

        with patch.object(
            LocalEnvironmentManager, "_create_display", return_value=mock_display
        ), patch.object(
            LocalEnvironmentManager, "_create_browser", return_value=mock_browser
        ):
            manager = LocalEnvironmentManager(headless=True)

            result = manager.wait_for_ready(timeout_seconds=0.1)

            assert result is False


class TestEnvironmentManagerContextManager:
    """Tests for context manager interface."""

    def test_context_manager_starts_and_stops(self):
        """Context manager should start on enter and stop on exit."""
        from src.environment.manager import LocalEnvironmentManager

        mock_browser = MagicMock(spec=BrowserRuntime)
        mock_browser.is_running = True
        mock_browser.get_viewport_size.return_value = (1920, 1080)

        mock_display = MagicMock()
        mock_display.is_running = True
        mock_display.width = 1920
        mock_display.height = 1080

        with patch.object(
            LocalEnvironmentManager, "_create_display", return_value=mock_display
        ), patch.object(
            LocalEnvironmentManager, "_create_browser", return_value=mock_browser
        ):
            with LocalEnvironmentManager(headless=True) as manager:
                assert manager.is_running()

            mock_browser.stop.assert_called()
            mock_display.stop.assert_called()


class TestEnvironmentManagerAutoRestart:
    """Tests for auto-restart functionality."""

    def test_status_shows_crashed_when_browser_dies(self):
        """status() should show CRASHED if browser unexpectedly stops."""
        from src.environment.manager import LocalEnvironmentManager

        mock_browser = MagicMock(spec=BrowserRuntime)
        mock_browser.is_running = True
        mock_browser.get_viewport_size.return_value = (1920, 1080)

        mock_display = MagicMock()
        mock_display.is_running = True
        mock_display.width = 1920
        mock_display.height = 1080

        with patch.object(
            LocalEnvironmentManager, "_create_display", return_value=mock_display
        ), patch.object(
            LocalEnvironmentManager, "_create_browser", return_value=mock_browser
        ):
            manager = LocalEnvironmentManager(headless=True)
            manager.start()

            # Simulate browser crash
            mock_browser.is_running = False

            health = manager.status()

            assert health.status == EnvironmentStatus.CRASHED

    def test_auto_restart_on_crash_detection(self):
        """Manager should attempt auto-restart when crash is detected."""
        from src.environment.manager import LocalEnvironmentManager

        mock_browser = MagicMock(spec=BrowserRuntime)
        mock_browser.is_running = True
        mock_browser.get_viewport_size.return_value = (1920, 1080)

        mock_display = MagicMock()
        mock_display.is_running = True
        mock_display.width = 1920
        mock_display.height = 1080

        with patch.object(
            LocalEnvironmentManager, "_create_display", return_value=mock_display
        ), patch.object(
            LocalEnvironmentManager, "_create_browser", return_value=mock_browser
        ):
            manager = LocalEnvironmentManager(headless=True, auto_restart=True)
            manager.start()

            # Simulate crash
            mock_browser.is_running = False

            # Check status triggers auto-restart
            mock_browser.reset_mock()
            mock_display.reset_mock()

            # Pretend restart succeeds
            def start_side_effect():
                mock_browser.is_running = True

            mock_browser.start.side_effect = start_side_effect

            health = manager.status()

            # Should have attempted restart
            # The status might now be RUNNING if restart succeeded
            assert mock_browser.start.called or health.status == EnvironmentStatus.RUNNING


class TestEnvironmentModuleExports:
    """Tests that environment module exports correct classes."""

    def test_local_environment_manager_exported(self):
        """LocalEnvironmentManager should be exported from environment package."""
        from src.environment import LocalEnvironmentManager

        assert LocalEnvironmentManager is not None

    def test_virtual_display_exported(self):
        """VirtualDisplay should be exported from environment package."""
        from src.environment import VirtualDisplay

        assert VirtualDisplay is not None


@requires_playwright
class TestEnvironmentManagerIntegration:
    """Integration tests for LocalEnvironmentManager (requires Docker/display)."""

    @pytest.fixture
    def manager(self):
        """Create a manager for testing."""
        from src.environment.manager import LocalEnvironmentManager

        # Use headless mode for CI/testing environments
        manager = LocalEnvironmentManager(headless=True)
        yield manager
        # Cleanup
        if manager.is_running():
            manager.stop()

    def test_start_and_stop_lifecycle(self, manager):
        """Should be able to start and stop the environment."""
        manager.start()
        assert manager.is_running()

        manager.stop()
        assert not manager.is_running()

    def test_take_screenshot_after_start(self, manager):
        """Should be able to take screenshot after starting."""
        manager.start()
        manager.navigate("https://example.com")

        screenshot = manager.screenshot()

        assert isinstance(screenshot, bytes)
        assert len(screenshot) > 1000  # Should have substantial content

    def test_screenshot_pil_returns_image(self, manager):
        """screenshot_pil() should return PIL Image."""
        from PIL import Image

        manager.start()
        manager.navigate("https://example.com")

        img = manager.screenshot_pil()

        assert isinstance(img, Image.Image)
        width, height = img.size
        assert width > 0
        assert height > 0
