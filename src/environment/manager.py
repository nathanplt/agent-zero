"""Environment manager implementation.

This module provides a LocalEnvironmentManager that manages the execution
environment including virtual display and browser runtime. It implements
the EnvironmentManager interface.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from src.environment.browser import BrowserRuntime, BrowserRuntimeError
from src.environment.display import VirtualDisplay, VirtualDisplayError
from src.interfaces.environment import (
    EnvironmentHealth,
    EnvironmentManager,
    EnvironmentSetupError,
    EnvironmentStatus,
)

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


class LocalEnvironmentManager(EnvironmentManager):
    """Manages a local execution environment with virtual display and browser.

    This class coordinates:
    - Virtual display (Xvfb) for graphical rendering
    - Browser runtime (Chromium via Playwright) for web interaction
    - Environment lifecycle and health monitoring

    Example:
        >>> with LocalEnvironmentManager(headless=True) as env:
        ...     env.navigate("https://example.com")
        ...     screenshot = env.screenshot()

    Attributes:
        headless: Whether to run browser in headless mode.
        auto_restart: Whether to automatically restart on crash.
    """

    def __init__(
        self,
        headless: bool = False,
        viewport_width: int = 1920,
        viewport_height: int = 1080,
        display: str = ":99",
        auto_restart: bool = False,
        storage_state_path: str | Path | None = None,
        use_virtual_display: bool = True,
    ) -> None:
        """Initialize the environment manager.

        Args:
            headless: Run browser in headless mode (no display needed).
            viewport_width: Browser viewport width in pixels.
            viewport_height: Browser viewport height in pixels.
            display: X display number for virtual display.
            auto_restart: Automatically restart on crash detection.
        """
        self._headless = headless
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._display_number = display
        self._auto_restart = auto_restart
        self._storage_state_path = storage_state_path
        self._use_virtual_display = use_virtual_display

        self._display: VirtualDisplay | None = None
        self._browser: BrowserRuntime | None = None
        self._start_time: float | None = None
        self._was_started = False  # Tracks if start() was ever called

    def _create_display(self) -> VirtualDisplay:
        """Create a virtual display instance.

        This is separated for easier testing/mocking.
        """
        return VirtualDisplay(
            display=self._display_number,
            width=self._viewport_width,
            height=self._viewport_height,
        )

    def _create_browser(self) -> BrowserRuntime:
        """Create a browser runtime instance.

        This is separated for easier testing/mocking.
        """
        return BrowserRuntime(
            headless=self._headless,
            viewport_width=self._viewport_width,
            viewport_height=self._viewport_height,
            storage_state_path=self._storage_state_path,
        )

    def start(self) -> None:
        """Start the execution environment.

        This launches the virtual display (if not headless) and browser.

        Raises:
            EnvironmentSetupError: If startup fails.
        """
        if self._was_started and self.is_running():
            logger.warning("Environment is already running")
            return

        try:
            logger.info("Starting environment...")

            # Create components if needed
            if self._use_virtual_display and self._display is None:
                self._display = self._create_display()
            if self._browser is None:
                self._browser = self._create_browser()

            # Start virtual display if not in headless mode
            if not self._headless and self._use_virtual_display:
                logger.info("Starting virtual display...")
                self._display.start()
            elif not self._headless:
                logger.info("Using native desktop display (virtual display disabled)")

            # Start browser
            logger.info("Starting browser...")
            self._browser.start()

            self._start_time = time.time()
            self._was_started = True
            logger.info("Environment started successfully")

        except VirtualDisplayError as e:
            self._cleanup()
            raise EnvironmentSetupError(f"Failed to start display: {e}") from e
        except BrowserRuntimeError as e:
            self._cleanup()
            raise EnvironmentSetupError(f"Failed to start browser: {e}") from e
        except Exception as e:
            self._cleanup()
            raise EnvironmentSetupError(f"Failed to start environment: {e}") from e

    def stop(self) -> None:
        """Stop the execution environment.

        This cleanly shuts down the browser and virtual display.
        """
        logger.info("Stopping environment...")
        self._cleanup()
        self._was_started = False
        logger.info("Environment stopped")

    def _cleanup(self) -> None:
        """Clean up environment resources."""
        # Stop browser first
        if self._browser is not None:
            try:
                self._browser.stop()
            except Exception as e:
                logger.warning(f"Error stopping browser: {e}")

        # Then stop display
        if self._display is not None:
            try:
                self._display.stop()
            except Exception as e:
                logger.warning(f"Error stopping display: {e}")

        self._start_time = None

    def restart(self) -> None:
        """Restart the execution environment.

        Equivalent to stop() followed by start().
        """
        logger.info("Restarting environment...")
        self.stop()
        self.start()

    def status(self) -> EnvironmentHealth:
        """Get current environment health status.

        Returns:
            EnvironmentHealth with current metrics and status.
        """
        # Determine status
        display_active = (
            self._headless
            or not self._use_virtual_display
            or (self._display is not None and self._display.is_running)
        )
        browser_active = (
            self._browser is not None
            and self._browser.is_running
        )

        # Calculate uptime
        uptime = 0.0
        if self._start_time is not None:
            uptime = time.time() - self._start_time

        # Determine overall status
        if not self._was_started:
            status = EnvironmentStatus.STOPPED
        elif display_active and browser_active:
            status = EnvironmentStatus.RUNNING
        elif self._was_started and (not display_active or not browser_active):
            # Was started but something died
            status = EnvironmentStatus.CRASHED

            # Auto-restart if enabled
            if self._auto_restart:
                logger.warning("Crash detected, attempting auto-restart...")
                try:
                    self.restart()
                    # Check if restart worked
                    display_active = (
                        self._headless
                        or not self._use_virtual_display
                        or (self._display is not None and self._display.is_running)
                    )
                    browser_active = (
                        self._browser is not None
                        and self._browser.is_running
                    )
                    if display_active and browser_active:
                        status = EnvironmentStatus.RUNNING
                        logger.info("Auto-restart successful")
                except Exception as e:
                    logger.error(f"Auto-restart failed: {e}")
        else:
            status = EnvironmentStatus.STOPPED

        error_message = None
        if status == EnvironmentStatus.CRASHED:
            error_message = "Environment crashed unexpectedly"

        return EnvironmentHealth(
            status=status,
            uptime_seconds=uptime,
            cpu_percent=0.0,  # TODO: Implement CPU monitoring
            memory_mb=0.0,  # TODO: Implement memory monitoring
            display_active=display_active,
            browser_active=browser_active,
            error_message=error_message,
        )

    def screenshot(self) -> bytes:
        """Capture a screenshot from the browser.

        Returns:
            PNG-encoded screenshot bytes.

        Raises:
            EnvironmentSetupError: If capture fails or environment not running.
        """
        if not self.is_running():
            raise EnvironmentSetupError("Environment not running. Call start() first.")

        if self._browser is None:
            raise EnvironmentSetupError("Browser not initialized")

        try:
            result = self._browser.screenshot_with_recovery()
            if isinstance(result, bytes):
                return result
            return self._browser.screenshot()
        except BrowserRuntimeError as e:
            raise EnvironmentSetupError(f"Screenshot failed: {e}") from e

    def screenshot_pil(self) -> Image.Image:
        """Capture a screenshot as a PIL Image.

        Returns:
            PIL Image object.

        Raises:
            EnvironmentSetupError: If capture fails or environment not running.
        """
        if not self.is_running():
            raise EnvironmentSetupError("Environment not running. Call start() first.")

        if self._browser is None:
            raise EnvironmentSetupError("Browser not initialized")

        try:
            return self._browser.screenshot_pil()
        except BrowserRuntimeError as e:
            raise EnvironmentSetupError(f"Screenshot failed: {e}") from e

    def navigate(self, url: str) -> None:
        """Navigate the browser to a URL.

        Args:
            url: The URL to navigate to.

        Raises:
            EnvironmentSetupError: If navigation fails or environment not running.
        """
        if not self.is_running():
            raise EnvironmentSetupError("Environment not running. Call start() first.")

        if self._browser is None:
            raise EnvironmentSetupError("Browser not initialized")

        try:
            self._browser.navigate(url)
        except BrowserRuntimeError as e:
            raise EnvironmentSetupError(f"Navigation failed: {e}") from e

    def is_running(self) -> bool:
        """Check if the environment is running.

        Returns:
            True if both display (or headless) and browser are running.
        """
        display_ok = (
            self._headless
            or not self._use_virtual_display
            or (self._display is not None and self._display.is_running)
        )
        browser_ok = self._browser is not None and self._browser.is_running

        return display_ok and browser_ok

    def wait_for_ready(self, timeout_seconds: float = 30.0) -> bool:
        """Wait for the environment to be ready.

        Args:
            timeout_seconds: Maximum time to wait.

        Returns:
            True if ready within timeout, False otherwise.
        """
        start = time.time()
        poll_interval = 0.1

        while time.time() - start < timeout_seconds:
            if self.is_running():
                return True
            time.sleep(poll_interval)

        return False

    def get_display_size(self) -> tuple[int, int]:
        """Get the display/viewport size.

        Returns:
            Tuple of (width, height) in pixels.
        """
        if self._display is not None:
            return (self._display.width, self._display.height)
        return (self._viewport_width, self._viewport_height)

    def get_browser_runtime(self) -> BrowserRuntime | None:
        """Expose browser runtime through a public accessor."""
        return self._browser

    def __enter__(self) -> LocalEnvironmentManager:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        """Context manager exit."""
        self.stop()

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        import contextlib

        with contextlib.suppress(Exception):
            self._cleanup()
