"""Virtual display management using Xvfb.

This module provides a VirtualDisplay class that manages a virtual X11 display
using Xvfb. This allows running graphical applications without a physical display.

Note: This is primarily used within Docker containers. On macOS/Windows development
machines, the display may not be available and headless mode should be used instead.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class VirtualDisplayError(Exception):
    """Error raised when virtual display operations fail."""

    pass


class VirtualDisplay:
    """Manages a virtual X11 display using Xvfb.

    This class provides methods to:
    - Start/stop Xvfb virtual display
    - Take screenshots of the display
    - Check display status

    Example:
        >>> display = VirtualDisplay(display=":99", width=1920, height=1080)
        >>> display.start()
        >>> # Run graphical applications...
        >>> display.stop()

    Note:
        On systems without Xvfb (macOS, Windows), the display will run in
        a "mock" mode where is_running always returns True but no actual
        display process is started.
    """

    def __init__(
        self,
        display: str = ":99",
        width: int = 1920,
        height: int = 1080,
        depth: int = 24,
    ) -> None:
        """Initialize virtual display configuration.

        Args:
            display: X display number (e.g., ":99").
            width: Display width in pixels.
            height: Display height in pixels.
            depth: Color depth (typically 24 for true color).
        """
        self._display = display
        self._width = width
        self._height = height
        self._depth = depth

        self._process: subprocess.Popen[bytes] | None = None
        self._is_running = False
        self._mock_mode = False
        self._start_time: float | None = None

    @property
    def display(self) -> str:
        """Get the display number."""
        return self._display

    @property
    def width(self) -> int:
        """Get the display width."""
        return self._width

    @property
    def height(self) -> int:
        """Get the display height."""
        return self._height

    @property
    def depth(self) -> int:
        """Get the display color depth."""
        return self._depth

    @property
    def is_running(self) -> bool:
        """Check if the virtual display is running.

        Returns:
            True if display is running (or mock mode is active).
        """
        if self._mock_mode:
            return self._is_running

        if self._process is None:
            return False

        # Check if process is still alive
        poll_result = self._process.poll()
        if poll_result is not None:
            # Process exited
            self._is_running = False
            self._process = None
            return False

        return self._is_running

    def _xvfb_available(self) -> bool:
        """Check if Xvfb is available on the system."""
        return shutil.which("Xvfb") is not None

    def start(self) -> None:
        """Start the virtual display.

        If Xvfb is not available (e.g., on macOS/Windows), this will
        activate mock mode instead.

        Raises:
            VirtualDisplayError: If display fails to start.
        """
        if self._is_running:
            logger.warning("Virtual display is already running")
            return

        if not self._xvfb_available():
            logger.info(
                "Xvfb not available, activating mock mode. "
                "Use headless browser mode for proper operation."
            )
            self._mock_mode = True
            self._is_running = True
            self._start_time = time.time()
            return

        try:
            # Build Xvfb command
            screen_config = f"{self._width}x{self._height}x{self._depth}"
            cmd = [
                "Xvfb",
                self._display,
                "-screen",
                "0",
                screen_config,
                "-ac",  # Disable access control
                "-nolisten",
                "tcp",  # Don't listen on TCP (security)
            ]

            logger.info(f"Starting Xvfb: {' '.join(cmd)}")

            # Start Xvfb process
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            # Wait a moment for startup
            time.sleep(0.5)

            # Check if it started successfully
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                raise VirtualDisplayError(f"Xvfb failed to start: {stderr}")

            # Set DISPLAY environment variable
            os.environ["DISPLAY"] = self._display

            self._is_running = True
            self._start_time = time.time()
            logger.info(f"Virtual display started on {self._display} ({screen_config})")

        except FileNotFoundError:
            # Xvfb not found, use mock mode
            logger.warning("Xvfb not found, using mock mode")
            self._mock_mode = True
            self._is_running = True
            self._start_time = time.time()
        except Exception as e:
            self._cleanup()
            raise VirtualDisplayError(f"Failed to start virtual display: {e}") from e

    def stop(self) -> None:
        """Stop the virtual display."""
        logger.info("Stopping virtual display...")
        self._cleanup()
        logger.info("Virtual display stopped")

    def _cleanup(self) -> None:
        """Clean up display resources."""
        if self._process is not None:
            with contextlib.suppress(Exception):
                self._process.terminate()
                self._process.wait(timeout=5)

            if self._process.poll() is None:
                # Still running, force kill
                with contextlib.suppress(Exception):
                    self._process.kill()
                    self._process.wait(timeout=2)

            self._process = None

        self._is_running = False
        self._mock_mode = False
        self._start_time = None

    def screenshot(self) -> bytes:
        """Take a screenshot of the virtual display.

        This uses xwd + ImageMagick to capture the display.
        Only works when Xvfb is running (not in mock mode).

        Returns:
            PNG-encoded screenshot bytes.

        Raises:
            VirtualDisplayError: If screenshot fails.
        """
        if not self._is_running:
            raise VirtualDisplayError("Virtual display not running")

        if self._mock_mode:
            raise VirtualDisplayError(
                "Cannot take screenshot in mock mode. Use browser screenshot instead."
            )

        # Check if required tools are available
        if not shutil.which("xwd") or not shutil.which("convert"):
            raise VirtualDisplayError(
                "xwd or ImageMagick not installed. Use browser screenshot instead."
            )

        try:
            # Use xwd to capture the root window, pipe to convert for PNG output
            xwd_cmd = ["xwd", "-root", "-display", self._display, "-silent"]
            convert_cmd = ["convert", "xwd:-", "png:-"]

            xwd_proc = subprocess.Popen(
                xwd_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            convert_proc = subprocess.Popen(
                convert_cmd,
                stdin=xwd_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Close xwd's stdout in parent so convert gets EOF when done
            if xwd_proc.stdout:
                xwd_proc.stdout.close()

            png_data, stderr = convert_proc.communicate(timeout=10)

            if convert_proc.returncode != 0:
                raise VirtualDisplayError(f"Screenshot conversion failed: {stderr.decode()}")

            return png_data

        except subprocess.TimeoutExpired as e:
            raise VirtualDisplayError("Screenshot capture timed out") from e
        except Exception as e:
            raise VirtualDisplayError(f"Screenshot failed: {e}") from e

    def get_uptime(self) -> float:
        """Get display uptime in seconds.

        Returns:
            Uptime in seconds, or 0.0 if not running.
        """
        if not self._is_running or self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def __enter__(self) -> VirtualDisplay:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        """Context manager exit."""
        self.stop()

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        self._cleanup()
