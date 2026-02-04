"""Screenshot capture module for the vision system.

This module provides fast, reliable screenshot capture from the virtual display
with a configurable buffer for temporal analysis.

Features:
- Captures at 10+ FPS
- Returns raw bytes and PIL Image
- Buffer stores last N frames
- Timestamps accurate to millisecond

Example:
    >>> from src.environment.manager import LocalEnvironmentManager
    >>> from src.vision.capture import ScreenshotCapture
    >>>
    >>> with LocalEnvironmentManager(headless=True) as env:
    ...     env.navigate("https://example.com")
    ...     capture = ScreenshotCapture(env)
    ...     screenshot = capture.capture()
    ...     print(f"Captured {screenshot.width}x{screenshot.height}")
"""

from __future__ import annotations

import io
import logging
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING

from PIL import Image

from src.interfaces.environment import EnvironmentSetupError
from src.interfaces.vision import Screenshot, VisionError

if TYPE_CHECKING:

    from src.interfaces.environment import EnvironmentManager

logger = logging.getLogger(__name__)


class ScreenshotCapture:
    """Fast screenshot capture with buffering for temporal analysis.

    This class wraps an EnvironmentManager to provide:
    - Fast screenshot capture as Screenshot objects
    - A configurable rolling buffer of recent frames
    - Timestamp tracking for each capture

    The buffer uses a deque for O(1) append and automatic size limiting.

    Attributes:
        buffer_size: Maximum number of screenshots to keep in buffer.

    Example:
        >>> capture = ScreenshotCapture(environment_manager, buffer_size=20)
        >>> screenshot = capture.capture()
        >>> recent = capture.get_buffer(count=5)  # Last 5 screenshots
    """

    def __init__(
        self,
        environment_manager: EnvironmentManager,
        buffer_size: int = 10,
    ) -> None:
        """Initialize the screenshot capture system.

        Args:
            environment_manager: The environment manager to capture from.
            buffer_size: Maximum number of screenshots to keep in buffer.
                Defaults to 10.
        """
        self._environment_manager = environment_manager
        self._buffer_size = buffer_size
        self._buffer: deque[Screenshot] = deque(maxlen=buffer_size)

        logger.debug(f"ScreenshotCapture initialized with buffer_size={buffer_size}")

    @property
    def buffer_size(self) -> int:
        """Get the maximum buffer size."""
        return self._buffer_size

    def capture(self) -> Screenshot:
        """Capture a screenshot from the virtual display.

        This method:
        1. Checks that the environment is running
        2. Captures raw bytes and PIL Image from the environment
        3. Creates a Screenshot object with timestamp
        4. Adds to the rolling buffer

        Returns:
            Screenshot object with image data and metadata.

        Raises:
            VisionError: If capture fails or environment not running.
        """
        # Check environment is running
        if not self._environment_manager.is_running():
            raise VisionError("Environment not running. Cannot capture screenshot.")

        try:
            # Capture timestamp immediately before capture
            timestamp = datetime.now()

            # Get raw bytes from environment (single screenshot call)
            raw_bytes = self._environment_manager.screenshot()

            # Derive PIL Image from the same bytes to avoid a second capture
            pil_image = Image.open(io.BytesIO(raw_bytes))

            # Get dimensions from PIL image
            width, height = pil_image.size

            # Create Screenshot object
            screenshot = Screenshot(
                image=pil_image,
                raw_bytes=raw_bytes,
                timestamp=timestamp,
                width=width,
                height=height,
            )

            # Add to buffer
            self._buffer.append(screenshot)

            logger.debug(f"Captured screenshot: {width}x{height} at {timestamp}")

            return screenshot

        except EnvironmentSetupError as e:
            raise VisionError(f"Screenshot capture failed: {e}") from e
        except Exception as e:
            raise VisionError(f"Unexpected error during capture: {e}") from e

    def get_buffer(self, count: int | None = None) -> list[Screenshot]:
        """Get recent screenshots from the buffer.

        Screenshots are returned in reverse chronological order
        (most recent first).

        Args:
            count: Number of screenshots to return. None returns all buffered.

        Returns:
            List of screenshots, most recent first.
        """
        # Convert deque to list in reverse order (most recent first)
        buffer_list = list(reversed(self._buffer))

        if count is None:
            return buffer_list

        return buffer_list[:count]

    def clear_buffer(self) -> None:
        """Clear all screenshots from the buffer."""
        self._buffer.clear()
        logger.debug("Screenshot buffer cleared")
