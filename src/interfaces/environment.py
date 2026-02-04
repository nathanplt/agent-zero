"""Environment manager interface for container/runtime control."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


class EnvironmentStatus(Enum):
    """Status of the execution environment."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"
    CRASHED = "crashed"


class EnvironmentHealth:
    """Health information about the environment."""

    __slots__ = (
        "status",
        "uptime_seconds",
        "cpu_percent",
        "memory_mb",
        "display_active",
        "browser_active",
        "error_message",
    )

    def __init__(
        self,
        status: EnvironmentStatus,
        uptime_seconds: float = 0.0,
        cpu_percent: float = 0.0,
        memory_mb: float = 0.0,
        display_active: bool = False,
        browser_active: bool = False,
        error_message: str | None = None,
    ) -> None:
        """Initialize environment health.

        Args:
            status: Current status of the environment.
            uptime_seconds: How long the environment has been running.
            cpu_percent: Current CPU usage percentage.
            memory_mb: Current memory usage in megabytes.
            display_active: Whether the virtual display is running.
            browser_active: Whether the browser is running.
            error_message: Error message if status is ERROR or CRASHED.
        """
        self.status = status
        self.uptime_seconds = uptime_seconds
        self.cpu_percent = cpu_percent
        self.memory_mb = memory_mb
        self.display_active = display_active
        self.browser_active = browser_active
        self.error_message = error_message


class EnvironmentManager(ABC):
    """Abstract interface for environment management.

    The environment manager is responsible for:
    - Starting/stopping the container or VM
    - Managing the virtual display
    - Managing the browser runtime
    - Providing screenshots
    - Health monitoring
    """

    @abstractmethod
    def start(self) -> None:
        """Start the execution environment.

        This launches the container, virtual display, and browser.

        Raises:
            EnvironmentSetupError: If startup fails.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the execution environment.

        This cleanly shuts down the browser, display, and container.
        """
        ...

    @abstractmethod
    def restart(self) -> None:
        """Restart the execution environment.

        Equivalent to stop() followed by start().
        """
        ...

    @abstractmethod
    def status(self) -> EnvironmentHealth:
        """Get current environment health status.

        Returns:
            EnvironmentHealth with current metrics.
        """
        ...

    @abstractmethod
    def screenshot(self) -> bytes:
        """Capture a screenshot from the virtual display.

        Returns:
            PNG-encoded screenshot bytes.

        Raises:
            EnvironmentSetupError: If capture fails.
        """
        ...

    @abstractmethod
    def screenshot_pil(self) -> Image.Image:
        """Capture a screenshot as a PIL Image.

        Returns:
            PIL Image object.

        Raises:
            EnvironmentSetupError: If capture fails.
        """
        ...

    @abstractmethod
    def navigate(self, url: str) -> None:
        """Navigate the browser to a URL.

        Args:
            url: The URL to navigate to.

        Raises:
            EnvironmentSetupError: If navigation fails.
        """
        ...

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the environment is running.

        Returns:
            True if environment is running, False otherwise.
        """
        ...

    @abstractmethod
    def wait_for_ready(self, timeout_seconds: float = 30.0) -> bool:
        """Wait for the environment to be ready.

        Args:
            timeout_seconds: Maximum time to wait.

        Returns:
            True if ready within timeout, False otherwise.
        """
        ...

    @abstractmethod
    def get_display_size(self) -> tuple[int, int]:
        """Get the virtual display size.

        Returns:
            Tuple of (width, height) in pixels.
        """
        ...


class EnvironmentSetupError(Exception):
    """Error raised when environment operations fail.

    Named EnvironmentSetupError to avoid shadowing the Python
    built-in EnvironmentError (OSError subclass).
    """

    pass
