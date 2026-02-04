"""Communication server interface for streaming and control API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any


class LogLevel(Enum):
    """Log levels for streaming logs."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    DECISION = "decision"  # Special level for agent decisions


class LogEntry:
    """A log entry to be streamed."""

    __slots__ = ("timestamp", "level", "message", "data")

    def __init__(
        self,
        timestamp: str,
        level: LogLevel,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a log entry.

        Args:
            timestamp: ISO format timestamp.
            level: Log level.
            message: Log message.
            data: Optional structured data.
        """
        self.timestamp = timestamp
        self.level = level
        self.message = message
        self.data = data


class AgentCommand(Enum):
    """Commands that can be sent to the agent."""

    START = "start"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    RESTART = "restart"


class AgentState(Enum):
    """Current state of the agent."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class CommunicationServer(ABC):
    """Abstract interface for the communication server.

    The communication server is responsible for:
    - Streaming screen frames to clients
    - Streaming logs to clients
    - Providing metrics
    - Handling control commands
    - Serving the web dashboard
    """

    # Lifecycle

    @abstractmethod
    def start(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Start the communication server.

        Args:
            host: Host to bind to.
            port: Port to listen on.

        Raises:
            CommunicationError: If server fails to start.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the communication server."""
        ...

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the server is running.

        Returns:
            True if running, False otherwise.
        """
        ...

    # Screen Streaming

    @abstractmethod
    def push_frame(self, frame_data: bytes) -> None:
        """Push a new frame to all connected clients.

        Args:
            frame_data: JPEG or PNG encoded frame bytes.
        """
        ...

    @abstractmethod
    def set_frame_rate(self, fps: int) -> None:
        """Set the target frame rate for streaming.

        Args:
            fps: Target frames per second.
        """
        ...

    @abstractmethod
    def get_connected_clients(self) -> int:
        """Get the number of connected clients.

        Returns:
            Number of active client connections.
        """
        ...

    # Log Streaming

    @abstractmethod
    def push_log(self, entry: LogEntry) -> None:
        """Push a log entry to all connected clients.

        Args:
            entry: The log entry to push.
        """
        ...

    @abstractmethod
    def set_log_level(self, level: LogLevel) -> None:
        """Set minimum log level to stream.

        Args:
            level: Minimum level to include.
        """
        ...

    # Metrics

    @abstractmethod
    def update_metrics(self, metrics: dict[str, Any]) -> None:
        """Update the current metrics.

        Args:
            metrics: Dictionary of metric name to value.
        """
        ...

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics.

        Returns:
            Dictionary of all current metrics.
        """
        ...

    # Control API

    @abstractmethod
    def register_command_handler(
        self,
        callback: Callable[[AgentCommand], None],
    ) -> None:
        """Register a callback for agent commands.

        Args:
            callback: Function to call when a command is received.
        """
        ...

    @abstractmethod
    def set_agent_state(self, state: AgentState) -> None:
        """Update the current agent state.

        Args:
            state: New agent state.
        """
        ...

    @abstractmethod
    def get_agent_state(self) -> AgentState:
        """Get the current agent state.

        Returns:
            Current agent state.
        """
        ...

    # Configuration

    @abstractmethod
    def update_config(self, config: dict[str, Any]) -> None:
        """Update agent configuration at runtime.

        Args:
            config: Configuration values to update.
        """
        ...


class CommunicationError(Exception):
    """Error raised when communication operations fail."""

    pass
