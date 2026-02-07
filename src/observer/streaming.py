"""Screen/action streaming services for observer WebSocket delivery."""

from __future__ import annotations

import io
import logging
import threading
from collections import deque
from datetime import datetime
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_STREAM_FPS = 10
DEFAULT_STREAM_QUALITY = 80


def compress_frame_to_jpeg(
    frame: bytes | Image.Image,
    quality: int = DEFAULT_STREAM_QUALITY,
) -> bytes:
    """Compress a frame (raw bytes or PIL Image) to JPEG.

    Args:
        frame: Raw image bytes (PNG/JPEG) or PIL Image.
        quality: JPEG quality 1-100.

    Returns:
        JPEG-encoded bytes.
    """
    quality = max(1, min(100, quality))
    if isinstance(frame, Image.Image):
        img = frame.convert("RGB")
    else:
        img = Image.open(io.BytesIO(frame)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()


class ScreenStreamingService:
    """Holds latest frame as JPEG; supports configurable quality and FPS.

    Producers call push_frame() with raw or PIL data; consumers (e.g. WebSocket
    handler) call get_latest_frame() to get JPEG bytes. Thread-safe.
    """

    def __init__(
        self,
        stream_fps: int = DEFAULT_STREAM_FPS,
        stream_quality: int = DEFAULT_STREAM_QUALITY,
    ) -> None:
        self._stream_fps = stream_fps
        self._stream_quality = stream_quality
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None

    def push_frame(self, frame: bytes | Image.Image) -> None:
        """Store a new frame as JPEG. Thread-safe."""
        jpeg = compress_frame_to_jpeg(frame, self._stream_quality)
        with self._lock:
            self._latest_jpeg = jpeg

    def get_latest_frame(self) -> bytes | None:
        """Return the latest JPEG frame, or None if none pushed yet. Thread-safe."""
        with self._lock:
            return self._latest_jpeg

    def set_stream_fps(self, fps: int) -> None:
        """Set target frame rate (1-60)."""
        self._stream_fps = max(1, min(60, fps))

    def get_stream_fps(self) -> int:
        """Get current target frame rate."""
        return self._stream_fps

    def set_stream_quality(self, quality: int) -> None:
        """Set JPEG quality (1-100). Affects future push_frame calls."""
        self._stream_quality = max(1, min(100, quality))

    def get_stream_quality(self) -> int:
        """Get current JPEG quality."""
        return self._stream_quality


class ActionStreamingService:
    """Thread-safe bounded event stream for action/decision observability."""

    def __init__(self, max_events: int = 500) -> None:
        self._max_events = max(1, max_events)
        self._lock = threading.Lock()
        self._events: deque[dict[str, Any]] = deque(maxlen=self._max_events)
        self._next_id = 1
        self._control_commands: deque[dict[str, Any]] = deque(maxlen=self._max_events)
        self._next_control_id = 1

    def push_event(self, payload: dict[str, Any]) -> int:
        """Push an event payload and return its assigned event id."""
        with self._lock:
            event_id = self._next_id
            self._next_id += 1
            event = {
                "id": event_id,
                "timestamp": datetime.now().isoformat(),
                "payload": payload,
            }
            self._events.append(event)
            return event_id

    def get_events_since(self, last_event_id: int) -> list[dict[str, Any]]:
        """Get events with id greater than `last_event_id`."""
        with self._lock:
            return [event for event in self._events if int(event["id"]) > last_event_id]

    def push_control_command(
        self,
        command: str,
        payload: dict[str, Any] | None = None,
    ) -> int:
        """Queue a control command for runtime consumers."""
        with self._lock:
            command_id = self._next_control_id
            self._next_control_id += 1
            entry = {
                "id": command_id,
                "timestamp": datetime.now().isoformat(),
                "command": command,
                "payload": payload or {},
            }
            self._control_commands.append(entry)
            return command_id

    def pop_control_commands(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Pop queued control commands in FIFO order."""
        with self._lock:
            if limit is None or limit <= 0:
                limit = len(self._control_commands)
            popped: list[dict[str, Any]] = []
            while self._control_commands and len(popped) < limit:
                popped.append(self._control_commands.popleft())
            return popped
