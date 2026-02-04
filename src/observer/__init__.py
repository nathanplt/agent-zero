"""Observer package: screen streaming, logs, and control API."""

from src.observer.streaming import ScreenStreamingService, compress_frame_to_jpeg

__all__ = ["ScreenStreamingService", "compress_frame_to_jpeg"]
