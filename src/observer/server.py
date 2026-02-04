"""FastAPI application and WebSocket endpoint for screen streaming."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.observer.streaming import ScreenStreamingService

logger = logging.getLogger(__name__)


def _apply_stream_config(msg: str, service: ScreenStreamingService) -> None:
    """Apply quality/fps from JSON message to service."""
    try:
        data = json.loads(msg)
        if "quality" in data:
            service.set_stream_quality(int(data["quality"]))
        if "fps" in data:
            service.set_stream_fps(int(data["fps"]))
    except (ValueError, TypeError):
        pass


def create_app(streaming_service: ScreenStreamingService | None = None) -> FastAPI:
    """Create FastAPI app with /ws/screen endpoint.

    Args:
        streaming_service: Shared service for frame delivery. If None, a new
            instance is created and stored on app.state.
    """
    app = FastAPI(title="Agent Zero Observer", version="0.1.0")
    svc = streaming_service or ScreenStreamingService()
    app.state.streaming_service = svc

    @app.websocket("/ws/screen")
    async def screen_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        service: ScreenStreamingService = app.state.streaming_service
        fps = service.get_stream_fps()
        interval = 1.0 / fps if fps > 0 else 0.1
        try:
            while True:
                frame = service.get_latest_frame()
                if frame:
                    await websocket.send_bytes(frame)
                try:
                    msg = await asyncio.wait_for(websocket.receive_text(), timeout=interval)
                except TimeoutError:
                    pass
                else:
                    _apply_stream_config(msg, service)
                    fps = service.get_stream_fps()
                    interval = 1.0 / fps if fps > 0 else 0.1
        except WebSocketDisconnect:
            logger.debug("Screen WebSocket client disconnected")
        except Exception as e:
            logger.warning("Screen stream error: %s", e)

    return app
