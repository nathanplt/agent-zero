"""FastAPI application for screen and action log streaming."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from src.observer.streaming import ActionStreamingService, ScreenStreamingService

logger = logging.getLogger(__name__)

LIVE_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Agent Zero Live</title>
  <style>
    body { font-family: sans-serif; margin: 0; padding: 1rem; background: #111; color: #eee; }
    .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 1rem; }
    .panel { background: #1a1a1a; border: 1px solid #2f2f2f; border-radius: 8px; padding: 0.75rem; }
    img { width: 100%; border-radius: 6px; border: 1px solid #333; background: #000; }
    pre { white-space: pre-wrap; max-height: 70vh; overflow-y: auto; font-size: 12px; }
  </style>
</head>
<body>
  <h2>Agent Zero Live Observer</h2>
  <div class="grid">
    <div class="panel">
      <h3>Screen</h3>
      <img id="screen" alt="Live screen stream" />
    </div>
    <div class="panel">
      <h3>Action Log</h3>
      <pre id="logs"></pre>
    </div>
  </div>
  <script>
    const logsEl = document.getElementById("logs");
    const screenEl = document.getElementById("screen");

    const screenProto = location.protocol === "https:" ? "wss" : "ws";
    const screenWs = new WebSocket(`${screenProto}://${location.host}/ws/screen`);
    screenWs.binaryType = "arraybuffer";
    screenWs.onmessage = (ev) => {
      const blob = new Blob([ev.data], { type: "image/jpeg" });
      const url = URL.createObjectURL(blob);
      screenEl.src = url;
    };

    const logsWs = new WebSocket(`${screenProto}://${location.host}/ws/logs`);
    logsWs.onmessage = (ev) => {
      const data = JSON.parse(ev.data);
      logsEl.textContent = `${JSON.stringify(data)}\\n` + logsEl.textContent;
    };
  </script>
</body>
</html>
"""


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


def create_app(
    streaming_service: ScreenStreamingService | None = None,
    action_streaming_service: ActionStreamingService | None = None,
) -> FastAPI:
    """Create FastAPI app with screen and log streaming endpoints.

    Args:
        streaming_service: Shared service for frame delivery. If None, a new
            instance is created and stored on app.state.
        action_streaming_service: Shared service for action/log events.
    """
    app = FastAPI(title="Agent Zero Observer", version="0.1.0")
    svc = streaming_service or ScreenStreamingService()
    action_svc = action_streaming_service or ActionStreamingService()
    app.state.streaming_service = svc
    app.state.action_streaming_service = action_svc

    @app.get("/live")
    async def live_page() -> HTMLResponse:
        return HTMLResponse(LIVE_PAGE_HTML)

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

    @app.websocket("/ws/logs")
    async def logs_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        service: ActionStreamingService = app.state.action_streaming_service
        last_event_id = 0
        try:
            while True:
                events = service.get_events_since(last_event_id)
                for event in events:
                    await websocket.send_json(event)
                    last_event_id = int(event["id"])
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            logger.debug("Logs WebSocket client disconnected")
        except Exception as e:
            logger.warning("Logs stream error: %s", e)

    return app
