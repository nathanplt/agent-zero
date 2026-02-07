"""Tests for screen and log streaming (Feature 7.1/7.2)."""

from __future__ import annotations

import io

import pytest

from src.observer.streaming import (
    ActionStreamingService,
    ScreenStreamingService,
    compress_frame_to_jpeg,
)

# PIL for test images
pytest.importorskip("PIL")
from PIL import Image


def _make_test_image(width: int = 100, height: int = 80) -> Image.Image:
    """Create a small RGB image."""
    return Image.new("RGB", (width, height), color=(128, 64, 192))


def _png_bytes(width: int = 50, height: int = 50) -> bytes:
    """Minimal PNG as bytes."""
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestCompressFrameToJpeg:
    """Tests for compress_frame_to_jpeg."""

    def test_compresses_pil_image_to_jpeg_bytes(self) -> None:
        img = _make_test_image()
        out = compress_frame_to_jpeg(img, quality=85)
        assert isinstance(out, bytes)
        assert out[:2] == b"\xff\xd8"
        assert out[-2:] == b"\xff\xd9"

    def test_compresses_raw_bytes_to_jpeg(self) -> None:
        raw = _png_bytes()
        out = compress_frame_to_jpeg(raw, quality=70)
        assert isinstance(out, bytes)
        assert len(out) > 0
        assert out[:2] == b"\xff\xd8"

    def test_quality_affects_size(self) -> None:
        img = _make_test_image(200, 200)
        low = compress_frame_to_jpeg(img, quality=10)
        high = compress_frame_to_jpeg(img, quality=95)
        assert len(low) <= len(high) or len(high) > 0


class TestScreenStreamingService:
    """Tests for ScreenStreamingService."""

    def test_get_latest_frame_none_initially(self) -> None:
        svc = ScreenStreamingService()
        assert svc.get_latest_frame() is None

    def test_push_and_get_frame(self) -> None:
        svc = ScreenStreamingService()
        img = _make_test_image()
        svc.push_frame(img)
        frame = svc.get_latest_frame()
        assert frame is not None
        assert frame[:2] == b"\xff\xd8"

    def test_frames_at_target_fps(self) -> None:
        svc = ScreenStreamingService(stream_fps=10)
        assert svc.get_stream_fps() == 10
        svc.set_stream_fps(15)
        assert svc.get_stream_fps() == 15

    def test_quality_adjustment(self) -> None:
        svc = ScreenStreamingService(stream_quality=80)
        assert svc.get_stream_quality() == 80
        svc.set_stream_quality(50)
        assert svc.get_stream_quality() == 50
        img = _make_test_image()
        svc.push_frame(img)
        frame = svc.get_latest_frame()
        assert frame is not None


class TestActionStreamingService:
    """Tests for ActionStreamingService."""

    def test_push_and_retrieve_events_with_incrementing_ids(self) -> None:
        svc = ActionStreamingService(max_events=5)
        first_id = svc.push_event({"event": "a"})
        second_id = svc.push_event({"event": "b"})

        assert first_id == 1
        assert second_id == 2
        events = svc.get_events_since(0)
        assert [event["id"] for event in events] == [1, 2]
        assert events[1]["payload"]["event"] == "b"

    def test_event_buffer_is_bounded(self) -> None:
        svc = ActionStreamingService(max_events=2)
        svc.push_event({"event": "a"})
        svc.push_event({"event": "b"})
        svc.push_event({"event": "c"})

        events = svc.get_events_since(0)
        assert len(events) == 2
        assert events[0]["payload"]["event"] == "b"
        assert events[1]["payload"]["event"] == "c"

    def test_push_and_pop_control_commands(self) -> None:
        svc = ActionStreamingService(max_events=5)
        cmd_id = svc.push_control_command("auth_click_login")

        assert cmd_id == 1
        commands = svc.pop_control_commands()
        assert len(commands) == 1
        assert commands[0]["command"] == "auth_click_login"
        assert svc.pop_control_commands() == []


class TestWebSocketEndpoint:
    """Tests for FastAPI WebSocket /ws/screen (requires fastapi)."""

    @pytest.fixture
    def app_and_service(self):
        pytest.importorskip("fastapi")
        from src.observer.server import create_app
        from src.observer.streaming import ActionStreamingService, ScreenStreamingService

        svc = ScreenStreamingService(stream_fps=10, stream_quality=75)
        action_svc = ActionStreamingService()
        app = create_app(streaming_service=svc, action_streaming_service=action_svc)
        return app, svc, action_svc

    def test_websocket_connect_and_receive_frames(self, app_and_service) -> None:
        pytest.importorskip("fastapi")
        from starlette.testclient import TestClient

        app, svc, _ = app_and_service
        svc.push_frame(_make_test_image(64, 64))
        with TestClient(app) as client, client.websocket_connect("/ws/screen") as ws:
            data = ws.receive_bytes()
            assert isinstance(data, bytes)
            assert len(data) > 0
            assert data[:2] == b"\xff\xd8"

    def test_websocket_accepts_connection(self, app_and_service) -> None:
        pytest.importorskip("fastapi")
        from starlette.testclient import TestClient

        app, _, _ = app_and_service
        with TestClient(app) as client, client.websocket_connect("/ws/screen") as ws:
            ws.close()

    def test_log_websocket_receives_action_events(self, app_and_service) -> None:
        pytest.importorskip("fastapi")
        from starlette.testclient import TestClient

        app, _, action_svc = app_and_service
        action_svc.push_event({"event": "decision", "action": "click"})
        with TestClient(app) as client, client.websocket_connect("/ws/logs") as ws:
            data = ws.receive_json()
            assert data["payload"]["event"] == "decision"
            assert data["payload"]["action"] == "click"

    def test_live_page_is_served(self, app_and_service) -> None:
        pytest.importorskip("fastapi")
        from starlette.testclient import TestClient

        app, _, _ = app_and_service
        with TestClient(app) as client:
            response = client.get("/live")
        assert response.status_code == 200
        assert "ws/screen" in response.text
        assert "ws/logs" in response.text
        assert "/control/auth/click-login" in response.text
        assert "Click Login In Agent Session" in response.text

    def test_control_click_login_endpoint_enqueues_command_and_event(self, app_and_service) -> None:
        pytest.importorskip("fastapi")
        from starlette.testclient import TestClient

        app, _, action_svc = app_and_service
        with TestClient(app) as client:
            response = client.post("/control/auth/click-login")

        assert response.status_code == 200
        events = action_svc.get_events_since(0)
        assert events
        assert events[-1]["payload"]["event"] == "auth_click_login_request"
        commands = action_svc.pop_control_commands()
        assert commands
        assert commands[-1]["command"] == "auth_click_login"
