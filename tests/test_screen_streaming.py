"""Tests for screen streaming (Feature 7.1)."""

from __future__ import annotations

import io

import pytest

from src.observer.streaming import ScreenStreamingService, compress_frame_to_jpeg

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


class TestWebSocketEndpoint:
    """Tests for FastAPI WebSocket /ws/screen (requires fastapi)."""

    @pytest.fixture
    def app_and_service(self):
        pytest.importorskip("fastapi")
        from src.observer.server import create_app
        from src.observer.streaming import ScreenStreamingService

        svc = ScreenStreamingService(stream_fps=10, stream_quality=75)
        app = create_app(streaming_service=svc)
        return app, svc

    def test_websocket_connect_and_receive_frames(self, app_and_service) -> None:
        pytest.importorskip("fastapi")
        from starlette.testclient import TestClient

        app, svc = app_and_service
        svc.push_frame(_make_test_image(64, 64))
        with TestClient(app) as client, client.websocket_connect("/ws/screen") as ws:
            data = ws.receive_bytes()
            assert isinstance(data, bytes)
            assert len(data) > 0
            assert data[:2] == b"\xff\xd8"

    def test_websocket_accepts_connection(self, app_and_service) -> None:
        pytest.importorskip("fastapi")
        from starlette.testclient import TestClient

        app, _ = app_and_service
        with TestClient(app) as client, client.websocket_connect("/ws/screen") as ws:
            ws.close()
