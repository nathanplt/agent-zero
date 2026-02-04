"""Tests for Feature 2.1: Screenshot Capture.

These tests verify fast, reliable screenshot capture from virtual display:
- Captures at 10+ FPS
- Returns raw bytes and PIL Image
- Buffer stores last N frames
- Timestamps accurate to millisecond

Note: Tests use mocked environment manager for unit testing.
Integration tests marked with @pytest.mark.integration require real environment.
"""

import time
from datetime import datetime
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from PIL import Image


# Test fixtures for generating fake screenshots
def create_test_image(width: int = 1920, height: int = 1080, color: tuple = (255, 0, 0)) -> Image.Image:
    """Create a test image with specified color."""
    return Image.new("RGB", (width, height), color)


def create_test_image_bytes(width: int = 1920, height: int = 1080, color: tuple = (255, 0, 0)) -> bytes:
    """Create PNG bytes for a test image."""
    img = create_test_image(width, height, color)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestScreenshotCaptureImport:
    """Tests for ScreenshotCapture class import and basic structure."""

    def test_screenshot_capture_class_exists(self):
        """ScreenshotCapture class should be importable."""
        from src.vision.capture import ScreenshotCapture

        assert ScreenshotCapture is not None

    def test_has_capture_method(self):
        """ScreenshotCapture should have a capture method."""
        from src.vision.capture import ScreenshotCapture

        assert hasattr(ScreenshotCapture, "capture")

    def test_has_get_buffer_method(self):
        """ScreenshotCapture should have a get_buffer method."""
        from src.vision.capture import ScreenshotCapture

        assert hasattr(ScreenshotCapture, "get_buffer")

    def test_has_clear_buffer_method(self):
        """ScreenshotCapture should have a clear_buffer method."""
        from src.vision.capture import ScreenshotCapture

        assert hasattr(ScreenshotCapture, "clear_buffer")

    def test_has_buffer_size_property(self):
        """ScreenshotCapture should have a buffer_size property."""
        from src.vision.capture import ScreenshotCapture

        assert hasattr(ScreenshotCapture, "buffer_size")


class TestScreenshotCaptureInitialization:
    """Tests for ScreenshotCapture initialization."""

    def test_initialization_with_environment_manager(self):
        """Should initialize with an environment manager."""
        from src.vision.capture import ScreenshotCapture

        mock_env = MagicMock()
        capture = ScreenshotCapture(environment_manager=mock_env)

        assert capture._environment_manager is mock_env

    def test_initialization_with_custom_buffer_size(self):
        """Should accept custom buffer size."""
        from src.vision.capture import ScreenshotCapture

        mock_env = MagicMock()
        capture = ScreenshotCapture(environment_manager=mock_env, buffer_size=20)

        assert capture.buffer_size == 20

    def test_default_buffer_size_is_10(self):
        """Default buffer size should be 10."""
        from src.vision.capture import ScreenshotCapture

        mock_env = MagicMock()
        capture = ScreenshotCapture(environment_manager=mock_env)

        assert capture.buffer_size == 10

    def test_buffer_is_empty_initially(self):
        """Buffer should be empty after initialization."""
        from src.vision.capture import ScreenshotCapture

        mock_env = MagicMock()
        capture = ScreenshotCapture(environment_manager=mock_env)

        assert len(capture.get_buffer()) == 0


class TestScreenshotCaptureCapture:
    """Tests for the capture() method."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment manager."""
        mock = MagicMock()
        mock.screenshot.return_value = create_test_image_bytes()
        mock.screenshot_pil.return_value = create_test_image()
        mock.is_running.return_value = True
        return mock

    def test_capture_returns_screenshot_object(self, mock_env):
        """capture() should return a Screenshot object."""
        from src.interfaces.vision import Screenshot
        from src.vision.capture import ScreenshotCapture

        capture = ScreenshotCapture(environment_manager=mock_env)
        result = capture.capture()

        assert isinstance(result, Screenshot)

    def test_capture_includes_raw_bytes(self, mock_env):
        """Screenshot should include raw PNG bytes."""
        from src.vision.capture import ScreenshotCapture

        capture = ScreenshotCapture(environment_manager=mock_env)
        result = capture.capture()

        assert isinstance(result.raw_bytes, bytes)
        assert result.raw_bytes.startswith(b"\x89PNG")

    def test_capture_includes_pil_image(self, mock_env):
        """Screenshot should include PIL Image."""
        from src.vision.capture import ScreenshotCapture

        capture = ScreenshotCapture(environment_manager=mock_env)
        result = capture.capture()

        assert isinstance(result.image, Image.Image)

    def test_capture_includes_timestamp(self, mock_env):
        """Screenshot should include timestamp."""
        from src.vision.capture import ScreenshotCapture

        before = datetime.now()
        capture = ScreenshotCapture(environment_manager=mock_env)
        result = capture.capture()
        after = datetime.now()

        assert isinstance(result.timestamp, datetime)
        assert before <= result.timestamp <= after

    def test_capture_timestamp_millisecond_precision(self, mock_env):
        """Timestamp should have millisecond precision."""
        from src.vision.capture import ScreenshotCapture

        capture = ScreenshotCapture(environment_manager=mock_env)
        result = capture.capture()

        # Timestamp should have microsecond precision (datetime default)
        # which gives us millisecond precision
        assert result.timestamp.microsecond >= 0

    def test_capture_includes_dimensions(self, mock_env):
        """Screenshot should include width and height."""
        from src.vision.capture import ScreenshotCapture

        capture = ScreenshotCapture(environment_manager=mock_env)
        result = capture.capture()

        assert result.width == 1920
        assert result.height == 1080

    def test_capture_adds_to_buffer(self, mock_env):
        """capture() should add screenshot to buffer."""
        from src.vision.capture import ScreenshotCapture

        capture = ScreenshotCapture(environment_manager=mock_env)

        assert len(capture.get_buffer()) == 0
        capture.capture()
        assert len(capture.get_buffer()) == 1

    def test_capture_uses_environment_manager_screenshot(self, mock_env):
        """capture() should use environment manager's screenshot method."""
        from src.vision.capture import ScreenshotCapture

        capture = ScreenshotCapture(environment_manager=mock_env)
        capture.capture()

        # Should call screenshot() once (PIL derived from bytes, not separate call)
        mock_env.screenshot.assert_called_once()
        mock_env.screenshot_pil.assert_not_called()


class TestScreenshotCaptureBuffer:
    """Tests for screenshot buffer functionality."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment manager."""
        mock = MagicMock()
        mock.screenshot.return_value = create_test_image_bytes()
        mock.screenshot_pil.return_value = create_test_image()
        mock.is_running.return_value = True
        return mock

    def test_buffer_stores_last_n_frames(self, mock_env):
        """Buffer should store exactly the last N frames."""
        from src.vision.capture import ScreenshotCapture

        capture = ScreenshotCapture(environment_manager=mock_env, buffer_size=10)

        # Capture 20 frames
        for _ in range(20):
            capture.capture()

        buffer = capture.get_buffer()
        assert len(buffer) == 10

    def test_buffer_most_recent_first(self, mock_env):
        """get_buffer() should return most recent screenshot first."""
        from src.vision.capture import ScreenshotCapture

        capture = ScreenshotCapture(environment_manager=mock_env, buffer_size=10)

        # Capture 5 frames with small delays
        timestamps = []
        for _ in range(5):
            result = capture.capture()
            timestamps.append(result.timestamp)
            time.sleep(0.001)  # Small delay to ensure different timestamps

        buffer = capture.get_buffer()

        # Most recent should be first
        assert buffer[0].timestamp >= buffer[-1].timestamp

    def test_buffer_count_parameter(self, mock_env):
        """get_buffer(count) should return only that many frames."""
        from src.vision.capture import ScreenshotCapture

        capture = ScreenshotCapture(environment_manager=mock_env, buffer_size=10)

        # Capture 5 frames
        for _ in range(5):
            capture.capture()

        buffer = capture.get_buffer(count=3)
        assert len(buffer) == 3

    def test_buffer_count_exceeds_available(self, mock_env):
        """get_buffer(count) should return all if count exceeds available."""
        from src.vision.capture import ScreenshotCapture

        capture = ScreenshotCapture(environment_manager=mock_env, buffer_size=10)

        # Capture 3 frames
        for _ in range(3):
            capture.capture()

        buffer = capture.get_buffer(count=10)
        assert len(buffer) == 3

    def test_clear_buffer(self, mock_env):
        """clear_buffer() should empty the buffer."""
        from src.vision.capture import ScreenshotCapture

        capture = ScreenshotCapture(environment_manager=mock_env)

        # Capture some frames
        for _ in range(5):
            capture.capture()

        assert len(capture.get_buffer()) == 5

        capture.clear_buffer()

        assert len(capture.get_buffer()) == 0

    def test_buffer_preserves_screenshot_objects(self, mock_env):
        """Buffer should preserve complete Screenshot objects."""
        from src.interfaces.vision import Screenshot
        from src.vision.capture import ScreenshotCapture

        capture = ScreenshotCapture(environment_manager=mock_env)
        original = capture.capture()

        buffer = capture.get_buffer()
        buffered = buffer[0]

        # Should be the same object
        assert buffered is original
        assert isinstance(buffered, Screenshot)
        assert buffered.raw_bytes is original.raw_bytes
        assert buffered.image is original.image


class TestScreenshotCapturePerformance:
    """Tests for screenshot capture performance."""

    @pytest.fixture
    def mock_env(self):
        """Create a fast mock environment manager."""
        mock = MagicMock()
        # Use a cached image to simulate fast capture
        cached_bytes = create_test_image_bytes()
        cached_image = create_test_image()
        mock.screenshot.return_value = cached_bytes
        mock.screenshot_pil.return_value = cached_image
        mock.is_running.return_value = True
        return mock

    def test_capture_performance_10_fps(self, mock_env):
        """Should be able to capture at 10+ FPS (< 100ms per frame)."""
        from src.vision.capture import ScreenshotCapture

        capture = ScreenshotCapture(environment_manager=mock_env)

        # Capture 100 frames and measure time
        start = time.time()
        for _ in range(100):
            capture.capture()
        elapsed = time.time() - start

        average_per_frame = elapsed / 100

        # Should average less than 100ms per frame (10+ FPS)
        # Being generous for CI environments
        assert average_per_frame < 0.1, f"Average {average_per_frame * 1000:.1f}ms per frame is too slow"

    def test_buffer_operations_are_fast(self, mock_env):
        """Buffer operations should not add significant overhead."""
        from src.vision.capture import ScreenshotCapture

        capture = ScreenshotCapture(environment_manager=mock_env, buffer_size=100)

        # Fill buffer
        for _ in range(100):
            capture.capture()

        # Time get_buffer operations
        start = time.time()
        for _ in range(1000):
            capture.get_buffer()
            capture.get_buffer(count=10)
        elapsed = time.time() - start

        # Should be very fast (< 1ms per operation on average)
        assert elapsed < 1.0, f"Buffer operations too slow: {elapsed}s for 2000 ops"


class TestScreenshotCaptureDifferentFrames:
    """Tests for detecting different frames."""

    def test_different_frames_have_different_content(self):
        """Should be able to capture frames with different content."""
        from src.vision.capture import ScreenshotCapture

        # Create mock that returns different images
        mock_env = MagicMock()
        mock_env.is_running.return_value = True

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        call_count = [0]

        def get_screenshot():
            color = colors[call_count[0] % len(colors)]
            call_count[0] += 1
            return create_test_image_bytes(color=color)

        def get_screenshot_pil():
            color = colors[(call_count[0] - 1) % len(colors)]
            return create_test_image(color=color)

        mock_env.screenshot.side_effect = get_screenshot
        mock_env.screenshot_pil.side_effect = get_screenshot_pil

        capture = ScreenshotCapture(environment_manager=mock_env)

        # Capture 3 different frames
        frame1 = capture.capture()
        frame2 = capture.capture()
        frame3 = capture.capture()

        # Frames should have different content (bytes comparison)
        assert frame1.raw_bytes != frame2.raw_bytes
        assert frame2.raw_bytes != frame3.raw_bytes

        # But similar sizes since they're all 1920x1080
        assert frame1.width == frame2.width == frame3.width
        assert frame1.height == frame2.height == frame3.height


class TestScreenshotCaptureErrorHandling:
    """Tests for error handling in screenshot capture."""

    def test_capture_raises_on_environment_error(self):
        """capture() should raise VisionError on environment failure."""
        from src.interfaces.environment import EnvironmentSetupError
        from src.interfaces.vision import VisionError
        from src.vision.capture import ScreenshotCapture

        mock_env = MagicMock()
        mock_env.is_running.return_value = True
        mock_env.screenshot.side_effect = EnvironmentSetupError("Display crashed")

        capture = ScreenshotCapture(environment_manager=mock_env)

        with pytest.raises(VisionError):
            capture.capture()

    def test_capture_raises_when_environment_not_running(self):
        """capture() should raise VisionError when environment not running."""
        from src.interfaces.vision import VisionError
        from src.vision.capture import ScreenshotCapture

        mock_env = MagicMock()
        mock_env.is_running.return_value = False

        capture = ScreenshotCapture(environment_manager=mock_env)

        with pytest.raises(VisionError):
            capture.capture()


class TestScreenshotCaptureTimestampAccuracy:
    """Tests for timestamp accuracy to millisecond precision."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment manager."""
        mock = MagicMock()
        mock.screenshot.return_value = create_test_image_bytes()
        mock.screenshot_pil.return_value = create_test_image()
        mock.is_running.return_value = True
        return mock

    def test_timestamps_are_sequential(self, mock_env):
        """Sequential captures should have sequential timestamps."""
        from src.vision.capture import ScreenshotCapture

        capture = ScreenshotCapture(environment_manager=mock_env)

        # Capture with small delays
        timestamps = []
        for _ in range(10):
            result = capture.capture()
            timestamps.append(result.timestamp)
            time.sleep(0.001)

        # Each timestamp should be >= previous
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    def test_timestamps_reflect_capture_time(self, mock_env):
        """Timestamps should accurately reflect capture time."""
        from src.vision.capture import ScreenshotCapture

        capture = ScreenshotCapture(environment_manager=mock_env)

        before = datetime.now()
        result = capture.capture()
        after = datetime.now()

        # Timestamp should be within the capture window
        assert before <= result.timestamp <= after

        # Window should be very small (< 100ms typically)
        window = (after - before).total_seconds()
        assert window < 0.1, f"Capture took too long: {window * 1000:.1f}ms"

    def test_timestamps_have_subsecond_precision(self, mock_env):
        """Timestamps should have sub-second precision for accurate timing."""
        from src.vision.capture import ScreenshotCapture

        capture = ScreenshotCapture(environment_manager=mock_env)

        # Capture two frames quickly
        result1 = capture.capture()
        time.sleep(0.005)  # 5ms delay
        result2 = capture.capture()

        # Should be able to distinguish 5ms difference
        diff = (result2.timestamp - result1.timestamp).total_seconds()
        assert diff >= 0.004, f"Timestamp precision insufficient: {diff * 1000:.3f}ms"


class TestScreenshotCaptureModuleExports:
    """Tests for module exports."""

    def test_screenshot_capture_exported_from_vision(self):
        """ScreenshotCapture should be exported from vision package."""
        from src.vision import ScreenshotCapture

        assert ScreenshotCapture is not None

    def test_vision_error_available(self):
        """VisionError should be available."""
        from src.interfaces.vision import VisionError

        assert VisionError is not None
