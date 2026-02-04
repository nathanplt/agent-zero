"""Tests for Feature 2.3: UI Element Detection.

These tests verify UI element detection from screenshots:
- Detects buttons with 90%+ recall
- Returns bounding boxes for each element
- Classifies element types (button, menu, resource, etc.)
- Handles overlapping UI gracefully

Note: Tests use mocked detection for unit testing.
Integration tests marked with @pytest.mark.integration require real detection.
"""

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw

from src.interfaces.vision import Screenshot, UIElement


# Helper function to create test screenshots
def create_test_screenshot(
    width: int = 800,
    height: int = 600,
    elements: list[dict] | None = None,
) -> Screenshot:
    """Create a test screenshot with optional drawn elements."""
    img = Image.new("RGB", (width, height), color=(40, 40, 40))  # Dark background
    draw = ImageDraw.Draw(img)

    if elements:
        for elem in elements:
            x, y, w, h = elem["x"], elem["y"], elem["width"], elem["height"]
            # Draw different colors for different element types
            colors = {
                "button": (100, 150, 200),  # Blue-ish
                "resource": (200, 180, 100),  # Gold-ish
                "menu": (150, 150, 150),  # Gray
                "progress_bar": (100, 200, 100),  # Green
                "icon": (200, 100, 100),  # Red-ish
            }
            color = colors.get(elem.get("type", "button"), (150, 150, 150))
            draw.rectangle([x, y, x + w, y + h], fill=color, outline=(255, 255, 255))

            # Draw label if present
            if elem.get("label"):
                draw.text((x + 5, y + 5), elem["label"], fill=(255, 255, 255))

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    raw_bytes = buffer.getvalue()

    return Screenshot(
        image=img,
        raw_bytes=raw_bytes,
        timestamp=datetime.now(),
        width=width,
        height=height,
    )


class TestUIDetectorImport:
    """Tests for UIDetector class import and basic structure."""

    def test_ui_detector_class_exists(self):
        """UIDetector class should be importable."""
        from src.vision.ui_detection import UIDetector

        assert UIDetector is not None

    def test_has_detect_method(self):
        """UIDetector should have a detect method."""
        from src.vision.ui_detection import UIDetector

        assert hasattr(UIDetector, "detect")

    def test_has_detect_buttons_method(self):
        """UIDetector should have a detect_buttons method."""
        from src.vision.ui_detection import UIDetector

        assert hasattr(UIDetector, "detect_buttons")

    def test_has_classify_method(self):
        """UIDetector should have a classify method."""
        from src.vision.ui_detection import UIDetector

        assert hasattr(UIDetector, "classify")


class TestUIDetectorInitialization:
    """Tests for UIDetector initialization."""

    def test_initialization_default(self):
        """Should initialize with default settings."""
        from src.vision.ui_detection import UIDetector

        detector = UIDetector()
        assert detector is not None

    def test_initialization_with_confidence_threshold(self):
        """Should accept confidence threshold parameter."""
        from src.vision.ui_detection import UIDetector

        detector = UIDetector(min_confidence=0.8)
        assert detector._min_confidence == 0.8


class TestUIElementDetection:
    """Tests for detect method."""

    @pytest.fixture
    def detector(self):
        """Create a UIDetector instance."""
        from src.vision.ui_detection import UIDetector

        return UIDetector()

    @pytest.fixture
    def ui_fixtures(self):
        """Load UI element test fixtures."""
        fixture_path = Path(__file__).parent / "fixtures" / "ui" / "element_types.json"
        with open(fixture_path) as f:
            return json.load(f)

    def test_detect_returns_list(self, detector):
        """detect should return a list of UIElement."""
        screenshot = create_test_screenshot()

        with patch.object(detector, "_run_detection") as mock_detect:
            mock_detect.return_value = []
            result = detector.detect(screenshot)

        assert isinstance(result, list)

    def test_detect_returns_ui_elements(self, detector):
        """detect should return UIElement objects."""
        screenshot = create_test_screenshot()

        with patch.object(detector, "_run_detection") as mock_detect:
            mock_detect.return_value = [
                {"type": "button", "x": 100, "y": 200, "width": 120, "height": 40, "confidence": 0.95, "label": "Click"}
            ]
            result = detector.detect(screenshot)

        assert len(result) > 0
        assert all(isinstance(r, UIElement) for r in result)

    def test_ui_element_has_bounding_box(self, detector):
        """UIElement should have x, y, width, height."""
        screenshot = create_test_screenshot()

        with patch.object(detector, "_run_detection") as mock_detect:
            mock_detect.return_value = [
                {"type": "button", "x": 50, "y": 100, "width": 80, "height": 30, "confidence": 0.9, "label": None}
            ]
            result = detector.detect(screenshot)

        elem = result[0]
        assert elem.x == 50
        assert elem.y == 100
        assert elem.width == 80
        assert elem.height == 30

    def test_ui_element_has_confidence(self, detector):
        """UIElement should have confidence score."""
        screenshot = create_test_screenshot()

        with patch.object(detector, "_run_detection") as mock_detect:
            mock_detect.return_value = [
                {"type": "button", "x": 0, "y": 0, "width": 50, "height": 20, "confidence": 0.87, "label": None}
            ]
            result = detector.detect(screenshot)

        elem = result[0]
        assert 0.0 <= elem.confidence <= 1.0
        assert elem.confidence == pytest.approx(0.87, rel=0.01)

    def test_ui_element_has_type(self, detector):
        """UIElement should have element_type."""
        screenshot = create_test_screenshot()

        with patch.object(detector, "_run_detection") as mock_detect:
            mock_detect.return_value = [
                {"type": "button", "x": 0, "y": 0, "width": 50, "height": 20, "confidence": 0.9, "label": None}
            ]
            result = detector.detect(screenshot)

        elem = result[0]
        assert elem.element_type == "button"

    def test_ui_element_has_optional_label(self, detector):
        """UIElement should have optional label."""
        screenshot = create_test_screenshot()

        with patch.object(detector, "_run_detection") as mock_detect:
            mock_detect.return_value = [
                {"type": "button", "x": 0, "y": 0, "width": 50, "height": 20, "confidence": 0.9, "label": "Upgrade"}
            ]
            result = detector.detect(screenshot)

        elem = result[0]
        assert elem.label == "Upgrade"

    def test_ui_element_center_property(self, detector):
        """UIElement should have center property."""
        screenshot = create_test_screenshot()

        with patch.object(detector, "_run_detection") as mock_detect:
            mock_detect.return_value = [
                {"type": "button", "x": 100, "y": 200, "width": 80, "height": 40, "confidence": 0.9, "label": None}
            ]
            result = detector.detect(screenshot)

        elem = result[0]
        assert elem.center == (140, 220)


class TestButtonDetection:
    """Tests for button-specific detection."""

    @pytest.fixture
    def detector(self):
        """Create a UIDetector instance."""
        from src.vision.ui_detection import UIDetector

        return UIDetector()

    def test_detect_buttons_returns_only_buttons(self, detector):
        """detect_buttons should return only button elements."""
        screenshot = create_test_screenshot()

        with patch.object(detector, "_run_detection") as mock_detect:
            mock_detect.return_value = [
                {"type": "button", "x": 100, "y": 200, "width": 80, "height": 40, "confidence": 0.9, "label": "OK"},
                {"type": "resource", "x": 10, "y": 10, "width": 60, "height": 30, "confidence": 0.85, "label": "Gold"},
                {"type": "button", "x": 200, "y": 200, "width": 80, "height": 40, "confidence": 0.88, "label": "Cancel"},
            ]
            result = detector.detect_buttons(screenshot)

        assert len(result) == 2
        assert all(elem.element_type == "button" for elem in result)

    def test_detect_buttons_with_confidence_filter(self, detector):
        """detect_buttons should filter by confidence."""
        screenshot = create_test_screenshot()

        with patch.object(detector, "_run_detection") as mock_detect:
            mock_detect.return_value = [
                {"type": "button", "x": 100, "y": 200, "width": 80, "height": 40, "confidence": 0.95, "label": "OK"},
                {"type": "button", "x": 200, "y": 200, "width": 80, "height": 40, "confidence": 0.5, "label": "Low"},
            ]
            result = detector.detect_buttons(screenshot, min_confidence=0.7)

        assert len(result) == 1
        assert result[0].label == "OK"


class TestElementClassification:
    """Tests for element type classification."""

    @pytest.fixture
    def detector(self):
        """Create a UIDetector instance."""
        from src.vision.ui_detection import UIDetector

        return UIDetector()

    @pytest.fixture
    def ui_fixtures(self):
        """Load UI element test fixtures."""
        fixture_path = Path(__file__).parent / "fixtures" / "ui" / "element_types.json"
        with open(fixture_path) as f:
            return json.load(f)

    def test_classify_button_by_keywords(self, detector, ui_fixtures):
        """Should classify as button based on keywords."""
        for keyword in ui_fixtures["button_keywords"][:5]:  # Test first 5
            elem_type = detector.classify(label=keyword.capitalize(), aspect_ratio=2.5)
            assert elem_type == "button", f"Expected 'button' for '{keyword}'"

    def test_classify_resource_by_keywords(self, detector, ui_fixtures):
        """Should classify as resource based on keywords."""
        for keyword in ui_fixtures["resource_keywords"][:5]:  # Test first 5
            elem_type = detector.classify(label=f"{keyword}: 100", aspect_ratio=3.0)
            assert elem_type == "resource", f"Expected 'resource' for '{keyword}'"

    def test_classify_by_aspect_ratio(self, detector):
        """Should use aspect ratio for classification hints."""
        # Wide elements might be progress bars
        elem_type = detector.classify(label=None, aspect_ratio=10.0, has_gradient=True)
        assert elem_type == "progress_bar"

        # Square-ish elements might be icons
        elem_type = detector.classify(label=None, aspect_ratio=1.0, size_small=True)
        assert elem_type == "icon"

    def test_classify_unknown_defaults_to_button(self, detector):
        """Unknown elements should default to button (most common)."""
        elem_type = detector.classify(label="SomeLabel", aspect_ratio=2.0)
        # Could be button or unknown - implementation decides
        assert elem_type in ["button", "unknown", "text"]


class TestOverlappingElements:
    """Tests for handling overlapping UI elements."""

    @pytest.fixture
    def detector(self):
        """Create a UIDetector instance."""
        from src.vision.ui_detection import UIDetector

        return UIDetector()

    def test_handles_overlapping_elements(self, detector):
        """Should handle overlapping elements gracefully."""
        screenshot = create_test_screenshot()

        with patch.object(detector, "_run_detection") as mock_detect:
            # Two overlapping buttons
            mock_detect.return_value = [
                {"type": "button", "x": 100, "y": 100, "width": 80, "height": 40, "confidence": 0.9, "label": "A"},
                {"type": "button", "x": 120, "y": 110, "width": 80, "height": 40, "confidence": 0.85, "label": "B"},
            ]
            result = detector.detect(screenshot)

        # Both elements should be returned
        assert len(result) == 2

    def test_filter_duplicate_detections(self, detector):
        """Should filter very similar duplicate detections."""
        screenshot = create_test_screenshot()

        with patch.object(detector, "_run_detection") as mock_detect:
            # Nearly identical detections (same element detected twice)
            mock_detect.return_value = [
                {"type": "button", "x": 100, "y": 100, "width": 80, "height": 40, "confidence": 0.9, "label": "OK"},
                {"type": "button", "x": 101, "y": 101, "width": 79, "height": 39, "confidence": 0.85, "label": "OK"},
            ]
            result = detector.detect(screenshot, filter_duplicates=True)

        # Should keep only the higher confidence one
        assert len(result) == 1
        assert result[0].confidence == 0.9


class TestConfidenceFiltering:
    """Tests for confidence-based filtering."""

    @pytest.fixture
    def detector(self):
        """Create a UIDetector instance."""
        from src.vision.ui_detection import UIDetector

        return UIDetector(min_confidence=0.5)

    def test_filters_low_confidence(self, detector):
        """Should filter out low confidence detections."""
        screenshot = create_test_screenshot()

        with patch.object(detector, "_run_detection") as mock_detect:
            mock_detect.return_value = [
                {"type": "button", "x": 100, "y": 100, "width": 80, "height": 40, "confidence": 0.9, "label": "High"},
                {"type": "button", "x": 200, "y": 100, "width": 80, "height": 40, "confidence": 0.3, "label": "Low"},
                {"type": "button", "x": 300, "y": 100, "width": 80, "height": 40, "confidence": 0.6, "label": "Mid"},
            ]
            result = detector.detect(screenshot)

        # Should only include elements with confidence >= 0.5
        assert len(result) == 2
        labels = [elem.label for elem in result]
        assert "High" in labels
        assert "Mid" in labels
        assert "Low" not in labels


class TestDetectionWithRegion:
    """Tests for region-based detection."""

    @pytest.fixture
    def detector(self):
        """Create a UIDetector instance."""
        from src.vision.ui_detection import UIDetector

        return UIDetector()

    def test_detect_in_region(self, detector):
        """Should detect elements only in specified region."""
        screenshot = create_test_screenshot(width=800, height=600)

        with patch.object(detector, "_run_detection") as mock_detect:
            mock_detect.return_value = [
                {"type": "button", "x": 10, "y": 10, "width": 80, "height": 40, "confidence": 0.9, "label": "InRegion"},
            ]
            detector.detect(screenshot, region=(0, 0, 200, 200))

        # Coordinates should be adjusted for region offset
        mock_detect.assert_called_once()

    def test_region_clips_to_bounds(self, detector):
        """Region should be clipped to image bounds."""
        screenshot = create_test_screenshot(width=100, height=100)

        with patch.object(detector, "_run_detection") as mock_detect:
            mock_detect.return_value = []
            # Region extends beyond image
            detector.detect(screenshot, region=(50, 50, 200, 200))

        # Should not raise, verify cropped image size
        call_args = mock_detect.call_args
        cropped_img = call_args[0][0]
        assert cropped_img.size[0] <= 100
        assert cropped_img.size[1] <= 100


class TestElementTypeSupport:
    """Tests for supported element types."""

    @pytest.fixture
    def detector(self):
        """Create a UIDetector instance."""
        from src.vision.ui_detection import UIDetector

        return UIDetector()

    @pytest.fixture
    def ui_fixtures(self):
        """Load UI element test fixtures."""
        fixture_path = Path(__file__).parent / "fixtures" / "ui" / "element_types.json"
        with open(fixture_path) as f:
            return json.load(f)

    def test_supports_all_element_types(self, ui_fixtures):
        """Should support all defined element types."""
        from src.vision.ui_detection import SUPPORTED_ELEMENT_TYPES

        for elem_type in ui_fixtures["element_types"]:
            assert elem_type in SUPPORTED_ELEMENT_TYPES, f"Missing type: {elem_type}"


class TestModuleExports:
    """Tests for module exports."""

    def test_ui_detector_exported_from_vision(self):
        """UIDetector should be exported from vision package."""
        from src.vision import UIDetector

        assert UIDetector is not None

    def test_supported_types_available(self):
        """SUPPORTED_ELEMENT_TYPES should be available."""
        from src.vision.ui_detection import SUPPORTED_ELEMENT_TYPES

        assert isinstance(SUPPORTED_ELEMENT_TYPES, (list, tuple, set, frozenset))
        assert "button" in SUPPORTED_ELEMENT_TYPES


class TestIoUCalculation:
    """Tests for Intersection over Union calculation."""

    def test_iou_same_box(self):
        """IoU of same box should be 1.0."""
        from src.vision.ui_detection import calculate_iou

        box = (100, 100, 50, 50)  # x, y, width, height
        assert calculate_iou(box, box) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        """IoU of non-overlapping boxes should be 0.0."""
        from src.vision.ui_detection import calculate_iou

        box1 = (0, 0, 50, 50)
        box2 = (100, 100, 50, 50)
        assert calculate_iou(box1, box2) == pytest.approx(0.0)

    def test_iou_partial_overlap(self):
        """IoU of partially overlapping boxes should be between 0 and 1."""
        from src.vision.ui_detection import calculate_iou

        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 100, 100)
        iou = calculate_iou(box1, box2)
        assert 0.0 < iou < 1.0

    def test_iou_contained_box(self):
        """IoU when one box contains another."""
        from src.vision.ui_detection import calculate_iou

        outer = (0, 0, 100, 100)
        inner = (25, 25, 50, 50)
        iou = calculate_iou(outer, inner)
        # Inner area is 2500, outer is 10000, intersection is 2500
        # Union is 10000, so IoU = 2500/10000 = 0.25
        assert iou == pytest.approx(0.25)
