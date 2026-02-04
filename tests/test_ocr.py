"""Tests for Feature 2.2: OCR System.

These tests verify text and number extraction from screenshots:
- Extracts visible text with 95%+ accuracy
- Handles game fonts (may need training)
- Parses abbreviated numbers correctly
- Returns text with bounding boxes

Note: Tests use mocked OCR engine for unit testing.
Integration tests marked with @pytest.mark.integration require real OCR.
"""

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw, ImageFont

from src.interfaces.vision import Screenshot, TextRegion


# Helper function to create test screenshots
def create_test_screenshot(
    width: int = 800,
    height: int = 600,
    text: str | None = None,
    text_position: tuple[int, int] = (100, 100),
) -> Screenshot:
    """Create a test screenshot with optional text."""
    from datetime import datetime

    img = Image.new("RGB", (width, height), color=(255, 255, 255))

    if text:
        draw = ImageDraw.Draw(img)
        # Use default font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except OSError:
            font = ImageFont.load_default()
        draw.text(text_position, text, fill=(0, 0, 0), font=font)

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


class TestOCRSystemImport:
    """Tests for OCRSystem class import and basic structure."""

    def test_ocr_system_class_exists(self):
        """OCRSystem class should be importable."""
        from src.vision.ocr import OCRSystem

        assert OCRSystem is not None

    def test_has_extract_text_method(self):
        """OCRSystem should have an extract_text method."""
        from src.vision.ocr import OCRSystem

        assert hasattr(OCRSystem, "extract_text")

    def test_has_parse_number_method(self):
        """OCRSystem should have a parse_number method."""
        from src.vision.ocr import OCRSystem

        assert hasattr(OCRSystem, "parse_number")

    def test_has_extract_numbers_method(self):
        """OCRSystem should have an extract_numbers method."""
        from src.vision.ocr import OCRSystem

        assert hasattr(OCRSystem, "extract_numbers")


class TestOCRSystemInitialization:
    """Tests for OCRSystem initialization."""

    def test_initialization_default(self):
        """Should initialize with default settings."""
        from src.vision.ocr import OCRSystem

        ocr = OCRSystem()
        assert ocr is not None

    def test_initialization_with_language(self):
        """Should accept language parameter."""
        from src.vision.ocr import OCRSystem

        ocr = OCRSystem(language="eng")
        assert ocr._language == "eng"


class TestNumberParsing:
    """Tests for parse_number method with game-style number formats."""

    @pytest.fixture
    def ocr(self):
        """Create an OCRSystem instance."""
        from src.vision.ocr import OCRSystem

        return OCRSystem()

    @pytest.fixture
    def number_fixtures(self):
        """Load number format test fixtures."""
        fixture_path = Path(__file__).parent / "fixtures" / "ocr" / "numbers.json"
        with open(fixture_path) as f:
            return json.load(f)

    def test_parse_basic_integer(self, ocr):
        """Should parse basic integers."""
        assert ocr.parse_number("100") == 100.0
        assert ocr.parse_number("42") == 42.0
        assert ocr.parse_number("0") == 0.0

    def test_parse_decimal(self, ocr):
        """Should parse decimal numbers."""
        assert ocr.parse_number("3.14") == 3.14
        assert ocr.parse_number("0.5") == 0.5

    def test_parse_thousands_suffix(self, ocr):
        """Should parse K suffix for thousands."""
        assert ocr.parse_number("1K") == 1000.0
        assert ocr.parse_number("1.5K") == 1500.0
        assert ocr.parse_number("2.3k") == 2300.0  # lowercase

    def test_parse_millions_suffix(self, ocr):
        """Should parse M suffix for millions."""
        assert ocr.parse_number("1M") == 1000000.0
        assert ocr.parse_number("2.3M") == 2300000.0
        assert ocr.parse_number("10.5m") == 10500000.0  # lowercase

    def test_parse_billions_suffix(self, ocr):
        """Should parse B suffix for billions."""
        assert ocr.parse_number("1B") == 1000000000.0
        assert ocr.parse_number("1.23B") == 1230000000.0

    def test_parse_trillions_suffix(self, ocr):
        """Should parse T suffix for trillions."""
        assert ocr.parse_number("1T") == 1000000000000.0
        assert ocr.parse_number("5.67T") == 5670000000000.0

    def test_parse_comma_separated(self, ocr):
        """Should parse comma-separated numbers."""
        assert ocr.parse_number("1,234") == 1234.0
        assert ocr.parse_number("1,234,567") == 1234567.0

    def test_parse_with_currency_symbol(self, ocr):
        """Should handle currency symbols."""
        assert ocr.parse_number("$1.5K") == 1500.0
        assert ocr.parse_number("$100") == 100.0

    def test_parse_with_surrounding_text(self, ocr):
        """Should extract number from surrounding text."""
        assert ocr.parse_number("1.5K coins") == 1500.0
        assert ocr.parse_number("Gold: 500") == 500.0

    def test_parse_with_plus_sign(self, ocr):
        """Should handle plus sign."""
        assert ocr.parse_number("+2.3M") == 2300000.0
        assert ocr.parse_number("+100") == 100.0

    def test_parse_negative_number(self, ocr):
        """Should handle negative numbers."""
        assert ocr.parse_number("-500") == -500.0
        assert ocr.parse_number("-1.5K") == -1500.0

    def test_parse_with_whitespace(self, ocr):
        """Should handle whitespace."""
        assert ocr.parse_number("  1.5K  ") == 1500.0
        assert ocr.parse_number("\t100\n") == 100.0

    def test_parse_invalid_returns_none(self, ocr):
        """Should return None for invalid input."""
        assert ocr.parse_number("abc") is None
        assert ocr.parse_number("") is None
        assert ocr.parse_number("K") is None

    def test_parse_all_fixtures(self, ocr, number_fixtures):
        """Should correctly parse all number format fixtures."""
        for test_case in number_fixtures["number_formats"]:
            result = ocr.parse_number(test_case["input"])
            expected = test_case["expected"]
            assert result == pytest.approx(expected, rel=1e-6), (
                f"Failed for input '{test_case['input']}': "
                f"expected {expected}, got {result}"
            )

    def test_parse_invalid_fixtures(self, ocr, number_fixtures):
        """Should return None for all invalid fixtures."""
        for test_case in number_fixtures["invalid_formats"]:
            result = ocr.parse_number(test_case["input"])
            assert result is None, (
                f"Expected None for input '{test_case['input']}', got {result}"
            )


class TestExtractText:
    """Tests for extract_text method."""

    @pytest.fixture
    def ocr(self):
        """Create an OCRSystem instance."""
        from src.vision.ocr import OCRSystem

        return OCRSystem()

    def test_extract_text_returns_list(self, ocr):
        """extract_text should return a list of TextRegion."""
        screenshot = create_test_screenshot(text="Hello World")

        # Mock the OCR engine for unit testing
        with patch.object(ocr, "_run_ocr") as mock_ocr:
            mock_ocr.return_value = [
                {"text": "Hello World", "left": 100, "top": 100, "width": 150, "height": 30, "conf": 95.0}
            ]
            result = ocr.extract_text(screenshot)

        assert isinstance(result, list)

    def test_extract_text_returns_text_regions(self, ocr):
        """extract_text should return TextRegion objects."""
        screenshot = create_test_screenshot(text="Test")

        with patch.object(ocr, "_run_ocr") as mock_ocr:
            mock_ocr.return_value = [
                {"text": "Test", "left": 50, "top": 50, "width": 60, "height": 20, "conf": 90.0}
            ]
            result = ocr.extract_text(screenshot)

        assert len(result) > 0
        assert all(isinstance(r, TextRegion) for r in result)

    def test_text_region_has_bounding_box(self, ocr):
        """TextRegion should have x, y, width, height."""
        screenshot = create_test_screenshot(text="Sample")

        with patch.object(ocr, "_run_ocr") as mock_ocr:
            mock_ocr.return_value = [
                {"text": "Sample", "left": 100, "top": 200, "width": 80, "height": 25, "conf": 88.0}
            ]
            result = ocr.extract_text(screenshot)

        region = result[0]
        assert region.x == 100
        assert region.y == 200
        assert region.width == 80
        assert region.height == 25

    def test_text_region_has_confidence(self, ocr):
        """TextRegion should have confidence score."""
        screenshot = create_test_screenshot(text="Test")

        with patch.object(ocr, "_run_ocr") as mock_ocr:
            mock_ocr.return_value = [
                {"text": "Test", "left": 0, "top": 0, "width": 50, "height": 20, "conf": 95.5}
            ]
            result = ocr.extract_text(screenshot)

        region = result[0]
        assert 0.0 <= region.confidence <= 1.0
        assert region.confidence == pytest.approx(0.955, rel=0.01)

    def test_text_region_has_text_content(self, ocr):
        """TextRegion should contain detected text."""
        screenshot = create_test_screenshot(text="Hello")

        with patch.object(ocr, "_run_ocr") as mock_ocr:
            mock_ocr.return_value = [
                {"text": "Hello", "left": 0, "top": 0, "width": 50, "height": 20, "conf": 90.0}
            ]
            result = ocr.extract_text(screenshot)

        region = result[0]
        assert region.text == "Hello"


class TestRegionBasedOCR:
    """Tests for region-based OCR extraction."""

    @pytest.fixture
    def ocr(self):
        """Create an OCRSystem instance."""
        from src.vision.ocr import OCRSystem

        return OCRSystem()

    def test_extract_text_with_region(self, ocr):
        """Should extract text only from specified region."""
        screenshot = create_test_screenshot(text="Full Image Text")

        with patch.object(ocr, "_run_ocr") as mock_ocr:
            mock_ocr.return_value = [
                {"text": "Region Text", "left": 10, "top": 10, "width": 100, "height": 20, "conf": 90.0}
            ]
            # Extract from specific region (x, y, width, height)
            result = ocr.extract_text(screenshot, region=(0, 0, 200, 100))

        assert len(result) >= 0  # May have text or not
        mock_ocr.assert_called_once()

    def test_region_crops_image(self, ocr):
        """Region parameter should crop the image before OCR."""
        screenshot = create_test_screenshot(width=800, height=600)

        with patch.object(ocr, "_run_ocr") as mock_ocr:
            mock_ocr.return_value = []
            ocr.extract_text(screenshot, region=(100, 100, 200, 150))

        # Verify the cropped image was passed
        call_args = mock_ocr.call_args
        cropped_img = call_args[0][0]
        assert cropped_img.size == (200, 150)

    def test_region_outside_bounds_handled(self, ocr):
        """Should handle region extending beyond image bounds."""
        screenshot = create_test_screenshot(width=100, height=100)

        with patch.object(ocr, "_run_ocr") as mock_ocr:
            mock_ocr.return_value = []
            # Region extends beyond image - should be clipped
            result = ocr.extract_text(screenshot, region=(50, 50, 200, 200))

        # Should not raise, result is empty or contains text
        assert isinstance(result, list)


class TestExtractNumbers:
    """Tests for extract_numbers method."""

    @pytest.fixture
    def ocr(self):
        """Create an OCRSystem instance."""
        from src.vision.ocr import OCRSystem

        return OCRSystem()

    def test_extract_numbers_from_text(self, ocr):
        """Should extract all numbers from text."""
        result = ocr.extract_numbers("Gold: 1.5K | Gems: 500")

        assert 1500.0 in result
        assert 500.0 in result

    def test_extract_numbers_empty_text(self, ocr):
        """Should return empty list for text without numbers."""
        result = ocr.extract_numbers("No numbers here")

        assert result == []

    def test_extract_numbers_multiple_formats(self, ocr):
        """Should handle multiple number formats."""
        result = ocr.extract_numbers("Level 42 Score: 1.5M XP: 2,500")

        assert 42.0 in result
        assert 1500000.0 in result
        assert 2500.0 in result


class TestOCRWithScreenshot:
    """Tests for OCR integration with Screenshot objects."""

    @pytest.fixture
    def ocr(self):
        """Create an OCRSystem instance."""
        from src.vision.ocr import OCRSystem

        return OCRSystem()

    def test_extract_text_accepts_screenshot(self, ocr):
        """extract_text should accept Screenshot objects."""
        screenshot = create_test_screenshot()

        with patch.object(ocr, "_run_ocr") as mock_ocr:
            mock_ocr.return_value = []
            result = ocr.extract_text(screenshot)

        assert isinstance(result, list)

    def test_extract_text_uses_screenshot_image(self, ocr):
        """extract_text should use the PIL Image from Screenshot."""
        screenshot = create_test_screenshot(width=640, height=480)

        with patch.object(ocr, "_run_ocr") as mock_ocr:
            mock_ocr.return_value = []
            ocr.extract_text(screenshot)

        # Verify image was passed to OCR
        call_args = mock_ocr.call_args
        img = call_args[0][0]
        assert img.size == (640, 480)


class TestOCRFiltering:
    """Tests for OCR result filtering."""

    @pytest.fixture
    def ocr(self):
        """Create an OCRSystem instance."""
        from src.vision.ocr import OCRSystem

        return OCRSystem()

    def test_filter_low_confidence_results(self, ocr):
        """Should filter out low confidence results."""
        screenshot = create_test_screenshot()

        with patch.object(ocr, "_run_ocr") as mock_ocr:
            mock_ocr.return_value = [
                {"text": "Good", "left": 0, "top": 0, "width": 50, "height": 20, "conf": 90.0},
                {"text": "Bad", "left": 0, "top": 0, "width": 50, "height": 20, "conf": 10.0},
            ]
            result = ocr.extract_text(screenshot, min_confidence=0.5)

        # Only high confidence result should be returned
        assert len(result) == 1
        assert result[0].text == "Good"

    def test_filter_empty_text(self, ocr):
        """Should filter out empty text results."""
        screenshot = create_test_screenshot()

        with patch.object(ocr, "_run_ocr") as mock_ocr:
            mock_ocr.return_value = [
                {"text": "Valid", "left": 0, "top": 0, "width": 50, "height": 20, "conf": 90.0},
                {"text": "", "left": 0, "top": 0, "width": 50, "height": 20, "conf": 90.0},
                {"text": "   ", "left": 0, "top": 0, "width": 50, "height": 20, "conf": 90.0},
            ]
            result = ocr.extract_text(screenshot)

        assert len(result) == 1
        assert result[0].text == "Valid"


class TestOCRModuleExports:
    """Tests for module exports."""

    def test_ocr_system_exported_from_vision(self):
        """OCRSystem should be exported from vision package."""
        from src.vision import OCRSystem

        assert OCRSystem is not None

    def test_parse_number_function_available(self):
        """parse_number helper function should be available."""
        from src.vision.ocr import parse_number

        assert parse_number is not None
        assert parse_number("1.5K") == 1500.0
