"""OCR system for text and number extraction from screenshots.

This module provides text extraction and number parsing capabilities:
- Extract visible text from screenshots using OCR
- Parse game-style number formats (1.5K, 2.3M, etc.)
- Region-based OCR for specific screen areas
- Returns text with bounding boxes and confidence scores

Example:
    >>> from src.vision.ocr import OCRSystem
    >>> ocr = OCRSystem()
    >>> regions = ocr.extract_text(screenshot)
    >>> for region in regions:
    ...     print(f"{region.text} at ({region.x}, {region.y})")
    ...
    >>> # Parse game numbers
    >>> value = ocr.parse_number("1.5K")  # Returns 1500.0
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from src.interfaces.vision import Screenshot, TextRegion, VisionError

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

# Number suffix multipliers
SUFFIXES = {
    "k": 1_000,
    "m": 1_000_000,
    "b": 1_000_000_000,
    "t": 1_000_000_000_000,
}

# Regex pattern for parsing numbers with optional suffixes
NUMBER_PATTERN = re.compile(
    r"""
    ^\s*                           # Optional leading whitespace
    [\$\+]?                        # Optional currency symbol or plus sign
    (-?)                           # Optional negative sign (captured)
    (\d{1,3}(?:,\d{3})*|\d+)       # Integer part (with optional commas)
    (?:\.(\d+))?                   # Optional decimal part
    \s*([KkMmBbTt])?               # Optional suffix
    """,
    re.VERBOSE,
)


def parse_number(text: str) -> float | None:
    """Parse a number from text, handling game formats like 1.5K, 2.3M.

    Supported formats:
    - Basic integers: 100, 42, 0
    - Decimals: 3.14, 0.5
    - Thousands suffix: 1K, 1.5K, 2.3k
    - Millions suffix: 1M, 2.3M, 10.5m
    - Billions suffix: 1B, 1.23B
    - Trillions suffix: 1T, 5.67T
    - Comma separated: 1,234, 1,234,567
    - With currency: $1.5K, $100
    - With plus/minus: +2.3M, -500
    - With trailing text: "1.5K coins", "Gold: 500"

    Args:
        text: Text containing a number.

    Returns:
        Parsed number value, or None if parsing fails.

    Example:
        >>> parse_number("1.5K")
        1500.0
        >>> parse_number("2.3M")
        2300000.0
        >>> parse_number("abc")
        None
    """
    if not text or not text.strip():
        return None

    # Try to find a number pattern in the text
    # First, try to extract just the number portion
    cleaned = text.strip()

    # Try direct match first
    match = NUMBER_PATTERN.match(cleaned)

    # If no match, try to find number anywhere in string
    if not match:
        # Look for number patterns in the text
        search_pattern = re.compile(
            r"[\$\+]?(-?)(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?\s*([KkMmBbTt])?"
        )
        match = search_pattern.search(cleaned)

    if not match:
        return None

    negative_sign, integer_part, decimal_part, suffix = match.groups()

    # Remove commas from integer part
    integer_part = integer_part.replace(",", "")

    # Build the base number
    number = float(f"{integer_part}.{decimal_part}") if decimal_part else float(integer_part)

    # Apply suffix multiplier
    if suffix:
        multiplier = SUFFIXES.get(suffix.lower())
        if multiplier:
            number *= multiplier

    # Apply negative sign
    if negative_sign:
        number = -number

    return number


class OCRSystem:
    """OCR system for extracting text from screenshots.

    This class provides:
    - Text extraction with bounding boxes
    - Number parsing with game format support
    - Region-based OCR for specific areas
    - Confidence filtering

    The OCR engine uses pytesseract when available, with graceful
    fallback for testing environments.

    Attributes:
        _language: OCR language code (default: "eng").

    Example:
        >>> ocr = OCRSystem()
        >>> regions = ocr.extract_text(screenshot)
        >>> numbers = ocr.extract_numbers("Gold: 1.5K | Gems: 500")
    """

    def __init__(self, language: str = "eng") -> None:
        """Initialize the OCR system.

        Args:
            language: OCR language code (default: "eng" for English).
        """
        self._language = language
        self._tesseract_available = self._check_tesseract()

        if self._tesseract_available:
            logger.info("OCRSystem initialized with pytesseract")
        else:
            logger.warning("pytesseract not available, OCR will be limited")

    def _check_tesseract(self) -> bool:
        """Check if pytesseract is available."""
        try:
            import pytesseract

            # Try to get tesseract version to verify it's working
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def _run_ocr(self, image: Image.Image) -> list[dict[str, Any]]:
        """Run OCR on an image and return raw results.

        This is the internal OCR method that interfaces with the OCR engine.
        Results are returned as a list of dictionaries with:
        - text: The detected text
        - left, top, width, height: Bounding box
        - conf: Confidence score (0-100)

        Args:
            image: PIL Image to process.

        Returns:
            List of OCR result dictionaries.
        """
        if not self._tesseract_available:
            # Return empty results when tesseract is not available
            # This allows unit tests to mock this method
            return []

        try:
            import pytesseract

            # Use image_to_data for detailed results with bounding boxes
            data = pytesseract.image_to_data(
                image,
                lang=self._language,
                output_type=pytesseract.Output.DICT,
            )

            results = []
            n_boxes = len(data["text"])

            for i in range(n_boxes):
                text = data["text"][i].strip()
                conf = float(data["conf"][i])

                # Skip empty text or very low confidence
                if not text or conf < 0:
                    continue

                results.append(
                    {
                        "text": text,
                        "left": int(data["left"][i]),
                        "top": int(data["top"][i]),
                        "width": int(data["width"][i]),
                        "height": int(data["height"][i]),
                        "conf": conf,
                    }
                )

            return results

        except Exception as e:
            logger.error(f"OCR failed: {e}")
            raise VisionError(f"OCR processing failed: {e}") from e

    def extract_text(
        self,
        screenshot: Screenshot,
        region: tuple[int, int, int, int] | None = None,
        min_confidence: float = 0.0,
    ) -> list[TextRegion]:
        """Extract text from a screenshot using OCR.

        Args:
            screenshot: The screenshot to process.
            region: Optional (x, y, width, height) to limit extraction area.
                If provided, only text within this region is extracted.
            min_confidence: Minimum confidence threshold (0.0 to 1.0).
                Results below this threshold are filtered out.

        Returns:
            List of detected text regions with bounding boxes.

        Raises:
            VisionError: If OCR processing fails.
        """
        image = screenshot.image

        # Crop to region if specified
        if region is not None:
            x, y, width, height = region

            # Clip to image bounds
            x = max(0, x)
            y = max(0, y)
            right = min(image.width, x + width)
            bottom = min(image.height, y + height)

            # Crop the image
            image = image.crop((x, y, right, bottom))

            # Adjust width/height for clipped region
            region_offset_x = x
            region_offset_y = y
        else:
            region_offset_x = 0
            region_offset_y = 0

        # Run OCR
        ocr_results = self._run_ocr(image)

        # Convert to TextRegion objects
        text_regions = []
        for result in ocr_results:
            text = str(result["text"])
            conf = float(result["conf"]) / 100.0  # Convert to 0-1 range

            # Filter by confidence
            if conf < min_confidence:
                continue

            # Filter empty or whitespace-only text
            if not text.strip():
                continue

            # Adjust coordinates for region offset
            text_region = TextRegion(
                text=text,
                x=int(result["left"]) + region_offset_x,
                y=int(result["top"]) + region_offset_y,
                width=int(result["width"]),
                height=int(result["height"]),
                confidence=conf,
            )
            text_regions.append(text_region)

        logger.debug(f"Extracted {len(text_regions)} text regions")
        return text_regions

    def parse_number(self, text: str) -> float | None:
        """Parse a number from text, handling game formats like 1.5K, 2.3M.

        This is an instance method wrapper around the module-level parse_number
        function for convenience.

        Args:
            text: Text containing a number.

        Returns:
            Parsed number value, or None if parsing fails.
        """
        return parse_number(text)

    def extract_numbers(self, text: str) -> list[float]:
        """Extract all numbers from text.

        Finds and parses all number-like patterns in the text,
        including game formats like 1.5K, 2.3M.

        Args:
            text: Text containing numbers.

        Returns:
            List of parsed number values.

        Example:
            >>> ocr.extract_numbers("Gold: 1.5K | Gems: 500")
            [1500.0, 500.0]
        """
        if not text:
            return []

        # Pattern to find all number-like strings
        pattern = re.compile(
            r"[\$\+\-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*[KkMmBbTt]?"
            r"|[\$\+\-]?\d+(?:\.\d+)?\s*[KkMmBbTt]?"
        )

        numbers = []
        for match in pattern.finditer(text):
            num = parse_number(match.group())
            if num is not None:
                numbers.append(num)

        return numbers
