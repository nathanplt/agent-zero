"""Vision system interface for screen capture and game state extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


class Screenshot:
    """A captured screenshot with metadata."""

    __slots__ = ("image", "raw_bytes", "timestamp", "width", "height")

    def __init__(
        self,
        image: Image.Image,
        raw_bytes: bytes,
        timestamp: datetime,
        width: int,
        height: int,
    ) -> None:
        """Initialize a screenshot.

        Args:
            image: PIL Image object.
            raw_bytes: Raw image bytes (PNG or JPEG encoded).
            timestamp: When the screenshot was captured.
            width: Image width in pixels.
            height: Image height in pixels.
        """
        self.image = image
        self.raw_bytes = raw_bytes
        self.timestamp = timestamp
        self.width = width
        self.height = height


class TextRegion:
    """A region of detected text in a screenshot."""

    __slots__ = ("text", "x", "y", "width", "height", "confidence")

    def __init__(
        self,
        text: str,
        x: int,
        y: int,
        width: int,
        height: int,
        confidence: float,
    ) -> None:
        """Initialize a text region.

        Args:
            text: The detected text content.
            x: Left coordinate of bounding box.
            y: Top coordinate of bounding box.
            width: Width of bounding box.
            height: Height of bounding box.
            confidence: OCR confidence score (0.0 to 1.0).
        """
        self.text = text
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence


class UIElement:
    """A detected UI element in a screenshot."""

    __slots__ = ("element_type", "x", "y", "width", "height", "confidence", "label")

    def __init__(
        self,
        element_type: str,
        x: int,
        y: int,
        width: int,
        height: int,
        confidence: float,
        label: str | None = None,
    ) -> None:
        """Initialize a UI element.

        Args:
            element_type: Type of element (button, menu, resource, etc.).
            x: Left coordinate of bounding box.
            y: Top coordinate of bounding box.
            width: Width of bounding box.
            height: Height of bounding box.
            confidence: Detection confidence score (0.0 to 1.0).
            label: Optional text label for the element.
        """
        self.element_type = element_type
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence
        self.label = label

    @property
    def center(self) -> tuple[int, int]:
        """Get the center point of this element."""
        return (self.x + self.width // 2, self.y + self.height // 2)


class VisionSystem(ABC):
    """Abstract interface for the vision system.

    The vision system is responsible for:
    - Capturing screenshots from the virtual display
    - Extracting text using OCR
    - Detecting UI elements
    - Using LLM vision for complex understanding
    """

    @abstractmethod
    def capture(self) -> Screenshot:
        """Capture a screenshot from the virtual display.

        Returns:
            Screenshot object with image data and metadata.

        Raises:
            VisionError: If capture fails.
        """
        ...

    @abstractmethod
    def get_buffer(self, count: int | None = None) -> list[Screenshot]:
        """Get recent screenshots from the buffer.

        Args:
            count: Number of screenshots to return. None for all buffered.

        Returns:
            List of screenshots, most recent first.
        """
        ...

    @abstractmethod
    def extract_text(
        self,
        screenshot: Screenshot,
        region: tuple[int, int, int, int] | None = None,
    ) -> list[TextRegion]:
        """Extract text from a screenshot using OCR.

        Args:
            screenshot: The screenshot to process.
            region: Optional (x, y, width, height) to limit extraction area.

        Returns:
            List of detected text regions.
        """
        ...

    @abstractmethod
    def detect_ui_elements(self, screenshot: Screenshot) -> list[UIElement]:
        """Detect UI elements in a screenshot.

        Args:
            screenshot: The screenshot to analyze.

        Returns:
            List of detected UI elements.
        """
        ...

    @abstractmethod
    def parse_number(self, text: str) -> float | None:
        """Parse a number from text, handling game formats like 1.5K, 2.3M.

        Args:
            text: Text containing a number.

        Returns:
            Parsed number value, or None if parsing fails.
        """
        ...


class VisionError(Exception):
    """Error raised when vision operations fail."""

    pass
