"""UI element detection module for the vision system.

This module provides UI element detection from screenshots:
- Detect buttons, menus, resources, and other UI elements
- Return bounding boxes with confidence scores
- Classify element types based on visual features and labels
- Handle overlapping elements gracefully

Example:
    >>> from src.vision.ui_detection import UIDetector
    >>> detector = UIDetector()
    >>> elements = detector.detect(screenshot)
    >>> for elem in elements:
    ...     print(f"{elem.element_type}: {elem.label} at ({elem.x}, {elem.y})")
    ...
    >>> buttons = detector.detect_buttons(screenshot)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from src.interfaces.vision import Screenshot, UIElement, VisionError

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

# Supported UI element types
SUPPORTED_ELEMENT_TYPES = frozenset([
    "button",
    "menu",
    "resource",
    "text",
    "icon",
    "progress_bar",
    "input",
    "panel",
    "tooltip",
    "dialog",
])

# Keywords for element classification
BUTTON_KEYWORDS = frozenset([
    "click", "buy", "sell", "upgrade", "start", "ok", "cancel",
    "confirm", "yes", "no", "play", "next", "back", "close",
    "submit", "apply", "save", "delete", "reset", "skip",
])

RESOURCE_KEYWORDS = frozenset([
    "gold", "gems", "coins", "diamonds", "energy", "health",
    "mana", "xp", "level", "score", "cash", "points", "money",
    "credits", "tokens", "power", "stamina",
])


def calculate_iou(
    box1: tuple[int, int, int, int],
    box2: tuple[int, int, int, int],
) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: First box as (x, y, width, height).
        box2: Second box as (x, y, width, height).

    Returns:
        IoU value between 0.0 and 1.0.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to corner coordinates
    x1_min, y1_min = x1, y1
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_min, y2_min = x2, y2
    x2_max, y2_max = x2 + w2, y2 + h2

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


class UIDetector:
    """UI element detector for game screenshots.

    This class provides:
    - Detection of UI elements (buttons, menus, resources, etc.)
    - Element type classification
    - Confidence-based filtering
    - Duplicate detection filtering

    The detector uses heuristic-based detection when no ML model is available,
    making it suitable for testing and basic scenarios.

    Attributes:
        _min_confidence: Minimum confidence threshold for detections.

    Example:
        >>> detector = UIDetector(min_confidence=0.7)
        >>> elements = detector.detect(screenshot)
        >>> buttons = detector.detect_buttons(screenshot)
    """

    def __init__(self, min_confidence: float = 0.5) -> None:
        """Initialize the UI detector.

        Args:
            min_confidence: Minimum confidence threshold (0.0 to 1.0).
                Detections below this threshold are filtered out.
        """
        self._min_confidence = min_confidence
        logger.debug(f"UIDetector initialized with min_confidence={min_confidence}")

    def _run_detection(self, image: Image.Image) -> list[dict[str, Any]]:
        """Run UI element detection on an image.

        This is the internal detection method. Override or mock for testing.
        Returns raw detection results as dictionaries.

        Args:
            image: PIL Image to process.

        Returns:
            List of detection dictionaries with keys:
            - type: Element type string
            - x, y, width, height: Bounding box
            - confidence: Detection confidence (0-1)
            - label: Optional text label
        """
        # Default implementation returns empty results
        # This allows unit tests to mock this method
        # Real implementations would use CV or ML models
        _ = image  # Mark as used (for mocking/override)
        return []

    def detect(
        self,
        screenshot: Screenshot,
        region: tuple[int, int, int, int] | None = None,
        filter_duplicates: bool = False,
    ) -> list[UIElement]:
        """Detect UI elements in a screenshot.

        Args:
            screenshot: The screenshot to analyze.
            region: Optional (x, y, width, height) to limit detection area.
            filter_duplicates: Whether to filter near-duplicate detections.

        Returns:
            List of detected UI elements.

        Raises:
            VisionError: If detection fails.
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
            region_offset_x = x
            region_offset_y = y
        else:
            region_offset_x = 0
            region_offset_y = 0

        try:
            # Run detection
            detections = self._run_detection(image)
        except Exception as e:
            raise VisionError(f"UI detection failed: {e}") from e

        # Convert to UIElement objects
        elements = []
        for det in detections:
            confidence = float(det["confidence"])

            # Filter by confidence
            if confidence < self._min_confidence:
                continue

            element = UIElement(
                element_type=str(det["type"]),
                x=int(det["x"]) + region_offset_x,
                y=int(det["y"]) + region_offset_y,
                width=int(det["width"]),
                height=int(det["height"]),
                confidence=confidence,
                label=det.get("label"),
            )
            elements.append(element)

        # Filter duplicates if requested
        if filter_duplicates:
            elements = self._filter_duplicates(elements)

        logger.debug(f"Detected {len(elements)} UI elements")
        return elements

    def _filter_duplicates(
        self,
        elements: list[UIElement],
        iou_threshold: float = 0.8,
    ) -> list[UIElement]:
        """Filter near-duplicate detections.

        Keeps the higher-confidence detection when two elements have
        IoU above the threshold.

        Args:
            elements: List of detected elements.
            iou_threshold: IoU threshold for considering duplicates.

        Returns:
            Filtered list of elements.
        """
        if not elements:
            return elements

        # Sort by confidence (highest first)
        sorted_elements = sorted(elements, key=lambda e: e.confidence, reverse=True)
        kept: list[UIElement] = []

        for elem in sorted_elements:
            box = (elem.x, elem.y, elem.width, elem.height)
            is_duplicate = False

            for kept_elem in kept:
                kept_box = (kept_elem.x, kept_elem.y, kept_elem.width, kept_elem.height)
                if calculate_iou(box, kept_box) > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(elem)

        return kept

    def detect_buttons(
        self,
        screenshot: Screenshot,
        min_confidence: float | None = None,
    ) -> list[UIElement]:
        """Detect only button elements in a screenshot.

        Args:
            screenshot: The screenshot to analyze.
            min_confidence: Optional confidence threshold override.

        Returns:
            List of detected button elements.
        """
        elements = self.detect(screenshot)

        threshold = min_confidence if min_confidence is not None else self._min_confidence

        return [
            elem for elem in elements
            if elem.element_type == "button" and elem.confidence >= threshold
        ]

    def classify(
        self,
        label: str | None = None,
        aspect_ratio: float = 1.0,
        has_gradient: bool = False,
        size_small: bool = False,
    ) -> str:
        """Classify an element type based on features.

        Uses heuristics based on:
        - Label text (keywords)
        - Aspect ratio
        - Visual features (gradient, size)

        Args:
            label: Text label of the element (if any).
            aspect_ratio: Width / height ratio.
            has_gradient: Whether element has gradient fill.
            size_small: Whether element is small (icon-sized).

        Returns:
            Classified element type string.
        """
        # Check label keywords first
        if label:
            label_lower = label.lower()

            # Check for button keywords
            for keyword in BUTTON_KEYWORDS:
                if keyword in label_lower:
                    return "button"

            # Check for resource keywords
            for keyword in RESOURCE_KEYWORDS:
                if keyword in label_lower:
                    return "resource"

            # Check for number pattern (likely resource)
            if re.search(r"\d+[KkMmBbTt]?", label):
                return "resource"

        # Use aspect ratio and visual features
        if aspect_ratio > 5.0 and has_gradient:
            return "progress_bar"

        if size_small and 0.8 <= aspect_ratio <= 1.2:
            return "icon"

        if aspect_ratio > 3.0:
            return "text"

        # Default to button (most common interactive element)
        return "button"
