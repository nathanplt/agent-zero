"""Observation pipeline for capturing and processing game state.

This module orchestrates all vision components to produce unified observations:
- Screenshot capture
- OCR text extraction (parallel)
- UI element detection (parallel)
- LLM Vision for complex understanding (conditional)

The pipeline is designed for efficiency:
- OCR and UI detection run in parallel
- LLM Vision is only called when needed (first obs, transitions, low confidence)
- Performance target: <500ms without LLM, <2s with LLM

Example:
    >>> from src.core.observation import ObservationPipeline
    >>> from src.vision import ScreenshotCapture, OCRSystem, UIDetector, LLMVision
    >>> from src.environment import LocalEnvironmentManager
    >>>
    >>> env = LocalEnvironmentManager()
    >>> env.start()
    >>> env.navigate("https://game.example.com")
    >>>
    >>> pipeline = ObservationPipeline(
    ...     capture=ScreenshotCapture(env),
    ...     ocr=OCRSystem(),
    ...     ui_detector=UIDetector(),
    ...     llm_vision=LLMVision(),
    ... )
    >>>
    >>> obs = pipeline.observe()
    >>> print(f"Screen: {obs.game_state.current_screen}")
    >>> print(f"Elements: {len(obs.ui_elements)}")
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.interfaces.vision import Screenshot, TextRegion, UIElement, VisionError
from src.models.game_state import GameState, Resource, ScreenType
from src.models.game_state import UIElement as GameStateUIElement

if TYPE_CHECKING:
    from src.vision.capture import ScreenshotCapture
    from src.vision.llm_vision import LLMVision
    from src.vision.ocr import OCRSystem
    from src.vision.ui_detection import UIDetector

logger = logging.getLogger(__name__)


# Resource patterns for parsing from OCR text
RESOURCE_PATTERNS = [
    # "Gold: 1.5K" or "Gold 1.5K"
    re.compile(
        r"(?P<name>gold|gems|coins|diamonds|energy|health|mana|xp|level|score|"
        r"cash|points|money|credits|tokens|power|stamina)"
        r"[\s:]*(?P<value>[\d,.]+[KkMmBbTt]?)",
        re.IGNORECASE,
    ),
    # "1.5K Gold" (value before name)
    re.compile(
        r"(?P<value>[\d,.]+[KkMmBbTt]?)\s*"
        r"(?P<name>gold|gems|coins|diamonds|energy|health|mana|xp|level|score|"
        r"cash|points|money|credits|tokens|power|stamina)",
        re.IGNORECASE,
    ),
]


def parse_resource_value(text: str) -> float | None:
    """Parse a resource value from text.

    Handles formats like:
    - 1500, 1,500
    - 1.5K, 2.3M, 1B
    - 1.5k (lowercase)

    Args:
        text: Text containing a number.

    Returns:
        Parsed value or None if parsing fails.
    """
    text = text.strip().replace(",", "")

    # Check for suffix
    multiplier = 1.0
    if text and text[-1].upper() in "KMBT":
        suffix = text[-1].upper()
        multiplier = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}[
            suffix
        ]
        text = text[:-1]

    try:
        return float(text) * multiplier
    except ValueError:
        return None


@dataclass
class Observation:
    """A complete observation of the game state.

    Contains all information captured at a single point in time:
    - The raw screenshot
    - Extracted game state (resources, screen type, etc.)
    - Detected UI elements
    - OCR text regions
    - Timing and confidence information

    Attributes:
        screenshot: The captured screenshot.
        game_state: Extracted game state (resources, screen type, etc.).
        ui_elements: Detected UI elements from UIDetector.
        text_regions: OCR text regions.
        timestamp: When the observation was made.
        llm_used: Whether LLM Vision was used for this observation.
        confidence: Overall confidence in the observation (0.0-1.0).
        duration_ms: How long the observation took in milliseconds.
    """

    screenshot: Screenshot
    game_state: GameState
    ui_elements: list[UIElement] = field(default_factory=list)
    text_regions: list[TextRegion] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    llm_used: bool = False
    confidence: float = 1.0
    duration_ms: float = 0.0


@dataclass
class PipelineConfig:
    """Configuration for the observation pipeline.

    Attributes:
        use_llm: Whether to use LLM Vision at all.
        llm_interval: Call LLM every N observations for validation.
        llm_on_transition: Call LLM when screen transition detected.
        llm_on_low_confidence: Call LLM when confidence below threshold.
        confidence_threshold: Threshold for triggering LLM on low confidence.
        parallel_timeout: Timeout for parallel OCR/UI detection in seconds.
    """

    use_llm: bool = True
    llm_interval: int = 10
    llm_on_transition: bool = True
    llm_on_low_confidence: bool = True
    confidence_threshold: float = 0.5
    parallel_timeout: float = 5.0


class ObservationPipeline:
    """Orchestrates vision components to produce observations.

    This class combines:
    - ScreenshotCapture for frame capture
    - OCRSystem for text extraction
    - UIDetector for UI element detection
    - LLMVision for complex understanding (conditional)

    OCR and UI detection run in parallel for efficiency.
    LLM Vision is only called when needed to save costs.

    Attributes:
        _capture: Screenshot capture component.
        _ocr: OCR system component.
        _ui_detector: UI element detector component.
        _llm_vision: LLM Vision component (optional).
        _config: Pipeline configuration.
        _observation_count: Number of observations made.
        _last_screen: Last observed screen type for transition detection.

    Example:
        >>> pipeline = ObservationPipeline(capture, ocr, ui_detector, llm_vision)
        >>> obs = pipeline.observe()
        >>> print(f"Resources: {list(obs.game_state.resources.keys())}")
    """

    def __init__(
        self,
        capture: ScreenshotCapture,
        ocr: OCRSystem,
        ui_detector: UIDetector,
        llm_vision: LLMVision | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        """Initialize the observation pipeline.

        Args:
            capture: Screenshot capture component.
            ocr: OCR system component.
            ui_detector: UI element detector component.
            llm_vision: LLM Vision component (optional).
            config: Pipeline configuration. Uses defaults if None.
        """
        self._capture = capture
        self._ocr = ocr
        self._ui_detector = ui_detector
        self._llm_vision = llm_vision
        self._config = config or PipelineConfig()

        self._observation_count = 0
        self._last_screen: ScreenType | None = None
        self._executor = ThreadPoolExecutor(max_workers=2)

        logger.debug(
            f"ObservationPipeline initialized (LLM: {llm_vision is not None})"
        )

    def observe(self) -> Observation:
        """Capture and process a complete observation.

        This method:
        1. Captures a screenshot
        2. Runs OCR and UI detection in parallel
        3. Parses resources from OCR text
        4. Conditionally calls LLM Vision for complex understanding
        5. Builds and returns an Observation object

        Returns:
            Complete Observation with all extracted data.

        Raises:
            VisionError: If capture or critical processing fails.
        """
        start_time = time.time()
        self._observation_count += 1

        # Step 1: Capture screenshot
        try:
            screenshot = self._capture.capture()
        except Exception as e:
            raise VisionError(f"Screenshot capture failed: {e}") from e

        # Step 2: Run OCR and UI detection in parallel
        text_regions, ui_elements, ocr_error, ui_error = self._run_parallel_detection(
            screenshot
        )

        # Log any errors but continue with partial data
        if ocr_error:
            logger.warning(f"OCR failed: {ocr_error}")
        if ui_error:
            logger.warning(f"UI detection failed: {ui_error}")

        # Step 3: Parse resources from OCR results
        resources = self._parse_resources(text_regions)

        # Step 4: Calculate confidence based on extraction quality
        confidence = self._calculate_confidence(text_regions, ui_elements)

        # Step 5: Determine if LLM Vision should be used
        use_llm = self._should_use_llm(confidence)

        # Step 6: Build game state
        if use_llm and self._llm_vision is not None:
            try:
                game_state = self._llm_vision.analyze(screenshot)
                llm_used = True
                logger.debug("LLM Vision used for observation")
            except Exception as e:
                logger.warning(f"LLM Vision failed, using local extraction: {e}")
                game_state = self._build_game_state(resources, ui_elements, text_regions)
                llm_used = False
        else:
            game_state = self._build_game_state(resources, ui_elements, text_regions)
            llm_used = False

        # Track screen transitions
        self._last_screen = game_state.current_screen

        duration_ms = (time.time() - start_time) * 1000

        observation = Observation(
            screenshot=screenshot,
            game_state=game_state,
            ui_elements=ui_elements,
            text_regions=text_regions,
            timestamp=datetime.now(),
            llm_used=llm_used,
            confidence=confidence,
            duration_ms=duration_ms,
        )

        logger.debug(
            f"Observation #{self._observation_count}: "
            f"{len(ui_elements)} elements, {len(text_regions)} text regions, "
            f"LLM={llm_used}, {duration_ms:.1f}ms"
        )

        return observation

    def _run_parallel_detection(
        self, screenshot: Screenshot
    ) -> tuple[list[TextRegion], list[UIElement], Exception | None, Exception | None]:
        """Run OCR and UI detection in parallel.

        Args:
            screenshot: Screenshot to process.

        Returns:
            Tuple of (text_regions, ui_elements, ocr_error, ui_error).
            Errors are None if successful.
        """
        text_regions: list[TextRegion] = []
        ui_elements: list[UIElement] = []
        ocr_error: Exception | None = None
        ui_error: Exception | None = None

        # Submit tasks to thread pool
        ocr_future = self._executor.submit(self._ocr.extract_text, screenshot)
        ui_future = self._executor.submit(self._ui_detector.detect, screenshot)

        futures: dict[Future[Any], str] = {
            ocr_future: "ocr",
            ui_future: "ui",
        }

        # Wait for completion with timeout
        try:
            for future in as_completed(futures, timeout=self._config.parallel_timeout):
                task_name = futures[future]
                try:
                    result = future.result()
                    if task_name == "ocr":
                        text_regions = result
                    else:
                        ui_elements = result
                except Exception as e:
                    if task_name == "ocr":
                        ocr_error = e
                    else:
                        ui_error = e
        except TimeoutError:
            logger.warning("Parallel detection timed out")
        finally:
            # Cancel unfinished futures to avoid dangling background work.
            for future in futures:
                done = False
                with suppress(Exception):
                    done = bool(future.done())
                if not done:
                    future.cancel()

        return text_regions, ui_elements, ocr_error, ui_error

    def _parse_resources(self, text_regions: list[TextRegion]) -> dict[str, Resource]:
        """Parse resources from OCR text regions.

        Args:
            text_regions: OCR text regions to parse.

        Returns:
            Dictionary of resource name -> Resource.
        """
        resources: dict[str, Resource] = {}

        for region in text_regions:
            text = region.text.strip()
            if not text:
                continue

            # Try each resource pattern
            for pattern in RESOURCE_PATTERNS:
                match = pattern.search(text)
                if match:
                    name = match.group("name").lower()
                    value_str = match.group("value")
                    value = parse_resource_value(value_str)

                    if value is not None and name not in resources:
                        resources[name] = Resource(name=name, amount=value)
                        logger.debug(f"Parsed resource: {name}={value}")
                    break

        return resources

    def _calculate_confidence(
        self, text_regions: list[TextRegion], ui_elements: list[UIElement]
    ) -> float:
        """Calculate overall confidence in the observation.

        Args:
            text_regions: OCR text regions.
            ui_elements: Detected UI elements.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not text_regions and not ui_elements:
            return 0.0

        # Average OCR confidence
        ocr_confidence = 0.0
        if text_regions:
            ocr_confidence = sum(r.confidence for r in text_regions) / len(text_regions)

        # Average UI detection confidence
        ui_confidence = 0.0
        if ui_elements:
            ui_confidence = sum(e.confidence for e in ui_elements) / len(ui_elements)

        # Weight OCR and UI equally if both present
        if text_regions and ui_elements:
            return (ocr_confidence + ui_confidence) / 2
        elif text_regions:
            return ocr_confidence
        else:
            return ui_confidence

    def _should_use_llm(self, confidence: float) -> bool:
        """Determine if LLM Vision should be used.

        Args:
            confidence: Current observation confidence.

        Returns:
            True if LLM should be used.
        """
        config = self._config

        if not config.use_llm or self._llm_vision is None:
            return False

        # First observation - always use LLM for baseline
        if self._observation_count == 1:
            logger.debug("Using LLM: first observation")
            return True

        # Periodic validation
        if config.llm_interval > 0 and self._observation_count % config.llm_interval == 0:
            logger.debug(f"Using LLM: periodic ({self._observation_count})")
            return True

        # Low confidence
        if config.llm_on_low_confidence and confidence < config.confidence_threshold:
            logger.debug(f"Using LLM: low confidence ({confidence:.2f})")
            return True

        return False

    def _build_game_state(
        self,
        resources: dict[str, Resource],
        ui_elements: list[UIElement],
        text_regions: list[TextRegion],
    ) -> GameState:
        """Build game state from local extraction results.

        Args:
            resources: Parsed resources.
            ui_elements: Detected UI elements.
            text_regions: OCR text regions.

        Returns:
            GameState object.
        """
        # Convert interface UIElements to model UIElements
        model_ui_elements = [
            GameStateUIElement(
                element_type=e.element_type,
                x=e.x,
                y=e.y,
                width=e.width,
                height=e.height,
                label=e.label,
                confidence=e.confidence,
                clickable=e.clickable,
                metadata=e.metadata,
            )
            for e in ui_elements
        ]

        # Extract raw text
        raw_text = [r.text for r in text_regions if r.text.strip()]

        # Infer screen type from UI elements and text
        screen_type = self._infer_screen_type(ui_elements, raw_text)

        return GameState(
            resources=resources,
            upgrades=[],  # Would need more sophisticated parsing
            current_screen=screen_type,
            ui_elements=model_ui_elements,
            timestamp=datetime.now(),
            raw_text=raw_text,
        )

    def _infer_screen_type(
        self, ui_elements: list[UIElement], raw_text: list[str]
    ) -> ScreenType:
        """Infer the screen type from UI elements and text.

        Args:
            ui_elements: Detected UI elements.
            raw_text: OCR text lines.

        Returns:
            Inferred ScreenType.
        """
        text_lower = " ".join(raw_text).lower()

        # Check for common screen indicators
        if "loading" in text_lower:
            return ScreenType.LOADING
        if "settings" in text_lower or "options" in text_lower:
            return ScreenType.SETTINGS
        if "shop" in text_lower or "store" in text_lower or "buy" in text_lower:
            return ScreenType.SHOP
        if "inventory" in text_lower or "items" in text_lower:
            return ScreenType.INVENTORY
        if "menu" in text_lower:
            return ScreenType.MENU
        if "prestige" in text_lower or "rebirth" in text_lower or "ascend" in text_lower:
            return ScreenType.PRESTIGE

        # Check for dialog indicators
        dialog_labels = {"ok", "cancel", "yes", "no", "confirm", "close"}
        for element in ui_elements:
            if element.label and element.label.lower() in dialog_labels:
                return ScreenType.DIALOG

        # If we have buttons and resources, likely main game screen
        has_buttons = any(e.element_type == "button" for e in ui_elements)
        has_resources = any(
            word in text_lower
            for word in ["gold", "coins", "gems", "energy", "level"]
        )

        if has_buttons and has_resources:
            return ScreenType.MAIN

        return ScreenType.UNKNOWN

    def close(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=False)
        logger.debug("ObservationPipeline closed")

    def __enter__(self) -> ObservationPipeline:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()
