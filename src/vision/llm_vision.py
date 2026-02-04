"""LLM Vision integration for game state extraction.

This module provides vision-language model (VLM) integration for extracting
structured game state from screenshots. It supports multiple providers
(Anthropic, OpenAI) and includes features like:
- Caching for identical frames
- Set-of-Mark (SoM) annotation for improved element grounding
- Retry logic with exponential backoff
- Structured output parsing

Example:
    >>> from src.vision.llm_vision import LLMVision
    >>> from src.vision.capture import ScreenshotCapture
    >>>
    >>> llm_vision = LLMVision(provider="anthropic")
    >>> game_state = llm_vision.analyze(screenshot)
    >>> print(f"Screen: {game_state.current_screen}")
    >>> print(f"Gold: {game_state.resources.get('gold')}")
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from collections import OrderedDict
from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw, ImageFont

from src.interfaces.vision import Screenshot
from src.models.game_state import GameState, Resource, ScreenType, UIElement, Upgrade

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LLMVisionError(Exception):
    """Error raised when LLM vision operations fail."""

    pass


# Default prompt template for game state extraction
DEFAULT_PROMPT_TEMPLATE = """Analyze this game screenshot and extract the game state information.

You are an expert at understanding game UIs, particularly incremental/idle games.
Extract the following information from the screenshot:

1. **Current Screen Type**: Identify what type of screen is shown (main, menu, shop, inventory, settings, loading, dialog, prestige, or unknown)

2. **Resources**: List all visible resources with their current amounts. Parse numbers like "1.5K" as 1500, "2.3M" as 2300000, etc.

3. **Upgrades**: List any visible upgrades/purchases with their names, levels, costs, and availability.

4. **UI Elements**: Identify clickable buttons, menus, and interactive elements with their approximate locations.

5. **Raw Text**: Extract any other visible text that might be relevant.

{format_instructions}

Respond with a JSON object in this exact format:
{{
    "current_screen": "main|menu|shop|inventory|settings|loading|dialog|prestige|unknown",
    "resources": {{
        "resource_name": {{"name": "resource_name", "amount": 1234.5, "max_amount": null, "rate": null}}
    }},
    "upgrades": [
        {{"id": "upgrade_id", "name": "Upgrade Name", "level": 1, "max_level": null, "cost": {{"gold": 100}}, "available": true}}
    ],
    "ui_elements": [
        {{"element_type": "button", "x": 100, "y": 200, "width": 80, "height": 40, "label": "Click Me", "clickable": true}}
    ],
    "raw_text": ["Text line 1", "Text line 2"]
}}

Be precise with coordinates and numbers. If you're unsure about something, use reasonable estimates.
"""

SOM_PROMPT_ADDITION = """
The screenshot has been annotated with numbered markers (Set-of-Mark).
Each marker is shown as a colored circle with a number.
When identifying UI elements, include the mark_id that corresponds to each element.
This helps ensure accurate element identification and click targeting.
"""

# Valid providers
VALID_PROVIDERS = {"anthropic", "openai"}


def add_set_of_mark(
    image: Image.Image,
    regions: list[dict[str, int]],
    marker_radius: int = 15,
    font_size: int = 12,
) -> Image.Image:
    """Add Set-of-Mark annotations to an image.

    Set-of-Mark (SoM) is a technique where visual markers are overlaid on
    an image to help vision models ground their responses to specific
    locations. Each marker is a numbered circle.

    Args:
        image: PIL Image to annotate.
        regions: List of regions to mark, each with x, y, width, height.
        marker_radius: Radius of marker circles.
        font_size: Font size for marker numbers.

    Returns:
        Annotated PIL Image with numbered markers.

    Example:
        >>> regions = [{"x": 100, "y": 100, "width": 50, "height": 30}]
        >>> marked = add_set_of_mark(screenshot.image, regions)
    """
    # Create a copy to avoid modifying original
    marked = image.copy()
    draw = ImageDraw.Draw(marked)

    # Colors for markers (cycle through for visibility)
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
    ]

    # Try to load a font, fall back to default
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for idx, region in enumerate(regions):
        mark_id = idx + 1
        color = colors[idx % len(colors)]

        # Calculate center of region
        center_x = region["x"] + region["width"] // 2
        center_y = region["y"] + region["height"] // 2

        # Draw filled circle
        bbox = [
            center_x - marker_radius,
            center_y - marker_radius,
            center_x + marker_radius,
            center_y + marker_radius,
        ]
        draw.ellipse(bbox, fill=color, outline=(255, 255, 255), width=2)

        # Draw number in center
        text = str(mark_id)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = center_x - text_width // 2
        text_y = center_y - text_height // 2
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

    return marked


class LLMVision:
    """LLM-based vision system for game state extraction.

    This class provides the ability to analyze game screenshots using
    vision-language models (VLMs) like Claude or GPT-4V. It extracts
    structured game state information including resources, upgrades,
    UI elements, and screen type.

    Features:
    - Multiple provider support (Anthropic, OpenAI)
    - Response caching to avoid duplicate API calls
    - Set-of-Mark annotation for improved grounding
    - Retry logic with exponential backoff
    - Structured output parsing to GameState objects

    Attributes:
        cache_stats: Dictionary with cache hit/miss statistics.

    Example:
        >>> llm_vision = LLMVision(provider="anthropic")
        >>> game_state = llm_vision.analyze(screenshot)
        >>> print(game_state.current_screen)
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str | None = None,
        cache_size: int = 50,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        prompt_template: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the LLM Vision system.

        Args:
            provider: LLM provider ("anthropic" or "openai").
            model: Model name. Defaults based on provider.
            cache_size: Maximum number of cached responses.
            max_retries: Maximum retry attempts on API failure.
            retry_delay: Initial delay between retries (exponential backoff).
            prompt_template: Custom prompt template for extraction.
            api_key: API key for the provider. Falls back to environment variable.

        Raises:
            ValueError: If provider is not supported.
        """
        if provider not in VALID_PROVIDERS:
            raise ValueError(f"Invalid provider: {provider}. Must be one of {VALID_PROVIDERS}")

        self._provider = provider
        self._model = model or self._default_model(provider)
        self._cache_size = cache_size
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self._api_key = api_key

        # LRU cache for responses (OrderedDict for LRU behavior)
        self._cache: OrderedDict[str, GameState] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

        logger.debug(
            f"LLMVision initialized: provider={provider}, model={self._model}, "
            f"cache_size={cache_size}"
        )

    @staticmethod
    def _default_model(provider: str) -> str:
        """Get default model for a provider."""
        defaults = {
            "anthropic": "claude-3-sonnet-20240229",
            "openai": "gpt-4-vision-preview",
        }
        return defaults.get(provider, "claude-3-sonnet-20240229")

    @property
    def cache_stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with 'size', 'hits', and 'misses' counts.
        """
        return {
            "size": len(self._cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
        }

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("LLMVision cache cleared")

    def _format_prompt(self, use_som: bool = False) -> str:
        """Format the prompt template with instructions.

        Args:
            use_som: Whether to include Set-of-Mark instructions.

        Returns:
            Formatted prompt string.
        """
        format_instructions = (
            "Provide your response as valid JSON only, with no additional text or explanation."
        )

        prompt = self._prompt_template.format(format_instructions=format_instructions)

        if use_som:
            prompt += "\n\n" + SOM_PROMPT_ADDITION

        return prompt

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string.

        Args:
            image: PIL Image to convert.

        Returns:
            Base64-encoded PNG string.
        """
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _compute_cache_key(self, screenshot: Screenshot) -> str:
        """Compute a cache key for a screenshot.

        Uses MD5 hash of image bytes for fast comparison.

        Args:
            screenshot: Screenshot to compute key for.

        Returns:
            Hex digest of image hash.
        """
        return hashlib.md5(screenshot.raw_bytes).hexdigest()

    def _call_vision_api(
        self,
        image_base64: str,
        prompt: str,
    ) -> dict[str, Any]:
        """Call the vision API with image and prompt.

        This method handles the actual API call to the vision LLM.
        It's separated to allow easy mocking in tests.

        Args:
            image_base64: Base64-encoded image.
            prompt: Text prompt for the model.

        Returns:
            Parsed JSON response from the model.

        Raises:
            LLMVisionError: If API call fails.
        """
        try:
            if self._provider == "anthropic":
                return self._call_anthropic(image_base64, prompt)
            elif self._provider == "openai":
                return self._call_openai(image_base64, prompt)
            else:
                raise LLMVisionError(f"Unknown provider: {self._provider}")
        except Exception as e:
            if isinstance(e, LLMVisionError):
                raise
            raise LLMVisionError(f"API call failed: {e}") from e

    def _call_anthropic(self, image_base64: str, prompt: str) -> dict[str, Any]:
        """Call Anthropic's Claude API.

        Args:
            image_base64: Base64-encoded image.
            prompt: Text prompt for the model.

        Returns:
            Parsed JSON response.

        Raises:
            LLMVisionError: If API call fails.
        """
        try:
            import anthropic
        except ImportError as e:
            raise LLMVisionError(
                "anthropic package not installed. Install with: pip install anthropic"
            ) from e

        try:
            client = anthropic.Anthropic(api_key=self._api_key)

            message = client.messages.create(
                model=self._model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )

            # Extract text content from response
            content_block = message.content[0]
            if hasattr(content_block, "text"):
                response_text = content_block.text
            else:
                raise LLMVisionError("Unexpected response format from Anthropic API")

            # Parse JSON from response
            return self._parse_json_response(response_text)

        except anthropic.APIError as e:
            raise LLMVisionError(f"Anthropic API error: {e}") from e

    def _call_openai(self, image_base64: str, prompt: str) -> dict[str, Any]:
        """Call OpenAI's GPT-4V API.

        Args:
            image_base64: Base64-encoded image.
            prompt: Text prompt for the model.

        Returns:
            Parsed JSON response.

        Raises:
            LLMVisionError: If API call fails.
        """
        try:
            import openai
        except ImportError as e:
            raise LLMVisionError(
                "openai package not installed. Install with: pip install openai"
            ) from e

        try:
            client = openai.OpenAI(api_key=self._api_key)

            response = client.chat.completions.create(
                model=self._model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                },
                            },
                        ],
                    }
                ],
            )

            # Extract text content
            response_text = response.choices[0].message.content
            if response_text is None:
                raise LLMVisionError("OpenAI returned empty response")

            # Parse JSON from response
            return self._parse_json_response(response_text)

        except openai.APIError as e:
            raise LLMVisionError(f"OpenAI API error: {e}") from e

    def _parse_json_response(self, response_text: str) -> dict[str, Any]:
        """Parse JSON from LLM response text.

        Handles cases where JSON is wrapped in markdown code blocks.

        Args:
            response_text: Raw text response from LLM.

        Returns:
            Parsed dictionary.

        Raises:
            LLMVisionError: If JSON parsing fails.
        """
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            result = json.loads(text)
            if not isinstance(result, dict):
                logger.warning(f"JSON response is not a dict: {type(result)}")
                return {}
            return dict(result)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {text[:500]}")
            # Return empty dict to allow graceful degradation
            return {}

    def _parse_game_state(self, data: dict[str, Any]) -> GameState:
        """Parse raw response data into a GameState object.

        Handles missing fields gracefully with defaults.

        Args:
            data: Dictionary from LLM response.

        Returns:
            GameState object.
        """
        # Parse screen type
        screen_str = data.get("current_screen", "unknown")
        try:
            current_screen = ScreenType(screen_str)
        except ValueError:
            current_screen = ScreenType.UNKNOWN

        # Parse resources
        resources: dict[str, Resource] = {}
        raw_resources = data.get("resources", {})
        if isinstance(raw_resources, dict):
            for name, res_data in raw_resources.items():
                if isinstance(res_data, dict):
                    try:
                        res_name = res_data.get("name") or name
                        resources[name] = Resource(
                            name=str(res_name),
                            amount=float(res_data.get("amount", 0)),
                            max_amount=res_data.get("max_amount"),
                            rate=res_data.get("rate"),
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to parse resource {name}: {e}")

        # Parse upgrades
        upgrades = []
        raw_upgrades = data.get("upgrades", [])
        if isinstance(raw_upgrades, list):
            for upg_data in raw_upgrades:
                if isinstance(upg_data, dict):
                    try:
                        upgrades.append(Upgrade(
                            id=upg_data.get("id", "unknown"),
                            name=upg_data.get("name", "Unknown"),
                            description=upg_data.get("description"),
                            level=int(upg_data.get("level", 0)),
                            max_level=upg_data.get("max_level"),
                            cost=upg_data.get("cost", {}),
                            effect=upg_data.get("effect", {}),
                            available=upg_data.get("available", True),
                        ))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to parse upgrade: {e}")

        # Parse UI elements
        ui_elements = []
        raw_elements = data.get("ui_elements", [])
        if isinstance(raw_elements, list):
            for elem_data in raw_elements:
                if isinstance(elem_data, dict):
                    try:
                        metadata = {}
                        if "mark_id" in elem_data:
                            metadata["mark_id"] = elem_data["mark_id"]

                        ui_elements.append(UIElement(
                            element_type=elem_data.get("element_type", "unknown"),
                            x=int(elem_data.get("x", 0)),
                            y=int(elem_data.get("y", 0)),
                            width=int(elem_data.get("width", 10)),
                            height=int(elem_data.get("height", 10)),
                            label=elem_data.get("label"),
                            confidence=float(elem_data.get("confidence", 1.0)),
                            clickable=elem_data.get("clickable", True),
                            metadata=metadata,
                        ))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to parse UI element: {e}")

        # Parse raw text
        raw_text = data.get("raw_text", [])
        if not isinstance(raw_text, list):
            raw_text = []

        return GameState(
            resources=resources,
            upgrades=upgrades,
            current_screen=current_screen,
            ui_elements=ui_elements,
            timestamp=datetime.now(),
            raw_text=raw_text,
        )

    def analyze(self, screenshot: Screenshot) -> GameState:
        """Analyze a screenshot and extract game state.

        This method:
        1. Checks the cache for identical frames
        2. If not cached, sends screenshot to the vision LLM
        3. Parses the response into a GameState object
        4. Caches the result

        Args:
            screenshot: Screenshot to analyze.

        Returns:
            GameState extracted from the screenshot.

        Raises:
            LLMVisionError: If analysis fails after retries.
        """
        # Check cache
        cache_key = self._compute_cache_key(screenshot)
        if cache_key in self._cache:
            self._cache_hits += 1
            # Move to end for LRU
            self._cache.move_to_end(cache_key)
            logger.debug(f"Cache hit for screenshot {cache_key[:8]}")
            return self._cache[cache_key]

        self._cache_misses += 1

        # Convert image to base64
        image_base64 = self._image_to_base64(screenshot.image)
        prompt = self._format_prompt(use_som=False)

        # Call API with retries
        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = self._call_vision_api(image_base64, prompt)
                game_state = self._parse_game_state(response)

                # Cache the result
                self._cache[cache_key] = game_state
                # Evict oldest if cache is full
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)

                logger.debug(f"Analyzed screenshot {cache_key[:8]}: screen={game_state.current_screen}")
                return game_state

            except LLMVisionError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        f"API call failed (attempt {attempt + 1}/{self._max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)

        raise LLMVisionError(f"Failed after {self._max_retries} attempts: {last_error}")

    def analyze_with_som(
        self,
        screenshot: Screenshot,
        regions: list[dict[str, int]] | None = None,
    ) -> tuple[GameState, Image.Image]:
        """Analyze a screenshot with Set-of-Mark annotation.

        Set-of-Mark adds numbered visual markers to the image before
        sending to the LLM. This improves grounding accuracy by giving
        the model explicit reference points for UI elements.

        Args:
            screenshot: Screenshot to analyze.
            regions: Optional list of regions to mark. If None, will be
                auto-detected or use a grid.

        Returns:
            Tuple of (GameState, marked_image).

        Raises:
            LLMVisionError: If analysis fails.
        """
        # Generate regions if not provided
        if regions is None:
            # Create a simple grid of regions for initial marking
            # In a full implementation, this would use pre-detection
            regions = self._generate_grid_regions(
                screenshot.width,
                screenshot.height,
                grid_size=5,
            )

        # Add markers to image
        marked_image = add_set_of_mark(screenshot.image, regions)

        # Convert to base64
        image_base64 = self._image_to_base64(marked_image)
        prompt = self._format_prompt(use_som=True)

        # Call API with retries
        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = self._call_vision_api(image_base64, prompt)
                game_state = self._parse_game_state(response)

                # Add mark_id to UI elements based on position
                ui_elements_with_marks = []
                for elem in game_state.ui_elements:
                    # Find closest mark
                    mark_id = self._find_closest_mark(elem, regions)
                    metadata = dict(elem.metadata)
                    metadata["mark_id"] = mark_id
                    ui_elements_with_marks.append(UIElement(
                        element_type=elem.element_type,
                        x=elem.x,
                        y=elem.y,
                        width=elem.width,
                        height=elem.height,
                        label=elem.label,
                        confidence=elem.confidence,
                        clickable=elem.clickable,
                        metadata=metadata,
                    ))

                # Create new GameState with marked elements
                game_state = GameState(
                    resources=game_state.resources,
                    upgrades=game_state.upgrades,
                    current_screen=game_state.current_screen,
                    ui_elements=ui_elements_with_marks,
                    timestamp=game_state.timestamp,
                    raw_text=game_state.raw_text,
                )

                logger.debug(f"Analyzed with SoM: {len(ui_elements_with_marks)} elements marked")
                return game_state, marked_image

            except LLMVisionError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        f"API call failed (attempt {attempt + 1}/{self._max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)

        raise LLMVisionError(f"Failed after {self._max_retries} attempts: {last_error}")

    def _generate_grid_regions(
        self,
        width: int,
        height: int,
        grid_size: int = 5,
    ) -> list[dict[str, int]]:
        """Generate a grid of regions for Set-of-Mark.

        Args:
            width: Image width.
            height: Image height.
            grid_size: Number of cells per dimension.

        Returns:
            List of region dictionaries.
        """
        regions = []
        cell_w = width // grid_size
        cell_h = height // grid_size

        for row in range(grid_size):
            for col in range(grid_size):
                regions.append({
                    "x": col * cell_w,
                    "y": row * cell_h,
                    "width": cell_w,
                    "height": cell_h,
                })

        return regions

    def _find_closest_mark(
        self,
        element: UIElement,
        regions: list[dict[str, int]],
    ) -> int:
        """Find the closest mark to a UI element.

        Args:
            element: UI element to match.
            regions: List of marked regions.

        Returns:
            Mark ID (1-indexed) of closest mark.
        """
        elem_center_x = element.x + element.width // 2
        elem_center_y = element.y + element.height // 2

        min_dist = float("inf")
        closest_idx = 0

        for idx, region in enumerate(regions):
            region_center_x = region["x"] + region["width"] // 2
            region_center_y = region["y"] + region["height"] // 2

            dist = (
                (elem_center_x - region_center_x) ** 2 +
                (elem_center_y - region_center_y) ** 2
            ) ** 0.5

            if dist < min_dist:
                min_dist = dist
                closest_idx = idx

        return closest_idx + 1  # 1-indexed
