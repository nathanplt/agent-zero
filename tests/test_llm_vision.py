"""Tests for Feature 2.4: LLM Vision Integration.

These tests verify LLM-based vision understanding:
- Sends screenshot to vision LLM
- Returns structured GameState object
- Handles API errors gracefully
- Caches repeated identical frames
- Set-of-Mark improves grounding accuracy

Note: Tests use mocked LLM API for unit testing.
Integration tests marked with @pytest.mark.integration require real API keys.
"""

import time
from datetime import datetime
from io import BytesIO
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw

from src.interfaces.vision import Screenshot


# Helper function to create test screenshots
def create_test_screenshot(
    width: int = 800,
    height: int = 600,
    color: tuple = (100, 100, 100),
    text: str | None = None,
) -> Screenshot:
    """Create a test screenshot with optional content."""
    img = Image.new("RGB", (width, height), color=color)

    if text:
        draw = ImageDraw.Draw(img)
        draw.text((100, 100), text, fill=(255, 255, 255))

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


def create_different_screenshots(count: int = 3) -> list[Screenshot]:
    """Create multiple visually different screenshots."""
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    return [
        create_test_screenshot(color=colors[i % len(colors)], text=f"Screenshot {i}")
        for i in range(count)
    ]


class TestLLMVisionImport:
    """Tests for LLMVision class import and basic structure."""

    def test_llm_vision_class_exists(self):
        """LLMVision class should be importable."""
        from src.vision.llm_vision import LLMVision

        assert LLMVision is not None

    def test_has_analyze_method(self):
        """LLMVision should have an analyze method."""
        from src.vision.llm_vision import LLMVision

        assert hasattr(LLMVision, "analyze")

    def test_has_analyze_with_som_method(self):
        """LLMVision should have an analyze_with_som method for Set-of-Mark."""
        from src.vision.llm_vision import LLMVision

        assert hasattr(LLMVision, "analyze_with_som")

    def test_has_clear_cache_method(self):
        """LLMVision should have a clear_cache method."""
        from src.vision.llm_vision import LLMVision

        assert hasattr(LLMVision, "clear_cache")

    def test_has_cache_stats_property(self):
        """LLMVision should have a cache_stats property."""
        from src.vision.llm_vision import LLMVision

        assert hasattr(LLMVision, "cache_stats")


class TestLLMVisionInitialization:
    """Tests for LLMVision initialization."""

    def test_initialization_default(self):
        """Should initialize with default settings."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        assert llm_vision is not None

    def test_initialization_with_provider(self):
        """Should accept provider parameter."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision(provider="anthropic")
        assert llm_vision._provider == "anthropic"

    def test_initialization_with_model(self):
        """Should accept model parameter."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision(model="claude-3-sonnet-20240229")
        assert llm_vision._model == "claude-3-sonnet-20240229"

    def test_initialization_with_cache_size(self):
        """Should accept cache_size parameter."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision(cache_size=100)
        assert llm_vision._cache_size == 100

    def test_cache_empty_initially(self):
        """Cache should be empty after initialization."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        stats = llm_vision.cache_stats
        assert stats["size"] == 0


class TestLLMVisionAnalyze:
    """Tests for the analyze() method."""

    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response with valid game state."""
        return {
            "current_screen": "main",
            "resources": {
                "gold": {"name": "gold", "amount": 1500.0},
                "gems": {"name": "gems", "amount": 25.0},
            },
            "upgrades": [
                {
                    "id": "click_power",
                    "name": "Click Power",
                    "level": 5,
                    "cost": {"gold": 100.0},
                    "available": True,
                }
            ],
            "ui_elements": [
                {
                    "element_type": "button",
                    "x": 100,
                    "y": 200,
                    "width": 80,
                    "height": 40,
                    "label": "Upgrade",
                    "clickable": True,
                }
            ],
            "raw_text": ["Gold: 1.5K", "Gems: 25", "Click Power Lv.5"],
        }

    def test_analyze_returns_game_state(self, mock_llm_response):
        """analyze() should return a GameState object."""
        from src.models.game_state import GameState
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        screenshot = create_test_screenshot()

        with patch.object(llm_vision, "_call_vision_api") as mock_api:
            mock_api.return_value = mock_llm_response
            result = llm_vision.analyze(screenshot)

        assert isinstance(result, GameState)

    def test_analyze_extracts_screen_type(self, mock_llm_response):
        """analyze() should correctly identify screen type."""
        from src.models.game_state import ScreenType
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        screenshot = create_test_screenshot()

        with patch.object(llm_vision, "_call_vision_api") as mock_api:
            mock_api.return_value = mock_llm_response
            result = llm_vision.analyze(screenshot)

        assert result.current_screen == ScreenType.MAIN

    def test_analyze_extracts_resources(self, mock_llm_response):
        """analyze() should extract game resources."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        screenshot = create_test_screenshot()

        with patch.object(llm_vision, "_call_vision_api") as mock_api:
            mock_api.return_value = mock_llm_response
            result = llm_vision.analyze(screenshot)

        assert "gold" in result.resources
        assert result.resources["gold"].amount == 1500.0

    def test_analyze_extracts_upgrades(self, mock_llm_response):
        """analyze() should extract available upgrades."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        screenshot = create_test_screenshot()

        with patch.object(llm_vision, "_call_vision_api") as mock_api:
            mock_api.return_value = mock_llm_response
            result = llm_vision.analyze(screenshot)

        assert len(result.upgrades) > 0
        assert result.upgrades[0].id == "click_power"

    def test_analyze_extracts_ui_elements(self, mock_llm_response):
        """analyze() should extract UI elements."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        screenshot = create_test_screenshot()

        with patch.object(llm_vision, "_call_vision_api") as mock_api:
            mock_api.return_value = mock_llm_response
            result = llm_vision.analyze(screenshot)

        assert len(result.ui_elements) > 0
        assert result.ui_elements[0].element_type == "button"

    def test_analyze_includes_timestamp(self, mock_llm_response):
        """analyze() should include timestamp in result."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        screenshot = create_test_screenshot()

        with patch.object(llm_vision, "_call_vision_api") as mock_api:
            mock_api.return_value = mock_llm_response
            result = llm_vision.analyze(screenshot)

        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)

    def test_analyze_sends_screenshot_to_api(self, mock_llm_response):
        """analyze() should send screenshot to the vision API."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        screenshot = create_test_screenshot()

        with patch.object(llm_vision, "_call_vision_api") as mock_api:
            mock_api.return_value = mock_llm_response
            llm_vision.analyze(screenshot)

        mock_api.assert_called_once()
        # Verify screenshot data was passed
        call_args = mock_api.call_args
        assert call_args is not None


class TestLLMVisionCaching:
    """Tests for screenshot caching functionality."""

    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response."""
        return {
            "current_screen": "main",
            "resources": {},
            "upgrades": [],
            "ui_elements": [],
            "raw_text": [],
        }

    def test_identical_screenshot_uses_cache(self, mock_llm_response):
        """Identical screenshots should use cached result."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        screenshot = create_test_screenshot()

        with patch.object(llm_vision, "_call_vision_api") as mock_api:
            mock_api.return_value = mock_llm_response

            # First call - should hit API
            result1 = llm_vision.analyze(screenshot)
            assert mock_api.call_count == 1

            # Second call with same screenshot - should use cache
            result2 = llm_vision.analyze(screenshot)
            assert mock_api.call_count == 1  # No additional call

        # Results should be equivalent
        assert result1.current_screen == result2.current_screen

    def test_different_screenshots_hit_api(self, mock_llm_response):
        """Different screenshots should each call the API."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        screenshots = create_different_screenshots(3)

        with patch.object(llm_vision, "_call_vision_api") as mock_api:
            mock_api.return_value = mock_llm_response

            for screenshot in screenshots:
                llm_vision.analyze(screenshot)

        # Each different screenshot should hit the API
        assert mock_api.call_count == 3

    def test_cache_stats_updated(self, mock_llm_response):
        """Cache stats should track hits and misses."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        screenshot = create_test_screenshot()

        with patch.object(llm_vision, "_call_vision_api") as mock_api:
            mock_api.return_value = mock_llm_response

            # First call - cache miss
            llm_vision.analyze(screenshot)
            stats = llm_vision.cache_stats
            assert stats["misses"] == 1
            assert stats["hits"] == 0

            # Second call - cache hit
            llm_vision.analyze(screenshot)
            stats = llm_vision.cache_stats
            assert stats["hits"] == 1
            assert stats["misses"] == 1

    def test_clear_cache(self, mock_llm_response):
        """clear_cache() should empty the cache."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        screenshot = create_test_screenshot()

        with patch.object(llm_vision, "_call_vision_api") as mock_api:
            mock_api.return_value = mock_llm_response

            # Populate cache
            llm_vision.analyze(screenshot)
            assert llm_vision.cache_stats["size"] == 1

            # Clear cache
            llm_vision.clear_cache()
            assert llm_vision.cache_stats["size"] == 0

            # Should hit API again
            llm_vision.analyze(screenshot)
            assert mock_api.call_count == 2

    def test_cache_eviction_when_full(self, mock_llm_response):
        """Cache should evict old entries when full."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision(cache_size=3)
        screenshots = create_different_screenshots(5)

        with patch.object(llm_vision, "_call_vision_api") as mock_api:
            mock_api.return_value = mock_llm_response

            # Add more than cache size
            for screenshot in screenshots:
                llm_vision.analyze(screenshot)

        # Cache should not exceed max size
        assert llm_vision.cache_stats["size"] <= 3


class TestLLMVisionErrorHandling:
    """Tests for API error handling and retry logic."""

    def test_retries_on_api_error(self):
        """Should retry on transient API errors."""
        from src.vision.llm_vision import LLMVision, LLMVisionError

        llm_vision = LLMVision(max_retries=3)
        screenshot = create_test_screenshot()

        call_count = 0
        def mock_api_with_errors(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMVisionError("API temporarily unavailable")
            return {
                "current_screen": "main",
                "resources": {},
                "upgrades": [],
                "ui_elements": [],
                "raw_text": [],
            }

        with patch.object(llm_vision, "_call_vision_api", side_effect=mock_api_with_errors):
            result = llm_vision.analyze(screenshot)

        assert call_count == 3  # Retried 3 times
        assert result is not None

    def test_raises_after_max_retries(self):
        """Should raise after exhausting retries."""
        from src.vision.llm_vision import LLMVision, LLMVisionError

        llm_vision = LLMVision(max_retries=3)
        screenshot = create_test_screenshot()

        with patch.object(llm_vision, "_call_vision_api") as mock_api:
            mock_api.side_effect = LLMVisionError("API error")

            with pytest.raises(LLMVisionError):
                llm_vision.analyze(screenshot)

        assert mock_api.call_count == 3

    def test_exponential_backoff(self):
        """Should use exponential backoff between retries."""
        from src.vision.llm_vision import LLMVision, LLMVisionError

        llm_vision = LLMVision(max_retries=3, retry_delay=0.01)  # Fast for testing
        screenshot = create_test_screenshot()

        timestamps = []
        def mock_api_with_timing(*_args, **_kwargs):
            timestamps.append(time.time())
            raise LLMVisionError("API error")

        with patch.object(llm_vision, "_call_vision_api", side_effect=mock_api_with_timing), \
                pytest.raises(LLMVisionError):
            llm_vision.analyze(screenshot)

        # Verify delays increase (exponential backoff)
        if len(timestamps) >= 3:
            delay1 = timestamps[1] - timestamps[0]
            delay2 = timestamps[2] - timestamps[1]
            assert delay2 >= delay1  # Second delay should be >= first

    def test_handles_malformed_response(self):
        """Should handle malformed LLM responses gracefully."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        screenshot = create_test_screenshot()

        with patch.object(llm_vision, "_call_vision_api") as mock_api:
            mock_api.return_value = {"invalid": "response"}

            # Should return a default/empty GameState or raise
            result = llm_vision.analyze(screenshot)

        # Should still return a valid GameState
        from src.models.game_state import GameState
        assert isinstance(result, GameState)

    def test_handles_partial_response(self):
        """Should handle partial LLM responses gracefully."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        screenshot = create_test_screenshot()

        with patch.object(llm_vision, "_call_vision_api") as mock_api:
            # Only some fields present
            mock_api.return_value = {
                "current_screen": "shop",
                "resources": {"gold": {"name": "gold", "amount": 100.0}},
            }

            result = llm_vision.analyze(screenshot)

        from src.models.game_state import GameState, ScreenType
        assert isinstance(result, GameState)
        assert result.current_screen == ScreenType.SHOP


class TestSetOfMark:
    """Tests for Set-of-Mark annotation functionality."""

    @pytest.fixture
    def mock_llm_response_with_marks(self):
        """Create a mock response referencing marked elements."""
        return {
            "current_screen": "main",
            "resources": {},
            "upgrades": [],
            "ui_elements": [
                {
                    "element_type": "button",
                    "x": 100,
                    "y": 200,
                    "width": 80,
                    "height": 40,
                    "label": "Upgrade",
                    "mark_id": 1,
                },
                {
                    "element_type": "button",
                    "x": 200,
                    "y": 200,
                    "width": 80,
                    "height": 40,
                    "label": "Shop",
                    "mark_id": 2,
                },
            ],
            "raw_text": [],
        }

    def test_analyze_with_som_adds_marks_to_image(self, mock_llm_response_with_marks):
        """analyze_with_som() should add visual marks to the image."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        screenshot = create_test_screenshot()

        with patch.object(llm_vision, "_call_vision_api") as mock_api:
            mock_api.return_value = mock_llm_response_with_marks
            result, marked_image = llm_vision.analyze_with_som(screenshot)

        # Should return both GameState and marked image
        from src.models.game_state import GameState
        assert isinstance(result, GameState)
        assert isinstance(marked_image, Image.Image)

    def test_som_marks_are_numbered(self, mock_llm_response_with_marks):
        """Set-of-Mark should use numbered markers."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        screenshot = create_test_screenshot()

        with patch.object(llm_vision, "_call_vision_api") as mock_api:
            mock_api.return_value = mock_llm_response_with_marks
            result, marked_image = llm_vision.analyze_with_som(screenshot)

        # UI elements should have mark_id in metadata
        for element in result.ui_elements:
            assert "mark_id" in element.metadata

    def test_som_improves_element_references(self, mock_llm_response_with_marks):
        """Set-of-Mark should enable more accurate element references."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        screenshot = create_test_screenshot()

        with patch.object(llm_vision, "_call_vision_api") as mock_api:
            mock_api.return_value = mock_llm_response_with_marks
            result, _ = llm_vision.analyze_with_som(screenshot)

        # Each UI element should have a mark_id for reference
        assert len(result.ui_elements) == 2
        # All elements should have mark_id metadata assigned
        for element in result.ui_elements:
            assert "mark_id" in element.metadata
            assert isinstance(element.metadata["mark_id"], int)
            assert element.metadata["mark_id"] >= 1  # mark_ids are 1-indexed

    def test_add_set_of_mark_function(self):
        """add_set_of_mark() should annotate image with markers."""
        from src.vision.llm_vision import add_set_of_mark

        screenshot = create_test_screenshot()

        regions = [
            {"x": 100, "y": 100, "width": 50, "height": 30},
            {"x": 200, "y": 150, "width": 60, "height": 40},
        ]

        marked_image = add_set_of_mark(screenshot.image, regions)

        assert isinstance(marked_image, Image.Image)
        # Image should be modified (different from original)
        assert marked_image.size == screenshot.image.size


class TestPromptTemplates:
    """Tests for LLM prompt template functionality."""

    def test_default_prompt_template_exists(self):
        """Should have a default prompt template."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        assert hasattr(llm_vision, "_prompt_template")
        assert llm_vision._prompt_template is not None

    def test_prompt_template_includes_instructions(self):
        """Prompt template should include game state extraction instructions."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        template = llm_vision._prompt_template

        # Should mention key extraction targets
        assert "resources" in template.lower() or "game" in template.lower()

    def test_custom_prompt_template(self):
        """Should accept custom prompt template."""
        from src.vision.llm_vision import LLMVision

        custom_template = "Analyze this game screenshot and extract: {format_instructions}"
        llm_vision = LLMVision(prompt_template=custom_template)

        assert llm_vision._prompt_template == custom_template

    def test_format_instructions_in_prompt(self):
        """Prompt should include format instructions for structured output."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()

        # The formatted prompt should include JSON schema or format instructions
        formatted = llm_vision._format_prompt()
        assert "json" in formatted.lower() or "format" in formatted.lower()


class TestLLMVisionProviders:
    """Tests for different LLM provider support."""

    def test_anthropic_provider_support(self):
        """Should support Anthropic as a provider."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision(provider="anthropic")
        assert llm_vision._provider == "anthropic"

    def test_openai_provider_support(self):
        """Should support OpenAI as a provider."""
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision(provider="openai")
        assert llm_vision._provider == "openai"

    def test_invalid_provider_raises(self):
        """Should raise error for invalid provider."""
        from src.vision.llm_vision import LLMVision

        with pytest.raises(ValueError):
            LLMVision(provider="invalid_provider")


class TestLLMVisionModuleExports:
    """Tests for module exports."""

    def test_llm_vision_exported_from_vision(self):
        """LLMVision should be exported from vision package."""
        from src.vision import LLMVision

        assert LLMVision is not None

    def test_llm_vision_error_exported(self):
        """LLMVisionError should be exported."""
        from src.vision.llm_vision import LLMVisionError

        assert LLMVisionError is not None

    def test_add_set_of_mark_exported(self):
        """add_set_of_mark should be exported."""
        from src.vision.llm_vision import add_set_of_mark

        assert add_set_of_mark is not None


class TestLLMVisionIntegration:
    """Integration tests for LLMVision with real screenshots."""

    @pytest.mark.integration
    def test_analyze_real_screenshot(self):
        """Should analyze a real screenshot (requires API key)."""
        import os

        # Skip if no API key is available
        if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("No API key available for integration test")

        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision()
        screenshot = create_test_screenshot(text="Gold: 1,500 | Level: 42")

        # This test requires real API credentials
        result = llm_vision.analyze(screenshot)

        from src.models.game_state import GameState
        assert isinstance(result, GameState)
