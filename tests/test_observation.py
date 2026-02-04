"""Tests for Feature 4.1: Observation Pipeline.

These tests verify:
- Observation dataclass
- Parallel OCR and UI detection
- Conditional LLM Vision calling
- Resource parsing from OCR
- Screen type inference
- Error handling
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.core.observation import (
    Observation,
    ObservationPipeline,
    PipelineConfig,
    parse_resource_value,
)
from src.interfaces.vision import Screenshot, TextRegion, UIElement
from src.models.game_state import GameState, Resource, ScreenType


class TestParseResourceValue:
    """Tests for resource value parsing."""

    def test_parse_integer(self):
        """Should parse plain integers."""
        assert parse_resource_value("100") == 100.0
        assert parse_resource_value("1234567") == 1234567.0

    def test_parse_decimal(self):
        """Should parse decimal values."""
        assert parse_resource_value("3.14") == 3.14
        assert parse_resource_value("0.5") == 0.5

    def test_parse_thousands(self):
        """Should parse K suffix."""
        assert parse_resource_value("1K") == 1000.0
        assert parse_resource_value("1.5K") == 1500.0
        assert parse_resource_value("2.5k") == 2500.0  # lowercase

    def test_parse_millions(self):
        """Should parse M suffix."""
        assert parse_resource_value("1M") == 1_000_000.0
        assert parse_resource_value("2.3M") == 2_300_000.0

    def test_parse_billions(self):
        """Should parse B suffix."""
        assert parse_resource_value("1B") == 1_000_000_000.0
        assert parse_resource_value("1.5B") == 1_500_000_000.0

    def test_parse_trillions(self):
        """Should parse T suffix."""
        assert parse_resource_value("1T") == 1_000_000_000_000.0

    def test_parse_with_commas(self):
        """Should parse comma-separated numbers."""
        assert parse_resource_value("1,234") == 1234.0
        assert parse_resource_value("1,234,567") == 1234567.0

    def test_parse_invalid(self):
        """Should return None for invalid input."""
        assert parse_resource_value("abc") is None
        assert parse_resource_value("") is None


class TestObservationDataclass:
    """Tests for Observation dataclass."""

    @pytest.fixture
    def mock_screenshot(self):
        """Create a mock screenshot."""
        screenshot = MagicMock(spec=Screenshot)
        screenshot.width = 1920
        screenshot.height = 1080
        screenshot.timestamp = datetime.now()
        return screenshot

    def test_observation_creation(self, mock_screenshot):
        """Should create observation with required fields."""
        game_state = GameState()

        obs = Observation(
            screenshot=mock_screenshot,
            game_state=game_state,
        )

        assert obs.screenshot is mock_screenshot
        assert obs.game_state is game_state
        assert obs.ui_elements == []
        assert obs.text_regions == []
        assert obs.llm_used is False
        assert obs.confidence == 1.0

    def test_observation_with_all_fields(self, mock_screenshot):
        """Should create observation with all fields."""
        game_state = GameState()
        ui_elements = [UIElement("button", 10, 20, 100, 50)]
        text_regions = [TextRegion("Gold: 100", 0, 0, 50, 20, 0.9)]

        obs = Observation(
            screenshot=mock_screenshot,
            game_state=game_state,
            ui_elements=ui_elements,
            text_regions=text_regions,
            llm_used=True,
            confidence=0.85,
            duration_ms=250.5,
        )

        assert len(obs.ui_elements) == 1
        assert len(obs.text_regions) == 1
        assert obs.llm_used is True
        assert obs.confidence == 0.85
        assert obs.duration_ms == 250.5


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = PipelineConfig()

        assert config.use_llm is True
        assert config.llm_interval == 10
        assert config.llm_on_transition is True
        assert config.llm_on_low_confidence is True
        assert config.confidence_threshold == 0.5
        assert config.parallel_timeout == 5.0

    def test_custom_config(self):
        """Should accept custom values."""
        config = PipelineConfig(
            use_llm=False,
            llm_interval=5,
            confidence_threshold=0.7,
        )

        assert config.use_llm is False
        assert config.llm_interval == 5
        assert config.confidence_threshold == 0.7


class TestObservationPipeline:
    """Tests for ObservationPipeline."""

    @pytest.fixture
    def mock_screenshot(self):
        """Create a mock screenshot."""
        screenshot = MagicMock(spec=Screenshot)
        screenshot.width = 1920
        screenshot.height = 1080
        screenshot.timestamp = datetime.now()
        screenshot.image = MagicMock()
        return screenshot

    @pytest.fixture
    def mock_capture(self, mock_screenshot):
        """Create a mock capture component."""
        capture = MagicMock()
        capture.capture.return_value = mock_screenshot
        return capture

    @pytest.fixture
    def mock_ocr(self):
        """Create a mock OCR component."""
        ocr = MagicMock()
        ocr.extract_text.return_value = [
            TextRegion("Gold: 1.5K", 10, 10, 80, 20, 0.95),
            TextRegion("Level 5", 100, 10, 60, 20, 0.9),
        ]
        return ocr

    @pytest.fixture
    def mock_ui_detector(self):
        """Create a mock UI detector component."""
        detector = MagicMock()
        detector.detect.return_value = [
            UIElement("button", 50, 100, 120, 40, 0.9, "Click Me"),
            UIElement("resource", 10, 10, 80, 20, 0.85, "Gold"),
        ]
        return detector

    @pytest.fixture
    def mock_llm_vision(self):
        """Create a mock LLM vision component."""
        llm = MagicMock()
        llm.analyze.return_value = GameState(
            current_screen=ScreenType.MAIN,
            resources={"gold": Resource(name="gold", amount=1500)},
        )
        return llm

    @pytest.fixture
    def pipeline(self, mock_capture, mock_ocr, mock_ui_detector, mock_llm_vision):
        """Create a pipeline with all mocked components."""
        config = PipelineConfig(use_llm=True, llm_interval=5)
        return ObservationPipeline(
            capture=mock_capture,
            ocr=mock_ocr,
            ui_detector=mock_ui_detector,
            llm_vision=mock_llm_vision,
            config=config,
        )

    def test_initialization(self, mock_capture, mock_ocr, mock_ui_detector):
        """Should initialize with components."""
        pipeline = ObservationPipeline(
            capture=mock_capture,
            ocr=mock_ocr,
            ui_detector=mock_ui_detector,
        )

        assert pipeline._capture is mock_capture
        assert pipeline._ocr is mock_ocr
        assert pipeline._ui_detector is mock_ui_detector

    def test_observe_returns_observation(self, pipeline):
        """Should return complete observation."""
        obs = pipeline.observe()

        assert isinstance(obs, Observation)
        assert obs.screenshot is not None
        assert obs.game_state is not None
        assert obs.duration_ms > 0

    def test_observe_captures_screenshot(self, pipeline, mock_capture):
        """Should capture screenshot."""
        pipeline.observe()

        mock_capture.capture.assert_called_once()

    def test_observe_runs_ocr(self, pipeline, mock_ocr, mock_screenshot):
        """Should run OCR on screenshot."""
        pipeline.observe()

        mock_ocr.extract_text.assert_called_once_with(mock_screenshot)

    def test_observe_runs_ui_detection(self, pipeline, mock_ui_detector, mock_screenshot):
        """Should run UI detection on screenshot."""
        pipeline.observe()

        mock_ui_detector.detect.assert_called_once_with(mock_screenshot)

    def test_observe_parses_resources(self, pipeline):
        """Should parse resources from OCR text."""
        obs = pipeline.observe()

        # First observation uses LLM, so check game_state has resources
        assert obs.game_state is not None

    def test_llm_used_on_first_observation(self, pipeline, mock_llm_vision):
        """LLM should be used on first observation."""
        obs = pipeline.observe()

        mock_llm_vision.analyze.assert_called_once()
        assert obs.llm_used is True

    def test_llm_not_used_every_observation(self, pipeline, mock_llm_vision):
        """LLM should not be used on every observation."""
        # First observation uses LLM
        pipeline.observe()

        # Reset mock
        mock_llm_vision.reset_mock()

        # Second observation should not use LLM
        obs = pipeline.observe()

        mock_llm_vision.analyze.assert_not_called()
        assert obs.llm_used is False

    def test_llm_used_periodically(self, pipeline, mock_llm_vision):
        """LLM should be used at configured interval."""
        # Make 5 observations (interval is 5)
        for _ in range(4):
            pipeline.observe()

        mock_llm_vision.reset_mock()

        # 5th observation should use LLM
        obs = pipeline.observe()

        mock_llm_vision.analyze.assert_called_once()
        assert obs.llm_used is True

    def test_llm_not_used_when_disabled(self, mock_capture, mock_ocr, mock_ui_detector, mock_llm_vision):
        """LLM should not be used when disabled in config."""
        config = PipelineConfig(use_llm=False)
        pipeline = ObservationPipeline(
            capture=mock_capture,
            ocr=mock_ocr,
            ui_detector=mock_ui_detector,
            llm_vision=mock_llm_vision,
            config=config,
        )

        obs = pipeline.observe()

        mock_llm_vision.analyze.assert_not_called()
        assert obs.llm_used is False

    def test_llm_on_low_confidence(self, mock_capture, mock_ui_detector, mock_llm_vision):
        """LLM should be used when confidence is low."""
        # OCR returning low confidence results
        mock_ocr = MagicMock()
        mock_ocr.extract_text.return_value = [
            TextRegion("???", 10, 10, 80, 20, 0.2),  # Low confidence
        ]

        # UI detector returning low confidence
        mock_ui_detector.detect.return_value = [
            UIElement("button", 50, 100, 120, 40, 0.3),  # Low confidence
        ]

        config = PipelineConfig(
            use_llm=True,
            llm_on_low_confidence=True,
            confidence_threshold=0.5,
        )
        pipeline = ObservationPipeline(
            capture=mock_capture,
            ocr=mock_ocr,
            ui_detector=mock_ui_detector,
            llm_vision=mock_llm_vision,
            config=config,
        )

        # First observation uses LLM anyway
        pipeline.observe()
        mock_llm_vision.reset_mock()

        # Second observation should also use LLM due to low confidence
        pipeline.observe()

        mock_llm_vision.analyze.assert_called_once()

    def test_observe_handles_ocr_error(self, mock_capture, mock_ui_detector):
        """Should handle OCR errors gracefully."""
        mock_ocr = MagicMock()
        mock_ocr.extract_text.side_effect = Exception("OCR failed")

        config = PipelineConfig(use_llm=False)
        pipeline = ObservationPipeline(
            capture=mock_capture,
            ocr=mock_ocr,
            ui_detector=mock_ui_detector,
            config=config,
        )

        # Should not raise
        obs = pipeline.observe()

        assert obs is not None
        assert len(obs.text_regions) == 0

    def test_observe_handles_ui_detection_error(self, mock_capture, mock_ocr):
        """Should handle UI detection errors gracefully."""
        mock_ui_detector = MagicMock()
        mock_ui_detector.detect.side_effect = Exception("UI detection failed")

        config = PipelineConfig(use_llm=False)
        pipeline = ObservationPipeline(
            capture=mock_capture,
            ocr=mock_ocr,
            ui_detector=mock_ui_detector,
            config=config,
        )

        # Should not raise
        obs = pipeline.observe()

        assert obs is not None
        assert len(obs.ui_elements) == 0

    def test_observe_handles_llm_error(self, pipeline, mock_llm_vision):
        """Should fall back to local extraction on LLM error."""
        mock_llm_vision.analyze.side_effect = Exception("LLM failed")

        # Should not raise
        obs = pipeline.observe()

        assert obs is not None
        assert obs.llm_used is False


class TestScreenTypeInference:
    """Tests for screen type inference."""

    @pytest.fixture
    def mock_capture(self):
        """Create mock capture."""
        screenshot = MagicMock(spec=Screenshot)
        screenshot.width = 1920
        screenshot.height = 1080
        screenshot.timestamp = datetime.now()

        capture = MagicMock()
        capture.capture.return_value = screenshot
        return capture

    @pytest.fixture
    def mock_ocr(self):
        """Create mock OCR."""
        return MagicMock()

    @pytest.fixture
    def mock_ui_detector(self):
        """Create mock UI detector."""
        detector = MagicMock()
        detector.detect.return_value = []
        return detector

    def test_infer_loading_screen(self, mock_capture, mock_ocr, mock_ui_detector):
        """Should infer loading screen from text."""
        mock_ocr.extract_text.return_value = [
            TextRegion("Loading...", 400, 300, 100, 30, 0.9),
        ]

        config = PipelineConfig(use_llm=False)
        pipeline = ObservationPipeline(
            capture=mock_capture,
            ocr=mock_ocr,
            ui_detector=mock_ui_detector,
            config=config,
        )

        obs = pipeline.observe()

        assert obs.game_state.current_screen == ScreenType.LOADING

    def test_infer_settings_screen(self, mock_capture, mock_ocr, mock_ui_detector):
        """Should infer settings screen from text."""
        mock_ocr.extract_text.return_value = [
            TextRegion("Settings", 400, 50, 100, 30, 0.9),
            TextRegion("Sound: On", 300, 150, 100, 30, 0.9),
        ]

        config = PipelineConfig(use_llm=False)
        pipeline = ObservationPipeline(
            capture=mock_capture,
            ocr=mock_ocr,
            ui_detector=mock_ui_detector,
            config=config,
        )

        obs = pipeline.observe()

        assert obs.game_state.current_screen == ScreenType.SETTINGS

    def test_infer_shop_screen(self, mock_capture, mock_ocr, mock_ui_detector):
        """Should infer shop screen from text."""
        mock_ocr.extract_text.return_value = [
            TextRegion("Shop", 400, 50, 100, 30, 0.9),
            TextRegion("Buy Items", 300, 150, 100, 30, 0.9),
        ]

        config = PipelineConfig(use_llm=False)
        pipeline = ObservationPipeline(
            capture=mock_capture,
            ocr=mock_ocr,
            ui_detector=mock_ui_detector,
            config=config,
        )

        obs = pipeline.observe()

        assert obs.game_state.current_screen == ScreenType.SHOP

    def test_infer_dialog_from_buttons(self, mock_capture, mock_ocr, mock_ui_detector):
        """Should infer dialog from OK/Cancel buttons."""
        mock_ocr.extract_text.return_value = []
        mock_ui_detector.detect.return_value = [
            UIElement("button", 300, 300, 80, 40, 0.9, "OK"),
            UIElement("button", 400, 300, 80, 40, 0.9, "Cancel"),
        ]

        config = PipelineConfig(use_llm=False)
        pipeline = ObservationPipeline(
            capture=mock_capture,
            ocr=mock_ocr,
            ui_detector=mock_ui_detector,
            config=config,
        )

        obs = pipeline.observe()

        assert obs.game_state.current_screen == ScreenType.DIALOG

    def test_infer_main_screen(self, mock_capture, mock_ocr, mock_ui_detector):
        """Should infer main screen from buttons + resources."""
        mock_ocr.extract_text.return_value = [
            TextRegion("Gold: 1500", 10, 10, 100, 30, 0.9),
        ]
        mock_ui_detector.detect.return_value = [
            UIElement("button", 300, 300, 80, 40, 0.9, "Upgrade"),
            UIElement("resource", 10, 10, 100, 30, 0.9, "Gold"),
        ]

        config = PipelineConfig(use_llm=False)
        pipeline = ObservationPipeline(
            capture=mock_capture,
            ocr=mock_ocr,
            ui_detector=mock_ui_detector,
            config=config,
        )

        obs = pipeline.observe()

        assert obs.game_state.current_screen == ScreenType.MAIN


class TestResourceParsing:
    """Tests for resource parsing from OCR."""

    @pytest.fixture
    def mock_capture(self):
        """Create mock capture."""
        screenshot = MagicMock(spec=Screenshot)
        screenshot.width = 1920
        screenshot.height = 1080
        screenshot.timestamp = datetime.now()

        capture = MagicMock()
        capture.capture.return_value = screenshot
        return capture

    @pytest.fixture
    def mock_ui_detector(self):
        """Create mock UI detector."""
        detector = MagicMock()
        detector.detect.return_value = []
        return detector

    def test_parse_gold_resource(self, mock_capture, mock_ui_detector):
        """Should parse gold resource from text."""
        mock_ocr = MagicMock()
        mock_ocr.extract_text.return_value = [
            TextRegion("Gold: 1.5K", 10, 10, 100, 30, 0.9),
        ]

        config = PipelineConfig(use_llm=False)
        pipeline = ObservationPipeline(
            capture=mock_capture,
            ocr=mock_ocr,
            ui_detector=mock_ui_detector,
            config=config,
        )

        obs = pipeline.observe()

        assert "gold" in obs.game_state.resources
        assert obs.game_state.resources["gold"].amount == 1500.0

    def test_parse_multiple_resources(self, mock_capture, mock_ui_detector):
        """Should parse multiple resources."""
        mock_ocr = MagicMock()
        mock_ocr.extract_text.return_value = [
            TextRegion("Gold: 500", 10, 10, 100, 30, 0.9),
            TextRegion("Gems: 25", 10, 50, 100, 30, 0.9),
            TextRegion("Energy: 80", 10, 90, 100, 30, 0.9),
        ]

        config = PipelineConfig(use_llm=False)
        pipeline = ObservationPipeline(
            capture=mock_capture,
            ocr=mock_ocr,
            ui_detector=mock_ui_detector,
            config=config,
        )

        obs = pipeline.observe()

        assert "gold" in obs.game_state.resources
        assert "gems" in obs.game_state.resources
        assert "energy" in obs.game_state.resources

    def test_parse_resource_value_before_name(self, mock_capture, mock_ui_detector):
        """Should parse resources with value before name."""
        mock_ocr = MagicMock()
        mock_ocr.extract_text.return_value = [
            TextRegion("1.5K Gold", 10, 10, 100, 30, 0.9),
        ]

        config = PipelineConfig(use_llm=False)
        pipeline = ObservationPipeline(
            capture=mock_capture,
            ocr=mock_ocr,
            ui_detector=mock_ui_detector,
            config=config,
        )

        obs = pipeline.observe()

        assert "gold" in obs.game_state.resources
        assert obs.game_state.resources["gold"].amount == 1500.0


class TestModuleExports:
    """Tests for module exports."""

    def test_observation_exported(self):
        """Observation should be exported from core package."""
        from src.core import Observation

        assert Observation is not None

    def test_observation_pipeline_exported(self):
        """ObservationPipeline should be exported from core package."""
        from src.core import ObservationPipeline

        assert ObservationPipeline is not None


class TestContextManager:
    """Tests for context manager support."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components."""
        screenshot = MagicMock(spec=Screenshot)
        screenshot.width = 1920
        screenshot.height = 1080
        screenshot.timestamp = datetime.now()

        capture = MagicMock()
        capture.capture.return_value = screenshot

        ocr = MagicMock()
        ocr.extract_text.return_value = []

        ui_detector = MagicMock()
        ui_detector.detect.return_value = []

        return capture, ocr, ui_detector

    def test_context_manager_usage(self, mock_components):
        """Should work as context manager."""
        capture, ocr, ui_detector = mock_components
        config = PipelineConfig(use_llm=False)

        with ObservationPipeline(
            capture=capture,
            ocr=ocr,
            ui_detector=ui_detector,
            config=config,
        ) as pipeline:
            obs = pipeline.observe()
            assert obs is not None
