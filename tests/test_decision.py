"""Tests for Feature 4.2: Decision Engine.

These tests verify:
- DecisionEngine initialization and configuration
- ReAct prompt building
- LLM response parsing
- Decision caching
- Fallback decisions
- Module exports
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.core.decision import (
    DecisionConfig,
    DecisionEngine,
    DecisionEngineError,
)
from src.core.observation import Observation
from src.core.prompts import PROMPT_VERSION, DecisionPrompts
from src.interfaces.actions import ActionType
from src.interfaces.vision import Screenshot, UIElement
from src.models.decisions import Decision
from src.models.game_state import GameState, Resource, ScreenType


class TestDecisionConfig:
    """Tests for DecisionConfig."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = DecisionConfig()

        assert config.provider == "anthropic"
        assert config.cache_size == 100
        assert config.max_retries == 3
        assert config.temperature == 0.3

    def test_custom_config(self):
        """Should accept custom values."""
        config = DecisionConfig(
            provider="openai",
            model="gpt-4",
            cache_size=50,
            temperature=0.5,
        )

        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.cache_size == 50
        assert config.temperature == 0.5


class TestDecisionPrompts:
    """Tests for DecisionPrompts."""

    @pytest.fixture
    def mock_observation(self):
        """Create a mock observation."""
        screenshot = MagicMock(spec=Screenshot)
        screenshot.width = 1920
        screenshot.height = 1080
        screenshot.timestamp = datetime.now()

        game_state = GameState(
            current_screen=ScreenType.MAIN,
            resources={
                "gold": Resource(name="gold", amount=1500),
                "gems": Resource(name="gems", amount=25),
            },
        )

        return Observation(
            screenshot=screenshot,
            game_state=game_state,
            ui_elements=[
                UIElement("button", 100, 200, 80, 40, 0.9, "Upgrade"),
                UIElement("resource", 10, 10, 100, 30, 0.85, "Gold"),
            ],
            timestamp=datetime.now(),
        )

    def test_prompt_version_exists(self):
        """Should have a version string."""
        assert PROMPT_VERSION is not None
        assert len(PROMPT_VERSION) > 0

    def test_build_react_prompt(self, mock_observation):
        """Should build a complete ReAct prompt."""
        prompts = DecisionPrompts()

        prompt = prompts.build_react_prompt(mock_observation)

        # Should contain key sections
        assert "Screen Type" in prompt
        assert "main" in prompt.lower()
        assert "Resources" in prompt
        assert "Gold" in prompt or "gold" in prompt.lower()
        assert "UI Elements" in prompt
        assert "button" in prompt.lower()
        assert "JSON" in prompt

    def test_build_react_prompt_with_goal(self, mock_observation):
        """Should include goal in prompt."""
        prompts = DecisionPrompts()

        prompt = prompts.build_react_prompt(
            mock_observation,
            goal="Maximize gold production",
        )

        assert "Maximize gold production" in prompt

    def test_build_react_prompt_with_recent_actions(self, mock_observation):
        """Should include recent actions."""
        prompts = DecisionPrompts()
        recent_actions = [
            {"type": "click", "result": "success"},
            {"type": "wait", "result": "success"},
        ]

        prompt = prompts.build_react_prompt(
            mock_observation,
            recent_actions=recent_actions,
        )

        assert "Recent Actions" in prompt
        assert "click" in prompt.lower()

    def test_format_number_thousands(self):
        """Should format thousands with K suffix."""
        prompts = DecisionPrompts()

        assert prompts._format_number(1500) == "1.50K"
        assert prompts._format_number(10000) == "10.00K"

    def test_format_number_millions(self):
        """Should format millions with M suffix."""
        prompts = DecisionPrompts()

        assert prompts._format_number(1500000) == "1.50M"
        assert prompts._format_number(10000000) == "10.00M"

    def test_format_number_billions(self):
        """Should format billions with B suffix."""
        prompts = DecisionPrompts()

        assert prompts._format_number(1500000000) == "1.50B"

    def test_format_number_small(self):
        """Should format small numbers without suffix."""
        prompts = DecisionPrompts()

        assert prompts._format_number(50) == "50.00"
        assert prompts._format_number(500) == "500"


class TestDecisionEngineInitialization:
    """Tests for DecisionEngine initialization."""

    def test_initialization_default(self):
        """Should initialize with default config."""
        engine = DecisionEngine()

        assert engine._config.provider == "anthropic"
        assert engine._prompts is not None

    def test_initialization_with_config(self):
        """Should accept custom config."""
        config = DecisionConfig(provider="openai", cache_size=50)
        engine = DecisionEngine(config=config)

        assert engine._config.provider == "openai"
        assert engine._config.cache_size == 50

    def test_initialization_invalid_provider(self):
        """Should reject invalid provider."""
        config = DecisionConfig(provider="invalid")

        with pytest.raises(ValueError) as exc_info:
            DecisionEngine(config=config)

        assert "invalid" in str(exc_info.value).lower()


class TestDecisionEngineCache:
    """Tests for decision caching."""

    @pytest.fixture
    def engine(self):
        """Create engine with small cache."""
        config = DecisionConfig(cache_size=5)
        return DecisionEngine(config=config)

    @pytest.fixture
    def mock_observation(self):
        """Create a mock observation."""
        screenshot = MagicMock(spec=Screenshot)
        screenshot.timestamp = datetime.now()

        game_state = GameState(
            current_screen=ScreenType.MAIN,
            resources={"gold": Resource(name="gold", amount=100)},
        )

        return Observation(
            screenshot=screenshot,
            game_state=game_state,
            ui_elements=[],
            timestamp=datetime.now(),
        )

    def test_cache_stats_initial(self, engine):
        """Should have zero stats initially."""
        stats = engine.cache_stats

        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_clear_cache(self, engine):
        """Should clear cache and stats."""
        # Add something to cache (via direct manipulation for test)
        engine._cache["test_key"] = MagicMock()
        engine._cache_hits = 5
        engine._cache_misses = 10

        engine.clear_cache()

        stats = engine.cache_stats
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0


class TestDecisionEngineParsing:
    """Tests for LLM response parsing."""

    @pytest.fixture
    def engine(self):
        """Create engine for parsing tests."""
        return DecisionEngine()

    @pytest.fixture
    def mock_observation(self):
        """Create a mock observation."""
        screenshot = MagicMock(spec=Screenshot)
        screenshot.timestamp = datetime.now()

        game_state = GameState(
            current_screen=ScreenType.MAIN,
            resources={},
        )

        return Observation(
            screenshot=screenshot,
            game_state=game_state,
            ui_elements=[],
            timestamp=datetime.now(),
            confidence=0.9,
        )

    def test_parse_valid_response(self, engine, mock_observation):
        """Should parse valid JSON response."""
        response = json.dumps({
            "thought": "Gold is low, should click upgrade button",
            "action": {
                "type": "click",
                "target": {"x": 100, "y": 200},
                "parameters": {},
                "description": "Click upgrade button",
            },
            "confidence": 0.85,
            "expected_outcome": "Upgrade will be purchased",
            "alternatives": [
                {"action_type": "wait", "reason": "Could wait for more gold"}
            ],
        })

        decision = engine._parse_response(response, mock_observation)

        assert isinstance(decision, Decision)
        assert decision.reasoning == "Gold is low, should click upgrade button"
        assert decision.action.type == ActionType.CLICK
        assert decision.confidence == 0.85
        assert decision.expected_outcome == "Upgrade will be purchased"

    def test_parse_response_with_markdown(self, engine, mock_observation):
        """Should handle markdown code blocks."""
        response = """```json
{
    "thought": "Need to upgrade",
    "action": {
        "type": "click",
        "target": {"x": 50, "y": 50},
        "parameters": {},
        "description": "Click"
    },
    "confidence": 0.7,
    "expected_outcome": "Button clicked"
}
```"""

        decision = engine._parse_response(response, mock_observation)

        assert decision.reasoning == "Need to upgrade"
        assert decision.action.type == ActionType.CLICK

    def test_parse_response_unknown_action_type(self, engine, mock_observation):
        """Should handle unknown action types gracefully."""
        response = json.dumps({
            "thought": "Testing unknown action",
            "action": {
                "type": "unknown_action",
                "target": None,
                "parameters": {},
                "description": "Unknown",
            },
            "confidence": 0.5,
            "expected_outcome": "Unknown",
        })

        decision = engine._parse_response(response, mock_observation)

        # Should default to WAIT
        assert decision.action.type == ActionType.WAIT

    def test_parse_response_missing_target(self, engine, mock_observation):
        """Should handle missing target."""
        response = json.dumps({
            "thought": "Waiting for resources",
            "action": {
                "type": "wait",
                "target": None,
                "parameters": {"duration_ms": 1000},
                "description": "Wait",
            },
            "confidence": 0.9,
            "expected_outcome": "Resources will accumulate",
        })

        decision = engine._parse_response(response, mock_observation)

        assert decision.action.type == ActionType.WAIT
        assert decision.action.target is None

    def test_parse_wait_without_duration_adds_default(self, engine, mock_observation):
        """Wait actions without duration should get a safe default."""
        response = json.dumps({
            "thought": "No safe click available",
            "action": {
                "type": "wait",
                "target": None,
                "parameters": {},
                "description": "Wait",
            },
            "confidence": 0.7,
            "expected_outcome": "State stabilizes",
        })

        decision = engine._parse_response(response, mock_observation)

        assert decision.action.type == ActionType.WAIT
        assert decision.action.parameters.get("duration_ms") == 1000

    def test_parse_unknown_action_defaults_to_wait_with_duration(self, engine, mock_observation):
        """Unknown actions should degrade to a valid wait action."""
        response = json.dumps({
            "thought": "Try unsupported action",
            "action": {
                "type": "unsupported",
                "parameters": {},
                "description": "Unknown action",
            },
            "confidence": 0.5,
            "expected_outcome": "No-op",
        })

        decision = engine._parse_response(response, mock_observation)

        assert decision.action.type == ActionType.WAIT
        assert decision.action.parameters.get("duration_ms") == 1000

    def test_parse_invalid_json(self, engine, mock_observation):
        """Should raise error for invalid JSON."""
        response = "This is not JSON"

        with pytest.raises(ValueError) as exc_info:
            engine._parse_response(response, mock_observation)

        assert "json" in str(exc_info.value).lower()


class TestDecisionEngineFallback:
    """Tests for fallback decisions."""

    @pytest.fixture
    def engine(self):
        """Create engine for fallback tests."""
        return DecisionEngine()

    @pytest.fixture
    def mock_observation(self):
        """Create a mock observation."""
        screenshot = MagicMock(spec=Screenshot)
        screenshot.timestamp = datetime.now()

        game_state = GameState(current_screen=ScreenType.MAIN)

        return Observation(
            screenshot=screenshot,
            game_state=game_state,
            ui_elements=[],
            timestamp=datetime.now(),
        )

    def test_create_fallback_decision(self, engine, mock_observation):
        """Should create safe fallback decision."""
        decision = engine.create_fallback_decision(
            mock_observation,
            reason="LLM timeout",
        )

        assert isinstance(decision, Decision)
        assert decision.action.type == ActionType.WAIT
        assert decision.confidence < 0.5  # Low confidence
        assert "LLM timeout" in decision.reasoning
        assert decision.context.get("fallback") is True


class TestDecisionEngineActionRecording:
    """Tests for action result recording."""

    @pytest.fixture
    def engine(self):
        """Create engine for recording tests."""
        return DecisionEngine()

    def test_record_action_result(self, engine):
        """Should record action results."""
        action = MagicMock()
        action.action_type.value = "click"

        engine.record_action_result(action, success=True, outcome="Button clicked")

        assert len(engine._recent_actions) == 1
        assert engine._recent_actions[0]["type"] == "click"
        assert engine._recent_actions[0]["result"] == "success"

    def test_record_action_limit(self, engine):
        """Should limit recent actions."""
        action = MagicMock()
        action.action_type.value = "click"

        # Record many actions
        for i in range(20):
            engine.record_action_result(action, success=True, outcome=f"Action {i}")

        # Should only keep max_recent_actions
        assert len(engine._recent_actions) <= engine._max_recent_actions


class TestDecisionEngineDecide:
    """Tests for the decide method."""

    @pytest.fixture
    def mock_observation(self):
        """Create a mock observation."""
        screenshot = MagicMock(spec=Screenshot)
        screenshot.timestamp = datetime.now()

        game_state = GameState(
            current_screen=ScreenType.MAIN,
            resources={"gold": Resource(name="gold", amount=1000)},
        )

        return Observation(
            screenshot=screenshot,
            game_state=game_state,
            ui_elements=[
                UIElement("button", 100, 200, 80, 40, 0.9, "Upgrade"),
            ],
            timestamp=datetime.now(),
        )

    def test_decide_calls_llm(self, mock_observation):
        """Should call LLM and return decision."""
        engine = DecisionEngine()

        # Mock the LLM call
        mock_response = json.dumps({
            "thought": "Should upgrade",
            "action": {
                "type": "click",
                "target": {"x": 100, "y": 200},
                "parameters": {},
                "description": "Click upgrade",
            },
            "confidence": 0.85,
            "expected_outcome": "Upgrade purchased",
        })

        with patch.object(engine, "_call_llm", return_value=mock_response):
            decision = engine.decide(mock_observation)

        assert isinstance(decision, Decision)
        assert decision.action.type == ActionType.CLICK

    def test_decide_uses_cache(self, mock_observation):
        """Should use cached decision on identical state."""
        engine = DecisionEngine()

        mock_response = json.dumps({
            "thought": "Test",
            "action": {"type": "wait", "target": None, "parameters": {}, "description": "Wait"},
            "confidence": 0.5,
            "expected_outcome": "Waiting",
        })

        with patch.object(engine, "_call_llm", return_value=mock_response) as mock_llm:
            # First call
            decision1 = engine.decide(mock_observation)

            # Second call with same observation
            decision2 = engine.decide(mock_observation)

        # LLM should only be called once
        assert mock_llm.call_count == 1

        # Decisions should be the same
        assert decision1.reasoning == decision2.reasoning

        # Cache should have hit
        assert engine.cache_stats["hits"] == 1

    def test_decide_bypass_cache(self, mock_observation):
        """Should bypass cache when requested."""
        engine = DecisionEngine()

        mock_response = json.dumps({
            "thought": "Test",
            "action": {"type": "wait", "target": None, "parameters": {}, "description": "Wait"},
            "confidence": 0.5,
            "expected_outcome": "Waiting",
        })

        with patch.object(engine, "_call_llm", return_value=mock_response) as mock_llm:
            # First call
            engine.decide(mock_observation, use_cache=False)

            # Second call bypassing cache
            engine.decide(mock_observation, use_cache=False)

        # LLM should be called twice
        assert mock_llm.call_count == 2

    def test_decide_handles_llm_error(self, mock_observation):
        """Should raise error on LLM failure."""
        engine = DecisionEngine()

        with (
            patch.object(engine, "_call_llm", side_effect=Exception("API error")),
            pytest.raises(DecisionEngineError) as exc_info,
        ):
            engine.decide(mock_observation)

        assert "failed" in str(exc_info.value).lower()


class TestModuleExports:
    """Tests for module exports."""

    def test_decision_engine_exported(self):
        """DecisionEngine should be exported from core package."""
        from src.core import DecisionEngine

        assert DecisionEngine is not None

    def test_decision_config_exported(self):
        """DecisionConfig should be exported from core package."""
        from src.core import DecisionConfig

        assert DecisionConfig is not None

    def test_decision_prompts_exported(self):
        """DecisionPrompts should be exported from core package."""
        from src.core import DecisionPrompts

        assert DecisionPrompts is not None

    def test_decision_engine_error_exported(self):
        """DecisionEngineError should be exported from core package."""
        from src.core import DecisionEngineError

        assert DecisionEngineError is not None
