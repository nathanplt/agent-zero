"""Tests for shared data models.

These tests verify that:
1. Valid instances can be created
2. Invalid instances raise ValidationError
3. Serialization/deserialization works correctly
4. Model methods work as expected
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.models.actions import Action, ActionType, Point
from src.models.decisions import Decision
from src.models.game_state import GameState, Resource, ScreenType, UIElement, Upgrade
from src.models.observations import Observation


class TestResourceModel:
    """Tests for Resource model."""

    def test_create_valid_resource(self) -> None:
        """Test creating a valid resource."""
        resource = Resource(name="gold", amount=100.0)
        assert resource.name == "gold"
        assert resource.amount == 100.0
        assert resource.max_amount is None
        assert resource.rate is None

    def test_create_resource_with_all_fields(self) -> None:
        """Test creating a resource with all optional fields."""
        resource = Resource(
            name="energy",
            amount=50.0,
            max_amount=100.0,
            rate=1.5,
        )
        assert resource.max_amount == 100.0
        assert resource.rate == 1.5

    def test_resource_invalid_negative_amount(self) -> None:
        """Test that negative amount is rejected."""
        with pytest.raises(ValidationError):
            Resource(name="gold", amount=-10.0)

    def test_resource_invalid_empty_name(self) -> None:
        """Test that empty name is rejected."""
        with pytest.raises(ValidationError):
            Resource(name="", amount=10.0)

    def test_resource_is_frozen(self) -> None:
        """Test that resource is immutable."""
        resource = Resource(name="gold", amount=100.0)
        with pytest.raises(ValidationError):
            resource.amount = 200.0  # type: ignore


class TestUpgradeModel:
    """Tests for Upgrade model."""

    def test_create_valid_upgrade(self) -> None:
        """Test creating a valid upgrade."""
        upgrade = Upgrade(id="sword-1", name="Iron Sword")
        assert upgrade.id == "sword-1"
        assert upgrade.name == "Iron Sword"
        assert upgrade.level == 0
        assert upgrade.available is True

    def test_upgrade_is_maxed_no_max(self) -> None:
        """Test is_maxed when no max_level set."""
        upgrade = Upgrade(id="test", name="Test", level=100)
        assert upgrade.is_maxed is False

    def test_upgrade_is_maxed_at_max(self) -> None:
        """Test is_maxed when at max level."""
        upgrade = Upgrade(id="test", name="Test", level=10, max_level=10)
        assert upgrade.is_maxed is True

    def test_upgrade_is_maxed_below_max(self) -> None:
        """Test is_maxed when below max level."""
        upgrade = Upgrade(id="test", name="Test", level=5, max_level=10)
        assert upgrade.is_maxed is False


class TestUIElementModel:
    """Tests for UIElement model."""

    def test_create_valid_ui_element(self) -> None:
        """Test creating a valid UI element."""
        elem = UIElement(
            element_type="button",
            x=100,
            y=200,
            width=50,
            height=30,
        )
        assert elem.element_type == "button"
        assert elem.confidence == 1.0  # default

    def test_ui_element_center(self) -> None:
        """Test center calculation."""
        elem = UIElement(
            element_type="button",
            x=100,
            y=200,
            width=50,
            height=30,
        )
        assert elem.center == (125, 215)

    def test_ui_element_bounds(self) -> None:
        """Test bounds calculation."""
        elem = UIElement(
            element_type="button",
            x=100,
            y=200,
            width=50,
            height=30,
        )
        assert elem.bounds == (100, 200, 150, 230)

    def test_ui_element_invalid_negative_coords(self) -> None:
        """Test that negative coordinates are rejected."""
        with pytest.raises(ValidationError):
            UIElement(
                element_type="button",
                x=-10,
                y=200,
                width=50,
                height=30,
            )

    def test_ui_element_invalid_zero_dimensions(self) -> None:
        """Test that zero dimensions are rejected."""
        with pytest.raises(ValidationError):
            UIElement(
                element_type="button",
                x=100,
                y=200,
                width=0,
                height=30,
            )

    def test_ui_element_invalid_confidence(self) -> None:
        """Test that confidence outside 0-1 is rejected."""
        with pytest.raises(ValidationError):
            UIElement(
                element_type="button",
                x=100,
                y=200,
                width=50,
                height=30,
                confidence=1.5,
            )


class TestGameStateModel:
    """Tests for GameState model."""

    def test_create_empty_game_state(self) -> None:
        """Test creating an empty game state."""
        state = GameState()
        assert state.resources == {}
        assert state.upgrades == []
        assert state.current_screen == ScreenType.UNKNOWN
        assert isinstance(state.timestamp, datetime)

    def test_create_game_state_with_resources(self) -> None:
        """Test creating a game state with resources."""
        gold = Resource(name="gold", amount=100.0)
        state = GameState(resources={"gold": gold})
        assert state.get_resource("gold") == gold
        assert state.get_resource("silver") is None

    def test_game_state_find_element(self) -> None:
        """Test finding UI elements."""
        elem1 = UIElement(element_type="button", x=0, y=0, width=10, height=10, label="Buy")
        elem2 = UIElement(element_type="button", x=100, y=0, width=10, height=10, label="Sell")
        state = GameState(ui_elements=[elem1, elem2])

        found = state.find_element("button", "Sell")
        assert found == elem2

        found_none = state.find_element("menu")
        assert found_none is None

    def test_game_state_find_elements(self) -> None:
        """Test finding all elements of a type."""
        elem1 = UIElement(element_type="button", x=0, y=0, width=10, height=10)
        elem2 = UIElement(element_type="button", x=100, y=0, width=10, height=10)
        elem3 = UIElement(element_type="text", x=50, y=0, width=10, height=10)
        state = GameState(ui_elements=[elem1, elem2, elem3])

        buttons = state.find_elements("button")
        assert len(buttons) == 2

    def test_game_state_serialization(self) -> None:
        """Test serialization to JSON and back."""
        gold = Resource(name="gold", amount=100.0)
        state = GameState(
            resources={"gold": gold},
            current_screen=ScreenType.MAIN,
        )

        json_str = state.model_dump_json()
        assert "gold" in json_str
        assert "100" in json_str

        # Deserialize
        restored = GameState.model_validate_json(json_str)
        assert restored.get_resource("gold") is not None
        assert restored.get_resource("gold").amount == 100.0  # type: ignore


class TestPointModel:
    """Tests for Point model."""

    def test_create_valid_point(self) -> None:
        """Test creating a valid point."""
        point = Point(x=100, y=200)
        assert point.x == 100
        assert point.y == 200

    def test_point_invalid_negative(self) -> None:
        """Test that negative coordinates are rejected."""
        with pytest.raises(ValidationError):
            Point(x=-10, y=200)


class TestActionModel:
    """Tests for Action model."""

    def test_create_click_action_factory(self) -> None:
        """Test creating a click action with factory method."""
        action = Action.click(100, 200, description="Click button")
        assert action.type == ActionType.CLICK
        assert action.target is not None
        assert action.target.x == 100
        assert action.target.y == 200
        assert action.parameters == {"button": "left"}

    def test_create_type_action_factory(self) -> None:
        """Test creating a type action with factory method."""
        action = Action.type_text("hello world")
        assert action.type == ActionType.TYPE
        assert action.parameters["text"] == "hello world"

    def test_create_key_combo_factory(self) -> None:
        """Test creating a key combo action."""
        action = Action.key_combo(["ctrl", "c"])
        assert action.type == ActionType.KEY_COMBO
        assert action.parameters["keys"] == ["ctrl", "c"]

    def test_create_scroll_factory(self) -> None:
        """Test creating a scroll action."""
        action = Action.scroll("down", amount=3)
        assert action.type == ActionType.SCROLL
        assert action.parameters["direction"] == "down"
        assert action.parameters["amount"] == 3

    def test_create_wait_factory(self) -> None:
        """Test creating a wait action."""
        action = Action.wait(1000)
        assert action.type == ActionType.WAIT
        assert action.parameters["duration_ms"] == 1000

    def test_action_serialization(self) -> None:
        """Test action serialization."""
        action = Action.click(100, 200)
        json_str = action.model_dump_json()

        restored = Action.model_validate_json(json_str)
        assert restored.type == ActionType.CLICK
        assert restored.target is not None
        assert restored.target.x == 100


class TestObservationModel:
    """Tests for Observation model."""

    def test_create_valid_observation(self) -> None:
        """Test creating a valid observation."""
        obs = Observation(
            screenshot=b"fake image data",
            game_state=GameState(),
        )
        assert obs.is_valid
        assert obs.screenshot_size == (1920, 1080)

    def test_observation_with_error(self) -> None:
        """Test observation with processing error."""
        obs = Observation(
            screenshot=b"",
            game_state=GameState(),
            error="Failed to process",
        )
        assert not obs.is_valid

    def test_observation_invalid_dimensions(self) -> None:
        """Test that zero dimensions are rejected."""
        with pytest.raises(ValidationError):
            Observation(
                screenshot=b"data",
                game_state=GameState(),
                screenshot_width=0,
            )


class TestDecisionModel:
    """Tests for Decision model."""

    def test_create_valid_decision(self) -> None:
        """Test creating a valid decision."""
        action = Action.click(100, 200)
        decision = Decision(
            reasoning="The button is visible and should be clicked",
            action=action,
            confidence=0.9,
            expected_outcome="Button will be activated",
        )
        assert decision.is_high_confidence
        assert not decision.is_low_confidence

    def test_low_confidence_decision(self) -> None:
        """Test low confidence detection."""
        action = Action.wait(1000)
        decision = Decision(
            reasoning="Not sure what to do",
            action=action,
            confidence=0.3,
            expected_outcome="Nothing will happen",
        )
        assert decision.is_low_confidence
        assert not decision.is_high_confidence

    def test_decision_invalid_empty_reasoning(self) -> None:
        """Test that empty reasoning is rejected."""
        with pytest.raises(ValidationError):
            Decision(
                reasoning="",
                action=Action.wait(1000),
                confidence=0.5,
                expected_outcome="Nothing",
            )

    def test_decision_invalid_confidence_range(self) -> None:
        """Test that confidence outside 0-1 is rejected."""
        with pytest.raises(ValidationError):
            Decision(
                reasoning="test",
                action=Action.wait(1000),
                confidence=1.5,
                expected_outcome="Nothing",
            )

    def test_decision_serialization(self) -> None:
        """Test decision serialization round-trip."""
        action = Action.click(100, 200)
        decision = Decision(
            reasoning="Click the buy button",
            action=action,
            confidence=0.85,
            expected_outcome="Item will be purchased",
            alternatives=[{"action": "wait", "reason": "could wait for price drop"}],
        )

        json_str = decision.model_dump_json()
        restored = Decision.model_validate_json(json_str)

        assert restored.reasoning == "Click the buy button"
        assert restored.confidence == 0.85
        assert len(restored.alternatives) == 1


class TestModelImports:
    """Tests that models can be imported from the package."""

    def test_import_from_package(self) -> None:
        """Test importing models from src.models."""
        from src.models import (
            Action,
            ActionType,
            Decision,
            GameState,
            Observation,
            Resource,
            ScreenType,
            UIElement,
            Upgrade,
        )

        # Verify they are classes
        assert isinstance(Action, type)
        assert isinstance(GameState, type)
        assert isinstance(Decision, type)
        assert isinstance(Observation, type)

        # Verify enums
        assert ScreenType.MAIN.value == "main"
        assert ActionType.CLICK.value == "click"
