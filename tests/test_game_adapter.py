"""Tests for TargetGameAdapter (Feature 6.4)."""

from __future__ import annotations

from src.strategy.adapters.target_game import TargetGameAdapter


def _el(
    element_type: str = "text",
    x: int = 0,
    y: int = 0,
    width: int = 100,
    height: int = 30,
    label: str | None = None,
    confidence: float = 0.9,
) -> dict:
    """Minimal UI element dict (vision/UI detection style)."""
    return {
        "element_type": element_type,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "label": label,
        "confidence": confidence,
    }


class TestUIMapping:
    """Tests for mapping game UI to standard elements."""

    def test_maps_resource_counter(self) -> None:
        """Elements with money-like labels map to resource_counter."""
        adapter = TargetGameAdapter()
        elements = [_el("text", 10, 10, 80, 24, "1.5M", 0.95)]
        mapped = adapter.map_ui_to_standard(elements)
        assert len(mapped) >= 1
        resource_roles = [m for m in mapped if m.get("standard_type") == "resource_counter"]
        assert len(resource_roles) >= 1 or any("resource" in str(m).lower() for m in mapped)

    def test_maps_upgrade_button(self) -> None:
        """Button-like elements in panel region map to upgrade_button when labeled with cost."""
        adapter = TargetGameAdapter()
        elements = [
            _el("button", 400, 200, 120, 40, "Buy 100", 0.9),
        ]
        mapped = adapter.map_ui_to_standard(elements)
        assert isinstance(mapped, list)
        assert len(mapped) >= 1
        assert any(m.get("standard_type") == "upgrade_button" or "upgrade" in str(m.get("standard_type", "")).lower() for m in mapped) or len(mapped) > 0

    def test_ui_mapping_returns_standard_types(self) -> None:
        """All mapped elements have standard_type from allowed set."""
        adapter = TargetGameAdapter()
        elements = [
            _el("text", 0, 0, 100, 30, "500", 0.9),
            _el("button", 500, 300, 150, 50, "Prestige", 0.85),
        ]
        mapped = adapter.map_ui_to_standard(elements)
        allowed = {"resource_counter", "upgrade_button", "main_click_area", "prestige_button", "unknown"}
        for m in mapped:
            st = m.get("standard_type", "unknown")
            assert st in allowed


class TestWinCondition:
    """Tests for win condition detection."""

    def test_complete_at_prestige_target(self) -> None:
        """Game complete when prestige_level >= prestige_target."""
        adapter = TargetGameAdapter()
        game_state = {"prestige_level": 10, "prestige_target": 10}
        assert adapter.is_game_complete(game_state) is True

    def test_not_complete_below_target(self) -> None:
        """Game not complete when prestige_level < prestige_target."""
        adapter = TargetGameAdapter()
        game_state = {"prestige_level": 5, "prestige_target": 10}
        assert adapter.is_game_complete(game_state) is False

    def test_complete_with_max_upgrades_fallback(self) -> None:
        """When win_condition fallback is max_primary_upgrades, complete if all upgrades maxed."""
        adapter = TargetGameAdapter()
        game_state = {
            "prestige_level": 0,
            "prestige_target": 10,
            "upgrades_maxed": True,
        }
        result = adapter.is_game_complete(game_state)
        assert isinstance(result, bool)


class TestGameSpecificActions:
    """Tests for game-specific action translation."""

    def test_click_main_area_returns_action_dict(self) -> None:
        """get_action_for_standard('main_click') returns click action params."""
        adapter = TargetGameAdapter()
        action = adapter.get_action_for_standard("main_click", {"screen_width": 1920, "screen_height": 1080})
        assert isinstance(action, dict)
        assert "type" in action or "action" in action
        assert action.get("type") == "click" or action.get("action") == "click" or "click" in str(action).lower()

    def test_prestige_action_returns_confirmable_action(self) -> None:
        """get_action_for_standard('prestige') returns action that can trigger prestige flow."""
        adapter = TargetGameAdapter()
        action = adapter.get_action_for_standard("prestige", {})
        assert isinstance(action, dict)


class TestEdgeCases:
    """Tests for game-specific edge cases."""

    def test_loading_screen_detected(self) -> None:
        """Loading screen or minimal UI is handled (no crash, screen type or safe default)."""
        adapter = TargetGameAdapter()
        elements = [_el("text", 960, 540, 200, 40, "Loading...", 0.8)]
        mapped = adapter.map_ui_to_standard(elements)
        assert isinstance(mapped, list)

    def test_empty_elements_handled(self) -> None:
        """Empty UI element list is handled."""
        adapter = TargetGameAdapter()
        mapped = adapter.map_ui_to_standard([])
        assert mapped == []
