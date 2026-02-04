"""Game adapter for Money Clicker Incremental (target game from 6.0).

Maps game UI to standard elements, provides game-specific actions,
and detects win condition. Uses configs/game_config.yaml when available.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

# Standard element types for mapping
STANDARD_TYPES = frozenset({
    "resource_counter",
    "upgrade_button",
    "main_click_area",
    "prestige_button",
    "unknown",
})

# Default config if file not loaded
DEFAULT_PRESTIGE_TARGET = 10


class TargetGameAdapter:
    """Adapter for Money Clicker Incremental: UI mapping, actions, win condition."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Load game config from config_path or repo configs/game_config.yaml."""
        self._config: dict[str, Any] = {}
        if config_path is not None:
            self._load_config(Path(config_path))
        else:
            repo = Path(__file__).resolve().parent.parent.parent.parent
            default = repo / "configs" / "game_config.yaml"
            if default.exists():
                self._load_config(default)

    def _load_config(self, path: Path) -> None:
        if yaml is None:
            return
        try:
            with open(path) as f:
                self._config = yaml.safe_load(f) or {}
        except Exception:
            self._config = {}

    def _prestige_target(self, game_state: dict[str, Any]) -> int:
        return int(
            game_state.get("prestige_target")
            or self._config.get("win_condition", {}).get("prestige_target", DEFAULT_PRESTIGE_TARGET)
        )

    def map_ui_to_standard(self, ui_elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Map raw UI elements to standard element types (resource_counter, upgrade_button, etc.)."""
        result: list[dict[str, Any]] = []
        primary = (self._config.get("resources") or {}).get("primary", "money")
        for el in ui_elements:
            label = (el.get("label") or "").strip()
            el_type = (el.get("element_type") or "unknown").lower()
            x = int(el.get("x", 0))
            y = int(el.get("y", 0))
            w = int(el.get("width", 1))
            h = int(el.get("height", 1))
            center_x = x + w // 2
            center_y = y + h // 2

            standard_type = "unknown"
            if self._looks_like_resource(label, primary):
                standard_type = "resource_counter"
            elif "prestige" in label.lower() or "rebirth" in label.lower():
                standard_type = "prestige_button"
            elif el_type == "button" and (re.search(r"\d+", label) or "buy" in label.lower() or "upgrade" in label.lower()):
                standard_type = "upgrade_button"
            elif self._in_center_region(center_x, center_y):
                standard_type = "main_click_area"

            result.append({
                **el,
                "standard_type": standard_type,
                "center": (center_x, center_y),
            })
        return result

    def _looks_like_resource(self, label: str, primary: str) -> bool:
        if not label:
            return False
        return (
            primary.lower() in label.lower()
            or bool(re.match(r"^[\d.,]+[KMBT]?$", label.replace(" ", ""), re.IGNORECASE))
        )

    def _in_center_region(self, cx: int, cy: int, width: int = 1920, height: int = 1080) -> bool:
        margin = 0.2
        return (
            width * margin < cx < width * (1 - margin)
            and height * margin < cy < height * (1 - margin)
        )

    def is_game_complete(self, game_state: dict[str, Any]) -> bool:
        """True when prestige_level >= prestige_target or upgrades_maxed (fallback)."""
        prestige_level = int(game_state.get("prestige_level", 0))
        target = self._prestige_target(game_state)
        if prestige_level >= target:
            return True
        return game_state.get("upgrades_maxed") is True

    def get_action_for_standard(
        self,
        standard_action: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Return action params for executor (type, target, etc.) for a standard action."""
        screen_width = int(context.get("screen_width", 1920))
        screen_height = int(context.get("screen_height", 1080))
        center_x = screen_width // 2
        center_y = screen_height // 2

        if standard_action == "main_click":
            return {"type": "click", "target": [center_x, center_y], "description": "Click main area"}
        if standard_action == "prestige":
            return {"type": "click", "target": "prestige_button", "description": "Prestige/Rebirth"}
        if standard_action == "upgrade":
            return {"type": "click", "target": "upgrade_button", "description": "Buy upgrade"}
        return {"type": "click", "target": [center_x, center_y], "description": standard_action}
