"""Tests for game configuration and game fixtures (Feature 6.0)."""

from __future__ import annotations

from pathlib import Path

import yaml

# Paths relative to repo root (tests/ is one level down)
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
FIXTURES_GAME_DIR = REPO_ROOT / "tests" / "fixtures" / "game"


def test_game_config_file_exists() -> None:
    """Game config YAML file exists."""
    path = CONFIGS_DIR / "game_config.yaml"
    assert path.exists(), f"Expected {path} to exist"


def test_game_config_loads_without_errors() -> None:
    """Game config loads without errors and has required fields."""
    path = CONFIGS_DIR / "game_config.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)

    assert data is not None, "Config should not be empty"

    # Required top-level keys per PROJECT_PLAN and configs/game_config.yaml
    required = ["game", "resources", "ui_regions", "win_condition", "fixtures"]
    for key in required:
        assert key in data, f"Missing required key: {key}"

    # Game identity
    assert "url" in data["game"]
    assert "game_id" in data["game"] or "name" in data["game"]

    # Fixtures section
    assert "required_screenshots" in data["fixtures"]
    assert isinstance(data["fixtures"]["required_screenshots"], list)
    assert len(data["fixtures"]["required_screenshots"]) >= 1


def test_fixtures_directory_contains_required_screenshots() -> None:
    """Fixtures directory contains required screenshot files."""
    assert FIXTURES_GAME_DIR.exists(), f"Expected {FIXTURES_GAME_DIR} to exist"

    # Required screenshots per PROJECT_PLAN and game_config.yaml
    required = ["main_screen.png", "upgrade_menu.png", "prestige_dialog.png"]
    for name in required:
        path = FIXTURES_GAME_DIR / name
        assert path.exists(), f"Required screenshot missing: {name}"
        assert path.stat().st_size > 0, f"Required screenshot empty: {name}"


def test_game_config_required_screenshots_match_fixtures() -> None:
    """Required screenshots in config match files present in fixtures dir."""
    path = CONFIGS_DIR / "game_config.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)

    required = data["fixtures"]["required_screenshots"]
    for name in required:
        fixture_path = FIXTURES_GAME_DIR / name
        assert fixture_path.exists(), (
            f"Config requires {name} but file not found in {FIXTURES_GAME_DIR}"
        )
