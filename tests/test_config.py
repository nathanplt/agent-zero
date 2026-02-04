"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import load_config


class TestConfigLoader:
    """Tests for load_config function."""

    def test_load_default_config(self) -> None:
        """Test loading the default configuration file."""
        config = load_config()

        # Verify structure exists
        assert config.agent is not None
        assert config.environment is not None
        assert config.vision is not None
        assert config.llm is not None
        assert config.memory is not None
        assert config.observer is not None
        assert config.logging is not None

    def test_load_default_values(self) -> None:
        """Test that default values are correctly loaded."""
        config = load_config()

        # Check some default values from default.yaml
        assert config.agent.loop_rate == 10
        assert config.environment.display_width == 1920
        assert config.environment.display_height == 1080
        assert config.vision.capture_fps == 10
        assert config.llm.provider == "anthropic"

    def test_load_custom_config_file(self, tmp_path: Path) -> None:
        """Test loading from a custom config file path."""
        custom_config = {
            "agent": {"loop_rate": 5, "max_retries": 5},
            "vision": {"capture_fps": 30},
        }

        config_file = tmp_path / "custom.yaml"
        with open(config_file, "w") as f:
            yaml.dump(custom_config, f)

        config = load_config(config_file)

        assert config.agent.loop_rate == 5
        assert config.agent.max_retries == 5
        assert config.vision.capture_fps == 30
        # Default values still apply for unspecified fields
        assert config.agent.timeout == 30

    def test_missing_config_file_raises_error(self) -> None:
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_empty_config_file_uses_defaults(self, tmp_path: Path) -> None:
        """Test that empty config file uses all defaults."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        config = load_config(config_file)

        # Should use all default values
        assert config.agent.loop_rate == 10
        assert config.environment.display_width == 1920


class TestEnvironmentOverrides:
    """Tests for environment variable overrides."""

    def test_env_override_simple_value(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variables override YAML values."""
        config_data = {"agent": {"loop_rate": 10}}
        config_file = tmp_path / "test.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Set environment variable
        monkeypatch.setenv("AGENTZERO_AGENT__LOOP_RATE", "20")

        config = load_config(config_file)

        assert config.agent.loop_rate == 20

    def test_env_override_nested_value(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment override for nested config values."""
        config_data = {"llm": {"provider": "openai", "max_tokens": 1000}}
        config_file = tmp_path / "test.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        monkeypatch.setenv("AGENTZERO_LLM__MAX_TOKENS", "8000")

        config = load_config(config_file)

        assert config.llm.max_tokens == 8000
        assert config.llm.provider == "openai"  # unchanged

    def test_env_override_boolean_value(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment override for boolean values."""
        config_data = {"vision": {"ocr_enabled": True}}
        config_file = tmp_path / "test.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        monkeypatch.setenv("AGENTZERO_VISION__OCR_ENABLED", "false")

        config = load_config(config_file)

        assert config.vision.ocr_enabled is False

    def test_env_override_float_value(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment override for float values."""
        config_data = {"llm": {"temperature": 0.7}}
        config_file = tmp_path / "test.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        monkeypatch.setenv("AGENTZERO_LLM__TEMPERATURE", "0.3")

        config = load_config(config_file)

        assert config.llm.temperature == 0.3


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_invalid_loop_rate_too_high(self, tmp_path: Path) -> None:
        """Test that loop_rate above max is rejected."""
        config_data = {"agent": {"loop_rate": 500}}  # max is 100
        config_file = tmp_path / "test.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValidationError):
            load_config(config_file)

    def test_invalid_loop_rate_too_low(self, tmp_path: Path) -> None:
        """Test that loop_rate below min is rejected."""
        config_data = {"agent": {"loop_rate": 0}}  # min is 1
        config_file = tmp_path / "test.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValidationError):
            load_config(config_file)

    def test_invalid_provider(self, tmp_path: Path) -> None:
        """Test that invalid LLM provider is rejected."""
        config_data = {"llm": {"provider": "invalid_provider"}}
        config_file = tmp_path / "test.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValidationError):
            load_config(config_file)

    def test_invalid_log_level(self, tmp_path: Path) -> None:
        """Test that invalid log level is rejected."""
        config_data = {"logging": {"level": "INVALID"}}
        config_file = tmp_path / "test.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValidationError):
            load_config(config_file)


class TestConfigSerialization:
    """Tests for config serialization."""

    def test_config_to_dict(self) -> None:
        """Test that config can be serialized to dict."""
        config = load_config()
        data = config.model_dump()

        assert isinstance(data, dict)
        assert "agent" in data
        assert "loop_rate" in data["agent"]

    def test_config_to_json(self) -> None:
        """Test that config can be serialized to JSON."""
        config = load_config()
        json_str = config.model_dump_json()

        assert isinstance(json_str, str)
        assert "loop_rate" in json_str
