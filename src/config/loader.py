"""Configuration loader for Agent Zero.

Loads configuration from YAML files and environment variables.
Environment variables override YAML values using the AGENTZERO_ prefix.
Nested keys use double underscores: AGENTZERO_AGENT__LOOP_RATE=5
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Agent core settings."""

    loop_rate: int = Field(default=10, ge=1, le=100, description="Iterations per second")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    timeout: int = Field(default=30, ge=1, le=300, description="Timeout in seconds")


class EnvironmentConfig(BaseModel):
    """Environment/display settings."""

    display_width: int = Field(default=1920, ge=640, le=3840)
    display_height: int = Field(default=1080, ge=480, le=2160)
    virtual_display: str = Field(default=":99")


class VisionConfig(BaseModel):
    """Vision system settings."""

    capture_fps: int = Field(default=10, ge=1, le=60)
    buffer_size: int = Field(default=10, ge=1, le=100)
    ocr_enabled: bool = Field(default=True)


class LLMConfig(BaseModel):
    """LLM provider settings."""

    provider: str = Field(default="anthropic", pattern="^(anthropic|openai)$")
    model: str = Field(default="claude-3-sonnet-20240229")
    max_tokens: int = Field(default=4096, ge=1, le=100000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class MemoryConfig(BaseModel):
    """Memory/persistence settings."""

    database_path: str = Field(default="data/memory.db")
    max_episodes: int = Field(default=10000, ge=100, le=1000000)


class ObserverConfig(BaseModel):
    """Observer/web UI settings."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080, ge=1, le=65535)
    stream_fps: int = Field(default=10, ge=1, le=60)


class LoggingConfig(BaseModel):
    """Logging settings."""

    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file: str = Field(default="logs/agent.log")


class Config(BaseModel):
    """Root configuration model."""

    agent: AgentConfig = Field(default_factory=AgentConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    observer: ObserverConfig = Field(default_factory=ObserverConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _get_env_value(key: str) -> str | None:
    """Get environment variable with AGENTZERO_ prefix."""
    env_key = f"AGENTZERO_{key.upper()}"
    return os.environ.get(env_key)


def _apply_env_overrides(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Apply environment variable overrides to config data.

    Environment variables use AGENTZERO_ prefix with double underscores for nesting.
    Example: AGENTZERO_AGENT__LOOP_RATE=5 sets agent.loop_rate to 5
    """
    result = data.copy()

    for key, value in result.items():
        env_key = f"{prefix}__{key}" if prefix else key

        if isinstance(value, dict):
            result[key] = _apply_env_overrides(value, env_key)
        else:
            env_value = _get_env_value(env_key)
            if env_value is not None:
                # Convert to appropriate type based on original value
                if isinstance(value, bool):
                    result[key] = env_value.lower() in ("true", "1", "yes")
                elif isinstance(value, int):
                    result[key] = int(env_value)
                elif isinstance(value, float):
                    result[key] = float(env_value)
                else:
                    result[key] = env_value

    return result


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from YAML file with environment variable overrides.

    Args:
        config_path: Path to YAML config file. If None, uses default.yaml.

    Returns:
        Validated Config object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValidationError: If config values are invalid.
    """
    # Determine config file path
    if config_path is None:
        # Look for default config relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)

    # Load YAML file
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    # Apply environment variable overrides
    data = _apply_env_overrides(data)

    # Create and validate config
    return Config.model_validate(data)


def get_default_config() -> Config:
    """Get default configuration without loading from file."""
    return Config()
