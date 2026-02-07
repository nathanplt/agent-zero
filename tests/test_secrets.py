"""Tests for secure secret loading from .env files."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config.secrets import load_environment_secrets


class TestSecretsLoader:
    """Secret loading behavior for .env files."""

    def test_loads_openai_api_key_from_secure_env_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=from-dotenv\n")
        os.chmod(env_file, 0o600)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        loaded = load_environment_secrets(env_file=env_file)

        assert loaded == env_file
        assert os.environ.get("OPENAI_API_KEY") == "from-dotenv"

    def test_does_not_override_existing_environment_value(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=from-dotenv\n")
        os.chmod(env_file, 0o600)
        monkeypatch.setenv("OPENAI_API_KEY", "already-set")

        load_environment_secrets(env_file=env_file)

        assert os.environ.get("OPENAI_API_KEY") == "already-set"

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits differ on Windows")
    def test_rejects_group_or_world_readable_env_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=from-dotenv\n")
        os.chmod(env_file, 0o644)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(PermissionError):
            load_environment_secrets(env_file=env_file)

        assert os.environ.get("OPENAI_API_KEY") is None

    def test_missing_env_file_returns_none(self, tmp_path: Path) -> None:
        missing = tmp_path / ".env"

        loaded = load_environment_secrets(env_file=missing, strict=False)

        assert loaded is None

    def test_directory_env_path_has_actionable_error(self, tmp_path: Path) -> None:
        env_dir = tmp_path / ".env"
        env_dir.mkdir()

        with pytest.raises(ValueError, match="is a directory"):
            load_environment_secrets(env_file=env_dir)
