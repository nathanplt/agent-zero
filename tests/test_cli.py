"""Tests for CLI runtime orchestration."""

from __future__ import annotations

from argparse import Namespace
from datetime import datetime
from unittest.mock import MagicMock

from src.cli import (
    AgentRuntime,
    _resolve_llm_runtime,
    _start_environment,
    build_arg_parser,
    build_credentials_from_args,
    main,
    run_command,
)
from src.core.observation import Observation
from src.interfaces.actions import ActionResult, ActionType
from src.interfaces.environment import EnvironmentSetupError
from src.interfaces.vision import Screenshot
from src.models.actions import Action
from src.models.decisions import Decision
from src.models.game_state import GameState, ScreenType


class TestCLIParser:
    """Argument parsing behavior."""

    def test_parses_run_command_options(self) -> None:
        parser = build_arg_parser()

        args = parser.parse_args([
            "run",
            "--max-iterations",
            "5",
            "--game-url",
            "https://example.com/game",
            "--headless",
        ])

        assert args.command == "run"
        assert args.max_iterations == 5
        assert args.game_url == "https://example.com/game"
        assert args.headless is True

    def test_parses_observer_and_credentials_options(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "run",
                "--observe",
                "--observer-host",
                "127.0.0.1",
                "--observer-port",
                "9999",
                "--username",
                "alice",
                "--password",
                "secret",
                "--totp-secret",
                "ABCD1234",
            ]
        )

        assert args.observe is True
        assert args.observer_host == "127.0.0.1"
        assert args.observer_port == 9999
        assert args.username == "alice"
        assert args.password == "secret"
        assert args.totp_secret == "ABCD1234"


class TestCLIRunCommand:
    """Run command should wire and drive the runtime."""

    def test_run_command_executes_requested_iterations(self, monkeypatch) -> None:
        fake_runtime = MagicMock()
        fake_runtime.run_iterations.return_value = 3
        monkeypatch.setattr("src.cli._create_runtime", lambda _args: fake_runtime)

        args = Namespace(
            command="run",
            max_iterations=3,
            game_url="https://example.com/game",
            headless=True,
            skip_auth=True,
        )

        exit_code = run_command(args)

        assert exit_code == 0
        fake_runtime.run_iterations.assert_called_once_with(3)
        fake_runtime.shutdown.assert_called_once()

    def test_main_returns_nonzero_on_runtime_error(self, monkeypatch) -> None:
        def _raise(_: Namespace) -> int:
            raise RuntimeError("boom")

        monkeypatch.setattr("src.cli.run_command", _raise)

        code = main(["run"])

        assert code == 1

    def test_main_loads_environment_secrets_before_run(self, monkeypatch) -> None:
        calls: list[str] = []

        def _load() -> None:
            calls.append("secrets")

        def _run(_: Namespace) -> int:
            assert calls == ["secrets"]
            return 0

        monkeypatch.setattr("src.cli.load_environment_secrets", _load)
        monkeypatch.setattr("src.cli.run_command", _run)

        code = main(["run"])

        assert code == 0
        assert calls == ["secrets"]


class TestCLICredentials:
    """Credential helper behavior."""

    def test_build_credentials_from_args_returns_none_without_user_pass(self) -> None:
        args = Namespace(username=None, password=None, totp_secret=None)
        creds = build_credentials_from_args(args)
        assert creds is None

    def test_build_credentials_from_args_uses_explicit_values(self) -> None:
        args = Namespace(username="alice", password="secret", totp_secret="AAAA")
        creds = build_credentials_from_args(args)
        assert creds is not None
        assert creds.username == "alice"
        assert creds.password == "secret"
        assert creds.totp_secret == "AAAA"


class TestAgentRuntimeStreaming:
    """Runtime should publish frames and action events for observation."""

    def test_run_iterations_pushes_frames_and_events(self) -> None:
        screenshot = MagicMock(spec=Screenshot)
        screenshot.width = 1920
        screenshot.height = 1080
        screenshot.timestamp = datetime.now()
        screenshot.image = MagicMock()
        screenshot.raw_bytes = b"frame"

        observation = Observation(
            screenshot=screenshot,
            game_state=GameState(current_screen=ScreenType.MAIN),
            ui_elements=[],
            text_regions=[],
            timestamp=datetime.now(),
            llm_used=False,
            confidence=0.9,
            duration_ms=10.0,
        )
        decision = Decision(
            reasoning="click main",
            action=Action(type=ActionType.CLICK, description="click"),
            confidence=0.8,
            expected_outcome="resource increases",
            context={"decision_source": "policy"},
        )
        action_result = ActionResult(
            success=True,
            action=MagicMock(),
            error=None,
            duration_ms=5.0,
        )

        loop = MagicMock()
        loop.run_once.return_value = (observation, decision, action_result)
        screen_stream = MagicMock()
        action_stream = MagicMock()

        runtime = AgentRuntime(
            environment=MagicMock(),
            loop=loop,
            pipeline=MagicMock(),
            screen_streaming_service=screen_stream,
            action_streaming_service=action_stream,
            observer_server=None,
        )
        completed = runtime.run_iterations(1)

        assert completed == 1
        screen_stream.push_frame.assert_called_once()
        action_stream.push_event.assert_called_once()


class TestCLIEnvironmentStartup:
    """Environment startup fallback behavior."""

    def test_retries_headless_when_display_start_fails(self, monkeypatch) -> None:
        created: list[object] = []

        class FakeEnv:
            def __init__(
                self,
                *,
                headless: bool,
                viewport_width: int,
                viewport_height: int,
                display: str,
            ):
                _ = (viewport_width, viewport_height, display)
                self.headless = headless
                created.append(self)

            def start(self) -> None:
                if not self.headless:
                    raise EnvironmentSetupError("Failed to start display: Xvfb failed")

        monkeypatch.setattr("src.cli.LocalEnvironmentManager", FakeEnv)

        args = Namespace(headless=False)
        cfg = Namespace(
            environment=Namespace(display_width=1920, display_height=1080, virtual_display=":99")
        )

        env = _start_environment(args, cfg)

        assert env.headless is True
        assert len(created) == 2
        assert created[0].headless is False
        assert created[1].headless is True

    def test_does_not_retry_on_non_display_error(self, monkeypatch) -> None:
        class FakeEnv:
            def __init__(
                self,
                *,
                headless: bool,
                viewport_width: int,
                viewport_height: int,
                display: str,
            ):
                _ = (viewport_width, viewport_height, display)
                self.headless = headless

            def start(self) -> None:
                raise EnvironmentSetupError("Failed to start browser: boom")

        monkeypatch.setattr("src.cli.LocalEnvironmentManager", FakeEnv)

        args = Namespace(headless=False)
        cfg = Namespace(
            environment=Namespace(display_width=1920, display_height=1080, virtual_display=":99")
        )

        try:
            _start_environment(args, cfg)
            raise AssertionError("Expected EnvironmentSetupError")
        except EnvironmentSetupError as exc:
            assert "Failed to start browser" in str(exc)


class TestCLILLMRuntimeResolution:
    """Provider/model/key resolution behavior."""

    def test_uses_configured_provider_when_key_available(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        resolved = _resolve_llm_runtime(
            configured_provider="anthropic",
            configured_model="claude-3-sonnet-20240229",
        )

        assert resolved.provider == "anthropic"
        assert resolved.model == "claude-3-sonnet-20240229"
        assert resolved.api_key == "anthropic-key"

    def test_falls_back_to_openai_when_configured_provider_key_missing(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

        resolved = _resolve_llm_runtime(
            configured_provider="anthropic",
            configured_model="claude-3-sonnet-20240229",
        )

        assert resolved.provider == "openai"
        assert resolved.api_key == "openai-key"
        assert "gpt" in str(resolved.model).lower()

    def test_corrects_mismatched_model_for_openai_provider(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

        resolved = _resolve_llm_runtime(
            configured_provider="openai",
            configured_model="claude-3-sonnet-20240229",
        )

        assert resolved.provider == "openai"
        assert resolved.api_key == "openai-key"
        assert "gpt" in str(resolved.model).lower()

    def test_returns_none_api_key_when_no_provider_keys_available(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        resolved = _resolve_llm_runtime(
            configured_provider="anthropic",
            configured_model="claude-3-sonnet-20240229",
        )

        assert resolved.provider == "anthropic"
        assert resolved.api_key is None
