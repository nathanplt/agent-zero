"""Tests for CLI runtime orchestration."""

from __future__ import annotations

import logging
import time
from argparse import Namespace
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.cli import (
    AgentRuntime,
    AuthChallengeMode,
    AuthMode,
    RuntimeTarget,
    _authenticate_session,
    _configure_logging,
    _create_runtime,
    _determine_startup_headless,
    _enforce_auth_mode,
    _resolve_llm_runtime,
    _resolve_runtime_target,
    _run_with_heartbeat,
    _start_environment,
    auth_command,
    build_arg_parser,
    build_credentials_from_args,
    main,
    run_command,
)
from src.core.observation import Observation
from src.environment.auth import AuthOutcome, AuthResult
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

    def test_parses_launch_stability_options(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "run",
                "--runtime-target",
                "native",
                "--auth-mode",
                "best-effort",
                "--recovery-profile",
                "aggressive",
                "--log-format",
                "json",
                "--progress-heartbeat-seconds",
                "7",
                "--auth-challenge-mode",
                "fail",
                "--auth-manual-timeout-seconds",
                "45",
            ]
        )
        assert args.runtime_target == "native"
        assert args.auth_mode == "best-effort"
        assert args.recovery_profile == "aggressive"
        assert args.log_format == "json"
        assert args.progress_heartbeat_seconds == 7.0
        assert args.auth_challenge_mode == "fail"
        assert args.auth_manual_timeout_seconds == 45.0

    def test_launch_stability_option_defaults(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["run"])

        assert args.runtime_target == "auto"
        assert args.auth_mode == "strict"
        assert args.recovery_profile == "balanced"
        assert args.log_format == "readable"
        assert args.progress_heartbeat_seconds == 5.0
        assert args.auth_challenge_mode == "manual"
        assert args.auth_manual_timeout_seconds == 180.0

    def test_parses_auth_checkpoint_command_options(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "auth",
                "--observe",
                "--observer-host",
                "127.0.0.1",
                "--observer-port",
                "9091",
                "--username",
                "alice",
                "--password",
                "secret",
            ]
        )

        assert args.command == "auth"
        assert args.observe is True
        assert args.observer_host == "127.0.0.1"
        assert args.observer_port == 9091
        assert args.username == "alice"
        assert args.password == "secret"


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

    def test_main_loads_environment_secrets_before_auth(self, monkeypatch) -> None:
        calls: list[str] = []

        def _load() -> None:
            calls.append("secrets")

        def _auth(_: Namespace) -> int:
            assert calls == ["secrets"]
            return 0

        monkeypatch.setattr("src.cli.load_environment_secrets", _load)
        monkeypatch.setattr("src.cli.auth_command", _auth)

        code = main(["auth"])

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

    def test_run_iterations_uses_recovery_when_iteration_raises(self) -> None:
        loop = MagicMock()
        loop.run_once.side_effect = RuntimeError("capture failed")
        recovery = MagicMock()
        recovery.handle.side_effect = [True, False]

        runtime = AgentRuntime(
            environment=MagicMock(),
            loop=loop,
            pipeline=MagicMock(),
            recovery_coordinator=recovery,
            screen_streaming_service=None,
            action_streaming_service=None,
            observer_server=None,
        )

        try:
            runtime.run_iterations(1)
            raise AssertionError("Expected runtime to raise after recovery budget exhaustion")
        except RuntimeError:
            pass

        assert recovery.handle.call_count == 2


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
                storage_state_path=None,
                use_virtual_display: bool = True,
            ):
                _ = (
                    viewport_width,
                    viewport_height,
                    display,
                    storage_state_path,
                    use_virtual_display,
                )
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
                storage_state_path=None,
                use_virtual_display: bool = True,
            ):
                _ = (
                    viewport_width,
                    viewport_height,
                    display,
                    storage_state_path,
                    use_virtual_display,
                )
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

    def test_uses_native_display_on_macos_when_headed(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        class FakeEnv:
            def __init__(
                self,
                *,
                headless: bool,
                viewport_width: int,
                viewport_height: int,
                display: str,
                storage_state_path=None,
                use_virtual_display: bool = True,
            ):
                _ = (viewport_width, viewport_height, display, storage_state_path)
                captured["headless"] = headless
                captured["use_virtual_display"] = use_virtual_display

            def start(self) -> None:
                return None

        monkeypatch.setattr("src.cli.LocalEnvironmentManager", FakeEnv)
        monkeypatch.setattr("src.cli.sys.platform", "darwin")

        args = Namespace(headless=False)
        cfg = Namespace(
            environment=Namespace(display_width=1920, display_height=1080, virtual_display=":99")
        )

        _start_environment(args, cfg)

        assert captured["headless"] is False
        assert captured["use_virtual_display"] is False


class TestCLIObserverStartupOrder:
    """Observer server should be available during auth and startup failures."""

    def test_create_runtime_starts_observer_before_auth(self, monkeypatch) -> None:
        calls: list[str] = []

        cfg = Namespace(
            logging=Namespace(level="INFO"),
            llm=Namespace(provider="openai", model="gpt-4o-mini"),
            environment=Namespace(display_width=1920, display_height=1080, virtual_display=":99"),
            observer=Namespace(host="127.0.0.1", port=9090, stream_fps=10),
            vision=Namespace(buffer_size=4),
            agent=Namespace(loop_rate=1),
        )
        monkeypatch.setattr("src.cli.load_config", lambda: cfg)
        monkeypatch.setattr(
            "src.cli._resolve_llm_runtime",
            lambda **_kwargs: Namespace(provider="openai", model="gpt-4o-mini", api_key=None),
        )

        env = MagicMock()
        env.get_browser_runtime.return_value = MagicMock(page=MagicMock())
        monkeypatch.setattr("src.cli._start_environment", lambda *_args: env)

        def _start_observer_server(*_args, **_kwargs):
            calls.append("observer")
            return MagicMock()

        monkeypatch.setattr("src.cli._start_observer_server", _start_observer_server)
        monkeypatch.setattr(
            "src.cli._authenticate_session",
            lambda **_kwargs: (calls.append("auth") or AuthResult(AuthOutcome.SUCCESS, "ok", False, 1.0)),
        )
        monkeypatch.setattr("src.cli.ScreenshotCapture", lambda **_kwargs: MagicMock())
        monkeypatch.setattr("src.cli.OCRSystem", lambda: MagicMock())
        monkeypatch.setattr("src.cli.UIDetector", lambda: MagicMock())
        monkeypatch.setattr("src.cli.ObservationPipeline", lambda **_kwargs: MagicMock())
        monkeypatch.setattr("src.cli.DecisionEngine", lambda **_kwargs: MagicMock())
        monkeypatch.setattr("src.cli.GameActionExecutor", lambda **_kwargs: MagicMock())
        monkeypatch.setattr("src.cli.AgentLoop", lambda **_kwargs: MagicMock())

        args = Namespace(
            observe=True,
            observer_host="127.0.0.1",
            observer_port=9090,
            game_url=None,
            skip_auth=False,
            username="user",
            password="pass",
            totp_secret=None,
            auth_mode="strict",
            recovery_profile="balanced",
            log_format="readable",
            progress_heartbeat_seconds=5.0,
            headless=True,
            runtime_target="auto",
        )

        runtime = _create_runtime(args)
        runtime.shutdown()

        assert calls[:2] == ["observer", "auth"]

    def test_create_runtime_stops_observer_when_strict_auth_fails(self, monkeypatch) -> None:
        cfg = Namespace(
            logging=Namespace(level="INFO"),
            llm=Namespace(provider="openai", model="gpt-4o-mini"),
            environment=Namespace(display_width=1920, display_height=1080, virtual_display=":99"),
            observer=Namespace(host="127.0.0.1", port=9090, stream_fps=10),
            vision=Namespace(buffer_size=4),
            agent=Namespace(loop_rate=1),
        )
        monkeypatch.setattr("src.cli.load_config", lambda: cfg)
        monkeypatch.setattr(
            "src.cli._resolve_llm_runtime",
            lambda **_kwargs: Namespace(provider="openai", model="gpt-4o-mini", api_key=None),
        )

        env = MagicMock()
        env.get_browser_runtime.return_value = MagicMock(page=MagicMock())
        monkeypatch.setattr("src.cli._start_environment", lambda *_args: env)
        server = MagicMock()
        monkeypatch.setattr("src.cli._start_observer_server", lambda *_args, **_kwargs: server)
        monkeypatch.setattr(
            "src.cli._authenticate_session",
            lambda **_kwargs: AuthResult(AuthOutcome.NETWORK_TIMEOUT, "timeout", True, 30000.0),
        )

        args = Namespace(
            observe=True,
            observer_host="127.0.0.1",
            observer_port=9090,
            game_url=None,
            skip_auth=False,
            username="user",
            password="pass",
            totp_secret=None,
            auth_mode="strict",
            recovery_profile="balanced",
            log_format="readable",
            progress_heartbeat_seconds=5.0,
            headless=True,
            runtime_target="auto",
        )

        with pytest.raises(RuntimeError):
            _create_runtime(args)

        server.stop.assert_called_once()

    def test_create_runtime_reenters_game_after_successful_auth(self, monkeypatch) -> None:
        cfg = Namespace(
            logging=Namespace(level="INFO"),
            llm=Namespace(provider="openai", model="gpt-4o-mini"),
            environment=Namespace(display_width=1920, display_height=1080, virtual_display=":99"),
            observer=Namespace(host="127.0.0.1", port=9090, stream_fps=10),
            vision=Namespace(buffer_size=4),
            agent=Namespace(loop_rate=1),
        )
        monkeypatch.setattr("src.cli.load_config", lambda: cfg)
        monkeypatch.setattr(
            "src.cli._resolve_llm_runtime",
            lambda **_kwargs: Namespace(provider="openai", model="gpt-4o-mini", api_key=None),
        )

        env = MagicMock()
        env.get_browser_runtime.return_value = MagicMock(page=MagicMock())
        monkeypatch.setattr("src.cli._start_environment", lambda *_args: env)
        monkeypatch.setattr(
            "src.cli._authenticate_session",
            lambda **_kwargs: AuthResult(AuthOutcome.SUCCESS, "ok", False, 10.0),
        )
        monkeypatch.setattr("src.cli.ScreenshotCapture", lambda **_kwargs: MagicMock())
        monkeypatch.setattr("src.cli.OCRSystem", lambda: MagicMock())
        monkeypatch.setattr("src.cli.UIDetector", lambda: MagicMock())
        monkeypatch.setattr("src.cli.ObservationPipeline", lambda **_kwargs: MagicMock())
        monkeypatch.setattr("src.cli.DecisionEngine", lambda **_kwargs: MagicMock())
        monkeypatch.setattr("src.cli.GameActionExecutor", lambda **_kwargs: MagicMock())
        monkeypatch.setattr("src.cli.AgentLoop", lambda **_kwargs: MagicMock())

        args = Namespace(
            observe=False,
            observer_host=None,
            observer_port=None,
            game_url="https://www.roblox.com/games/18408132742/Money-Clicker-Incremental",
            skip_auth=False,
            username="user",
            password="pass",
            totp_secret=None,
            auth_mode="strict",
            auth_challenge_mode="manual",
            auth_manual_timeout_seconds=180.0,
            recovery_profile="balanced",
            log_format="readable",
            progress_heartbeat_seconds=5.0,
            headless=False,
            runtime_target="auto",
        )

        runtime = _create_runtime(args)
        runtime.shutdown()

        assert env.navigate.call_count == 2


class TestCLILoggingAndRuntimeTarget:
    """Logging and runtime target helpers."""

    def test_configure_logging_applies_readable_format(self, caplog) -> None:
        _configure_logging(level="INFO", log_format="readable", quiet_uvicorn=True)
        logger = logging.getLogger("agentzero.test")
        with caplog.at_level(logging.INFO):
            logger.info("[BOOT] hello")

        assert any("[BOOT] hello" in msg for msg in caplog.messages)
        # Access logs should be suppressed by default.
        assert logging.getLogger("uvicorn.access").level >= logging.WARNING

    def test_resolve_runtime_target_native_falls_back_to_browser(self) -> None:
        resolved = _resolve_runtime_target(RuntimeTarget.NATIVE)
        assert resolved == RuntimeTarget.BROWSER

    def test_determine_startup_headless_respects_headed_on_macos(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.setattr("src.cli.sys.platform", "darwin")
        result = _determine_startup_headless(
            requested_headless=False,
            runtime_target=RuntimeTarget.BROWSER,
        )
        assert result is False

    def test_determine_startup_headless_keeps_user_headless(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.setattr("src.cli.sys.platform", "linux")
        result = _determine_startup_headless(
            requested_headless=True,
            runtime_target=RuntimeTarget.BROWSER,
        )
        assert result is True

    def test_run_with_heartbeat_emits_progress_during_blocking_operation(
        self,
        caplog,
    ) -> None:
        with caplog.at_level(logging.INFO):
            result = _run_with_heartbeat(
                stage="ENV",
                operation="start browser",
                heartbeat_seconds=0.05,
                operation_fn=lambda: time.sleep(0.12) or "ok",
            )

        assert result == "ok"
        assert any("[ENV]" in msg and "still running" in msg for msg in caplog.messages)


class TestCLIAuthMode:
    """Auth-mode behavior should enforce strict launch guarantees."""

    def test_enforce_auth_mode_allows_success_in_strict_mode(self) -> None:
        result = AuthResult(
            outcome=AuthOutcome.SUCCESS,
            message="ok",
            retryable=False,
            latency_ms=12.0,
        )

        _enforce_auth_mode(AuthMode.STRICT, result)

    def test_enforce_auth_mode_raises_in_strict_mode_for_failure(self) -> None:
        result = AuthResult(
            outcome=AuthOutcome.NETWORK_TIMEOUT,
            message="timeout",
            retryable=True,
            latency_ms=30123.0,
        )

        try:
            _enforce_auth_mode(AuthMode.STRICT, result)
            raise AssertionError("Expected strict auth mode to raise")
        except RuntimeError as exc:
            assert "NETWORK_TIMEOUT" in str(exc)

    def test_enforce_auth_mode_allows_failure_in_best_effort(self) -> None:
        result = AuthResult(
            outcome=AuthOutcome.CHALLENGE_BLOCKED,
            message="captcha",
            retryable=False,
            latency_ms=205.0,
        )

        _enforce_auth_mode(AuthMode.BEST_EFFORT, result)


class TestCLISessionAuthReuse:
    """Authentication should prefer a valid saved session before login."""

    def test_authenticate_session_uses_saved_session_when_valid(self) -> None:
        auth = MagicMock()
        auth.load_session.return_value = True
        auth.is_authenticated.return_value = True

        result = _authenticate_session(
            auth=auth,
            credentials=None,
            heartbeat_seconds=0.01,
        )

        assert result.outcome == AuthOutcome.SUCCESS
        assert "restored" in result.message.lower()
        auth.login.assert_not_called()

    def test_authenticate_session_falls_back_to_login_when_saved_session_invalid(self) -> None:
        auth = MagicMock()
        auth.load_session.return_value = True
        auth.is_authenticated.return_value = False
        auth.login.return_value = AuthResult(
            outcome=AuthOutcome.SUCCESS,
            message="login ok",
            retryable=False,
            latency_ms=10.0,
        )

        result = _authenticate_session(
            auth=auth,
            credentials=None,
            heartbeat_seconds=0.01,
        )

        assert result.outcome == AuthOutcome.SUCCESS
        auth.login.assert_called_once()

    def test_authenticate_session_saves_session_after_manual_resolution(self) -> None:
        auth = MagicMock()
        auth.load_session.return_value = False
        auth.login.return_value = AuthResult(
            outcome=AuthOutcome.CHALLENGE_BLOCKED,
            message="captcha",
            retryable=False,
            latency_ms=10.0,
        )
        auth.browser_is_headless = False
        auth.is_authenticated.side_effect = [False, True]

        result = _authenticate_session(
            auth=auth,
            credentials=None,
            heartbeat_seconds=0.01,
            challenge_mode=AuthChallengeMode.MANUAL,
            manual_timeout_seconds=0.3,
            manual_poll_seconds=0.01,
        )

        assert result.outcome == AuthOutcome.SUCCESS
        auth.save_session.assert_called_once()


class TestCLIAuthChallengeHandling:
    """Authentication challenge flow should be deterministic."""

    def test_authenticate_session_manual_challenge_resolves_to_success(self) -> None:
        auth = MagicMock()
        auth.load_session.return_value = False
        auth.login.return_value = AuthResult(
            outcome=AuthOutcome.CHALLENGE_BLOCKED,
            message="captcha",
            retryable=False,
            latency_ms=10.0,
        )
        auth.browser_is_headless = False
        auth.is_authenticated.side_effect = [False, True]

        result = _authenticate_session(
            auth=auth,
            credentials=None,
            heartbeat_seconds=0.01,
            challenge_mode=AuthChallengeMode.MANUAL,
            manual_timeout_seconds=0.3,
            manual_poll_seconds=0.01,
        )

        assert result.outcome == AuthOutcome.SUCCESS
        assert "manual" in result.message.lower()

    def test_authenticate_session_manual_challenge_fails_when_headless(self) -> None:
        auth = MagicMock()
        auth.load_session.return_value = False
        auth.login.return_value = AuthResult(
            outcome=AuthOutcome.CHALLENGE_BLOCKED,
            message="captcha",
            retryable=False,
            latency_ms=10.0,
        )
        auth.browser_is_headless = True

        result = _authenticate_session(
            auth=auth,
            credentials=None,
            heartbeat_seconds=0.01,
            challenge_mode=AuthChallengeMode.MANUAL,
            manual_timeout_seconds=1.0,
            manual_poll_seconds=0.01,
        )

        assert result.outcome == AuthOutcome.CHALLENGE_BLOCKED
        assert "without --headless" in result.message.lower()
        auth.is_authenticated.assert_not_called()

    def test_authenticate_session_fail_mode_keeps_challenge_failure(self) -> None:
        auth = MagicMock()
        auth.load_session.return_value = False
        auth.login.return_value = AuthResult(
            outcome=AuthOutcome.CHALLENGE_BLOCKED,
            message="captcha",
            retryable=False,
            latency_ms=10.0,
        )

        result = _authenticate_session(
            auth=auth,
            credentials=None,
            heartbeat_seconds=0.01,
            challenge_mode=AuthChallengeMode.FAIL,
            manual_timeout_seconds=1.0,
            manual_poll_seconds=0.01,
        )

        assert result.outcome == AuthOutcome.CHALLENGE_BLOCKED

    def test_authenticate_session_manual_challenge_calls_progress_hook(self) -> None:
        auth = MagicMock()
        auth.load_session.return_value = False
        auth.login.return_value = AuthResult(
            outcome=AuthOutcome.CHALLENGE_BLOCKED,
            message="captcha",
            retryable=False,
            latency_ms=10.0,
        )
        auth.browser_is_headless = False
        auth.is_authenticated.side_effect = [False, False, True]
        progress_calls: list[float] = []

        def _progress(elapsed_seconds: float) -> None:
            progress_calls.append(elapsed_seconds)

        result = _authenticate_session(
            auth=auth,
            credentials=None,
            heartbeat_seconds=0.01,
            challenge_mode=AuthChallengeMode.MANUAL,
            manual_timeout_seconds=0.6,
            manual_poll_seconds=0.1,
            manual_progress_hook=_progress,
        )

        assert result.outcome == AuthOutcome.SUCCESS
        assert progress_calls

    def test_authenticate_session_manual_challenge_handles_login_click_command(self) -> None:
        auth = MagicMock()
        auth.load_session.return_value = False
        auth.login.return_value = AuthResult(
            outcome=AuthOutcome.CHALLENGE_BLOCKED,
            message="captcha",
            retryable=False,
            latency_ms=10.0,
        )
        auth.browser_is_headless = False
        auth.is_authenticated.side_effect = [False, True]
        command_checks = {"count": 0}
        handled: list[str] = []

        def _command_consumer() -> list[dict[str, object]]:
            command_checks["count"] += 1
            if command_checks["count"] == 1:
                return [{"command": "auth_click_login", "id": 1, "payload": {}}]
            return []

        def _command_handler(command: dict[str, object]) -> None:
            handled.append(str(command.get("command")))

        result = _authenticate_session(
            auth=auth,
            credentials=None,
            heartbeat_seconds=0.01,
            challenge_mode=AuthChallengeMode.MANUAL,
            manual_timeout_seconds=0.5,
            manual_poll_seconds=0.1,
            manual_command_consumer=_command_consumer,
            manual_command_handler=_command_handler,
        )

        assert result.outcome == AuthOutcome.SUCCESS
        assert command_checks["count"] >= 1
        assert "auth_click_login" in handled


class TestCLIAuthCommand:
    """Dedicated auth checkpoint command behavior."""

    def test_auth_command_completes_on_success(self, monkeypatch) -> None:
        cfg = Namespace(
            logging=Namespace(level="INFO"),
            environment=Namespace(display_width=1920, display_height=1080, virtual_display=":99"),
            observer=Namespace(host="127.0.0.1", port=9090, stream_fps=10),
        )
        monkeypatch.setattr("src.cli.load_config", lambda: cfg)

        env = MagicMock()
        env.get_browser_runtime.return_value = MagicMock(page=MagicMock())
        monkeypatch.setattr("src.cli._start_environment", lambda *_args: env)
        monkeypatch.setattr(
            "src.cli._authenticate_session",
            lambda **_kwargs: AuthResult(AuthOutcome.SUCCESS, "ok", False, 10.0),
        )

        args = Namespace(
            command="auth",
            observe=False,
            observer_host=None,
            observer_port=None,
            username="user",
            password="pass",
            totp_secret=None,
            auth_manual_timeout_seconds=120.0,
            progress_heartbeat_seconds=5.0,
            log_format="readable",
            headless=False,
            runtime_target="auto",
        )

        code = auth_command(args)

        assert code == 0
        env.stop.assert_called_once()

    def test_auth_command_raises_when_checkpoint_fails(self, monkeypatch) -> None:
        cfg = Namespace(
            logging=Namespace(level="INFO"),
            environment=Namespace(display_width=1920, display_height=1080, virtual_display=":99"),
            observer=Namespace(host="127.0.0.1", port=9090, stream_fps=10),
        )
        monkeypatch.setattr("src.cli.load_config", lambda: cfg)

        env = MagicMock()
        env.get_browser_runtime.return_value = MagicMock(page=MagicMock())
        monkeypatch.setattr("src.cli._start_environment", lambda *_args: env)
        monkeypatch.setattr(
            "src.cli._authenticate_session",
            lambda **_kwargs: AuthResult(AuthOutcome.CHALLENGE_BLOCKED, "captcha", False, 10.0),
        )

        args = Namespace(
            command="auth",
            observe=False,
            observer_host=None,
            observer_port=None,
            username="user",
            password="pass",
            totp_secret=None,
            auth_manual_timeout_seconds=120.0,
            progress_heartbeat_seconds=5.0,
            log_format="readable",
            headless=False,
            runtime_target="auto",
        )

        with pytest.raises(RuntimeError):
            auth_command(args)


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

    def test_ignores_placeholder_openai_key_and_disables_llm(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "your_openai_api_key_here")

        resolved = _resolve_llm_runtime(
            configured_provider="anthropic",
            configured_model="claude-3-sonnet-20240229",
        )

        assert resolved.provider == "anthropic"
        assert resolved.api_key is None

    def test_ignores_blank_key_values(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        resolved = _resolve_llm_runtime(
            configured_provider="anthropic",
            configured_model="claude-3-sonnet-20240229",
        )

        assert resolved.provider == "anthropic"
        assert resolved.api_key is None
