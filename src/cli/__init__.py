"""CLI entrypoint for running Agent Zero gameplay sessions."""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
from collections.abc import Callable
from typing import Any

from src.actions.backend import InputBackend, NullInputBackend, PlaywrightInputBackend
from src.actions.executor import GameActionExecutor
from src.cli.helpers import (
    ObserverServer,
    _authenticate_session,
    _build_manual_command_consumer,
    _configure_logging,
    _determine_startup_headless,
    _emit_observer_event,
    _enforce_auth_mode,
    _resolve_llm_runtime,
    _resolve_runtime_target,
    _run_with_heartbeat,
    _start_environment as _start_environment_impl,
    _start_observer_server,
    _warn_roblox_runtime_constraints,
)
from src.cli.options import (
    AuthChallengeMode,
    AuthMode,
    LogFormat,
    RecoveryProfile,
    RuntimeTarget,
    build_arg_parser,
    build_credentials_from_args,
)
from src.cli.runtime import AgentRuntime
from src.config.loader import load_config
from src.config.secrets import load_environment_secrets
from src.core.decision import DecisionConfig, DecisionEngine
from src.core.loop import AgentLoop, LoopConfig
from src.core.observation import ObservationPipeline, PipelineConfig
from src.environment.auth import AuthOutcome, RobloxAuth
from src.environment.manager import LocalEnvironmentManager
from src.observer.streaming import ActionStreamingService, ScreenStreamingService
from src.runtime.recovery import RecoveryPolicy, RuntimeRecoveryCoordinator
from src.vision.capture import ScreenshotCapture
from src.vision.ocr import OCRSystem
from src.vision.ui_detection import UIDetector

logger = logging.getLogger(__name__)


def _start_environment(args: argparse.Namespace, config: Any) -> LocalEnvironmentManager:
    """Start environment with the local module's environment class for patchable tests."""
    return _start_environment_impl(args, config, local_environment_cls=LocalEnvironmentManager)


def _create_runtime(args: argparse.Namespace) -> AgentRuntime:
    """Create and initialize a playable runtime from CLI arguments."""
    config = load_config()
    _configure_logging(
        level=str(config.logging.level),
        log_format=str(getattr(args, "log_format", LogFormat.READABLE.value)),
        quiet_uvicorn=True,
    )

    _warn_roblox_runtime_constraints()
    llm_runtime = _resolve_llm_runtime(
        configured_provider=str(config.llm.provider),
        configured_model=str(config.llm.model) if config.llm.model else None,
    )
    heartbeat_seconds = float(getattr(args, "progress_heartbeat_seconds", 5.0))
    auth_mode = AuthMode(str(getattr(args, "auth_mode", AuthMode.STRICT.value)))
    challenge_mode = AuthChallengeMode(
        str(getattr(args, "auth_challenge_mode", AuthChallengeMode.MANUAL.value))
    )
    auth_manual_timeout_seconds = float(getattr(args, "auth_manual_timeout_seconds", 180.0))
    screen_service: ScreenStreamingService | None = None
    action_service: ActionStreamingService | None = None
    observer_server: ObserverServer | None = None
    environment: LocalEnvironmentManager | None = None
    manual_command_consumer: Callable[[], list[dict[str, object]]] | None = None

    try:
        environment = _start_environment(args, config)

        if bool(args.observe):
            observer_host = str(args.observer_host or config.observer.host)
            observer_port = int(args.observer_port or config.observer.port)
            screen_service = ScreenStreamingService(stream_fps=config.observer.stream_fps)
            action_service = ActionStreamingService()
            manual_command_consumer = _build_manual_command_consumer(action_service)
            observer_server = _start_observer_server(
                host=observer_host,
                port=observer_port,
                screen_service=screen_service,
                action_service=action_service,
            )
            logger.info("[OBSERVER] live page: http://%s:%s/live", observer_host, observer_port)
            _emit_observer_event(
                action_service,
                {"event": "observer_started", "host": observer_host, "port": observer_port},
            )

        if args.game_url:
            _run_with_heartbeat(
                stage="ENV",
                operation=f"navigate to {args.game_url}",
                heartbeat_seconds=heartbeat_seconds,
                operation_fn=lambda: environment.navigate(str(args.game_url)),
            )
            _emit_observer_event(
                action_service,
                {"event": "navigation_complete", "url": str(args.game_url)},
            )
            if screen_service is not None:
                with contextlib.suppress(Exception):
                    screen_service.push_frame(environment.screenshot())

        explicit_credentials = build_credentials_from_args(args)
        if not bool(args.skip_auth):
            browser_runtime = environment.get_browser_runtime()
            if browser_runtime is None:
                raise RuntimeError("Browser runtime unavailable for authentication")
            auth = RobloxAuth(browser_runtime)
            _emit_observer_event(action_service, {"event": "auth_started"})

            def _manual_auth_progress(elapsed_seconds: float) -> None:
                _emit_observer_event(
                    action_service,
                    {"event": "auth_manual_wait", "elapsed_seconds": round(elapsed_seconds, 1)},
                )
                if screen_service is not None:
                    with contextlib.suppress(Exception):
                        screen_service.push_frame(environment.screenshot())

            def _manual_auth_command_handler(command: dict[str, object]) -> None:
                command_name = str(command.get("command", "")).strip().lower()
                if command_name != "auth_click_login":
                    return
                clicked = auth.click_login_button()
                _emit_observer_event(
                    action_service,
                    {
                        "event": (
                            "auth_click_login_executed"
                            if clicked
                            else "auth_click_login_not_found"
                        )
                    },
                )
                if screen_service is not None:
                    with contextlib.suppress(Exception):
                        screen_service.push_frame(environment.screenshot())

            auth_result = _authenticate_session(
                auth=auth,
                credentials=explicit_credentials,
                heartbeat_seconds=heartbeat_seconds,
                challenge_mode=challenge_mode,
                manual_timeout_seconds=auth_manual_timeout_seconds,
                manual_progress_hook=_manual_auth_progress,
                manual_command_consumer=manual_command_consumer,
                manual_command_handler=_manual_auth_command_handler,
            )
            _emit_observer_event(
                action_service,
                {
                    "event": "auth_result",
                    "outcome": auth_result.outcome.value,
                    "retryable": auth_result.retryable,
                    "latency_ms": auth_result.latency_ms,
                },
            )
            _enforce_auth_mode(auth_mode, auth_result)
            if auth_result.outcome != AuthOutcome.SUCCESS:
                logger.warning(
                    "Authentication failed (%s); continuing unauthenticated session",
                    auth_result.outcome,
                )
            elif args.game_url:
                logger.info("[AUTH] Re-entering target game after successful authentication.")
                _run_with_heartbeat(
                    stage="ENV",
                    operation=f"re-enter game {args.game_url}",
                    heartbeat_seconds=heartbeat_seconds,
                    operation_fn=lambda: environment.navigate(str(args.game_url)),
                )
                _emit_observer_event(
                    action_service,
                    {"event": "post_auth_navigation_complete", "url": str(args.game_url)},
                )
                if screen_service is not None:
                    with contextlib.suppress(Exception):
                        screen_service.push_frame(environment.screenshot())

        capture = ScreenshotCapture(environment_manager=environment, buffer_size=config.vision.buffer_size)
        ocr = OCRSystem()
        ui_detector = UIDetector()
        llm_vision = None
        if llm_runtime.api_key is not None:
            try:
                from src.vision.llm_vision import LLMVision

                llm_vision = LLMVision(
                    provider=llm_runtime.provider,
                    model=llm_runtime.model,
                    api_key=llm_runtime.api_key,
                )
            except Exception as exc:  # pragma: no cover - optional dependency path
                logger.warning("LLM vision unavailable (%s); using local extraction only", exc)
        else:
            logger.warning("LLM vision disabled because no compatible API key is available.")

        pipeline = ObservationPipeline(
            capture=capture,
            ocr=ocr,
            ui_detector=ui_detector,
            llm_vision=llm_vision,
            config=PipelineConfig(use_llm=llm_vision is not None),
        )

        decision_engine = DecisionEngine(
            config=DecisionConfig(
                provider=llm_runtime.provider,
                model=llm_runtime.model,
                policy_enabled=True,
                policy_confidence_threshold=(0.0 if llm_runtime.api_key is None else 0.7),
            ),
            api_key=llm_runtime.api_key,
        )

        backend: InputBackend = NullInputBackend()
        browser_runtime = environment.get_browser_runtime()
        if browser_runtime is not None and browser_runtime.page is not None:
            backend = PlaywrightInputBackend(browser_runtime.page)

        action_executor = GameActionExecutor(backend=backend)

        loop = AgentLoop(
            observation_pipeline=pipeline,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=LoopConfig(target_rate_hz=float(config.agent.loop_rate), enable_signal_handlers=False),
        )

        recovery_profile = RecoveryProfile(str(getattr(args, "recovery_profile", RecoveryProfile.BALANCED.value)))
        recovery_policy = RecoveryPolicy.for_profile(recovery_profile.value)

        return AgentRuntime(
            environment=environment,
            loop=loop,
            pipeline=pipeline,
            recovery_coordinator=RuntimeRecoveryCoordinator(recovery_policy),
            screen_streaming_service=screen_service,
            action_streaming_service=action_service,
            observer_server=observer_server,
        )
    except Exception:
        if observer_server is not None:
            with contextlib.suppress(Exception):
                observer_server.stop()
        if environment is not None:
            with contextlib.suppress(Exception):
                environment.stop()
        raise


def run_command(args: argparse.Namespace) -> int:
    """Execute the `run` command."""
    if args.command != "run":
        raise ValueError(f"Unsupported command: {args.command}")

    runtime = _create_runtime(args)
    try:
        runtime.run_iterations(int(args.max_iterations))
        return 0
    finally:
        runtime.shutdown()


def auth_command(args: argparse.Namespace) -> int:
    """Execute the dedicated authentication checkpoint command."""
    if args.command != "auth":
        raise ValueError(f"Unsupported command: {args.command}")

    config = load_config()
    _configure_logging(
        level=str(config.logging.level),
        log_format=str(getattr(args, "log_format", LogFormat.READABLE.value)),
        quiet_uvicorn=True,
    )
    _warn_roblox_runtime_constraints()

    heartbeat_seconds = float(getattr(args, "progress_heartbeat_seconds", 5.0))
    challenge_mode = AuthChallengeMode(
        str(getattr(args, "auth_challenge_mode", AuthChallengeMode.MANUAL.value))
    )
    auth_manual_timeout_seconds = float(getattr(args, "auth_manual_timeout_seconds", 240.0))
    screen_service: ScreenStreamingService | None = None
    action_service: ActionStreamingService | None = None
    observer_server: ObserverServer | None = None
    environment: LocalEnvironmentManager | None = None

    try:
        environment = _start_environment(args, config)

        if bool(getattr(args, "observe", False)):
            observer_host = str(args.observer_host or config.observer.host)
            observer_port = int(args.observer_port or config.observer.port)
            screen_service = ScreenStreamingService(stream_fps=config.observer.stream_fps)
            action_service = ActionStreamingService()
            observer_server = _start_observer_server(
                host=observer_host,
                port=observer_port,
                screen_service=screen_service,
                action_service=action_service,
            )
            logger.info("[OBSERVER] live page: http://%s:%s/live", observer_host, observer_port)
            _emit_observer_event(
                action_service,
                {"event": "auth_checkpoint_started", "host": observer_host, "port": observer_port},
            )

        browser_runtime = environment.get_browser_runtime()
        if browser_runtime is None:
            raise RuntimeError("Browser runtime unavailable for authentication checkpoint")
        auth = RobloxAuth(browser_runtime)
        explicit_credentials = build_credentials_from_args(args)

        def _manual_auth_progress(elapsed_seconds: float) -> None:
            _emit_observer_event(
                action_service,
                {"event": "auth_manual_wait", "elapsed_seconds": round(elapsed_seconds, 1)},
            )
            if screen_service is not None:
                with contextlib.suppress(Exception):
                    screen_service.push_frame(environment.screenshot())

        auth_result = _authenticate_session(
            auth=auth,
            credentials=explicit_credentials,
            heartbeat_seconds=heartbeat_seconds,
            challenge_mode=challenge_mode,
            manual_timeout_seconds=auth_manual_timeout_seconds,
            manual_progress_hook=_manual_auth_progress,
        )
        _emit_observer_event(
            action_service,
            {
                "event": "auth_result",
                "outcome": auth_result.outcome.value,
                "retryable": auth_result.retryable,
                "latency_ms": auth_result.latency_ms,
            },
        )
        if auth_result.outcome != AuthOutcome.SUCCESS:
            raise RuntimeError(
                f"Authentication checkpoint failed ({auth_result.outcome}): {auth_result.message}"
            )

        _emit_observer_event(action_service, {"event": "auth_checkpoint_complete"})
        logger.info("[AUTH] Authentication checkpoint complete; session persisted.")
        return 0
    finally:
        if observer_server is not None:
            with contextlib.suppress(Exception):
                observer_server.stop()
        if environment is not None:
            with contextlib.suppress(Exception):
                environment.stop()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint function."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    # Configure a sane bootstrap logger before config loading.
    _configure_logging(
        level="INFO",
        log_format=str(getattr(args, "log_format", LogFormat.READABLE.value)),
        quiet_uvicorn=True,
    )

    try:
        if args.command == "run":
            load_environment_secrets()
            return run_command(args)
        if args.command == "auth":
            load_environment_secrets()
            return auth_command(args)
        raise ValueError(f"Unsupported command: {args.command}")
    except Exception as exc:
        logger.error("[BOOT] CLI execution failed: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
