"""CLI entrypoint for running Agent Zero gameplay sessions."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from src.actions.backend import InputBackend, NullInputBackend, PlaywrightInputBackend
from src.actions.executor import GameActionExecutor
from src.config.loader import load_config
from src.config.secrets import load_environment_secrets
from src.core.decision import DecisionConfig, DecisionEngine
from src.core.loop import AgentLoop, LoopConfig
from src.core.observation import ObservationPipeline, PipelineConfig
from src.environment.auth import (
    DEFAULT_STORAGE_PATH,
    AuthOutcome,
    AuthResult,
    Credentials,
    RobloxAuth,
)
from src.environment.manager import LocalEnvironmentManager
from src.interfaces.environment import EnvironmentSetupError
from src.observer.server import create_app
from src.observer.streaming import ActionStreamingService, ScreenStreamingService
from src.runtime.recovery import RecoveryPolicy, RuntimeRecoveryCoordinator
from src.runtime.targets import RuntimeBackendKind, resolve_runtime_backend
from src.vision.capture import ScreenshotCapture
from src.vision.ocr import OCRSystem
from src.vision.ui_detection import UIDetector

if TYPE_CHECKING:
    from src.interfaces.vision import Screenshot

logger = logging.getLogger(__name__)

_LLM_PROVIDER_ENV_KEYS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}
_RUNTIME_MODEL_DEFAULTS: dict[str, str] = {
    "anthropic": "claude-3-sonnet-20240229",
    "openai": "gpt-4o-mini",
}
_PLACEHOLDER_API_KEYS = frozenset(
    {
        "your_openai_api_key_here",
        "your_anthropic_api_key_here",
        "your_api_key_here",
        "changeme",
        "replace_me",
    }
)


class RuntimeTarget(StrEnum):
    """Runtime target selector for launch strategy."""

    AUTO = "auto"
    BROWSER = "browser"
    NATIVE = "native"


class AuthMode(StrEnum):
    """Authentication behavior for launch flow."""

    STRICT = "strict"
    BEST_EFFORT = "best-effort"


class AuthChallengeMode(StrEnum):
    """Behavior when a challenge/captcha blocks login."""

    FAIL = "fail"
    MANUAL = "manual"


class RecoveryProfile(StrEnum):
    """Recovery budget profile."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class LogFormat(StrEnum):
    """CLI log formatter mode."""

    READABLE = "readable"
    JSON = "json"


class _JSONLogFormatter(logging.Formatter):
    """Compact JSON formatter for machine-readable logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


class ObserverServer(Protocol):
    """Protocol for running observer server handles."""

    def stop(self) -> None:
        """Stop observer server."""
        ...


@dataclass
class _UvicornObserverServer:
    """Background uvicorn server handle."""

    server: Any
    thread: threading.Thread

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5.0)


@dataclass(frozen=True)
class LLMRuntimeConfig:
    """Resolved LLM runtime settings for this process."""

    provider: str
    model: str
    api_key: str | None


@dataclass
class AgentRuntime:
    """Runtime wrapper for an assembled agent session."""

    environment: LocalEnvironmentManager
    loop: AgentLoop
    pipeline: ObservationPipeline
    recovery_coordinator: RuntimeRecoveryCoordinator | None = None
    screen_streaming_service: ScreenStreamingService | None = None
    action_streaming_service: ActionStreamingService | None = None
    observer_server: ObserverServer | None = None

    def run_iterations(self, max_iterations: int) -> int:
        """Run the agent for a fixed number of iterations."""
        completed = 0
        while completed < max_iterations:
            try:
                observation, decision, action_result = self.loop.run_once()
            except Exception as exc:
                if self.recovery_coordinator is None:
                    raise
                should_continue = self.recovery_coordinator.handle(exc, environment=self.environment)
                if not should_continue:
                    raise
                continue

            self._publish_observation(observation.screenshot)
            self._publish_action_event(completed + 1, decision, action_result.success, action_result.error)
            completed += 1
        return completed

    def _publish_observation(self, screenshot: Screenshot) -> None:
        if self.screen_streaming_service is None:
            return
        frame = screenshot.raw_bytes if screenshot.raw_bytes else screenshot.image
        self.screen_streaming_service.push_frame(frame)

    def _publish_action_event(
        self,
        iteration: int,
        decision: object,
        success: bool,
        error: str | None,
    ) -> None:
        if self.action_streaming_service is None:
            return
        action_type = None
        decision_context: dict[str, object] = {}
        if hasattr(decision, "action") and hasattr(decision.action, "type"):
            action_type = getattr(decision.action.type, "value", str(decision.action.type))
        if hasattr(decision, "context") and isinstance(decision.context, dict):
            decision_context = decision.context
        self.action_streaming_service.push_event(
            {
                "event": "action_executed",
                "iteration": iteration,
                "action_type": action_type,
                "success": success,
                "error": error,
                "decision_source": decision_context.get("decision_source"),
                "policy_strategy": decision_context.get("policy_strategy"),
            }
        )

    def shutdown(self) -> None:
        """Shutdown runtime resources."""
        self.loop.stop()
        self.pipeline.close()
        self.environment.stop()
        if self.observer_server is not None:
            self.observer_server.stop()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(prog="agent-zero", description="Roblox incremental game agent")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the agent")
    run_parser.add_argument("--max-iterations", type=int, default=100, help="Max loop iterations")
    run_parser.add_argument("--game-url", type=str, default=None, help="Game URL to open")
    run_parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    run_parser.add_argument("--skip-auth", action="store_true", help="Skip Roblox authentication")
    run_parser.add_argument("--username", type=str, default=None, help="Roblox username/email")
    run_parser.add_argument("--password", type=str, default=None, help="Roblox password")
    run_parser.add_argument("--totp-secret", type=str, default=None, help="Roblox TOTP secret")
    run_parser.add_argument("--observe", action="store_true", help="Enable live observer web app")
    run_parser.add_argument("--observer-host", type=str, default=None, help="Observer host override")
    run_parser.add_argument("--observer-port", type=int, default=None, help="Observer port override")
    run_parser.add_argument(
        "--runtime-target",
        type=str,
        default=RuntimeTarget.AUTO.value,
        choices=[target.value for target in RuntimeTarget],
        help="Runtime backend target selection",
    )
    run_parser.add_argument(
        "--auth-mode",
        type=str,
        default=AuthMode.STRICT.value,
        choices=[mode.value for mode in AuthMode],
        help="Authentication enforcement mode",
    )
    run_parser.add_argument(
        "--auth-challenge-mode",
        type=str,
        default=AuthChallengeMode.MANUAL.value,
        choices=[mode.value for mode in AuthChallengeMode],
        help="How to handle challenge/captcha prompts during login",
    )
    run_parser.add_argument(
        "--auth-manual-timeout-seconds",
        type=float,
        default=180.0,
        help="Maximum time to wait for manual challenge completion",
    )
    run_parser.add_argument(
        "--recovery-profile",
        type=str,
        default=RecoveryProfile.BALANCED.value,
        choices=[profile.value for profile in RecoveryProfile],
        help="Recovery aggressiveness profile",
    )
    run_parser.add_argument(
        "--log-format",
        type=str,
        default=LogFormat.READABLE.value,
        choices=[fmt.value for fmt in LogFormat],
        help="Terminal log format",
    )
    run_parser.add_argument(
        "--progress-heartbeat-seconds",
        type=float,
        default=5.0,
        help="Emit progress heartbeat every N seconds during blocking operations",
    )

    auth_parser = subparsers.add_parser(
        "auth",
        help="Run dedicated human authentication checkpoint and persist session",
    )
    auth_parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    auth_parser.add_argument("--username", type=str, default=None, help="Roblox username/email")
    auth_parser.add_argument("--password", type=str, default=None, help="Roblox password")
    auth_parser.add_argument("--totp-secret", type=str, default=None, help="Roblox TOTP secret")
    auth_parser.add_argument("--observe", action="store_true", help="Enable live observer web app")
    auth_parser.add_argument("--observer-host", type=str, default=None, help="Observer host override")
    auth_parser.add_argument("--observer-port", type=int, default=None, help="Observer port override")
    auth_parser.add_argument(
        "--runtime-target",
        type=str,
        default=RuntimeTarget.AUTO.value,
        choices=[target.value for target in RuntimeTarget],
        help="Runtime backend target selection",
    )
    auth_parser.add_argument(
        "--auth-challenge-mode",
        type=str,
        default=AuthChallengeMode.MANUAL.value,
        choices=[mode.value for mode in AuthChallengeMode],
        help="How to handle challenge/captcha prompts during login",
    )
    auth_parser.add_argument(
        "--auth-manual-timeout-seconds",
        type=float,
        default=240.0,
        help="Maximum time to wait for manual challenge completion",
    )
    auth_parser.add_argument(
        "--log-format",
        type=str,
        default=LogFormat.READABLE.value,
        choices=[fmt.value for fmt in LogFormat],
        help="Terminal log format",
    )
    auth_parser.add_argument(
        "--progress-heartbeat-seconds",
        type=float,
        default=5.0,
        help="Emit progress heartbeat every N seconds during blocking operations",
    )

    return parser


def build_credentials_from_args(args: argparse.Namespace) -> Credentials | None:
    """Build optional credentials from explicit CLI args."""
    if not args.username or not args.password:
        return None
    return Credentials(
        username=str(args.username),
        password=str(args.password),
        totp_secret=str(args.totp_secret) if args.totp_secret else None,
    )


def _configure_logging(
    level: str = "INFO",
    log_format: str = LogFormat.READABLE.value,
    quiet_uvicorn: bool = True,
) -> None:
    """Configure process-wide logging with stable launch-friendly defaults."""
    normalized_level = level.upper()
    resolved_level = getattr(logging, normalized_level, logging.INFO)
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not getattr(h, "_agentzero_handler", False)]

    handler = logging.StreamHandler()
    handler._agentzero_handler = True  # type: ignore[attr-defined]
    if log_format == LogFormat.JSON.value:
        formatter: logging.Formatter = _JSONLogFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(resolved_level)

    if quiet_uvicorn:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    # Third-party HTTP transport logs are noisy at INFO during loop execution.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def _run_with_heartbeat(
    *,
    stage: str,
    operation: str,
    heartbeat_seconds: float,
    operation_fn: Callable[[], Any],
) -> Any:
    """Run a blocking operation while emitting periodic heartbeat logs."""
    if heartbeat_seconds <= 0:
        return operation_fn()

    done = threading.Event()
    started = time.monotonic()

    def _heartbeat() -> None:
        while not done.wait(heartbeat_seconds):
            elapsed = time.monotonic() - started
            logger.info("[%s] %s still running (elapsed=%.1fs)", stage, operation, elapsed)

    thread = threading.Thread(target=_heartbeat, daemon=True, name=f"{stage}-heartbeat")
    thread.start()
    try:
        return operation_fn()
    finally:
        done.set()
        thread.join(timeout=0.1)


def _warn_roblox_runtime_constraints() -> None:
    """Emit runtime constraints relevant to real Roblox playability."""
    logger.warning(
        "Roblox game launch may require the native Roblox client; browser-only automation can be limited."
    )
    logger.warning("Roblox anti-cheat may block virtualized environments for some sessions.")


def _start_observer_server(
    host: str,
    port: int,
    screen_service: ScreenStreamingService,
    action_service: ActionStreamingService,
) -> ObserverServer | None:
    """Start observer FastAPI server in a background thread."""
    try:
        import uvicorn
    except ImportError:  # pragma: no cover - optional dependency
        logger.warning("uvicorn not installed; observer web app disabled")
        return None

    app = create_app(
        streaming_service=screen_service,
        action_streaming_service=action_service,
    )
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="ObserverServer")
    thread.start()
    # Best-effort brief wait for initial bind.
    time.sleep(0.1)
    return _UvicornObserverServer(server=server, thread=thread)


def _is_display_start_failure(error: EnvironmentSetupError) -> bool:
    """Determine whether startup failure is related to virtual display/Xvfb."""
    message = str(error).lower()
    display_markers = (
        "failed to start display",
        "xvfb",
        "virtual display",
        ".x11-unix",
        "cannot establish any listening sockets",
    )
    return any(marker in message for marker in display_markers)


def _resolve_runtime_target(runtime_target: RuntimeTarget) -> RuntimeTarget:
    """Resolve effective runtime target for this process."""
    selection = resolve_runtime_backend(runtime_target.value, platform=sys.platform)
    effective = RuntimeTarget(selection.effective.value)
    if runtime_target == RuntimeTarget.NATIVE and effective != RuntimeTarget.NATIVE:
        logger.warning("Native runtime target is not available yet; falling back to browser.")
    if runtime_target == RuntimeTarget.AUTO and selection.requested == RuntimeBackendKind.AUTO:
        return effective
    return effective


def _determine_startup_headless(
    requested_headless: bool,
    runtime_target: RuntimeTarget,
) -> bool:
    """Determine startup headless behavior based on OS and target."""
    if requested_headless:
        return True
    if runtime_target == RuntimeTarget.BROWSER and sys.platform == "darwin":
        logger.info("[ENV] macOS detected; preferring headed startup for interactive auth.")
        return False
    return False


def _determine_virtual_display_usage(
    *,
    startup_headless: bool,
    runtime_target: RuntimeTarget,
) -> bool:
    """Determine whether virtual-display startup should be used."""
    if startup_headless:
        return False
    if runtime_target == RuntimeTarget.BROWSER and sys.platform == "darwin":
        logger.info("[ENV] macOS detected; using native display (virtual display disabled).")
        return False
    return True


def _start_environment(args: argparse.Namespace, config: Any) -> LocalEnvironmentManager:
    """Start environment with deterministic startup strategy and fallback."""
    storage_state_path = _resolve_storage_state_path()

    requested_runtime_target = getattr(args, "runtime_target", None)
    if requested_runtime_target is None:
        runtime_target = RuntimeTarget.BROWSER
        startup_headless = bool(args.headless)
        use_virtual_display = _determine_virtual_display_usage(
            startup_headless=startup_headless,
            runtime_target=runtime_target,
        )

        def create_environment(*, headless: bool, use_virtual_display: bool) -> LocalEnvironmentManager:
            return LocalEnvironmentManager(
                headless=headless,
                viewport_width=config.environment.display_width,
                viewport_height=config.environment.display_height,
                display=config.environment.virtual_display,
                storage_state_path=storage_state_path,
                use_virtual_display=use_virtual_display,
            )
    else:
        selection = resolve_runtime_backend(str(requested_runtime_target), platform=sys.platform)
        runtime_target = RuntimeTarget(selection.effective.value)
        startup_headless = _determine_startup_headless(
            requested_headless=bool(args.headless),
            runtime_target=runtime_target,
        )
        use_virtual_display = _determine_virtual_display_usage(
            startup_headless=startup_headless,
            runtime_target=runtime_target,
        )

        def create_environment(*, headless: bool, use_virtual_display: bool) -> LocalEnvironmentManager:
            return selection.backend.create_environment(
                headless=headless,
                viewport_width=config.environment.display_width,
                viewport_height=config.environment.display_height,
                display=config.environment.virtual_display,
                storage_state_path=storage_state_path,
                use_virtual_display=use_virtual_display,
            )

    heartbeat_seconds = float(getattr(args, "progress_heartbeat_seconds", 5.0))
    env = create_environment(headless=startup_headless, use_virtual_display=use_virtual_display)
    try:
        _run_with_heartbeat(
            stage="ENV",
            operation="start environment",
            heartbeat_seconds=heartbeat_seconds,
            operation_fn=env.start,
        )
        return env
    except EnvironmentSetupError as error:
        if startup_headless or not _is_display_start_failure(error):
            raise
        logger.warning("Display startup failed (%s). Retrying with headless browser mode.", error)
        fallback = create_environment(headless=True, use_virtual_display=False)
        _run_with_heartbeat(
            stage="ENV",
            operation="start environment (headless fallback)",
            heartbeat_seconds=heartbeat_seconds,
            operation_fn=fallback.start,
        )
        return fallback


def _model_matches_provider(provider: str, model: str) -> bool:
    """Validate basic provider/model compatibility."""
    lowered = model.strip().lower()
    if provider == "anthropic":
        return "claude" in lowered
    if provider == "openai":
        return any(token in lowered for token in ("gpt", "o1", "o3", "o4"))
    return False


def _normalize_runtime_model(provider: str, configured_model: str | None) -> str:
    """Normalize model so provider/model pairs are valid by default."""
    fallback = _RUNTIME_MODEL_DEFAULTS[provider]
    if configured_model is None:
        return fallback

    if _model_matches_provider(provider, configured_model):
        return configured_model

    logger.warning(
        "Configured model '%s' does not look compatible with provider '%s'; using '%s' instead.",
        configured_model,
        provider,
        fallback,
    )
    return fallback


def _is_placeholder_api_key(value: str) -> bool:
    """Best-effort placeholder detection for env-provided API keys."""
    lowered = value.strip().lower()
    if lowered in _PLACEHOLDER_API_KEYS:
        return True
    return lowered.startswith("your_") and lowered.endswith("_here")


def _read_provider_api_key(provider: str) -> str | None:
    """Read and sanitize provider API key from environment."""
    env_key = _LLM_PROVIDER_ENV_KEYS[provider]
    raw = os.environ.get(env_key)
    if raw is None:
        return None

    cleaned = raw.strip().strip('"').strip("'")
    if not cleaned:
        logger.warning("%s is set but blank; ignoring it.", env_key)
        return None

    if _is_placeholder_api_key(cleaned):
        logger.warning("%s appears to be a placeholder value; ignoring it.", env_key)
        return None

    return cleaned


def _resolve_llm_runtime(
    configured_provider: str,
    configured_model: str | None,
) -> LLMRuntimeConfig:
    """Resolve provider/model/api-key for runtime execution."""
    provider = configured_provider.strip().lower()
    if provider not in _LLM_PROVIDER_ENV_KEYS:
        raise ValueError(f"Unsupported LLM provider: {configured_provider}")

    primary_env = _LLM_PROVIDER_ENV_KEYS[provider]
    primary_key = _read_provider_api_key(provider)
    if primary_key:
        model = _normalize_runtime_model(provider, configured_model)
        return LLMRuntimeConfig(provider=provider, model=model, api_key=primary_key)

    fallback_provider = "openai" if provider == "anthropic" else "anthropic"
    fallback_key = _read_provider_api_key(fallback_provider)
    if fallback_key:
        model = _normalize_runtime_model(fallback_provider, configured_model)
        logger.warning(
            "Configured provider '%s' is missing %s; falling back to '%s'.",
            provider,
            primary_env,
            fallback_provider,
        )
        return LLMRuntimeConfig(
            provider=fallback_provider,
            model=model,
            api_key=fallback_key,
        )

    model = _normalize_runtime_model(provider, configured_model)
    logger.warning(
        "No LLM API key found (%s or %s). Running policy-only decision mode.",
        _LLM_PROVIDER_ENV_KEYS["anthropic"],
        _LLM_PROVIDER_ENV_KEYS["openai"],
    )
    return LLMRuntimeConfig(provider=provider, model=model, api_key=None)


def _enforce_auth_mode(auth_mode: AuthMode, result: AuthResult) -> None:
    """Enforce strict auth requirements based on configured mode."""
    if auth_mode == AuthMode.STRICT and result.outcome != AuthOutcome.SUCCESS:
        raise RuntimeError(f"Authentication failed ({result.outcome}): {result.message}")


def _resolve_storage_state_path() -> Path:
    """Resolve storage-state path for Roblox session reuse."""
    raw = os.environ.get("AGENTZERO_SESSION_STATE_FILE")
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_STORAGE_PATH.resolve()


def _authenticate_session(
    *,
    auth: RobloxAuth,
    credentials: Credentials | None,
    heartbeat_seconds: float,
    challenge_mode: AuthChallengeMode = AuthChallengeMode.MANUAL,
    manual_timeout_seconds: float = 180.0,
    manual_poll_seconds: float = 1.0,
    manual_progress_hook: Callable[[float], None] | None = None,
    manual_command_consumer: Callable[[], list[dict[str, object]]] | None = None,
    manual_command_handler: Callable[[dict[str, object]], None] | None = None,
) -> AuthResult:
    """Authenticate using saved session first, then credentials login."""
    start = time.time()
    try:
        if auth.load_session():
            restored = _run_with_heartbeat(
                stage="AUTH",
                operation="validate restored session",
                heartbeat_seconds=heartbeat_seconds,
                operation_fn=auth.is_authenticated,
            )
            if restored:
                return AuthResult(
                    outcome=AuthOutcome.SUCCESS,
                    message="Session restored from saved state",
                    retryable=False,
                    latency_ms=(time.time() - start) * 1000,
                )
    except Exception as exc:
        logger.warning("[AUTH] Session restore check failed; falling back to login: %s", exc)

    login_result = _run_with_heartbeat(
        stage="AUTH",
        operation="authenticate session",
        heartbeat_seconds=heartbeat_seconds,
        operation_fn=lambda: auth.login(credentials=credentials),
    )
    if login_result.outcome != AuthOutcome.CHALLENGE_BLOCKED:
        return login_result
    if challenge_mode != AuthChallengeMode.MANUAL:
        return login_result
    if auth.browser_is_headless:
        return AuthResult(
            outcome=AuthOutcome.CHALLENGE_BLOCKED,
            message=(
                "Challenge detected but browser is headless. "
                "Re-run without --headless to complete challenge manually."
            ),
            retryable=False,
            latency_ms=(time.time() - start) * 1000,
        )

    logger.warning("[AUTH] Challenge detected. Complete it in the browser window.")

    def _wait_for_manual_auth() -> bool:
        deadline = time.monotonic() + max(manual_timeout_seconds, 0.0)
        poll_seconds = max(manual_poll_seconds, 0.1)
        started_wait = time.monotonic()
        while time.monotonic() < deadline:
            if manual_progress_hook is not None:
                with contextlib.suppress(Exception):
                    manual_progress_hook(time.monotonic() - started_wait)
            if manual_command_consumer is not None:
                with contextlib.suppress(Exception):
                    commands = manual_command_consumer()
                    if commands:
                        logger.info("[AUTH] Received %d manual control command(s).", len(commands))
                        if manual_command_handler is not None:
                            for command in commands:
                                with contextlib.suppress(Exception):
                                    manual_command_handler(command)
            if auth.is_authenticated():
                return True
            time.sleep(poll_seconds)
        return False

    solved = _run_with_heartbeat(
        stage="AUTH",
        operation="wait for manual challenge completion",
        heartbeat_seconds=heartbeat_seconds,
        operation_fn=_wait_for_manual_auth,
    )
    if solved:
        with contextlib.suppress(Exception):
            auth.save_session()
        return AuthResult(
            outcome=AuthOutcome.SUCCESS,
            message="Authentication completed after manual challenge",
            retryable=False,
            latency_ms=(time.time() - start) * 1000,
        )
    return AuthResult(
        outcome=AuthOutcome.CHALLENGE_BLOCKED,
        message=(
            "Login blocked by challenge/captcha flow; "
            f"manual completion timed out after {manual_timeout_seconds:.0f}s."
        ),
        retryable=True,
        latency_ms=(time.time() - start) * 1000,
    )


def _emit_observer_event(
    action_streaming_service: ActionStreamingService | None,
    payload: dict[str, object],
) -> None:
    """Emit a best-effort observer event."""
    if action_streaming_service is None:
        return
    action_streaming_service.push_event(payload)


def _build_manual_command_consumer(
    action_streaming_service: ActionStreamingService | None,
) -> Callable[[], list[dict[str, object]]] | None:
    """Build a consumer for observer-issued manual auth commands."""
    if action_streaming_service is None:
        return None

    def _consumer() -> list[dict[str, object]]:
        commands = action_streaming_service.pop_control_commands()
        normalized: list[dict[str, object]] = []
        for command in commands:
            if isinstance(command, dict):
                normalized.append({k: v for k, v in command.items() if isinstance(k, str)})
        return normalized

    return _consumer


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
