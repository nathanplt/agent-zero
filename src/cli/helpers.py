"""Shared helper utilities for CLI runtime/auth orchestration."""

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
from pathlib import Path
from typing import Any, Protocol

from src.cli.options import (
    AuthChallengeMode,
    AuthMode,
    LLMRuntimeConfig,
    LogFormat,
    RuntimeTarget,
)
from src.environment.auth import DEFAULT_STORAGE_PATH, AuthOutcome, AuthResult, Credentials, RobloxAuth
from src.environment.manager import LocalEnvironmentManager
from src.interfaces.environment import EnvironmentSetupError
from src.observer.server import create_app
from src.observer.streaming import ActionStreamingService, ScreenStreamingService
from src.runtime.targets import RuntimeBackendKind, resolve_runtime_backend

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


class ObserverServer(Protocol):
    """Protocol for running observer server handles."""

    def stop(self) -> None:
        """Stop observer server."""
        ...


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


@dataclass
class _UvicornObserverServer:
    """Background uvicorn server handle."""

    server: Any
    thread: threading.Thread

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5.0)


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


def _resolve_storage_state_path() -> Path:
    """Resolve storage-state path for Roblox session reuse."""
    raw = os.environ.get("AGENTZERO_SESSION_STATE_FILE")
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_STORAGE_PATH.resolve()


def _start_environment(
    args: argparse.Namespace,
    config: Any,
    local_environment_cls: type[LocalEnvironmentManager] = LocalEnvironmentManager,
) -> LocalEnvironmentManager:
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
            return local_environment_cls(
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
