"""Compatibility module for legacy imports of `src.cli.main`."""

from __future__ import annotations

from src.cli import (
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
    main,
    run_command,
)

__all__ = [
    "_authenticate_session",
    "_configure_logging",
    "_create_runtime",
    "_determine_startup_headless",
    "_enforce_auth_mode",
    "_resolve_llm_runtime",
    "_resolve_runtime_target",
    "_run_with_heartbeat",
    "_start_environment",
    "auth_command",
    "main",
    "run_command",
]
