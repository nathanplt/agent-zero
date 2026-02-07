"""CLI option models and parser helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import StrEnum

from src.environment.auth import Credentials


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


@dataclass(frozen=True)
class LLMRuntimeConfig:
    """Resolved LLM runtime settings for this process."""

    provider: str
    model: str
    api_key: str | None


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
