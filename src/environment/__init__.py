"""Environment management package.

This package provides implementations for managing the execution environment:
- LocalEnvironmentManager: Coordinates display and browser lifecycle
- VirtualDisplay: Manages virtual X11 display (Xvfb)
- BrowserRuntime: Manages browser lifecycle via Playwright
- RobloxAuth: Handles Roblox authentication, session persistence, and 2FA
"""

from src.environment.auth import (
    AuthenticationError,
    AuthOutcome,
    AuthResult,
    Credentials,
    RobloxAuth,
    generate_totp_code,
)
from src.environment.browser import BrowserRuntime
from src.environment.display import VirtualDisplay
from src.environment.manager import LocalEnvironmentManager

__all__ = [
    "AuthenticationError",
    "AuthOutcome",
    "AuthResult",
    "BrowserRuntime",
    "Credentials",
    "LocalEnvironmentManager",
    "RobloxAuth",
    "VirtualDisplay",
    "generate_totp_code",
]
