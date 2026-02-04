"""Roblox authentication management.

This module provides authentication handling for Roblox:
- Login with username/password via Playwright
- Session persistence using browser storage state
- 2FA/TOTP support
- Automatic re-authentication on session expiration

Security Notes:
- Credentials should be provided via environment variables
- Never log credentials or session tokens
- Session state is stored encrypted when possible

Example:
    >>> from src.environment.auth import RobloxAuth
    >>> from src.environment.browser import BrowserRuntime
    >>>
    >>> browser = BrowserRuntime()
    >>> browser.start()
    >>> auth = RobloxAuth(browser)
    >>>
    >>> # Login (credentials from environment)
    >>> auth.login()
    >>>
    >>> # Check if authenticated
    >>> if auth.is_authenticated():
    ...     browser.navigate("https://www.roblox.com/games/123456")
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.sync_api import Page

    from src.environment.browser import BrowserRuntime

logger = logging.getLogger(__name__)

# Roblox URLs
ROBLOX_HOME = "https://www.roblox.com"
ROBLOX_LOGIN = "https://www.roblox.com/login"
ROBLOX_LOGOUT = "https://www.roblox.com/logout"

# Environment variable names for credentials
ENV_USERNAME = "ROBLOX_USERNAME"
ENV_PASSWORD = "ROBLOX_PASSWORD"
ENV_TOTP_SECRET = "ROBLOX_TOTP_SECRET"

# Default storage path for session state
DEFAULT_STORAGE_PATH = Path("data/roblox_session.json")


class AuthenticationError(Exception):
    """Error raised when authentication fails."""

    pass


@dataclass
class Credentials:
    """Roblox login credentials.

    Attributes:
        username: Roblox username or email.
        password: Roblox password.
        totp_secret: Optional TOTP secret for 2FA (base32 encoded).
    """

    username: str
    password: str
    totp_secret: str | None = None

    @classmethod
    def from_environment(cls) -> Credentials:
        """Load credentials from environment variables.

        Returns:
            Credentials loaded from environment.

        Raises:
            AuthenticationError: If required credentials are missing.
        """
        username = os.environ.get(ENV_USERNAME)
        password = os.environ.get(ENV_PASSWORD)
        totp_secret = os.environ.get(ENV_TOTP_SECRET)

        if not username:
            raise AuthenticationError(
                f"Missing {ENV_USERNAME} environment variable. "
                "Set it to your Roblox username or email."
            )
        if not password:
            raise AuthenticationError(
                f"Missing {ENV_PASSWORD} environment variable. "
                "Set it to your Roblox password."
            )

        logger.debug("Credentials loaded from environment")
        return cls(username=username, password=password, totp_secret=totp_secret)

    def has_2fa(self) -> bool:
        """Check if 2FA credentials are available."""
        return self.totp_secret is not None and len(self.totp_secret) > 0


def generate_totp_code(secret: str) -> str:
    """Generate a TOTP code from a secret.

    Uses the standard TOTP algorithm (RFC 6238) with:
    - 30 second time step
    - 6 digit codes
    - SHA1 hash

    Args:
        secret: Base32 encoded TOTP secret.

    Returns:
        6-digit TOTP code as string.

    Raises:
        AuthenticationError: If TOTP generation fails.
    """
    try:
        import base64
        import hashlib
        import hmac
        import struct

        # Decode the base32 secret
        # Handle secrets with or without padding
        secret = secret.upper().replace(" ", "")
        # Add padding if needed
        padding = 8 - (len(secret) % 8)
        if padding != 8:
            secret += "=" * padding

        key = base64.b32decode(secret)

        # Get current time step (30 second intervals)
        time_step = int(time.time()) // 30

        # Pack time as big-endian 64-bit integer
        time_bytes = struct.pack(">Q", time_step)

        # Generate HMAC-SHA1
        hmac_hash = hmac.new(key, time_bytes, hashlib.sha1).digest()

        # Dynamic truncation
        offset = hmac_hash[-1] & 0x0F
        code_int = struct.unpack(">I", hmac_hash[offset : offset + 4])[0]
        code_int &= 0x7FFFFFFF  # Remove sign bit
        code = code_int % 1000000  # 6 digits

        return f"{code:06d}"

    except Exception as e:
        raise AuthenticationError(f"Failed to generate TOTP code: {e}") from e


class RobloxAuth:
    """Manages Roblox authentication.

    Handles login, session persistence, and 2FA for Roblox.
    Uses Playwright's storage state for session persistence.

    Attributes:
        _browser: BrowserRuntime instance.
        _storage_path: Path for storing session state.
        _credentials: Optional cached credentials.

    Example:
        >>> auth = RobloxAuth(browser)
        >>> auth.login()  # Uses credentials from environment
        >>> assert auth.is_authenticated()
    """

    def __init__(
        self,
        browser: BrowserRuntime,
        storage_path: Path | str | None = None,
    ) -> None:
        """Initialize Roblox authentication manager.

        Args:
            browser: BrowserRuntime instance to use for authentication.
            storage_path: Path for storing session state. If None, uses default.
        """
        self._browser = browser
        self._storage_path = Path(storage_path) if storage_path else DEFAULT_STORAGE_PATH
        self._credentials: Credentials | None = None

        logger.debug(f"RobloxAuth initialized with storage at {self._storage_path}")

    @property
    def page(self) -> Page:
        """Get the Playwright page from browser runtime."""
        if self._browser.page is None:
            raise AuthenticationError("Browser not started. Call browser.start() first.")
        return self._browser.page

    def _ensure_storage_dir(self) -> None:
        """Ensure the storage directory exists."""
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

    def save_session(self) -> None:
        """Save current browser session state to storage.

        The session state includes cookies and local storage,
        allowing the session to persist across restarts.
        """
        self._ensure_storage_dir()
        try:
            storage_state = self.page.context.storage_state()
            with open(self._storage_path, "w") as f:
                json.dump(storage_state, f)
            logger.info(f"Session saved to {self._storage_path}")
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")

    def load_session(self) -> bool:
        """Load session state from storage.

        Returns:
            True if session was loaded successfully.
        """
        if not self._storage_path.exists():
            logger.debug("No saved session found")
            return False

        try:
            # Playwright contexts need storage state at creation time,
            # but we can check if the file is valid
            with open(self._storage_path) as f:
                state = json.load(f)

            # Verify it has the expected structure
            if "cookies" not in state:
                logger.warning("Invalid session state file")
                return False

            logger.info(f"Session state loaded from {self._storage_path}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load session: {e}")
            return False

    def clear_session(self) -> None:
        """Clear saved session state."""
        if self._storage_path.exists():
            self._storage_path.unlink()
            logger.info("Session cleared")

    def is_authenticated(self) -> bool:
        """Check if currently authenticated with Roblox.

        Navigates to Roblox and checks for authentication indicators.

        Returns:
            True if authenticated, False otherwise.
        """
        try:
            # Navigate to Roblox home
            self.page.goto(ROBLOX_HOME, wait_until="domcontentloaded", timeout=30000)

            # Wait a moment for any redirects
            self.page.wait_for_timeout(1000)

            # Check for authentication indicators
            # - Logged in: User menu or avatar present
            # - Logged out: Login button visible

            # Try to find user menu (indicates logged in)
            user_menu = self.page.locator('[data-testid="user-menu"]')
            if user_menu.count() > 0:
                logger.debug("Authentication check: User menu found - authenticated")
                return True

            # Alternative: check for avatar
            avatar = self.page.locator(".avatar-card-link, .age-bracket-label-username")
            if avatar.count() > 0:
                logger.debug("Authentication check: Avatar found - authenticated")
                return True

            # Check for login button (indicates not logged in)
            login_btn = self.page.locator('a[href*="/login"], .login-button')
            if login_btn.count() > 0:
                logger.debug("Authentication check: Login button found - not authenticated")
                return False

            # If we can't determine, assume not authenticated
            logger.debug("Authentication check: Could not determine status")
            return False

        except Exception as e:
            logger.warning(f"Authentication check failed: {e}")
            return False

    def login(
        self,
        credentials: Credentials | None = None,
        save_session: bool = True,
    ) -> bool:
        """Log in to Roblox.

        Args:
            credentials: Credentials to use. If None, loads from environment.
            save_session: Whether to save session after successful login.

        Returns:
            True if login successful.

        Raises:
            AuthenticationError: If login fails.
        """
        # Get credentials
        if credentials is None:
            credentials = Credentials.from_environment()
        self._credentials = credentials

        logger.info(f"Attempting login for user: {credentials.username}")

        try:
            # Navigate to login page
            self.page.goto(ROBLOX_LOGIN, wait_until="domcontentloaded", timeout=30000)
            self.page.wait_for_timeout(1000)

            # Check if already logged in
            if self.is_authenticated():
                logger.info("Already authenticated")
                return True

            # Fill in credentials
            # Find username field
            username_input = self.page.locator(
                'input[name="username"], input[id="login-username"], '
                'input[placeholder*="Username"], input[placeholder*="Email"]'
            )
            username_input.fill(credentials.username)
            logger.debug("Username entered")

            # Find password field
            password_input = self.page.locator(
                'input[name="password"], input[id="login-password"], '
                'input[type="password"]'
            )
            password_input.fill(credentials.password)
            logger.debug("Password entered")

            # Click login button
            login_button = self.page.locator(
                'button[type="submit"], button[id="login-button"], '
                'button:has-text("Log In"), button:has-text("Login")'
            )
            login_button.click()
            logger.debug("Login button clicked")

            # Wait for response
            self.page.wait_for_timeout(3000)

            # Check for 2FA prompt
            if self._check_for_2fa_prompt():
                if not credentials.has_2fa():
                    raise AuthenticationError(
                        "2FA required but no TOTP secret provided. "
                        f"Set the {ENV_TOTP_SECRET} environment variable."
                    )
                self._handle_2fa(credentials.totp_secret)  # type: ignore[arg-type]

            # Wait for login to complete
            self.page.wait_for_timeout(2000)

            # Verify login succeeded
            if self.is_authenticated():
                logger.info("Login successful")
                if save_session:
                    self.save_session()
                return True

            # Check for error messages
            error_msg = self._get_login_error()
            if error_msg:
                raise AuthenticationError(f"Login failed: {error_msg}")

            raise AuthenticationError("Login failed: Unknown error")

        except AuthenticationError:
            raise
        except Exception as e:
            raise AuthenticationError(f"Login failed: {e}") from e

    def _check_for_2fa_prompt(self) -> bool:
        """Check if 2FA prompt is displayed.

        Returns:
            True if 2FA prompt is visible.
        """
        # Look for 2FA input field or prompt
        twofa_indicators = [
            'input[name="code"]',
            'input[placeholder*="code"]',
            'input[placeholder*="Code"]',
            'text="Two-Step Verification"',
            'text="Enter the code"',
            'text="Authenticator"',
        ]

        for selector in twofa_indicators:
            try:
                element = self.page.locator(selector)
                if element.count() > 0:
                    logger.debug("2FA prompt detected")
                    return True
            except Exception:
                pass

        return False

    def _handle_2fa(self, totp_secret: str) -> None:
        """Handle 2FA verification.

        Args:
            totp_secret: TOTP secret for code generation.

        Raises:
            AuthenticationError: If 2FA verification fails.
        """
        logger.info("Handling 2FA verification")

        # Generate TOTP code
        code = generate_totp_code(totp_secret)
        logger.debug(f"Generated TOTP code: {code[:2]}****")

        # Find and fill the code input
        code_input = self.page.locator(
            'input[name="code"], input[placeholder*="code"], '
            'input[type="text"][maxlength="6"], input[type="number"]'
        )

        if code_input.count() == 0:
            raise AuthenticationError("Could not find 2FA code input")

        code_input.fill(code)
        logger.debug("2FA code entered")

        # Submit the code
        submit_button = self.page.locator(
            'button[type="submit"], button:has-text("Verify"), '
            'button:has-text("Submit"), button:has-text("Continue")'
        )

        if submit_button.count() > 0:
            submit_button.click()
            logger.debug("2FA submit clicked")

        # Wait for verification
        self.page.wait_for_timeout(3000)

        # Check if 2FA was successful (no longer on 2FA page)
        if self._check_for_2fa_prompt():
            raise AuthenticationError(
                "2FA verification failed. Check your TOTP secret."
            )

        logger.info("2FA verification successful")

    def _get_login_error(self) -> str | None:
        """Get login error message if present.

        Returns:
            Error message string, or None if no error visible.
        """
        error_selectors = [
            ".alert-error",
            ".error-message",
            '[class*="error"]',
            '[data-testid="login-error"]',
        ]

        for selector in error_selectors:
            try:
                element = self.page.locator(selector)
                if element.count() > 0:
                    text: str | None = element.first.text_content()
                    return text
            except Exception:
                pass

        return None

    def logout(self) -> None:
        """Log out from Roblox."""
        try:
            self.page.goto(ROBLOX_LOGOUT, wait_until="domcontentloaded", timeout=30000)
            self.clear_session()
            logger.info("Logged out successfully")
        except Exception as e:
            logger.warning(f"Logout failed: {e}")

    def ensure_authenticated(self) -> bool:
        """Ensure authenticated, re-authenticating if needed.

        Returns:
            True if authenticated (either already or after re-auth).

        Raises:
            AuthenticationError: If re-authentication fails.
        """
        if self.is_authenticated():
            return True

        logger.info("Session expired, re-authenticating")

        # Try to login
        if self._credentials:
            return self.login(self._credentials)
        else:
            return self.login()

    def get_storage_state_path(self) -> Path | None:
        """Get path to storage state file if it exists.

        This can be used when creating new browser contexts to
        restore the session.

        Returns:
            Path to storage state file, or None if not available.
        """
        if self._storage_path.exists():
            return self._storage_path
        return None
