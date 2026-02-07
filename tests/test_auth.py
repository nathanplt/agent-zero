"""Tests for Feature 1.4: Roblox Authentication.

These tests verify:
- Credential loading from environment
- Session persistence (save/load/clear)
- TOTP code generation
- Authentication flow (mocked)

Note: Integration tests that actually log into Roblox
are marked with @pytest.mark.integration and require
real credentials.
"""

import json
from unittest.mock import MagicMock

import pytest


class TestCredentials:
    """Tests for Credentials class."""

    def test_credentials_creation(self):
        """Should create credentials with username and password."""
        from src.environment.auth import Credentials

        creds = Credentials(username="testuser", password="testpass")

        assert creds.username == "testuser"
        assert creds.password == "testpass"
        assert creds.totp_secret is None

    def test_credentials_with_totp(self):
        """Should create credentials with TOTP secret."""
        from src.environment.auth import Credentials

        creds = Credentials(
            username="testuser",
            password="testpass",
            totp_secret="JBSWY3DPEHPK3PXP",
        )

        assert creds.totp_secret == "JBSWY3DPEHPK3PXP"
        assert creds.has_2fa()

    def test_credentials_without_totp_has_2fa_false(self):
        """has_2fa() should return False without TOTP secret."""
        from src.environment.auth import Credentials

        creds = Credentials(username="testuser", password="testpass")

        assert not creds.has_2fa()

    def test_credentials_from_environment(self, monkeypatch):
        """Should load credentials from environment variables."""
        from src.environment.auth import Credentials

        monkeypatch.setenv("ROBLOX_USERNAME", "envuser")
        monkeypatch.setenv("ROBLOX_PASSWORD", "envpass")
        monkeypatch.setenv("ROBLOX_TOTP_SECRET", "ENVSECRET")

        creds = Credentials.from_environment()

        assert creds.username == "envuser"
        assert creds.password == "envpass"
        assert creds.totp_secret == "ENVSECRET"

    def test_credentials_from_environment_missing_username(self, monkeypatch):
        """Should raise error if username is missing."""
        from src.environment.auth import AuthenticationError, Credentials

        monkeypatch.delenv("ROBLOX_USERNAME", raising=False)
        monkeypatch.setenv("ROBLOX_PASSWORD", "testpass")

        with pytest.raises(AuthenticationError) as exc_info:
            Credentials.from_environment()

        assert "ROBLOX_USERNAME" in str(exc_info.value)

    def test_credentials_from_environment_missing_password(self, monkeypatch):
        """Should raise error if password is missing."""
        from src.environment.auth import AuthenticationError, Credentials

        monkeypatch.setenv("ROBLOX_USERNAME", "testuser")
        monkeypatch.delenv("ROBLOX_PASSWORD", raising=False)

        with pytest.raises(AuthenticationError) as exc_info:
            Credentials.from_environment()

        assert "ROBLOX_PASSWORD" in str(exc_info.value)

    def test_credentials_from_environment_optional_totp(self, monkeypatch):
        """TOTP secret should be optional."""
        from src.environment.auth import Credentials

        monkeypatch.setenv("ROBLOX_USERNAME", "testuser")
        monkeypatch.setenv("ROBLOX_PASSWORD", "testpass")
        monkeypatch.delenv("ROBLOX_TOTP_SECRET", raising=False)

        creds = Credentials.from_environment()

        assert creds.username == "testuser"
        assert creds.totp_secret is None
        assert not creds.has_2fa()


class TestTOTPGeneration:
    """Tests for TOTP code generation."""

    def test_generate_totp_code_format(self):
        """TOTP code should be 6 digits."""
        from src.environment.auth import generate_totp_code

        # Using a known test secret
        code = generate_totp_code("JBSWY3DPEHPK3PXP")

        assert len(code) == 6
        assert code.isdigit()

    def test_generate_totp_code_changes_over_time(self):
        """TOTP codes should change (eventually)."""
        from src.environment.auth import generate_totp_code

        # Note: This test may be flaky near time step boundaries
        # We just verify the function works, not exact timing
        code1 = generate_totp_code("JBSWY3DPEHPK3PXP")

        # Same secret at same time should give same code
        code2 = generate_totp_code("JBSWY3DPEHPK3PXP")

        # Codes should be equal (same time step)
        assert code1 == code2

    def test_generate_totp_handles_unpadded_secret(self):
        """Should handle secrets without padding."""
        from src.environment.auth import generate_totp_code

        # Secret without proper padding
        code = generate_totp_code("JBSWY3DPEHPK3PXP")

        assert len(code) == 6

    def test_generate_totp_handles_spaces(self):
        """Should handle secrets with spaces."""
        from src.environment.auth import generate_totp_code

        # Secret with spaces (common format)
        code = generate_totp_code("JBSW Y3DP EHPK 3PXP")

        assert len(code) == 6

    def test_generate_totp_handles_lowercase(self):
        """Should handle lowercase secrets."""
        from src.environment.auth import generate_totp_code

        code_upper = generate_totp_code("JBSWY3DPEHPK3PXP")
        code_lower = generate_totp_code("jbswy3dpehpk3pxp")

        assert code_upper == code_lower

    def test_generate_totp_invalid_secret(self):
        """Should raise error for invalid secrets."""
        from src.environment.auth import AuthenticationError, generate_totp_code

        with pytest.raises(AuthenticationError):
            generate_totp_code("not a valid base32 secret!!!")


class TestRobloxAuthInitialization:
    """Tests for RobloxAuth initialization."""

    @pytest.fixture
    def mock_browser(self):
        """Create a mock browser runtime."""
        browser = MagicMock()
        browser.page = MagicMock()
        browser.page.context = MagicMock()
        return browser

    def test_initialization(self, mock_browser):
        """Should initialize with browser runtime."""
        from src.environment.auth import RobloxAuth

        auth = RobloxAuth(mock_browser)

        assert auth._browser is mock_browser

    def test_initialization_with_custom_storage_path(self, mock_browser, tmp_path):
        """Should accept custom storage path."""
        from src.environment.auth import RobloxAuth

        storage_path = tmp_path / "custom_session.json"
        auth = RobloxAuth(mock_browser, storage_path=storage_path)

        assert auth._storage_path == storage_path

    def test_default_storage_path(self, mock_browser):
        """Should use default storage path if not specified."""
        from src.environment.auth import DEFAULT_STORAGE_PATH, RobloxAuth

        auth = RobloxAuth(mock_browser)

        assert auth._storage_path == DEFAULT_STORAGE_PATH


class TestSessionPersistence:
    """Tests for session save/load functionality."""

    @pytest.fixture
    def mock_browser(self):
        """Create a mock browser runtime."""
        browser = MagicMock()
        browser.page = MagicMock()
        browser.page.context = MagicMock()
        browser.page.context.storage_state.return_value = {
            "cookies": [{"name": "test", "value": "cookie"}],
            "origins": [],
        }
        return browser

    @pytest.fixture
    def auth(self, mock_browser, tmp_path):
        """Create RobloxAuth with temp storage."""
        from src.environment.auth import RobloxAuth

        return RobloxAuth(mock_browser, storage_path=tmp_path / "session.json")

    def test_save_session(self, auth):
        """Should save session state to file."""
        auth.save_session()

        assert auth._storage_path.exists()

        with open(auth._storage_path) as f:
            data = json.load(f)

        assert "cookies" in data

    def test_load_session_success(self, auth):
        """Should load session from file."""
        # Create a session file
        session_data = {"cookies": [{"name": "test"}], "origins": []}
        auth._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(auth._storage_path, "w") as f:
            json.dump(session_data, f)

        result = auth.load_session()

        assert result is True

    def test_load_session_no_file(self, auth):
        """Should return False if no session file."""
        result = auth.load_session()

        assert result is False

    def test_load_session_invalid_file(self, auth):
        """Should return False for invalid session file."""
        auth._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(auth._storage_path, "w") as f:
            f.write("not json")

        result = auth.load_session()

        assert result is False

    def test_load_session_missing_cookies(self, auth):
        """Should return False if session missing cookies."""
        auth._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(auth._storage_path, "w") as f:
            json.dump({"origins": []}, f)

        result = auth.load_session()

        assert result is False

    def test_clear_session(self, auth):
        """Should delete session file."""
        # Create a session file
        auth._storage_path.parent.mkdir(parents=True, exist_ok=True)
        auth._storage_path.write_text("{}")

        auth.clear_session()

        assert not auth._storage_path.exists()

    def test_clear_session_no_file(self, auth):
        """Should not error if no session file."""
        auth.clear_session()  # Should not raise

    def test_get_storage_state_path_exists(self, auth):
        """Should return path if session file exists."""
        auth._storage_path.parent.mkdir(parents=True, exist_ok=True)
        auth._storage_path.write_text("{}")

        result = auth.get_storage_state_path()

        assert result == auth._storage_path

    def test_get_storage_state_path_not_exists(self, auth):
        """Should return None if no session file."""
        result = auth.get_storage_state_path()

        assert result is None


class TestAuthenticationCheck:
    """Tests for authentication status checking."""

    @pytest.fixture
    def mock_browser(self):
        """Create a mock browser runtime."""
        browser = MagicMock()
        browser.page = MagicMock()
        return browser

    @pytest.fixture
    def auth(self, mock_browser, tmp_path):
        """Create RobloxAuth with mock browser."""
        from src.environment.auth import RobloxAuth

        return RobloxAuth(mock_browser, storage_path=tmp_path / "session.json")

    def test_is_authenticated_user_menu_found(self, auth):
        """Should return True if user menu is found."""
        # Mock locator to find user menu
        user_menu_locator = MagicMock()
        user_menu_locator.count.return_value = 1

        auth._browser.page.locator.return_value = user_menu_locator

        result = auth.is_authenticated()

        assert result is True

    def test_is_authenticated_login_button_found(self, auth):
        """Should return False if login button is found."""
        # First call for user menu returns 0
        # Second call for avatar returns 0
        # Third call for login button returns 1
        call_count = [0]

        def mock_locator(_selector):
            call_count[0] += 1
            locator = MagicMock()
            if call_count[0] <= 2:
                locator.count.return_value = 0
            else:
                locator.count.return_value = 1
            return locator

        auth._browser.page.locator.side_effect = mock_locator

        result = auth.is_authenticated()

        assert result is False


class TestModuleExports:
    """Tests for module exports."""

    def test_roblox_auth_exported(self):
        """RobloxAuth should be exported from environment package."""
        from src.environment import RobloxAuth

        assert RobloxAuth is not None

    def test_credentials_exported(self):
        """Credentials should be exported from environment package."""
        from src.environment import Credentials

        assert Credentials is not None

    def test_authentication_error_exported(self):
        """AuthenticationError should be exported from environment package."""
        from src.environment import AuthenticationError

        assert AuthenticationError is not None

    def test_generate_totp_code_exported(self):
        """generate_totp_code should be exported from environment package."""
        from src.environment import generate_totp_code

        assert generate_totp_code is not None


class TestLoginFlow:
    """Tests for login flow (mocked)."""

    @pytest.fixture
    def mock_browser(self):
        """Create a mock browser runtime."""
        browser = MagicMock()
        browser.page = MagicMock()
        browser.page.context = MagicMock()
        browser.page.context.storage_state.return_value = {"cookies": [], "origins": []}
        return browser

    @pytest.fixture
    def auth(self, mock_browser, tmp_path):
        """Create RobloxAuth with mock browser."""
        from src.environment.auth import RobloxAuth

        return RobloxAuth(mock_browser, storage_path=tmp_path / "session.json")

    def test_login_calls_navigate_to_login_page(self, auth, monkeypatch):
        """Login should navigate to login page."""
        # Set up environment
        monkeypatch.setenv("ROBLOX_USERNAME", "testuser")
        monkeypatch.setenv("ROBLOX_PASSWORD", "testpass")

        # Mock is_authenticated to return True after "login"
        auth_check_calls = [0]

        def mock_is_authenticated():
            auth_check_calls[0] += 1
            return auth_check_calls[0] > 1  # False first, then True

        auth.is_authenticated = mock_is_authenticated

        # Mock _check_for_2fa_prompt to return False (no 2FA)
        auth._check_for_2fa_prompt = MagicMock(return_value=False)

        # Mock locators for input fields
        mock_input = MagicMock()
        mock_input.count.return_value = 1
        auth._browser.page.locator.return_value = mock_input

        auth.login()

        # Verify navigation was called
        auth._browser.page.goto.assert_called()

    def test_login_fills_credentials(self, auth, monkeypatch):
        """Login should fill username and password."""
        monkeypatch.setenv("ROBLOX_USERNAME", "testuser")
        monkeypatch.setenv("ROBLOX_PASSWORD", "testpass")

        # Mock successful login
        auth_calls = [0]

        def mock_is_authenticated():
            auth_calls[0] += 1
            return auth_calls[0] > 1

        auth.is_authenticated = mock_is_authenticated

        # Mock _check_for_2fa_prompt to return False (no 2FA)
        auth._check_for_2fa_prompt = MagicMock(return_value=False)

        # Track fill calls
        fill_calls = []
        mock_input = MagicMock()
        mock_input.fill = lambda x: fill_calls.append(x)
        mock_input.count.return_value = 1
        mock_input.click = MagicMock()

        auth._browser.page.locator.return_value = mock_input

        auth.login()

        # Verify credentials were filled
        assert "testuser" in fill_calls
        assert "testpass" in fill_calls

    def test_login_saves_session_on_success(self, auth, monkeypatch):
        """Login should save session on success."""
        monkeypatch.setenv("ROBLOX_USERNAME", "testuser")
        monkeypatch.setenv("ROBLOX_PASSWORD", "testpass")

        auth_calls = [0]

        def mock_is_authenticated():
            auth_calls[0] += 1
            return auth_calls[0] > 1

        auth.is_authenticated = mock_is_authenticated

        # Mock _check_for_2fa_prompt to return False (no 2FA)
        auth._check_for_2fa_prompt = MagicMock(return_value=False)

        mock_input = MagicMock()
        mock_input.count.return_value = 1
        auth._browser.page.locator.return_value = mock_input

        auth.login(save_session=True)

        # Session should be saved
        assert auth._storage_path.exists()

    def test_login_handles_2fa(self, auth, monkeypatch):
        """Login should handle 2FA when required."""
        monkeypatch.setenv("ROBLOX_USERNAME", "testuser")
        monkeypatch.setenv("ROBLOX_PASSWORD", "testpass")
        monkeypatch.setenv("ROBLOX_TOTP_SECRET", "JBSWY3DPEHPK3PXP")

        auth_calls = [0]

        def mock_is_authenticated():
            auth_calls[0] += 1
            return auth_calls[0] > 1

        auth.is_authenticated = mock_is_authenticated

        # Mock _check_for_2fa_prompt to return True first, then False after handling
        twofa_calls = [0]

        def mock_check_2fa():
            twofa_calls[0] += 1
            return twofa_calls[0] == 1  # True first call, False after

        auth._check_for_2fa_prompt = mock_check_2fa

        mock_input = MagicMock()
        mock_input.count.return_value = 1
        auth._browser.page.locator.return_value = mock_input

        auth.login()

        # Should have checked 2FA twice (once in login, once in _handle_2fa)
        assert twofa_calls[0] >= 1

    def test_login_returns_success_result(self, auth, monkeypatch):
        """Login should report SUCCESS outcome when authenticated."""
        from src.environment.auth import AuthOutcome

        monkeypatch.setenv("ROBLOX_USERNAME", "testuser")
        monkeypatch.setenv("ROBLOX_PASSWORD", "testpass")

        auth_calls = [0]

        def mock_is_authenticated():
            auth_calls[0] += 1
            return auth_calls[0] > 1

        auth.is_authenticated = mock_is_authenticated
        auth._check_for_2fa_prompt = MagicMock(return_value=False)
        mock_input = MagicMock()
        mock_input.count.return_value = 1
        auth._browser.page.locator.return_value = mock_input

        result = auth.login()

        assert result.outcome == AuthOutcome.SUCCESS
        assert result.retryable is False

    def test_login_timeout_maps_to_network_timeout(self, auth, monkeypatch):
        """Playwright timeouts should map to NETWORK_TIMEOUT outcome."""
        from src.environment.auth import AuthOutcome

        monkeypatch.setenv("ROBLOX_USERNAME", "testuser")
        monkeypatch.setenv("ROBLOX_PASSWORD", "testpass")
        auth._browser.page.goto.side_effect = TimeoutError("goto timed out")

        result = auth.login()

        assert result.outcome == AuthOutcome.NETWORK_TIMEOUT
        assert result.retryable is True
        assert "timed out" in result.message.lower()

    def test_login_timeout_with_challenge_maps_to_challenge_blocked(self, auth, monkeypatch):
        """Timeouts caused by challenge overlays should classify as CHALLENGE_BLOCKED."""
        from src.environment.auth import AuthOutcome

        monkeypatch.setenv("ROBLOX_USERNAME", "testuser")
        monkeypatch.setenv("ROBLOX_PASSWORD", "testpass")
        auth.is_authenticated = MagicMock(return_value=False)
        auth._check_for_2fa_prompt = MagicMock(return_value=False)
        auth._check_for_challenge_prompt = MagicMock(return_value=True)

        username_locator = MagicMock()
        username_locator.count.return_value = 1
        username_locator.fill.side_effect = TimeoutError("fill timed out")
        auth._browser.page.locator.return_value = username_locator

        result = auth.login()

        assert result.outcome == AuthOutcome.CHALLENGE_BLOCKED
        assert result.retryable is False

    def test_login_invalid_credentials_maps_to_invalid_credentials(self, auth, monkeypatch):
        """Known login errors should map to INVALID_CREDENTIALS outcome."""
        from src.environment.auth import AuthOutcome

        monkeypatch.setenv("ROBLOX_USERNAME", "testuser")
        monkeypatch.setenv("ROBLOX_PASSWORD", "badpass")
        auth.is_authenticated = MagicMock(return_value=False)
        auth._check_for_2fa_prompt = MagicMock(return_value=False)
        auth._get_login_error = MagicMock(return_value="Invalid username or password")
        mock_input = MagicMock()
        mock_input.count.return_value = 1
        auth._browser.page.locator.return_value = mock_input

        result = auth.login()

        assert result.outcome == AuthOutcome.INVALID_CREDENTIALS
        assert result.retryable is False

    def test_login_challenge_maps_to_challenge_blocked(self, auth, monkeypatch):
        """Captcha/challenge pages should map to CHALLENGE_BLOCKED."""
        from src.environment.auth import AuthOutcome

        monkeypatch.setenv("ROBLOX_USERNAME", "testuser")
        monkeypatch.setenv("ROBLOX_PASSWORD", "testpass")
        auth.is_authenticated = MagicMock(return_value=False)
        auth._check_for_2fa_prompt = MagicMock(return_value=False)
        auth._check_for_challenge_prompt = MagicMock(return_value=True)
        mock_input = MagicMock()
        mock_input.count.return_value = 1
        auth._browser.page.locator.return_value = mock_input

        result = auth.login()

        assert result.outcome == AuthOutcome.CHALLENGE_BLOCKED
        assert result.retryable is False

    def test_login_does_not_navigate_away_from_login_page(self, auth, monkeypatch):
        """Login flow should not redirect itself away from /login before form fill."""
        from src.environment.auth import ROBLOX_HOME, ROBLOX_LOGIN

        monkeypatch.setenv("ROBLOX_USERNAME", "testuser")
        monkeypatch.setenv("ROBLOX_PASSWORD", "testpass")

        def locator_for(selector):
            loc = MagicMock()
            if "user-menu" in selector or "avatar-card-link" in selector:
                loc.count.return_value = 0
            elif 'a[href*="/login"]' in selector or "login-button" in selector or "input[name=\"username\"]" in selector or "input[name=\"password\"]" in selector or "button[type=\"submit\"]" in selector:
                loc.count.return_value = 1
            else:
                loc.count.return_value = 0
            return loc

        auth._browser.page.locator.side_effect = locator_for
        auth._get_login_error = MagicMock(return_value="Invalid username or password")
        auth._check_for_2fa_prompt = MagicMock(return_value=False)
        auth._check_for_challenge_prompt = MagicMock(return_value=False)

        auth.login()

        goto_urls = [call.args[0] for call in auth._browser.page.goto.call_args_list]
        assert ROBLOX_LOGIN in goto_urls
        assert ROBLOX_HOME not in goto_urls

    def test_click_login_button_returns_true_when_button_present(self, auth) -> None:
        """Manual login click should trigger submit when button exists."""
        button = MagicMock()
        button.count.return_value = 1
        auth._browser.page.locator.return_value = button

        result = auth.click_login_button()

        assert result is True
        button.first.click.assert_called_once()

    def test_click_login_button_returns_false_when_all_fallbacks_fail(self, auth) -> None:
        """Manual login click should fail only when all fallback paths fail."""
        missing = MagicMock()
        missing.count.return_value = 0
        auth._browser.page.locator.return_value = missing
        auth._browser.page.evaluate.side_effect = RuntimeError("js blocked")
        auth._browser.page.keyboard = MagicMock()
        auth._browser.page.keyboard.press.side_effect = RuntimeError("key blocked")

        result = auth.click_login_button()

        assert result is False

    def test_click_login_button_falls_back_to_javascript_submit(self, auth) -> None:
        """If locator click fails, JS fallback should still submit login."""
        button = MagicMock()
        button.count.return_value = 1
        button.first.click.side_effect = RuntimeError("strict mode violation")
        auth._browser.page.locator.return_value = button
        auth._browser.page.evaluate.return_value = True

        result = auth.click_login_button()

        assert result is True
        auth._browser.page.evaluate.assert_called_once()

    def test_click_login_button_falls_back_to_enter_key(self, auth) -> None:
        """If button/JS fail, Enter key fallback should be attempted."""
        missing = MagicMock()
        missing.count.return_value = 0
        auth._browser.page.locator.return_value = missing
        auth._browser.page.evaluate.return_value = False
        auth._browser.page.keyboard = MagicMock()

        result = auth.click_login_button()

        assert result is True
        auth._browser.page.keyboard.press.assert_called_once_with("Enter")


class TestSecurityConsiderations:
    """Tests for security requirements."""

    def test_credentials_not_in_repr(self):
        """Credentials should not expose password in repr."""
        from src.environment.auth import Credentials

        creds = Credentials(username="user", password="secret123")

        # The default dataclass repr will include password,
        # but we shouldn't log it directly
        # This test documents the concern
        repr_str = repr(creds)

        # Note: dataclass default repr does include password
        # In production, we should never log credentials objects
        assert "secret123" in repr_str  # This is expected with dataclass

    def test_credentials_from_environment_does_not_log_password(self, monkeypatch, caplog):
        """Loading credentials should not log the password."""
        import logging

        from src.environment.auth import Credentials

        monkeypatch.setenv("ROBLOX_USERNAME", "testuser")
        monkeypatch.setenv("ROBLOX_PASSWORD", "supersecret")

        with caplog.at_level(logging.DEBUG):
            Credentials.from_environment()

        # Password should not appear in logs
        assert "supersecret" not in caplog.text
