"""Unit tests for browser screenshot recovery orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.environment.browser import BrowserRuntime, BrowserRuntimeError


class TestBrowserRecovery:
    """Browser runtime should recover from transient capture failures."""

    def test_screenshot_with_recovery_recovers_after_page_restart(self) -> None:
        runtime = BrowserRuntime(headless=True)
        runtime._page = MagicMock()  # noqa: SLF001
        runtime._page.screenshot.side_effect = [TimeoutError("timeout"), b"ok"]  # noqa: SLF001
        runtime.recover_page = MagicMock()  # type: ignore[method-assign]
        runtime.recover_context = MagicMock()  # type: ignore[method-assign]

        screenshot = runtime.screenshot_with_recovery(
            timeout_ms=100,
            page_retry_budget=1,
            context_retry_budget=1,
        )

        assert screenshot == b"ok"
        runtime.recover_page.assert_called_once()
        runtime.recover_context.assert_not_called()

    def test_screenshot_with_recovery_escalates_to_context_recovery(self) -> None:
        runtime = BrowserRuntime(headless=True)
        runtime._page = MagicMock()  # noqa: SLF001
        runtime._page.screenshot.side_effect = TimeoutError("timeout")  # noqa: SLF001
        runtime.recover_page = MagicMock()  # type: ignore[method-assign]
        runtime.recover_context = MagicMock()  # type: ignore[method-assign]

        with pytest.raises(BrowserRuntimeError):
            runtime.screenshot_with_recovery(
                timeout_ms=100,
                page_retry_budget=1,
                context_retry_budget=1,
            )

        assert runtime.recover_page.call_count >= 1
        assert runtime.recover_context.call_count >= 1

    def test_recover_page_rotates_page_instance(self) -> None:
        runtime = BrowserRuntime(headless=True)
        old_page = MagicMock()
        new_page = MagicMock()
        runtime._page = old_page  # noqa: SLF001
        runtime._context = MagicMock()  # noqa: SLF001
        runtime._context.new_page.return_value = new_page  # noqa: SLF001

        runtime.recover_page()

        old_page.close.assert_called_once()
        assert runtime.page is new_page

    def test_recover_context_requires_active_browser(self) -> None:
        runtime = BrowserRuntime(headless=True)
        runtime._browser = None  # noqa: SLF001

        with pytest.raises(BrowserRuntimeError):
            runtime.recover_context()
