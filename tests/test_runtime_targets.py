"""Tests for runtime backend target scaffolding."""

from __future__ import annotations

from src.runtime.targets import RuntimeBackendKind, resolve_runtime_backend


class TestRuntimeBackendSelection:
    """Runtime target resolution should be deterministic and extensible."""

    def test_browser_target_resolves_to_browser_backend(self) -> None:
        selection = resolve_runtime_backend("browser", platform="darwin")

        assert selection.requested == RuntimeBackendKind.BROWSER
        assert selection.effective == RuntimeBackendKind.BROWSER

    def test_native_target_falls_back_to_browser_until_native_backend_ready(self) -> None:
        selection = resolve_runtime_backend("native", platform="darwin")

        assert selection.requested == RuntimeBackendKind.NATIVE
        assert selection.effective == RuntimeBackendKind.BROWSER

    def test_auto_target_prefers_browser_when_native_not_available(self) -> None:
        selection = resolve_runtime_backend("auto", platform="linux")

        assert selection.requested == RuntimeBackendKind.AUTO
        assert selection.effective == RuntimeBackendKind.BROWSER
