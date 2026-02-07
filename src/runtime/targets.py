"""Runtime backend target selection and scaffolding."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from src.environment.manager import LocalEnvironmentManager

logger = logging.getLogger(__name__)


class RuntimeBackendKind(StrEnum):
    """Supported runtime backend targets."""

    AUTO = "auto"
    BROWSER = "browser"
    NATIVE = "native"


@dataclass(frozen=True)
class EnvironmentCapabilities:
    """Detected runtime capabilities for backend selection."""

    platform: str
    native_available: bool


class RuntimeBackend:
    """Abstract runtime backend for environment creation."""

    def __init__(self, kind: RuntimeBackendKind) -> None:
        self.kind = kind

    def create_environment(
        self,
        *,
        headless: bool,
        viewport_width: int,
        viewport_height: int,
        display: str,
        storage_state_path: Path | None,
        use_virtual_display: bool,
    ) -> LocalEnvironmentManager:
        """Create a runtime-specific environment manager."""
        raise NotImplementedError


class BrowserRuntimeBackend(RuntimeBackend):
    """Browser-first runtime backend."""

    def __init__(self) -> None:
        super().__init__(RuntimeBackendKind.BROWSER)

    def create_environment(
        self,
        *,
        headless: bool,
        viewport_width: int,
        viewport_height: int,
        display: str,
        storage_state_path: Path | None,
        use_virtual_display: bool,
    ) -> LocalEnvironmentManager:
        return LocalEnvironmentManager(
            headless=headless,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            display=display,
            storage_state_path=storage_state_path,
            use_virtual_display=use_virtual_display,
        )


class NativeRuntimeBackend(RuntimeBackend):
    """Native backend scaffold.

    For now, this delegates to the browser backend until a true native adapter
    is implemented.
    """

    def __init__(self) -> None:
        super().__init__(RuntimeBackendKind.NATIVE)

    def create_environment(
        self,
        *,
        headless: bool,
        viewport_width: int,
        viewport_height: int,
        display: str,
        storage_state_path: Path | None,
        use_virtual_display: bool,
    ) -> LocalEnvironmentManager:
        logger.warning(
            "[ENV] Native runtime backend scaffold is not yet implemented; using browser backend."
        )
        return BrowserRuntimeBackend().create_environment(
            headless=headless,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            display=display,
            storage_state_path=storage_state_path,
            use_virtual_display=use_virtual_display,
        )


@dataclass(frozen=True)
class RuntimeBackendSelection:
    """Resolved runtime backend selection details."""

    requested: RuntimeBackendKind
    effective: RuntimeBackendKind
    backend: RuntimeBackend
    capabilities: EnvironmentCapabilities


def _detect_capabilities(platform: str) -> EnvironmentCapabilities:
    """Detect capabilities for runtime backend choice."""
    # Native control path is a planned capability and currently disabled.
    native_available = False
    return EnvironmentCapabilities(platform=platform, native_available=native_available)


def resolve_runtime_backend(
    runtime_target: str | RuntimeBackendKind,
    *,
    platform: str,
) -> RuntimeBackendSelection:
    """Resolve requested runtime target to an effective backend."""
    requested = (
        runtime_target
        if isinstance(runtime_target, RuntimeBackendKind)
        else RuntimeBackendKind(str(runtime_target).lower())
    )
    capabilities = _detect_capabilities(platform)

    if requested == RuntimeBackendKind.BROWSER:
        return RuntimeBackendSelection(
            requested=requested,
            effective=RuntimeBackendKind.BROWSER,
            backend=BrowserRuntimeBackend(),
            capabilities=capabilities,
        )

    if requested == RuntimeBackendKind.NATIVE:
        effective = RuntimeBackendKind.NATIVE if capabilities.native_available else RuntimeBackendKind.BROWSER
        if effective == RuntimeBackendKind.BROWSER:
            logger.warning("[ENV] Native runtime target requested but not available; falling back to browser.")
        return RuntimeBackendSelection(
            requested=requested,
            effective=effective,
            backend=(NativeRuntimeBackend() if effective == RuntimeBackendKind.NATIVE else BrowserRuntimeBackend()),
            capabilities=capabilities,
        )

    # AUTO: prefer native when available, otherwise browser.
    effective = RuntimeBackendKind.NATIVE if capabilities.native_available else RuntimeBackendKind.BROWSER
    return RuntimeBackendSelection(
        requested=requested,
        effective=effective,
        backend=(NativeRuntimeBackend() if effective == RuntimeBackendKind.NATIVE else BrowserRuntimeBackend()),
        capabilities=capabilities,
    )
