"""Runtime error classification and bounded recovery policies."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from src.interfaces.vision import VisionError

if TYPE_CHECKING:
    from src.environment.manager import LocalEnvironmentManager

logger = logging.getLogger(__name__)


class RuntimeErrorClass(StrEnum):
    """Typed runtime failure classes used for deterministic recovery."""

    AUTH = "auth"
    NAVIGATION = "navigation"
    SCREENSHOT = "screenshot"
    CAPTURE = "capture"
    ENVIRONMENT = "environment"
    LLM = "llm"
    ACTION = "action"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RecoveryPolicy:
    """Retry budgets for each runtime error class."""

    auth_retries: int = 0
    navigation_retries: int = 1
    screenshot_retries: int = 1
    capture_retries: int = 2
    environment_retries: int = 1
    llm_retries: int = 2
    action_retries: int = 2
    unknown_retries: int = 0

    @classmethod
    def for_profile(cls, profile: str) -> RecoveryPolicy:
        """Resolve a named recovery profile to retry budgets."""
        normalized = profile.strip().lower()
        if normalized == "conservative":
            return cls(
                auth_retries=0,
                navigation_retries=1,
                screenshot_retries=1,
                capture_retries=1,
                environment_retries=1,
                llm_retries=1,
                action_retries=1,
                unknown_retries=0,
            )
        if normalized == "aggressive":
            return cls(
                auth_retries=0,
                navigation_retries=3,
                screenshot_retries=3,
                capture_retries=3,
                environment_retries=2,
                llm_retries=4,
                action_retries=4,
                unknown_retries=1,
            )
        # Default "balanced"
        return cls()


class RuntimeRecoveryCoordinator:
    """Stateful recovery coordinator with bounded retries per error class."""

    def __init__(self, policy: RecoveryPolicy) -> None:
        self._policy = policy
        self._attempts: dict[RuntimeErrorClass, int] = {}

    def classify(self, error: Exception) -> RuntimeErrorClass:
        """Classify an exception into a runtime error class."""
        if isinstance(error, VisionError):
            return RuntimeErrorClass.CAPTURE

        message = str(error).lower()

        if any(token in message for token in ("auth", "login", "credential", "captcha", "challenge")):
            return RuntimeErrorClass.AUTH
        if any(token in message for token in ("navigate", "navigation", "goto", "domcontentloaded")):
            return RuntimeErrorClass.NAVIGATION
        if "screenshot" in message:
            return RuntimeErrorClass.SCREENSHOT
        if any(token in message for token in ("capture", "frame")):
            return RuntimeErrorClass.CAPTURE
        if any(token in message for token in ("display", "xvfb", "environment", "browser start")):
            return RuntimeErrorClass.ENVIRONMENT
        if any(token in message for token in ("llm", "openai", "anthropic", "api key", "api call failed")):
            return RuntimeErrorClass.LLM
        if any(token in message for token in ("action", "validation")):
            return RuntimeErrorClass.ACTION
        return RuntimeErrorClass.UNKNOWN

    def handle(
        self,
        error: Exception,
        environment: LocalEnvironmentManager | None = None,
    ) -> bool:
        """Attempt recovery for an error if budget remains.

        Returns:
            True if the caller should continue execution, False if the
            error budget is exhausted and runtime should abort.
        """
        error_class = self.classify(error)
        budget = self._budget_for(error_class)
        attempts = self._attempts.get(error_class, 0)
        if attempts >= budget:
            logger.error(
                "[RECOVERY] Budget exhausted for %s (%s/%s): %s",
                error_class.value,
                attempts,
                budget,
                error,
            )
            return False

        self._attempts[error_class] = attempts + 1
        logger.warning(
            "[RECOVERY] Attempt %s/%s for %s: %s",
            self._attempts[error_class],
            budget,
            error_class.value,
            error,
        )

        if error_class in {
            RuntimeErrorClass.CAPTURE,
            RuntimeErrorClass.SCREENSHOT,
            RuntimeErrorClass.NAVIGATION,
            RuntimeErrorClass.ENVIRONMENT,
        }:
            if environment is None:
                return False
            try:
                environment.restart()
                return True
            except Exception as restart_error:
                logger.error("[RECOVERY] Environment restart failed: %s", restart_error)
                return False

        return error_class != RuntimeErrorClass.AUTH

    def _budget_for(self, error_class: RuntimeErrorClass) -> int:
        if error_class == RuntimeErrorClass.AUTH:
            return self._policy.auth_retries
        if error_class == RuntimeErrorClass.NAVIGATION:
            return self._policy.navigation_retries
        if error_class == RuntimeErrorClass.SCREENSHOT:
            return self._policy.screenshot_retries
        if error_class == RuntimeErrorClass.CAPTURE:
            return self._policy.capture_retries
        if error_class == RuntimeErrorClass.ENVIRONMENT:
            return self._policy.environment_retries
        if error_class == RuntimeErrorClass.LLM:
            return self._policy.llm_retries
        if error_class == RuntimeErrorClass.ACTION:
            return self._policy.action_retries
        return self._policy.unknown_retries
