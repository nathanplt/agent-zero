"""Tests for runtime error classification and bounded recovery."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.interfaces.vision import VisionError
from src.runtime.recovery import (
    RecoveryPolicy,
    RuntimeErrorClass,
    RuntimeRecoveryCoordinator,
)


class TestRuntimeRecoveryClassification:
    """Error classification should map failures to deterministic classes."""

    def test_classifies_authentication_errors(self) -> None:
        coordinator = RuntimeRecoveryCoordinator(policy=RecoveryPolicy())
        error = RuntimeError("Authentication failed: invalid credentials")

        error_class = coordinator.classify(error)

        assert error_class == RuntimeErrorClass.AUTH

    def test_classifies_capture_errors(self) -> None:
        coordinator = RuntimeRecoveryCoordinator(policy=RecoveryPolicy())
        error = VisionError("Screenshot capture failed")

        error_class = coordinator.classify(error)

        assert error_class == RuntimeErrorClass.CAPTURE


class TestRuntimeRecoveryBudgets:
    """Recovery attempts should be bounded by policy budgets."""

    def test_capture_recovery_uses_restart_then_stops_at_budget(self) -> None:
        policy = RecoveryPolicy(capture_retries=1)
        coordinator = RuntimeRecoveryCoordinator(policy=policy)
        environment = MagicMock()

        first = coordinator.handle(VisionError("capture failed"), environment=environment)
        second = coordinator.handle(VisionError("capture failed"), environment=environment)

        assert first is True
        assert second is False
        environment.restart.assert_called_once()

    def test_action_errors_recover_within_budget(self) -> None:
        policy = RecoveryPolicy(action_retries=2)
        coordinator = RuntimeRecoveryCoordinator(policy=policy)

        first = coordinator.handle(RuntimeError("Action validation failed"))
        second = coordinator.handle(RuntimeError("Action validation failed"))
        third = coordinator.handle(RuntimeError("Action validation failed"))

        assert first is True
        assert second is True
        assert third is False
