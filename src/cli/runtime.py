"""Runtime container used by CLI commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.loop import AgentLoop
from src.core.observation import ObservationPipeline
from src.environment.manager import LocalEnvironmentManager
from src.observer.streaming import ActionStreamingService, ScreenStreamingService
from src.runtime.recovery import RuntimeRecoveryCoordinator

if TYPE_CHECKING:
    from src.interfaces.vision import Screenshot


@dataclass
class AgentRuntime:
    """Runtime wrapper for an assembled agent session."""

    environment: LocalEnvironmentManager
    loop: AgentLoop
    pipeline: ObservationPipeline
    recovery_coordinator: RuntimeRecoveryCoordinator | None = None
    screen_streaming_service: ScreenStreamingService | None = None
    action_streaming_service: ActionStreamingService | None = None
    observer_server: object | None = None

    def run_iterations(self, max_iterations: int) -> int:
        """Run the agent for a fixed number of iterations."""
        completed = 0
        while completed < max_iterations:
            try:
                observation, decision, action_result = self.loop.run_once()
            except Exception as exc:
                if self.recovery_coordinator is None:
                    raise
                should_continue = self.recovery_coordinator.handle(exc, environment=self.environment)
                if not should_continue:
                    raise
                continue

            self._publish_observation(observation.screenshot)
            self._publish_action_event(completed + 1, decision, action_result.success, action_result.error)
            completed += 1
        return completed

    def _publish_observation(self, screenshot: Screenshot) -> None:
        if self.screen_streaming_service is None:
            return
        frame = screenshot.raw_bytes if screenshot.raw_bytes else screenshot.image
        self.screen_streaming_service.push_frame(frame)

    def _publish_action_event(
        self,
        iteration: int,
        decision: object,
        success: bool,
        error: str | None,
    ) -> None:
        if self.action_streaming_service is None:
            return
        action_type = None
        decision_context: dict[str, object] = {}
        if hasattr(decision, "action") and hasattr(decision.action, "type"):
            action_type = getattr(decision.action.type, "value", str(decision.action.type))
        if hasattr(decision, "context") and isinstance(decision.context, dict):
            decision_context = decision.context
        self.action_streaming_service.push_event(
            {
                "event": "action_executed",
                "iteration": iteration,
                "action_type": action_type,
                "success": success,
                "error": error,
                "decision_source": decision_context.get("decision_source"),
                "policy_strategy": decision_context.get("policy_strategy"),
            }
        )

    def shutdown(self) -> None:
        """Shutdown runtime resources."""
        self.loop.stop()
        self.pipeline.close()
        self.environment.stop()
        if self.observer_server is not None and hasattr(self.observer_server, "stop"):
            self.observer_server.stop()
