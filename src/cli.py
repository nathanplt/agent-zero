"""CLI entrypoint for running Agent Zero gameplay sessions."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from src.actions.backend import InputBackend, NullInputBackend, PlaywrightInputBackend
from src.actions.executor import GameActionExecutor
from src.config.loader import load_config
from src.core.decision import DecisionConfig, DecisionEngine
from src.core.loop import AgentLoop, LoopConfig
from src.core.observation import ObservationPipeline, PipelineConfig
from src.environment.auth import AuthenticationError, Credentials, RobloxAuth
from src.environment.manager import LocalEnvironmentManager
from src.observer.server import create_app
from src.observer.streaming import ActionStreamingService, ScreenStreamingService
from src.vision.capture import ScreenshotCapture
from src.vision.ocr import OCRSystem
from src.vision.ui_detection import UIDetector

if TYPE_CHECKING:
    from src.interfaces.vision import Screenshot

logger = logging.getLogger(__name__)


class ObserverServer(Protocol):
    """Protocol for running observer server handles."""

    def stop(self) -> None:
        """Stop observer server."""
        ...


@dataclass
class _UvicornObserverServer:
    """Background uvicorn server handle."""

    server: Any
    thread: threading.Thread

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5.0)


@dataclass
class AgentRuntime:
    """Runtime wrapper for an assembled agent session."""

    environment: LocalEnvironmentManager
    loop: AgentLoop
    pipeline: ObservationPipeline
    screen_streaming_service: ScreenStreamingService | None = None
    action_streaming_service: ActionStreamingService | None = None
    observer_server: ObserverServer | None = None

    def run_iterations(self, max_iterations: int) -> int:
        """Run the agent for a fixed number of iterations.

        Args:
            max_iterations: Number of iterations to execute.

        Returns:
            Number of completed iterations.
        """
        completed = 0
        while completed < max_iterations:
            observation, decision, action_result = self.loop.run_once()
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
        if self.observer_server is not None:
            self.observer_server.stop()


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


def _warn_roblox_runtime_constraints() -> None:
    """Emit runtime constraints relevant to real Roblox playability."""
    logger.warning(
        "Roblox game launch may require the native Roblox client; browser-only automation can be limited."
    )
    logger.warning(
        "Roblox anti-cheat may block virtualized environments for some sessions."
    )


def _start_observer_server(
    host: str,
    port: int,
    screen_service: ScreenStreamingService,
    action_service: ActionStreamingService,
) -> ObserverServer | None:
    """Start observer FastAPI server in a background thread."""
    try:
        import uvicorn
    except ImportError:  # pragma: no cover - optional dependency
        logger.warning("uvicorn not installed; observer web app disabled")
        return None

    app = create_app(
        streaming_service=screen_service,
        action_streaming_service=action_service,
    )
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="ObserverServer")
    thread.start()
    # Best-effort brief wait for initial bind.
    time.sleep(0.1)
    return _UvicornObserverServer(server=server, thread=thread)


def _create_runtime(args: argparse.Namespace) -> AgentRuntime:
    """Create and initialize a playable runtime from CLI arguments."""
    config = load_config()
    _warn_roblox_runtime_constraints()

    environment = LocalEnvironmentManager(
        headless=bool(args.headless),
        viewport_width=config.environment.display_width,
        viewport_height=config.environment.display_height,
        display=config.environment.virtual_display,
    )
    environment.start()

    if args.game_url:
        environment.navigate(str(args.game_url))

    explicit_credentials = build_credentials_from_args(args)
    if not bool(args.skip_auth):
        # Accessing the browser runtime is needed to perform login flow.
        browser_runtime = environment._browser  # noqa: SLF001
        if browser_runtime is None:
            raise RuntimeError("Browser runtime unavailable for authentication")
        auth = RobloxAuth(browser_runtime)
        try:
            auth.login(credentials=explicit_credentials)
        except AuthenticationError:
            logger.warning("Authentication failed; continuing unauthenticated session")

    capture = ScreenshotCapture(environment_manager=environment, buffer_size=config.vision.buffer_size)
    ocr = OCRSystem()
    ui_detector = UIDetector()

    llm_vision = None
    try:
        from src.vision.llm_vision import LLMVision

        llm_vision = LLMVision(provider=config.llm.provider, model=config.llm.model)
    except Exception as exc:  # pragma: no cover - optional dependency path
        logger.warning("LLM vision unavailable (%s); using local extraction only", exc)

    pipeline = ObservationPipeline(
        capture=capture,
        ocr=ocr,
        ui_detector=ui_detector,
        llm_vision=llm_vision,
        config=PipelineConfig(use_llm=llm_vision is not None),
    )

    decision_engine = DecisionEngine(
        config=DecisionConfig(
            provider=config.llm.provider,
            model=config.llm.model,
            policy_enabled=True,
        )
    )

    backend: InputBackend = NullInputBackend()
    browser_runtime = environment._browser  # noqa: SLF001
    if browser_runtime is not None and browser_runtime.page is not None:
        backend = PlaywrightInputBackend(browser_runtime.page)

    action_executor = GameActionExecutor(backend=backend)

    loop = AgentLoop(
        observation_pipeline=pipeline,
        decision_engine=decision_engine,
        action_executor=action_executor,
        config=LoopConfig(target_rate_hz=float(config.agent.loop_rate), enable_signal_handlers=False),
    )

    screen_service: ScreenStreamingService | None = None
    action_service: ActionStreamingService | None = None
    observer_server: ObserverServer | None = None

    if bool(args.observe):
        observer_host = str(args.observer_host or config.observer.host)
        observer_port = int(args.observer_port or config.observer.port)
        screen_service = ScreenStreamingService(stream_fps=config.observer.stream_fps)
        action_service = ActionStreamingService()
        observer_server = _start_observer_server(
            host=observer_host,
            port=observer_port,
            screen_service=screen_service,
            action_service=action_service,
        )
        logger.info("Observer live page: http://%s:%s/live", observer_host, observer_port)

    return AgentRuntime(
        environment=environment,
        loop=loop,
        pipeline=pipeline,
        screen_streaming_service=screen_service,
        action_streaming_service=action_service,
        observer_server=observer_server,
    )


def run_command(args: argparse.Namespace) -> int:
    """Execute the `run` command."""
    if args.command != "run":
        raise ValueError(f"Unsupported command: {args.command}")

    runtime = _create_runtime(args)
    try:
        runtime.run_iterations(int(args.max_iterations))
        return 0
    finally:
        runtime.shutdown()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint function."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    try:
        if args.command == "run":
            return run_command(args)
        raise ValueError(f"Unsupported command: {args.command}")
    except Exception as exc:
        logger.error("CLI execution failed: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
