"""Decision engine for ReAct-style reasoning and action selection.

This module provides the DecisionEngine class that:
- Takes observations and produces decisions using LLM
- Uses ReAct (Reasoning + Acting) prompting
- Includes chain-of-thought reasoning
- Provides confidence-calibrated action selection
- Caches decisions for identical game states

The Decision Engine shares the LLM client approach with the Vision module
but uses different prompts optimized for decision-making.

Example:
    >>> from src.core.decision import DecisionEngine
    >>> from src.core.observation import Observation
    >>>
    >>> engine = DecisionEngine(provider="anthropic")
    >>> decision = engine.decide(observation)
    >>> print(f"Action: {decision.action.type}")
    >>> print(f"Reasoning: {decision.reasoning}")
    >>> print(f"Confidence: {decision.confidence}")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.core.prompts import DecisionPrompts
from src.interfaces.actions import Action, ActionType, Point
from src.models.actions import Action as ModelAction
from src.models.decisions import Decision
from src.strategy.policy import IncrementalPolicyEngine

if TYPE_CHECKING:
    from src.core.observation import Observation

logger = logging.getLogger(__name__)

# Valid LLM providers
VALID_PROVIDERS = {"anthropic", "openai"}

# Default models for decision making (faster/cheaper than vision models)
DEFAULT_MODELS = {
    "anthropic": "claude-3-haiku-20240307",  # Faster for decisions
    "openai": "gpt-4-turbo-preview",
}


class DecisionEngineError(Exception):
    """Error raised when decision engine operations fail."""

    pass


@dataclass
class DecisionConfig:
    """Configuration for the decision engine.

    Attributes:
        provider: LLM provider ("anthropic" or "openai").
        model: Model name (defaults based on provider).
        cache_size: Maximum number of cached decisions.
        max_retries: Maximum retry attempts on API failure.
        retry_delay: Initial delay between retries.
        temperature: LLM temperature (0.0-1.0).
        max_tokens: Maximum tokens in response.
    """

    provider: str = "anthropic"
    model: str | None = None
    cache_size: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0
    temperature: float = 0.3  # Lower for more deterministic decisions
    max_tokens: int = 1024
    policy_enabled: bool = False
    policy_confidence_threshold: float = 0.7


class DecisionEngine:
    """ReAct-style decision engine for game playing.

    This class uses LLM to analyze game observations and decide on
    the best action to take. It follows the ReAct paradigm:
    - Observation: Current game state
    - Thought: Reasoning about what to do
    - Action: Selected action with target
    - Expected: What should happen next

    The engine:
    - Caches decisions for identical game states
    - Provides confidence-calibrated action selection
    - Tracks reasoning traces for debugging
    - Handles LLM errors with retries

    Attributes:
        cache_stats: Dictionary with cache hit/miss statistics.

    Example:
        >>> engine = DecisionEngine(provider="anthropic")
        >>> decision = engine.decide(observation)
        >>> if decision.confidence > 0.7:
        ...     execute(decision.action)
    """

    def __init__(
        self,
        config: DecisionConfig | None = None,
        prompts: DecisionPrompts | None = None,
        api_key: str | None = None,
        policy_engine: IncrementalPolicyEngine | None = None,
    ) -> None:
        """Initialize the decision engine.

        Args:
            config: Engine configuration. Uses defaults if None.
            prompts: Prompt templates. Uses defaults if None.
            api_key: API key for the provider. Falls back to environment.

        Raises:
            ValueError: If provider is not supported.
        """
        self._config = config or DecisionConfig()
        self._prompts = prompts or DecisionPrompts()
        self._api_key = api_key
        self._policy = policy_engine or IncrementalPolicyEngine()

        if self._config.provider not in VALID_PROVIDERS:
            raise ValueError(
                f"Invalid provider: {self._config.provider}. "
                f"Must be one of {VALID_PROVIDERS}"
            )

        # Set default model if not specified
        if self._config.model is None:
            self._config.model = DEFAULT_MODELS.get(
                self._config.provider, "claude-3-haiku-20240307"
            )

        # LRU cache for decisions
        self._cache: OrderedDict[str, Decision] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

        # Recent actions for context
        self._recent_actions: list[dict[str, Any]] = []
        self._max_recent_actions = 10

        logger.debug(
            f"DecisionEngine initialized: provider={self._config.provider}, "
            f"model={self._config.model}, cache_size={self._config.cache_size}"
        )

    @property
    def cache_stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with 'size', 'hits', and 'misses' counts.
        """
        return {
            "size": len(self._cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
        }

    def clear_cache(self) -> None:
        """Clear the decision cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("DecisionEngine cache cleared")

    def decide(
        self,
        observation: Observation,
        goal: str | None = None,
        use_cache: bool = True,
    ) -> Decision:
        """Make a decision based on the current observation.

        Uses ReAct-style prompting to analyze the game state and
        select the best action with confidence scoring.

        Args:
            observation: Current game observation.
            goal: Optional current goal being pursued.
            use_cache: Whether to use cached decisions.

        Returns:
            Decision with action, reasoning, and confidence.

        Raises:
            DecisionEngineError: If decision-making fails.
        """
        start_time = time.time()

        # Check cache
        if use_cache:
            cache_key = self._compute_cache_key(observation)
            if cache_key in self._cache:
                self._cache_hits += 1
                # Move to end (LRU)
                self._cache.move_to_end(cache_key)
                logger.debug(f"Decision cache hit for state {cache_key[:8]}...")
                return self._cache[cache_key]
            self._cache_misses += 1

        # Try deterministic policy first for fast/cheap decisions.
        if self._config.policy_enabled:
            try:
                policy = self._policy.propose(
                    observation=observation,
                    recent_actions=self._recent_actions,
                )
                if policy.confidence >= self._config.policy_confidence_threshold:
                    decision = self._decision_from_policy(policy, observation)
                    if use_cache:
                        self._add_to_cache(cache_key, decision)
                    duration_ms = (time.time() - start_time) * 1000
                    logger.debug(
                        f"Policy decision made in {duration_ms:.1f}ms: "
                        f"{decision.action.type} with confidence {decision.confidence:.2f}"
                    )
                    return decision
            except Exception as e:
                logger.warning(f"Policy decision failed, falling back to LLM: {e}")

        # Build prompt
        prompt = self._prompts.build_react_prompt(
            observation=observation,
            recent_actions=self._recent_actions,
            goal=goal,
        )

        # Call LLM
        try:
            response = self._call_llm(prompt)
        except Exception as e:
            raise DecisionEngineError(f"LLM call failed: {e}") from e

        # Parse response
        try:
            decision = self._parse_response(response, observation)
            decision = decision.model_copy(
                update={
                    "context": {
                        **decision.context,
                        "decision_source": "llm",
                    }
                }
            )
        except Exception as e:
            raise DecisionEngineError(f"Failed to parse decision: {e}") from e

        # Cache the decision
        if use_cache:
            self._add_to_cache(cache_key, decision)

        duration_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"Decision made in {duration_ms:.1f}ms: "
            f"{decision.action.type} with confidence {decision.confidence:.2f}"
        )

        return decision

    def _decision_from_policy(
        self,
        policy: Any,
        observation: Observation,
    ) -> Decision:
        """Convert a policy proposal into a Decision model."""
        return Decision(
            reasoning=policy.rationale,
            action=policy.action,
            confidence=float(policy.confidence),
            expected_outcome=policy.expected_outcome,
            timestamp=datetime.now(),
            alternatives=[],
            context={
                "screen_type": observation.game_state.current_screen.value,
                "observation_confidence": observation.confidence,
                "decision_source": "policy",
                "policy_strategy": policy.strategy,
            },
        )

    def record_action_result(
        self,
        action: Action,
        success: bool,
        outcome: str = "",
    ) -> None:
        """Record the result of an executed action.

        This helps the engine learn from past actions and provide
        better context for future decisions.

        Args:
            action: The action that was executed.
            success: Whether the action succeeded.
            outcome: Description of what happened.
        """
        record = {
            "type": action.action_type.value if hasattr(action, "action_type") else str(action),
            "result": "success" if success else "failed",
            "outcome": outcome,
            "timestamp": datetime.now().isoformat(),
            "strategy": (
                action.parameters.get("strategy")
                if hasattr(action, "parameters") and isinstance(action.parameters, dict)
                else None
            ),
        }

        self._recent_actions.insert(0, record)

        # Keep only recent actions
        if len(self._recent_actions) > self._max_recent_actions:
            self._recent_actions = self._recent_actions[: self._max_recent_actions]

    def _compute_cache_key(self, observation: Observation) -> str:
        """Compute a cache key for an observation.

        The key is based on the game state, not the screenshot,
        to allow caching across visually identical but separately
        captured frames.

        Args:
            observation: Observation to compute key for.

        Returns:
            Cache key string.
        """
        game_state = observation.game_state

        # Build state representation
        state_repr = {
            "screen": game_state.current_screen.value,
            "resources": {
                name: res.amount for name, res in game_state.resources.items()
            },
            "ui_elements": len(observation.ui_elements),
        }

        # Hash the state
        state_str = json.dumps(state_repr, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()

    def _add_to_cache(self, key: str, decision: Decision) -> None:
        """Add a decision to the cache with LRU eviction.

        Args:
            key: Cache key.
            decision: Decision to cache.
        """
        # Remove oldest if at capacity
        while len(self._cache) >= self._config.cache_size:
            self._cache.popitem(last=False)

        self._cache[key] = decision

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with retry logic.

        Args:
            prompt: Prompt to send to the LLM.

        Returns:
            LLM response string.

        Raises:
            DecisionEngineError: If all retries fail.
        """
        config = self._config
        last_error: Exception | None = None

        for attempt in range(config.max_retries):
            try:
                if config.provider == "anthropic":
                    return self._call_anthropic(prompt)
                else:
                    return self._call_openai(prompt)
            except Exception as e:
                last_error = e
                if attempt < config.max_retries - 1:
                    delay = config.retry_delay * (2**attempt)
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}), "
                        f"retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)

        raise DecisionEngineError(
            f"LLM call failed after {config.max_retries} attempts: {last_error}"
        )

    def _call_anthropic(self, prompt: str) -> str:
        """Call the Anthropic API.

        Args:
            prompt: Prompt to send.

        Returns:
            Response text.
        """
        try:
            import anthropic
        except ImportError as e:
            raise DecisionEngineError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from e

        api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise DecisionEngineError(
                "ANTHROPIC_API_KEY not set. Set it in environment or pass to constructor."
            )

        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            system=self._prompts.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        content = message.content[0]
        if hasattr(content, "text"):
            return str(content.text)
        return ""

    def _call_openai(self, prompt: str) -> str:
        """Call the OpenAI API.

        Args:
            prompt: Prompt to send.

        Returns:
            Response text.
        """
        try:
            import openai
        except ImportError as e:
            raise DecisionEngineError(
                "openai package not installed. Run: pip install openai"
            ) from e

        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise DecisionEngineError(
                "OPENAI_API_KEY not set. Set it in environment or pass to constructor."
            )

        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            messages=[
                {"role": "system", "content": self._prompts.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )

        return response.choices[0].message.content or ""

    def _parse_response(self, response: str, observation: Observation) -> Decision:
        """Parse LLM response into a Decision object.

        Args:
            response: Raw LLM response.
            observation: Original observation for context.

        Returns:
            Parsed Decision object.

        Raises:
            ValueError: If response cannot be parsed.
        """
        # Extract JSON from response
        response = response.strip()

        # Handle markdown code blocks
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove first and last lines (```json and ```)
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```") and not in_json:
                    in_json = True
                    continue
                elif line.startswith("```") and in_json:
                    break
                elif in_json:
                    json_lines.append(line)
            response = "\n".join(json_lines)

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}") from e

        # Extract fields
        thought = data.get("thought", "")
        action_data = data.get("action", {})
        confidence = float(data.get("confidence", 0.5))
        expected_outcome = data.get("expected_outcome", "")
        alternatives = data.get("alternatives", [])

        # Build action
        action_type_str = action_data.get("type", "wait")
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            logger.warning(f"Unknown action type: {action_type_str}, defaulting to wait")
            action_type = ActionType.WAIT

        target = None
        target_data = action_data.get("target")
        if target_data and isinstance(target_data, dict):
            target = Point(int(target_data.get("x", 0)), int(target_data.get("y", 0)))

        parameters = action_data.get("parameters", {})
        description = action_data.get("description", "")

        # Create model action (Pydantic)
        from src.models.actions import Point as ModelPoint

        model_target = None
        if target:
            model_target = ModelPoint(x=target.x, y=target.y)

        model_action = ModelAction(
            type=action_type,
            target=model_target,
            parameters=parameters,
            description=description,
        )

        # Create decision
        return Decision(
            reasoning=thought,
            action=model_action,
            confidence=confidence,
            expected_outcome=expected_outcome,
            timestamp=datetime.now(),
            alternatives=alternatives,
            context={
                "screen_type": observation.game_state.current_screen.value,
                "observation_confidence": observation.confidence,
            },
        )

    def create_fallback_decision(
        self,
        observation: Observation,
        reason: str = "Fallback due to error",
    ) -> Decision:
        """Create a safe fallback decision when LLM fails.

        This creates a low-risk "wait" action that allows the game
        to continue while logging the issue.

        Args:
            observation: Current observation.
            reason: Why fallback is being used.

        Returns:
            Safe fallback Decision.
        """
        model_action = ModelAction(
            type=ActionType.WAIT,
            target=None,
            parameters={"duration_ms": 1000},
            description=f"Waiting (fallback: {reason})",
        )

        return Decision(
            reasoning=f"Fallback decision: {reason}. Waiting to observe further.",
            action=model_action,
            confidence=0.3,
            expected_outcome="Game state will be re-evaluated after wait.",
            timestamp=datetime.now(),
            alternatives=[],
            context={
                "fallback": True,
                "reason": reason,
                "screen_type": observation.game_state.current_screen.value,
            },
        )
