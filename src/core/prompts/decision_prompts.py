"""Decision prompt templates for the agent.

This module contains the prompt templates used by the DecisionEngine
for ReAct-style reasoning and action selection.

Prompt versions are tracked for debugging and A/B testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.observation import Observation
    from src.models.game_state import GameState

# Prompt version for tracking
PROMPT_VERSION = "1.0.0"

SYSTEM_PROMPT = """You are an expert AI game player agent. Your goal is to play incremental/idle games optimally.

You follow the ReAct (Reasoning + Acting) framework:
1. OBSERVE: Analyze the current game state
2. THINK: Reason about what to do next
3. ACT: Select the best action
4. EXPECT: Predict what should happen

You make decisions based on:
- Maximizing resource generation
- Efficient upgrade paths
- Long-term strategy over short-term gains
- Avoiding wasteful actions

Always provide your response in the exact JSON format specified."""

REACT_PROMPT_TEMPLATE = """## Current Observation

**Screen Type**: {screen_type}
**Timestamp**: {timestamp}

### Resources
{resources_section}

### UI Elements
{ui_elements_section}

### Available Actions
The following action types are available:
- click: Click on a UI element or coordinate
- double_click: Double-click on a location
- type: Type text into an input field
- key_press: Press a single key (enter, escape, tab, etc.)
- key_combo: Press key combination (ctrl+a, etc.)
- scroll: Scroll up/down/left/right
- wait: Wait for a specified duration
- move: Move mouse to a location

{recent_actions_section}

## Your Task

Analyze the current game state and decide the best action to take.
Consider:
1. What resources are available?
2. What can be upgraded or purchased?
3. What will maximize progress?
4. Is there anything time-sensitive?

Respond with a JSON object in this exact format:
{{
    "thought": "Your reasoning about the current situation and what to do",
    "action": {{
        "type": "click|double_click|type|key_press|key_combo|scroll|wait|move",
        "target": {{"x": 100, "y": 200}} or null,
        "parameters": {{"key": "value"}} or {{}},
        "description": "Human-readable description of the action"
    }},
    "confidence": 0.85,
    "expected_outcome": "What should happen after this action",
    "alternatives": [
        {{"action_type": "...", "reason": "Why this wasn't chosen"}}
    ]
}}

Confidence should be:
- 0.9-1.0: Obvious best action, no ambiguity
- 0.7-0.9: Good action, minor alternatives exist
- 0.5-0.7: Reasonable action, significant uncertainty
- 0.3-0.5: Guessing, multiple valid options
- 0.0-0.3: Very uncertain, need more information

Respond with valid JSON only, no additional text."""


@dataclass
class DecisionPrompts:
    """Manager for decision prompt templates.

    Provides methods to build prompts for the decision engine
    with proper formatting of game state information.

    Attributes:
        version: Prompt version string.
        system_prompt: System prompt for the LLM.
        react_template: ReAct prompt template.
    """

    version: str = PROMPT_VERSION
    system_prompt: str = SYSTEM_PROMPT
    react_template: str = REACT_PROMPT_TEMPLATE

    def build_react_prompt(
        self,
        observation: Observation,
        recent_actions: list[dict[str, Any]] | None = None,
        goal: str | None = None,
    ) -> str:
        """Build a ReAct-style decision prompt.

        Args:
            observation: Current observation with game state.
            recent_actions: List of recent actions taken (for context).
            goal: Current goal being pursued (optional).

        Returns:
            Formatted prompt string.
        """
        game_state = observation.game_state

        # Format resources section
        resources_section = self._format_resources(game_state)

        # Format UI elements section
        ui_elements_section = self._format_ui_elements(observation)

        # Format recent actions section
        recent_actions_section = self._format_recent_actions(recent_actions)

        # Build prompt
        prompt = self.react_template.format(
            screen_type=game_state.current_screen.value,
            timestamp=observation.timestamp.isoformat(),
            resources_section=resources_section,
            ui_elements_section=ui_elements_section,
            recent_actions_section=recent_actions_section,
        )

        # Add goal if specified
        if goal:
            prompt = f"**Current Goal**: {goal}\n\n" + prompt

        return prompt

    def _format_resources(self, game_state: GameState) -> str:
        """Format resources for the prompt.

        Args:
            game_state: Current game state.

        Returns:
            Formatted resources string.
        """
        if not game_state.resources:
            return "No resources detected."

        lines = []
        for name, resource in game_state.resources.items():
            line = f"- **{name.title()}**: {self._format_number(resource.amount)}"
            if resource.max_amount:
                line += f" / {self._format_number(resource.max_amount)}"
            if resource.rate:
                line += f" (+{self._format_number(resource.rate)}/s)"
            lines.append(line)

        return "\n".join(lines)

    def _format_ui_elements(self, observation: Observation) -> str:
        """Format UI elements for the prompt.

        Args:
            observation: Current observation.

        Returns:
            Formatted UI elements string.
        """
        elements = observation.ui_elements
        if not elements:
            return "No clickable UI elements detected."

        lines = []
        for i, elem in enumerate(elements[:20], 1):  # Limit to top 20
            label = elem.label or "unlabeled"
            lines.append(
                f"{i}. [{elem.element_type}] \"{label}\" "
                f"at ({elem.x}, {elem.y}) size {elem.width}x{elem.height}"
            )

        if len(elements) > 20:
            lines.append(f"... and {len(elements) - 20} more elements")

        return "\n".join(lines)

    def _format_recent_actions(
        self, recent_actions: list[dict[str, Any]] | None
    ) -> str:
        """Format recent actions for context.

        Args:
            recent_actions: List of recent actions.

        Returns:
            Formatted recent actions string.
        """
        if not recent_actions:
            return "### Recent Actions\nNo recent actions."

        lines = ["### Recent Actions (most recent first)"]
        for i, action in enumerate(recent_actions[:5], 1):  # Last 5 actions
            action_type = action.get("type", "unknown")
            result = action.get("result", "unknown")
            lines.append(f"{i}. {action_type}: {result}")

        return "\n".join(lines)

    @staticmethod
    def _format_number(value: float) -> str:
        """Format a number with K/M/B suffixes.

        Args:
            value: Number to format.

        Returns:
            Formatted string.
        """
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}B"
        elif value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        elif value >= 1_000:
            return f"{value / 1_000:.2f}K"
        elif value >= 100:
            return f"{value:.0f}"
        else:
            return f"{value:.2f}"
