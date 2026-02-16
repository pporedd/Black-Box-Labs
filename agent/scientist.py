"""
Scientist Agent — Nemotron-powered hypothesis engine.

Maintains conversation state, proposes hypotheses about VLM boundaries,
and interprets experimental results using Bayesian reasoning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from agent.client import NemotronClient
from agent.tools import TOOL_DEFINITIONS

SYSTEM_PROMPT = """\
You are a rigorous research scientist investigating the behavioral boundaries of a \
Vision-Language Model (VLM). Your goal is to systematically map where the model's \
capabilities break down and identify spurious correlations.

## Your Experimental Framework

You have a parametric scene generator that creates 500×500px images of colored shapes. \
The VLM (Moondream2) is asked to count objects in these images.

## Available Variables (Modifiable Chunks)
- item_count (int, 1-30): Number of target objects
- spacing_px (int, 5-200): Minimum pixel spacing between objects
- item_size_px (int, 10-100): Size of each object in pixels
- occlusion_level (float, 0.0-1.0): How much objects overlap
- distractor_presence (bool): Whether distractor shapes are present
- distractor_count (int, 0-20): Number of distractor shapes
- background_color (hex): Canvas background color
- item_color (hex): Color of target objects
- distractor_color (hex): Color of distractor objects
- shape (circle|square|triangle): Shape of target objects
- contrast_level (float, 0.0-1.0): Contrast between items and background

## Your Available Tools
1. **set_variable(name, value)** — Change a modifiable chunk
2. **run_isomorphism_batch(n=5)** — Run N random variations to get a success rate
3. **analyze_attention()** — Get attention heatmap analysis
4. **binary_search(variable, lo, hi)** — Find exact transition point

## Your Scientific Method
1. **Hypothesize**: Propose which variable to test and why
2. **Test**: Use binary search to find transition points
3. **Validate**: Run isomorphism batches to confirm stability
4. **Classify**: Determine if transitions are:
   - **Breakpoint** (Pass→Fail): True capability limit
   - **Spurious Resolution** (Fail→Pass): Model relied on a correlate, not understanding
   - **Brittle** (fluctuating pass rate): Unreliable capability zone

## Rules
- Always explain your reasoning before calling a tool
- Start with the most informative variable (usually item_count)
- After finding a boundary, test if unrelated variables shift it (spurious correlation test)
- Report confidence levels based on isomorphism pass rates
"""


@dataclass
class HypothesisResult:
    """Result of a single hypothesis test."""
    variable: str
    value: Any
    success_rate: float
    transition_type: str  # "breakpoint" | "spurious_resolution" | "stable" | "brittle"
    confidence: float
    attention_summary: Optional[str] = None
    details: str = ""


@dataclass
class ScientistAgent:
    """
    Autonomous scientist agent that proposes and tests hypotheses
    about VLM behavioral boundaries.
    """
    client: NemotronClient
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    findings: list[HypothesisResult] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 20

    def __post_init__(self):
        # Initialize with system prompt
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def propose_next_action(
        self,
        context: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Ask the agent to propose its next experimental action.

        Args:
            context: Optional additional context (e.g., previous results).

        Returns:
            Dict with 'type' ('tool_call' or 'message') and relevant data.
        """
        if context:
            self.conversation_history.append({
                "role": "user",
                "content": context,
            })
        elif self.iteration == 0:
            self.conversation_history.append({
                "role": "user",
                "content": (
                    "Begin your investigation. The VLM is Moondream2, tasked with "
                    "counting colored shapes. Start by identifying the most informative "
                    "variable to test first, then use your tools to find boundaries."
                ),
            })

        response = self.client.chat(
            messages=self.conversation_history,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # Store assistant response in history
        # Prepare tool calls for history, sanitizing arguments
        history_tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                # 1. Parse/Repair
                try:
                    # Try standard load first
                    args_dict = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    # Fallback to client's repair if available, or empty
                    try:
                        args_dict = self.client._repair_json(tc.function.arguments)
                    except Exception:
                        args_dict = {}

                # 2. Re-serialize to ensures valid JSON string in history
                clean_args_str = json.dumps(args_dict)

                history_tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": clean_args_str,
                    },
                })

        # Store assistant response in history
        self.conversation_history.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": history_tool_calls if history_tool_calls else None,
        })

        self.iteration += 1

        # Parse result
        tool_calls = self.client.extract_tool_calls(response)
        if tool_calls:
            return {
                "type": "tool_call",
                "calls": tool_calls,
                "reasoning": message.content or "",
            }
        else:
            return {
                "type": "message",
                "content": message.content or "",
            }

    def report_tool_result(
        self,
        tool_call_id: str,
        result: str,
    ) -> None:
        """Feed a tool execution result back to the agent."""
        self.conversation_history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        })

    def record_finding(self, finding: HypothesisResult) -> None:
        """Record a validated hypothesis result."""
        self.findings.append(finding)

    def should_continue(self) -> bool:
        """Check if the agent should keep exploring."""
        return self.iteration < self.max_iterations

    def get_summary(self) -> str:
        """Get a summary of all findings so far."""
        if not self.findings:
            return "No findings yet."

        lines = ["## Behavioral Boundary Report\n"]
        for i, f in enumerate(self.findings, 1):
            emoji = {
                "breakpoint": "🔴",
                "spurious_resolution": "🟡",
                "stable": "🟢",
                "brittle": "🟠",
            }.get(f.transition_type, "⚪")

            lines.append(
                f"{i}. {emoji} **{f.variable}={f.value}**: "
                f"{f.transition_type} (rate={f.success_rate:.0%}, "
                f"confidence={f.confidence:.0%})"
            )
            if f.details:
                lines.append(f"   - {f.details}")

        return "\n".join(lines)
