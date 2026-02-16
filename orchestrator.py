"""
Orchestrator — The main experimental loop.

Connects the Scientist Agent, Stimulus Generator, and VLM Evaluator
to autonomously search for behavioral boundaries.

Workflow:
1. Agent proposes variable + action (via Nemotron tool calls)
2. Orchestrator executes the tool (set_variable, binary_search, etc.)
3. Results fed back to the agent for analysis
4. Repeat until agent decides to stop or max iterations reached
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from PIL import Image

from agent.scientist import ScientistAgent, HypothesisResult
from evaluator.vlm import MoondreamEvaluator, EvalResult
from evaluator.attention import extract_attention_heatmap
from generator.state import ModifiableParams, SceneState, calculate_scene
from generator.renderer import render_scene
from generator.isomorphic import generate_isomorphisms
from session import Session, StepRecord
from config import settings


@dataclass
class OrchestratorState:
    """Mutable state of the orchestration loop."""
    params: ModifiableParams = field(default_factory=ModifiableParams)
    latest_image: Optional[Image.Image] = None
    latest_scene: Optional[SceneState] = None
    latest_eval: Optional[EvalResult] = None
    latest_attention_map: Optional[np.ndarray] = None
    is_running: bool = False
    is_paused: bool = False


class BoundarySearchOrchestrator:
    """
    Main orchestrator that connects all components.

    Can be used programmatically or driven by the Streamlit UI.
    """

    def __init__(
        self,
        evaluator: MoondreamEvaluator,
        agent: ScientistAgent,
        session_id: Optional[str] = None,
    ):
        self.evaluator = evaluator
        self.agent = agent
        self.session = Session(session_id=session_id or str(uuid.uuid4())[:8])
        self.state = OrchestratorState()

        # Store params in session
        self.session.current_params = self.state.params.to_dict()

    # ── Tool Executors ───────────────────────────────────────────────

    def _execute_set_variable(self, name: str, value: Any) -> str:
        """Execute set_variable tool call."""
        params_dict = self.state.params.to_dict()

        # Type coercion
        field_type = type(params_dict.get(name))
        if field_type == int:
            value = int(value)
        elif field_type == float:
            value = float(value)
        elif field_type == bool:
            value = bool(value)

        if name not in params_dict:
            return json.dumps({"error": f"Unknown variable: {name}"})

        setattr(self.state.params, name, value)
        self.session.current_params = self.state.params.to_dict()

        return json.dumps({
            "success": True,
            "variable": name,
            "new_value": value,
            "all_params": self.state.params.to_dict(),
        })

    def _execute_isomorphism_batch(self, n: int = 5) -> str:
        """Execute run_isomorphism_batch tool call."""
        results_data = []
        pass_count = 0

        variations = generate_isomorphisms(self.state.params, n=n)

        for i, (scene, image) in enumerate(variations):
            eval_result = self.evaluator.evaluate(
                image, scene.question, scene.ground_truth_answer
            )
            if eval_result.passed:
                pass_count += 1

            results_data.append({
                "seed": scene.seed,
                "answer": eval_result.raw_answer,
                "parsed": eval_result.parsed_answer,
                "ground_truth": eval_result.ground_truth,
                "passed": eval_result.passed,
            })

            # Keep latest for UI
            self.state.latest_image = image
            self.state.latest_scene = scene
            self.state.latest_eval = eval_result

        success_rate = pass_count / n if n > 0 else 0.0

        # Log step
        step = StepRecord(
            step_id=self.session.next_step_id(),
            timestamp=time.time(),
            variable_name=self._get_last_changed_variable(),
            variable_value=self._get_last_changed_value(),
            success_rate=success_rate,
            num_isomorphisms=n,
            transition_type=self._classify_rate(success_rate),
            individual_results=results_data,
        )
        self.session.add_step(step)

        return json.dumps({
            "success_rate": success_rate,
            "pass_count": pass_count,
            "total": n,
            "transition_type": step.transition_type,
            "individual_results": results_data,
        })

    def _execute_analyze_attention(self) -> str:
        """Execute analyze_attention tool call."""
        # Generate a single scene and evaluate with attention
        scene = calculate_scene(self.state.params, seed=42)
        image = render_scene(scene)

        eval_result = self.evaluator.evaluate_with_attention(
            image, scene.question, scene.ground_truth_answer
        )

        self.state.latest_image = image
        self.state.latest_scene = scene
        self.state.latest_eval = eval_result
        self.state.latest_attention_map = eval_result.attention_map

        # Summarize attention distribution
        summary = self._summarize_attention(eval_result.attention_map, image.size)

        return json.dumps({
            "has_attention_map": eval_result.attention_map is not None,
            "summary": summary,
            "answer": eval_result.raw_answer,
            "passed": eval_result.passed,
        })

    def _execute_binary_search(
        self,
        variable: str,
        lo: float,
        hi: float,
    ) -> str:
        """
        Binary search for behavioral transition point.

        Tests midpoints between lo and hi, evaluating the VLM at each,
        until the transition boundary is found (within integer precision
        for int variables, or 0.05 precision for floats).
        """
        params_dict = self.state.params.to_dict()
        is_int = isinstance(params_dict.get(variable), int)
        precision = 1 if is_int else 0.05

        lo_pass = self._quick_eval(variable, lo)
        hi_pass = self._quick_eval(variable, hi)

        search_history = [
            {"value": lo, "passed": lo_pass},
            {"value": hi, "passed": hi_pass},
        ]

        if lo_pass == hi_pass:
            return json.dumps({
                "boundary_found": False,
                "message": f"No transition found: both lo={lo} and hi={hi} {'pass' if lo_pass else 'fail'}",
                "history": search_history,
            })

        # Binary search
        iterations = 0
        while (hi - lo) > precision and iterations < 30:
            mid = (lo + hi) / 2
            if is_int:
                mid = int(round(mid))
            mid_pass = self._quick_eval(variable, mid)
            search_history.append({"value": mid, "passed": mid_pass})

            if mid_pass == lo_pass:
                lo = mid
            else:
                hi = mid
            iterations += 1

        boundary = (lo + hi) / 2
        if is_int:
            boundary = int(round(boundary))

        transition_type = "breakpoint" if lo_pass else "spurious_resolution"

        # Set the variable to the boundary value for further testing
        setattr(self.state.params, variable, int(boundary) if is_int else boundary)

        return json.dumps({
            "boundary_found": True,
            "boundary_value": boundary,
            "transition_type": transition_type,
            "direction": "pass→fail" if lo_pass else "fail→pass",
            "search_history": search_history,
            "iterations": iterations,
        })

    def _quick_eval(self, variable: str, value: Any) -> bool:
        """Quick single-shot evaluation at a specific variable value."""
        old_value = getattr(self.state.params, variable)
        setattr(self.state.params, variable, int(value) if isinstance(old_value, int) else value)

        scene = calculate_scene(self.state.params, seed=0)
        image = render_scene(scene)
        result = self.evaluator.evaluate(image, scene.question, scene.ground_truth_answer)

        setattr(self.state.params, variable, old_value)  # Restore
        return result.passed

    # ── Main Loop ────────────────────────────────────────────────────

    def step(self, context: Optional[str] = None) -> dict[str, Any]:
        """
        Execute one step of the orchestration loop.

        Returns a dict describing what happened (for the UI to display).
        """
        # Get agent's next action
        action = self.agent.propose_next_action(context)

        if action["type"] == "message":
            return {
                "type": "message",
                "content": action["content"],
                "done": not self.agent.should_continue(),
            }

        # Execute tool calls
        results = []
        for call in action["calls"]:
            result = self._dispatch_tool(call["name"], call["arguments"])
            self.agent.report_tool_result(call["id"], result)
            results.append({
                "tool": call["name"],
                "args": call["arguments"],
                "result": json.loads(result),
            })

        return {
            "type": "tool_execution",
            "reasoning": action.get("reasoning", ""),
            "tool_results": results,
            "state": {
                "params": self.state.params.to_dict(),
                "has_image": self.state.latest_image is not None,
                "has_attention": self.state.latest_attention_map is not None,
            },
            "done": not self.agent.should_continue(),
        }

    def _dispatch_tool(self, name: str, args: dict) -> str:
        """Dispatch a tool call to the appropriate executor."""
        if name == "set_variable":
            return self._execute_set_variable(args["name"], args["value"])
        elif name == "run_isomorphism_batch":
            return self._execute_isomorphism_batch(args.get("n", 5))
        elif name == "analyze_attention":
            return self._execute_analyze_attention()
        elif name == "binary_search":
            return self._execute_binary_search(
                args["variable"], args["lo"], args["hi"]
            )
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    def run(self, max_steps: Optional[int] = None) -> None:
        """
        Run the full orchestration loop until the agent stops.

        For use as a headless/batch process (Streamlit uses step() instead).
        """
        self.state.is_running = True
        steps = 0
        max_steps = max_steps or self.agent.max_iterations

        while self.agent.should_continue() and steps < max_steps:
            if self.state.is_paused:
                time.sleep(0.1)
                continue

            result = self.step()
            steps += 1

            if result.get("done"):
                break

        self.state.is_running = False
        self.session.save_summary()

    # ── Helpers ──────────────────────────────────────────────────────

    def _get_last_changed_variable(self) -> str:
        """Get the name of the most recently changed variable."""
        # Compare current params to session's stored params
        for key, val in self.state.params.to_dict().items():
            stored = self.session.current_params.get(key)
            if stored != val:
                return key
        return "item_count"  # Default

    def _get_last_changed_value(self) -> Any:
        var = self._get_last_changed_variable()
        return getattr(self.state.params, var)

    def _classify_rate(self, rate: float) -> str:
        """Classify a success rate into a transition type."""
        if rate >= 0.8:
            return "stable"
        elif rate <= 0.2:
            return "breakpoint"
        else:
            return "brittle"

    @staticmethod
    def _summarize_attention(
        attention_map: Optional[np.ndarray],
        image_size: tuple[int, int],
    ) -> str:
        """Generate a textual summary of the attention distribution."""
        if attention_map is None:
            return "Attention map unavailable (model architecture may not support extraction)."

        h, w = attention_map.shape

        # Divide into quadrants
        mid_h, mid_w = h // 2, w // 2
        quadrants = {
            "top-left": attention_map[:mid_h, :mid_w].mean(),
            "top-right": attention_map[:mid_h, mid_w:].mean(),
            "bottom-left": attention_map[mid_h:, :mid_w].mean(),
            "bottom-right": attention_map[mid_h:, mid_w:].mean(),
        }

        # Center vs periphery
        margin = h // 4
        center = attention_map[margin:h-margin, margin:w-margin].mean()
        periphery = (attention_map.mean() * h * w - center * (h - 2*margin) * (w - 2*margin))
        periphery /= max(h * w - (h - 2*margin) * (w - 2*margin), 1)

        max_quadrant = max(quadrants, key=quadrants.get)
        concentration = max(quadrants.values()) / (sum(quadrants.values()) / 4 + 1e-8)

        if concentration > 2.0:
            pattern = f"highly concentrated in {max_quadrant}"
        elif center > periphery * 1.5:
            pattern = "focused on center"
        elif periphery > center * 1.5:
            pattern = "diffuse toward edges"
        else:
            pattern = "relatively uniform"

        return (
            f"Attention is {pattern}. "
            f"Center intensity: {center:.2f}, Periphery: {periphery:.2f}. "
            f"Strongest quadrant: {max_quadrant} ({quadrants[max_quadrant]:.2f})."
        )
