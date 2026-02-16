"""
Session logging — JSON-lines format for experiment history.

Records every step of the boundary search for analysis and resumability.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from config import settings


@dataclass
class StepRecord:
    """A single experimental step."""
    step_id: int
    timestamp: float
    variable_name: str
    variable_value: Any
    success_rate: float
    num_isomorphisms: int
    transition_type: str  # "breakpoint" | "spurious_resolution" | "stable" | "brittle"
    agent_reasoning: str = ""
    attention_summary: Optional[str] = None
    individual_results: list[dict] = field(default_factory=list)


@dataclass
class Session:
    """Experiment session with full history."""
    session_id: str
    started_at: float = field(default_factory=time.time)
    steps: list[StepRecord] = field(default_factory=list)
    current_params: dict[str, Any] = field(default_factory=dict)
    agent_findings: list[dict] = field(default_factory=list)

    @property
    def log_path(self) -> Path:
        return settings.SESSIONS_DIR / f"{self.session_id}.jsonl"

    @property
    def summary_path(self) -> Path:
        return settings.SESSIONS_DIR / f"{self.session_id}_summary.json"

    def add_step(self, step: StepRecord) -> None:
        """Add a step and append to the log file."""
        self.steps.append(step)
        self._append_to_log(step)

    def _append_to_log(self, step: StepRecord) -> None:
        """Append a single step to the JSONL log."""
        settings.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(step), default=str) + "\n")

    def save_summary(self) -> None:
        """Save a full session summary as JSON."""
        summary = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "total_steps": len(self.steps),
            "current_params": self.current_params,
            "findings": self.agent_findings,
            "boundary_map": self._build_boundary_map(),
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

    def _build_boundary_map(self) -> dict[str, list[dict]]:
        """Build a per-variable boundary map from steps."""
        boundaries: dict[str, list[dict]] = {}
        for step in self.steps:
            if step.variable_name not in boundaries:
                boundaries[step.variable_name] = []
            boundaries[step.variable_name].append({
                "value": step.variable_value,
                "success_rate": step.success_rate,
                "transition_type": step.transition_type,
            })
        return boundaries

    @classmethod
    def load(cls, session_id: str) -> Session:
        """Load a session from its JSONL log."""
        log_path = settings.SESSIONS_DIR / f"{session_id}.jsonl"
        session = cls(session_id=session_id)
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    session.steps.append(StepRecord(**data))
        return session

    def get_boundary_data(self, variable: str) -> list[tuple[Any, float]]:
        """
        Get (value, success_rate) pairs for a specific variable.
        Useful for plotting the boundary histogram.
        """
        return [
            (s.variable_value, s.success_rate)
            for s in self.steps
            if s.variable_name == variable
        ]

    def next_step_id(self) -> int:
        return len(self.steps) + 1
