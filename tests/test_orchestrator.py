"""Tests for the orchestrator — binary search and transition classification."""

import json
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from orchestrator import BoundarySearchOrchestrator, OrchestratorState
from generator.state import ModifiableParams


class MockEvalResult:
    """Mock evaluation result."""
    def __init__(self, passed: bool, answer: str = "5"):
        self.raw_answer = answer
        self.parsed_answer = answer
        self.ground_truth = "5"
        self.passed = passed
        self.attention_map = None


class TestClassifyRate:
    """Tests for _classify_rate()."""

    def setup_method(self):
        # Create a minimal orchestrator with mocks
        self.evaluator = MagicMock()
        self.agent = MagicMock()
        self.agent.findings = []
        self.agent.max_iterations = 20
        self.agent.should_continue.return_value = True

        self.orch = BoundarySearchOrchestrator.__new__(BoundarySearchOrchestrator)
        self.orch.evaluator = self.evaluator
        self.orch.agent = self.agent
        self.orch.state = OrchestratorState()

    def test_stable(self):
        assert self.orch._classify_rate(1.0) == "stable"
        assert self.orch._classify_rate(0.8) == "stable"

    def test_breakpoint(self):
        assert self.orch._classify_rate(0.0) == "breakpoint"
        assert self.orch._classify_rate(0.2) == "breakpoint"

    def test_brittle(self):
        assert self.orch._classify_rate(0.5) == "brittle"
        assert self.orch._classify_rate(0.6) == "brittle"
        assert self.orch._classify_rate(0.4) == "brittle"


class TestSetVariable:
    """Tests for _execute_set_variable()."""

    def setup_method(self):
        self.evaluator = MagicMock()
        self.agent = MagicMock()
        self.agent.findings = []

        self.orch = BoundarySearchOrchestrator.__new__(BoundarySearchOrchestrator)
        self.orch.evaluator = self.evaluator
        self.orch.agent = self.agent
        self.orch.state = OrchestratorState()

        from session import Session
        self.orch.session = Session(session_id="test")

    def test_set_int_variable(self):
        result = json.loads(self.orch._execute_set_variable("item_count", 10))
        assert result["success"] is True
        assert self.orch.state.params.item_count == 10

    def test_set_float_variable(self):
        result = json.loads(self.orch._execute_set_variable("occlusion_level", 0.5))
        assert result["success"] is True
        assert self.orch.state.params.occlusion_level == 0.5

    def test_set_string_variable(self):
        result = json.loads(self.orch._execute_set_variable("background_color", "#000000"))
        assert result["success"] is True
        assert self.orch.state.params.background_color == "#000000"

    def test_unknown_variable(self):
        result = json.loads(self.orch._execute_set_variable("not_a_variable", 5))
        assert "error" in result
