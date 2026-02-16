"""Scientist Agent — Nemotron-powered hypothesis engine."""

from agent.client import NemotronClient
from agent.scientist import ScientistAgent
from agent.tools import TOOL_DEFINITIONS

__all__ = ["NemotronClient", "ScientistAgent", "TOOL_DEFINITIONS"]
