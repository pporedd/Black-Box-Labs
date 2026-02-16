"""Stimulus Generator — Parametric HTML → Image pipeline."""

from generator.state import SceneState, calculate_scene, ModifiableParams
from generator.renderer import render_scene
from generator.isomorphic import generate_isomorphisms

__all__ = [
    "SceneState",
    "calculate_scene",
    "ModifiableParams",
    "render_scene",
    "generate_isomorphisms",
]
