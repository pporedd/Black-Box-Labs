"""
Isomorphic variation generator.

Generates N images with identical parameter values but different random seeds,
producing visually distinct scenes with the same ground truth. This is the
"Bayesian" validation step — testing whether a result is stable across
random layout variations.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image

from generator.state import ModifiableParams, SceneState, calculate_scene
from generator.renderer import render_scene
from config import settings


def generate_isomorphisms(
    params: ModifiableParams,
    n: Optional[int] = None,
    base_seed: int = 0,
) -> list[tuple[SceneState, Image.Image]]:
    """
    Generate N isomorphic scene variations.

    Each variation has identical ModifiableParams but a different seed,
    producing different random positions/rotations while maintaining
    the same ground truth answer.

    Args:
        params: The modifiable chunk values to hold constant.
        n: Number of isomorphisms (defaults to settings.NUM_ISOMORPHISMS).
        base_seed: Starting seed; variations use base_seed + i.

    Returns:
        List of (SceneState, PIL.Image) tuples.
    """
    if n is None:
        n = settings.NUM_ISOMORPHISMS

    results: list[tuple[SceneState, Image.Image]] = []
    for i in range(n):
        scene = calculate_scene(params, seed=base_seed + i)
        image = render_scene(scene)
        results.append((scene, image))

    return results
