"""
Scene state calculator — the ground-truth engine.

Python defines the exact visual state BEFORE rendering, providing an
indisputable answer key without needing secondary vision models.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ItemState:
    """State of a single visual item in the scene."""
    x: int
    y: int
    width: int
    height: int
    color: str
    shape: str  # "circle" | "square" | "triangle"
    z_index: int = 0
    rotation_deg: float = 0.0


@dataclass
class ModifiableParams:
    """
    The independent variables exposed to the Scientist Agent.
    These are the 'Modifiable Chunks' that get swept during search.
    """
    item_count: int = 5
    spacing_px: int = 40
    item_size_px: int = 40
    occlusion_level: float = 0.0      # 0.0 = no overlap, 1.0 = full overlap
    distractor_presence: bool = False
    distractor_count: int = 0
    background_color: str = "#FFFFFF"
    item_color: str = "#E74C3C"        # red
    distractor_color: str = "#BDC3C7"  # grey
    shape: str = "circle"              # "circle" | "square" | "triangle"
    contrast_level: float = 1.0        # 1.0 = full, 0.0 = no contrast

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_count": self.item_count,
            "spacing_px": self.spacing_px,
            "item_size_px": self.item_size_px,
            "occlusion_level": self.occlusion_level,
            "distractor_presence": self.distractor_presence,
            "distractor_count": self.distractor_count,
            "background_color": self.background_color,
            "item_color": self.item_color,
            "distractor_color": self.distractor_color,
            "shape": self.shape,
            "contrast_level": self.contrast_level,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModifiableParams:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SceneState:
    """Complete scene state — serves as the absolute ground truth."""
    params: ModifiableParams
    seed: int
    items: list[ItemState] = field(default_factory=list)
    distractors: list[ItemState] = field(default_factory=list)
    ground_truth_answer: str = ""
    question: str = ""
    canvas_width: int = 500
    canvas_height: int = 500


def _place_items_with_spacing(
    count: int,
    size: int,
    spacing: int,
    canvas_w: int,
    canvas_h: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Place items with minimum spacing, using rejection sampling."""
    margin = size
    positions: list[tuple[int, int]] = []
    max_attempts = count * 200

    for _ in range(max_attempts):
        if len(positions) >= count:
            break
        x = rng.randint(margin, canvas_w - margin)
        y = rng.randint(margin, canvas_h - margin)

        # Check spacing constraint
        too_close = False
        for px, py in positions:
            dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if dist < spacing:
                too_close = True
                break
        if not too_close:
            positions.append((x, y))

    # If rejection sampling couldn't place all items (tight spacing),
    # fall back to grid layout
    if len(positions) < count:
        positions = _grid_fallback(count, size, canvas_w, canvas_h, rng)

    return positions[:count]


def _grid_fallback(
    count: int,
    size: int,
    canvas_w: int,
    canvas_h: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Deterministic grid layout as a fallback."""
    import math

    cols = max(1, int(math.ceil(math.sqrt(count))))
    rows = max(1, int(math.ceil(count / cols)))
    cell_w = (canvas_w - size) // max(cols, 1)
    cell_h = (canvas_h - size) // max(rows, 1)
    positions = []
    for i in range(count):
        col = i % cols
        row = i // cols
        x = size // 2 + col * cell_w + rng.randint(-5, 5)
        y = size // 2 + row * cell_h + rng.randint(-5, 5)
        positions.append((x, y))
    return positions


def _apply_occlusion(
    positions: list[tuple[int, int]],
    occlusion_level: float,
    size: int,
) -> list[tuple[int, int]]:
    """Shift items closer together to simulate occlusion."""
    if occlusion_level <= 0.0 or len(positions) < 2:
        return positions

    # Pull items toward the centroid proportional to occlusion_level
    cx = sum(p[0] for p in positions) / len(positions)
    cy = sum(p[1] for p in positions) / len(positions)

    result = []
    for x, y in positions:
        dx = cx - x
        dy = cy - y
        new_x = int(x + dx * occlusion_level)
        new_y = int(y + dy * occlusion_level)
        result.append((new_x, new_y))
    return result


def calculate_scene(params: ModifiableParams, seed: int = 0) -> SceneState:
    """
    Calculate the full scene state from parameters and seed.

    This is the GROUND TRUTH engine. Everything about the scene's visual
    state is determined here, before any rendering occurs.
    """
    rng = random.Random(seed)

    canvas_w = 500
    canvas_h = 500

    # ── Place primary items ──────────────────────────────────────────
    positions = _place_items_with_spacing(
        count=params.item_count,
        size=params.item_size_px,
        spacing=params.spacing_px,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        rng=rng,
    )
    positions = _apply_occlusion(positions, params.occlusion_level, params.item_size_px)

    # Blend item color toward background based on contrast
    item_color = params.item_color
    if params.contrast_level < 1.0:
        item_color = _blend_color(
            params.item_color, params.background_color, params.contrast_level
        )

    items = []
    for i, (x, y) in enumerate(positions):
        rotation = rng.uniform(0, 360)
        items.append(ItemState(
            x=x,
            y=y,
            width=params.item_size_px,
            height=params.item_size_px,
            color=item_color,
            shape=params.shape,
            z_index=i,
            rotation_deg=rotation,
        ))

    # ── Place distractors ────────────────────────────────────────────
    distractors = []
    if params.distractor_presence and params.distractor_count > 0:
        d_positions = _place_items_with_spacing(
            count=params.distractor_count,
            size=params.item_size_px,
            spacing=max(params.spacing_px // 2, params.item_size_px),
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            rng=rng,
        )
        for i, (x, y) in enumerate(d_positions):
            rotation = rng.uniform(0, 360)
            distractors.append(ItemState(
                x=x,
                y=y,
                width=params.item_size_px,
                height=params.item_size_px,
                color=params.distractor_color,
                shape="square" if params.shape == "circle" else "circle",
                z_index=100 + i,
                rotation_deg=rotation,
            ))

    # ── Ground truth ─────────────────────────────────────────────────
    ground_truth = str(params.item_count)
    question = f"How many {params.shape}s are in the image?"

    return SceneState(
        params=params,
        seed=seed,
        items=items,
        distractors=distractors,
        ground_truth_answer=ground_truth,
        question=question,
        canvas_width=canvas_w,
        canvas_height=canvas_h,
    )


def _blend_color(fg_hex: str, bg_hex: str, alpha: float) -> str:
    """Blend foreground toward background by alpha (1.0 = full fg)."""
    fg = _hex_to_rgb(fg_hex)
    bg = _hex_to_rgb(bg_hex)
    blended = tuple(int(f * alpha + b * (1 - alpha)) for f, b in zip(fg, bg))
    return f"#{blended[0]:02x}{blended[1]:02x}{blended[2]:02x}"


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' to (R, G, B) tuple."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
