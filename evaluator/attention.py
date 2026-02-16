"""
Attention map extraction utilities.

Registers forward hooks on the vision encoder's attention layers
to capture attention weights for heatmap visualization.

Uses attn_implementation="eager" + output_attentions=True to make
the attention matrices available (at the cost of speed/VRAM).
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def extract_attention_heatmap(
    attention_weights: tuple | list | None,
    target_size: tuple[int, int] = (500, 500),
) -> np.ndarray | None:
    """
    Convert raw attention weight tensors into a 2D heatmap.

    Args:
        attention_weights: Tuple of attention tensors from the model's
            output_attentions=True. Each tensor has shape
            (batch, num_heads, seq_len, seq_len).
        target_size: (width, height) to resize the heatmap to.

    Returns:
        Normalized 2D numpy array (0.0–1.0) of shape (height, width),
        or None if attention_weights is unavailable.
    """
    if attention_weights is None:
        return None

    try:
        import torch

        # Use the last layer's attention as it's most task-relevant
        last_layer_attn = attention_weights[-1]  # (batch, heads, seq, seq)

        # Average across all heads
        avg_attn = last_layer_attn.mean(dim=1)  # (batch, seq, seq)

        # Take the CLS token's attention over spatial tokens (first token → rest)
        # This represents "what the model looked at"
        cls_attn = avg_attn[0, 0, 1:]  # (seq_len - 1,)

        # Try to reshape into a square spatial grid
        num_patches = cls_attn.shape[0]
        grid_size = int(num_patches ** 0.5)

        if grid_size * grid_size == num_patches:
            heatmap = cls_attn.reshape(grid_size, grid_size)
        else:
            # Non-square: reshape to closest rectangle
            import math
            h = int(math.sqrt(num_patches))
            while num_patches % h != 0 and h > 1:
                h -= 1
            w = num_patches // h
            heatmap = cls_attn[:h * w].reshape(h, w)

        # Convert to numpy and normalize
        heatmap = heatmap.detach().cpu().float().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # Resize to target dimensions
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8), mode="L")
        heatmap_img = heatmap_img.resize(target_size, Image.BILINEAR)
        return np.array(heatmap_img).astype(np.float32) / 255.0

    except Exception:
        return None


def overlay_heatmap(
    base_image: Image.Image,
    heatmap: np.ndarray | None,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> Image.Image:
    """
    Overlay a heatmap onto a base image using a matplotlib colormap.

    Args:
        base_image: The original stimulus image.
        heatmap: 2D numpy array (0.0–1.0), or None (returns base unchanged).
        alpha: Blend factor for the heatmap overlay.
        colormap: Matplotlib colormap name.

    Returns:
        PIL Image with the heatmap overlay.
    """
    if heatmap is None:
        return base_image.copy()

    import matplotlib.cm as cm

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(heatmap)[:, :, :3]  # Drop alpha channel, keep RGB
    colored = (colored * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(colored, mode="RGB")

    # Ensure same size
    heatmap_img = heatmap_img.resize(base_image.size, Image.BILINEAR)

    # Blend
    return Image.blend(base_image.convert("RGB"), heatmap_img, alpha)
