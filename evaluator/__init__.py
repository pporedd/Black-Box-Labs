"""VLM Evaluator — Moondream2 subject model with attention extraction."""

from evaluator.vlm import MoondreamEvaluator, EvalResult
from evaluator.attention import extract_attention_heatmap

__all__ = ["MoondreamEvaluator", "EvalResult", "extract_attention_heatmap"]
