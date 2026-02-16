"""
Moondream2 VLM Evaluator.

Loads the model with attn_implementation="eager" and output_attentions=True
to expose attention matrices for heatmap visualization.

Configured for GTX 1080 (CUDA).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from config import settings


@dataclass
class EvalResult:
    """Result of evaluating a single image."""
    raw_answer: str
    parsed_answer: str
    ground_truth: str
    passed: bool
    attention_map: Optional[np.ndarray] = None  # 2D heatmap (H, W) or None


class MoondreamEvaluator:
    """
    Wraps Moondream2 for deterministic VQA evaluation.

    Uses temp=0, max_tokens=20 for short deterministic answers.
    Captures attention weights for visualization.
    """

    def __init__(self, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, GenerationConfig

        self.device = device or settings.VLM_DEVICE
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        print(f"[MoondreamEvaluator] Loading {settings.VLM_MODEL_ID} on {self.device}...")

        # ── MONKEY PATCH for transformers >= 4.41 compatibility ──
        # Fixes: 'HfMoondream' object has no attribute 'all_tied_weights_keys'
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            @property
            def all_tied_weights_keys(self):
                return {}
            PreTrainedModel.all_tied_weights_keys = all_tied_weights_keys

        # Load weights and move to device manually
        # device_map in transformers can be buggy with custom Moondream implementation
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.VLM_MODEL_ID,
            revision=settings.VLM_REVISION,
            trust_remote_code=True,
            attn_implementation=settings.VLM_ATTN_IMPLEMENTATION,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        # Explicitly set generation config to prevent 'NoneType' object has no attribute 'keys'
        try:
            self.model.generation_config = GenerationConfig.from_pretrained(
                settings.VLM_MODEL_ID, 
                trust_remote_code=True
            )
        except Exception:
            # Fallback to a basic config if remote loading fails
            self.model.generation_config = GenerationConfig()

        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.VLM_MODEL_ID,
            revision=settings.VLM_REVISION,
            trust_remote_code=True,
        )

        self.model.eval()
        print("[MoondreamEvaluator] Model loaded.")

    def evaluate(
        self,
        image: Image.Image,
        question: str,
        ground_truth: str,
    ) -> EvalResult:
        """
        Ask Moondream2 a question about an image and compare to ground truth.

        Args:
            image: PIL Image (stimulus).
            question: The visual question to ask.
            ground_truth: Expected correct answer string.

        Returns:
            EvalResult with pass/fail, raw/parsed answers, and attention map.
        """
        import torch

        # Query the model
        with torch.no_grad():
            # Moondream's answer method (trust_remote_code API)
            raw_answer = self.model.answer_question(
                self.model.encode_image(image),
                question,
                self.tokenizer,
            )

        # Parse and compare
        parsed = self._parse_answer(raw_answer)
        passed = self._compare(parsed, ground_truth)

        # Attention map extraction is best-effort
        attention_map = None  # Will be populated by the attention hooks when possible

        return EvalResult(
            raw_answer=raw_answer.strip(),
            parsed_answer=parsed,
            ground_truth=ground_truth,
            passed=passed,
            attention_map=attention_map,
        )

    def evaluate_with_attention(
        self,
        image: Image.Image,
        question: str,
        ground_truth: str,
    ) -> EvalResult:
        """
        Evaluate with attention map extraction enabled.

        This is slower and uses more VRAM due to output_attentions=True,
        but provides the attention heatmap for visualization.
        """
        import torch
        from evaluator.attention import extract_attention_heatmap

        # Encode image
        image_embeds = self.model.encode_image(image)

        # Use attention hooks — attempt to capture encoder attentions
        attention_map = None

        try:
            # Try to get attention from the vision encoder
            # This is best-effort; architecture may vary
            enc_image = self.model.vision_encoder(
                image.resize((384, 384)),
                output_attentions=True,
            ) if hasattr(self.model, 'vision_encoder') else None

            if enc_image is not None and hasattr(enc_image, 'attentions'):
                attention_map = extract_attention_heatmap(
                    enc_image.attentions,
                    target_size=(image.width, image.height),
                )
        except Exception:
            pass  # Attention extraction is best-effort

        with torch.no_grad():
            raw_answer = self.model.answer_question(
                image_embeds,
                question,
                self.tokenizer,
            )

        parsed = self._parse_answer(raw_answer)
        passed = self._compare(parsed, ground_truth)

        return EvalResult(
            raw_answer=raw_answer.strip(),
            parsed_answer=parsed,
            ground_truth=ground_truth,
            passed=passed,
            attention_map=attention_map,
        )

    @staticmethod
    def _parse_answer(raw: str) -> str:
        """
        Parse the raw model output into a normalized answer.

        Handles common formats:
        - "3"
        - " 3 "
        - "There are 3 circles"
        - "three"
        - "Three items"
        """
        raw = raw.strip().lower()

        # Try to extract a number directly
        numbers = re.findall(r'\d+', raw)
        if numbers:
            return numbers[0]

        # Word-to-number mapping
        word_map = {
            "zero": "0", "one": "1", "two": "2", "three": "3",
            "four": "4", "five": "5", "six": "6", "seven": "7",
            "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
            "twelve": "12", "thirteen": "13", "fourteen": "14",
            "fifteen": "15", "sixteen": "16", "seventeen": "17",
            "eighteen": "18", "nineteen": "19", "twenty": "20",
        }
        for word, digit in word_map.items():
            if word in raw:
                return digit

        return raw

    @staticmethod
    def _compare(parsed: str, ground_truth: str) -> bool:
        """Compare parsed answer to ground truth."""
        return parsed.strip() == ground_truth.strip()
