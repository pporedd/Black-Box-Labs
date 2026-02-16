"""
Central configuration for the VLM Behavioral Boundary system.
All core constants and environment-driven settings live here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # ── Canvas ──────────────────────────────────────────────────────
    CANVAS_WIDTH: int = 500
    CANVAS_HEIGHT: int = 500

    # ── VLM (Moondream2) ────────────────────────────────────────────
    VLM_MODEL_ID: str = "vikhyatk/moondream2"
    VLM_REVISION: str = "2025-01-09"
    VLM_TEMP: float = 0.0
    VLM_MAX_TOKENS: int = 20
    VLM_DEVICE: str = "cuda"  # GTX 1080
    VLM_ATTN_IMPLEMENTATION: str = "eager"  # required for output_attentions

    # ── Scientist Agent (Nemotron via OpenRouter) ───────────────────
    OPENROUTER_API_KEY: str = ""
    NEMOTRON_MODEL: str = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
    NEMOTRON_BASE_URL: str = "https://openrouter.ai/api/v1"
    NEMOTRON_TEMP: float = 0.3
    NEMOTRON_MAX_TOKENS: int = 4096

    # ── Isomorphic Validation ───────────────────────────────────────
    NUM_ISOMORPHISMS: int = 5

    # ── Paths ───────────────────────────────────────────────────────
    PROJECT_ROOT: Path = Path(__file__).parent
    SESSIONS_DIR: Optional[Path] = None
    OUTPUTS_DIR: Optional[Path] = None

    @model_validator(mode="after")
    def _set_default_paths(self) -> Settings:
        if self.SESSIONS_DIR is None:
            self.SESSIONS_DIR = self.PROJECT_ROOT / "sessions"
        if self.OUTPUTS_DIR is None:
            self.OUTPUTS_DIR = self.PROJECT_ROOT / "outputs"
        self.SESSIONS_DIR.mkdir(exist_ok=True)
        self.OUTPUTS_DIR.mkdir(exist_ok=True)
        return self

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton instance
settings = Settings()
