"""
Nemotron OpenRouter API Client.

Wraps the OpenRouter endpoint (OpenAI-compatible) with tool-calling support.
Uses the `openai` Python SDK pointed at the OpenRouter base URL.
"""

from __future__ import annotations

from typing import Any, Optional

from openai import OpenAI

from config import settings


class NemotronClient:
    """
    Client for the OpenRouter API (OpenAI-compatible).

    Supports chat completions with tool/function calling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.base_url = base_url or settings.NEMOTRON_BASE_URL
        self.model = model or settings.NEMOTRON_MODEL

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required. Set it in .env or pass directly."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Send a chat completion request with optional tool calling.

        Args:
            messages: Conversation history in OpenAI format.
            tools: Tool definitions in OpenAI function-calling schema.
            tool_choice: "auto", "none", "required", or specific tool name.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            The full API response as a dict-like object.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or settings.NEMOTRON_TEMP,
            "max_tokens": max_tokens or settings.NEMOTRON_MAX_TOKENS,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**kwargs)
        return response

    def extract_tool_calls(self, response) -> list[dict[str, Any]]:
        """
        Parse tool calls from a completion response.

        Returns:
            List of dicts with 'name', 'arguments' (parsed JSON), and 'id'.
        """
        import json

        message = response.choices[0].message
        if not message.tool_calls:
            return []

        calls = []
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                # Attempt to repair malformed JSON from the model
                try:
                    args = self._repair_json(tc.function.arguments)
                except Exception:
                    # If repair fails, return empty dict or raw string (to avoid crash)
                    print(f"Failed to parse tool arguments: {tc.function.arguments}")
                    args = {}

            calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": args,
            })
        return calls

    def _repair_json(self, json_str: str) -> dict[str, Any]:
        """Attempt to repair common JSON syntax errors from LLMs."""
        import json
        import re

        # Strip markdown code blocks
        if "```" in json_str:
            pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
            match = re.search(pattern, json_str, re.DOTALL)
            if match:
                json_str = match.group(1)

        json_str = json_str.strip()

        # Strategy 1: Attempt to parse the whole string
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Handle concatenated JSONs (e.g. "{...}{...}")
        # We take the *last* valid JSON object, assuming it's the correction
        # or the most recent intent.
        potential_jsons = []
        
        # Regex to find top-level braces (non-nested assumption for simplicity, or greedy)
        # This regex tries to find {...} blocks. It's not perfect for nesting.
        # But for tool args usually simple.
        candidates = re.findall(r'\{.*?\}', json_str)
        if hasattr(re, 'DOTALL'):
             candidates = re.findall(r'\{.*?\}', json_str, re.DOTALL)
        
        for candidate in candidates:
             try:
                 # Clean candidate keys (single quotes)
                 if "'" in candidate and '"' not in candidate:
                     candidate = candidate.replace("'", '"')
                 return json.loads(candidate)
             except json.JSONDecodeError:
                 continue
        
        # Strategy 3: Brute force cleanup (single quotes)
        if "'" in json_str and '"' not in json_str:
             json_str = json_str.replace("'", '"')
             return json.loads(json_str)
        
        raise ValueError("Could not repair JSON")

    def extract_text(self, response) -> str:
        """Extract plain text content from a response."""
        return response.choices[0].message.content or ""
