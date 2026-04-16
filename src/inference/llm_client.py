"""Inference clients for open-weight judge model endpoints."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests
from pydantic import BaseModel, Field

from utils.constants import CONFIG_DIR
from utils.io import load_yaml


class InferenceSettings(BaseModel):
    """Runtime inference configuration."""

    provider: str = "none"
    model_name: str = "qwen2.5-7b-instruct"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout_seconds: int = 45
    retries: int = Field(default=2, ge=0, le=5)


@dataclass
class InferenceResponse:
    """Normalized model response payload."""

    text: str
    raw: dict[str, Any]


def load_inference_settings() -> InferenceSettings:
    """Load inference settings from config and environment."""
    payload = load_yaml(CONFIG_DIR / "models" / "inference.yaml") or {}
    payload.update(
        {
            "provider": os.getenv("FACETFORGE_MODEL_PROVIDER", payload.get("provider", "none")),
            "model_name": os.getenv("FACETFORGE_MODEL_NAME", payload.get("model_name", "qwen2.5-7b-instruct")),
            "base_url": os.getenv("FACETFORGE_MODEL_BASE_URL", payload.get("base_url")),
            "api_key": os.getenv("FACETFORGE_MODEL_API_KEY", payload.get("api_key")),
            "timeout_seconds": int(os.getenv("FACETFORGE_MODEL_TIMEOUT_SECONDS", payload.get("timeout_seconds", 45))),
            "retries": int(os.getenv("FACETFORGE_MODEL_RETRIES", payload.get("retries", 2))),
        }
    )
    return InferenceSettings(**payload)


class LLMClient:
    """Thin multi-provider client for open-weight judging."""

    def __init__(self, settings: InferenceSettings) -> None:
        self.settings = settings

    @property
    def is_configured(self) -> bool:
        """Return True when a live provider is configured."""
        if self.settings.provider == "none":
            return False
        return bool(self.settings.base_url)

    def generate(self, messages: list[dict[str, str]]) -> InferenceResponse:
        """Generate a model response from the configured provider."""
        if not self.is_configured:
            raise RuntimeError("No inference provider configured.")

        if self.settings.provider in {"openai_compatible", "vllm"}:
            return self._generate_openai_compatible(messages)
        if self.settings.provider == "ollama":
            return self._generate_ollama(messages)
        raise ValueError(f"Unsupported provider: {self.settings.provider}")

    def generate_many(self, message_batches: list[list[dict[str, str]]]) -> list[InferenceResponse]:
        """Run multiple prompt batches sequentially."""
        return [self.generate(messages) for messages in message_batches]

    def _generate_openai_compatible(self, messages: list[dict[str, str]]) -> InferenceResponse:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        payload = {
            "model": self.settings.model_name,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        endpoint = self.settings.base_url.rstrip("/") + "/chat/completions"
        response = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=self.settings.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return InferenceResponse(text=content, raw=data)

    def _generate_ollama(self, messages: list[dict[str, str]]) -> InferenceResponse:
        payload = {
            "model": self.settings.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.settings.temperature},
        }
        endpoint = self.settings.base_url.rstrip("/") + "/api/chat"
        response = requests.post(endpoint, json=payload, timeout=self.settings.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "")
        return InferenceResponse(text=content, raw=data)

