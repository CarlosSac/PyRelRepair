"""Ollama LLM interface for local model inference."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OllamaClient:
    """Client for interacting with a local Ollama instance."""

    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.ollama_base_url
        self.model = config.ollama_model

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        num_return: int = 1,
        stop: list[str] | None = None,
    ) -> list[LLMResponse]:
        """Generate one or more completions from the LLM.

        For num_return > 1, makes multiple independent calls since
        Ollama's /api/generate does not natively support n > 1.
        """
        temp = temperature if temperature is not None else self.config.temperature
        responses = []

        for _ in range(num_return):
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temp,
                },
            }
            if stop:
                payload["options"]["stop"] = stop

            try:
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.config.llm_timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                responses.append(
                    LLMResponse(
                        text=data.get("response", ""),
                        model=data.get("model", self.model),
                        prompt_tokens=data.get("prompt_eval_count", 0),
                        completion_tokens=data.get("eval_count", 0),
                    )
                )
            except requests.RequestException as e:
                logger.error("Ollama request failed: %s", e)
                responses.append(LLMResponse(text="", model=self.model))

        return responses

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> LLMResponse:
        """Send a chat-style request (system/user/assistant messages)."""
        temp = temperature if temperature is not None else self.config.temperature

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temp,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.config.llm_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message", {})
            return LLMResponse(
                text=msg.get("content", ""),
                model=data.get("model", self.model),
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
            )
        except requests.RequestException as e:
            logger.error("Ollama chat request failed: %s", e)
            return LLMResponse(text="", model=self.model)

    def is_available(self) -> bool:
        """Check if the Ollama server is reachable and the model is loaded."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            # Match with or without tag suffix
            model_base = self.model.split(":")[0]
            return any(model_base in m for m in models)
        except requests.RequestException:
            return False
