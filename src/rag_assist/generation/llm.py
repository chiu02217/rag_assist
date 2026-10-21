"""OpenAI-compatible chat completion client for Ollama, LM Studio, vLLM, or hosted APIs."""
from __future__ import annotations
from dataclasses import dataclass
import os
import requests

@dataclass
class OpenAICompatibleLLM:
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:3b"
    api_key: str = "ollama"
    timeout_seconds: int = 120
    temperature: float = 0.1
    max_tokens: int = 800

    @classmethod
    def from_env(cls) -> "OpenAICompatibleLLM":
        return cls(
            base_url=os.getenv("RAG_LLM_BASE_URL", "http://localhost:11434/v1"),
            model=os.getenv("RAG_LLM_MODEL", "qwen2.5:3b"),
            api_key=os.getenv("RAG_LLM_API_KEY", "ollama"),
            timeout_seconds=int(os.getenv("RAG_LLM_TIMEOUT_SECONDS", "120")),
            temperature=float(os.getenv("RAG_LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("RAG_LLM_MAX_TOKENS", "800")),
        )

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
                "reasoning_effort": "none",
            }
        try:
            response = requests.post(endpoint, headers=headers, json=payload,
                                     timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Cannot connect to LLM endpoint {endpoint}. Start the LLM server or "
                f"check RAG_LLM_BASE_URL / RAG_LLM_MODEL. Original error: {exc}"
            ) from exc
        if not response.ok:
            raise RuntimeError(
                f"LLM request failed (HTTP {response.status_code}) at {endpoint}: "
                f"{response.text[:1200]}"
            )
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Unexpected LLM response. Expected OpenAI-compatible /chat/completions."
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM returned an empty answer.")
        return content.strip()
