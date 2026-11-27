from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .utils import extract_json


@dataclass
class LLMResponse:
    text: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class OllamaClient:
    def __init__(
        self,
        model: str = "llama3.2:latest",
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 300,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        json_mode: bool = False,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": 8192,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama is unavailable at {self.base_url}. Start it with 'ollama serve' "
                f"and pull the configured model. Original error: {exc}"
            ) from exc
        latency_ms = int((time.monotonic() - started) * 1000)
        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            latency_ms=latency_ms,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )

    def chat_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        attempts: int = 3,
    ) -> tuple[Any, LLMResponse]:
        last_error: Exception | None = None
        last_response: LLMResponse | None = None
        for _ in range(attempts):
            response = self.chat(
                prompt,
                system=system,
                temperature=temperature,
                json_mode=True,
                max_tokens=max_tokens,
            )
            last_response = response
            try:
                return extract_json(response.text), response
            except ValueError as exc:
                last_error = exc
                prompt = (
                    "Return only one valid JSON value. Do not include markdown or commentary. "
                    f"Previous invalid response: {response.text[:1000]}\n\nOriginal request:\n{prompt}"
                )
        raise RuntimeError(f"Model did not return valid JSON after {attempts} attempts: {last_error}") from last_error
