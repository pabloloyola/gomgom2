"""Small OpenAI-compatible chat client used by EIPG LLM experiments.

The client intentionally depends only on ``requests`` so the same code works
against LM Studio, OpenRouter, or another OpenAI-compatible endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import os
import time

import requests


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    model: str
    api_key_env: str | None = None
    temperature: float | None = 0.0
    max_tokens: int = 256
    max_tokens_field: str = "max_tokens"
    timeout_seconds: float = 90.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    min_request_interval_seconds: float = 0.0
    cache_dir: str | None = ".cache/eipg/openai_compatible"
    extra_headers: dict[str, str] | None = None
    extra_body: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> "OpenAICompatibleConfig":
        return cls(
            base_url=str(section["base_url"]).rstrip("/"),
            model=str(section["model"]),
            api_key_env=(str(section["api_key_env"]) if section.get("api_key_env") else None),
            temperature=(None if section.get("temperature", 0.0) is None else float(section.get("temperature", 0.0))),
            max_tokens=int(section.get("max_tokens", 256)),
            max_tokens_field=str(section.get("max_tokens_field", "max_tokens")),
            timeout_seconds=float(section.get("timeout_seconds", 90.0)),
            max_retries=int(section.get("max_retries", 2)),
            retry_backoff_seconds=float(section.get("retry_backoff_seconds", 1.0)),
            min_request_interval_seconds=float(section.get("min_request_interval_seconds", 0.0)),
            cache_dir=(str(section["cache_dir"]) if section.get("cache_dir") else None),
            extra_headers={str(k): str(v) for k, v in section.get("extra_headers", {}).items()} or None,
            extra_body=dict(section.get("extra_body", {})) or None,
        )


@dataclass(frozen=True)
class ChatResult:
    text: str
    cached: bool
    latency_seconds: float
    prompt_hash: str


class OpenAICompatibleChatClient:
    def __init__(self, config: OpenAICompatibleConfig, *, session: requests.Session | None = None) -> None:
        self.config = config
        self.session = session or requests.Session()
        self.cache_dir = Path(config.cache_dir) if config.cache_dir else None
        self._last_request_started: float | None = None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            token = os.environ.get(self.config.api_key_env)
            if not token:
                raise RuntimeError(f"missing API key environment variable {self.config.api_key_env}")
            headers["Authorization"] = f"Bearer {token}"
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers

    def _cache_key(self, messages: list[dict[str, str]], *, response_format: dict[str, Any] | None) -> str:
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "max_tokens_field": self.config.max_tokens_field,
            "messages": messages,
            "response_format": response_format,
            "extra_body": self.config.extra_body,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return sha256(raw.encode("utf-8")).hexdigest()

    def _pace(self) -> None:
        interval = max(float(self.config.min_request_interval_seconds), 0.0)
        if interval <= 0.0 or self._last_request_started is None:
            return
        elapsed = time.monotonic() - self._last_request_started
        if elapsed < interval:
            time.sleep(interval - elapsed)

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        """Respect Retry-After when available; otherwise use exponential backoff."""
        delay = float(self.config.retry_backoff_seconds) * (2**attempt)
        response = getattr(exc, "response", None)
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    delay = max(delay, float(retry_after))
                except (TypeError, ValueError):
                    pass
        return max(delay, 0.0)

    def chat(self, messages: list[dict[str, str]], *, response_format: dict[str, Any] | None = None) -> ChatResult:
        key = self._cache_key(messages, response_format=response_format)
        path = None if self.cache_dir is None else self.cache_dir / f"{key}.json"
        if path is not None and path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return ChatResult(str(payload["text"]), True, 0.0, key)

        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            self.config.max_tokens_field: self.config.max_tokens,
        }
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        if self.config.extra_body:
            body.update(self.config.extra_body)
        if response_format is not None:
            body["response_format"] = response_format

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            self._pace()
            started = time.perf_counter()
            self._last_request_started = time.monotonic()
            try:
                response = self.session.post(
                    f"{self.config.base_url}/chat/completions",
                    json=body,
                    headers=self._headers(),
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                # Some OpenAI-compatible providers occasionally return a successful
                # HTTP response with null/empty assistant content. Treat that as a
                # transient completion failure so the retry policy applies.
                if content is None or not str(content).strip():
                    raise ValueError("empty assistant content")
                text = str(content)
                latency = time.perf_counter() - started
                if path is not None:
                    path.write_text(
                        json.dumps(
                            {
                                "text": text,
                                "model": self.config.model,
                                "latency_seconds": latency,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                return ChatResult(text, False, latency, key)
            except (requests.RequestException, KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(self._retry_delay(exc, attempt))
        assert last_error is not None
        raise RuntimeError(f"OpenAI-compatible chat failed after retries: {last_error}") from last_error
