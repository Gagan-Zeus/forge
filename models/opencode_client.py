from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

import httpx

LOGGER = logging.getLogger(__name__)


class OpenCodeAuthError(RuntimeError):
    """Raised when OpenCode authentication fails."""


class OpenCodeAPIError(RuntimeError):
    """Raised when OpenCode API calls fail."""


class OpenCodeClient:
    """Client for OpenCode Zen API (OpenAI-compatible)."""

    DEFAULT_BASE_URL = "https://opencode.ai/zen/v1"
    CHAT_MAX_ATTEMPTS = 3
    MAX_CALL_TIMEOUT_SECONDS = 300.0
    DEFAULT_TIMEOUT = 60.0

    # OpenCode Zen available models (from documentation)
    DEFAULT_MODELS = (
        # GPT models
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4",
        "gpt-5.4-pro",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.3-codex",
        "gpt-5.3-codex-spark",
        "gpt-5.2",
        "gpt-5.2-codex",
        "gpt-5.1",
        "gpt-5.1-codex",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex-mini",
        "gpt-5",
        "gpt-5-codex",
        "gpt-5-nano",
        # Claude models
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-opus-4-1",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-sonnet-4",
        "claude-haiku-4-5",
        "claude-3-5-haiku",
        # Gemini models
        "gemini-3.1-pro",
        "gemini-3-flash",
        # Other models
        "qwen3.6-plus",
        "qwen3.5-plus",
        "minimax-m2.7",
        "minimax-m2.5",
        "minimax-m2.5-free",
        "glm-5.1",
        "glm-5",
        "kimi-k2.5",
        "kimi-k2.6",
        "big-pickle",
        "deepseek-v4-flash-free",
        "ring-2.6-1t-free",
        "nemotron-3-super-free",
    )

    # Map model names to endpoint types
    MODEL_ENDPOINTS = {
        # GPT models use /responses endpoint
        "gpt-5.5": "responses",
        "gpt-5.5-pro": "responses",
        "gpt-5.4": "responses",
        "gpt-5.4-pro": "responses",
        "gpt-5.4-mini": "responses",
        "gpt-5.4-nano": "responses",
        "gpt-5.3-codex": "responses",
        "gpt-5.3-codex-spark": "responses",
        "gpt-5.2": "responses",
        "gpt-5.2-codex": "responses",
        "gpt-5.1": "responses",
        "gpt-5.1-codex": "responses",
        "gpt-5.1-codex-max": "responses",
        "gpt-5.1-codex-mini": "responses",
        "gpt-5": "responses",
        "gpt-5-codex": "responses",
        "gpt-5-nano": "responses",
        # Claude models use /messages endpoint
        "claude-opus-4-7": "messages",
        "claude-opus-4-6": "messages",
        "claude-opus-4-5": "messages",
        "claude-opus-4-1": "messages",
        "claude-sonnet-4-6": "messages",
        "claude-sonnet-4-5": "messages",
        "claude-sonnet-4": "messages",
        "claude-haiku-4-5": "messages",
        "claude-3-5-haiku": "messages",
        # Gemini models use specific model endpoints
        "gemini-3.1-pro": "models/gemini-3.1-pro",
        "gemini-3-flash": "models/gemini-3-flash",
        # All others use /chat/completions
    }

    _known_models: tuple[str, ...] = DEFAULT_MODELS

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        base_system_prompt_path: str | None = None,
    ) -> None:
        self._api_key = (api_key or os.getenv("OPENCODE_API_KEY", "")).strip()
        self._base_url = (base_url or os.getenv("OPENCODE_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self._timeout_seconds = timeout_seconds if timeout_seconds and timeout_seconds > 0 else self.DEFAULT_TIMEOUT
        self._base_system_prompt = self._load_base_system_prompt(base_system_prompt_path)

        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    @staticmethod
    def _resolve_base_system_prompt_path(base_system_prompt_path: str | None) -> Path:
        if base_system_prompt_path:
            return Path(base_system_prompt_path).expanduser().resolve()
        return (Path(__file__).resolve().parent.parent / "system-prompt.txt").resolve()

    @classmethod
    def _load_base_system_prompt(cls, base_system_prompt_path: str | None) -> str:
        candidate = cls._resolve_base_system_prompt_path(base_system_prompt_path)
        if not candidate.exists() or not candidate.is_file():
            LOGGER.info("Base system prompt file not found at %s; continuing without it.", candidate)
            return ""
        try:
            content = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            LOGGER.warning("Could not read base system prompt file at %s; continuing without it.", candidate)
            return ""
        return content

    def _merged_system_prompt(self, system_prompt: str | None) -> str | None:
        base = self._base_system_prompt.strip()
        scoped = (system_prompt or "").strip()
        if base and scoped:
            return f"{base}\n\n{scoped}"
        if base:
            return base
        if scoped:
            return scoped
        return None

    def is_authenticated(self) -> bool:
        return bool(self._api_key)

    async def ensure_ready(self) -> None:
        if not self._api_key:
            raise OpenCodeAuthError(
                "OpenCode API key is not configured. "
                "Set OPENCODE_API_KEY in your .env file or environment. "
                "Get your API key at https://opencode.ai/auth"
            )
        # Test the connection
        await self.refresh_available_models()

    async def get_token(self) -> str:
        """Return the API key (for compatibility with CopilotClient interface)."""
        if self._api_key:
            return self._api_key
        raise OpenCodeAuthError("OpenCode API key not configured.")

    async def get_access_token(self) -> str:
        """Return the API key (for compatibility with CopilotClient interface)."""
        return await self.get_token()

    async def refresh_available_models(self) -> tuple[str, ...]:
        if not self._api_key:
            return self._known_models

        try:
            client = await self._ensure_client()
            # Try to fetch models from the API
            response = await client.get(f"{self._base_url}/models")
            if response.status_code == 200:
                data = response.json()
                if "data" in data:
                    models = tuple(sorted([m["id"] for m in data["data"] if "id" in m]))
                    if models:
                        OpenCodeClient._known_models = models
                elif isinstance(data, list):
                    models = tuple(sorted([m["id"] for m in data if "id" in m]))
                    if models:
                        OpenCodeClient._known_models = models
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Could not refresh OpenCode model list: %s", exc)

        return OpenCodeClient._known_models

    @classmethod
    def available_models(cls) -> tuple[str, ...]:
        return cls._known_models

    def _get_endpoint_for_model(self, model: str) -> str:
        """Get the appropriate endpoint for a model."""
        endpoint = self.MODEL_ENDPOINTS.get(model, "chat/completions")
        return f"{self._base_url}/{endpoint}"

    def _convert_messages(self, messages: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
        """Convert message format to OpenAI-compatible format."""
        converted = []
        for msg in messages:
            role = str(msg.get("role", "user")).strip().lower()
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            converted.append({"role": role, "content": content})
        return converted

    async def call(
        self,
        messages: Sequence[dict[str, str]],
        model: str = "gpt-5-mini",
        system_prompt: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        on_assistant_delta: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> str:
        await self.ensure_ready()

        # Prepare messages
        converted_messages = self._convert_messages(messages)

        # Add system prompt if provided
        merged_system = self._merged_system_prompt(system_prompt)
        if merged_system:
            converted_messages.insert(0, {"role": "system", "content": merged_system})

        # Handle attachments (convert to message format if needed)
        if attachments:
            for attachment in attachments:
                content = attachment.get("content") or attachment.get("text", "")
                if content:
                    converted_messages.append({
                        "role": "user",
                        "content": f"[Attachment]\n{content}"
                    })

        if not converted_messages:
            raise OpenCodeAPIError("No valid messages to send.")

        # Prepare request payload
        payload = {
            "model": model,
            "messages": converted_messages,
            "stream": on_assistant_delta is not None,
            "temperature": 0.7,
        }

        endpoint = self._get_endpoint_for_model(model)
        max_attempts = 1 if on_assistant_delta else self.CHAT_MAX_ATTEMPTS

        for attempt in range(1, max_attempts + 1):
            try:
                client = await self._ensure_client()

                if on_assistant_delta is not None:
                    # Streaming response
                    streamed_chunks: list[str] = []
                    async with client.stream("POST", endpoint, json=payload) as response:
                        if response.status_code == 401:
                            raise OpenCodeAuthError("Invalid OpenCode API key.")
                        if response.status_code != 200:
                            body = await response.aread()
                            raise OpenCodeAPIError(f"API error {response.status_code}: {body.decode()}")

                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data = line[6:]
                                if data == "[DONE]":
                                    break
                                try:
                                    import json
                                    chunk = json.loads(data)
                                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                    if delta:
                                        streamed_chunks.append(delta)
                                        maybe_awaitable = on_assistant_delta(delta)
                                        if asyncio.iscoroutine(maybe_awaitable):
                                            await maybe_awaitable
                                except json.JSONDecodeError:
                                    continue

                    return "".join(streamed_chunks).strip()
                else:
                    # Non-streaming response
                    response = await client.post(endpoint, json=payload)

                    if response.status_code == 401:
                        raise OpenCodeAuthError("Invalid OpenCode API key.")
                    if response.status_code != 200:
                        raise OpenCodeAPIError(f"API error {response.status_code}: {response.text}")

                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    if content:
                        return content

                    if attempt < max_attempts:
                        LOGGER.warning("OpenCode returned empty response on attempt %s/%s", attempt, max_attempts)
                        await asyncio.sleep(1.5 * attempt)
                        continue

                    raise OpenCodeAPIError("OpenCode API returned an empty response.")

            except OpenCodeAuthError:
                raise
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("OpenCode API error on attempt %s/%s: %s", attempt, max_attempts, exc)
                if attempt >= max_attempts:
                    raise OpenCodeAPIError(f"Could not complete OpenCode API request: {exc}") from exc
                await asyncio.sleep(1.5 * attempt)

        raise OpenCodeAPIError("OpenCode API call failed after retries.")

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client

        async with self._client_lock:
            if self._client is not None:
                return self._client

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=self._timeout_seconds,
                follow_redirects=True,
            )
            return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get_authenticated_login(self) -> str:
        """Return user info for OpenCode."""
        if not self._api_key:
            return "not authenticated"
        try:
            # Try to get user info (if available)
            return "authenticated"
        except Exception:  # noqa: BLE001
            return "authenticated (details unavailable)"
