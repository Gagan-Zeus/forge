from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from models.copilot_client import CopilotAPIError, CopilotAuthError, CopilotClient
from models.opencode_client import OpenCodeAPIError, OpenCodeAuthError, OpenCodeClient

LOGGER = logging.getLogger(__name__)


class ModelAuthError(RuntimeError):
    """Raised when model authentication fails (unified)."""


class ModelAPIError(RuntimeError):
    """Raised when model API calls fail (unified)."""


class ModelClient(ABC):
    """Abstract base class for unified model client interface."""

    @property
    @abstractmethod
    def provider(self) -> Literal["copilot", "opencode"]:
        """Return the provider type."""
        ...

    @property
    @abstractmethod
    def available_models(self) -> tuple[str, ...]:
        """Return available models."""
        ...

    @abstractmethod
    async def is_authenticated(self) -> bool:
        """Check if the client is authenticated."""
        ...

    @abstractmethod
    async def ensure_ready(self) -> None:
        """Ensure the client is ready (authenticated)."""
        ...

    @abstractmethod
    async def call(
        self,
        messages: Sequence[dict[str, str]],
        model: str = "gpt-5-mini",
        system_prompt: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        on_assistant_delta: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> str:
        """Call the model API."""
        ...

    @abstractmethod
    async def get_access_token(self) -> str:
        """Get access token for GitHub operations."""
        ...

    @abstractmethod
    async def refresh_available_models(self) -> tuple[str, ...]:
        """Refresh and return available models."""
        ...


class CopilotModelClient(ModelClient):
    """Wrapper for CopilotClient to implement ModelClient interface."""

    def __init__(self, client: CopilotClient) -> None:
        self._client = client

    @property
    def provider(self) -> Literal["copilot", "opencode"]:
        return "copilot"

    @property
    def available_models(self) -> tuple[str, ...]:
        return self._client.available_models()

    async def is_authenticated(self) -> bool:
        return self._client.is_authenticated()

    async def ensure_ready(self) -> None:
        try:
            await self._client.ensure_ready()
        except CopilotAuthError as e:
            raise ModelAuthError(str(e)) from e
        except CopilotAPIError as e:
            raise ModelAPIError(str(e)) from e

    async def call(
        self,
        messages: Sequence[dict[str, str]],
        model: str = "gpt-5-mini",
        system_prompt: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        on_assistant_delta: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> str:
        try:
            return await self._client.call(
                messages=messages,
                model=model,
                system_prompt=system_prompt,
                attachments=attachments,
                on_assistant_delta=on_assistant_delta,
            )
        except CopilotAuthError as e:
            raise ModelAuthError(str(e)) from e
        except CopilotAPIError as e:
            raise ModelAPIError(str(e)) from e

    async def get_access_token(self) -> str:
        try:
            return await self._client.get_access_token()
        except CopilotAuthError as e:
            raise ModelAuthError(str(e)) from e

    async def refresh_available_models(self) -> tuple[str, ...]:
        try:
            return await self._client.refresh_available_models()
        except CopilotAuthError as e:
            raise ModelAuthError(str(e)) from e


class OpenCodeModelClient(ModelClient):
    """Wrapper for OpenCodeClient to implement ModelClient interface."""

    def __init__(self, client: OpenCodeClient) -> None:
        self._client = client

    @property
    def provider(self) -> Literal["copilot", "opencode"]:
        return "opencode"

    @property
    def available_models(self) -> tuple[str, ...]:
        return self._client.available_models()

    async def is_authenticated(self) -> bool:
        return self._client.is_authenticated()

    async def ensure_ready(self) -> None:
        try:
            await self._client.ensure_ready()
        except OpenCodeAuthError as e:
            raise ModelAuthError(str(e)) from e
        except OpenCodeAPIError as e:
            raise ModelAPIError(str(e)) from e

    async def call(
        self,
        messages: Sequence[dict[str, str]],
        model: str = "gpt-5-mini",
        system_prompt: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        on_assistant_delta: Callable[[str], Awaitable[None] | None] = None,
    ) -> str:
        try:
            return await self._client.call(
                messages=messages,
                model=model,
                system_prompt=system_prompt,
                attachments=attachments,
                on_assistant_delta=on_assistant_delta,
            )
        except OpenCodeAuthError as e:
            raise ModelAuthError(str(e)) from e
        except OpenCodeAPIError as e:
            raise ModelAPIError(str(e)) from e

    async def get_access_token(self) -> str:
        # OpenCode doesn't provide GitHub tokens, return empty
        return ""

    async def refresh_available_models(self) -> tuple[str, ...]:
        try:
            return await self._client.refresh_available_models()
        except OpenCodeAuthError as e:
            raise ModelAuthError(str(e)) from e


class UnifiedModelClient:
    """Manages both Copilot and OpenCode clients and switches between them."""

    # Provider-specific model defaults
    COPILOT_DEFAULT_MODEL = "gpt-5-mini"
    OPENCODE_DEFAULT_MODEL = "minimax-m2.5-free"

    def __init__(
        self,
        copilot_client: CopilotClient,
        opencode_client: OpenCodeClient | None = None,
        default_provider: Literal["copilot", "opencode"] = "copilot",
    ) -> None:
        self._copilot = CopilotModelClient(copilot_client)
        self._opencode = OpenCodeModelClient(opencode_client) if opencode_client else None
        self._default_provider = default_provider
        self._active_provider: Literal["copilot", "opencode"] = default_provider

    @property
    def active_client(self) -> ModelClient:
        """Get the currently active client."""
        if self._active_provider == "opencode" and self._opencode:
            return self._opencode
        return self._copilot

    @property
    def provider(self) -> Literal["copilot", "opencode"]:
        """Get the current provider."""
        return self._active_provider

    @provider.setter
    def provider(self, value: Literal["copilot", "opencode"]) -> None:
        """Set the current provider."""
        if value == "opencode" and not self._opencode:
            raise ModelAuthError("OpenCode client not configured. Set OPENCODE_API_KEY in environment.")
        self._active_provider = value

    def get_client_for_provider(self, provider: Literal["copilot", "opencode"]) -> ModelClient:
        """Get client for a specific provider."""
        if provider == "opencode":
            if not self._opencode:
                raise ModelAuthError("OpenCode client not configured. Set OPENCODE_API_KEY in environment.")
            return self._opencode
        return self._copilot

    @property
    def available_models(self) -> tuple[str, ...]:
        """Get available models for current provider."""
        return self.active_client.available_models

    @property
    def copilot_models(self) -> tuple[str, ...]:
        """Get Copilot models."""
        return self._copilot.available_models

    @property
    def opencode_models(self) -> tuple[str, ...]:
        """Get OpenCode models."""
        if self._opencode:
            return self._opencode.available_models
        return ()

    def get_default_model(self, provider: Literal["copilot", "opencode"] | None = None) -> str:
        """Get default model for provider."""
        prov = provider or self._active_provider
        if prov == "copilot":
            return self.COPILOT_DEFAULT_MODEL
        return self.OPENCODE_DEFAULT_MODEL

    async def is_authenticated(self) -> bool:
        """Check if active client is authenticated."""
        return await self.active_client.is_authenticated()

    async def ensure_ready(self) -> None:
        """Ensure active client is ready."""
        try:
            await self.active_client.ensure_ready()
        except ModelAuthError:
            raise

    def _get_client_for_model(self, model: str) -> ModelClient:
        """Get the appropriate client based on model name."""
        # Check if it's an OpenCode model
        opencode_models = {
            "minimax-m2.5-free",
            "deepseek-v4-flash-free", 
            "ring-2.6-1t-free",
            "nemotron-3-super-free",
        }
        if model in opencode_models and self._opencode:
            return self._opencode
        return self._copilot

    async def call(
        self,
        messages: Sequence[dict[str, str]],
        model: str = "gpt-5-mini",
        system_prompt: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        on_assistant_delta: Callable[[str], Awaitable[None] | None] = None,
    ) -> str:
        """Call the appropriate client based on model."""
        client = self._get_client_for_model(model)
        try:
            return await client.call(
                messages=messages,
                model=model,
                system_prompt=system_prompt,
                attachments=attachments,
                on_assistant_delta=on_assistant_delta,
            )
        except ModelAuthError:
            raise

    async def get_access_token(self) -> str:
        """Get access token from Copilot (for GitHub operations)."""
        try:
            return await self._copilot.get_access_token()
        except ModelAuthError:
            return ""

    async def get_authenticated_login(self) -> str:
        """Get authenticated user info from the active provider."""
        try:
            if self._active_provider == "opencode":
                return "OpenCode user"
            return await self._copilot.get_authenticated_login()
        except ModelAuthError:
            return "unknown"

    async def refresh_available_models(self) -> tuple[str, ...]:
        """Refresh available models for active provider."""
        try:
            return await self.active_client.refresh_available_models()
        except ModelAuthError:
            raise

    def is_provider_available(self, provider: Literal["copilot", "opencode"]) -> bool:
        """Check if a provider is available."""
        if provider == "copilot":
            return True  # Copilot is always available
        return self._opencode is not None

    @property
    def has_opencode(self) -> bool:
        """Check if OpenCode client is configured."""
        return self._opencode is not None
