from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

import httpx

LOGGER = logging.getLogger(__name__)


class CopilotAuthError(RuntimeError):
    """Raised when Copilot authentication fails."""


class CopilotAPIError(RuntimeError):
    """Raised when Copilot model calls fail."""


@dataclass(frozen=True)
class DeviceFlowResponse:
    device_code: str
    user_code: str
    verification_uri: str
    interval: int


class CopilotClient:
    CLIENT_ID = "Iv1.b507a08c87ecfe98"
    DEVICE_CODE_URL = "https://github.com/login/device/code"
    ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
    COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
    CHAT_COMPLETIONS_URL = "https://api.githubcopilot.com/chat/completions"
    EDITOR_VERSION = "vscode/1.85.0"
    EDITOR_PLUGIN_VERSION = "copilot-chat/0.13.0"
    USER_AGENT = "GitHubCopilotChat/0.13.0"
    CHAT_MAX_ATTEMPTS = 4
    CHAT_RETRYABLE_STATUS_CODES = (408, 409, 425, 429, 500, 502, 503, 504)

    SUPPORTED_MODELS = (
        "gpt-5.2",
        "gpt-5.2-codex",
        "gpt-5.3-codex",
        "gpt-5.4-mini",
        "gpt-5.1",
        "gpt-5.1-codex",
        "gpt-5.1-codex-mini",
        "gpt-5.1-codex-max",
        "gpt-5-mini",
        "gpt-4o-mini",
        "gpt-4o",
        "grok-code-fast-1",
        "claude-opus-4.5",
        "claude-sonnet-4.5",
        "claude-sonnet-4",
        "claude-haiku-4.5",
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-3-pro",
        "gemini-3-flash",
        "gpt-4.1",
        "o3-mini",
        "claude-3.5-sonnet",
        "gemini-1.5-pro",
        "o1-mini",
    )

    def __init__(self, tokens_path: str | Path = "auth/tokens.json", timeout_seconds: float = 60.0) -> None:
        self._tokens_path = Path(tokens_path)
        self._timeout_seconds = timeout_seconds

    def is_authenticated(self) -> bool:
        tokens = self._read_tokens()
        return bool(tokens.get("access_token"))

    async def request_device_code(self) -> DeviceFlowResponse:
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(
                self.DEVICE_CODE_URL,
                headers={"Accept": "application/json"},
                data={"client_id": self.CLIENT_ID, "scope": "copilot"},
            )
        response.raise_for_status()
        data = response.json()
        required_fields = {"device_code", "user_code", "verification_uri", "interval"}
        if not required_fields.issubset(data):
            raise CopilotAuthError(f"Invalid device code payload: {data}")
        return DeviceFlowResponse(
            device_code=data["device_code"],
            user_code=data["user_code"],
            verification_uri=data["verification_uri"],
            interval=int(data["interval"]),
        )

    async def poll_for_access_token(self, device_code: str, interval: int) -> str:
        poll_interval = max(interval, 1)
        while True:
            await asyncio.sleep(poll_interval)
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.post(
                    self.ACCESS_TOKEN_URL,
                    headers={"Accept": "application/json"},
                    data={
                        "client_id": self.CLIENT_ID,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                )
            response.raise_for_status()
            data = response.json()
            access_token = data.get("access_token")
            if access_token:
                return str(access_token)

            error_code = data.get("error")
            if error_code == "authorization_pending":
                continue
            if error_code == "slow_down":
                poll_interval += 5
                continue
            if error_code == "expired_token":
                raise CopilotAuthError("Device code expired. Please run /start again.")
            raise CopilotAuthError(f"Access token polling failed: {data}")

    async def exchange_copilot_token(self, access_token: str) -> tuple[str, int]:
        attempts = (
            ("bearer", f"Bearer {access_token}"),
            ("token", f"token {access_token}"),
        )
        errors: list[str] = []

        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            for label, authorization in attempts:
                response = await client.get(
                    self.COPILOT_TOKEN_URL,
                    headers={
                        "Authorization": authorization,
                        "Accept": "application/json",
                        "User-Agent": self.USER_AGENT,
                        "Editor-Version": self.EDITOR_VERSION,
                        "Editor-Plugin-Version": self.EDITOR_PLUGIN_VERSION,
                        "Copilot-Integration-Id": "vscode-chat",
                        "X-GitHub-Api-Version": "2022-11-28",
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    token = data.get("token")
                    expires_at = self._parse_expires_at(data.get("expires_at"))
                    if not token or not expires_at:
                        raise CopilotAuthError(f"Invalid Copilot token payload: {data}")
                    return str(token), expires_at

                detail = self._extract_error_detail(response)
                errors.append(f"{label} auth -> HTTP {response.status_code}: {detail}")

                if response.status_code not in {401, 403}:
                    response.raise_for_status()

        raise CopilotAuthError(
            "GitHub denied Copilot token exchange. Make sure the authorized GitHub account has "
            "active Copilot access and that you completed the device flow with the same account. "
            f"Details: {' | '.join(errors)}"
        )

    async def authenticate(
        self,
        on_device_code: Callable[[DeviceFlowResponse], Awaitable[None]] | None = None,
    ) -> None:
        device_flow = await self.request_device_code()
        if on_device_code:
            await on_device_code(device_flow)
        else:
            LOGGER.info(
                "To connect GitHub Copilot, go to %s and enter code %s",
                device_flow.verification_uri,
                device_flow.user_code,
            )

        access_token = await self.poll_for_access_token(device_flow.device_code, device_flow.interval)
        copilot_token, expires_at = await self.exchange_copilot_token(access_token)
        self._write_tokens(
            {
                "access_token": access_token,
                "copilot_token": copilot_token,
                "expires_at": expires_at,
            }
        )

    async def get_access_token(self) -> str:
        tokens = self._read_tokens()
        access_token = tokens.get("access_token")
        if not access_token:
            raise CopilotAuthError("No access token found. Run authenticate() first.")
        return str(access_token)

    async def get_token(self) -> str:
        tokens = self._read_tokens()
        access_token = tokens.get("access_token")
        if not access_token:
            raise CopilotAuthError("No access token found. Run authenticate() first.")

        copilot_token = tokens.get("copilot_token")
        expires_at = self._parse_expires_at(tokens.get("expires_at"))
        if not copilot_token or self._is_expiring_soon(expires_at):
            refreshed_token, refreshed_expires_at = await self.exchange_copilot_token(str(access_token))
            tokens["copilot_token"] = refreshed_token
            tokens["expires_at"] = refreshed_expires_at
            self._write_tokens(tokens)
            copilot_token = refreshed_token
        return str(copilot_token)

    async def call(
        self,
        messages: Sequence[dict[str, str]],
        model: str = "gpt-4o",
        system_prompt: str | None = None,
    ) -> str:
        if model not in self.SUPPORTED_MODELS:
            raise CopilotAPIError(f"Unsupported model '{model}'. Supported: {', '.join(self.SUPPORTED_MODELS)}")

        token = await self.get_token()
        request_messages = list(messages)
        if system_prompt:
            request_messages = [{"role": "system", "content": system_prompt}] + request_messages

        payload = {
            "model": model,
            "messages": request_messages,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Editor-Version": self.EDITOR_VERSION,
            "Editor-Plugin-Version": self.EDITOR_PLUGIN_VERSION,
            "Copilot-Integration-Id": "vscode-chat",
            "User-Agent": self.USER_AGENT,
            "Accept": "text/event-stream",
        }

        for attempt in range(1, self.CHAT_MAX_ATTEMPTS + 1):
            should_retry_status = False
            retry_status_delay = 0.0
            output_chunks: list[str] = []

            try:
                async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                    async with client.stream("POST", self.CHAT_COMPLETIONS_URL, headers=headers, json=payload) as response:
                        if response.status_code >= 400:
                            error_text = (await response.aread()).decode("utf-8", errors="replace")
                            if (
                                response.status_code in self.CHAT_RETRYABLE_STATUS_CODES
                                and attempt < self.CHAT_MAX_ATTEMPTS
                            ):
                                retry_status_delay = self._retry_delay_seconds(attempt)
                                should_retry_status = True
                                LOGGER.warning(
                                    "Copilot API transient HTTP %s on attempt %s/%s. Retrying in %.1fs.",
                                    response.status_code,
                                    attempt,
                                    self.CHAT_MAX_ATTEMPTS,
                                    retry_status_delay,
                                )
                            else:
                                raise CopilotAPIError(
                                    f"Copilot API call failed ({response.status_code}): {error_text[:500]}"
                                )
                        else:
                            async for line in response.aiter_lines():
                                if not line.startswith("data:"):
                                    continue
                                data = line[len("data:") :].strip()
                                if not data or data == "[DONE]":
                                    if data == "[DONE]":
                                        break
                                    continue
                                try:
                                    chunk = json.loads(data)
                                except json.JSONDecodeError:
                                    continue

                                choices = chunk.get("choices") or []
                                if not choices:
                                    continue
                                choice = choices[0]
                                delta = choice.get("delta") or {}
                                content = delta.get("content")
                                if content:
                                    output_chunks.append(str(content))
                                    continue

                                message = choice.get("message") or {}
                                message_content = message.get("content")
                                if message_content:
                                    output_chunks.append(str(message_content))

                if should_retry_status:
                    await asyncio.sleep(retry_status_delay)
                    continue

                text = "".join(output_chunks).strip()
                if text:
                    return text

                if attempt < self.CHAT_MAX_ATTEMPTS:
                    empty_retry_delay = self._retry_delay_seconds(attempt)
                    LOGGER.warning(
                        "Copilot API returned an empty response on attempt %s/%s. Retrying in %.1fs.",
                        attempt,
                        self.CHAT_MAX_ATTEMPTS,
                        empty_retry_delay,
                    )
                    await asyncio.sleep(empty_retry_delay)
                    continue

                raise CopilotAPIError("Copilot API returned an empty response.")
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.NetworkError) as exc:
                if attempt >= self.CHAT_MAX_ATTEMPTS:
                    raise CopilotAPIError(
                        "Could not reach the Copilot API after multiple attempts. "
                        "Check your internet connection and DNS, then retry the build. "
                        f"Last error: {exc}"
                    ) from exc

                retry_delay = self._retry_delay_seconds(attempt)
                LOGGER.warning(
                    "Copilot API network error on attempt %s/%s: %s. Retrying in %.1fs.",
                    attempt,
                    self.CHAT_MAX_ATTEMPTS,
                    exc,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)

        raise CopilotAPIError("Copilot API call failed after retries.")

    @classmethod
    def available_models(cls) -> tuple[str, ...]:
        return cls.SUPPORTED_MODELS

    def _read_tokens(self) -> dict[str, Any]:
        if not self._tokens_path.exists():
            return {}
        try:
            return json.loads(self._tokens_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            LOGGER.warning("Failed to read Copilot token file: %s", exc)
            return {}

    def _write_tokens(self, tokens: dict[str, Any]) -> None:
        self._tokens_path.parent.mkdir(parents=True, exist_ok=True)
        self._tokens_path.write_text(json.dumps(tokens, indent=2), encoding="utf-8")

    @staticmethod
    def _is_expiring_soon(expires_at: int) -> bool:
        return expires_at <= int(time.time()) + 300

    @staticmethod
    def _parse_expires_at(raw_value: Any) -> int:
        if isinstance(raw_value, (int, float)):
            return int(raw_value)
        if isinstance(raw_value, str):
            if raw_value.isdigit():
                return int(raw_value)
            try:
                timestamp = datetime.fromisoformat(raw_value.replace("Z", "+00:00")).timestamp()
                return int(timestamp)
            except ValueError:
                return 0
        return 0

    @staticmethod
    def _extract_error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                message = str(payload.get("message", "")).strip()
                error = str(payload.get("error", "")).strip()
                if message and error:
                    return f"{message} ({error})"
                if message:
                    return message
                if error:
                    return error
                return json.dumps(payload)[:300]
            return str(payload)[:300]
        except ValueError:
            return response.text.strip()[:300] or "No response body"

    @staticmethod
    def _retry_delay_seconds(attempt: int) -> float:
        # Exponential backoff capped to keep the overall build responsive.
        return min(1.5 * (2 ** (attempt - 1)), 10.0)
