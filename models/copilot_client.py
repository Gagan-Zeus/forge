from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from copilot import CopilotClient as SDKCopilotClient
from copilot.client import SubprocessConfig
from copilot.session import PermissionHandler

LOGGER = logging.getLogger(__name__)


class CopilotAuthError(RuntimeError):
    """Raised when Copilot authentication fails."""


class CopilotAPIError(RuntimeError):
    """Raised when Copilot model calls fail."""


class CopilotClient:
    CHAT_MAX_ATTEMPTS = 4
    MAX_CALL_TIMEOUT_SECONDS = 300.0

    DEFAULT_MODELS = (
        "gpt-5.3-codex",
        "gpt-5.2-codex",
        "gpt-5.2",
        "gpt-5.4-mini",
        "gpt-5-mini",
        "gpt-4.1",
        "claude-haiku-4.5",
    )

    _known_models: tuple[str, ...] = DEFAULT_MODELS

    def __init__(
        self,
        timeout_seconds: float | None = None,
        cli_path: str | None = None,
        github_token: str | None = None,
    ) -> None:
        self._timeout_seconds = timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
        self._cli_path = (cli_path or os.getenv("COPILOT_CLI_PATH", "")).strip() or None
        self._github_token = (github_token or "").strip()

        self._sdk_client: SDKCopilotClient | None = None
        self._client_lock = asyncio.Lock()
        self._model_refresh_warning_emitted = False

    def is_authenticated(self) -> bool:
        return self._sdk_client is not None and self._sdk_client.get_state() == "connected"

    async def authenticate(self, on_device_code: Any = None) -> None:
        raise CopilotAuthError(
            "Device-flow authentication was removed. "
            "Authenticate Copilot CLI directly in your terminal with: copilot auth login"
        )

    async def get_token(self) -> str:
        raise CopilotAuthError(
            "Raw Copilot token access is not exposed by the Copilot SDK. "
            "Use CopilotClient.ensure_ready() and CopilotClient.call() instead."
        )

    async def get_access_token(self) -> str:
        token = self._github_token or os.getenv("GITHUB_TOKEN", "").strip()
        if token:
            return token
        raise CopilotAuthError(
            "GITHUB_TOKEN is required for GitHub push operations when using SDK-based Copilot auth."
        )

    async def ensure_ready(self) -> None:
        sdk_client = await self._ensure_sdk_client()
        try:
            auth_status = await sdk_client.get_auth_status()
        except Exception as exc:  # noqa: BLE001
            raise CopilotAuthError(
                "Failed to query GitHub Copilot CLI auth state. "
                "Make sure Copilot CLI is installed and reachable in PATH."
            ) from exc

        if not getattr(auth_status, "isAuthenticated", False):
            status_message = getattr(auth_status, "statusMessage", "") or "Not authenticated"
            raise CopilotAuthError(
                "GitHub Copilot CLI is not authenticated. "
                "Run `copilot auth login` in terminal and then send /start again. "
                f"Status: {status_message}"
            )

        await self.refresh_available_models()

    async def refresh_available_models(self) -> tuple[str, ...]:
        sdk_client = await self._ensure_sdk_client()
        try:
            models = await sdk_client.list_models()
            model_ids = tuple(sorted({str(item.id).strip() for item in models if getattr(item, "id", "")}))
            if model_ids:
                CopilotClient._known_models = model_ids
            self._model_refresh_warning_emitted = False
        except Exception as exc:  # noqa: BLE001
            message = f"Could not refresh Copilot SDK model list, using cached defaults: {exc}"
            known_partial_metadata_issue = (
                "missing required fields in modelcapabilities" in str(exc).lower()
            )

            if known_partial_metadata_issue:
                LOGGER.debug(message)
            elif self._model_refresh_warning_emitted:
                LOGGER.debug(message)
            else:
                LOGGER.warning(message)
                self._model_refresh_warning_emitted = True
        return CopilotClient._known_models

    @classmethod
    def available_models(cls) -> tuple[str, ...]:
        return cls._known_models

    async def call(
        self,
        messages: Sequence[dict[str, str]],
        model: str = "gpt-5-mini",
        system_prompt: str | None = None,
        on_assistant_delta: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> str:
        await self.ensure_ready()
        prompt = self._messages_to_prompt(messages)
        max_attempts = 1 if on_assistant_delta else self.CHAT_MAX_ATTEMPTS

        for attempt in range(1, max_attempts + 1):
            session = None
            unsubscribe = None
            streamed_chunks: list[str] = []
            pending_stream_tasks: set[asyncio.Task[None]] = set()
            loop = asyncio.get_running_loop()

            def _consume_stream_task(task: asyncio.Task[None]) -> None:
                pending_stream_tasks.discard(task)
                try:
                    task.result()
                except Exception:  # noqa: BLE001
                    LOGGER.debug("Ignoring assistant delta callback failure", exc_info=True)

            def _publish_stream_delta(delta_text: str) -> None:
                if not on_assistant_delta or not delta_text:
                    return
                try:
                    maybe_awaitable = on_assistant_delta(delta_text)
                except Exception:  # noqa: BLE001
                    LOGGER.debug("Ignoring assistant delta callback failure", exc_info=True)
                    return
                if asyncio.iscoroutine(maybe_awaitable):
                    task = loop.create_task(maybe_awaitable)
                    pending_stream_tasks.add(task)
                    task.add_done_callback(_consume_stream_task)

            def _on_session_event(event: Any) -> None:
                delta_text = self._extract_assistant_delta_text(event)
                if not delta_text:
                    return
                streamed_chunks.append(delta_text)
                _publish_stream_delta(delta_text)

            try:
                sdk_client = await self._ensure_sdk_client()
                session_kwargs: dict[str, Any] = {
                    "on_permission_request": PermissionHandler.approve_all,
                    "model": model,
                    "streaming": on_assistant_delta is not None,
                }
                if system_prompt:
                    session_kwargs["system_message"] = {
                        "mode": "append",
                        "content": system_prompt,
                    }

                session = await sdk_client.create_session(**session_kwargs)
                if on_assistant_delta is not None:
                    unsubscribe = session.on(_on_session_event)
                attempt_timeout = self._attempt_timeout_seconds(attempt)
                event = await self._send_and_wait(session=session, prompt=prompt, timeout_seconds=attempt_timeout)
                text = self._extract_assistant_text(event)

                if not text:
                    events = await session.get_messages()
                    text = self._extract_assistant_text_from_events(events)

                if not text and streamed_chunks:
                    text = "".join(streamed_chunks).strip()

                if pending_stream_tasks:
                    await asyncio.gather(*pending_stream_tasks, return_exceptions=True)

                if text:
                    return text

                if attempt < max_attempts:
                    retry_delay = self._retry_delay_seconds(attempt)
                    LOGGER.warning(
                        "Copilot SDK returned an empty response on attempt %s/%s. Retrying in %.1fs.",
                        attempt,
                        max_attempts,
                        retry_delay,
                    )
                    await asyncio.sleep(retry_delay)
                    continue

                raise CopilotAPIError("Copilot SDK returned an empty response.")
            except CopilotAuthError:
                raise
            except Exception as exc:  # noqa: BLE001
                if self._looks_like_auth_error(exc):
                    raise CopilotAuthError(
                        "GitHub Copilot CLI is not authenticated. "
                        "Run `copilot auth login` and retry. "
                        f"Details: {exc}"
                    ) from exc

                is_timeout = self._looks_like_timeout_error(exc)
                if is_timeout:
                    configured_timeout = (
                        f"{self._timeout_seconds:.1f}s" if self._timeout_seconds is not None else "disabled"
                    )
                    LOGGER.warning(
                        "Copilot SDK timed out on attempt %s/%s (base timeout %s).",
                        attempt,
                        max_attempts,
                        configured_timeout,
                    )

                if attempt >= max_attempts:
                    partial_text = "".join(streamed_chunks).strip()
                    if partial_text:
                        if pending_stream_tasks:
                            await asyncio.gather(*pending_stream_tasks, return_exceptions=True)
                        return partial_text
                    if is_timeout:
                        await self._reset_sdk_client()
                    raise CopilotAPIError(
                        "Could not complete Copilot SDK request after multiple attempts. "
                        f"Last error: {exc}"
                    ) from exc

                if is_timeout:
                    await self._reset_sdk_client()
                retry_delay = self._retry_delay_seconds(attempt)
                LOGGER.warning(
                    "Copilot SDK network/session error on attempt %s/%s: %s. Retrying in %.1fs.",
                    attempt,
                    max_attempts,
                    exc,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)
            finally:
                if unsubscribe is not None:
                    unsubscribe()
                if session is not None:
                    try:
                        if hasattr(session, "disconnect"):
                            await session.disconnect()
                        else:
                            await session.destroy()
                    except Exception:  # noqa: BLE001
                        LOGGER.debug("Ignoring Copilot SDK session destroy failure", exc_info=True)

        raise CopilotAPIError("Copilot SDK call failed after retries.")

    async def _ensure_sdk_client(self) -> SDKCopilotClient:
        if self._sdk_client is not None and self._sdk_client.get_state() == "connected":
            return self._sdk_client

        async with self._client_lock:
            if self._sdk_client is not None and self._sdk_client.get_state() == "connected":
                return self._sdk_client

            if self._sdk_client is not None:
                try:
                    await self._sdk_client.stop()
                except Exception:  # noqa: BLE001
                    LOGGER.debug("Ignoring SDK stop failure while recreating client", exc_info=True)

            # Copilot model endpoints require CLI user auth; PAT-backed transport can fail.
            sdk_env = dict(os.environ)
            sdk_env.pop("GITHUB_TOKEN", None)
            sdk_env.pop("GH_TOKEN", None)
            config = SubprocessConfig(
                cli_path=self._cli_path,
                log_level="warning",
                use_logged_in_user=True,
                env=sdk_env,
            )

            self._sdk_client = SDKCopilotClient(config, auto_start=False)
            try:
                await self._sdk_client.start()
            except Exception as exc:  # noqa: BLE001
                raise CopilotAuthError(
                    "Could not start Copilot SDK client. "
                    "Ensure `copilot` CLI is installed and available in PATH."
                ) from exc

            return self._sdk_client

    async def _reset_sdk_client(self) -> None:
        async with self._client_lock:
            if self._sdk_client is None:
                return
            try:
                await self._sdk_client.stop()
            except Exception:  # noqa: BLE001
                LOGGER.debug("Ignoring SDK stop failure while resetting client", exc_info=True)
            finally:
                self._sdk_client = None

    @staticmethod
    def _messages_to_prompt(messages: Sequence[dict[str, str]]) -> str:
        normalized = [item for item in messages if item.get("content")]
        if not normalized:
            return ""

        if len(normalized) == 1 and normalized[0].get("role") == "user":
            return str(normalized[0].get("content", ""))

        lines = [
            "Use the conversation transcript below and respond to the latest user request.",
            "",
            "Transcript:",
        ]
        for item in normalized:
            role = str(item.get("role", "user")).strip().lower() or "user"
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            if role == "assistant":
                speaker = "Assistant"
            elif role == "system":
                speaker = "System"
            else:
                speaker = "User"
            lines.append(f"{speaker}:\n{content}")
            lines.append("")

        lines.append("Respond as the assistant to the latest user message.")
        return "\n".join(lines).strip()

    @staticmethod
    def _extract_assistant_text(event: Any) -> str:
        if event is None:
            return ""

        event_type_raw = getattr(event, "type", "")
        event_type = getattr(event_type_raw, "value", None) or str(event_type_raw)
        if event_type != "assistant.message":
            return ""

        data = getattr(event, "data", None)
        content = getattr(data, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(data, dict):
            value = data.get("content")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _extract_assistant_delta_text(event: Any) -> str:
        if event is None:
            return ""

        event_type_raw = getattr(event, "type", "")
        event_type = getattr(event_type_raw, "value", None) or str(event_type_raw)
        if event_type != "assistant.message_delta":
            return ""

        data = getattr(event, "data", None)
        delta_content = getattr(data, "delta_content", None)
        if isinstance(delta_content, str) and delta_content:
            return delta_content
        if isinstance(data, dict):
            value = data.get("delta_content") or data.get("deltaContent")
            if isinstance(value, str) and value:
                return value
        return ""

    @classmethod
    def _extract_assistant_text_from_events(cls, events: list[Any]) -> str:
        for event in reversed(events):
            text = cls._extract_assistant_text(event)
            if text:
                return text
        return ""

    @staticmethod
    def _retry_delay_seconds(attempt: int) -> float:
        return min(1.5 * (2 ** (attempt - 1)), 12.0)

    async def _send_and_wait(self, session: Any, prompt: str, timeout_seconds: float | None) -> Any:
        if timeout_seconds is not None:
            return await session.send_and_wait(prompt, timeout=timeout_seconds)

        idle_event = asyncio.Event()
        error_event: Exception | None = None
        last_assistant_message: Any = None

        def _handler(event: Any) -> None:
            nonlocal last_assistant_message, error_event
            event_type_raw = getattr(event, "type", "")
            event_type = getattr(event_type_raw, "value", None) or str(event_type_raw)
            if event_type == "assistant.message":
                last_assistant_message = event
            elif event_type == "session.idle":
                idle_event.set()
            elif event_type == "session.error":
                message = getattr(getattr(event, "data", None), "message", str(getattr(event, "data", "")))
                error_event = Exception(f"Session error: {message}")
                idle_event.set()

        unsubscribe = session.on(_handler)
        try:
            await session.send(prompt)
            await idle_event.wait()
            if error_event is not None:
                raise error_event
            return last_assistant_message
        finally:
            unsubscribe()

    def _attempt_timeout_seconds(self, attempt: int) -> float | None:
        if self._timeout_seconds is None:
            return None
        scaled_timeout = self._timeout_seconds * (1.5 ** (attempt - 1))
        return min(scaled_timeout, self.MAX_CALL_TIMEOUT_SECONDS)

    @staticmethod
    def _looks_like_auth_error(exc: Exception) -> bool:
        text = str(exc).lower()
        patterns = (
            "not authenticated",
            "auth.getstatus",
            "login required",
            "permission denied",
            "copilot auth",
            "no-auto-login",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def _looks_like_timeout_error(exc: Exception) -> bool:
        if isinstance(exc, TimeoutError | asyncio.TimeoutError):
            return True
        text = str(exc).lower()
        return "timeout" in text or "timed out" in text or "session.idle" in text
