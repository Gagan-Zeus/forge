from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent.orchestrator import BuildOrchestrator, BuildResult
from agent.planner import ProjectPlanner
from bot.session import SessionStore, UserSession
from models.copilot_client import CopilotAPIError, CopilotAuthError, CopilotClient
from tools.file_writer import FileWriter
from tools.github_pusher import GitHubPusher
from tools.shell_runner import ShellRunner

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5-mini"
MODEL_OPTIONS: list[tuple[str, str]] = [
    ("GPT-5.3-Codex", "gpt-5.3-codex"),
    ("GPT-5.2-Codex", "gpt-5.2-codex"),
    ("GPT-5.2", "gpt-5.2"),
    ("GPT-5.4 Mini", "gpt-5.4-mini"),
    ("GPT-5 Mini", "gpt-5-mini"),
    ("GPT-4.1", "gpt-4.1"),
    ("Claude Haiku 4.5", "claude-haiku-4.5"),
]
MODEL_LABELS = {value: label for label, value in MODEL_OPTIONS}
ALLOWED_MODEL_IDS = {value for _, value in MODEL_OPTIONS}
PROJECT_CHAT_HISTORY_TURNS = 8
PROJECT_CHAT_MAX_REPLY_CHARS = 3500
PROJECT_CHAT_STREAM_MIN_CHARS = 120
PROJECT_CHAT_STREAM_FLUSH_SECONDS = 1.0
PROJECT_ACTION_MAX_FILES = 12
PROJECT_FILE_TREE_MAX_ENTRIES = 240
PROJECT_FILE_TREE_MAX_CHARS = 12000


@dataclass
class _StreamedReplyState:
    message_id: int | None = None
    buffered_text: str = ""
    rendered_text: str = ""
    last_flush_at: float = 0.0


class _StreamingChatReplyPublisher:
    def __init__(self, application: Application, chat_id: int) -> None:
        self._application = application
        self._chat_id = chat_id
        self._state = _StreamedReplyState()
        self._lock = asyncio.Lock()

    def current_text(self) -> str:
        return self._state.buffered_text or self._state.rendered_text

    async def push_delta(self, delta_text: str) -> None:
        if not delta_text:
            return

        async with self._lock:
            self._state.buffered_text = _bounded_chat_reply(self._state.buffered_text + delta_text)
            if self._state.buffered_text == self._state.rendered_text:
                return

            now = asyncio.get_running_loop().time()
            pending_chars = len(self._state.buffered_text) - len(self._state.rendered_text)
            should_flush = pending_chars >= PROJECT_CHAT_STREAM_MIN_CHARS
            if not should_flush and (now - self._state.last_flush_at) >= PROJECT_CHAT_STREAM_FLUSH_SECONDS:
                should_flush = True
            if not should_flush and "\n\n" in delta_text:
                should_flush = True

            if should_flush:
                await self._flush_locked(self._state.buffered_text, now)

    async def finalize(self, final_text: str) -> None:
        text = _bounded_chat_reply(final_text.strip())
        if not text:
            text = self.current_text().strip()
        if not text:
            return

        async with self._lock:
            self._state.buffered_text = text
            if self._state.rendered_text == text:
                return
            now = asyncio.get_running_loop().time()
            await self._flush_locked(text, now)

    async def _flush_locked(self, text: str, now: float) -> None:
        if not text:
            return

        if self._state.message_id is None:
            try:
                sent_message = await self._application.bot.send_message(chat_id=self._chat_id, text=text)
            except TelegramError:
                LOGGER.exception("Failed to send streamed Telegram message to chat_id=%s", self._chat_id)
                return

            self._state.message_id = sent_message.message_id
            self._state.rendered_text = text
            self._state.last_flush_at = now
            return

        try:
            await self._application.bot.edit_message_text(
                chat_id=self._chat_id,
                message_id=self._state.message_id,
                text=text,
            )
            self._state.rendered_text = text
            self._state.last_flush_at = now
        except TelegramError as exc:
            if "message is not modified" in str(exc).lower():
                self._state.rendered_text = text
                self._state.last_flush_at = now
                return
            LOGGER.debug("Falling back to new streamed message for chat_id=%s", self._chat_id, exc_info=True)
            try:
                sent_message = await self._application.bot.send_message(chat_id=self._chat_id, text=text)
            except TelegramError:
                LOGGER.exception("Failed to send fallback streamed Telegram message to chat_id=%s", self._chat_id)
                return

            self._state.message_id = sent_message.message_id
            self._state.rendered_text = text
            self._state.last_flush_at = now


@dataclass
class RuntimeServices:
    sessions: SessionStore
    copilot_client: CopilotClient
    orchestrator: BuildOrchestrator
    file_writer: FileWriter
    shell_runner: ShellRunner
    github_pusher: GitHubPusher | None


def run_bot(telegram_token: str, github_username: str, github_token: str, projects_dir: str) -> None:
    if not telegram_token:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN in environment.")

    projects_path = Path(projects_dir).resolve()
    file_writer = FileWriter(projects_path)
    shell_runner = ShellRunner(allowed_root=projects_path)
    github_pat = github_token.strip()
    copilot_client = CopilotClient(
        cli_path=os.getenv("COPILOT_CLI_PATH", ""),
        github_token=github_pat,
    )
    planner = ProjectPlanner(copilot_client)

    async def github_access_token_provider() -> str:
        if github_pat:
            return github_pat
        return await copilot_client.get_access_token()

    github_pusher = GitHubPusher(
        github_username=github_username,
        access_token_provider=github_access_token_provider,
        shell_runner=shell_runner,
    )
    orchestrator = BuildOrchestrator(
        copilot_client=copilot_client,
        planner=planner,
        file_writer=file_writer,
        shell_runner=shell_runner,
        github_pusher=github_pusher,
    )

    app = Application.builder().token(telegram_token).build()
    app.bot_data["services"] = RuntimeServices(
        sessions=SessionStore(),
        copilot_client=copilot_client,
        orchestrator=orchestrator,
        file_writer=file_writer,
        shell_runner=shell_runner,
        github_pusher=github_pusher,
    )
    app.bot_data["build_tasks"] = {}
    app.bot_data["projects_root"] = projects_path
    app.bot_data["env_file"] = (Path.cwd() / ".env").resolve()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("cancel", cancel_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CallbackQueryHandler(model_selection_callback, pattern=r"^model:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, workspace_message_handler))
    app.add_error_handler(global_error_handler)

    LOGGER.info("Telegram bot is starting polling loop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = _chat_id(update)
    if chat_id is None:
        return

    services = _services(context)
    session = services.sessions.reset(chat_id, keep_auth=False)

    try:
        await services.copilot_client.ensure_ready()
        session.is_authenticated = True
        session.model = DEFAULT_MODEL
        await _safe_send_message(
            context.application,
            chat_id,
            (
                "Copilot SDK connected. Chatbot mode is active.\n"
                f"Default model: {MODEL_LABELS.get(session.model, session.model)}\n\n"
                "Commands:\n"
                "/model - change model\n"
                "/status - show current state\n"
                "/cancel - cancel running build\n"
                "/reset - reset chat state"
            ),
        )
        return
    except CopilotAuthError as exc:
        session.is_authenticated = False
        await _safe_send_message(
            context.application,
            chat_id,
            (
                "Copilot SDK is not ready.\n"
                "1) Install GitHub Copilot CLI if missing\n"
                "2) Run `copilot auth login` in terminal\n"
                "3) Send /start again\n\n"
                f"Details: {exc}"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        session.is_authenticated = False
        LOGGER.exception("Copilot SDK initialization failed for chat_id=%s", chat_id)
        await _safe_send_message(context.application, chat_id, f"Copilot SDK initialization failed: {exc}")


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = _chat_id(update)
    if chat_id is None:
        return

    services = _services(context)
    previous = services.sessions.get(chat_id)
    preserved_model = previous.model if previous.model in ALLOWED_MODEL_IDS else DEFAULT_MODEL
    session = services.sessions.reset(chat_id, keep_auth=True)
    session.model = preserved_model
    session.is_authenticated = previous.is_authenticated
    await _safe_send_message(
        context.application,
        chat_id,
        (
            "Session reset. Chatbot mode is active.\n"
            f"Current model: {MODEL_LABELS.get(session.model, session.model)}"
        ),
    )


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = _chat_id(update)
    if chat_id is None:
        return

    services = _services(context)
    session = services.sessions.get(chat_id)

    if not session.is_authenticated:
        await _safe_send_message(
            context.application,
            chat_id,
            "Use /start first to initialize Copilot SDK.",
        )
        return

    chosen = _resolve_model_choice(" ".join(context.args or []))
    if chosen:
        session.model = chosen
        await _safe_send_message(
            context.application,
            chat_id,
            f"Model set to {MODEL_LABELS.get(chosen, chosen)}.",
        )
        return

    if context.args:
        await _safe_send_message(
            context.application,
            chat_id,
            "Unsupported model. Select one from the list below.",
        )

    await _safe_send_message(
        context.application,
        chat_id,
        (
            f"Current model: {MODEL_LABELS.get(session.model, session.model)}\n"
            "Choose a model:"
        ),
        reply_markup=_model_keyboard(),
    )


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = _chat_id(update)
    if chat_id is None:
        return

    services = _services(context)
    session = services.sessions.get(chat_id)

    if not session.is_building:
        await _safe_send_message(context.application, chat_id, "No build is currently running.")
        return

    services.orchestrator.cancel(chat_id)
    session.build_progress = "Cancel requested. Stopping build..."
    await _safe_send_message(
        context.application,
        chat_id,
        "Cancellation requested. I will stop after the current active step.",
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = _chat_id(update)
    if chat_id is None:
        return

    services = _services(context)
    session = services.sessions.get(chat_id)

    if not session.is_authenticated:
        await _safe_send_message(
            context.application,
            chat_id,
            "Copilot SDK is not connected. Run /start after `copilot auth login` in terminal.",
        )
        return
    if session.is_building:
        await _safe_send_message(
            context.application,
            chat_id,
            f"Build in progress:\n{session.build_progress}",
        )
        return

    projects_root = _projects_root(context.application)
    active_project = session.active_project_path or "none"
    project_count = len(_list_generated_projects(projects_root, limit=200))
    await _safe_send_message(
        context.application,
        chat_id,
        (
            "Chatbot mode is active.\n"
            f"Authenticated: {'yes' if session.is_authenticated else 'no'}\n"
            f"Current model: {MODEL_LABELS.get(session.model, session.model)}\n"
            f"Generated projects: {project_count}\n"
            f"Active project: {active_project}"
        ),
    )


async def model_selection_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.message:
        return

    await query.answer()
    chat_id = query.message.chat_id
    services = _services(context)
    session = services.sessions.get(chat_id)

    if not session.is_authenticated:
        await _safe_send_message(context.application, chat_id, "Please authenticate first with /start.")
        return

    model = (query.data or "").split(":", maxsplit=1)[-1]
    if model not in ALLOWED_MODEL_IDS:
        await _safe_send_message(context.application, chat_id, "Unsupported model choice.")
        return

    session.model = model
    await _safe_send_message(context.application, chat_id, f"Selected model: {MODEL_LABELS.get(model, model)}")


async def workspace_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    chat_id = update.message.chat_id
    text = (update.message.text or "").strip()
    services = _services(context)
    session = services.sessions.get(chat_id)

    typing_stop = asyncio.Event()
    typing_task = asyncio.create_task(_typing_indicator_loop(context.application, chat_id, typing_stop))
    try:
        if not session.is_authenticated:
            await _safe_send_message(
                context.application,
                chat_id,
                "Use /start to initialize Copilot SDK. If needed, run `copilot auth login` in terminal first.",
            )
            return

        if session.is_building:
            await _safe_send_message(
                context.application,
                chat_id,
                "A build is already running. Use /status to check progress or /cancel to stop it.",
            )
            return

        if _looks_like_project_build_request(text):
            await _handle_project_build_request(context.application, services, chat_id, session, text)
            return

        await _handle_workspace_chat(context.application, services, chat_id, session, text)
    finally:
        typing_stop.set()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass


async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    LOGGER.exception("Unhandled Telegram error", exc_info=context.error)
    if isinstance(update, Update):
        chat_id = _chat_id(update)
        if chat_id is not None:
            await _safe_send_message(
                context.application,
                chat_id,
                "A temporary error occurred. The bot is still running; please retry your last message.",
            )


async def _handle_workspace_chat(
    application: Application,
    services: RuntimeServices,
    chat_id: int,
    session: UserSession,
    user_text: str,
) -> None:
    if not user_text:
        await _safe_send_message(application, chat_id, "Send a message and I will help from your generated workspace context.")
        return

    selected_model = session.model.strip()
    if selected_model not in ALLOWED_MODEL_IDS:
        selected_model = DEFAULT_MODEL

    projects_root = _projects_root(application)
    workspace_projects_text = _render_workspace_projects(projects_root, limit=40)
    env_status = _load_env_key_status(_env_file_path(application))
    env_keys_text = _render_env_key_status_lines(env_status)
    integration_status_text = _render_integration_status(env_status)
    active_project = session.active_project_path or "none"
    history_window = session.chat_history[-(PROJECT_CHAT_HISTORY_TURNS * 2) :]

    prompt = (
        "Respond as a workspace-aware coding chatbot.\n"
        f"Generated projects root: {projects_root}\n"
        f"Known generated projects:\n{workspace_projects_text}\n\n"
        f".env key status (values hidden):\n{env_keys_text}\n\n"
        f"Integration status:\n{integration_status_text}\n\n"
        f"Active project path: {active_project}\n\n"
        f"User message:\n{user_text}\n"
    )
    stream_publisher = _StreamingChatReplyPublisher(application, chat_id)

    try:
        response = await services.copilot_client.call(
            messages=[*history_window, {"role": "user", "content": prompt}],
            model=selected_model,
            system_prompt=(
                "You are a Telegram coding assistant. "
                "Only assume access to the generated projects directory provided in context and .env key metadata. "
                "Do not claim access outside that directory. "
                "If GITHUB_TOKEN is set, you can suggest GitHub push operations for generated projects. "
                "If VERCEL_TOKEN is set, acknowledge token availability; if automatic deploy wiring is not explicit, state that clearly and provide safe next steps."
            ),
            on_assistant_delta=stream_publisher.push_delta,
        )
    except CopilotAPIError as exc:
        LOGGER.warning("Workspace chatbot Copilot API failure for chat_id=%s: %s", chat_id, exc)
        partial_response = stream_publisher.current_text().strip()
        if partial_response:
            _append_project_chat_history(session, user_text, partial_response)
            await stream_publisher.finalize(partial_response)
            return
        await _safe_send_message(application, chat_id, f"Copilot request failed: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Workspace chatbot request failed for chat_id=%s", chat_id)
        await _safe_send_message(application, chat_id, f"Workspace chat failed: {exc}")
        return

    response = response.strip() or stream_publisher.current_text().strip()
    if not response:
        await _safe_send_message(application, chat_id, "Copilot returned an empty response.")
        return

    _append_project_chat_history(session, user_text, response)
    await stream_publisher.finalize(response)


async def _handle_project_build_request(
    application: Application,
    services: RuntimeServices,
    chat_id: int,
    session: UserSession,
    user_text: str,
) -> None:
    session.idea = user_text.strip()
    session.stack = _infer_stack_from_text(user_text)
    session.requirements = user_text.strip()
    session.push_to_github = _looks_like_push_request(user_text)
    session.repo_name = ""
    session.repo_visibility = "public" if "public" in user_text.lower() else "private"
    session.is_building = True
    session.build_progress = "Queued"

    await _safe_send_message(application, chat_id, "Starting project build in file-by-file mode.")

    async def _progress_update(message: str) -> None:
        session.build_progress = message
        await _safe_send_message(application, chat_id, message)

    try:
        result = await services.orchestrator.build_project(
            chat_id=chat_id,
            session=session,
            progress_callback=_progress_update,
        )
    finally:
        session.is_building = False

    if result.success:
        session.active_project_path = result.project_path
        session.active_github_url = result.github_url or ""
        session.project_context = _build_project_chat_context(copy.deepcopy(session), result)
        warnings_text = (
            "\nWarnings:\n" + "\n".join(f"- {item}" for item in result.warnings)
            if result.warnings
            else ""
        )
        await _safe_send_message(
            application,
            chat_id,
            (
                "Build completed.\n"
                f"Project path: {result.project_path}\n"
                f"Files created: {len(result.files_created)}"
                f"{warnings_text}"
            ),
        )
        return

    await _safe_send_message(
        application,
        chat_id,
        (
            "Build failed.\n"
            f"Project path: {result.project_path}\n"
            f"Error: {result.error or 'Unknown error'}"
        ),
    )


def _looks_like_project_build_request(user_text: str) -> bool:
    lowered = user_text.lower()
    build_verbs = (
        "build",
        "create",
        "generate",
        "make",
        "scaffold",
        "spin up",
    )
    project_targets = (
        "project",
        "app",
        "website",
        "web app",
        "frontend",
        "backend",
        "api",
        "bot",
        "dashboard",
    )
    has_build_verb = any(token in lowered for token in build_verbs)
    has_project_target = any(token in lowered for token in project_targets)
    return has_build_verb and has_project_target


def _infer_stack_from_text(user_text: str) -> str:
    lowered = user_text.lower()
    if any(token in lowered for token in ("next", "next.js")):
        return "next.js"
    if any(token in lowered for token in ("react", "vite", "typescript", "javascript", "node")):
        return "node/react"
    if any(token in lowered for token in ("fastapi", "flask", "django", "python", "pytest")):
        return "python"
    if any(token in lowered for token in ("go", "golang")):
        return "go"
    if "rust" in lowered:
        return "rust"
    return "general"

def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        payload = json.loads(fenced.group(1))
        if isinstance(payload, dict):
            return payload

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        payload = json.loads(text[start : end + 1])
        if isinstance(payload, dict):
            return payload

    raise ValueError("Could not parse JSON follow-up plan from model response.")


def _is_safe_project_relative_path(path: str) -> bool:
    candidate = PurePosixPath(path)
    if candidate.is_absolute():
        return False
    return ".." not in candidate.parts


def _read_project_file_if_exists(project_root: Path, relative_path: str) -> str:
    target = (project_root / relative_path).resolve()
    try:
        target.relative_to(project_root.resolve())
    except ValueError:
        return ""
    if not target.exists() or not target.is_file():
        return ""
    try:
        return target.read_text(encoding="utf-8")
    except OSError:
        return ""


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    fenced = re.match(r"^```[a-zA-Z0-9_+-]*\n(.*)\n```$", stripped, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    return stripped


def _looks_like_deletion_intent(user_text: str, change_request: str) -> bool:
    lowered = f"{user_text}\n{change_request}".lower()
    keywords = (
        "remove",
        "delete",
        "purge",
        "drop",
        "strip",
        "rewrite from scratch",
        "replace entire",
        "replace whole",
        "start over",
        "reset file",
    )
    return any(keyword in lowered for keyword in keywords)


def _is_overly_destructive_update(relative_path: str, current_content: str, updated_content: str) -> bool:
    current = current_content.strip()
    updated = updated_content.strip()
    if not current:
        return False
    if not updated:
        return True

    current_len = len(current)
    updated_len = len(updated)
    if updated_len < max(80, int(current_len * 0.45)):
        return True

    current_lines = [line.strip() for line in current.splitlines() if line.strip()]
    meaningful_lines = [line for line in current_lines if len(line) >= 3]
    if len(meaningful_lines) >= 10:
        retained = sum(1 for line in meaningful_lines if line in updated_content)
        retention_ratio = retained / len(meaningful_lines)
        if retention_ratio < 0.35:
            return True

    if relative_path.lower().endswith(".html"):
        current_lower = current_content.lower()
        has_stylesheet_link = bool(
            re.search(r"<link\\b[^>]*rel=[\"']stylesheet[\"']", updated_content, flags=re.IGNORECASE)
        )
        if "<style" in current_lower and "<style" not in updated_content.lower() and not has_stylesheet_link:
            return True

    return False


def _apply_file_language_boundary_rules(relative_path: str, content: str) -> tuple[str, dict[str, str]]:
    if not relative_path.lower().endswith(".html"):
        return content, {}
    return _externalize_inline_assets_from_html(relative_path, content)


def _recover_missing_html_assets(
    relative_path: str,
    current_content: str,
    updated_content: str,
) -> tuple[str, dict[str, str]]:
    if not relative_path.lower().endswith(".html"):
        return updated_content, {}

    recovered_html = updated_content
    recovered_assets: dict[str, str] = {}
    html_path = PurePosixPath(relative_path)
    folder = html_path.parent.as_posix()
    stem = html_path.stem or "page"
    css_href = f"{stem}.css"
    js_src = f"{stem}.js"
    css_path = _join_posix_path(folder, css_href)
    js_path = _join_posix_path(folder, js_src)

    current_style_blocks = _extract_html_inline_style_blocks(current_content)
    updated_has_style = bool(_extract_html_inline_style_blocks(updated_content))
    updated_has_stylesheet = bool(
        re.search(r"<link\\b[^>]*rel=[\"']stylesheet[\"']", updated_content, flags=re.IGNORECASE)
    )
    if current_style_blocks and not updated_has_style and not updated_has_stylesheet:
        recovered_assets[css_path] = "\n\n".join(current_style_blocks).strip() + "\n"
        recovered_html = _inject_stylesheet_link(recovered_html, css_href)

    current_script_blocks = _extract_html_inline_script_blocks(current_content)
    updated_has_inline_script = bool(_extract_html_inline_script_blocks(updated_content))
    updated_has_script_src = bool(re.search(r"<script\\b[^>]*\\bsrc=", updated_content, flags=re.IGNORECASE))
    if current_script_blocks and not updated_has_inline_script and not updated_has_script_src:
        recovered_assets[js_path] = "\n\n".join(current_script_blocks).strip() + "\n"
        recovered_html = _inject_script_src(recovered_html, js_src)

    return recovered_html, recovered_assets


def _externalize_inline_assets_from_html(relative_path: str, html_content: str) -> tuple[str, dict[str, str]]:
    html_path = PurePosixPath(relative_path)
    folder = html_path.parent.as_posix()
    stem = html_path.stem or "page"
    css_href = f"{stem}.css"
    js_src = f"{stem}.js"
    css_path = _join_posix_path(folder, css_href)
    js_path = _join_posix_path(folder, js_src)

    style_blocks = _extract_html_inline_style_blocks(html_content)
    script_blocks = _extract_html_inline_script_blocks(html_content)
    updated_html = re.sub(r"<style\b[^>]*>.*?</style>", "", html_content, flags=re.IGNORECASE | re.DOTALL)
    updated_html = re.sub(
        r"<script\b(?![^>]*\bsrc=)[^>]*>.*?</script>",
        "",
        updated_html,
        flags=re.IGNORECASE | re.DOTALL,
    )

    generated_assets: dict[str, str] = {}
    if style_blocks:
        generated_assets[css_path] = "\n\n".join(style_blocks).strip() + "\n"
        updated_html = _inject_stylesheet_link(updated_html, css_href)

    if script_blocks:
        generated_assets[js_path] = "\n\n".join(script_blocks).strip() + "\n"
        updated_html = _inject_script_src(updated_html, js_src)

    return updated_html, generated_assets


def _extract_html_inline_style_blocks(html_content: str) -> list[str]:
    return [
        block.strip()
        for block in re.findall(r"<style\b[^>]*>(.*?)</style>", html_content, flags=re.IGNORECASE | re.DOTALL)
        if block.strip()
    ]


def _extract_html_inline_script_blocks(html_content: str) -> list[str]:
    return [
        block.strip()
        for block in re.findall(
            r"<script\b(?![^>]*\bsrc=)[^>]*>(.*?)</script>",
            html_content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if block.strip()
    ]


def _inject_stylesheet_link(html_content: str, href: str) -> str:
    if re.search(rf"<link\\b[^>]*href=[\"']{re.escape(href)}[\"']", html_content, flags=re.IGNORECASE):
        return html_content

    tag = f'<link rel="stylesheet" href="{href}">'
    if re.search(r"</head>", html_content, flags=re.IGNORECASE):
        return re.sub(r"</head>", f"{tag}\n</head>", html_content, count=1, flags=re.IGNORECASE)

    return tag + "\n" + html_content


def _inject_script_src(html_content: str, src: str) -> str:
    if re.search(rf"<script\\b[^>]*src=[\"']{re.escape(src)}[\"']", html_content, flags=re.IGNORECASE):
        return html_content

    tag = f'<script src="{src}"></script>'
    if re.search(r"</body>", html_content, flags=re.IGNORECASE):
        return re.sub(r"</body>", f"{tag}\n</body>", html_content, count=1, flags=re.IGNORECASE)

    return html_content + "\n" + tag


def _join_posix_path(folder: str, file_name: str) -> str:
    normalized_folder = folder.strip()
    if not normalized_folder or normalized_folder == ".":
        return file_name
    return f"{normalized_folder}/{file_name}"


def _merge_generated_asset_content(target: dict[str, str], incoming: dict[str, str]) -> None:
    for path, content in incoming.items():
        snippet = content.strip()
        if not snippet:
            continue
        if path in target:
            target[path] = _merge_asset_content(target[path], snippet)
        else:
            target[path] = snippet + "\n"


def _merge_asset_content(existing: str, incoming: str) -> str:
    existing_trimmed = existing.strip()
    incoming_trimmed = incoming.strip()

    if not existing_trimmed:
        return incoming_trimmed + ("\n" if incoming_trimmed else "")
    if not incoming_trimmed:
        return existing
    if incoming_trimmed in existing_trimmed:
        return existing if existing.endswith("\n") else existing + "\n"

    return f"{existing_trimmed}\n\n{incoming_trimmed}\n"


def _append_project_chat_history(session: UserSession, user_text: str, assistant_text: str) -> None:
    session.chat_history.append({"role": "user", "content": user_text})
    session.chat_history.append({"role": "assistant", "content": assistant_text})
    max_history_items = PROJECT_CHAT_HISTORY_TURNS * 2
    if len(session.chat_history) > max_history_items:
        session.chat_history = session.chat_history[-max_history_items:]


def _bounded_chat_reply(text: str) -> str:
    return text if len(text) <= PROJECT_CHAT_MAX_REPLY_CHARS else text[:PROJECT_CHAT_MAX_REPLY_CHARS] + "..."


def _looks_like_push_request(user_text: str) -> bool:
    lowered = user_text.lower()
    keywords = (
        "push",
        "publish",
        "upload",
        "commit",
        "github",
        "repo",
        "repository",
        "open a pr",
    )
    return any(keyword in lowered for keyword in keywords)


def _repo_name_from_github_url(github_url: str) -> str:
    url = github_url.strip()
    if not url:
        return ""

    match = re.search(r"github\.com/[^/]+/([^/#?]+)", url)
    if not match:
        return ""

    repo = match.group(1)
    if repo.endswith(".git"):
        repo = repo[:-4]
    repo = re.sub(r"[^a-zA-Z0-9._-]+", "-", repo).strip("-.")
    return repo.lower()


def _derive_repo_name(project_root: Path, idea: str = "") -> str:
    base = re.sub(r"[^a-zA-Z0-9._-]+", "-", project_root.name).strip("-.")
    base_lower = base.lower()

    blocked_defaults = {"telegram-builder", "generated-project", "project", "app"}
    if (not base_lower or base_lower in blocked_defaults) and idea.strip():
        words = [word for word in re.split(r"\W+", idea.lower()) if word]
        slug = "-".join(words[:6]).strip("-.")
        if slug:
            return slug

    return base_lower or "project-repo"


def _build_project_chat_context(snapshot: UserSession, result: BuildResult) -> str:
    files = result.files_created[:120]
    file_lines = "\n".join(f"- {path}" for path in files) if files else "- none"
    warnings = "\n".join(f"- {item}" for item in result.warnings) if result.warnings else "- none"
    context = (
        f"Project name: {result.project_name}\n"
        f"Idea: {snapshot.idea}\n"
        f"Stack: {snapshot.stack}\n"
        f"Model used for generation: {snapshot.model}\n"
        f"Requirements: {snapshot.requirements or 'none'}\n"
        f"Local path: {result.project_path}\n"
        f"GitHub URL: {result.github_url or 'not pushed'}\n"
        "Files created:\n"
        f"{file_lines}\n"
        "Warnings:\n"
        f"{warnings}"
    )
    # Keep system prompt payload bounded for stable follow-up performance.
    return context[:8000]


def _services(context: ContextTypes.DEFAULT_TYPE) -> RuntimeServices:
    return cast(RuntimeServices, context.application.bot_data["services"])


def _chat_id(update: Update) -> int | None:
    if update.effective_chat:
        return update.effective_chat.id
    return None


def _model_keyboard(model_options: list[tuple[str, str]] | None = None) -> InlineKeyboardMarkup:
    options = model_options or MODEL_OPTIONS
    keyboard = [
        [InlineKeyboardButton(label, callback_data=f"model:{value}")]
        for label, value in options
    ]
    return InlineKeyboardMarkup(keyboard)


def _resolve_model_choice(raw_choice: str) -> str | None:
    raw = raw_choice.strip()
    if not raw:
        return None

    normalized = raw.lower().strip()
    normalized = re.sub(r"\s+", "-", normalized)
    if normalized in ALLOWED_MODEL_IDS:
        return normalized

    for label, value in MODEL_OPTIONS:
        if normalized == re.sub(r"\s+", "-", label.lower().strip()):
            return value

    return None


def _projects_root(application: Application) -> Path:
    configured = application.bot_data.get("projects_root")
    if isinstance(configured, Path):
        return configured.resolve()
    return Path("./generated_projects").resolve()


def _env_file_path(application: Application) -> Path:
    configured = application.bot_data.get("env_file")
    if isinstance(configured, Path):
        return configured.resolve()
    return (Path.cwd() / ".env").resolve()


def _list_generated_projects(projects_root: Path, limit: int = 40) -> list[Path]:
    if not projects_root.exists() or not projects_root.is_dir():
        return []

    projects = [
        item.resolve()
        for item in projects_root.iterdir()
        if item.is_dir()
    ]
    projects.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return projects[:limit]


def _render_workspace_projects(projects_root: Path, limit: int = 40) -> str:
    projects = _list_generated_projects(projects_root, limit=limit)
    if not projects:
        return "- none"

    lines: list[str] = []
    for project in projects:
        try:
            relative = project.relative_to(projects_root).as_posix()
        except ValueError:
            relative = project.name
        file_count = _count_project_files(project, limit=PROJECT_FILE_TREE_MAX_ENTRIES)
        lines.append(f"- {relative} ({file_count} files)")
    return "\n".join(lines)


def _count_project_files(project_root: Path, limit: int = PROJECT_FILE_TREE_MAX_ENTRIES) -> int:
    excluded_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv"}
    count = 0
    for root, dirs, file_names in os.walk(project_root):
        dirs[:] = [item for item in dirs if item not in excluded_dirs]
        root_path = Path(root)
        for file_name in file_names:
            path = root_path / file_name
            try:
                path.relative_to(project_root)
            except ValueError:
                continue
            count += 1
            if count >= limit:
                return count
    return count


def _load_env_key_status(env_file_path: Path) -> dict[str, bool]:
    status: dict[str, bool] = {}
    if not env_file_path.exists() or not env_file_path.is_file():
        return status

    try:
        raw_lines = env_file_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return status

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue

        value = value.strip()
        if value.startswith(("\"", "'")) and value.endswith(("\"", "'")) and len(value) >= 2:
            value = value[1:-1]

        runtime_value = os.getenv(key, value)
        status[key] = bool(str(runtime_value).strip())

    return status


def _render_env_key_status_lines(status: dict[str, bool]) -> str:
    if not status:
        return "- none"

    lines = [f"- {key}: {'set' if is_set else 'not set'}" for key, is_set in sorted(status.items())]
    return "\n".join(lines)


def _render_integration_status(status: dict[str, bool]) -> str:
    github_ready = status.get("GITHUB_TOKEN", False)
    vercel_ready = status.get("VERCEL_TOKEN", False)
    return "\n".join(
        [
            f"- GitHub push token available: {'yes' if github_ready else 'no'}",
            f"- Vercel token available: {'yes' if vercel_ready else 'no'}",
        ]
    )


async def _typing_indicator_loop(
    application: Application,
    chat_id: int,
    stop_event: asyncio.Event,
    interval_seconds: float = 4.0,
) -> None:
    while not stop_event.is_set():
        try:
            await application.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except TelegramError:
            LOGGER.debug("Failed to send typing action for chat_id=%s", chat_id, exc_info=True)

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except TimeoutError:
            continue


async def _safe_send_message(
    application: Application,
    chat_id: int,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    try:
        await application.bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)
    except TelegramError:
        LOGGER.exception("Failed to send Telegram message to chat_id=%s", chat_id)
