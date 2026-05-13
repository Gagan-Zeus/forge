from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

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
from models.opencode_client import OpenCodeAPIError, OpenCodeAuthError, OpenCodeClient
from models.unified_client import UnifiedModelClient, ModelAPIError, ModelAuthError
from tools.file_writer import FileWriter
from tools.github_pusher import GitHubPusher
from tools.shell_runner import ShellRunner

LOGGER = logging.getLogger(__name__)

# Provider settings
DEFAULT_PROVIDER = "copilot"

# Copilot models
COPILOT_DEFAULT_MODEL = "gpt-5-mini"
COPILOT_MODEL_OPTIONS = [
    ("GPT-5.3-Codex", "gpt-5.3-codex"),
    ("GPT-5.2-Codex", "gpt-5.2-codex"),
    ("GPT-5.2", "gpt-5.2"),
    ("GPT-5.4 Mini", "gpt-5.4-mini"),
    ("GPT-5 Mini", "gpt-5-mini"),
    ("GPT-4.1", "gpt-4.1"),
    ("Claude Haiku 4.5", "claude-haiku-4.5"),
]

# OpenCode models (Zen)
OPENCODE_DEFAULT_MODEL = "gpt-5.4-mini"
OPENCODE_MODEL_OPTIONS = [
    # GPT models
    ("GPT-5.5", "gpt-5.5"),
    ("GPT-5.5 Pro", "gpt-5.5-pro"),
    ("GPT-5.4", "gpt-5.4"),
    ("GPT-5.4 Pro", "gpt-5.4-pro"),
    ("GPT-5.4 Mini", "gpt-5.4-mini"),
    ("GPT-5.4 Nano", "gpt-5.4-nano"),
    ("GPT-5.3 Codex", "gpt-5.3-codex"),
    ("GPT-5.2", "gpt-5.2"),
    ("GPT-5.2 Codex", "gpt-5.2-codex"),
    ("GPT-5.1", "gpt-5.1"),
    ("GPT-5.1 Codex", "gpt-5.1-codex"),
    # Claude models
    ("Claude Opus 4.5", "claude-opus-4-5"),
    ("Claude Opus 4.6", "claude-opus-4-6"),
    ("Claude Opus 4.7", "claude-opus-4-7"),
    ("Claude Sonnet 4.5", "claude-sonnet-4-5"),
    ("Claude Sonnet 4.6", "claude-sonnet-4-6"),
    ("Claude Haiku 4.5", "claude-haiku-4-5"),
    ("Claude Haiku 3.5", "claude-3-5-haiku"),
    # Gemini models
    ("Gemini 3.1 Pro", "gemini-3.1-pro"),
    ("Gemini 3 Flash", "gemini-3-flash"),
    # Free models
    ("DeepSeek V4 Flash Free", "deepseek-v4-flash-free"),
    ("MiniMax M2.5 Free", "minimax-m2.5-free"),
    ("Qwen 3.5 Plus", "qwen3.5-plus"),
    ("Big Pickle", "big-pickle"),
]

# Combine all models
ALL_MODEL_OPTIONS = COPILOT_MODEL_OPTIONS + OPENCODE_MODEL_OPTIONS
MODEL_LABELS = {value: label for label, value in ALL_MODEL_OPTIONS}
COPILOT_MODEL_LABELS = {value: label for label, value in COPILOT_MODEL_OPTIONS}
OPENCODE_MODEL_LABELS = {value: label for label, value in OPENCODE_MODEL_OPTIONS}
COPILOT_MODEL_IDS = {value for _, value in COPILOT_MODEL_OPTIONS}
OPENCODE_MODEL_IDS = {value for _, value in OPENCODE_MODEL_OPTIONS}
ALLOWED_MODEL_IDS = COPILOT_MODEL_IDS | OPENCODE_MODEL_IDS
PROJECT_CHAT_HISTORY_TURNS = 8
PROJECT_CHAT_MAX_REPLY_CHARS = 3500
PROJECT_CHAT_STREAM_MIN_CHARS = 120
PROJECT_CHAT_STREAM_FLUSH_SECONDS = 1.0
PROJECT_ACTION_MAX_FILES = 12
PROJECT_FILE_TREE_MAX_ENTRIES = 240
PROJECT_FILE_TREE_MAX_CHARS = 12000


@dataclass
class _StreamedReplyState:
    buffered_text: str = ""
    sent_text: str = ""
    last_flush_at: float = 0.0


class _StreamingChatReplyPublisher:
    def __init__(self, application: Application, chat_id: int) -> None:
        self._application = application
        self._chat_id = chat_id
        self._state = _StreamedReplyState()
        self._lock = asyncio.Lock()

    def current_text(self) -> str:
        return self._state.buffered_text or self._state.sent_text

    async def push_delta(self, delta_text: str) -> None:
        if not delta_text:
            return

        async with self._lock:
            self._state.buffered_text = _bounded_chat_reply(self._state.buffered_text + delta_text)
            if self._state.buffered_text == self._state.sent_text:
                return

            now = asyncio.get_running_loop().time()
            pending_chars = len(self._state.buffered_text) - len(self._state.sent_text)
            should_flush = pending_chars >= PROJECT_CHAT_STREAM_MIN_CHARS
            if not should_flush and (now - self._state.last_flush_at) >= PROJECT_CHAT_STREAM_FLUSH_SECONDS:
                should_flush = True
            if not should_flush and "\n\n" in delta_text:
                should_flush = True

            if should_flush:
                await self._flush_locked(now)

    async def finalize(self, final_text: str) -> None:
        text = _bounded_chat_reply(final_text.strip())
        if not text:
            text = self.current_text().strip()
        if not text:
            return

        async with self._lock:
            self._state.buffered_text = text
            if self._state.sent_text == text:
                return
            now = asyncio.get_running_loop().time()
            await self._flush_locked(now)

    async def _flush_locked(self, now: float) -> None:
        pending_text = self._pending_text(self._state.buffered_text, self._state.sent_text)
        if not pending_text:
            return

        try:
            await self._application.bot.send_message(chat_id=self._chat_id, text=pending_text)
            self._state.sent_text = self._state.buffered_text
            self._state.last_flush_at = now
        except TelegramError:
            LOGGER.exception("Failed to send streamed Telegram message to chat_id=%s", self._chat_id)

    @staticmethod
    def _pending_text(full_text: str, sent_text: str) -> str:
        if not full_text:
            return ""
        if not sent_text:
            return full_text
        if full_text.startswith(sent_text):
            return full_text[len(sent_text):]
        return full_text


@dataclass
class RuntimeServices:
    sessions: SessionStore
    model_client: UnifiedModelClient
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

    # Create Copilot client
    copilot_client = CopilotClient(
        cli_path=os.getenv("COPILOT_CLI_PATH", ""),
        github_token=github_pat,
        base_system_prompt_path=os.getenv("SYSTEM_PROMPT_PATH", ""),
    )

    # Create OpenCode client if API key is configured
    opencode_api_key = os.getenv("OPENCODE_API_KEY", "").strip()
    opencode_client = None
    if opencode_api_key:
        opencode_client = OpenCodeClient(
            api_key=opencode_api_key,
            base_url=os.getenv("OPENCODE_BASE_URL"),
            base_system_prompt_path=os.getenv("SYSTEM_PROMPT_PATH", ""),
        )

    # Create unified model client
    default_provider = os.getenv("DEFAULT_MODEL_PROVIDER", "copilot")
    if default_provider not in ("copilot", "opencode"):
        default_provider = "copilot"

    model_client = UnifiedModelClient(
        copilot_client=copilot_client,
        opencode_client=opencode_client,
        default_provider=default_provider,  # type: ignore[arg-type]
    )

    planner = ProjectPlanner(model_client)

    async def github_access_token_provider() -> str:
        if github_pat:
            return github_pat
        # Try to get from Copilot (for backwards compatibility)
        try:
            return await copilot_client.get_access_token()
        except Exception:
            return ""

    github_pusher = GitHubPusher(
        github_username=github_username,
        access_token_provider=github_access_token_provider,
        shell_runner=shell_runner,
    )
    orchestrator = BuildOrchestrator(
        model_client=model_client,
        planner=planner,
        file_writer=file_writer,
        shell_runner=shell_runner,
        github_pusher=github_pusher,
    )

    app = Application.builder().token(telegram_token).build()
    app.bot_data["services"] = RuntimeServices(
        sessions=SessionStore(),
        model_client=model_client,
        orchestrator=orchestrator,
        file_writer=file_writer,
        shell_runner=shell_runner,
        github_pusher=github_pusher,
    )
    app.bot_data["build_tasks"] = {}
    app.bot_data["projects_root"] = projects_path
    app.bot_data["env_file"] = (Path.cwd() / ".env").resolve()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("provider", provider_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("create", create_command))
    app.add_handler(CommandHandler("project", project_command))
    app.add_handler(CommandHandler("github", github_command))
    app.add_handler(CommandHandler("update", update_command))
    app.add_handler(CommandHandler("install", install_command))
    app.add_handler(CommandHandler("delete", delete_command))
    app.add_handler(CommandHandler("cancel", cancel_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CallbackQueryHandler(model_selection_callback, pattern=r"^model:"))
    app.add_handler(CallbackQueryHandler(provider_selection_callback, pattern=r"^provider:"))
    app.add_handler(CallbackQueryHandler(project_selection_callback, pattern=r"^project:"))
    app.add_handler(
        MessageHandler(
            (filters.TEXT | filters.PHOTO | filters.Document.ALL) & ~filters.COMMAND,
            workspace_message_handler,
        )
    )
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
        await services.model_client.ensure_ready()
        session.is_authenticated = True
        session.model = services.model_client.get_default_model(session.provider)
        
        provider_name = "OpenCode" if session.provider == "opencode" else "Copilot"
        await _safe_send_message(
            context.application,
            chat_id,
            (
                f"{provider_name} SDK connected.\n"
                f"Provider: {session.provider}\n"
                f"Default model: {MODEL_LABELS.get(session.model, session.model)}\n\n"
                "Commands:\n"
                "/provider - change provider (copilot/opencode)\n"
                "/model - change model\n"
                "/create <prompt> - build a new project\n"
                "/project - select an existing project\n"
                "/github <repo_name> [--branch <branch>] - push to GitHub\n"
                "/update <prompt> - update the active project\n"
                "/delete - delete the active project directory\n"
                "/install - install project dependencies\n"
                "/status - show current state\n"
                "/cancel - cancel running build\n"
                "/reset - reset chat state"
            ),
        )
        return
    except ModelAuthError as exc:
        session.is_authenticated = False
        # Check if it's OpenCode-specific error
        if services.model_client.has_opencode:
            await _safe_send_message(
                context.application,
                chat_id,
                (
                    "Authentication failed.\n"
                    "For Copilot:\n"
                    "1) Install GitHub Copilot CLI if missing\n"
                    "2) Run `copilot -i auth login` in terminal\n\n"
                    "For OpenCode:\n"
                    "1) Get your API key at https://opencode.ai/auth\n"
                    "2) Set OPENCODE_API_KEY in your .env file\n\n"
                    f"Details: {exc}"
                ),
            )
        else:
            await _safe_send_message(
                context.application,
                chat_id,
                (
                    "Copilot SDK is not ready.\n"
                    "1) Install GitHub Copilot CLI if missing\n"
                    "2) Run `copilot -i auth login` in terminal\n"
                    "3) Send /start again\n\n"
                    f"Details: {exc}"
                ),
            )
    except Exception as exc:  # noqa: BLE001
        session.is_authenticated = False
        LOGGER.exception("Model client initialization failed for chat_id=%s", chat_id)
        await _safe_send_message(context.application, chat_id, f"Model client initialization failed: {exc}")


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = _chat_id(update)
    if chat_id is None:
        return

    services = _services(context)
    previous = services.sessions.get(chat_id)
    session = services.sessions.reset(chat_id, keep_auth=True)
    
    # Preserve provider and select appropriate default model
    session.provider = previous.provider
    if session.provider == "opencode":
        session.model = OPENCODE_DEFAULT_MODEL if previous.model not in OPENCODE_MODEL_IDS else previous.model
    else:
        session.model = COPILOT_DEFAULT_MODEL if previous.model not in COPILOT_MODEL_IDS else previous.model
    
    await _safe_send_message(
        context.application,
        chat_id,
        (
            "Session reset. Chatbot mode is active.\n"
            f"Provider: {session.provider}\n"
            f"Current model: {MODEL_LABELS.get(session.model, session.model)}"
        ),
    )


async def provider_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = _chat_id(update)
    if chat_id is None:
        return

    services = _services(context)
    session = services.sessions.get(chat_id)

    if not session.is_authenticated:
        await _safe_send_message(
            context.application,
            chat_id,
            "Use /start first to initialize.",
        )
        return

    args = context.args or []
    chosen = args[0].lower() if args else ""

    if chosen == "copilot":
        session.provider = "copilot"
        session.model = COPILOT_DEFAULT_MODEL
        await _safe_send_message(
            context.application,
            chat_id,
            "Provider set to Copilot.\nModel reset to " + COPILOT_DEFAULT_MODEL,
        )
        return
    elif chosen == "opencode":
        if not services.model_client.is_provider_available("opencode"):
            await _safe_send_message(
                context.application,
                chat_id,
                "OpenCode is not configured. Set OPENCODE_API_KEY in your .env file.",
            )
            return
        session.provider = "opencode"
        session.model = OPENCODE_DEFAULT_MODEL
        await _safe_send_message(
            context.application,
            chat_id,
            "Provider set to OpenCode.\nModel reset to " + OPENCODE_DEFAULT_MODEL,
        )
        return
    elif chosen:
        await _safe_send_message(
            context.application,
            chat_id,
            "Unknown provider. Choose copilot or opencode.",
        )
        return

    # Show provider selection
    keyboard = []
    if services.model_client.is_provider_available("copilot"):
        keyboard.append([InlineKeyboardButton("Copilot", callback_data="provider:copilot")])
    if services.model_client.is_provider_available("opencode"):
        keyboard.append([InlineKeyboardButton("OpenCode", callback_data="provider:opencode")])

    if not keyboard:
        await _safe_send_message(
            context.application,
            chat_id,
            "No providers available. Check your configuration.",
        )
        return

    await _safe_send_message(
        context.application,
        chat_id,
        f"Current provider: {session.provider}\nChoose a provider:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def provider_selection_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.message:
        return

    await query.answer()
    chat_id = query.message.chat_id
    services = _services(context)
    session = services.sessions.get(chat_id)

    provider = (query.data or "").split(":", maxsplit=1)[-1]
    if provider not in ("copilot", "opencode"):
        return

    if provider == "opencode" and not services.model_client.is_provider_available("opencode"):
        await _safe_send_message(
            context.application,
            chat_id,
            "OpenCode is not configured. Set OPENCODE_API_KEY in your .env file.",
        )
        return

    session.provider = provider  # type: ignore[assignment]
    if provider == "copilot":
        session.model = COPILOT_DEFAULT_MODEL
    else:
        session.model = OPENCODE_DEFAULT_MODEL

    provider_name = "OpenCode" if provider == "opencode" else "Copilot"
    await _safe_send_message(
        context.application,
        chat_id,
        f"Provider set to {provider_name}.\nModel reset to {session.model}",
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
            "Use /start first to initialize.",
        )
        return

    chosen = _resolve_model_choice(" ".join(context.args or ""), session.provider)
    if chosen:
        # Validate that the model is compatible with the current provider
        if session.provider == "copilot" and chosen not in COPILOT_MODEL_IDS:
            await _safe_send_message(
                context.application,
                chat_id,
                f"Model '{chosen}' is not available with Copilot. Use /provider to switch providers.",
            )
            return
        if session.provider == "opencode" and chosen not in OPENCODE_MODEL_IDS:
            await _safe_send_message(
                context.application,
                chat_id,
                f"Model '{chosen}' is not available with OpenCode. Use /provider to switch providers.",
            )
            return
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
            f"Provider: {session.provider}\n"
            f"Current model: {MODEL_LABELS.get(session.model, session.model)}\n"
            "Choose a model:"
        ),
        reply_markup=_model_keyboard(session.provider),
    )


async def create_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

    if session.is_building:
        await _safe_send_message(
            context.application,
            chat_id,
            "A build is already running. Use /status to check progress or /cancel to stop it.",
        )
        return

    project_prompt = " ".join(context.args or []).strip()
    if not project_prompt:
        await _safe_send_message(
            context.application,
            chat_id,
            "Usage: /create <build prompt>\nExample: /create build a hello world html page",
        )
        return

    await _handle_project_build_request(
        context.application,
        services,
        chat_id,
        session,
        project_prompt,
    )


async def project_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = _chat_id(update)
    if chat_id is None:
        return

    services = _services(context)
    session = services.sessions.get(chat_id)

    if not session.is_authenticated:
        await _safe_send_message(context.application, chat_id, "Use /start first to initialize Copilot SDK.")
        return

    projects_root = _projects_root(context.application)
    projects = _list_generated_projects(projects_root, limit=50)

    if not projects:
        await _safe_send_message(context.application, chat_id, "No generated projects found in your projects directory.")
        return

    keyboard = []
    for p in projects:
        name = p.name
        keyboard.append([InlineKeyboardButton(name, callback_data=f"project:{name}")])

    await _safe_send_message(
        context.application,
        chat_id,
        "Select a project to set as active:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def project_selection_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.message:
        return

    await query.answer()
    chat_id = query.message.chat_id
    services = _services(context)
    session = services.sessions.get(chat_id)

    project_name = (query.data or "").split(":", maxsplit=1)[-1]
    projects_root = _projects_root(context.application)
    project_path = projects_root / project_name

    if not project_path.exists() or not project_path.is_dir():
        await _safe_send_message(context.application, chat_id, f"Project '{project_name}' no longer exists.")
        return

    session.active_project_path = str(project_path)
    # Re-build project context for chatbot and updates
    session.project_context = f"Project name: {project_name}\nLocal path: {session.active_project_path}"
    
    await _safe_send_message(
        context.application,
        chat_id,
        f"Active project set to: {project_name}\n",
    )


async def github_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = _chat_id(update)
    if chat_id is None:
        return

    services = _services(context)
    session = services.sessions.get(chat_id)

    if not session.is_authenticated:
        await _safe_send_message(context.application, chat_id, "Use /start first to initialize Copilot SDK.")
        return

    if not session.active_project_path:
        await _safe_send_message(context.application, chat_id, "No active project found. Use /project or /create first.")
        return

    # Normalize dashes: em-dash (—) and en-dash (–) to standard dashes
    args = [a.replace("—", "--").replace("–", "-") for a in (context.args or [])]
    repo_name = ""
    branch = "main"

    i = 0
    while i < len(args):
        if args[i] in ("--branch", "-b") and i + 1 < len(args):
            branch = args[i+1]
            i += 2
        elif args[i].startswith("-"):
            i += 1
        else:
            if not repo_name:
                repo_name = args[i]
            i += 1

    if not repo_name:
        repo_name = session.repo_name

    if not repo_name:
        project_path = Path(session.active_project_path)
        repo_name = _derive_repo_name(project_path)
    
    session.repo_name = repo_name

    await _safe_send_message(context.application, chat_id, f"Pushing project to GitHub repository '{repo_name}' on branch '{branch}'...")

    try:
        if not services.github_pusher:
             await _safe_send_message(context.application, chat_id, "GitHub pusher not configured.")
             return
             
        github_url = await services.github_pusher.push_project(
            project_path=Path(session.active_project_path),
            repo_name=repo_name,
            visibility=session.repo_visibility,
            branch=branch
        )
        session.active_github_url = github_url
        await _safe_send_message(context.application, chat_id, f"Project successfully pushed to GitHub: {github_url}")
    except Exception as exc:
        LOGGER.exception("GitHub push failed for chat_id=%s", chat_id)
        await _safe_send_message(context.application, chat_id, f"GitHub push failed: {exc}")


async def update_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = _chat_id(update)
    if chat_id is None:
        return

    services = _services(context)
    session = services.sessions.get(chat_id)

    if not session.is_authenticated:
        await _safe_send_message(context.application, chat_id, "Use /start first to initialize Copilot SDK.")
        return

    if session.is_building:
        await _safe_send_message(context.application, chat_id, "A task is already running. Use /status to check progress.")
        return

    if not session.active_project_path:
        await _safe_send_message(context.application, chat_id, "No active project found. Use /project to select one or /create to generate a new one.")
        return

    update_prompt = " ".join(context.args or []).strip()
    if not update_prompt:
        await _safe_send_message(context.application, chat_id, "Usage: /update <changes to make>\nExample: /update add a login page")
        return

    session.is_building = True
    session.build_progress = "Initializing update..."

    async def _progress_update(message: str) -> None:
        session.build_progress = message
        await _safe_send_message(context.application, chat_id, message)

    try:
        result = await services.orchestrator.update_project(
            chat_id=chat_id,
            session=session,
            update_prompt=update_prompt,
            progress_callback=_progress_update,
        )
    finally:
        session.is_building = False

    if result.success:
        changed = result.files_created
        msg = f"Update completed successfully.\nFiles changed: {len(changed)}"
        if changed:
            msg += "\n\nChanges:\n" + "\n".join(f"- {f}" for f in changed[:20])
            if len(changed) > 20:
                msg += f"\n... and {len(changed) - 20} more."
        await _safe_send_message(context.application, chat_id, msg)
    else:
        await _safe_send_message(context.application, chat_id, f"Update failed: {result.error}")


async def delete_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = _chat_id(update)
    if chat_id is None:
        return

    services = _services(context)
    session = services.sessions.get(chat_id)

    if not session.is_authenticated:
        await _safe_send_message(context.application, chat_id, "Use /start first to initialize Copilot SDK.")
        return

    if not session.active_project_path:
        await _safe_send_message(context.application, chat_id, "No active project found. Use /project to select one or /create to generate a new one.")
        return

    project_path = Path(session.active_project_path)
    projects_root = _projects_root(context.application)

    # Ensure the project path is within the projects_root for safety
    try:
        project_path.relative_to(projects_root.resolve())
    except ValueError:
        await _safe_send_message(context.application, chat_id, f"Project path '{project_path}' is outside the projects directory. Deletion aborted.")
        return

    if not project_path.exists():
        await _safe_send_message(context.application, chat_id, f"Project path '{project_path}' does not exist.")
        session.active_project_path = ""
        session.active_github_url = ""
        return

    # Delete the project directory
    try:
        import shutil
        shutil.rmtree(project_path)
        await _safe_send_message(
            context.application,
            chat_id,
            f"Project '{project_path.name}' has been deleted successfully."
        )
        # Clear the active project from session
        session.active_project_path = ""
        session.active_github_url = ""
        session.repo_name = ""
        session.project_context = ""
    except Exception as exc:
        LOGGER.exception("Failed to delete project for chat_id=%s", chat_id)
        await _safe_send_message(context.application, chat_id, f"Failed to delete project: {exc}")


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


async def install_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = _chat_id(update)
    if chat_id is None:
        return

    services = _services(context)
    session = services.sessions.get(chat_id)

    if not session.is_authenticated:
        await _safe_send_message(context.application, chat_id, "Use /start first to initialize Copilot SDK.")
        return

    if not session.active_project_path:
        await _safe_send_message(context.application, chat_id, "No active project found. Use /project to select one or /create to generate a new one.")
        return

    project_path = Path(session.active_project_path)

    if not project_path.exists():
        await _safe_send_message(context.application, chat_id, f"Project path does not exist: {project_path}")
        return

    # Detect project type and determine install command
    await _safe_send_message(context.application, chat_id, "Analyzing project structure and installing dependencies...")

    install_result = await _analyze_and_install_dependencies(services, project_path, session)

    if install_result["success"]:
        msg = f"✅ Dependencies installed successfully!\n\n{install_result['message']}"
        if install_result.get("fixes_applied"):
            msg += f"\n\n🔧 Fixes applied:\n" + "\n".join(f"• {fix}" for fix in install_result["fixes_applied"])
        if install_result.get("commands_executed"):
            msg += f"\n\nCommands executed:\n" + "\n".join(f"• {cmd}" for cmd in install_result["commands_executed"])
        await _safe_send_message(context.application, chat_id, msg)
    else:
        msg = f"❌ Installation failed:\n\n{install_result['message']}"
        if install_result.get("fixes_applied"):
            msg += f"\n\n🔧 Attempted fixes:\n" + "\n".join(f"• {fix}" for fix in install_result["fixes_applied"])
        await _safe_send_message(context.application, chat_id, msg)


async def _analyze_and_install_dependencies(
    services: RuntimeServices,
    project_path: Path,
    session: UserSession,
) -> dict[str, Any]:
    """Analyze project, install dependencies, and fix issues if installation fails."""
    results = []
    commands_executed = []
    fixes_applied = []

    # Detect virtual environment
    venv_path, venv_python = _detect_virtual_env(project_path)

    # Check for various dependency files
    dependency_files = {
        "package.json": {
            "name": "Node.js/npm",
            "commands": ["npm install"],
            "lock_files": ["package-lock.json", "yarn.lock", "pnpm-lock.yaml"],
            "fixable": True,
        },
        "requirements.txt": {
            "name": "Python pip",
            "commands": _build_python_commands(venv_python, "-m pip install -r requirements.txt"),
            "lock_files": [],
            "fixable": True,
            "venv_required": True,
        },
        "pyproject.toml": {
            "name": "Python (Poetry/PDM/uv)",
            "commands": _get_pyproject_commands(project_path, venv_python),
            "lock_files": ["poetry.lock", "pdm.lock", "uv.lock"],
            "fixable": True,
            "venv_required": True,
        },
        "setup.py": {
            "name": "Python setuptools",
            "commands": _build_python_commands(venv_python, "-m pip install -e ."),
            "lock_files": [],
            "fixable": True,
            "venv_required": True,
        },
        "Pipfile": {
            "name": "Python pipenv",
            "commands": ["pipenv install"],
            "lock_files": ["Pipfile.lock"],
            "fixable": False,
            "venv_required": True,
        },
        "Cargo.toml": {
            "name": "Rust Cargo",
            "commands": ["cargo build"],
            "lock_files": ["Cargo.lock"],
            "fixable": False,
        },
        "go.mod": {
            "name": "Go modules",
            "commands": ["go mod download", "go mod tidy"],
            "lock_files": ["go.sum"],
            "fixable": True,
        },
        "Gemfile": {
            "name": "Ruby Bundler",
            "commands": ["bundle install"],
            "lock_files": ["Gemfile.lock"],
            "fixable": False,
        },
        "composer.json": {
            "name": "PHP Composer",
            "commands": ["composer install"],
            "lock_files": ["composer.lock"],
            "fixable": False,
        },
        "pom.xml": {
            "name": "Java Maven",
            "commands": ["mvn install -DskipTests"],
            "lock_files": [],
            "fixable": False,
        },
        "build.gradle": {
            "name": "Java Gradle",
            "commands": ["gradle build -x test"],
            "lock_files": [],
            "fixable": False,
        },
        # Additional lock files for reference
        "uv.lock": {
            "name": "Python uv (lock file)",
            "commands": ["uv sync", "uv pip install -r pyproject.toml"],
            "lock_files": [],
            "fixable": True,
            "venv_required": True,
        },
    }

    found_dependencies = []

    # Scan for dependency files
    for filename, config in dependency_files.items():
        file_path = project_path / filename
        if file_path.exists():
            found_dependencies.append((filename, config))

    if not found_dependencies:
        return {
            "success": True,
            "message": "No dependency files found in the project. The project may not require dependency installation.",
            "commands_executed": [],
            "fixes_applied": [],
        }

    # Install dependencies for each detected type
    for filename, config in found_dependencies:
        LOGGER.info("Installing dependencies for %s project (file: %s)", config["name"], filename)

        # For Python projects, ensure virtual environment exists
        if config.get("venv_required", False):
            venv_path, venv_python = _detect_virtual_env(project_path)
            if not venv_path:
                LOGGER.info("No virtual environment found for Python project, creating one...")
                venv_result = await services.shell_runner.run("python -m venv .venv", project_path)
                if venv_result["success"]:
                    fixes_applied.append("Created virtual environment (.venv)")
                    # Re-detect to get the new Python path
                    venv_path, venv_python = _detect_virtual_env(project_path)
                    # Update commands to use venv Python
                    if filename == "requirements.txt":
                        config["commands"] = _build_python_commands(venv_python, "-m pip install -r requirements.txt")
                    elif filename == "pyproject.toml":
                        config["commands"] = _get_pyproject_commands(project_path, venv_python)
                    elif filename == "setup.py":
                        config["commands"] = _build_python_commands(venv_python, "-m pip install -e .")
                else:
                    results.append(f"⚠️ {config['name']}: Could not create virtual environment, will try system Python")

        # Try standard commands first
        install_success = False
        last_error = ""
        last_output = ""

        for command in config["commands"]:
            result = await services.shell_runner.run(command, project_path)
            commands_executed.append(command)

            if result["success"]:
                results.append(f"✅ {config['name']}: Installed successfully using `{command}`")
                install_success = True
                break
            else:
                last_error = result.get("error", "")
                last_output = result.get("output", "")
                # Try next command if this one failed
                if result.get("exit_code") != 0:
                    continue

        # If install failed and this dependency type is fixable, try to fix it
        if not install_success and config.get("fixable", False):
            LOGGER.info("Install failed for %s, attempting to fix...", config["name"])
            fix_result = await _attempt_dependency_fix(
                services, session, project_path, filename, config, last_error, last_output
            )

            if fix_result["fixed"]:
                fixes_applied.extend(fix_result["fixes"])

                # Retry install after fix
                for command in config["commands"]:
                    retry_result = await services.shell_runner.run(command, project_path)
                    commands_executed.append(f"{command} (after fix)")

                    if retry_result["success"]:
                        results.append(f"✅ {config['name']}: Installed successfully after fixes using `{command}`")
                        install_success = True
                        break
                    else:
                        last_error = retry_result.get("error", "")

            if not install_success:
                # Report failure with fix attempts
                error_msg = f"❌ {config['name']}: Failed to install dependencies"
                if last_error:
                    error_msg += f"\n   Error: {last_error[:500]}"
                if fix_result["fixes"]:
                    error_msg += f"\n   Attempted fixes: {', '.join(fix_result['fixes'])}"
                results.append(error_msg)
        elif not install_success:
            # Not fixable, just report failure
            error_msg = f"❌ {config['name']}: Failed to install dependencies"
            if last_error:
                error_msg += f"\n   Error: {last_error[:500]}"
            results.append(error_msg)

    # Also check for virtual environment setup
    venv_paths = [project_path / ".venv", project_path / "venv", project_path / "env"]
    venv_found = any(v.exists() for v in venv_paths)

    if venv_found:
        results.append("ℹ️ Virtual environment detected. Make sure to activate it before running the project.")

    # Success if at least one dependency type was installed
    success = any("✅" in r for r in results)

    if success:
        validation = await _validate_and_fix_installed_project(services, session, project_path)
        commands_executed.extend(validation["commands_executed"])
        fixes_applied.extend(validation["fixes_applied"])
        results.append(validation["message"])
        success = bool(validation["success"])

    return {
        "success": success,
        "message": "\n\n".join(results),
        "commands_executed": commands_executed,
        "fixes_applied": fixes_applied,
    }


async def _attempt_dependency_fix(
    services: RuntimeServices,
    session: UserSession,
    project_path: Path,
    filename: str,
    config: dict[str, Any],
    error_output: str,
    command_output: str,
) -> dict[str, Any]:
    """Attempt to fix dependency installation issues."""
    fixes_applied = []
    file_path = project_path / filename

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        LOGGER.warning("Could not read %s for fixing: %s", filename, e)
        return {"fixed": False, "fixes": [], "error": str(e)}

    # Use Copilot to analyze the error and suggest fixes
    prompt = (
        f"Analyze this dependency installation error and suggest fixes.\n\n"
        f"Dependency file: {filename}\n"
        f"Project type: {config['name']}\n\n"
        f"Command output:\n```\n{command_output[:2000]}\n```\n\n"
        f"Error output:\n```\n{error_output[:2000]}\n```\n\n"
        f"Current {filename} content:\n```\n{content[:3000]}\n```\n\n"
        "Provide specific fixes for common issues:\n"
        "1. Version conflicts - suggest version changes\n"
        "2. Missing dependencies - identify what's missing\n"
        "3. Invalid syntax - suggest corrections\n"
        "4. Platform-specific issues - suggest alternatives\n\n"
        "Return a JSON response with:\n"
        '{"fixes": [{"type": "version_update|remove_package|add_package|syntax_fix", '
        '"target": "package_name", "action": "description", '
        '"new_content": "optional replacement content"}]}'
    )

    try:
        response = await services.model_client.call(
            messages=[{"role": "user", "content": prompt}],
            model=session.model,
            system_prompt=(
                "You are a dependency resolution expert. Analyze installation errors and provide "
                "concrete fixes. Return valid JSON with specific actions to take."
            ),
        )

        # Try to parse the JSON response
        try:
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                fix_data = json.loads(json_match.group())
            else:
                # Try to parse the whole response
                fix_data = json.loads(response)

            fixes = fix_data.get("fixes", [])

            for fix in fixes:
                fix_type = fix.get("type", "")
                target = fix.get("target", "")
                new_content = fix.get("new_content", "")

                if fix_type == "version_update" and new_content:
                    # Write updated content
                    await services.file_writer.write_file(project_path, filename, new_content)
                    fixes_applied.append(f"Updated versions in {filename}")

                elif fix_type == "remove_package" and target:
                    # Remove problematic package
                    lines = content.splitlines()
                    filtered_lines = [line for line in lines if target not in line]
                    new_content = "\n".join(filtered_lines)
                    if len(filtered_lines) < len(lines):
                        await services.file_writer.write_file(project_path, filename, new_content)
                        fixes_applied.append(f"Removed problematic package: {target}")

                elif fix_type == "syntax_fix" and new_content:
                    await services.file_writer.write_file(project_path, filename, new_content)
                    fixes_applied.append(f"Fixed syntax in {filename}")

                elif fix_type == "add_package" and target:
                    # This is informational, user needs to add manually
                    fixes_applied.append(f"Suggestion: Add package {target}")

        except json.JSONDecodeError:
            LOGGER.warning("Could not parse fix response as JSON: %s", response[:200])

    except Exception as e:
        LOGGER.exception("Failed to get fix suggestions from Copilot")
        return {"fixed": False, "fixes": [], "error": str(e)}

    # Also try common fixes based on error patterns
    if not fixes_applied:
        common_fixes = await _apply_common_dependency_fixes(
            services, project_path, filename, error_output, content
        )
        fixes_applied.extend(common_fixes)

    return {
        "fixed": len(fixes_applied) > 0,
        "fixes": fixes_applied,
    }


async def _apply_common_dependency_fixes(
    services: RuntimeServices,
    project_path: Path,
    filename: str,
    error_output: str,
    content: str,
) -> list[str]:
    """Apply common dependency fixes based on error patterns."""
    fixes_applied = []
    lowered_error = error_output.lower()

    # Node.js/npm specific fixes
    if filename == "package.json":
        # Clear node_modules and reinstall
        if "enoent" in lowered_error or "cannot find module" in lowered_error:
            await services.shell_runner.run("rm -rf node_modules package-lock.json", project_path)
            fixes_applied.append("Cleared node_modules and lock file")

        # Try npm audit fix
        if "vulnerability" in lowered_error or "audit" in lowered_error:
            await services.shell_runner.run("npm audit fix --force", project_path)
            fixes_applied.append("Ran npm audit fix")

    # Python specific fixes
    if filename in ("requirements.txt", "pyproject.toml", "setup.py"):
        # Check if virtual environment is missing and create it
        venv_path, _ = _detect_virtual_env(project_path)
        if not venv_path and ("venv" in lowered_error or "virtual" in lowered_error):
            await services.shell_runner.run("python -m venv .venv", project_path)
            fixes_applied.append("Created virtual environment (.venv)")

        # For pip - Remove version pins for packages that can't be found
        if filename == "requirements.txt":
            if "could not find" in lowered_error or "no matching distribution" in lowered_error:
                lines = content.splitlines()
                new_lines = []
                removed_packages = []

                for line in lines:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        new_lines.append(line)
                        continue

                    # Check if this package is mentioned in the error
                    package_match = re.match(r'^([a-zA-Z0-9_-]+)', stripped)
                    if package_match:
                        package_name = package_match.group(1).lower()
                        if package_name in lowered_error:
                            removed_packages.append(package_name)
                            continue

                    new_lines.append(line)

                if removed_packages:
                    new_content = "\n".join(new_lines)
                    await services.file_writer.write_file(project_path, filename, new_content)
                    fixes_applied.append(f"Removed problematic packages: {', '.join(removed_packages)}")

            # Try upgrading pip
            if "pip" in lowered_error:
                await services.shell_runner.run("python -m pip install --upgrade pip", project_path)
                fixes_applied.append("Upgraded pip")

    if filename == "pyproject.toml":
        # Clear lock files based on tool being used
        if "poetry.lock" in lowered_error or "poetry" in lowered_error:
            await services.shell_runner.run("rm -f poetry.lock", project_path)
            fixes_applied.append("Removed poetry.lock for regeneration")

        if "pdm.lock" in lowered_error or "pdm" in lowered_error:
            await services.shell_runner.run("rm -f pdm.lock", project_path)
            fixes_applied.append("Removed pdm.lock for regeneration")

        if "uv.lock" in lowered_error or "uv" in lowered_error:
            await services.shell_runner.run("rm -f uv.lock", project_path)
            fixes_applied.append("Removed uv.lock for regeneration")

        # Try installing uv as a fast alternative
        if "pip" in lowered_error and "uv" not in lowered_error:
            await services.shell_runner.run("pip install uv", project_path)
            fixes_applied.append("Installed uv for faster Python package management")

    # Go specific fixes
    if filename == "go.mod":
        if "go mod tidy" in lowered_error or "missing" in lowered_error:
            await services.shell_runner.run("go mod tidy", project_path)
            fixes_applied.append("Ran go mod tidy")

    return fixes_applied


async def _validate_and_fix_installed_project(
    services: RuntimeServices,
    session: UserSession,
    project_path: Path,
    max_attempts: int = 2,
) -> dict[str, Any]:
    commands_executed: list[str] = []
    fixes_applied: list[str] = []

    for attempt in range(1, max_attempts + 1):
        validation = await _run_project_validation_checks(services, project_path)
        commands_executed.extend(validation["commands_executed"])
        if validation["success"]:
            suffix = "" if attempt == 1 else " after fixes"
            return {
                "success": True,
                "message": f"✅ Project validation passed{suffix}.",
                "commands_executed": commands_executed,
                "fixes_applied": fixes_applied,
            }

        if attempt >= max_attempts:
            return {
                "success": False,
                "message": f"❌ Project validation failed after install:\n{validation['issue'][:1200]}",
                "commands_executed": commands_executed,
                "fixes_applied": fixes_applied,
            }

        fix_result = await _attempt_project_validation_fix(
            services=services,
            session=session,
            project_path=project_path,
            issue=validation["issue"],
        )
        fixes_applied.extend(fix_result["fixes_applied"])
        if not fix_result["success"]:
            return {
                "success": False,
                "message": f"❌ Project validation failed and auto-fix did not apply changes:\n{validation['issue'][:1200]}",
                "commands_executed": commands_executed,
                "fixes_applied": fixes_applied,
            }

    return {
        "success": False,
        "message": "❌ Project validation failed.",
        "commands_executed": commands_executed,
        "fixes_applied": fixes_applied,
    }


async def _run_project_validation_checks(services: RuntimeServices, project_path: Path) -> dict[str, Any]:
    commands = _detect_validation_commands(project_path)
    commands_executed: list[str] = []
    issues: list[str] = []

    static_issue = _detect_static_python_issues(project_path)
    if static_issue:
        issues.append(static_issue)

    for command, timeout_seconds, timeout_success_pattern in commands:
        result = await services.shell_runner.run(command, project_path, timeout_seconds=timeout_seconds)
        commands_executed.append(command)
        output = _combine_command_output(result)

        if result["success"]:
            continue

        if result.get("exit_code") == 124 and timeout_success_pattern:
            if re.search(timeout_success_pattern, output, flags=re.IGNORECASE):
                continue

        if _is_ignorable_runtime_output(output):
            continue

        issues.append(f"Command failed: {command}\n{output}")

    if issues:
        return {
            "success": False,
            "issue": "\n\n".join(issues),
            "commands_executed": commands_executed,
        }

    return {
        "success": True,
        "issue": "",
        "commands_executed": commands_executed,
    }


def _detect_validation_commands(project_path: Path) -> list[tuple[str, float | None, str | None]]:
    commands: list[tuple[str, float | None, str | None]] = []

    package_json = project_path / "package.json"
    if package_json.exists():
        scripts = _read_package_scripts(package_json)
        if "build" in scripts:
            commands.append(("npm run build", 180.0, None))
        if "test" in scripts and _is_meaningful_test_script(scripts["test"]):
            commands.append((_build_npm_test_command(scripts["test"]), 120.0, None))
        return commands

    python_files = list(project_path.glob("*.py")) + list((project_path / "src").glob("*.py") if (project_path / "src").exists() else [])
    if python_files or (project_path / "requirements.txt").exists() or (project_path / "pyproject.toml").exists():
        _, python_cmd = _detect_virtual_env(project_path)
        commands.append((f'"{python_cmd}" -m compileall -q .', 60.0, None))
        entrypoint = _detect_python_entrypoint(project_path)
        if entrypoint:
            commands.append((f'"{python_cmd}" -W error::DeprecationWarning {entrypoint}', 5.0, r"running on|serving flask app|started|uvicorn running"))
        return commands

    if (project_path / "go.mod").exists():
        commands.append(("go test ./...", 120.0, None))
    elif (project_path / "Cargo.toml").exists():
        commands.append(("cargo test", 180.0, None))

    return commands


def _read_package_scripts(package_json: Path) -> dict[str, str]:
    try:
        payload = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    scripts = payload.get("scripts")
    if not isinstance(scripts, dict):
        return {}
    return {str(key): str(value) for key, value in scripts.items()}


def _is_meaningful_test_script(script: str) -> bool:
    lowered = script.lower()
    return "no test specified" not in lowered and "exit 1" not in lowered


def _build_npm_test_command(script: str) -> str:
    lowered = script.lower()
    if "vitest" in lowered:
        return "npm test -- --run"
    if "jest" in lowered or "react-scripts test" in lowered:
        return "npm test -- --watchAll=false"
    return "npm test"


def _detect_python_entrypoint(project_path: Path) -> str | None:
    for candidate in ("app.py", "main.py", "run.py", "server.py", "src/app.py", "src/main.py"):
        if (project_path / candidate).exists():
            return candidate
    return None


def _detect_static_python_issues(project_path: Path) -> str:
    offenders: list[str] = []
    excluded_dirs = {".venv", "venv", "env", "__pycache__", ".git"}
    for path in project_path.rglob("*.py"):
        if any(part in excluded_dirs for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "datetime.utcnow()" in text or "datetime.utcfromtimestamp(" in text:
            try:
                relative = path.relative_to(project_path).as_posix()
            except ValueError:
                relative = path.name
            offenders.append(relative)

    if not offenders:
        return ""
    return "Deprecated naive UTC datetime usage found in: " + ", ".join(offenders)


def _combine_command_output(result: dict[str, Any]) -> str:
    output = str(result.get("output", "")).strip()
    error = str(result.get("error", "")).strip()
    if output and error:
        return f"stdout:\n{output}\n\nstderr:\n{error}"
    return output or error or "Unknown command failure"


def _is_ignorable_runtime_output(output: str) -> bool:
    lowered = output.lower()
    ignorable = (
        "warning: this is a development server",
        "press ctrl+c to quit",
        "command timed out",
    )
    has_real_failure = any(token in lowered for token in ("traceback", "error:", "exception", "failed", "deprecationwarning"))
    return any(token in lowered for token in ignorable) and not has_real_failure


async def _attempt_project_validation_fix(
    services: RuntimeServices,
    session: UserSession,
    project_path: Path,
    issue: str,
) -> dict[str, Any]:
    file_tree = _render_directory_tree(project_path, max_depth=4)
    snippets = _collect_validation_context(project_path, issue)
    prompt = (
        "Fix the project validation/runtime issue with minimal file-by-file changes.\n"
        "Return only JSON in this exact shape:\n"
        '{"files": [{"path": "relative/path", "content": "full replacement file content"}]}\n\n'
        "Rules:\n"
        "- Only change files that are necessary to fix the issue.\n"
        "- Return full replacement content for each changed file.\n"
        "- Do not include markdown fences.\n"
        "- Preserve the app's existing behavior and styling.\n\n"
        f"Validation issue:\n{issue[:4000]}\n\n"
        f"Project tree:\n{file_tree[:4000]}\n\n"
        f"Relevant files:\n{snippets[:12000]}\n"
    )

    try:
        response = await services.model_client.call(
            messages=[{"role": "user", "content": prompt}],
            model=session.model,
            system_prompt="You are a senior engineer fixing validation failures. Return strict JSON only.",
        )
        payload = _extract_json_object(response)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Validation auto-fix planning failed")
        return {"success": False, "fixes_applied": [f"Auto-fix planning failed: {exc}"]}

    files = payload.get("files")
    if not isinstance(files, list):
        return {"success": False, "fixes_applied": []}

    fixes_applied: list[str] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        relative_path = str(item.get("path", "")).strip()
        content = item.get("content")
        if not relative_path or not isinstance(content, str):
            continue
        if not _is_safe_project_relative_path(relative_path):
            continue
        await services.file_writer.write_file(project_path, relative_path, _strip_code_fences(content))
        fixes_applied.append(f"Updated {relative_path}")

    return {"success": bool(fixes_applied), "fixes_applied": fixes_applied}


def _collect_validation_context(project_path: Path, issue: str) -> str:
    candidates = _validation_context_candidates(project_path, issue)
    chunks: list[str] = []
    for path in candidates[:12]:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            relative = path.relative_to(project_path).as_posix()
        except OSError:
            continue
        chunks.append(f"### {relative}\n{content[:5000]}")
    return "\n\n".join(chunks)


def _validation_context_candidates(project_path: Path, issue: str) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    for match in re.findall(r"([A-Za-z0-9_./-]+\.(?:py|js|jsx|ts|tsx|json|toml|txt|html|css))", issue):
        path = (project_path / match).resolve()
        try:
            path.relative_to(project_path.resolve())
        except ValueError:
            continue
        if path.exists() and path.is_file() and path not in seen:
            candidates.append(path)
            seen.add(path)

    priority = (
        "app.py", "main.py", "run.py", "server.py", "package.json", "requirements.txt",
        "pyproject.toml", "src/app.py", "src/main.py", "src/index.js", "src/main.jsx",
        "src/App.jsx", "src/App.tsx",
    )
    for relative in priority:
        path = project_path / relative
        if path.exists() and path.is_file() and path not in seen:
            candidates.append(path)
            seen.add(path)

    return candidates


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
            "Model client not connected. Run /start first.",
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
    
    # Get provider status
    provider_status = session.provider
    opencode_available = "✓" if services.model_client.has_opencode else "✗"
    
    await _safe_send_message(
        context.application,
        chat_id,
        (
            "Chatbot mode is active.\n"
            f"Authenticated: {'yes' if session.is_authenticated else 'no'}\n"
            f"Provider: {provider_status}\n"
            f"OpenCode available: {opencode_available}\n"
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

    message = update.message
    chat_id = message.chat_id
    text = (message.text or message.caption or "").strip()
    image_attachments = await _extract_image_attachments_from_message(message)
    services = _services(context)
    session = services.sessions.get(chat_id)

    typing_stop = asyncio.Event()
    typing_task = asyncio.create_task(_typing_indicator_loop(context.application, chat_id, typing_stop))
    try:
        if not session.is_authenticated:
            await _safe_send_message(
                context.application,
                chat_id,
                "Use /start to initialize Copilot SDK. If needed, run `copilot -i auth login` in terminal first.",
            )
            return

        if session.is_building:
            await _safe_send_message(
                context.application,
                chat_id,
                "A build is already running. Use /status to check progress or /cancel to stop it.",
            )
            return

        await _handle_workspace_chat(
            context.application,
            services,
            chat_id,
            session,
            text,
            image_attachments=image_attachments,
        )
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
    image_attachments: list[dict[str, str]] | None = None,
) -> None:
    attachments = image_attachments or []

    if not user_text and not attachments:
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

    # Build available tools description
    tools_description = _build_tools_description()

    prompt = (
        "Respond as a workspace-aware coding chatbot with tool execution capabilities.\n"
        f"Generated projects root: {projects_root}\n"
        f"Known generated projects:\n{workspace_projects_text}\n\n"
        f".env key status (values hidden):\n{env_keys_text}\n\n"
        f"Integration status:\n{integration_status_text}\n\n"
        f"Active project path: {active_project}\n\n"
        f"Available tools:\n{tools_description}\n\n"
        "When you need to use a tool, include a TOOL_CALL block in your response like:\n"
        "```tool\n"
        "{\n"
        '  "tool": "tool_name",\n'
        '  "params": {\n'
        '    "param1": "value1"\n'
        "  }\n"
        "}\n"
        "```\n"
        "You can make multiple tool calls in sequence if needed.\n\n"
        f"User message:\n{user_text}\n"
    )

    max_tool_iterations = 5
    current_iteration = 0
    conversation_messages = [*history_window, {"role": "user", "content": prompt}]
    final_response = ""

    try:
        while current_iteration < max_tool_iterations:
            response = await services.model_client.call(
                messages=conversation_messages,
                model=selected_model,
                system_prompt=(
                    "You are a Telegram coding assistant with tool execution capabilities. "
                    "You can execute shell commands, read files, list directories, and more. "
                    "When the user asks you to perform an action like 'show me the files', 'read this file', "
                    "'run this command', or 'summarize the project', use the available tools to actually "
                    "perform the task and provide real results. "
                    "Always use TOOL_CALL blocks when you need to execute commands or access files. "
                    "After receiving tool results, analyze them and provide a helpful response to the user."
                ),
                attachments=attachments if current_iteration == 0 else None,
            )

            response = response.strip()
            if not response:
                break

            # Check for tool calls
            tool_calls = _extract_tool_calls(response)
            if not tool_calls:
                final_response = response
                break

            # Execute tool calls and collect results
            tool_results = []
            for tool_call in tool_calls:
                result = await _execute_tool(tool_call, services, projects_root, session.active_project_path)
                tool_results.append(result)

            # Append the assistant's response and tool results to conversation
            conversation_messages.append({"role": "assistant", "content": response})

            # Build tool results message
            tool_results_text = "Tool execution results:\n\n"
            for i, result in enumerate(tool_results):
                tool_results_text += f"Tool {i+1}: {result['tool']}\n"
                if result['success']:
                    tool_results_text += f"Success: {result['output'][:2000]}"
                    if len(result['output']) > 2000:
                        tool_results_text += "... (truncated)"
                    tool_results_text += "\n"
                else:
                    tool_results_text += f"Error: {result['error']}\n"
                tool_results_text += "\n"

            tool_results_text += "Now provide a helpful response to the user based on these results."
            conversation_messages.append({"role": "user", "content": tool_results_text})

            current_iteration += 1

        if not final_response:
            final_response = response

    except ModelAPIError as exc:
        LOGGER.warning("Workspace chatbot API failure for chat_id=%s: %s", chat_id, exc)
        await _safe_send_message(application, chat_id, f"API request failed: {exc}")
        return
    except ModelAuthError as exc:
        LOGGER.warning("Workspace chatbot auth failure for chat_id=%s: %s", chat_id, exc)
        await _safe_send_message(application, chat_id, f"Authentication failed: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Workspace chatbot request failed for chat_id=%s", chat_id)
        await _safe_send_message(application, chat_id, f"Workspace chat failed: {exc}")
        return

    final_response = final_response.strip()
    if not final_response:
        await _safe_send_message(application, chat_id, "API returned an empty response.")
        return

    _append_project_chat_history(session, user_text, final_response)
    await _safe_send_message(application, chat_id, _bounded_chat_reply(final_response))


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


async def _extract_image_attachments_from_message(message: Any) -> list[dict[str, str]]:
    attachments: list[dict[str, str]] = []

    if getattr(message, "photo", None):
        photo = message.photo[-1]
        try:
            telegram_file = await photo.get_file()
            payload = await telegram_file.download_as_bytearray()
            content = bytes(payload)
        except TelegramError:
            LOGGER.exception("Failed to download Telegram photo for chat_id=%s", message.chat_id)
            content = b""

        if content:
            attachments.append(
                {
                    "type": "blob",
                    "data": base64.b64encode(content).decode("ascii"),
                    "mimeType": "image/jpeg",
                    "displayName": f"{photo.file_unique_id}.jpg",
                }
            )

    document = getattr(message, "document", None)
    mime_type = str(getattr(document, "mime_type", "") or "").lower()
    if document and mime_type.startswith("image/"):
        try:
            telegram_file = await document.get_file()
            payload = await telegram_file.download_as_bytearray()
            content = bytes(payload)
        except TelegramError:
            LOGGER.exception("Failed to download Telegram image document for chat_id=%s", message.chat_id)
            content = b""

        if content:
            display_name = document.file_name or f"{document.file_unique_id}.img"
            attachments.append(
                {
                    "type": "blob",
                    "data": base64.b64encode(content).decode("ascii"),
                    "mimeType": mime_type,
                    "displayName": display_name,
                }
            )

    return attachments

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


def _build_tools_description() -> str:
    return """- shell_run: Execute shell commands in the projects directory
  Parameters: {"command": "string", "cwd": "optional path relative to projects root"}
  
- read_file: Read contents of a file
  Parameters: {"path": "relative path from active project or absolute from projects root"}
  
- list_directory: List files and directories
  Parameters: {"path": "optional directory path, defaults to active project"}
  
- file_info: Get file metadata (size, type, last modified)
  Parameters: {"path": "file path"}
  
- count_files: Count files in a directory
  Parameters: {"path": "directory path"}
  
- search_files: Search for files by pattern
  Parameters: {"pattern": "glob pattern", "path": "optional directory to search in"}
  
- get_project_structure: Get a tree-like view of project structure
  Parameters: {"path": "project path", "max_depth": "optional depth limit (default 3)"}"""


def _extract_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract tool calls from model response."""
    tool_calls = []

    # Match ```tool\n{...}\n``` blocks
    tool_pattern = re.compile(r'```tool\s*\n(.*?)\n```', re.DOTALL)
    for match in tool_pattern.finditer(text):
        try:
            json_str = match.group(1).strip()
            tool_call = json.loads(json_str)
            if "tool" in tool_call:
                tool_calls.append(tool_call)
        except json.JSONDecodeError:
            LOGGER.warning("Failed to parse tool call: %s", match.group(1)[:100])

    return tool_calls


async def _execute_tool(
    tool_call: dict[str, Any],
    services: RuntimeServices,
    projects_root: Path,
    active_project_path: str | None,
) -> dict[str, Any]:
    """Execute a tool call and return the result."""
    tool_name = tool_call.get("tool", "")
    params = tool_call.get("params", {})

    result: dict[str, Any] = {"tool": tool_name, "success": False, "output": "", "error": ""}

    try:
        if tool_name == "shell_run":
            command = params.get("command", "")
            cwd = params.get("cwd", "")
            if active_project_path:
                work_dir = Path(active_project_path)
            else:
                work_dir = projects_root
            if cwd:
                work_dir = projects_root / cwd

            shell_result = await services.shell_runner.run(command, work_dir)
            result["success"] = shell_result["success"]
            result["output"] = shell_result["output"]
            result["error"] = shell_result["error"]
            result["exit_code"] = shell_result.get("exit_code", 0)

        elif tool_name == "read_file":
            file_path = params.get("path", "")
            if active_project_path:
                full_path = Path(active_project_path) / file_path
            else:
                full_path = projects_root / file_path

            if not full_path.exists():
                result["error"] = f"File not found: {file_path}"
            elif not full_path.is_file():
                result["error"] = f"Path is not a file: {file_path}"
            else:
                try:
                    content = full_path.read_text(encoding="utf-8", errors="replace")
                    result["success"] = True
                    result["output"] = content
                except Exception as e:
                    result["error"] = f"Failed to read file: {e}"

        elif tool_name == "list_directory":
            dir_path = params.get("path", "")
            if dir_path:
                if active_project_path:
                    full_path = Path(active_project_path) / dir_path
                else:
                    full_path = projects_root / dir_path
            else:
                full_path = Path(active_project_path) if active_project_path else projects_root

            if not full_path.exists():
                result["error"] = f"Directory not found: {dir_path}"
            elif not full_path.is_dir():
                result["error"] = f"Path is not a directory: {dir_path}"
            else:
                try:
                    items = []
                    for item in sorted(full_path.iterdir()):
                        item_type = "dir" if item.is_dir() else "file"
                        items.append(f"{item.name} ({item_type})")
                    result["success"] = True
                    result["output"] = "\n".join(items) if items else "(empty directory)"
                except Exception as e:
                    result["error"] = f"Failed to list directory: {e}"

        elif tool_name == "file_info":
            file_path = params.get("path", "")
            if active_project_path:
                full_path = Path(active_project_path) / file_path
            else:
                full_path = projects_root / file_path

            if not full_path.exists():
                result["error"] = f"File not found: {file_path}"
            else:
                try:
                    stat = full_path.stat()
                    info_lines = [
                        f"Path: {file_path}",
                        f"Type: {'directory' if full_path.is_dir() else 'file'}",
                        f"Size: {stat.st_size} bytes",
                        f"Modified: {stat.st_mtime}",
                    ]
                    result["success"] = True
                    result["output"] = "\n".join(info_lines)
                except Exception as e:
                    result["error"] = f"Failed to get file info: {e}"

        elif tool_name == "count_files":
            dir_path = params.get("path", ".")
            if active_project_path:
                full_path = Path(active_project_path) / dir_path
            else:
                full_path = projects_root / dir_path

            if not full_path.exists():
                result["error"] = f"Directory not found: {dir_path}"
            elif not full_path.is_dir():
                result["error"] = f"Path is not a directory: {dir_path}"
            else:
                try:
                    count = _count_project_files(full_path, limit=10000)
                    result["success"] = True
                    result["output"] = f"Total files: {count}"
                except Exception as e:
                    result["error"] = f"Failed to count files: {e}"

        elif tool_name == "search_files":
            pattern = params.get("pattern", "")
            search_path = params.get("path", ".")
            if active_project_path:
                full_path = Path(active_project_path) / search_path
            else:
                full_path = projects_root / search_path

            if not full_path.exists():
                result["error"] = f"Directory not found: {search_path}"
            else:
                try:
                    matches = list(full_path.glob(pattern))
                    result["success"] = True
                    result["output"] = "\n".join(str(m.relative_to(full_path)) for m in matches[:100])
                    if len(matches) > 100:
                        result["output"] += f"\n... and {len(matches) - 100} more"
                except Exception as e:
                    result["error"] = f"Failed to search files: {e}"

        elif tool_name == "get_project_structure":
            project_path = params.get("path", ".")
            max_depth = params.get("max_depth", 3)
            if active_project_path:
                full_path = Path(active_project_path) / project_path
            else:
                full_path = projects_root / project_path

            if not full_path.exists():
                result["error"] = f"Directory not found: {project_path}"
            else:
                try:
                    structure = _render_directory_tree(full_path, max_depth=max_depth)
                    result["success"] = True
                    result["output"] = structure
                except Exception as e:
                    result["error"] = f"Failed to get project structure: {e}"

        else:
            result["error"] = f"Unknown tool: {tool_name}"

    except Exception as e:
        result["error"] = f"Tool execution error: {e}"
        LOGGER.exception("Tool execution failed for %s", tool_name)

    return result


def _render_directory_tree(path: Path, max_depth: int = 3, current_depth: int = 0, prefix: str = "") -> str:
    """Render a tree-like directory structure."""
    if current_depth >= max_depth:
        return ""

    lines = []
    try:
        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        excluded_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", ".pytest_cache"}

        for i, item in enumerate(items):
            if item.name.startswith(".") and item.name not in {".env", ".gitignore"}:
                continue
            if item.is_dir() and item.name in excluded_dirs:
                continue

            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{item.name}")

            if item.is_dir() and current_depth < max_depth - 1:
                extension = "    " if is_last else "│   "
                sub_tree = _render_directory_tree(item, max_depth, current_depth + 1, prefix + extension)
                if sub_tree:
                    lines.append(sub_tree)
    except PermissionError:
        lines.append(f"{prefix}(permission denied)")
    except Exception as e:
        lines.append(f"{prefix}(error: {e})")

    return "\n".join(lines)


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


def _model_keyboard(provider: str = "copilot") -> InlineKeyboardMarkup:
    if provider == "opencode":
        options = OPENCODE_MODEL_OPTIONS
    else:
        options = COPILOT_MODEL_OPTIONS
    keyboard = [
        [InlineKeyboardButton(label, callback_data=f"model:{value}")]
        for label, value in options
    ]
    return InlineKeyboardMarkup(keyboard)


def _resolve_model_choice(raw_choice: str, provider: str = "copilot") -> str | None:
    raw = raw_choice.strip()
    if not raw:
        return None

    normalized = raw.lower().strip()
    normalized = re.sub(r"\s+", "-", normalized)
    
    # Check in provider-specific models
    if provider == "opencode":
        if normalized in OPENCODE_MODEL_IDS:
            return normalized
        for label, value in OPENCODE_MODEL_OPTIONS:
            if normalized == re.sub(r"\s+", "-", label.lower().strip()):
                return value
    else:
        if normalized in COPILOT_MODEL_IDS:
            return normalized
        for label, value in COPILOT_MODEL_OPTIONS:
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


def _detect_virtual_env(project_path: Path) -> tuple[Path | None, str]:
    """Detect virtual environment in project and return (venv_path, python_path)."""
    venv_names = [".venv", "venv", "env", ".env", "virtualenv"]

    for venv_name in venv_names:
        venv_path = project_path / venv_name
        if venv_path.exists() and venv_path.is_dir():
            # Detect Python path based on OS
            if os.name == 'nt':  # Windows
                python_paths = [
                    venv_path / "Scripts" / "python.exe",
                    venv_path / "Scripts" / "python",
                ]
            else:  # Unix/Linux/Mac
                python_paths = [
                    venv_path / "bin" / "python",
                    venv_path / "bin" / "python3",
                ]

            for py_path in python_paths:
                if py_path.exists():
                    return (venv_path, str(py_path))

    return (None, "python")


def _build_python_commands(venv_python: str, pip_args: str) -> list[str]:
    """Build Python commands with virtual environment support."""
    commands = []

    # If venv exists, use its Python
    if venv_python and venv_python != "python":
        commands.append(f'"{venv_python}" {pip_args}')
        # Also try pip directly from venv
        venv_path = Path(venv_python).parent
        if os.name == 'nt':
            pip_path = venv_path / "pip.exe"
        else:
            pip_path = venv_path / "pip"
        if pip_path.exists():
            commands.append(f'"{pip_path}" {pip_args.replace("-m pip ", "")}')
    else:
        # No venv detected, use system Python but suggest creating one
        commands.append(f"python {pip_args}")
        commands.append(f"python3 {pip_args}")
        commands.append(f"py {pip_args}")

    return commands


def _get_pyproject_commands(project_path: Path, venv_python: str) -> list[str]:
    """Get install commands for pyproject.toml based on available tools."""
    commands = []

    # Detect which tool is being used
    pyproject_path = project_path / "pyproject.toml"
    if not pyproject_path.exists():
        return commands

    try:
        content = pyproject_path.read_text(encoding="utf-8")
    except Exception:
        content = ""

    # Check for uv (fastest, preferred)
    if '[tool.uv]' in content or 'requires = ["hatchling"]' in content:
        if venv_python and venv_python != "python":
            commands.append(f'"{venv_python}" -m pip install uv && uv sync')
            commands.append(f'"{venv_python}" -m pip install uv && uv pip install -e .')
        commands.append("pip install uv && uv sync")
        commands.append("pip install uv && uv pip install -e .")

    # Check for PDM
    if '[tool.pdm]' in content:
        if venv_python and venv_python != "python":
            commands.append(f'"{venv_python}" -m pip install pdm && pdm install')
        commands.append("pip install pdm && pdm install")

    # Check for Poetry
    if '[tool.poetry]' in content:
        commands.append("poetry install")
        if venv_python and venv_python != "python":
            commands.append(f'"{venv_python}" -m pip install poetry && poetry install')

    # Check for Hatch
    if '[tool.hatch]' in content:
        if venv_python and venv_python != "python":
            commands.append(f'"{venv_python}" -m pip install hatch && hatch env create')
        commands.append("pip install hatch && hatch env create")

    # Fallback to pip
    if venv_python and venv_python != "python":
        commands.append(f'"{venv_python}" -m pip install -e .')
    commands.append("pip install -e .")
    commands.append("python -m pip install -e .")

    return commands


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
    return "\n".join(
        [
            f"- GitHub push token available: {'yes' if github_ready else 'no'}",
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
