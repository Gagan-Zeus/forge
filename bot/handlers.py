from __future__ import annotations

import asyncio
import copy
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent.fixer import BuildFixer
from agent.orchestrator import BuildOrchestrator
from agent.planner import ProjectPlanner
from bot.session import SessionStore, UserSession, build_summary
from models.copilot_client import CopilotAuthError, CopilotClient
from tools.file_writer import FileWriter
from tools.github_pusher import GitHubPusher
from tools.shell_runner import ShellRunner

LOGGER = logging.getLogger(__name__)

MODEL_OPTIONS: list[tuple[str, str]] = [
    ("GPT-5.2", "gpt-5.2"),
    ("GPT-5.2 Codex", "gpt-5.2-codex"),
    ("GPT-5.3 Codex", "gpt-5.3-codex"),
    ("GPT-5.4 Mini", "gpt-5.4-mini"),
    ("GPT-5.1", "gpt-5.1"),
    ("GPT-5.1 Codex", "gpt-5.1-codex"),
    ("GPT-5.1 Codex Mini", "gpt-5.1-codex-mini"),
    ("GPT-5.1 Codex Max", "gpt-5.1-codex-max"),
    ("GPT-5 Mini", "gpt-5-mini"),
    ("GPT-4o Mini", "gpt-4o-mini"),
    ("GPT-4o", "gpt-4o"),
    ("Grok Code Fast 1", "grok-code-fast-1"),
    ("Claude Opus 4.5", "claude-opus-4.5"),
    ("Claude Sonnet 4.5", "claude-sonnet-4.5"),
    ("Claude Sonnet 4", "claude-sonnet-4"),
    ("Claude Haiku 4.5", "claude-haiku-4.5"),
    ("Gemini 3.1 Pro Preview", "gemini-3.1-pro-preview"),
    ("Gemini 3 Flash Preview", "gemini-3-flash-preview"),
    ("Gemini 2.5 Pro", "gemini-2.5-pro"),
    ("Gemini 3 Pro", "gemini-3-pro"),
    ("Gemini 3 Flash", "gemini-3-flash"),
    ("GPT-4.1", "gpt-4.1"),
]
MODEL_LABELS = {value: label for label, value in MODEL_OPTIONS}


@dataclass
class RuntimeServices:
    sessions: SessionStore
    copilot_client: CopilotClient
    orchestrator: BuildOrchestrator


def run_bot(telegram_token: str, github_username: str, github_token: str, projects_dir: str) -> None:
    if not telegram_token:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN in environment.")

    projects_path = Path(projects_dir).resolve()
    file_writer = FileWriter(projects_path)
    shell_runner = ShellRunner(allowed_root=projects_path, timeout_seconds=120)
    copilot_client = CopilotClient(tokens_path=Path("auth") / "tokens.json")
    planner = ProjectPlanner(copilot_client)
    fixer = BuildFixer(copilot_client)
    github_pat = github_token.strip()

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
        fixer=fixer,
        file_writer=file_writer,
        shell_runner=shell_runner,
        github_pusher=github_pusher,
    )

    app = Application.builder().token(telegram_token).build()
    app.bot_data["services"] = RuntimeServices(
        sessions=SessionStore(),
        copilot_client=copilot_client,
        orchestrator=orchestrator,
    )
    app.bot_data["build_tasks"] = {}
    app.bot_data["auth_tasks"] = {}

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("cancel", cancel_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CallbackQueryHandler(model_selection_callback, pattern=r"^model:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))
    app.add_error_handler(global_error_handler)

    LOGGER.info("Telegram bot is starting polling loop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = _chat_id(update)
    if chat_id is None:
        return

    services = _services(context)
    session = services.sessions.reset(chat_id, keep_auth=False)
    auth_tasks = _task_store(context, "auth_tasks")

    if await _has_valid_auth(services.copilot_client):
        session.is_authenticated = True
        await _safe_send_message(
            context.application,
            chat_id,
            "Copilot already connected. Send me your project idea to get started.",
        )
        await _ask_step_1(context.application, chat_id, session)
        return

    session.auth_in_progress = True
    await _safe_send_message(context.application, chat_id, "Starting GitHub Copilot connection...")

    previous_task = auth_tasks.get(chat_id)
    if previous_task and not previous_task.done():
        previous_task.cancel()

    task = asyncio.create_task(_run_auth_flow(context.application, services, chat_id))
    auth_tasks[chat_id] = task


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = _chat_id(update)
    if chat_id is None:
        return

    services = _services(context)
    services.sessions.reset(chat_id, keep_auth=False)
    await _safe_send_message(context.application, chat_id, "Session reset.")
    await start_command(update, context)


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

    if session.auth_in_progress:
        await _safe_send_message(context.application, chat_id, "Authentication is in progress.")
        return
    if session.is_building:
        await _safe_send_message(
            context.application,
            chat_id,
            f"Build in progress:\n{session.build_progress}",
        )
        return

    await _safe_send_message(
        context.application,
        chat_id,
        f"Current step: {session.current_step}\nAuthenticated: {'yes' if session.is_authenticated else 'no'}",
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

    if session.current_step != 3:
        await _safe_send_message(context.application, chat_id, "Model selection is not expected right now.")
        return

    model = (query.data or "").split(":", maxsplit=1)[-1]
    if model not in CopilotClient.available_models():
        await _safe_send_message(context.application, chat_id, "Unsupported model choice.")
        return

    session.model = model
    session.current_step = 4
    await _safe_send_message(
        context.application,
        chat_id,
        (
            f"Selected model: {MODEL_LABELS.get(model, model)}\n"
            "Step 4 - Any special requirements? (libraries, constraints, architecture)\n"
            "Reply 'none' to skip."
        ),
    )


async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    chat_id = update.message.chat_id
    text = (update.message.text or "").strip()
    services = _services(context)
    session = services.sessions.get(chat_id)

    if not session.is_authenticated:
        wait_text = "Authentication is in progress. Please complete it in GitHub." if session.auth_in_progress else "Use /start to connect GitHub Copilot first."
        await _safe_send_message(context.application, chat_id, wait_text)
        return

    if session.is_building:
        await _safe_send_message(
            context.application,
            chat_id,
            "A build is already running. Use /status to check progress or /cancel to stop it.",
        )
        return

    if session.current_step == 1:
        session.idea = text
        session.current_step = 2
        await _safe_send_message(
            context.application,
            chat_id,
            "Step 2 - Which language/stack? (e.g. Python, Node.js, React, FastAPI)",
        )
        return

    if session.current_step == 2:
        session.stack = text
        session.current_step = 3
        await _safe_send_message(
            context.application,
            chat_id,
            "Step 3 - Which AI model?",
            reply_markup=_model_keyboard(),
        )
        return

    if session.current_step == 3:
        await _safe_send_message(context.application, chat_id, "Please select the model from the inline buttons.")
        return

    if session.current_step == 4:
        session.requirements = "" if text.lower() == "none" else text
        session.current_step = 5
        await _safe_send_message(
            context.application,
            chat_id,
            "Step 5 - Push to GitHub when done? (yes / no)",
        )
        return

    if session.current_step == 5:
        await _handle_step_5(context.application, chat_id, session, text)
        return

    if session.current_step == 6:
        if text.lower() == "yes":
            await _start_build(context.application, services, chat_id, session)
            return
        if text.lower() == "no":
            await _safe_send_message(context.application, chat_id, "Build canceled. Send /reset to restart intake.")
            return
        await _safe_send_message(context.application, chat_id, "Please answer yes or no.")
        return

    await _safe_send_message(context.application, chat_id, "Use /start to begin.")


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


async def _run_auth_flow(application: Application, services: RuntimeServices, chat_id: int) -> None:
    session = services.sessions.get(chat_id)

    async def on_device_code(device_flow: Any) -> None:
        await _safe_send_message(
            application,
            chat_id,
            (
                "To connect GitHub Copilot, go to:\n"
                "https://github.com/login/device\n"
                f"Enter code: {device_flow.user_code}\n"
                "Waiting for you to authorize..."
            ),
        )

    try:
        await services.copilot_client.authenticate(on_device_code=on_device_code)
        session.is_authenticated = True
        session.auth_in_progress = False
        await _safe_send_message(
            application,
            chat_id,
            "Copilot connected! All models are now available.\nSend me your project idea to get started.",
        )
        await _ask_step_1(application, chat_id, session)
    except CopilotAuthError as exc:
        session.is_authenticated = False
        session.auth_in_progress = False
        LOGGER.warning("Copilot authentication rejected for chat_id=%s: %s", chat_id, exc)
        await _safe_send_message(
            application,
            chat_id,
            (
                "Copilot authentication failed. Confirm your GitHub account has Copilot access, "
                "then run /start again.\n"
                f"Details: {exc}"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        session.is_authenticated = False
        session.auth_in_progress = False
        LOGGER.exception("Copilot authentication failed for chat_id=%s", chat_id)
        await _safe_send_message(application, chat_id, f"Copilot authentication failed: {exc}")


async def _handle_step_5(application: Application, chat_id: int, session: UserSession, text: str) -> None:
    lowered = text.lower()

    if session.awaiting_repo_name:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", text):
            await _safe_send_message(
                application,
                chat_id,
                "Repo name can contain only letters, numbers, dots, dashes, and underscores. Try again.",
            )
            return
        session.repo_name = text
        session.awaiting_repo_name = False
        session.awaiting_repo_visibility = True
        await _safe_send_message(application, chat_id, "Public or private?")
        return

    if session.awaiting_repo_visibility:
        if lowered not in {"public", "private"}:
            await _safe_send_message(application, chat_id, "Please answer 'public' or 'private'.")
            return
        session.repo_visibility = lowered
        session.awaiting_repo_visibility = False
        session.current_step = 6
        await _send_confirmation(application, chat_id, session)
        return

    if lowered == "yes":
        session.push_to_github = True
        session.awaiting_repo_name = True
        await _safe_send_message(application, chat_id, "Repo name?")
        return

    if lowered == "no":
        session.push_to_github = False
        session.repo_name = ""
        session.repo_visibility = "private"
        session.current_step = 6
        await _send_confirmation(application, chat_id, session)
        return

    await _safe_send_message(application, chat_id, "Please answer yes or no.")


async def _start_build(
    application: Application,
    services: RuntimeServices,
    chat_id: int,
    session: UserSession,
) -> None:
    if session.is_building:
        await _safe_send_message(application, chat_id, "Build is already running.")
        return

    snapshot = copy.deepcopy(session)
    session.is_building = True
    session.build_progress = "Initializing build..."

    async def progress(message: str) -> None:
        live_session = services.sessions.get(chat_id)
        live_session.build_progress = message
        await _safe_send_message(application, chat_id, message)

    async def build_task() -> None:
        try:
            result = await services.orchestrator.build_project(
                chat_id=chat_id,
                session=snapshot,
                progress_callback=progress,
            )
            live_session = services.sessions.get(chat_id)
            live_session.is_building = False
            live_session.build_progress = "Idle"

            if result.success:
                warnings = "\n".join(f"- {item}" for item in result.warnings) if result.warnings else "- none"
                files = "\n".join(f"- {item}" for item in result.files_created) if result.files_created else "- none"
                summary_lines = [
                    "Build complete!",
                    f"Project name and description: {result.project_name} | {snapshot.idea}",
                    f"Stack: {snapshot.stack}",
                    f"Model used: {snapshot.model}",
                    "Files created:",
                    files,
                    f"Local path: {result.project_path}",
                    f"GitHub URL: {result.github_url or 'not pushed'}",
                    "Warnings:",
                    warnings,
                ]
                await _safe_send_message(application, chat_id, "\n".join(summary_lines))
                live_session.current_step = 1
                await _ask_step_1(application, chat_id, live_session)
            else:
                issue = (result.error or "Unknown error").strip()
                bounded_issue = issue if len(issue) <= 3500 else issue[:3500] + "..."
                await _safe_send_message(
                    application,
                    chat_id,
                    f"Build failed during validation/build.\nIssue:\n{bounded_issue}",
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Build task crashed for chat_id=%s", chat_id)
            live_session = services.sessions.get(chat_id)
            live_session.is_building = False
            live_session.build_progress = "Idle"
            await _safe_send_message(application, chat_id, f"Build crashed: {exc}")

    build_tasks = cast(dict[int, asyncio.Task[Any]], application.bot_data["build_tasks"])
    task = asyncio.create_task(build_task())
    build_tasks[chat_id] = task
    await _safe_send_message(application, chat_id, "Build started.")


async def _send_confirmation(application: Application, chat_id: int, session: UserSession) -> None:
    summary = build_summary(session)
    await _safe_send_message(
        application,
        chat_id,
        f"Step 6 - Summary\n{summary}\nReady to build? (yes / no)",
    )


async def _ask_step_1(application: Application, chat_id: int, session: UserSession) -> None:
    session.current_step = 1
    session.awaiting_repo_name = False
    session.awaiting_repo_visibility = False
    await _safe_send_message(
        application,
        chat_id,
        "Step 1 - What project do you want to build? Describe it in detail.",
    )


def _services(context: ContextTypes.DEFAULT_TYPE) -> RuntimeServices:
    return cast(RuntimeServices, context.application.bot_data["services"])


def _task_store(context: ContextTypes.DEFAULT_TYPE, key: str) -> dict[int, asyncio.Task[Any]]:
    return cast(dict[int, asyncio.Task[Any]], context.application.bot_data[key])


def _chat_id(update: Update) -> int | None:
    if update.effective_chat:
        return update.effective_chat.id
    return None


def _model_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton(label, callback_data=f"model:{value}")]
        for label, value in MODEL_OPTIONS
    ]
    return InlineKeyboardMarkup(keyboard)


async def _has_valid_auth(copilot_client: CopilotClient) -> bool:
    if not copilot_client.is_authenticated():
        return False
    try:
        await copilot_client.get_token()
        return True
    except (CopilotAuthError, TelegramError, OSError, ValueError):
        return False
    except Exception:  # noqa: BLE001
        LOGGER.exception("Unexpected token validation failure")
        return False


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
