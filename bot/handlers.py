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
from agent.orchestrator import BuildOrchestrator, BuildResult
from agent.planner import ProjectPlanner
from bot.session import SessionStore, UserSession, build_summary
from models.copilot_client import CopilotAuthError, CopilotClient
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
PROJECT_ACTION_MAX_FILES = 12
PROJECT_FILE_TREE_MAX_ENTRIES = 240
PROJECT_FILE_TREE_MAX_CHARS = 12000


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
    shell_runner = ShellRunner(allowed_root=projects_path, timeout_seconds=120)
    github_pat = github_token.strip()
    copilot_client = CopilotClient(
        timeout_seconds=120.0,
        cli_path=os.getenv("COPILOT_CLI_PATH", ""),
        github_token=github_pat,
    )
    planner = ProjectPlanner(copilot_client)
    fixer = BuildFixer(copilot_client)

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
        file_writer=file_writer,
        shell_runner=shell_runner,
        github_pusher=github_pusher,
    )
    app.bot_data["build_tasks"] = {}
    app.bot_data["projects_root"] = projects_path
    app.bot_data["env_file"] = (Path.cwd() / ".env").resolve()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("project", project_command))
    app.add_handler(CommandHandler("model", model_command))
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

    try:
        await services.copilot_client.ensure_ready()
        session.is_authenticated = True
        session.model = DEFAULT_MODEL
        _clear_project_intake_state(session)
        await _safe_send_message(
            context.application,
            chat_id,
            (
                "Copilot SDK connected. Chatbot mode is active.\n"
                f"Default model: {MODEL_LABELS.get(session.model, session.model)}\n\n"
                "Commands:\n"
                "/project - start project generation workflow\n"
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
    _clear_project_intake_state(session)
    await _safe_send_message(
        context.application,
        chat_id,
        (
            "Session reset. Chatbot mode is active.\n"
            f"Current model: {MODEL_LABELS.get(session.model, session.model)}"
        ),
    )


async def project_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
            "A build is already running. Use /status or /cancel.",
        )
        return

    if session.model not in ALLOWED_MODEL_IDS:
        session.model = DEFAULT_MODEL

    _begin_project_intake(session)
    await _safe_send_message(
        context.application,
        chat_id,
        (
            f"Project workflow started. Model locked to: {MODEL_LABELS.get(session.model, session.model)}\n"
            "Step 1 - What project do you want to build? Describe it in detail."
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

    if session.current_step > 0:
        await _safe_send_message(
            context.application,
            chat_id,
            (
                f"Project workflow in progress (step {session.current_step}).\n"
                "Continue by answering the current prompt, or send /reset to exit workflow."
            ),
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
    if session.current_step > 0:
        await _safe_send_message(
            context.application,
            chat_id,
            (
                f"Selected model: {MODEL_LABELS.get(model, model)}\n"
                "Project workflow is still active. Continue from your current step."
            ),
        )
        return

    await _safe_send_message(context.application, chat_id, f"Selected model: {MODEL_LABELS.get(model, model)}")


async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    chat_id = update.message.chat_id
    text = (update.message.text or "").strip()
    services = _services(context)
    session = services.sessions.get(chat_id)

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
            "Step 3 - Any special requirements? (libraries, constraints, architecture)\nReply 'none' to skip.",
        )
        return

    if session.current_step == 3:
        session.requirements = "" if text.lower() == "none" else text
        session.current_step = 4
        await _safe_send_message(
            context.application,
            chat_id,
            "Step 4 - Push to GitHub when done? (yes / no)",
        )
        return

    if session.current_step == 4:
        await _handle_step_4(context.application, chat_id, session, text)
        return

    if session.current_step == 5:
        if text.lower() == "yes":
            await _start_build(context.application, services, chat_id, session)
            return
        if text.lower() == "no":
            _clear_project_intake_state(session)
            await _safe_send_message(
                context.application,
                chat_id,
                "Build canceled. Chatbot mode is active. Send /project to start a new workflow.",
            )
            return
        await _safe_send_message(context.application, chat_id, "Please answer yes or no.")
        return

    if session.active_project_path:
        await _handle_project_chat(context.application, services, chat_id, session, text)
        return

    await _handle_workspace_chat(context.application, services, chat_id, session, text)


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


async def _handle_step_4(application: Application, chat_id: int, session: UserSession, text: str) -> None:
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
        session.current_step = 5
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
        session.current_step = 5
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

    if session.model not in ALLOWED_MODEL_IDS:
        session.model = DEFAULT_MODEL

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
                _clear_project_intake_state(live_session)
                live_session.project_chat_mode = True
                live_session.project_context = _build_project_chat_context(snapshot, result)
                live_session.active_project_path = result.project_path
                live_session.active_github_url = result.github_url or ""
                live_session.chat_history.clear()
                await _safe_send_message(
                    application,
                    chat_id,
                    (
                        "You are now in project follow-up chat mode. "
                        "Ask anything about this generated project (changes, fixes, explanations, next features). "
                        "Send /project anytime to start a brand new project workflow."
                    ),
                )
            else:
                issue = (result.error or "Unknown error").strip()
                bounded_issue = issue if len(issue) <= 3500 else issue[:3500] + "..."
                await _safe_send_message(
                    application,
                    chat_id,
                    f"Build failed during validation/build.\nIssue:\n{bounded_issue}",
                )
                _clear_project_intake_state(live_session)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Build task crashed for chat_id=%s", chat_id)
            live_session = services.sessions.get(chat_id)
            live_session.is_building = False
            live_session.build_progress = "Idle"
            _clear_project_intake_state(live_session)
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
        f"Step 5 - Summary\n{summary}\nReady to build? (yes / no)",
    )


def _clear_project_intake_state(session: UserSession) -> None:
    session.current_step = 0
    session.idea = ""
    session.stack = ""
    session.requirements = ""
    session.push_to_github = False
    session.repo_name = ""
    session.repo_visibility = "private"
    session.awaiting_repo_name = False
    session.awaiting_repo_visibility = False


def _begin_project_intake(session: UserSession) -> None:
    session.current_step = 1
    session.idea = ""
    session.stack = ""
    session.requirements = ""
    session.push_to_github = False
    session.repo_name = ""
    session.repo_visibility = "private"
    session.awaiting_repo_name = False
    session.awaiting_repo_visibility = False
    session.project_chat_mode = False
    session.chat_history.clear()


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
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Workspace chatbot request failed for chat_id=%s", chat_id)
        await _safe_send_message(application, chat_id, f"Workspace chat failed: {exc}")
        return

    _append_project_chat_history(session, user_text, response)
    await _safe_send_message(application, chat_id, _bounded_chat_reply(response))


async def _handle_project_chat(
    application: Application,
    services: RuntimeServices,
    chat_id: int,
    session: UserSession,
    user_text: str,
) -> None:
    if not user_text:
        await _safe_send_message(application, chat_id, "Ask a follow-up question about your generated project.")
        return

    if not session.project_context or not session.active_project_path:
        session.project_chat_mode = False
        await _handle_workspace_chat(application, services, chat_id, session, user_text)
        return

    project_root = Path(session.active_project_path).resolve()
    if not project_root.exists() or not project_root.is_dir():
        session.project_chat_mode = False
        session.active_project_path = ""
        session.project_context = ""
        await _handle_workspace_chat(application, services, chat_id, session, user_text)
        return

    if not session.project_context:
        session.project_context = _build_live_project_chat_context(
            session=session,
            project_root=project_root,
            changed_files=[],
            github_url=session.active_github_url,
            warnings=[],
        )

    selected_model = session.model.strip()
    if selected_model not in ALLOWED_MODEL_IDS:
        selected_model = DEFAULT_MODEL

    projects_root = _projects_root(application)
    workspace_projects_text = _render_workspace_projects(projects_root, limit=40)
    env_status = _load_env_key_status(_env_file_path(application))
    env_keys_text = _render_env_key_status_lines(env_status)
    integration_status_text = _render_integration_status(env_status)

    history_window = session.chat_history[-(PROJECT_CHAT_HISTORY_TURNS * 2) :]
    file_list = _collect_project_files(project_root, limit=PROJECT_FILE_TREE_MAX_ENTRIES)
    file_tree_text = _render_project_file_tree(file_list)
    history_text = _render_project_history(history_window)

    planner_prompt = (
        "You are controlling an autonomous coding assistant for an existing local project. "
        "Decide if the latest user message requires direct project actions (edit files, run validation, push) "
        "or just a conversational answer.\n\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        "  \"mode\": \"action\" | \"chat\",\n"
        "  \"assistant_reply\": \"short response for user\",\n"
        "  \"summary\": \"short internal plan summary\",\n"
        "  \"files\": [{\"path\": \"relative/path.ext\", \"change\": \"what to change\"}],\n"
        "  \"install_command\": \"optional install command\",\n"
        "  \"validation_command\": \"optional validation command\",\n"
        "  \"push_to_github\": true | false\n"
        "}\n\n"
        "Rules:\n"
        "- Use mode=action only when code/project changes are requested.\n"
        "- For mode=chat, return files as an empty list.\n"
        "- Never use absolute paths.\n"
        "- Keep files list minimal and focused.\n"
        "- Do not include README or dependency/lock files unless user explicitly asked to modify them.\n"
        "- For HTML changes, keep CSS and JS in their own files (no inline <style> or inline <script>).\n"
        "- Use safe, non-interactive commands only.\n\n"
        f"Generated projects root: {projects_root}\n"
        f"Workspace project inventory:\n{workspace_projects_text}\n\n"
        f".env key status (values hidden):\n{env_keys_text}\n\n"
        f"Integration status:\n{integration_status_text}\n\n"
        f"Project context:\n{session.project_context}\n\n"
        f"Project files:\n{file_tree_text}\n\n"
        f"Recent project chat history:\n{history_text}\n\n"
        f"Latest user message:\n{user_text}\n"
    )

    try:
        plan_raw = await services.copilot_client.call(
            messages=[{"role": "user", "content": planner_prompt}],
            model=selected_model,
            system_prompt="You are an autonomous coding planner. Return valid JSON only.",
        )
        plan = _parse_followup_plan(plan_raw)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Project follow-up chat failed for chat_id=%s", chat_id)
        await _safe_send_message(application, chat_id, f"Project follow-up chat failed: {exc}")
        return

    if plan["mode"] != "action":
        assistant_text = plan.get("assistant_reply") or plan.get("summary") or "Got it."
        _append_project_chat_history(session, user_text, assistant_text)
        await _safe_send_message(application, chat_id, _bounded_chat_reply(assistant_text))
        return

    file_specs_raw = plan.get("files") or []
    safe_file_specs: list[dict[str, str]] = []
    for item in file_specs_raw[:PROJECT_ACTION_MAX_FILES]:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "")).strip()
        change = str(item.get("change", "")).strip() or user_text
        if not path or not _is_safe_project_relative_path(path):
            continue
        safe_file_specs.append({"path": path, "change": change})

    push_requested = bool(plan.get("push_to_github")) or _looks_like_push_request(user_text)
    requested_install = str(plan.get("install_command") or "").strip()
    requested_validation = str(plan.get("validation_command") or "").strip()
    if not safe_file_specs and not push_requested and not requested_install and not requested_validation:
        assistant_text = (
            plan.get("assistant_reply")
            or "I could not determine safe file changes from that request. Please specify files or desired behavior."
        )
        _append_project_chat_history(session, user_text, assistant_text)
        await _safe_send_message(application, chat_id, _bounded_chat_reply(assistant_text))
        return

    changed_files: list[str] = []
    edit_warnings: list[str] = []
    if safe_file_specs:
        await _safe_send_message(application, chat_id, "Applying requested project changes...")
        for index, file_spec in enumerate(safe_file_specs, start=1):
            relative_path = file_spec["path"]
            change_request = file_spec["change"]
            current_content = _read_project_file_if_exists(project_root, relative_path)
            allow_destructive_update = _looks_like_deletion_intent(user_text, change_request)
            generated_assets: dict[str, str] = {}
            try:
                updated_content = await _generate_followup_file_content(
                    services=services,
                    selected_model=selected_model,
                    session=session,
                    user_text=user_text,
                    relative_path=relative_path,
                    change_request=change_request,
                    current_content=current_content,
                    project_files=file_list,
                    preserve_existing=not allow_destructive_update,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to generate follow-up file content for %s", relative_path)
                await _safe_send_message(
                    application,
                    chat_id,
                    f"Could not generate an update for {relative_path}: {exc}",
                )
                return

            if current_content and not allow_destructive_update and relative_path.lower().endswith(".html"):
                recovered_content, recovered_assets = _recover_missing_html_assets(
                    relative_path=relative_path,
                    current_content=current_content,
                    updated_content=updated_content,
                )
                updated_content = recovered_content
                _merge_generated_asset_content(generated_assets, recovered_assets)

            updated_content, boundary_assets = _apply_file_language_boundary_rules(relative_path, updated_content)
            _merge_generated_asset_content(generated_assets, boundary_assets)

            if current_content and not allow_destructive_update:
                if _is_overly_destructive_update(relative_path, current_content, updated_content):
                    warning = (
                        f"Skipped destructive rewrite for {relative_path}. "
                        "I preserved the existing file because your request did not explicitly ask to remove code."
                    )
                    edit_warnings.append(warning)
                    await _safe_send_message(application, chat_id, warning)
                    continue

            if updated_content == current_content and not generated_assets:
                await _safe_send_message(
                    application,
                    chat_id,
                    f"No effective code change generated for {relative_path}; keeping existing file.",
                )
                continue

            updated_targets: list[str] = []
            if updated_content != current_content:
                await services.file_writer.write_file(project_root, relative_path, updated_content)
                changed_files.append(relative_path)
                updated_targets.append(relative_path)

            for asset_path, asset_content in generated_assets.items():
                existing_asset_content = _read_project_file_if_exists(project_root, asset_path)
                merged_asset_content = _merge_asset_content(existing_asset_content, asset_content)
                if merged_asset_content.strip() == existing_asset_content.strip():
                    continue
                await services.file_writer.write_file(project_root, asset_path, merged_asset_content)
                if asset_path not in changed_files:
                    changed_files.append(asset_path)
                updated_targets.append(asset_path)

            if not updated_targets:
                await _safe_send_message(
                    application,
                    chat_id,
                    f"No effective code change generated for {relative_path}; keeping existing file.",
                )
                continue

            await _safe_send_message(
                application,
                chat_id,
                f"Updated {', '.join(updated_targets)}... ({index}/{len(safe_file_specs)})",
            )
    else:
        await _safe_send_message(application, chat_id, "No direct file edits requested; applying command/push actions.")

    all_project_files = _collect_project_files(project_root, limit=PROJECT_FILE_TREE_MAX_ENTRIES)
    readme_content = BuildOrchestrator._load_readme_content(project_root)

    install_command = _pick_followup_install_command(
        plan_install_command=requested_install,
        session=session,
        project_files=all_project_files,
        readme_content=readme_content,
        allow_fallback=bool(changed_files),
    )
    warnings: list[str] = list(edit_warnings)
    if install_command:
        await _safe_send_message(application, chat_id, f"Installing dependencies: {install_command}")
        install_result = await services.shell_runner.run(install_command, project_root)
        if not install_result["success"]:
            install_warning = BuildOrchestrator._format_command_warning("dependency install", install_result)
            warnings.append(install_warning)
            await _safe_send_message(application, chat_id, f"Install warning: {install_warning}")

    validation_command = _pick_followup_validation_command(
        plan_validation_command=requested_validation,
        session=session,
        project_files=all_project_files,
        readme_content=readme_content,
        allow_fallback=bool(changed_files),
    )

    if validation_command:
        await _safe_send_message(application, chat_id, f"Running validation: {validation_command}")
        validation_result = await services.shell_runner.run(validation_command, project_root)
        if not validation_result["success"]:
            initial_error = BuildOrchestrator._combine_output(validation_result)

            async def followup_fix_progress(message: str) -> None:
                await _safe_send_message(application, chat_id, message)

            fix_candidates = _filter_followup_fix_candidates(changed_files or all_project_files)
            fix_error = await services.orchestrator.attempt_single_fix(
                session=session,
                project_dir=project_root,
                candidate_files=fix_candidates,
                validation_command=validation_command,
                initial_error=initial_error,
                progress_callback=followup_fix_progress,
            )
            if fix_error:
                final_error = f"Follow-up action failed during validation.\nIssue:\n{fix_error}"
                _append_project_chat_history(session, user_text, final_error)
                await _safe_send_message(application, chat_id, _bounded_chat_reply(final_error))
                return
    else:
        if changed_files:
            warnings.append("Validation command not inferred for follow-up changes.")

    github_url = session.active_github_url
    if push_requested:
        if services.github_pusher is None:
            warnings.append("GitHub push requested but pusher is not configured.")
        else:
            repo_name = (
                session.repo_name.strip()
                or _repo_name_from_github_url(session.active_github_url)
                or _derive_repo_name(project_root, session.idea)
            )
            visibility = session.repo_visibility if session.repo_visibility in {"public", "private"} else "private"
            await _safe_send_message(application, chat_id, "Pushing updates to GitHub...")
            try:
                github_url = await services.github_pusher.push_project(
                    project_path=project_root,
                    repo_name=repo_name,
                    visibility=visibility,
                )
                session.repo_name = repo_name
                session.repo_visibility = visibility
            except Exception as exc:  # noqa: BLE001
                warning = f"GitHub push failed: {exc}"
                warnings.append(warning)
                await _safe_send_message(application, chat_id, _bounded_chat_reply(warning))

    session.active_project_path = str(project_root)
    session.active_github_url = github_url
    session.project_context = _build_live_project_chat_context(
        session=session,
        project_root=project_root,
        changed_files=changed_files,
        github_url=github_url,
        warnings=warnings,
    )

    summary_lines = [
        "Applied follow-up changes to the current project context.",
        f"Project path: {project_root}",
        "Changed files:",
        "\n".join(f"- {path}" for path in changed_files) if changed_files else "- none",
        f"GitHub URL: {github_url or 'not pushed'}",
    ]
    if warnings:
        summary_lines.extend(["Warnings:", "\n".join(f"- {item}" for item in warnings)])

    assistant_text = "\n".join(summary_lines)
    _append_project_chat_history(session, user_text, assistant_text)
    await _safe_send_message(application, chat_id, _bounded_chat_reply(assistant_text))


async def _generate_followup_file_content(
    services: RuntimeServices,
    selected_model: str,
    session: UserSession,
    user_text: str,
    relative_path: str,
    change_request: str,
    current_content: str,
    project_files: list[str],
    preserve_existing: bool,
) -> str:
    project_files_text = "\n".join(f"- {path}" for path in project_files[:PROJECT_FILE_TREE_MAX_ENTRIES])
    language_boundary_guidance = (
        "Language boundary rules:\n"
        "- Each file must contain only its own language.\n"
        "- .html files must not contain inline CSS (<style>) or inline JS (<script>...</script>).\n"
        "- Put CSS in .css files and JS/TS in .js/.ts files and reference them from HTML.\n"
        "- .css files contain only CSS; .js/.ts files contain only script code.\n\n"
    )
    preservation_guidance = ""
    if preserve_existing:
        preservation_guidance = (
            "Preservation rules:\n"
            "- Start from the current file and make the smallest possible edits.\n"
            "- Do not remove existing sections, styles, scripts, imports, or structure unless explicitly requested.\n"
            "- Keep all existing CSS and layout behavior unless the request clearly asks to replace them.\n"
            "- Avoid rewriting the entire file when a focused edit is enough.\n\n"
        )

    prompt = (
        "Update the target file for an existing project based on the user's follow-up request.\n"
        "Return only the full file content (no markdown fences).\n"
        "Preserve compatibility with existing project structure and imports.\n\n"
        f"{language_boundary_guidance}"
        f"{preservation_guidance}"
        f"Project context:\n{session.project_context}\n\n"
        f"User follow-up request:\n{user_text}\n\n"
        f"Requested change summary for this file:\n{change_request}\n\n"
        f"Project files:\n{project_files_text}\n\n"
        f"Target file: {relative_path}\n"
        "Current file content:\n"
        f"{current_content if current_content else '# File does not exist yet.'}\n"
    )
    response = await services.copilot_client.call(
        messages=[{"role": "user", "content": prompt}],
        model=selected_model,
        system_prompt="You are a senior software engineer. Return code only.",
    )
    return _strip_code_fences(response)


def _pick_followup_install_command(
    plan_install_command: str,
    session: UserSession,
    project_files: list[str],
    readme_content: str | None,
    allow_fallback: bool,
) -> str | None:
    if plan_install_command:
        command = plan_install_command.strip()
        if BuildOrchestrator._is_safe_for_runner(command) and BuildOrchestrator._is_install_command(command):
            return command

    if not allow_fallback:
        return None

    return BuildOrchestrator._pick_install_command(session.stack, project_files, readme_content)


def _pick_followup_validation_command(
    plan_validation_command: str,
    session: UserSession,
    project_files: list[str],
    readme_content: str | None,
    allow_fallback: bool,
) -> str | None:
    if plan_validation_command:
        command = plan_validation_command.strip()
        if (
            BuildOrchestrator._is_safe_for_runner(command)
            and BuildOrchestrator._is_validation_command(command)
            and not BuildOrchestrator._is_interactive_command(command)
        ):
            return command

    if not allow_fallback:
        return None

    fallback_command, _ = BuildOrchestrator._pick_validation_command(session.stack, project_files, readme_content)
    if (
        fallback_command
        and BuildOrchestrator._is_safe_for_runner(fallback_command)
        and BuildOrchestrator._is_validation_command(fallback_command)
        and not BuildOrchestrator._is_interactive_command(fallback_command)
    ):
        return fallback_command

    if any(path.endswith(".py") for path in project_files):
        return "python -m compileall -q ."

    return None


def _build_live_project_chat_context(
    session: UserSession,
    project_root: Path,
    changed_files: list[str],
    github_url: str,
    warnings: list[str],
) -> str:
    files = _collect_project_files(project_root, limit=120)
    file_lines = "\n".join(f"- {path}" for path in files) if files else "- none"
    changed_lines = "\n".join(f"- {path}" for path in changed_files) if changed_files else "- none"
    warning_lines = "\n".join(f"- {item}" for item in warnings) if warnings else "- none"
    context = (
        f"Project name: {project_root.name}\n"
        f"Idea: {session.idea}\n"
        f"Stack: {session.stack}\n"
        f"Model used for generation: {session.model}\n"
        f"Requirements: {session.requirements or 'none'}\n"
        f"Local path: {project_root}\n"
        f"GitHub URL: {github_url or 'not pushed'}\n"
        "Most recent follow-up changed files:\n"
        f"{changed_lines}\n"
        "Project files:\n"
        f"{file_lines}\n"
        "Warnings:\n"
        f"{warning_lines}"
    )
    return context[:8000]


def _filter_followup_fix_candidates(files: list[str]) -> list[str]:
    excluded_names = {
        "readme.md",
        "requirements.txt",
        "poetry.lock",
        "pipfile.lock",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
    }
    allowed_suffixes = {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".html",
        ".css",
        ".scss",
        ".sass",
        ".vue",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".swift",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".rb",
        ".php",
        ".json",
        ".yml",
        ".yaml",
        ".toml",
    }

    selected: list[str] = []
    seen: set[str] = set()
    for path in files:
        normalized = path.strip()
        if not normalized:
            continue
        name = Path(normalized).name.lower()
        if name in excluded_names:
            continue
        if Path(normalized).suffix.lower() not in allowed_suffixes:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(normalized)
    return selected


def _collect_project_files(project_root: Path, limit: int = PROJECT_FILE_TREE_MAX_ENTRIES) -> list[str]:
    excluded_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv"}
    files: list[str] = []

    for root, dirs, file_names in os.walk(project_root):
        dirs[:] = [item for item in dirs if item not in excluded_dirs]
        root_path = Path(root)
        for file_name in sorted(file_names):
            path = root_path / file_name
            try:
                relative = path.relative_to(project_root).as_posix()
            except ValueError:
                continue
            files.append(relative)
            if len(files) >= limit:
                return files
    return files


def _render_project_file_tree(files: list[str]) -> str:
    if not files:
        return "- none"

    rendered = "\n".join(f"- {path}" for path in files)
    if len(rendered) <= PROJECT_FILE_TREE_MAX_CHARS:
        return rendered
    return rendered[:PROJECT_FILE_TREE_MAX_CHARS] + "\n- ..."


def _render_project_history(history_window: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for item in history_window:
        role = item.get("role", "")
        content = item.get("content", "")
        if role not in {"user", "assistant"} or not content:
            continue
        trimmed = content if len(content) <= 400 else content[:400] + "..."
        lines.append(f"{role}: {trimmed}")
    return "\n".join(lines) if lines else "(no prior history)"


def _parse_followup_plan(raw_text: str) -> dict[str, Any]:
    payload = _extract_json_object(raw_text)
    mode = str(payload.get("mode", "chat")).strip().lower()
    if mode not in {"chat", "action"}:
        mode = "chat"

    files_raw = payload.get("files") or []
    files: list[dict[str, str]] = []
    if isinstance(files_raw, list):
        for item in files_raw:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).strip()
            change = str(item.get("change", "")).strip()
            if not path:
                continue
            files.append({"path": path, "change": change})

    push_raw = payload.get("push_to_github", False)
    push_to_github = bool(push_raw)
    if isinstance(push_raw, str):
        push_to_github = push_raw.strip().lower() in {"1", "true", "yes", "y"}

    return {
        "mode": mode,
        "assistant_reply": str(payload.get("assistant_reply", "")).strip(),
        "summary": str(payload.get("summary", "")).strip(),
        "files": files,
        "install_command": str(payload.get("install_command", "")).strip(),
        "validation_command": str(payload.get("validation_command", "")).strip(),
        "push_to_github": push_to_github,
    }


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
        file_count = len(_collect_project_files(project, limit=PROJECT_FILE_TREE_MAX_ENTRIES))
        lines.append(f"- {relative} ({file_count} files)")
    return "\n".join(lines)


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
