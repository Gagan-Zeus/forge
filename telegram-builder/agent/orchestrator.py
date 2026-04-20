from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from agent.fixer import BuildFixer
from agent.planner import ProjectPlanner
from bot.session import UserSession
from models.copilot_client import CopilotClient
from tools.file_writer import FileWriter
from tools.github_pusher import GitHubPusher
from tools.shell_runner import ShellRunner

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[str], Awaitable[None]]


@dataclass
class BuildResult:
    success: bool
    project_name: str
    project_path: str
    files_created: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    github_url: str | None = None
    error: str | None = None


class BuildCancelledError(RuntimeError):
    """Raised when a running build is canceled by the user."""


class BuildOrchestrator:
    def __init__(
        self,
        copilot_client: CopilotClient,
        planner: ProjectPlanner,
        fixer: BuildFixer,
        file_writer: FileWriter,
        shell_runner: ShellRunner,
        github_pusher: GitHubPusher | None = None,
    ) -> None:
        self._copilot_client = copilot_client
        self._planner = planner
        self._fixer = fixer
        self._file_writer = file_writer
        self._shell_runner = shell_runner
        self._github_pusher = github_pusher
        self._cancel_events: dict[int, asyncio.Event] = {}

    def cancel(self, chat_id: int) -> None:
        event = self._cancel_events.get(chat_id)
        if event:
            event.set()

    async def build_project(
        self,
        chat_id: int,
        session: UserSession,
        progress_callback: ProgressCallback,
    ) -> BuildResult:
        cancel_event = asyncio.Event()
        self._cancel_events[chat_id] = cancel_event

        project_name = self._derive_project_name(session)
        project_dir = self._file_writer.create_project_dir(project_name)
        created_files: list[str] = []
        warnings: list[str] = []

        try:
            await progress_callback("Planning your project...")
            plan = await self._planner.plan_files(
                idea=session.idea,
                stack=session.stack,
                requirements=session.requirements,
                model=session.model,
            )
            await self._raise_if_cancelled(cancel_event)

            await progress_callback(f"Got the plan - building {len(plan)} files...")
            for index, file_item in enumerate(plan, start=1):
                await self._raise_if_cancelled(cancel_event)
                relative_path = file_item["path"]
                description = file_item["description"]
                content = await self._generate_file_content(session, plan, relative_path, description)
                await self._file_writer.write_file(project_dir, relative_path, content)
                created_files.append(relative_path)
                await progress_callback(f"Writing {relative_path}... ({index}/{len(plan)})")

            install_command = self._pick_install_command(session.stack, created_files)
            if install_command:
                await self._raise_if_cancelled(cancel_event)
                await progress_callback("Installing dependencies...")
                install_result = await self._shell_runner.run(install_command, project_dir)
                if not install_result["success"]:
                    warnings.append(self._format_command_warning("dependency install", install_result))

            validation_command = self._pick_validation_command(session.stack, created_files)
            if validation_command:
                await self._raise_if_cancelled(cancel_event)
                await progress_callback("Running validation...")
                validation_result = await self._shell_runner.run(validation_command, project_dir)
                if not validation_result["success"]:
                    fix_error = await self._attempt_fixes(
                        session=session,
                        project_dir=project_dir,
                        created_files=created_files,
                        validation_command=validation_command,
                        initial_error=self._combine_output(validation_result),
                        progress_callback=progress_callback,
                    )
                    if fix_error:
                        return BuildResult(
                            success=False,
                            project_name=project_name,
                            project_path=str(project_dir),
                            files_created=created_files,
                            warnings=warnings,
                            error=fix_error,
                        )
            else:
                warnings.append("Validation skipped because no command could be inferred from the stack.")

            github_url: str | None = None
            if session.push_to_github:
                if not self._github_pusher:
                    warnings.append("GitHub push skipped: pusher is not configured.")
                else:
                    await self._raise_if_cancelled(cancel_event)
                    await progress_callback("Pushing to GitHub...")
                    github_url = await self._github_pusher.push_project(
                        project_path=project_dir,
                        repo_name=session.repo_name,
                        visibility=session.repo_visibility,
                    )
                    await progress_callback(f"Done! Repo: {github_url}")

            return BuildResult(
                success=True,
                project_name=project_name,
                project_path=str(project_dir),
                files_created=created_files,
                warnings=warnings,
                github_url=github_url,
            )
        except BuildCancelledError:
            return BuildResult(
                success=False,
                project_name=project_name,
                project_path=str(project_dir),
                files_created=created_files,
                warnings=warnings,
                error="Build canceled by user.",
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Build failed for chat_id=%s", chat_id)
            return BuildResult(
                success=False,
                project_name=project_name,
                project_path=str(project_dir),
                files_created=created_files,
                warnings=warnings,
                error=str(exc),
            )
        finally:
            self._cancel_events.pop(chat_id, None)

    async def _attempt_fixes(
        self,
        session: UserSession,
        project_dir: Path,
        created_files: list[str],
        validation_command: str,
        initial_error: str,
        progress_callback: ProgressCallback,
    ) -> str | None:
        attempts = {path: 0 for path in created_files}
        latest_error = initial_error
        context = self._render_project_context(session, created_files)

        while True:
            target = self._pick_target_file(latest_error, created_files, attempts)
            if not target:
                return (
                    "Validation failed and no further fixes are possible. "
                    f"Last error:\n{latest_error[:2500]}"
                )

            await progress_callback("Found an issue, fixing...")
            attempts[target] += 1
            current_content = await self._file_writer.read_file(project_dir, target)
            fixed_content = await self._fixer.fix_file(
                file_path=target,
                current_content=current_content,
                error_output=latest_error,
                project_context=context,
                model=session.model,
            )
            await self._file_writer.write_file(project_dir, target, fixed_content)

            validation_result = await self._shell_runner.run(validation_command, project_dir)
            if validation_result["success"]:
                return None

            latest_error = self._combine_output(validation_result)
            if all(count >= 3 for count in attempts.values()):
                return (
                    "Validation failed after max retries (3 attempts per file). "
                    f"Last error:\n{latest_error[:2500]}"
                )

    async def _generate_file_content(
        self,
        session: UserSession,
        plan: list[dict[str, str]],
        file_path: str,
        file_description: str,
    ) -> str:
        plan_summary = "\n".join(f"- {item['path']}: {item['description']}" for item in plan)
        prompt = (
            "Generate the full file content for the requested file.\n"
            "Return only file content without markdown fences.\n"
            "Keep production-quality style with clear error handling and type hints when relevant.\n\n"
            f"Project idea: {session.idea}\n"
            f"Stack: {session.stack}\n"
            f"Requirements: {session.requirements or 'none'}\n"
            f"All planned files:\n{plan_summary}\n\n"
            f"Target file: {file_path}\n"
            f"Target purpose: {file_description}\n"
        )
        response = await self._copilot_client.call(
            messages=[{"role": "user", "content": prompt}],
            model=session.model,
            system_prompt="You are a senior software engineer producing production-ready files.",
        )
        return self._strip_markdown_fences(response)

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        stripped = text.strip()
        fenced = re.match(r"^```[a-zA-Z0-9_+-]*\n(.*)\n```$", stripped, flags=re.DOTALL)
        if fenced:
            return fenced.group(1)
        return stripped

    @staticmethod
    def _derive_project_name(session: UserSession) -> str:
        if session.repo_name:
            return session.repo_name
        words = [word for word in re.split(r"\W+", session.idea.lower()) if word]
        return "-".join(words[:4]) or "generated-project"

    @staticmethod
    def _pick_install_command(stack: str, created_files: list[str]) -> str | None:
        lowered_stack = stack.lower()
        has_package_json = any(Path(path).name == "package.json" for path in created_files)
        has_requirements = any(Path(path).name == "requirements.txt" for path in created_files)
        has_pyproject = any(Path(path).name == "pyproject.toml" for path in created_files)

        if any(keyword in lowered_stack for keyword in ("node", "react", "next", "javascript", "typescript")) or has_package_json:
            return "npm install"
        if has_requirements:
            return "python -m pip install -r requirements.txt"
        if has_pyproject:
            return "python -m pip install -e ."
        return None

    @staticmethod
    def _pick_validation_command(stack: str, created_files: list[str]) -> str | None:
        lowered_stack = stack.lower()
        file_set = set(created_files)
        has_python_files = any(path.endswith(".py") for path in created_files)
        has_package_json = any(Path(path).name == "package.json" for path in created_files)

        if any(keyword in lowered_stack for keyword in ("node", "react", "next", "javascript", "typescript")) or has_package_json:
            return "npm run test --if-present && npm run build --if-present"

        if has_python_files:
            if any(path.startswith("tests/") for path in created_files):
                return "python -m pytest -q"
            for candidate in ("main.py", "app.py", "src/main.py", "src/app.py"):
                if candidate in file_set:
                    return f"python {candidate}"
            return "python -m compileall -q ."
        return None

    @staticmethod
    def _combine_output(command_result: dict[str, object]) -> str:
        output = str(command_result.get("output", "")).strip()
        error = str(command_result.get("error", "")).strip()
        if output and error:
            return f"stdout:\n{output}\n\nstderr:\n{error}"
        return output or error or "Unknown command failure"

    @staticmethod
    def _format_command_warning(operation: str, command_result: dict[str, object]) -> str:
        details = BuildOrchestrator._combine_output(command_result)
        return f"{operation.capitalize()} failed: {details[:1000]}"

    @staticmethod
    def _pick_target_file(error_output: str, files: list[str], attempts: dict[str, int]) -> str | None:
        lowered_error = error_output.lower()
        for path in files:
            if attempts[path] >= 3:
                continue
            if path.lower() in lowered_error:
                return path

        for path in files:
            if attempts[path] >= 3:
                continue
            if Path(path).name.lower() in lowered_error:
                return path

        available = [path for path in files if attempts[path] < 3]
        if not available:
            return None
        return min(available, key=lambda item: attempts[item])

    @staticmethod
    def _render_project_context(session: UserSession, created_files: list[str]) -> str:
        files_section = "\n".join(f"- {path}" for path in created_files)
        return (
            f"Idea: {session.idea}\n"
            f"Stack: {session.stack}\n"
            f"Requirements: {session.requirements or 'none'}\n"
            f"Generated files:\n{files_section}"
        )

    @staticmethod
    async def _raise_if_cancelled(cancel_event: asyncio.Event) -> None:
        if cancel_event.is_set():
            raise BuildCancelledError
