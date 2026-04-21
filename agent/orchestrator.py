from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

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
        file_writer: FileWriter,
        shell_runner: ShellRunner,
        github_pusher: GitHubPusher | None = None,
    ) -> None:
        self._copilot_client = copilot_client
        self._planner = planner
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

            readme_content = self._load_readme_content(project_dir)
            if not readme_content:
                await progress_callback("README.md missing or empty. Generating required README...")
                fallback_readme = self._fallback_readme(session, created_files)
                await self._file_writer.write_file(project_dir, "README.md", fallback_readme)
                if "README.md" not in created_files:
                    created_files.append("README.md")
                readme_content = fallback_readme

            install_command = self._pick_install_command(session.stack, created_files, readme_content)
            if install_command:
                await self._raise_if_cancelled(cancel_event)
                await progress_callback("Installing dependencies...")
                install_result = await self._shell_runner.run(install_command, project_dir)
                if not install_result["success"]:
                    warnings.append(self._format_command_warning("dependency install", install_result))

            validation_command, validation_from_readme = self._pick_validation_command(
                session.stack,
                created_files,
                readme_content,
            )
            if validation_command:
                await self._raise_if_cancelled(cancel_event)
                if validation_from_readme:
                    command_label = (
                        validation_command
                        if len(validation_command) <= 180
                        else validation_command[:180] + "..."
                    )
                    await progress_callback(f"Running validation from README command: {command_label}")
                else:
                    await progress_callback("Running validation...")
                validation_result = await self._shell_runner.run(validation_command, project_dir)
                if not validation_result["success"] and validation_from_readme:
                    validation_error_text = self._combine_output(validation_result)
                    if self._is_policy_block_error(validation_error_text):
                        fallback_validation_command, _ = self._pick_validation_command(
                            session.stack,
                            created_files,
                            None,
                        )
                        if fallback_validation_command and fallback_validation_command != validation_command:
                            fallback_label = (
                                fallback_validation_command
                                if len(fallback_validation_command) <= 180
                                else fallback_validation_command[:180] + "..."
                            )
                            await progress_callback(
                                "README validation command was blocked by safety policy. "
                                "Falling back to inferred validation command."
                            )
                            await progress_callback(f"Running fallback validation command: {fallback_label}")
                            validation_command = fallback_validation_command
                            validation_result = await self._shell_runner.run(validation_command, project_dir)

                if not validation_result["success"]:
                    validation_error = self._combine_output(validation_result)
                    return BuildResult(
                        success=False,
                        project_name=project_name,
                        project_path=str(project_dir),
                        files_created=created_files,
                        warnings=warnings,
                        error=(
                            "Validation failed after file-by-file generation. "
                            f"Issue: {self._summarize_issue(validation_error)}\n"
                            f"Details:\n{validation_error[:2500]}"
                        ),
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
            "Language boundary rules:\n"
            "- Keep each file language-pure.\n"
            "- .html files must not contain inline CSS (<style>) or inline JavaScript (<script>...</script>).\n"
            "- Put CSS in .css files and JS/TS in .js/.ts files, then reference them from HTML.\n"
            "- .css files contain only CSS; .js/.ts files contain only script code.\n\n"
            f"Project idea: {session.idea}\n"
            f"Stack: {session.stack}\n"
            f"Requirements: {session.requirements or 'none'}\n"
            f"All planned files:\n{plan_summary}\n\n"
            f"Target file: {file_path}\n"
            f"Target purpose: {file_description}\n"
        )
        lowered_file_path = file_path.lower()
        if lowered_file_path.endswith(".html"):
            prompt += (
                "\nHTML strictness:\n"
                "- Use markup structure only.\n"
                "- No <style> blocks.\n"
                "- No inline <script> blocks.\n"
                "- Use <link rel=\"stylesheet\" ...> and <script src=\"...\"></script> references only.\n"
            )
        elif lowered_file_path.endswith((".js", ".jsx", ".ts", ".tsx")):
            prompt += (
                "\nScript strictness:\n"
                "- Return only script code for this file type.\n"
                "- Do not emit HTML/CSS markup in this file.\n"
            )
        elif lowered_file_path.endswith((".css", ".scss", ".sass")):
            prompt += (
                "\nStyle strictness:\n"
                "- Return only style rules for this file type.\n"
                "- Do not emit HTML or JavaScript in this file.\n"
            )
        if file_path.lower() == "readme.md":
            prompt += (
                "\nREADME requirements:\n"
                "- Include sections: Overview, Setup, Run, Testing.\n"
                "- Include command examples in fenced bash blocks.\n"
                "- Testing section must contain non-interactive validation commands that can be run in CI.\n"
                "- Commands must exactly match this project's actual scripts/files.\n"
                "- Avoid curl, wget, sudo, eval, or destructive commands in any command examples.\n"
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

    @classmethod
    def _pick_install_command(
        cls,
        stack: str,
        created_files: list[str],
        readme_content: str | None = None,
    ) -> str | None:
        readme_install = cls._pick_install_command_from_readme(readme_content)
        if readme_install:
            return readme_install

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

    @classmethod
    def _pick_validation_command(
        cls,
        stack: str,
        created_files: list[str],
        readme_content: str | None = None,
    ) -> tuple[str | None, bool]:
        readme_validation = cls._pick_validation_command_from_readme(readme_content)
        if readme_validation:
            return readme_validation, True

        lowered_stack = stack.lower()
        file_set = set(created_files)
        has_python_files = any(path.endswith(".py") for path in created_files)
        has_package_json = any(Path(path).name == "package.json" for path in created_files)

        if any(keyword in lowered_stack for keyword in ("node", "react", "next", "javascript", "typescript")) or has_package_json:
            return "npm run test --if-present && npm run build --if-present", False

        if has_python_files:
            if any(path.startswith("tests/") for path in created_files):
                return "python -m pytest -q", False
            for candidate in ("main.py", "app.py", "src/main.py", "src/app.py"):
                if candidate in file_set:
                    return f"python {candidate}", False
            return "python -m compileall -q .", False
        return None, False

    @classmethod
    def _pick_install_command_from_readme(cls, readme_content: str | None) -> str | None:
        if not readme_content:
            return None
        for command in cls._extract_shell_commands_from_readme(readme_content):
            if cls._is_install_command(command) and cls._is_safe_for_runner(command):
                return command
        return None

    @classmethod
    def _pick_validation_command_from_readme(cls, readme_content: str | None) -> str | None:
        if not readme_content:
            return None

        section_commands = cls._extract_shell_commands_from_markdown_sections(
            readme_content,
            section_keywords=("test", "testing", "validation", "verify", "checks", "qa"),
        )
        selected = cls._filter_validation_commands(section_commands)
        if selected:
            return " && ".join(selected[:2])

        selected = cls._filter_validation_commands(cls._extract_shell_commands_from_readme(readme_content))
        if selected:
            return " && ".join(selected[:2])
        return None

    @classmethod
    def _filter_validation_commands(cls, commands: list[str]) -> list[str]:
        selected: list[str] = []
        for command in commands:
            if not cls._looks_like_shell_command(command):
                continue
            if not cls._is_safe_for_runner(command):
                continue
            if cls._is_environment_setup_command(command):
                continue
            if cls._is_install_command(command) or cls._is_interactive_command(command):
                continue
            if cls._is_validation_command(command):
                selected.append(command)
        return selected

    @classmethod
    def _extract_shell_commands_from_markdown_sections(
        cls,
        readme_content: str,
        section_keywords: tuple[str, ...],
    ) -> list[str]:
        heading_pattern = re.compile(r"^#{1,6}\s+(.+?)\s*$", flags=re.MULTILINE)
        matches = list(heading_pattern.finditer(readme_content))
        if not matches:
            return []

        commands: list[str] = []
        for index, match in enumerate(matches):
            title = match.group(1).strip().lower()
            if not any(keyword in title for keyword in section_keywords):
                continue

            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(readme_content)
            section_body = readme_content[start:end]
            commands.extend(cls._extract_shell_commands_from_readme(section_body))

        return commands

    @staticmethod
    def _extract_shell_commands_from_readme(readme_content: str) -> list[str]:
        fence_pattern = re.compile(
            r"```(?:bash|sh|zsh|shell|cmd|powershell|pwsh)?\s*\n(.*?)```",
            flags=re.DOTALL | re.IGNORECASE,
        )
        commands: list[str] = []

        for block in fence_pattern.findall(readme_content):
            for line in block.splitlines():
                command = line.strip()
                if not command or command.startswith("#"):
                    continue
                if command.startswith("$"):
                    command = command[1:].strip()
                if not command or command.startswith(("cd ", "export ", "set ")):
                    continue
                segments = BuildOrchestrator._split_compound_command(command)
                for segment in segments:
                    if not BuildOrchestrator._looks_like_shell_command(segment):
                        continue
                    if not BuildOrchestrator._is_safe_for_runner(segment):
                        continue
                    commands.append(segment)

        if not commands:
            inline = re.findall(r"`([^`\n]+)`", readme_content)
            for snippet in inline:
                command = snippet.strip()
                segments = BuildOrchestrator._split_compound_command(command)
                for segment in segments:
                    if BuildOrchestrator._looks_like_shell_command(segment):
                        if BuildOrchestrator._is_safe_for_runner(segment):
                            commands.append(segment)

        deduped: list[str] = []
        seen: set[str] = set()
        for command in commands:
            key = command.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(command)
        return deduped

    @staticmethod
    def _looks_like_shell_command(command: str) -> bool:
        lowered = command.lower()
        return lowered.startswith(
            (
                "npm ",
                "pnpm ",
                "yarn ",
                "bun ",
                "python ",
                "pytest",
                "pip ",
                "poetry ",
                "uv ",
                "go ",
                "cargo ",
                "dotnet ",
                "mvn ",
                "gradle ",
                "make ",
            )
        )

    @staticmethod
    def _is_install_command(command: str) -> bool:
        lowered = command.lower()
        return lowered.startswith(
            (
                "npm install",
                "npm ci",
                "pnpm install",
                "yarn install",
                "bun install",
                "python -m pip install",
                "pip install",
                "poetry install",
                "uv pip install",
                "pipenv install",
            )
        )

    @staticmethod
    def _is_interactive_command(command: str) -> bool:
        lowered = command.lower()
        interactive_prefixes = (
            "npm run dev",
            "pnpm dev",
            "yarn dev",
            "bun run dev",
            "npm start",
            "pnpm start",
            "yarn start",
            "vite",
            "next dev",
            "react-scripts start",
            "flask run",
            "python -m http.server",
        )
        if lowered.startswith(interactive_prefixes):
            if lowered.startswith("vite build"):
                return False
            return True

        if lowered.startswith("python -m "):
            parts = lowered.split()
            module = parts[2] if len(parts) >= 3 else ""
            if module in {"pytest", "unittest", "compileall", "py_compile", "mypy", "ruff"}:
                return False
            return True

        if lowered.startswith("python ") and not lowered.startswith("python -m "):
            parts = lowered.split()
            if len(parts) >= 2 and parts[1].endswith(".py"):
                script_name = Path(parts[1]).name
                if script_name in {"run.py", "app.py", "main.py", "manage.py", "server.py", "wsgi.py"}:
                    return True

        return " --watch" in lowered or " watch" in lowered or "--reload" in lowered

    @staticmethod
    def _is_validation_command(command: str) -> bool:
        lowered = command.lower()
        validation_tokens = (
            " test",
            "pytest",
            "vitest",
            "jest",
            "unittest",
            " build",
            " compile",
            "compileall",
            " lint",
            " typecheck",
        )
        if any(token in lowered for token in validation_tokens):
            return True

        if lowered.startswith(("go test", "cargo test", "dotnet test", "mvn test", "gradle test")):
            return True

        if lowered.startswith("python -m "):
            parts = lowered.split()
            module = parts[2] if len(parts) >= 3 else ""
            return module in {"pytest", "unittest", "compileall", "py_compile", "mypy", "ruff"}

        if lowered.startswith("python ") and not lowered.startswith("python -m "):
            parts = lowered.split()
            if len(parts) >= 2 and parts[1].endswith(".py"):
                script_name = Path(parts[1]).name
                if any(token in script_name for token in ("test", "check", "lint", "validate")):
                    return True

        return False

    @staticmethod
    def _is_environment_setup_command(command: str) -> bool:
        lowered = command.lower().strip()
        return lowered.startswith(
            (
                "python -m venv",
                "virtualenv",
                "conda create",
                "pyenv virtualenv",
                "pipenv --python",
            )
        )

    @staticmethod
    def _split_compound_command(command: str) -> list[str]:
        parts = re.split(r"\s*(?:&&|\|\||;)\s*", command)
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _is_safe_for_runner(command: str) -> bool:
        blocked_patterns = (
            r"rm\s+-rf\s+/",
            r"\bsudo\b",
            r"\bcurl\b",
            r"\bwget\b",
            r"\beval\b",
        )
        lowered = command.lower()
        return not any(re.search(pattern, lowered) for pattern in blocked_patterns)

    @staticmethod
    def _is_policy_block_error(error_text: str) -> bool:
        lowered = error_text.lower()
        return (
            "blocked unsafe command by policy" in lowered
            or "blocked command outside allowed directory" in lowered
        )

    @classmethod
    def _fallback_readme(cls, session: UserSession, created_files: list[str]) -> str:
        install_command = cls._pick_install_command(session.stack, created_files, None)
        validation_command, _ = cls._pick_validation_command(session.stack, created_files, None)
        run_command = cls._pick_run_command(session.stack, created_files)

        install_line = install_command or "# No dependency install command inferred"
        validation_line = validation_command or "# No validation command inferred"

        return (
            f"# {cls._derive_project_name(session)}\n\n"
            f"Generated project for: {session.idea}\n\n"
            "## Setup\n\n"
            "```bash\n"
            f"{install_line}\n"
            "```\n\n"
            "## Run\n\n"
            "```bash\n"
            f"{run_command}\n"
            "```\n\n"
            "## Testing\n\n"
            "```bash\n"
            f"{validation_line}\n"
            "```\n"
        )

    @staticmethod
    def _pick_run_command(stack: str, created_files: list[str]) -> str:
        lowered_stack = stack.lower()
        file_set = set(created_files)
        has_package_json = any(Path(path).name == "package.json" for path in created_files)
        has_python_files = any(path.endswith(".py") for path in created_files)

        if any(keyword in lowered_stack for keyword in ("node", "react", "next", "javascript", "typescript")) or has_package_json:
            return "npm run dev"

        if has_python_files:
            for candidate in ("main.py", "app.py", "src/main.py", "src/app.py"):
                if candidate in file_set:
                    return f"python {candidate}"
        return "# Add your run command here"

    @staticmethod
    def _load_readme_content(project_dir: Path) -> str | None:
        readme_path = project_dir / "README.md"
        if not readme_path.exists():
            return None
        try:
            content = readme_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return content or None

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
    async def _raise_if_cancelled(cancel_event: asyncio.Event) -> None:
        if cancel_event.is_set():
            raise BuildCancelledError

    @staticmethod
    def _summarize_issue(error_output: str, max_chars: int = 280) -> str:
        lines = [line.strip() for line in error_output.splitlines() if line.strip()]
        if not lines:
            return "Unknown validation issue."

        priority_tokens = ("error", "failed", "exception", "traceback", "npm err", "assert")
        selected = lines[0]
        for line in lines:
            if any(token in line.lower() for token in priority_tokens):
                selected = line
                break

        if len(selected) <= max_chars:
            return selected
        return selected[:max_chars] + "..."
