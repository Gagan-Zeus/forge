from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from agent.planner import ProjectPlan, ProjectPlanner
from bot.session import UserSession
from models.copilot_client import CopilotClient
from tools.dependency_version_resolver import DependencyVersionResolver
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
    MAX_FIX_ATTEMPTS = 3
    MAX_CONTEXT_FILE_CHARS = 6000
    MAX_CONTEXT_FILES = 8

    def __init__(
        self,
        copilot_client: CopilotClient,
        planner: ProjectPlanner,
        file_writer: FileWriter,
        shell_runner: ShellRunner,
        github_pusher: GitHubPusher | None = None,
        dependency_version_resolver: DependencyVersionResolver | None = None,
    ) -> None:
        self._copilot_client = copilot_client
        self._planner = planner
        self._file_writer = file_writer
        self._shell_runner = shell_runner
        self._github_pusher = github_pusher
        self._dependency_version_resolver = dependency_version_resolver or DependencyVersionResolver()
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

        project_name = await self._derive_project_name(session)
        project_dir = self._file_writer.create_project_dir(project_name)
        created_files: list[str] = []
        generated_contents: dict[str, str] = {}
        warnings: list[str] = []
        validation_status = "not run (skipped by /project)"

        try:
            await self._report_status(progress_callback, "PLAN", "Creating project plan...")
            project_plan = await self._planner.plan_project(
                idea=session.idea,
                stack=session.stack,
                requirements=session.requirements,
                model=session.model,
            )
            await self._raise_if_cancelled(cancel_event)
            await self._report_status(
                progress_callback,
                "PLAN",
                self._render_plan_message(project_plan=project_plan),
            )

            await self._report_status(
                progress_callback,
                "BUILD",
                f"Generating files incrementally ({len(project_plan.files)} planned)...",
            )
            build_files = [item for item in project_plan.files if item["path"].strip().lower() != "readme.md"]
            for index, file_item in enumerate(build_files, start=1):
                await self._raise_if_cancelled(cancel_event)
                relative_path = file_item["path"]
                description = file_item["description"]
                await self._report_status(
                    progress_callback,
                    "BUILD",
                    f"Creating {relative_path} ({index}/{len(build_files)})",
                )
                content = await self._generate_file_content(
                    session=session,
                    project_plan=project_plan,
                    file_path=relative_path,
                    file_description=description,
                    generated_contents=generated_contents,
                )
                content, dependency_warnings = await self._dependency_version_resolver.refresh_for_file(
                    relative_path,
                    content,
                )
                if dependency_warnings:
                    warnings.extend(dependency_warnings)
                await self._file_writer.write_file(project_dir, relative_path, content)
                created_files.append(relative_path)
                generated_contents[relative_path] = content
                await self._report_status(progress_callback, "BUILD", f"Created {relative_path}")

            install_command: str | None = None
            validation_command: str | None = None

            await self._raise_if_cancelled(cancel_event)
            await self._report_status(progress_callback, "README", "Generating README.md from actual project outputs...")
            run_command = self._pick_run_command(session.stack, created_files)
            readme_content = await self._generate_readme_content(
                session=session,
                project_plan=project_plan,
                created_files=created_files,
                generated_contents=generated_contents,
                install_command=install_command,
                run_command=run_command,
                validation_command=validation_command,
                validation_status=validation_status,
            )
            if not readme_content:
                readme_content = self._fallback_readme(
                    session=session,
                    created_files=created_files,
                    install_command=install_command,
                    run_command=run_command,
                    validation_command=validation_command,
                )
            await self._file_writer.write_file(project_dir, "README.md", readme_content)
            generated_contents["README.md"] = readme_content
            if "README.md" not in created_files:
                created_files.append("README.md")
            await self._report_status(progress_callback, "README", "Created README.md")

            github_url: str | None = None
            if session.push_to_github:
                if not self._github_pusher:
                    warnings.append("GitHub push skipped: pusher is not configured.")
                else:
                    await self._raise_if_cancelled(cancel_event)
                    await self._report_status(progress_callback, "FINAL", "Pushing project to GitHub...")
                    github_url = await self._github_pusher.push_project(
                        project_path=project_dir,
                        repo_name=session.repo_name,
                        visibility=session.repo_visibility,
                    )
                    await self._report_status(progress_callback, "FINAL", f"GitHub push complete: {github_url}")

            entrypoint = self._infer_entrypoint(created_files)
            run_hint = run_command if run_command and not run_command.startswith("#") else "No run command inferred"
            await self._report_status(
                progress_callback,
                "FINAL",
                (
                    "Project ready.\n"
                    f"Name: {project_name}\n"
                    f"Entry point: {entrypoint}\n"
                    f"Run: {run_hint}"
                ),
            )

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
        project_plan: ProjectPlan,
        file_path: str,
        file_description: str,
        generated_contents: dict[str, str],
        validation_error: str | None = None,
    ) -> str:
        plan_summary = "\n".join(
            f"- {item['path']}: {item['description']}" for item in project_plan.files
        )
        file_tree = "\n".join(f"- {item['path']}" for item in project_plan.files)
        related_context = self._render_related_file_context(
            target_path=file_path,
            plan_files=project_plan.files,
            generated_contents=generated_contents,
        )
        prompt = (
            "Generate the full file content for the requested file.\n"
            "Return only file content without markdown fences.\n"
            "Keep production-quality style with clear error handling and type hints when relevant.\n\n"
            "Cross-file consistency is mandatory:\n"
            "- Keep imports, links, selectors, class names, and API routes consistent with related files.\n"
            "- Do not invent references to files, symbols, or endpoints that are not in the plan.\n"
            "- Preserve compatibility with files that already exist.\n\n"
            "Language boundary rules:\n"
            "- Keep each file language-pure.\n"
            "- .html files must not contain inline CSS (<style>) or inline JavaScript (<script>...</script>).\n"
            "- Put CSS in .css files and JS/TS in .js/.ts files, then reference them from HTML.\n"
            "- .css files contain only CSS; .js/.ts files contain only script code.\n\n"
            f"Project idea: {session.idea}\n"
            f"Stack: {session.stack}\n"
            f"Requirements: {session.requirements or 'none'}\n"
            f"Project description: {project_plan.project_description}\n"
            f"Planned features:\n{self._render_features(project_plan.features)}\n\n"
            f"All planned files:\n{plan_summary}\n\n"
            f"Planned file tree:\n{file_tree}\n\n"
            f"Previously written relevant files:\n{related_context}\n\n"
            f"Target file: {file_path}\n"
            f"Target purpose: {file_description}\n"
        )
        if validation_error:
            prompt += (
                "\nThis file is being regenerated to fix validation failures.\n"
                "Apply the minimal safe change needed to fix the issue while preserving existing behavior.\n"
                f"Validation errors:\n{validation_error[:3000]}\n"
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

    async def _generate_readme_content(
        self,
        session: UserSession,
        project_plan: ProjectPlan,
        created_files: list[str],
        generated_contents: dict[str, str],
        install_command: str | None,
        run_command: str,
        validation_command: str | None,
        validation_status: str,
    ) -> str:
        file_tree = "\n".join(f"- {path}" for path in created_files)
        related_context = self._render_related_file_context(
            target_path="README.md",
            plan_files=project_plan.files,
            generated_contents=generated_contents,
        )
        prompt = (
            "Create an accurate README.md for the generated project.\n"
            "Return only markdown content (no code fences around the entire file).\n"
            "Use only commands and paths that truly exist in this project.\n"
            "Do not mention tools or files that are not listed.\n\n"
            f"Project idea: {session.idea}\n"
            f"Project description: {project_plan.project_description}\n"
            f"Target users: beginner developers and project users\n"
            f"Planned features:\n{self._render_features(project_plan.features)}\n\n"
            f"Actual file tree:\n{file_tree}\n\n"
            "Verified commands:\n"
            f"- Setup/install: {install_command or 'No dependency install command required'}\n"
            f"- Run: {run_command}\n"
            f"- Validation/tests: {validation_command or 'No validation command inferred'}\n"
            f"- Validation status: {validation_status}\n\n"
            f"Relevant source snippets:\n{related_context}\n\n"
            "README requirements:\n"
            "- Include sections: What this project is, Who it is for, Features, Tech stack.\n"
            "- Include Setup instructions and environment variables if used.\n"
            "- Include Run instructions and Testing/Build instructions.\n"
            "- Include Example usage and Limitations.\n"
            "- Keep instructions actionable and concise.\n"
        )
        response = await self._copilot_client.call(
            messages=[{"role": "user", "content": prompt}],
            model=session.model,
            system_prompt="You write practical, accurate project documentation.",
        )
        return self._strip_markdown_fences(response)

    async def _fix_files_from_validation_error(
        self,
        session: UserSession,
        project_plan: ProjectPlan,
        error_text: str,
        created_files: list[str],
        generated_contents: dict[str, str],
        project_dir: Path,
        progress_callback: ProgressCallback,
    ) -> int:
        candidates = self._select_fix_candidates(error_text=error_text, created_files=created_files)
        if not candidates:
            return 0

        fixed_count = 0
        for path in candidates:
            await self._report_status(progress_callback, "VALIDATE", f"Applying fix to {path}")
            description = self._description_for_path(project_plan.files, path)
            updated_content = await self._generate_file_content(
                session=session,
                project_plan=project_plan,
                file_path=path,
                file_description=description,
                generated_contents=generated_contents,
                validation_error=error_text,
            )
            await self._file_writer.write_file(project_dir, path, updated_content)
            generated_contents[path] = updated_content
            fixed_count += 1
            await self._report_status(progress_callback, "VALIDATE", f"Updated {path} to address validation errors")
        return fixed_count

    @classmethod
    def _select_fix_candidates(cls, error_text: str, created_files: list[str]) -> list[str]:
        lowered_error = error_text.lower()
        created = [path for path in created_files if path.lower() != "readme.md"]
        path_map = {path.lower(): path for path in created}

        selected: list[str] = []
        for lowered, original in path_map.items():
            if lowered in lowered_error:
                selected.append(original)

        path_like_matches = re.findall(r"([a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+)", error_text)
        for match in path_like_matches:
            key = match.lower()
            if key in path_map:
                selected.append(path_map[key])

        if not selected:
            selected = cls._default_fix_candidates_for_error(lowered_error, created)

        deduped: list[str] = []
        seen: set[str] = set()
        for item in selected:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
            if len(deduped) == 3:
                break
        return deduped

    @staticmethod
    def _default_fix_candidates_for_error(lowered_error: str, created: list[str]) -> list[str]:
        if any(token in lowered_error for token in ("npm", "node", "yarn", "pnpm")):
            preferred = [
                path
                for path in created
                if Path(path).name in {"package.json", "tsconfig.json", "vite.config.ts", "vite.config.js"}
            ]
            preferred.extend(path for path in created if path.endswith((".js", ".jsx", ".ts", ".tsx")))
            return preferred[:3] if preferred else created[:2]

        if any(token in lowered_error for token in ("pytest", "python", "traceback", "module not found")):
            preferred = [path for path in created if Path(path).name in {"requirements.txt", "pyproject.toml"}]
            preferred.extend(path for path in created if path.endswith(".py"))
            return preferred[:3] if preferred else created[:2]

        return created[:2]

    @staticmethod
    def _description_for_path(plan_files: list[dict[str, str]], file_path: str) -> str:
        for item in plan_files:
            if item["path"] == file_path:
                return item["description"]
        return "Update this file to satisfy project requirements."

    def _render_related_file_context(
        self,
        target_path: str,
        plan_files: list[dict[str, str]],
        generated_contents: dict[str, str],
    ) -> str:
        planned_paths = [item["path"] for item in plan_files]
        candidates = self._related_paths_for_target(target_path, planned_paths, list(generated_contents))
        if not candidates:
            return "(No related files written yet.)"

        chunks: list[str] = []
        for path in candidates[: self.MAX_CONTEXT_FILES]:
            content = generated_contents.get(path, "").strip()
            if not content:
                continue
            snippet = content[: self.MAX_CONTEXT_FILE_CHARS]
            chunks.append(f"### {path}\n{snippet}")
        return "\n\n".join(chunks) if chunks else "(No related files written yet.)"

    @classmethod
    def _related_paths_for_target(
        cls,
        target_path: str,
        planned_paths: list[str],
        available_paths: list[str],
    ) -> list[str]:
        target = Path(target_path)
        target_suffix = target.suffix.lower()
        target_parent = target.parent.as_posix()
        available = set(available_paths)
        scored: list[tuple[int, str]] = []

        for path in planned_paths:
            if path == target_path or path not in available:
                continue
            score = 0
            candidate = Path(path)
            if candidate.parent.as_posix() == target_parent:
                score -= 4
            if candidate.name in {"package.json", "requirements.txt", "pyproject.toml", "tsconfig.json"}:
                score -= 2
            suffix = candidate.suffix.lower()
            if target_suffix == ".html" and suffix in {".css", ".js", ".ts"}:
                score -= 3
            if target_suffix in {".css", ".js", ".ts", ".tsx", ".jsx"} and suffix == ".html":
                score -= 3
            if target_suffix in {".py", ".js", ".ts", ".tsx", ".jsx"} and suffix == target_suffix:
                score -= 2
            if candidate.parts[:1] == ("src",):
                score -= 1
            scored.append((score, path))

        scored.sort(key=lambda item: (item[0], item[1]))
        return [path for _, path in scored]

    @staticmethod
    async def _report_status(progress_callback: ProgressCallback, phase: str, message: str) -> None:
        await progress_callback(f"{phase}: {message}")

    @classmethod
    def _render_plan_message(cls, project_plan: ProjectPlan) -> str:
        feature_lines = "\n".join(f"- {item}" for item in project_plan.features[:6]) or "- (none)"
        file_lines = "\n".join(f"- {item['path']}" for item in project_plan.files[:20]) or "- (none)"
        return (
            "Plan ready.\n"
            f"{project_plan.project_description}\n"
            f"Features:\n{feature_lines}\n"
            f"File tree:\n{file_lines}"
        )

    @staticmethod
    def _render_features(features: list[str]) -> str:
        if not features:
            return "- (none)"
        return "\n".join(f"- {item}" for item in features)

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        stripped = text.strip()
        fenced = re.match(r"^```[a-zA-Z0-9_+-]*\n(.*)\n```$", stripped, flags=re.DOTALL)
        if fenced:
            return fenced.group(1)
        return stripped

    async def _derive_project_name(self, session: UserSession) -> str:
        if session.repo_name.strip():
            sanitized_repo_name = self._sanitize_project_name(session.repo_name)
            if sanitized_repo_name:
                return sanitized_repo_name

        prompt = (
            "Suggest a concise and relevant project folder name for this request.\n"
            "Return only one kebab-case name.\n"
            "Rules:\n"
            "- 2 to 5 words\n"
            "- lowercase letters and hyphen only\n"
            "- no numbers\n"
            "- no quotes, markdown, or explanation\n\n"
            f"Project request: {session.idea}\n"
            f"Stack: {session.stack or 'general'}\n"
            f"Requirements: {session.requirements or 'none'}\n"
        )
        try:
            response = await self._copilot_client.call(
                messages=[{"role": "user", "content": prompt}],
                model=session.model,
                system_prompt="You generate precise software project names.",
            )
            first_line = response.strip().splitlines()[0] if response.strip() else ""
            suggested_name = self._sanitize_project_name(first_line)
            if suggested_name:
                return suggested_name
        except Exception:  # noqa: BLE001
            LOGGER.debug("Could not derive project name from model; falling back to heuristic.", exc_info=True)

        return self._fallback_project_name(session.idea)

    @staticmethod
    def _sanitize_project_name(raw_name: str) -> str:
        lowered = raw_name.strip().lower()
        lowered = lowered.replace("_", "-").replace(" ", "-")
        lowered = re.sub(r"[^a-z-]+", "-", lowered)
        lowered = re.sub(r"-+", "-", lowered).strip("-")
        if not lowered:
            return ""
        parts = [part for part in lowered.split("-") if part]
        if len(parts) > 5:
            parts = parts[:5]
        return "-".join(parts)

    @classmethod
    def _fallback_project_name(cls, idea: str) -> str:
        words = [word for word in re.split(r"\W+", idea.lower()) if word]
        stop_words = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "contains",
            "contain",
            "single",
            "file",
            "files",
            "saying",
            "just",
            "only",
            "no",
            "to",
            "for",
            "with",
            "that",
            "this",
            "new",
            "create",
            "build",
            "generate",
            "make",
            "project",
            "app",
            "okay",
            "please",
        }
        filtered = [word for word in words if word not in stop_words and word.isalpha()]
        candidate_words = filtered[:4] if filtered else words[:4]
        candidate = cls._sanitize_project_name("-".join(candidate_words))
        return candidate or "generated-project"

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
        package_json_content: str | None = None,
    ) -> tuple[str | None, bool]:
        readme_validation = cls._pick_validation_command_from_readme(readme_content)
        if readme_validation:
            return readme_validation, True

        lowered_stack = stack.lower()
        file_set = set(created_files)
        has_python_files = any(path.endswith(".py") for path in created_files)
        has_package_json = any(Path(path).name == "package.json" for path in created_files)

        if any(keyword in lowered_stack for keyword in ("node", "react", "next", "javascript", "typescript")) or has_package_json:
            return cls._pick_node_validation_command(package_json_content), False

        if has_python_files:
            if any(path.startswith("tests/") for path in created_files):
                return "python -m pytest -q", False
            for candidate in ("main.py", "app.py", "src/main.py", "src/app.py"):
                if candidate in file_set:
                    return f"python {candidate}", False
            return "python -m compileall -q .", False
        if any(path.lower().endswith(".html") for path in created_files):
            entrypoint = cls._pick_static_html_entrypoint(created_files)
            return cls._build_static_web_validation_command(entrypoint), False
        return None, False

    @staticmethod
    def _pick_static_html_entrypoint(created_files: list[str]) -> str:
        for candidate in ("index.html", "public/index.html", "src/index.html"):
            if candidate in created_files:
                return candidate
        for path in created_files:
            if path.lower().endswith(".html"):
                return path
        return "index.html"

    @classmethod
    def _pick_node_validation_command(cls, package_json_content: str | None) -> str:
        scripts = cls._extract_package_scripts(package_json_content)
        if not scripts:
            return "npm run build --if-present && npm run lint --if-present"

        preferred_order = ("test:ci", "test", "check", "verify", "lint", "typecheck", "build")
        selected: list[str] = []
        for name in preferred_order:
            script = scripts.get(name)
            if not script:
                continue
            if not cls._is_meaningful_npm_script(name, script):
                continue
            selected.append(f"npm run {name}")
            if len(selected) == 2:
                break

        if selected:
            return " && ".join(selected)
        return "npm run build --if-present && npm run lint --if-present"

    @staticmethod
    def _extract_package_scripts(package_json_content: str | None) -> dict[str, str]:
        if not package_json_content:
            return {}
        try:
            payload = json.loads(package_json_content)
        except json.JSONDecodeError:
            return {}
        raw_scripts = payload.get("scripts")
        if not isinstance(raw_scripts, dict):
            return {}
        scripts: dict[str, str] = {}
        for key, value in raw_scripts.items():
            name = str(key).strip()
            command = str(value).strip()
            if not name or not command:
                continue
            scripts[name] = command
        return scripts

    @staticmethod
    def _is_meaningful_npm_script(name: str, script: str) -> bool:
        lowered = script.lower()
        if name == "test":
            if "no test specified" in lowered:
                return False
            if "exit 1" in lowered and "jest" not in lowered and "vitest" not in lowered and "mocha" not in lowered:
                return False
        blocked = ("npm run dev", "vite", "next dev", "react-scripts start", "--watch")
        return not any(token in lowered for token in blocked)

    @staticmethod
    def _build_static_web_validation_command(entrypoint: str) -> str:
        safe_entrypoint = entrypoint.replace('"', "")
        return (
            "python -c \"from pathlib import Path;import re,sys;root=Path('.');"
            f"html=root/'{safe_entrypoint}';"
            "text=html.read_text(encoding='utf-8');"
            "paths=re.findall(r'(?:href|src)=\\\"([^\\\"]+)\\\"', text);"
            "missing=[p for p in paths if p and not p.startswith(('http://','https://','#','mailto:','javascript:')) "
            "and not (root/p).exists()];"
            "print('static refs ok' if not missing else 'missing refs: ' + ', '.join(missing));"
            "sys.exit(0 if not missing else 1)\""
        )

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
    def _fallback_readme(
        cls,
        session: UserSession,
        created_files: list[str],
        install_command: str | None,
        run_command: str,
        validation_command: str | None,
    ) -> str:

        install_line = install_command or "# No dependency install command inferred"
        validation_line = validation_command or "# No validation command inferred"
        project_title = cls._fallback_project_name(session.idea)

        return (
            f"# {project_title}\n\n"
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
        if any(path.lower().endswith(".html") for path in created_files):
            entrypoint = BuildOrchestrator._pick_static_html_entrypoint(created_files)
            return f"open {entrypoint}"
        return "# Add your run command here"

    @staticmethod
    def _infer_entrypoint(created_files: list[str]) -> str:
        entrypoint_candidates = (
            "src/main.py",
            "main.py",
            "app.py",
            "src/app.py",
            "index.html",
            "public/index.html",
            "src/index.html",
        )
        file_set = set(created_files)
        for candidate in entrypoint_candidates:
            if candidate in file_set:
                return candidate
        return created_files[0] if created_files else "unknown"

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
