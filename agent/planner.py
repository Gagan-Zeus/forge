from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from models.copilot_client import CopilotClient

LOGGER = logging.getLogger(__name__)


@dataclass
class ProjectPlan:
    project_description: str
    features: list[str]
    files: list[dict[str, str]]


@dataclass
class UpdatePlan:
    reasoning: str
    files_to_update: list[dict[str, str]]


class ProjectPlanner:
    MIN_FILE_COUNT = 4
    DEFAULT_MAX_FILES = 10
    WEB_MAX_FILES = 14

    OPTIONAL_PREFIXES = (
        ".github/",
        ".vscode/",
        ".idea/",
        "docs/",
        "docker/",
        "examples/",
        "scripts/",
    )
    OPTIONAL_FILE_NAMES = {
        "contributing.md",
        "changelog.md",
        "license",
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        ".editorconfig",
        ".prettierrc",
        ".prettierignore",
        ".eslintrc",
        ".eslintrc.js",
        ".eslintrc.cjs",
        ".eslintrc.json",
        ".flake8",
        "mypy.ini",
        "pyrightconfig.json",
    }

    ESSENTIAL_PATHS = {
        "main.py",
        "app.py",
        "src/main.py",
        "src/app.py",
        "src/main.ts",
        "src/main.tsx",
        "src/main.js",
        "src/main.jsx",
        "src/index.ts",
        "src/index.tsx",
        "src/index.js",
        "src/index.jsx",
        "index.html",
    }
    ESSENTIAL_FILE_NAMES = {
        "readme.md",
        "package.json",
        "requirements.txt",
        "pyproject.toml",
        "setup.py",
        "tsconfig.json",
        "vite.config.ts",
        "vite.config.js",
        "next.config.js",
        "next.config.mjs",
    }

    def __init__(self, copilot_client: CopilotClient) -> None:
        self._copilot_client = copilot_client

    async def plan_project(
        self,
        idea: str,
        stack: str,
        requirements: str,
        model: str,
        conversation_key: str | None = None,
    ) -> ProjectPlan:
        prompt = (
            "Design a production-ready project plan as JSON only.\n"
            "Return exactly this schema:\n"
            "{"
            '"project_description":"...",'
            '"features":["..."],'
            '"files":[{"path":"...","description":"..."}]'
            "}\n"
            "Rules:\n"
            "- project_description must be 1-2 concise sentences.\n"
            "- features must include 3-8 concrete, user-facing capabilities.\n"
            "- README.md is mandatory and must always be included at project root.\n"
            "- Include only files that are strictly required for a runnable project.\n"
            "- Exclude optional files unless explicitly requested: tests, docs, CI, Docker, lint configs, licenses, examples.\n"
            "- Keep the file list lean (normally 6-14 files).\n"
            "- Use forward-slash paths.\n"
            "- Do not include binary files.\n"
            "- Keep file descriptions short and actionable.\n\n"
            "- Strict language boundaries: each file should contain only its own language.\n"
            "- HTML files must not include inline CSS or inline JS; place those in .css/.js files.\n"
            "- If the web UI needs styling or behavior, include separate stylesheet/script files and reference them from HTML.\n\n"
            f"Project idea: {idea}\n"
            f"Stack: {stack}\n"
            f"Special requirements: {requirements or 'none'}\n"
        )
        response = await self._copilot_client.call(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            system_prompt="You are a senior software architect. Return valid JSON only.",
            conversation_key=conversation_key,
        )
        payload = self._extract_json(response)
        project_description = str(payload.get("project_description", "")).strip()
        features = self._extract_features(payload.get("features"))
        files = self._extract_and_trim_files(payload, stack=stack, idea=idea, requirements=requirements)
        if not project_description:
            project_description = f"A generated project for: {idea.strip() or 'user request'}."
        if not features:
            features = self._fallback_features(files)
        return ProjectPlan(
            project_description=project_description,
            features=features,
            files=files,
        )

    async def plan_updates(
        self,
        update_prompt: str,
        project_context: str,
        file_tree: str,
        model: str,
        conversation_key: str | None = None,
    ) -> UpdatePlan:
        prompt = (
            "Plan the necessary file updates for the existing project based on the user's request.\n"
            "Return exactly this JSON schema:\n"
            "{\n"
            '  "reasoning": "Brief explanation of what changes are needed.",\n'
            '  "files_to_update": [\n'
            '    {"path": "...", "description": "What to change in this file or why it is being created."}\n'
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Include ONLY files that need to be changed or newly created.\n"
            "- If a file exists, its path must match exactly as shown in the file tree.\n"
            "- Do not include files that don't need changes.\n"
            "- Use forward-slash paths.\n"
            "- Keep descriptions technical and concise.\n\n"
            f"User update request: {update_prompt}\n\n"
            f"Project Context:\n{project_context}\n\n"
            f"Current File Tree:\n{file_tree}\n"
        )
        response = await self._copilot_client.call(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            system_prompt="You are a senior software architect planning a code update. Return valid JSON only.",
            conversation_key=conversation_key,
        )
        payload = self._extract_json(response)
        reasoning = str(payload.get("reasoning", "Updating project files.")).strip()
        files = payload.get("files_to_update", [])
        if not isinstance(files, list):
            files = []

        validated: list[dict[str, str]] = []
        for item in files:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).strip()
            description = str(item.get("description", "")).strip()
            if not path or not description:
                continue
            if not self._is_safe_relative_path(path):
                continue
            validated.append({"path": path, "description": description})

        return UpdatePlan(reasoning=reasoning, files_to_update=validated)

    async def plan_files(
        self,
        idea: str,
        stack: str,
        requirements: str,
        model: str,
        conversation_key: str | None = None,
    ) -> list[dict[str, str]]:
        return (
            await self.plan_project(
                idea=idea,
                stack=stack,
                requirements=requirements,
                model=model,
                conversation_key=conversation_key,
            )
        ).files

    def _extract_and_trim_files(
        self,
        payload: dict[str, Any],
        *,
        stack: str,
        idea: str,
        requirements: str,
    ) -> list[dict[str, str]]:
        files = payload.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("Planner returned no files.")

        validated: list[dict[str, str]] = []
        for item in files:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).strip()
            description = str(item.get("description", "")).strip()
            if not path or not description:
                continue
            if not self._is_safe_relative_path(path):
                LOGGER.warning("Skipping unsafe path from planner: %s", path)
                continue
            validated.append({"path": path, "description": description})

        if not validated:
            raise ValueError("Planner output did not include any valid file entries.")

        required = self._ensure_required_files(validated)
        trimmed = self._trim_to_required_files(required, stack=stack, idea=idea, requirements=requirements)
        if len(trimmed) != len(validated):
            LOGGER.info("Planner reduced file plan from %s to %s required files.", len(validated), len(trimmed))
        return trimmed

    @staticmethod
    def _extract_features(raw_features: Any) -> list[str]:
        if not isinstance(raw_features, list):
            return []
        features: list[str] = []
        for item in raw_features:
            feature = str(item).strip()
            if not feature:
                continue
            features.append(feature)
            if len(features) == 8:
                break
        return features

    @staticmethod
    def _fallback_features(files: list[dict[str, str]]) -> list[str]:
        features: list[str] = []
        for item in files:
            description = str(item.get("description", "")).strip()
            if not description:
                continue
            features.append(description[0].upper() + description[1:] if len(description) > 1 else description.upper())
            if len(features) == 5:
                break
        if features:
            return features
        return ["Core project scaffolding", "Runnable entry point", "Basic documentation"]

    @staticmethod
    def _extract_json(raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])

        raise ValueError("Could not parse JSON from planner response.")

    @staticmethod
    def _is_safe_relative_path(path: str) -> bool:
        candidate = PurePosixPath(path)
        if candidate.is_absolute():
            return False
        return ".." not in candidate.parts

    @classmethod
    def _trim_to_required_files(
        cls,
        files: list[dict[str, str]],
        stack: str,
        idea: str,
        requirements: str,
    ) -> list[dict[str, str]]:
        files = cls._ensure_required_files(files)
        deduped: list[dict[str, str]] = []
        seen_paths: set[str] = set()
        for item in files:
            path = item["path"].strip()
            key = path.lower()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            deduped.append(item)

        request_text = f"{idea}\n{requirements}".lower()
        include_tests = cls._has_any(request_text, ("test", "pytest", "jest", "vitest", "playwright", "cypress"))
        include_docs = cls._has_any(request_text, ("readme", "docs", "documentation"))
        include_docker = cls._has_any(request_text, ("docker", "container", "compose", "kubernetes"))
        include_ci = cls._has_any(request_text, ("github actions", "gitlab ci", "pipeline", "ci/cd", "ci"))
        include_lint = cls._has_any(
            request_text,
            ("eslint", "prettier", "ruff", "flake8", "mypy", "pyright", "lint"),
        )

        filtered = [
            item
            for item in deduped
            if not cls._is_optional(
                item["path"].lower(),
                include_tests=include_tests,
                include_docs=include_docs,
                include_docker=include_docker,
                include_ci=include_ci,
                include_lint=include_lint,
            )
        ]

        selected = filtered or deduped
        stack_lower = stack.lower()
        selected.sort(
            key=lambda item: (
                cls._priority(item["path"].lower(), stack_lower),
                len(PurePosixPath(item["path"]).parts),
                item["path"],
            )
        )

        max_files = cls._max_files_for_stack(stack_lower)
        trimmed = selected[:max_files]

        if len(trimmed) < cls.MIN_FILE_COUNT and len(selected) >= cls.MIN_FILE_COUNT:
            trimmed = selected[: cls.MIN_FILE_COUNT]
        return trimmed

    @staticmethod
    def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in text for keyword in keywords)

    @classmethod
    def _is_optional(
        cls,
        path: str,
        *,
        include_tests: bool,
        include_docs: bool,
        include_docker: bool,
        include_ci: bool,
        include_lint: bool,
    ) -> bool:
        file_name = PurePosixPath(path).name.lower()

        if file_name == "readme.md":
            return False

        if path.startswith("docs/"):
            return not include_docs
        if path.startswith(".github/"):
            return not include_ci
        if path.startswith("docker/"):
            return not include_docker

        if any(path.startswith(prefix) for prefix in cls.OPTIONAL_PREFIXES):
            return True

        if file_name in cls.OPTIONAL_FILE_NAMES:
            if file_name.startswith(".eslint") or file_name.startswith(".prettier"):
                return not include_lint
            if file_name in {".flake8", "mypy.ini", "pyrightconfig.json"}:
                return not include_lint
            if file_name.startswith("docker"):
                return not include_docker
            if file_name == "readme.md":
                return not include_docs
            return True

        if (
            "/tests/" in f"/{path}"
            or path.startswith("tests/")
            or file_name.startswith("test_")
            or file_name.endswith(".test.js")
            or file_name.endswith(".test.ts")
            or file_name.endswith(".spec.js")
            or file_name.endswith(".spec.ts")
        ):
            return not include_tests

        return False

    @staticmethod
    def _ensure_required_files(files: list[dict[str, str]]) -> list[dict[str, str]]:
        has_root_readme = any(item.get("path", "").strip().lower() == "readme.md" for item in files)
        if has_root_readme:
            return files

        with_readme = list(files)
        with_readme.append(
            {
                "path": "README.md",
                "description": "Project overview with setup, run, and testing instructions.",
            }
        )
        return with_readme

    @classmethod
    def _priority(cls, path: str, stack: str) -> int:
        file_name = PurePosixPath(path).name.lower()

        if path in cls.ESSENTIAL_PATHS or file_name in cls.ESSENTIAL_FILE_NAMES:
            return 0

        source_ext = (".py", ".js", ".jsx", ".ts", ".tsx", ".css", ".html")
        if path.startswith(("src/", "app/", "backend/", "frontend/")) and file_name.endswith(source_ext):
            return 1
        if file_name.endswith(source_ext):
            return 2

        if file_name.endswith((".json", ".toml", ".yaml", ".yml", ".env")):
            return 3

        if "react" in stack or "next" in stack or "vite" in stack:
            if file_name in {"index.html", "vite.config.ts", "vite.config.js", "tsconfig.json"}:
                return 0

        return 5

    @classmethod
    def _max_files_for_stack(cls, stack: str) -> int:
        if any(keyword in stack for keyword in ("react", "next", "vite", "frontend", "fullstack")):
            return cls.WEB_MAX_FILES
        return cls.DEFAULT_MAX_FILES
