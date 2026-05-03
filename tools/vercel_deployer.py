from __future__ import annotations

import logging
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from tools.shell_runner import ShellRunner

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VercelDeployTarget:
    name: str
    path: Path
    project_name: str
    env_file: Path | None
    env_keys: tuple[str, ...]
    env_values: dict[str, str]


@dataclass(frozen=True)
class VercelDeployResult:
    target: VercelDeployTarget
    success: bool
    deployment_url: str
    error: str
    command: str
    exit_code: int


class VercelDeployer:
    RESERVED_ENV_KEYS = {
        "COPILOT_CLI_PATH",
        "GITHUB_TOKEN",
        "GITHUB_USERNAME",
        "LOG_LEVEL",
        "PROJECTS_DIR",
        "SYSTEM_PROMPT_PATH",
        "TELEGRAM_BOT_TOKEN",
        "VERCEL_SCOPE",
        "VERCEL_TOKEN",
    }
    RESERVED_ENV_PREFIXES = ("COPILOT_", "TELEGRAM_", "VERCEL_")

    DEPLOYABLE_MARKERS = (
        "vercel.json",
        "package.json",
        "next.config.js",
        "next.config.mjs",
        "vite.config.js",
        "vite.config.ts",
        "requirements.txt",
        "pyproject.toml",
        "api",
        "public",
        "src",
        "app",
        "pages",
    )

    def __init__(self, shell_runner: ShellRunner, token: str = "", scope: str = "") -> None:
        self._shell_runner = shell_runner
        self._token = token.strip()
        self._scope = scope.strip()

    async def deploy_project(
        self,
        project_path: Path,
        *,
        production: bool = False,
        force: bool = False,
        target: str = "",
    ) -> list[VercelDeployResult]:
        deploy_targets = self.discover_targets(project_path)
        results: list[VercelDeployResult] = []

        for deploy_target in deploy_targets:
            command = self._build_deploy_command(
                deploy_target=deploy_target,
                production=production,
                force=force,
                target=target,
            )
            result = await self._shell_runner.run(command, deploy_target.path, timeout_seconds=900)
            output = str(result.get("output", "")).strip()
            error = str(result.get("error", "")).strip()
            sanitized_error = self.sanitize_env_values(self._sanitize(error or output), deploy_target.env_values)
            sanitized_command = self.sanitize_env_values(self._sanitize(command), deploy_target.env_values)
            results.append(
                VercelDeployResult(
                    target=deploy_target,
                    success=bool(result.get("success")),
                    deployment_url=self._extract_deployment_url(output),
                    error=sanitized_error,
                    command=sanitized_command,
                    exit_code=int(result.get("exit_code", 0)),
                )
            )

        return results

    def discover_targets(self, project_path: Path) -> list[VercelDeployTarget]:
        root = project_path.resolve()
        split_targets = [root / "frontend", root / "backend"]
        if all(path.is_dir() for path in split_targets):
            return [
                self._build_target(
                    path.name,
                    path,
                    root_env_file=root / ".env",
                    project_name=self._slugify_project_name(f"{root.name}-{path.name}"),
                )
                for path in split_targets
            ]

        child_targets = [
            self._build_target(
                path.name,
                path,
                root_env_file=root / ".env",
                project_name=self._slugify_project_name(f"{root.name}-{path.name}"),
            )
            for path in split_targets
            if path.is_dir() and self._looks_deployable(path)
        ]
        if child_targets:
            return child_targets

        return [self._build_target(root.name, root, root_env_file=root / ".env", project_name=self._slugify_project_name(root.name))]

    def _build_target(self, name: str, path: Path, root_env_file: Path, project_name: str) -> VercelDeployTarget:
        env_values = self._load_env_values(root_env_file)
        local_env_file = path / ".env"
        if local_env_file != root_env_file:
            env_values.update(self._load_env_values(local_env_file))

        env_file = local_env_file if local_env_file.exists() else root_env_file if root_env_file.exists() else None
        return VercelDeployTarget(
            name=name,
            path=path,
            project_name=project_name,
            env_file=env_file,
            env_keys=tuple(sorted(env_values)),
            env_values=env_values,
        )

    def _build_deploy_command(
        self,
        *,
        deploy_target: VercelDeployTarget,
        production: bool,
        force: bool,
        target: str,
    ) -> str:
        parts = ["vercel", "deploy", "--yes", "--archive=tgz"]
        if production:
            parts.append("--prod")
        if force:
            parts.append("--force")
        if target:
            parts.append(f"--target={shlex.quote(target)}")
        if self._token:
            parts.extend(["--token", shlex.quote(self._token)])
        if self._scope:
            parts.extend(["--scope", shlex.quote(self._scope)])
        env_values = deploy_target.env_values
        for key, value in env_values.items():
            assignment = shlex.quote(f"{key}={value}")
            parts.extend(["--env", assignment, "--build-env", assignment])

        return " ".join(parts)

    def _load_env_values(self, env_file: Path) -> dict[str, str]:
        if not env_file.exists() or not env_file.is_file():
            return {}

        values: dict[str, str] = {}
        try:
            parsed = dotenv_values(env_file)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to parse Vercel env file %s: %s", env_file, exc)
            return {}

        for key, value in parsed.items():
            if not key or value is None:
                continue
            if not key.replace("_", "").isalnum() or key[0].isdigit():
                LOGGER.warning("Skipping invalid Vercel env key from %s: %s", env_file, key)
                continue
            if key in self.RESERVED_ENV_KEYS or key.startswith(self.RESERVED_ENV_PREFIXES):
                continue
            values[key] = str(value)
        return values

    def _looks_deployable(self, path: Path) -> bool:
        return any((path / marker).exists() for marker in self.DEPLOYABLE_MARKERS)

    def _extract_deployment_url(self, output: str) -> str:
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("https://") and "vercel" in line.lower():
                return line
        return output.splitlines()[-1].strip() if output else ""

    @staticmethod
    def _slugify_project_name(name: str) -> str:
        slug = "".join(char.lower() if char.isalnum() else "-" for char in name)
        slug = "-".join(part for part in slug.split("-") if part)
        return slug[:100] or "vercel-project"

    def _sanitize(self, text: str) -> str:
        sanitized = text
        if self._token:
            sanitized = sanitized.replace(self._token, "***")
        for key, value in os.environ.items():
            if key.startswith("VERCEL_") and value:
                sanitized = sanitized.replace(value, "***")
        return sanitized

    @staticmethod
    def sanitize_env_values(text: str, values: dict[str, str]) -> str:
        sanitized = text
        for value in values.values():
            if value:
                sanitized = sanitized.replace(value, "***")
        return sanitized
