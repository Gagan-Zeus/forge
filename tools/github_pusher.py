from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from github import Github, GithubException

from tools.shell_runner import ShellRunner

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepositoryInfo:
    full_name: str
    html_url: str


class GitHubPusher:
    def __init__(
        self,
        github_username: str,
        access_token_provider: Callable[[], Awaitable[str]],
        shell_runner: ShellRunner,
    ) -> None:
        self._github_username = github_username
        self._access_token_provider = access_token_provider
        self._shell_runner = shell_runner

    async def push_project(self, project_path: Path, repo_name: str, visibility: str) -> str:
        access_token = await self._access_token_provider()
        repo = await asyncio.to_thread(self._create_or_get_repo, access_token, repo_name, visibility)
        remote_url = f"https://x-access-token:{access_token}@github.com/{repo.full_name}.git"

        await self._run_or_raise("git init", project_path, access_token)
        await self._run_or_raise('git config user.name "telegram-builder"', project_path, access_token)
        await self._run_or_raise('git config user.email "telegram-builder@local"', project_path, access_token)
        await self._run_or_raise("git add .", project_path, access_token)

        commit_result = await self._shell_runner.run('git commit -m "Initial commit from telegram-builder"', project_path)
        combined_commit_output = (commit_result["output"] + "\n" + commit_result["error"]).lower()
        if not commit_result["success"] and "nothing to commit" not in combined_commit_output:
            self._raise_git_error("git commit", commit_result, access_token)

        await self._run_or_raise("git branch -M main", project_path, access_token)
        await self._shell_runner.run("git remote remove origin", project_path)
        await self._run_or_raise(f"git remote add origin '{remote_url}'", project_path, access_token)
        await self._run_or_raise("git push -u origin main", project_path, access_token)
        return repo.html_url

    def _create_or_get_repo(self, access_token: str, repo_name: str, visibility: str) -> RepositoryInfo:
        client = Github(access_token)
        owner_login, owner = self._resolve_owner(client)
        private = visibility.lower() != "public"
        full_name = f"{owner_login}/{repo_name}"

        try:
            repository = client.get_repo(full_name)
            return RepositoryInfo(full_name=repository.full_name, html_url=repository.html_url)
        except GithubException as exc:
            if getattr(exc, "status", None) != 404:
                raise RuntimeError(
                    f"Could not access repository '{full_name}': {self._format_github_exception(exc)}"
                ) from exc

        try:
            repository = owner.create_repo(name=repo_name, private=private, auto_init=False)
        except GithubException as exc:
            details = self._format_github_exception(exc)
            if "resource not accessible by integration" in details.lower():
                raise RuntimeError(
                    "Failed to create GitHub repository with the current token. "
                    "Set GITHUB_TOKEN in .env to a Personal Access Token (classic) with 'repo' scope "
                    "or a fine-grained token with repository administration/write permissions, then retry. "
                    f"Details: {details}"
                ) from exc
            raise RuntimeError(
                "Failed to create GitHub repository. "
                "Ensure the authenticated account has permission to create repositories and write access to the target owner. "
                f"Details: {details}"
            ) from exc
        return RepositoryInfo(full_name=repository.full_name, html_url=repository.html_url)

    def _resolve_owner(self, client: Github):
        authenticated_user = client.get_user()
        authenticated_login = str(getattr(authenticated_user, "login", "")).strip()
        requested_owner = self._github_username.strip()

        if not requested_owner:
            return authenticated_login, authenticated_user

        if requested_owner.lower() == authenticated_login.lower():
            return authenticated_login, authenticated_user

        try:
            organization = client.get_organization(requested_owner)
            return str(getattr(organization, "login", requested_owner)), organization
        except GithubException as exc:
            if getattr(exc, "status", None) != 404:
                raise RuntimeError(
                    f"Failed to resolve GitHub owner '{requested_owner}': {self._format_github_exception(exc)}"
                ) from exc

        # If requested owner is not an organization and doesn't match the authenticated user,
        # push to the authenticated user's namespace to avoid hard failure on 404.
        LOGGER.warning(
            "GITHUB_USERNAME='%s' does not match authenticated user '%s' and is not an accessible organization. "
            "Falling back to authenticated user.",
            requested_owner,
            authenticated_login,
        )
        return authenticated_login, authenticated_user

    async def _run_or_raise(self, command: str, cwd: Path, token: str) -> None:
        result = await self._shell_runner.run(command, cwd)
        if not result["success"]:
            self._raise_git_error(command, result, token)

    def _raise_git_error(self, command: str, result: dict[str, object], token: str) -> None:
        output = self._sanitize(str(result.get("output", "")), token)
        error = self._sanitize(str(result.get("error", "")), token)
        raise RuntimeError(f"Command failed: {command}\nstdout:\n{output}\nstderr:\n{error}")

    @staticmethod
    def _format_github_exception(exc: GithubException) -> str:
        status = getattr(exc, "status", "unknown")
        data = getattr(exc, "data", "")
        if isinstance(data, dict):
            message = str(data.get("message", "")).strip()
            errors = data.get("errors")
            if errors:
                return f"HTTP {status}: {message} | errors={errors}"
            if message:
                return f"HTTP {status}: {message}"
            return f"HTTP {status}: {data}"
        if data:
            return f"HTTP {status}: {data}"
        return f"HTTP {status}: {exc}"

    @staticmethod
    def _sanitize(text: str, token: str) -> str:
        if not token:
            return text
        return text.replace(token, "***")
