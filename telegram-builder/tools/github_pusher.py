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
        user = client.get_user(self._github_username) if self._github_username else client.get_user()
        private = visibility.lower() != "public"

        try:
            repository = user.get_repo(repo_name)
            return RepositoryInfo(full_name=repository.full_name, html_url=repository.html_url)
        except GithubException as exc:
            if getattr(exc, "status", None) != 404:
                raise RuntimeError(f"Could not access repository '{repo_name}': {exc.data}") from exc

        try:
            repository = user.create_repo(name=repo_name, private=private, auto_init=False)
        except GithubException as exc:
            raise RuntimeError(
                "Failed to create GitHub repository. Ensure your token has repo permissions."
            ) from exc
        return RepositoryInfo(full_name=repository.full_name, html_url=repository.html_url)

    async def _run_or_raise(self, command: str, cwd: Path, token: str) -> None:
        result = await self._shell_runner.run(command, cwd)
        if not result["success"]:
            self._raise_git_error(command, result, token)

    def _raise_git_error(self, command: str, result: dict[str, object], token: str) -> None:
        output = self._sanitize(str(result.get("output", "")), token)
        error = self._sanitize(str(result.get("error", "")), token)
        raise RuntimeError(f"Command failed: {command}\nstdout:\n{output}\nstderr:\n{error}")

    @staticmethod
    def _sanitize(text: str, token: str) -> str:
        if not token:
            return text
        return text.replace(token, "***")
