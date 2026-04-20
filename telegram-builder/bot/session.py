from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class UserSession:
    current_step: int = 0
    idea: str = ""
    stack: str = ""
    model: str = ""
    requirements: str = ""
    push_to_github: bool = False
    repo_name: str = ""
    repo_visibility: str = "private"
    is_building: bool = False
    is_authenticated: bool = False
    build_progress: str = "Idle"
    warnings: list[str] = field(default_factory=list)
    awaiting_repo_name: bool = False
    awaiting_repo_visibility: bool = False
    auth_in_progress: bool = False


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[int, UserSession] = {}

    def get(self, chat_id: int) -> UserSession:
        if chat_id not in self._sessions:
            self._sessions[chat_id] = UserSession()
        return self._sessions[chat_id]

    def reset(self, chat_id: int, keep_auth: bool = True) -> UserSession:
        previous = self._sessions.get(chat_id)
        is_authenticated = previous.is_authenticated if previous and keep_auth else False
        auth_in_progress = previous.auth_in_progress if previous and keep_auth else False
        session = UserSession(is_authenticated=is_authenticated, auth_in_progress=auth_in_progress)
        self._sessions[chat_id] = session
        return session

    def clear(self, chat_id: int) -> None:
        self._sessions.pop(chat_id, None)


def build_summary(session: UserSession) -> str:
    requirements = session.requirements if session.requirements else "none"
    lines = [
        "Build summary:",
        f"- Idea: {session.idea}",
        f"- Stack: {session.stack}",
        f"- Model: {session.model}",
        f"- Requirements: {requirements}",
        f"- Push to GitHub: {'yes' if session.push_to_github else 'no'}",
    ]
    if session.push_to_github:
        lines.append(f"- Repo name: {session.repo_name}")
        lines.append(f"- Visibility: {session.repo_visibility}")
    return "\n".join(lines)
