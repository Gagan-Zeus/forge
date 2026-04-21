from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class ShellRunner:
    BLOCKED_PATTERNS = (
        re.compile(r"rm\s+-rf\s+/", flags=re.IGNORECASE),
        re.compile(r"\bsudo\b", flags=re.IGNORECASE),
        re.compile(r"\bcurl\b", flags=re.IGNORECASE),
        re.compile(r"\bwget\b", flags=re.IGNORECASE),
        re.compile(r"\beval\b", flags=re.IGNORECASE),
    )

    def __init__(self, allowed_root: str | Path) -> None:
        self.allowed_root = Path(allowed_root).resolve()

    async def run(self, command: str, cwd: str | Path) -> dict[str, Any]:
        work_dir = Path(cwd).resolve()
        if not self._is_allowed_directory(work_dir):
            error = f"Blocked command outside allowed directory: {work_dir}"
            LOGGER.warning(error)
            return self._result(False, "", error, 126)

        if self._contains_blocked_token(command):
            error = "Blocked unsafe command by policy."
            LOGGER.warning("Unsafe command blocked: %s", command)
            return self._result(False, "", error, 126)

        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(work_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        output = stdout.decode("utf-8", errors="replace")
        error = stderr.decode("utf-8", errors="replace")
        success = process.returncode == 0
        return self._result(success, output, error, int(process.returncode or 0))

    def _is_allowed_directory(self, path: Path) -> bool:
        try:
            path.relative_to(self.allowed_root)
        except ValueError:
            return False
        return True

    def _contains_blocked_token(self, command: str) -> bool:
        return any(pattern.search(command) for pattern in self.BLOCKED_PATTERNS)

    @staticmethod
    def _result(success: bool, output: str, error: str, exit_code: int) -> dict[str, Any]:
        return {
            "success": success,
            "output": output,
            "error": error,
            "exit_code": exit_code,
        }
