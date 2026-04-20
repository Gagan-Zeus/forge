from __future__ import annotations

import re

from models.copilot_client import CopilotClient


class BuildFixer:
    def __init__(self, copilot_client: CopilotClient) -> None:
        self._copilot_client = copilot_client

    async def fix_file(
        self,
        file_path: str,
        current_content: str,
        error_output: str,
        project_context: str,
        model: str,
    ) -> str:
        prompt = (
            "Fix the file so validation passes. Return only the full corrected file content, no markdown.\n\n"
            f"Project context:\n{project_context}\n\n"
            f"File path: {file_path}\n"
            f"Current file content:\n{current_content}\n\n"
            f"Validation error output:\n{error_output}\n"
        )
        response = await self._copilot_client.call(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            system_prompt="You are a senior debugging engineer. Return only code.",
        )
        return self._strip_code_fences(response)

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        stripped = text.strip()
        fenced = re.match(r"^```[a-zA-Z0-9_+-]*\n(.*)\n```$", stripped, flags=re.DOTALL)
        if fenced:
            return fenced.group(1)
        return stripped
