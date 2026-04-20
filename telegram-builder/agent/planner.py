from __future__ import annotations

import json
import logging
import re
from pathlib import PurePosixPath
from typing import Any

from models.copilot_client import CopilotClient

LOGGER = logging.getLogger(__name__)


class ProjectPlanner:
    def __init__(self, copilot_client: CopilotClient) -> None:
        self._copilot_client = copilot_client

    async def plan_files(self, idea: str, stack: str, requirements: str, model: str) -> list[dict[str, str]]:
        prompt = (
            "Design a production-ready project file plan as JSON only.\n"
            "Return exactly this schema: {\"files\":[{\"path\":\"...\",\"description\":\"...\"}]}\n"
            "Rules:\n"
            "- Include all critical files for a runnable project.\n"
            "- Use forward-slash paths.\n"
            "- Do not include binary files.\n"
            "- Keep file descriptions short and actionable.\n\n"
            f"Project idea: {idea}\n"
            f"Stack: {stack}\n"
            f"Special requirements: {requirements or 'none'}\n"
        )
        response = await self._copilot_client.call(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            system_prompt="You are a senior software architect. Return valid JSON only.",
        )
        payload = self._extract_json(response)
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
        return validated

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
