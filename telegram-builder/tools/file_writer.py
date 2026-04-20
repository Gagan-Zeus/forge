from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import aiofiles


class FileWriter:
    def __init__(self, projects_dir: str | Path) -> None:
        self.projects_dir = Path(projects_dir).resolve()
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def create_project_dir(self, project_name: str) -> Path:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", project_name).strip("-").lower()
        safe_slug = slug if slug else "generated-project"
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        suffix = uuid4().hex[:6]
        project_dir = self.projects_dir / f"{safe_slug}-{timestamp}-{suffix}"
        project_dir.mkdir(parents=True, exist_ok=False)
        return project_dir

    async def write_file(self, project_root: Path, relative_path: str, content: str) -> Path:
        target_path = self._resolve_safe_path(project_root, relative_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target_path, "w", encoding="utf-8") as file_handle:
            await file_handle.write(content)
        return target_path

    async def read_file(self, project_root: Path, relative_path: str) -> str:
        target_path = self._resolve_safe_path(project_root, relative_path)
        async with aiofiles.open(target_path, "r", encoding="utf-8") as file_handle:
            return await file_handle.read()

    @staticmethod
    def _resolve_safe_path(project_root: Path, relative_path: str) -> Path:
        root = project_root.resolve()
        target = (root / relative_path).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Unsafe path outside project root: {relative_path}") from exc
        return target
