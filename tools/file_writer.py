from __future__ import annotations

import re
from pathlib import Path

import aiofiles


class FileWriter:
    def __init__(self, projects_dir: str | Path) -> None:
        self.projects_dir = Path(projects_dir).resolve()
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def create_project_dir(self, project_name: str) -> Path:
        slug = re.sub(r"[^a-zA-Z]+", "-", project_name).strip("-").lower()
        safe_slug = slug if slug else "generated-project"

        base_dir = self.projects_dir / safe_slug
        if not base_dir.exists():
            base_dir.mkdir(parents=True, exist_ok=False)
            return base_dir

        suffix_index = 1
        while True:
            suffix = self._alpha_suffix(suffix_index)
            project_dir = self.projects_dir / f"{safe_slug}-{suffix}"
            if not project_dir.exists():
                project_dir.mkdir(parents=True, exist_ok=False)
                return project_dir
            suffix_index += 1

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

    @staticmethod
    def _alpha_suffix(index: int) -> str:
        letters: list[str] = []
        value = index
        while value > 0:
            value -= 1
            letters.append(chr(ord("a") + (value % 26)))
            value //= 26
        return "".join(reversed(letters))
