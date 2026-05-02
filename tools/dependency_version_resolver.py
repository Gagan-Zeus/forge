from __future__ import annotations

import asyncio
import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class DependencyVersionResolver:
    _REQ_SKIP_PREFIXES = (
        "-r ",
        "--requirement ",
        "-c ",
        "--constraint ",
        "-e ",
        "--editable ",
        "git+",
        "http://",
        "https://",
        "file:",
    )
    _REQ_NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?)\s*(?:[<>=!~].*)?$")
    _NPM_SKIP_PREFIXES = ("workspace:", "file:", "link:", "git+", "github:", "http://", "https://")

    def __init__(self, timeout_seconds: float = 6.0) -> None:
        self._timeout_seconds = timeout_seconds
        self._python_cache: dict[str, str | None] = {}
        self._npm_cache: dict[str, str | None] = {}

    async def refresh_for_file(self, relative_path: str, content: str) -> tuple[str, list[str]]:
        lowered = relative_path.lower()
        file_name = lowered.rsplit("/", maxsplit=1)[-1]
        if file_name == "requirements.txt":
            return await self._refresh_requirements_txt(content)
        if file_name == "package.json":
            return await self._refresh_package_json(content)
        return content, []

    async def _refresh_requirements_txt(self, content: str) -> tuple[str, list[str]]:
        warnings: list[str] = []
        updated_lines: list[str] = []
        changed = False

        for raw_line in content.splitlines():
            updated_line, warning, line_changed = await self._refresh_requirement_line(raw_line)
            if warning:
                warnings.append(warning)
            if line_changed:
                changed = True
            updated_lines.append(updated_line)

        updated_content = "\n".join(updated_lines)
        if content.endswith("\n"):
            updated_content += "\n"
        return (updated_content if changed else content), warnings

    async def _refresh_requirement_line(self, line: str) -> tuple[str, str | None, bool]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            return line, None, False
        if any(stripped.lower().startswith(prefix) for prefix in self._REQ_SKIP_PREFIXES):
            return line, None, False

        body = line
        inline_comment = ""
        if " #" in body:
            body, inline_comment = body.split(" #", maxsplit=1)
            inline_comment = f" #{inline_comment}"

        marker = ""
        if ";" in body:
            body, marker_tail = body.split(";", maxsplit=1)
            marker = f";{marker_tail.strip()}"

        match = self._REQ_NAME_PATTERN.match(body.strip())
        if not match:
            return line, None, False

        dependency = match.group(1)
        package_name = dependency.split("[", maxsplit=1)[0]
        latest = await self._latest_python_version(package_name)
        if not latest:
            return line, f"Could not resolve latest PyPI version for '{package_name}'.", False

        refreshed = f"{dependency}=={latest}"
        if marker:
            refreshed = f"{refreshed} {marker}"
        refreshed = f"{refreshed}{inline_comment}"
        return refreshed, None, refreshed != line

    async def _refresh_package_json(self, content: str) -> tuple[str, list[str]]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return content, ["Could not parse package.json to refresh dependency versions."]
        if not isinstance(payload, dict):
            return content, ["Could not parse package.json to refresh dependency versions."]

        warnings: list[str] = []
        changed = False
        dependency_sections = ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies")
        for section in dependency_sections:
            section_data = payload.get(section)
            if not isinstance(section_data, dict):
                continue
            for package, raw_spec in list(section_data.items()):
                spec = str(raw_spec).strip()
                if not spec or self._should_skip_npm_spec(spec):
                    continue
                latest = await self._latest_npm_version(package)
                if not latest:
                    warnings.append(f"Could not resolve latest npm version for '{package}'.")
                    continue
                prefix = "^" if not spec.startswith("~") else "~"
                refreshed = f"{prefix}{latest}"
                if refreshed != spec:
                    section_data[package] = refreshed
                    changed = True

        if not changed:
            return content, warnings
        return json.dumps(payload, indent=2, ensure_ascii=False) + "\n", warnings

    def _should_skip_npm_spec(self, spec: str) -> bool:
        lowered = spec.lower()
        if any(lowered.startswith(prefix) for prefix in self._NPM_SKIP_PREFIXES):
            return True
        return lowered in {"latest", "*"}

    async def _latest_python_version(self, package_name: str) -> str | None:
        normalized = package_name.lower().replace("_", "-")
        if normalized in self._python_cache:
            return self._python_cache[normalized]
        version = await asyncio.to_thread(self._fetch_latest_python_version, package_name)
        self._python_cache[normalized] = version
        return version

    async def _latest_npm_version(self, package_name: str) -> str | None:
        normalized = package_name.lower()
        if normalized in self._npm_cache:
            return self._npm_cache[normalized]
        version = await asyncio.to_thread(self._fetch_latest_npm_version, package_name)
        self._npm_cache[normalized] = version
        return version

    def _fetch_latest_python_version(self, package_name: str) -> str | None:
        url = f"https://pypi.org/pypi/{quote(package_name)}/json"
        data = self._fetch_json(url)
        if not isinstance(data, dict):
            return None
        info = data.get("info")
        if not isinstance(info, dict):
            return None
        version = info.get("version")
        if not version:
            return None
        return str(version).strip() or None

    def _fetch_latest_npm_version(self, package_name: str) -> str | None:
        encoded = quote(package_name, safe="")
        url = f"https://registry.npmjs.org/{encoded}/latest"
        data = self._fetch_json(url)
        if not isinstance(data, dict):
            return None
        version = data.get("version")
        if not version:
            return None
        return str(version).strip() or None

    def _fetch_json(self, url: str) -> Any | None:
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "forge-dependency-resolver/1.0"})
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310
                payload = response.read()
        except (HTTPError, URLError, TimeoutError, OSError):
            return None

        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
