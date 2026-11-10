"""File-backed prompt registry.

Each prompt lives at `prompts/<name>/<version>.yaml` and is reviewed in git like
code. Every request and every eval run is tagged with the version and a content
hash, so a faithfulness score always points at an exact prompt. `sync_to_mlflow`
mirrors the files into the MLflow Prompt Registry for browsing in the UI.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .vectorstore import Chunk


def format_documents(chunks: list[Chunk]) -> str:
    return "\n".join(
        f'<document index="{i}" source="{c.source}">\n{c.text}\n</document>'
        for i, c in enumerate(chunks, start=1)
    )


@dataclass(frozen=True)
class PromptVersion:
    name: str
    version: str
    description: str
    system: str
    user_template: str

    @property
    def sha(self) -> str:
        return hashlib.sha256(f"{self.system}\x00{self.user_template}".encode()).hexdigest()[:12]

    def render(self, question: str, chunks: list[Chunk]) -> tuple[str, str]:
        user = self.user_template.replace("{{documents}}", format_documents(chunks)).replace(
            "{{question}}", question
        )
        return self.system, user


def _version_key(version: str) -> tuple[int, str]:
    match = re.search(r"\d+", version)
    return (int(match.group()) if match else -1, version)


class PromptRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def versions(self, name: str) -> list[str]:
        return sorted((p.stem for p in (self.root / name).glob("*.yaml")), key=_version_key)

    def get(self, name: str, version: str = "latest") -> PromptVersion:
        available = self.versions(name)
        if not available:
            raise FileNotFoundError(f"No prompt versions under {self.root / name}")
        if version == "latest":
            version = available[-1]
        if version not in available:
            raise KeyError(f"Prompt {name}/{version} not found. Available: {available}")
        data = yaml.safe_load((self.root / name / f"{version}.yaml").read_text(encoding="utf-8"))
        for placeholder in ("{{documents}}", "{{question}}"):
            if placeholder not in data["user_template"]:
                raise ValueError(f"{name}/{version}: user_template is missing {placeholder}")
        return PromptVersion(
            name=name,
            version=version,
            description=data.get("description", ""),
            system=data["system"].strip(),
            user_template=data["user_template"].strip(),
        )

    def sync_to_mlflow(self, name: str) -> list[str]:
        """Register any file version whose content hash MLflow has not seen yet."""
        import mlflow

        seen: set[str] = set()
        n = 1
        while True:
            existing = mlflow.genai.load_prompt(f"prompts:/{name}/{n}", allow_missing=True)
            if existing is None:
                break
            seen.add((existing.tags or {}).get("sha", ""))
            n += 1
        registered = []
        for version in self.versions(name):
            prompt = self.get(name, version)
            if prompt.sha in seen:
                continue
            mlflow.genai.register_prompt(
                name=name,
                template=[
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user_template},
                ],
                commit_message=f"{version}: {prompt.description}",
                tags={"file_version": version, "sha": prompt.sha},
            )
            registered.append(version)
        return registered
