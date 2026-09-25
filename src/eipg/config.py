"""Configuration utilities for EIPG clean runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import yaml


@dataclass(frozen=True)
class RunConfig:
    """Thin wrapper around a YAML config dictionary.

    We intentionally keep this lightweight at the beginning of the rebuild. Once the
    controlled benchmark stabilizes, we can replace selected parts with stricter
    dataclasses or pydantic models.
    """

    path: Path
    data: dict[str, Any]

    @property
    def run_name(self) -> str:
        return str(self.data.get("run", {}).get("name", self.path.stem))

    @property
    def seed(self) -> int:
        return int(self.data.get("run", {}).get("seed", 0))

    @property
    def output_dir(self) -> Path:
        raw = self.data.get("run", {}).get("output_dir", f"outputs/{self.run_name}")
        return Path(raw)

    def require_section(self, name: str) -> dict[str, Any]:
        section = self.data.get(name)
        if not isinstance(section, dict):
            raise ValueError(f"Missing required config section: {name}")
        return section

    def write_manifest(self, output_dir: Path | None = None) -> Path:
        """Write a JSON manifest containing the resolved config."""

        out_dir = output_dir or self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "run_manifest.json"
        payload = {
            "config_path": str(self.path),
            "run_name": self.run_name,
            "seed": self.seed,
            "output_dir": str(out_dir),
            "config": self.data,
        }
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return manifest_path


def load_config(path: str | Path) -> RunConfig:
    """Load a YAML config file."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return RunConfig(path=config_path, data=data)
