"""Config loading with YAML support when PyYAML is installed and JSON fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read YAML configs; use JSON as a fallback") from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"config root must be an object: {config_path}")
    return data


def dump_config(data: dict[str, Any], path: str | Path) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to write YAML configs; use JSON as a fallback") from exc
        text = yaml.safe_dump(data, sort_keys=False)
    else:
        text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    config_path.write_text(text, encoding="utf-8")
