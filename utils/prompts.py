from typing import Any, Callable
import yaml


def _load_prompts(path: str = "promts/promts.yaml") -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

