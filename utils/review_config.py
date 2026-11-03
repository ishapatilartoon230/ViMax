"""Orchestrator review configuration.

Reads an optional ``review:`` section from ``configs/agent.local.yaml``:

    review:
      enabled: true
      max_retries: 2

Environment overrides: ``VIMAX_REVIEW_ENABLED`` (0/false/off disables) and
``VIMAX_REVIEW_MAX_RETRIES``. Defaults: enabled, 2 retries per stage.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

_TRUTHY = {"1", "true", "yes", "on"}


@lru_cache(maxsize=8)
def _load_review_section_cached(root_key: str) -> Dict[str, Any]:
    path = Path(root_key) / "configs" / "agent.local.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    section = payload.get("review") if isinstance(payload, dict) else None
    return section if isinstance(section, dict) else {}


def review_config_from_workspace(workspace_root: str | Path | None = None) -> Dict[str, Any]:
    """Resolve the review settings for a workspace root."""
    enabled = True
    max_retries = 2
    model = ""
    if workspace_root is not None:
        root = Path(workspace_root).resolve()
        section = _load_review_section_cached(str(root))
        enabled_raw = section.get("enabled")
        if enabled_raw is not None:
            enabled = str(enabled_raw).strip().lower() in _TRUTHY
        retries_raw = section.get("max_retries")
        if isinstance(retries_raw, int) and 0 <= retries_raw <= 5:
            max_retries = retries_raw
        model_raw = section.get("model")
        if isinstance(model_raw, str) and model_raw.strip():
            model = model_raw.strip()
    env_enabled = os.environ.get("VIMAX_REVIEW_ENABLED", "").strip().lower()
    if env_enabled:
        enabled = env_enabled in _TRUTHY
    env_retries = os.environ.get("VIMAX_REVIEW_MAX_RETRIES", "").strip()
    if env_retries.isdigit():
        max_retries = min(5, int(env_retries))
    env_model = os.environ.get("VIMAX_REVIEW_MODEL", "").strip()
    if env_model:
        model = env_model
    return {"enabled": enabled, "max_retries": max_retries, "model": model}
