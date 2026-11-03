"""Render geometry (aspect ratio, image size, resolution) shared by the agent
runtime and the rendering pipelines.

Resolution priority for every value:

1. Environment variable (``VIMAX_IMAGE_SIZE`` & friends) — highest.
2. The ``render:`` section of ``configs/agent.local.yaml`` — this is what the
   Web UI Settings page writes, so users can switch between landscape and
   vertical (9:16) output without touching code.
3. Built-in landscape defaults.

The Web UI stores every field as a string; ``allow_portrait`` accepts both
boolean and ``"true"``/``"false"`` string forms.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Union

import yaml

DEFAULT_IMAGE_SIZE = "1600x900"
DEFAULT_IMAGE_ASPECT_RATIO = "16:9"
DEFAULT_VIDEO_ASPECT_RATIO = "16:9"
DEFAULT_VIDEO_RESOLUTION = "720p"

_DEFAULTS: Dict[str, str] = {
    "image_size": DEFAULT_IMAGE_SIZE,
    "image_aspect_ratio": DEFAULT_IMAGE_ASPECT_RATIO,
    "video_aspect_ratio": DEFAULT_VIDEO_ASPECT_RATIO,
    "video_resolution": DEFAULT_VIDEO_RESOLUTION,
}

_ENV_NAMES: Dict[str, str] = {
    "image_size": "VIMAX_IMAGE_SIZE",
    "image_aspect_ratio": "VIMAX_IMAGE_ASPECT_RATIO",
    "video_aspect_ratio": "VIMAX_VIDEO_ASPECT_RATIO",
    "video_resolution": "VIMAX_VIDEO_RESOLUTION",
}

_TRUTHY = {"1", "true", "yes", "on"}


@lru_cache(maxsize=4)
def _load_render_section(workspace_root: str) -> Dict[str, Any]:
    path = Path(workspace_root).resolve() / "configs" / "agent.local.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    section = payload.get("render") if isinstance(payload, dict) else None
    return section if isinstance(section, dict) else {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY
    return False


def resolve_render_geometry(workspace_root: Union[str, Path, None] = None) -> Dict[str, Any]:
    """Resolve the full render geometry.

    ``workspace_root`` is required to consult a ``configs/agent.local.yaml``
    ``render:`` section; without it only environment variables and the
    built-in defaults apply (so pipelines never pick up an unrelated local
    config through the current working directory).
    """
    section = _load_render_section(str(workspace_root)) if workspace_root is not None else {}
    geometry: Dict[str, Any] = {}
    for key, env_name in _ENV_NAMES.items():
        value = os.environ.get(env_name, "").strip()
        if not value:
            raw = section.get(key)
            value = raw.strip() if isinstance(raw, str) else ""
        geometry[key] = value or _DEFAULTS[key]
    env_allow = os.environ.get("VIMAX_ALLOW_PORTRAIT", "").strip().lower()
    if env_allow:
        geometry["allow_portrait"] = env_allow in _TRUTHY
    else:
        geometry["allow_portrait"] = _as_bool(section.get("allow_portrait"))
    return geometry


def normalize_render_geometry(
    geometry: Dict[str, Any] | None = None,
    workspace_root: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """Merge a partial geometry dict over the resolved defaults."""
    resolved = resolve_render_geometry(workspace_root)
    if geometry:
        for key in resolved:
            if geometry.get(key) not in (None, ""):
                resolved[key] = geometry[key]
    resolved["allow_portrait"] = bool(resolved["allow_portrait"])
    return resolved
