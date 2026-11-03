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
DEFAULT_VIDEO_SECONDS = 8.0
DEFAULT_MAX_TOTAL_VIDEO_SECONDS = 40.0
DEFAULT_MAX_CLIP_SECONDS = 10.0
DEFAULT_DIALOGUE_WORDS_PER_SECOND = 2.8
DEFAULT_DIALOGUE_OVERAGE_TOLERANCE = 1.15

_DEFAULTS: Dict[str, str] = {
    "image_size": DEFAULT_IMAGE_SIZE,
    "image_aspect_ratio": DEFAULT_IMAGE_ASPECT_RATIO,
    "video_aspect_ratio": DEFAULT_VIDEO_ASPECT_RATIO,
    "video_resolution": DEFAULT_VIDEO_RESOLUTION,
    "video_seconds": str(DEFAULT_VIDEO_SECONDS),
    "max_total_video_seconds": str(DEFAULT_MAX_TOTAL_VIDEO_SECONDS),
    "max_clip_seconds": str(DEFAULT_MAX_CLIP_SECONDS),
    "dialogue_words_per_second": str(DEFAULT_DIALOGUE_WORDS_PER_SECOND),
    "dialogue_overage_tolerance": str(DEFAULT_DIALOGUE_OVERAGE_TOLERANCE),
}

_ENV_NAMES: Dict[str, str] = {
    "image_size": "VIMAX_IMAGE_SIZE",
    "image_aspect_ratio": "VIMAX_IMAGE_ASPECT_RATIO",
    "video_aspect_ratio": "VIMAX_VIDEO_ASPECT_RATIO",
    "video_resolution": "VIMAX_VIDEO_RESOLUTION",
    "video_seconds": "VIMAX_VIDEO_SECONDS",
    "max_total_video_seconds": "VIMAX_MAX_TOTAL_VIDEO_SECONDS",
    "max_clip_seconds": "VIMAX_MAX_CLIP_SECONDS",
    "dialogue_words_per_second": "VIMAX_DIALOGUE_WPS",
    "dialogue_overage_tolerance": "VIMAX_DIALOGUE_OVERAGE_TOLERANCE",
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
    env_chain = os.environ.get("VIMAX_CHAIN_FRAMES", "").strip().lower()
    if env_chain:
        geometry["chain_frames"] = env_chain in _TRUTHY
    else:
        geometry["chain_frames"] = _as_bool(section.get("chain_frames", True))
    geometry["video_seconds"] = clamp_video_seconds(geometry.get("video_seconds"))
    geometry["max_total_video_seconds"] = clamp_max_total_video_seconds(geometry.get("max_total_video_seconds"))
    geometry["max_clip_seconds"] = clamp_max_clip_seconds(geometry.get("max_clip_seconds"))
    geometry["dialogue_words_per_second"] = clamp_bounded_float(geometry.get("dialogue_words_per_second"), DEFAULT_DIALOGUE_WORDS_PER_SECOND, 1.0, 6.0)
    geometry["dialogue_overage_tolerance"] = clamp_bounded_float(geometry.get("dialogue_overage_tolerance"), DEFAULT_DIALOGUE_OVERAGE_TOLERANCE, 1.0, 2.0)
    return geometry


def clamp_video_seconds(value: Any) -> float:
    """Normalize a per-shot video budget to a sane positive float.

    Video models return clips at model-chosen lengths (the API treats duration
    as a suggestion at best), so the pipeline enforces the budget by trimming
    after generation; this value is that budget in seconds.
    """
    try:
        seconds = float(str(value).strip())
    except (TypeError, ValueError):
        return DEFAULT_VIDEO_SECONDS
    if seconds <= 0:
        return DEFAULT_VIDEO_SECONDS
    return min(seconds, 60.0)


def clamp_max_total_video_seconds(value: Any) -> float:
    """Normalize the hard cap on total rendered video length (default 40s).

    Planning and rendering refuse to produce a video longer than this; the
    final cut is also trimmed to it as a last line of defense.
    """
    try:
        seconds = float(str(value).strip())
    except (TypeError, ValueError):
        return DEFAULT_MAX_TOTAL_VIDEO_SECONDS
    if seconds <= 0:
        return DEFAULT_MAX_TOTAL_VIDEO_SECONDS
    return min(seconds, 600.0)


def clamp_bounded_float(value: Any, default: float, low: float, high: float) -> float:
    """Normalize a bounded positive float (fall back to ``default`` when unset)."""
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    if number <= 0:
        return default
    return min(max(number, low), high)


def clamp_max_clip_seconds(value: Any) -> float:
    """Normalize the maximum length of a SINGLE generated clip (default 10s).

    Video models cap what one generation call can return (commonly ~10s), so
    planning never budgets a shot longer than this — a longer beat is split
    into two shots instead of silently losing the tail (observed: a 14s plan
    delivered as a 10s clip with dialogue cut mid-line).
    """
    try:
        seconds = float(str(value).strip())
    except (TypeError, ValueError):
        return DEFAULT_MAX_CLIP_SECONDS
    if seconds <= 0:
        return DEFAULT_MAX_CLIP_SECONDS
    return min(seconds, 60.0)


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
    resolved["chain_frames"] = bool(resolved.get("chain_frames", True))
    resolved["video_seconds"] = clamp_video_seconds(resolved.get("video_seconds"))
    resolved["max_total_video_seconds"] = clamp_max_total_video_seconds(resolved.get("max_total_video_seconds"))
    resolved["max_clip_seconds"] = clamp_max_clip_seconds(resolved.get("max_clip_seconds"))
    resolved["dialogue_words_per_second"] = clamp_bounded_float(resolved.get("dialogue_words_per_second"), DEFAULT_DIALOGUE_WORDS_PER_SECOND, 1.0, 6.0)
    resolved["dialogue_overage_tolerance"] = clamp_bounded_float(resolved.get("dialogue_overage_tolerance"), DEFAULT_DIALOGUE_OVERAGE_TOLERANCE, 1.0, 2.0)
    return resolved
