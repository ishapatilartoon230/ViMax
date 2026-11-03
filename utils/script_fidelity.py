"""Mechanical script fidelity checks.

The storyboard agent rewrites the user's script into structured shots — an
LLM rewrite is lossy, so fidelity is verified mechanically, never by asking
another LLM whether it "looks fine":

- every quoted dialogue line in the user's script must appear verbatim
  (whitespace/punctuation-insensitive) in some shot's dialogue fields;
- when the script marks explicit shots/scenes (SHOT 1, SCENE 2, …), the plan
  must contain exactly that many shots.

A failure carries the exact missing text so the rebuild instruction is
actionable without interpretation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

_QUOTED_LINE_RE = re.compile(r'"([^"]{2,})"')
_SHOT_MARKER_RE = re.compile(r"^\s*(?:[#*\s-]*)(?:SHOT|SCENE)\s*([0-9]+)", re.IGNORECASE | re.MULTILINE)
_PUNCT_STRIP_RE = re.compile(r"[^\w\sऀ-ॿ]", re.UNICODE)


def extract_quoted_lines(text: str) -> List[str]:
    return [line.strip() for line in _QUOTED_LINE_RE.findall(str(text or "")) if line.strip()]


def normalize_line(line: str) -> str:
    """Comparison form: collapse whitespace, drop punctuation, keep script."""
    text = _PUNCT_STRIP_RE.sub(" ", str(line or ""))
    return " ".join(text.lower().split())


def expected_shot_count(script: str) -> int | None:
    markers = _SHOT_MARKER_RE.findall(str(script or ""))
    if not markers:
        return None
    return len({int(m) for m in markers})


def _plan_dialogue_text(shots: List[Any]) -> str:
    parts: List[str] = []
    for shot in shots:
        if isinstance(shot, dict):
            parts.append(str(shot.get("audio_desc", "") or ""))
            parts.append(str(shot.get("visual_desc", "") or ""))
        else:
            parts.append(str(getattr(shot, "audio_desc", "") or ""))
            parts.append(str(getattr(shot, "visual_desc", "") or ""))
    return "\n".join(parts)


def missing_dialogue_lines(source_text: str, target_text: str) -> List[str]:
    """Quoted lines present in the source but absent (normalized) from the target."""
    target_normalized = normalize_line(target_text)
    missing: List[str] = []
    for line in extract_quoted_lines(source_text):
        normalized = normalize_line(line)
        if normalized and normalized not in target_normalized:
            missing.append(line)
    return missing


def script_fidelity_report(script: str, shots: List[Any]) -> Dict[str, Any]:
    script_lines = extract_quoted_lines(script)
    plan_blob = _plan_dialogue_text(shots)
    plan_lines = extract_quoted_lines(plan_blob)
    plan_normalized = {normalize_line(line) for line in plan_lines}
    plan_blob_normalized = normalize_line(plan_blob)
    missing: List[str] = []
    for line in script_lines:
        normalized = normalize_line(line)
        if normalized and normalized not in plan_normalized and normalized not in plan_blob_normalized:
            missing.append(line)
    expected = expected_shot_count(script)
    shot_count_ok = expected is None or expected == len(shots)
    return {
        "script_dialogue_lines": len(script_lines),
        "missing_dialogue": missing,
        "expected_shots": expected,
        "actual_shots": len(shots),
        "shot_count_ok": shot_count_ok,
        "all_present": not missing and shot_count_ok,
    }


def script_fidelity_error_message(report: Dict[str, Any]) -> str | None:
    if report["all_present"]:
        return None
    parts: List[str] = []
    if report["missing_dialogue"]:
        quoted = " | ".join(f'"{line}"' for line in report["missing_dialogue"])
        parts.append(
            "The plan lost dialogue lines that exist in the user's script — they must appear verbatim: "
            f"{quoted}. Rebuild the affected shots keeping the user's exact wording."
        )
    if not report["shot_count_ok"]:
        parts.append(
            f"The user's script explicitly defines {report['expected_shots']} shots but the plan has "
            f"{report['actual_shots']}. The shot count must match exactly."
        )
    return "Script fidelity check failed. " + " ".join(parts)
