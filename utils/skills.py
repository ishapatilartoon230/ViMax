"""Per-agent skill loading for ViMax pipelines.

Skills are general craft-rule markdown files in ``<repo>/skills/``. There is
one file per pipeline stage (``screenwriter``, ``storyboard``,
``decomposition``, ``frame_prompts``, ``image_selection``, ``agent``) plus a
``shared`` file with rules that apply to every stage.

Skills are deliberately generic: they teach *how* to extract and enforce
project facts (characters, wardrobe, props, device orientation, product UI,
timing) from whatever material the user provides. Nothing in a skill file may
hardcode facts about a specific product — those come from the user's own
script/idea/requirements at runtime.

Loading model: ``StageSkills(workspace_root).stage("storyboard")`` returns
``shared.md + storyboard.md`` concatenated (empty string when the repo has no
``skills/`` directory), cached per root. Skill text is always *appended* to
system prompts, never passed through ``str.format``, so literal braces inside
a skill are safe.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

MAX_SKILL_CHARS = 4000
# Combined shared+stage cap. Script-writing agents stack up to three craft
# files (own stage + script_engagement + clip_contract) on top of shared:
# ~12.3k chars; 12500 keeps those intact. Storyboard and single-stage combos
# stay far below it.
MAX_STAGE_CHARS = 12500

STAGES = (
    "shared",
    "screenwriter",
    "storyboard",
    "decomposition",
    "frame_prompts",
    "image_selection",
    "character_extraction",
    "character_portraits",
    "reference_selection",
    "novel_adaptation",
    "script_planning",
    "script_enhancement",
    "script_engagement",
    "clip_contract",
    "orchestrator",
    "agent",
)


@lru_cache(maxsize=8)
def _read_skill_cached(root_key: str, stage: str) -> str:
    path = Path(root_key) / "skills" / f"{stage}.md"
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if len(text) > MAX_SKILL_CHARS:
        text = text[:MAX_SKILL_CHARS].rstrip() + "\n...[skill truncated]"
    return text


def load_skill(workspace_root: str | Path, stage: str) -> str:
    """Return the raw skill text for one stage ('' when absent)."""
    root = Path(workspace_root).resolve()
    return _read_skill_cached(str(root), stage)


class StageSkills:
    """Loads and combines ``shared`` + per-stage skills for a workspace root."""

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self.root = Path(workspace_root).resolve() if workspace_root is not None else None

    def stage(self, stage: str, *, include_shared: bool = True) -> str:
        return self.stages(stage, include_shared=include_shared)

    def stages(self, *names: str, include_shared: bool = True) -> str:
        """Combine ``shared`` (once) + several stage files, in the given order.

        Lets a dialogue-writing agent receive its own stage file plus a
        cross-cutting craft file (e.g. ``stages("screenwriter",
        "script_engagement")``) while the shared rules appear exactly once.
        """
        if self.root is None or not names:
            return ""
        parts: list[str] = []
        if include_shared and names[0] != "shared":
            shared = load_skill(self.root, "shared")
            if shared:
                parts.append(shared)
        seen: set[str] = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            own = load_skill(self.root, name)
            if own:
                parts.append(own)
        block = "\n\n".join(parts)
        if len(block) > MAX_STAGE_CHARS:
            block = block[:MAX_STAGE_CHARS].rstrip() + "\n...[skills truncated]"
        return block
