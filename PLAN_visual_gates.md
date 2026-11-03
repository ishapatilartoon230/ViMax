# NEXT STEP (planned, not yet implemented): Script2Video visual verification gates + skills hygiene + generation context

Status: plan approved-pending — implement next session. Source analysis done 2026-11-04.

## Problems being solved

1. **No review after the storyboard** — shot descriptions, keyframes, video clips, and the final video are never verified. Keyframes only get candidate *selection*, never *verification*. A bad keyframe poisons the whole chained render.
2. **Skills silently truncated** — `skills/shared.md` (5135 chars), `skills/storyboard.md` (4858), `skills/agent.md` (8389) all exceed the loader cap `MAX_SKILL_CHARS=4000` in `utils/skills.py`; their tails never reach any agent.
3. **Video model has no script understanding** — clip prompts are mechanics only (motion + audio + duration + chain instruction), no story context.
4. Dead commented-out code (e.g. `pipelines/script2video_pipeline.py:560-570`).

## Phase 1 — Visual verification gates (main ask)

- New agent `agents/visual_verifier.py` — `VisualVerifier` (VLM), built on the `BestImageSelector` multimodal pattern (b64 images, robust JSON parser, tenacity retry), returning `ReviewVerdict` from `agents/orchestrator.py`:
  - `verify_frame(frame_path, target_desc, reference_pairs)` — description match, character consistency vs portraits, prop facing, anatomy, no baked text.
  - `verify_clip(frame_paths, motion_desc, audio_desc, previous_end_frame_path, own_last_frame_path)` — all of the above + chain continuity (clip starts at previous clip's real end state; ends near own last_frame).
  - Fail-open: exception/unparseable ⇒ pass (never blocks render).
- **Gate 1 (keyframes)** in `generate_and_select_best_image` (`script2video_pipeline.py:956`): verify selected image; fail ⇒ regenerate with fix instruction (max `review.max_retries`, default 2).
- **Gate 2 (clips)** in `generate_video_for_single_shot` (`script2video_pipeline.py:908`): after save+trim, extract first/mid/last frames (helper `_extract_clip_frames`, modeled on `_ensure_video_end_frame`); fail ⇒ delete clip + regenerate with fix instruction (max 1 bounce — video is expensive).
- **Gate 3 (decomposition, text)**: reuse `_review_with_bounce` on shot descriptions vs script+storyboard in `plan_text_artifacts`.
- Config in `utils/review_config.py`: `frames: bool = True`, `clips: bool = True`, env `VIMAX_REVIEW_FRAMES` / `VIMAX_REVIEW_CLIPS`; reuse `review.model`.
- New skill `skills/visual_verification.md` (~1500 chars) registered in `STAGES`.

## Phase 2 — Skills hygiene

- Trim `shared.md`, `storyboard.md`, `agent.md` to ≤4000 chars content-preserving (their tails are already dropped by the loader, so this restores lost rules).
- Guard test in `tests/test_skills.py`: every shipped skill file ≤ `MAX_SKILL_CHARS`.

## Phase 3 — Script understanding

- Optional `context_desc` on `ShotDescription` (`interfaces/shot_description.py`): 1-2 sentences of story position/emotional state, generated at decomposition, prepended in `_build_video_prompt` as `Story context: ...`. Default None ⇒ backward compatible.

## Phase 4 — Light cleanup (no restructuring)

- Remove dead commented blocks + unused imports in touched files only.

## Tests & verification

- New `tests/test_visual_verifier.py` (verdict parsing, fail-open, gate bounces, config off ⇒ no calls); extend `tests/test_orchestrator_review.py` + `tests/test_skills.py`.
- Run: `pytest tests/test_visual_verifier.py tests/test_orchestrator_review.py tests/test_skills.py tests/test_script2video_pipeline_guards.py tests/test_vimax_adapters.py -q`, then full suite bounded (earlier background full-suite run produced no output — likely hung on a network test, pre-existing).

## Cost

Defaults: +1 VLM call per selected keyframe, +1 per clip, +1 regeneration only on failure (capped). Switchable via `review.frames` / `review.clips`.
