# ViMax — Features

Status: **395 tests green** (2026-11-26). Fix history lives in [`fixlog/`](fixlog/) (one file per date).

## Three ways to make a video
| Workflow | Input | What happens |
|---|---|---|
| **script2video** | A script, screenplay, or shot list (pasted into the chat) | Formatted into CLIPS → one approval → rendered |
| **idea2video** | A short concept/idea | Story developed → script → rendered |
| **novel2video** | Novel / book text | Events extracted → scenes → rendered |

Runs in the **web UI**, the **CLI**, or the **agent chat**. Progress events stream live, and every stage is **resume-safe** — re-running continues from the files already on disk (finished shots are never regenerated).

## Dialogue language — anything 🌍
Every quoted line is carried **verbatim** through storyboard, decomposition, and the video prompt, which explicitly says: *"Speak every quoted dialogue line exactly as written, in its original language — never translate, transliterate, or replace it."* Hindi, Tamil, Telugu, English, or mixed — whatever language the script is written in is what gets spoken. The pipeline never translates.

## Visual continuity — the strict Flow-style chain
```
AI character portraits → shot 0 first frame (the ONLY generated keyframe)
→ shot 0 video → real last frame extracted (video_last_frame.png)
→ copied as shot 1 first frame → shot 1 video → extract → shot 2 …
→ concat → final_video.mp4
```
- Chains **every shot — camera changes too** (a new angle is described in words; the model re-frames from the chained frame)
- **[Color lock]** in every chained clip: same white balance / exposure / contrast / saturation — no wash-out down the chain
- Broken-chain fallback: if a clip fails, the next shot's keyframe is generated so a missing `first_frame.png` never blocks the render
- Toggle: `render.chain_frames: false` restores the old camera-tree + per-shot keyframe + concurrent mode

## Character & voice
- Per-character **AI portraits** (front/side/back), used as references everywhere
- **[Identity lock]** in every image and video prompt — same face, hair, glasses, jewelry, exact garments
- **VLM best-image selection**: 2 candidates generated, a vision model picks the best
- **[Voice lock]** — and when the script gives no voice description, a voice is derived from the character's age/gender so the lock is never missing
- **Off-camera speakers** always sound distinct from the on-screen character, are never visible, and the on-screen mouth stays closed during their lines
- Shots with no dialogue carry a **[No dialogue]** block — the model may not invent speech

## Timing & speech — dialogue is never cut
- Speech is measured (2.8 words/sec, configurable) and shots **extend to fit** — lines are never trimmed
- Over the 10s clip cap → **split recommendation** instead of a cut
- Multi-speaker clips get a **[Pacing]** rule: lines back-to-back, no long silences
- `clip_short` / `clip_over` progress events warn when a clip under- or over-delivered its budget
- Script-stated durations are preserved exactly; total video capped at 40s (configurable)

## Script fidelity
- Explicit `SHOT`/`SCENE` markers → the plan keeps your **exact shot count and order**
- **Every dialogue line survives verbatim** — off-camera lines included
- Rotation is never planned (flagged as a feasibility note instead); devices keep their orientation for the whole clip
- Phone grip contract: character's face to camera, screen tilted toward her, screen content replaced in post
- **No baked-in text/logos/watermarks** in frames — overlays, captions, and end cards are listed for post-production

## Providers & config
- Video: **Google Omni** (current), **yunwu (Veo)**, **OpenRouter** — image and LLM providers swappable in `configs/agent.local.yaml`
- 9:16 / 16:9, resolution, seconds, clip caps — config keys + env overrides (`VIMAX_*`)

## Agent runtime safety
- Tool results capped before reaching the LLM context (no context bombs)
- `read_file` truncates huge files (keeps the tail, tells the agent to grep)
- `loop_history.jsonl` written slim and rotated at 100MB
- Context-limit 400s recover: fat content and stale images are dropped, then the turn retries instead of failing

## Pending (planned, not built)
- **VLM visual verification gates** (`PLAN_visual_gates.md`): auto-verify every keyframe and clip against the plan; regenerate on failure.
