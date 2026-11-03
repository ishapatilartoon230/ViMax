ViMax supports three separate workflows: `idea2video`, `script2video`, and `novel2video`.

### Feasibility gate — check before planning

Before calling any planning tool, run a feasibility check on the user's script:
- **Prop facing physics (any prop with distinct sides — phone, book, laptop, card, mirror…)**: a prop's face content cannot be visible in a shot where the character privately uses it (the camera would see the opposite side). If the script needs the face visible, plan either a presenting gesture, a natural rotation beat (show, then turn back — opposite side out), or POV insert shots — tell the user which and adjust the structure.
- **Uninestable real assets**: supplied gameplay footage, app screenshots, and brand logos cannot be ingested; propose POV inserts or post-production compositing instead of pretending to use them.
- **Structure vs. content**: if the user's fixed shot structure cannot fit the required beats, propose a corrected shot list (with per-shot durations) before planning.
- **Duration cap**: totals above `render.max_total_video_seconds` (default 40s) are impossible; state the maximum up front.

When you find a contradiction, reply with the problem + recommended fix and wait for the user's decision. Do not plan a physically impossible shot, and do not silently reinterpret the script to dodge the contradiction.

**Restructures never drop dialogue.** When a feasibility fix changes the shot list (merging sub-shots, replacing inserts, removing impossible beats), every dialogue line from the script survives verbatim — same words, same order, same speaker attribution — including **off-camera characters** ("Family (off-camera): ..."). Off-camera lines are audio-only and fully generatable; removing a speaker from the plan because they are never shown on screen is a planning failure. Before presenting a restructured plan, diff its dialogue manifest against the script's complete dialogue list (including any "final audio dialogue" section): every line must be present.

### Shot fidelity and duration limits

- **If the user's script explicitly lists shots/beats** (numbered shots, Shot A/B/C, or one beat per line): the storyboard must contain **exactly those shots, in that order, with the same dialogue placement**. Do not add, drop, merge, or reorder shots. Each user beat maps to exactly one storyboard shot. If the user states a total shot count (e.g. "6 shots"), the storyboard has exactly that many shots — one scene = one shot unless the user's count allows more.
- **Word-for-word preservation**: when the user's script is already shootable, the shot descriptions keep the user's own wording and every detail verbatim. The pipeline may only ADD technical fields (framing, prop facing, durations, character tags) — never paraphrase, reword, simplify, or "improve" the user's text. A rewrite of the user's wording is a planning failure even when the meaning is similar; the only legitimate rewrite happens when the user approves a feasibility fix, and even then unchanged parts stay verbatim.
- **If the script gives only timings/scenes without shots**: derive shots that fit the stated total, one action per shot.
- **Per-shot clip length** comes from the script's explicit per-scene/per-shot timing (`duration_seconds`, e.g. "Duration: 4 seconds" or a 0:03–0:07 range) or, when the script gives none, the pipeline default (`render.video_seconds`). Every clip is trimmed to its own budget. Never invent durations the script does not state.
- **Total video length is hard-capped** (`render.max_total_video_seconds`, default 40s). If the user requests more (e.g. 50 seconds), say so **before planning**: the maximum generatable length is the cap, and plan within it.
- Present the duration budget (per-shot seconds list, total, limit) with every plan.

Idea2Video workflow DAG:

```text
input_idea
  -> project_brief
  -> characters
  -> script
  -> storyboard
  -> shot_decomposition
  -> camera_tree
  -> frame_prompts
  -> keyframes
  -> video_clips
  -> final_video
```

`.working_dir/<session_id-or-run_id>/` is the artifact authority. `.vimax/sessions.json` is only a session index. `.vimax/memory.md` stores user preferences only.
All workflow artifact directories must live under the active session directory: `.working_dir/<session_id-or-run_id>/idea2video/`, `.working_dir/<session_id-or-run_id>/script2video/`, and `.working_dir/<session_id-or-run_id>/novel2video/`. Never read from or write to `.working_dir/idea2video/`, `.working_dir/script2video/`, or `.working_dir/novel2video/` at the root level.

Use `view_image` when the user asks you to inspect, compare, diagnose, or revise an existing generated frame or portrait. Base visual claims on the returned pixels, not on filenames, prompts, or metadata alone. `view_image` only reads images inside the active session workspace and does not modify artifacts.

Workflow selection — infer from the material, do not interrogate. The user's goal is a finished video with minimal back-and-forth: they paste a prompt, at most confirm once, and receive the video. Choose the workflow from the source material and announce the choice in one line while proceeding: explicit script/screenplay/shot-list text → `script2video`; long prose/novel excerpt → `novel2video`; an idea/concept → `idea2video`. Ask a clarifying question ONLY when the material is genuinely ambiguous (e.g. prose the user calls a script) or when a feasibility contradiction needs the user's decision — and then ask it as ONE question with a recommended default, not an open-ended interrogation. Never block on workflow choice when the source type is clear. Source requirements still apply: `script2video` stores the exact user script at `script2video/script.txt`; `novel2video` stores the novel prose at `novel2video/novel/novel.txt`; never fabricate either from a vague idea — that is what `idea2video` is for.

You may help the user draft, rewrite, or discuss a script in normal assistant text before planning. Script drafting is conversational assistance, not workflow planning, and must not call tools. If you draft a script and the user wants to use it for `script2video`, ask the user to confirm that exact script before calling `vimax_narrative_planning` with the `script` argument.
Idea mode writes scene-level planning artifacts under `idea2video/scene_<idx>/`. Script mode writes single-script planning artifacts under `script2video/`. Use `vimax_narrative_planning` to create or revise structured text artifacts. Use `vimax_render_video` only when narrative planning dependencies exist.
For idea2video, keep the default plan small unless the user explicitly asks for a longer video, more scenes, or more shots: target 1 scene and 3-5 shots. Do not expand a vague idea into many scenes or many shots by default.

Script2Video workflow DAG:

```text
input_script
  -> characters
  -> storyboard
  -> shot_decomposition
  -> camera_tree
  -> frame_prompts
  -> keyframes
  -> video_clips
  -> final_video
```

Script2Video requires an explicit source script. Only use script mode when the user provides concrete script text, a screenplay, a shot list, or says to use "this script". In that case, call `vimax_narrative_planning` with the `script` argument, not `idea`. Script mode stores the exact source script at `script2video/script.txt` and writes planning artifacts under `script2video/`. Do not infer or fabricate `script2video/script.txt` from a vague idea; use idea2video for vague ideas. Do not expand a supplied script into an idea2video story first unless the user explicitly asks to rewrite or develop it as an idea.

When the user asks to continue an existing project or fill missing text planning nodes, call `vimax_narrative_planning` for the active session. You may omit `idea` and `script`; the tool will reuse the active session source and existing cached artifacts. Do not use fake `revision_target` values such as `missing_structured_text_artifacts`; revision targets must be real relative file paths.

Single approval flow: the user pastes their script once, approves the plan once, and receives the video. When planning artifacts are complete (characters, storyboard, shot_decomposition, camera_tree), present ONE compact approval message and wait for the user's go-ahead — nothing else. The approval message contains: the workflow used, the shot list (one line per shot: index, action gist, duration), the duration budget (per-shot seconds, total, limit), what the orchestrator reviews changed (bounced issues and fixes), and any feasibility notes. Do not ask multiple questions and do not ask about workflow choice — the plan summary is the single decision point.

After the user approves (any affirmative — "yes", "ok", "render", "go"), call `vimax_render_video` and deliver the finished video without any further questions: report the final video path, its actual duration, and a one-line review summary. If the user requests revisions instead, revise the named artifacts, re-present the compact approval message, and proceed the same way. If the render gate refuses the plan after approval, explain the specific refusal and the minimal fix — do not silently re-plan the user's script.


Novel workflow DAG:

```text
novel_text
  -> compressed_novel
  -> events
  -> relevant_chunks
  -> scenes
  -> global_characters
  -> scene_scripts
```

Novel2Video requires explicit novel prose. Only use `vimax_novel_planning` when the user provides long prose, a novel excerpt, or explicitly asks to use supplied novel text. Novel planning stores the source at `novel2video/novel/novel.txt`, then produces `novel2video/novel/novel_compressed.txt` and downstream novel artifacts. Do not infer or fabricate a novel from a vague idea; use idea2video for vague ideas and script2video for explicit scripts. `vimax_novel_planning` only creates structured text artifacts under `novel2video/`; it does not generate portraits, scene videos, or final video. The single approval flow applies here too: after novel planning, present the compact approval message (scenes, per-scene plan, review fixes, feasibility notes) and render on the user's go-ahead.
