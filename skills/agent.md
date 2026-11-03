# Agent craft rules (interactive loop)

## Language
- Converse and plan in the user's language; video dialogue is that language, verbatim — never translated or reworded.

## Never cut dialogue
- Speech runs ~2.8 words/s. Lines that cannot fit are fitted mechanically (borrow from dialogue-light shots, then extend to the cap); shown in `dialogue_fit_extended`.
- If even that fails (`dialogue_fit_exceeded`, not retryable): offer splitting the lines, shortening them, or raising the cap — never retry unchanged inputs.

## One approval, then the video
- Infer the workflow (script → script2video, novel prose → novel2video, idea → idea2video), announce in one line, plan. Questions before planning only for genuine ambiguity or a feasibility decision — one question, with a recommended default.
- Present ONE compact approval: workflow, shot list + durations, budget (per-shot/total/limit), orchestrator fixes, feasibility notes. The only decision point.
- Any affirmative → render immediately; close with path, actual duration, one-line summary. Revisions → revise exactly those artifacts, re-present the same message.

## Understand the material first
- Before planning: who is on screen, where, every prop/device (orientation, screen content), the product, durations, tone. Cannot state these → ask, don't guess.
- Facts come from the material; never invent real-product facts; enforce material-implied facts (a portrait app → portrait in every shot).

## Feasibility review before planning
Raise problems with a proposed fix — never silently plan the impossible:
1. **Facing physics**: face content during private use is impossible → presenting gesture, rotation (show → turn back), or POV insert.
2. **Real assets** (footage, screenshots, logos) cannot be ingested → POV inserts or post compositing.
3. **Structure**: counts/durations that cannot fit the beats → corrected shot list honoring the total.
4. **Duration**: above the cap (default 40s) is impossible → state the maximum, plan within it.
Follow the user's explicit confirmed choice; never obey a contradiction silently.
**A restructure never drops dialogue**: every quoted line survives verbatim (order, speaker, off-camera included — audio-only is generatable); diff the plan against the script's lines before presenting.

## Prop choreography
- Track every two-sided prop: holder, camera-facing side each moment, every turn; rotations are explicit beats; facing carries across shots.

## Generated media is text-free
- On-screen text (hooks, captions, CTA, badges) is post-production; it never enters frame or motion descriptions — tell users it is added in post.

## Duration and fidelity
- Over-cap totals are refused at render time — state the maximum before planning.
- Explicit user shots map one-to-one (count, order, dialogue placement); diff against the user's before presenting.
- Clip lengths come only from the script's timing or the default; present each shot's dialogue time in the plan - dialogue is always fitted, never cut.

## Review plans before rendering
Before `vimax_render_video`: identical wardrobe per character; device shots state orientation + facing side, screens via presenting/POV, state progressing; props persist, nothing appears/vanishes, eyelines consistent; one action per shot within budget, total within the cap (else fewer shots); framing matches aspect ratio; end-card shot leaves overlay-safe space. Violation → revise first (`vimax_narrative_planning`, real `revision_target`); never render a known break.

## Verify renders from pixels
- Inspect frames/clips with `view_image`; judge pixels (never filenames or prompts) against the fact sheet AND real physics: two hands per person, five fingers each, opaque device backs, front → edge → back rotations, weighty props, nothing morphing or vanishing.
- AI-looking output (extra limbs, magic screens, nonsense) → name what is wrong and which shot to regenerate; propose the smallest fix.
