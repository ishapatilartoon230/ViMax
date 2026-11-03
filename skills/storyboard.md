# Storyboard design rules

## Prop facing and rotation
- Every prop with distinct sides (phone, tablet, book, laptop, card, mirror) has a facing: state which side faces the camera at each shot's start; a shot's end-facing is the next shot's start-facing.
- Face content is seen only via a presenting gesture or a **POV insert** (the face fills the frame). Never angle a prop toward the camera during private use — add a POV insert or presenting beat instead.
- **Prefer a cut or POV insert over a mid-shot rotation** — it is the hardest thing a video model does.
- **Rotation is never planned** — even when a script demands a device turn, flag it to the user as a feasibility note instead. A held device keeps its orientation every shot.

## No baked-in text, ever
- Generated frames and clips contain **no text** — no captions, headlines, labels, CTA buttons, logos, watermarks — even when the script asks for them. On-screen text is a post-production overlay; if the script lists it, keep it out of `visual_desc` and frame descriptions entirely.
- A device/app/game screen shot fills `screen_guidance`: minimal generic content, no readable UI text — the real screen is inserted in post. Never invent detailed app/game mechanics.

## Preserve the user's shot structure exactly
- When the script explicitly lists shots/beats, produce exactly one storyboard shot per user beat — same order, same dialogue placement. Adding, dropping, merging, or reordering shots is a planning failure, even when the result "flows better".
- **Keep the user's words**: when the script is shootable, build `visual_desc` from their sentences verbatim — add only technical fields, never paraphrase.
- Only scenes/timings given? Derive shots: one action per shot, counts that fit the total.
- **Every dialogue line survives — off-camera speakers included.** A character heard but never seen keeps their lines in the matching shot's `audio_desc` exactly as written; off-camera dialogue is audio and fully generatable.

## Timing is honest
- Copy per-shot durations from the script into `duration_seconds`; null when unstated (default applies). Totals are the sum; stay within the cap; never invent durations.
- **No shot longer than the model's maximum clip length** (default 10s): one generation call cannot return more, so a bigger budget silently cuts the tail mid-line. Split the beat into two shots at a natural pause.
- Keep the shot count within the total cap (max_seconds ÷ per_shot_seconds); if the material wants more, merge beats into fewer shots.

## Design from the fact sheet
- Restate internally before listing shots: characters, wardrobe, setting, light, props (orientation, screen content), budgets — every shot consistent with all, one action per shot.

## Every visual_desc must be drawable
- Self-contained: shot size, angle, subject position in frame, facing direction, expression, and the state of every visible prop.
- **Devices**: state the orientation (portrait/landscape) and the grip — the character's face stays toward the camera; the phone is held low with its screen tilted toward her, she looks down at it. Screen content is shown via a POV insert shot (the screen fills the frame), never by turning the screen toward the camera.
- POV screen inserts: say what fills the frame (the screen), the orientation, and that fingers/bezels do not cover the content.
- Prop state is progressive: what earlier shots tapped/played stays that way later.

## Continuity across shots
- Characters keep wardrobe and screen position between adjacent shots unless shown changing; adjacent shots hand over world state (objects moved, screens changed, cards played).
- Eyelines match what characters look at; off-camera speakers are never visualized.
- Reuse an established camera position; add one only for a genuinely different size/angle/focus. Compose for the target aspect ratio (portrait frames compose vertically).
