# Clip contract (per-clip script structure)

Every script ViMax writes (idea expansion, planning, enhancement, novel adaptation) is a sequence of CLIPS, each renderable as one continuous take. When reformatting a USER's script into this structure, carry EVERY section they wrote into the formatted script verbatim: character/reference locks, clothing locks, phone locks, voice continuity, off-camera rules, audio notes, screen guidance, negative constraints, end-card/post-production instructions, and their stated durations. Nothing from the user's script may be dropped, paraphrased, or summarized away - the formatting adds structure on top of their words, it never removes their words. Write each clip with these sections, in this order. When the user supplies a structured script, keep it verbatim — this contract governs scripts you author.

## Sections of one clip

1. **Header** — duration in seconds, aspect ratio/format, visual style and purpose (e.g. "10 seconds, vertical 9:16, photorealistic family-recorded UGC, mobile-game ad").
2. **Locks** — repeat them in every clip, never only the first:
   - Character identity: face, age, ethnicity, hair, glasses, jewelry — the details that make the person recognizable.
   - Wardrobe: garment, pattern, fabric, fit — "the same physical garment, not a similar replacement".
   - Environment: furniture, layout, light direction and quality.
   - Props: exact look + **orientation contract** (e.g. "black smartphone held horizontally in landscape, never vertical").
3. **Camera** — who films, height, distance, framing, movement texture (e.g. "family member filming at seated eye level, subtle handheld, imperfect home-video framing"). State what the camera never reveals (the filmer, a second person).
4. **Timecoded beats** — split the clip into 2-4 time windows (e.g. 0.0-2.5s / 2.5-5.8s). Each beat: one action, the quoted dialogue with its delivery tone ("playful confidence"), and **gesture-word synchronization** — gestures happen DURING the line they belong to ("she raises one finger as she says 'Now THIS'"), never after it.
5. **Voice lock** — one voice identity per speaker, repeated per clip: gender/age feel, pitch, accent, timbre, rhythm. "Excitement changes only expression and energy, never the underlying voice." Off-camera speakers: one unseen person behind the lens, same voice and acoustic distance every time, never shown, never a second visible person.
6. **Audio** — room ambience, dialogue dominant, music level underneath; no overlapping lines, no shouting, no robotic delivery.
7. **Screen guidance** — when a device shows an app/game: "keep on-screen content minimal and generic, no readable UI text or fake interface — the screen will be replaced with real footage in editing." Never invent detailed game mechanics.
8. **Negatives** — explicit ban list: no identity/wardrobe changes, no orientation flips, no extra or missing fingers, no new people entering, no model-generated text/logos/watermarks, no abrupt camera jumps.
9. **End-state handoff** — the exact final pose/expression of the clip and "ready for the next clip"; the next clip starts from that moment as if recording never stopped.
10. **Post-production** — overlays, captions, end cards, and real screen inserts are listed for the editor, never written into the generated visuals.

## Hard rules
- One clip = one continuous take, ≤ the model's maximum clip length (default 10s). Need more time? Split into two clips at a natural beat — never write one long clip.
- A prop never rotates mid-clip to face the camera. Show a screen over-the-shoulder, via a cut-in, or as a POV insert clip. A mid-clip rotation is a last resort only when the script demands it, written as its own explicit beat.
- End a clip on the last word or the held final pose — no trailing dead air.
