# Frame and video prompt rules

## The prompt you build is the only thing the image/video model sees
- Include every fact-sheet detail that matters for this frame: character appearance and wardrobe, setting and light, prop states, and each prop's facing (which side faces the camera, orientation). Do not assume the model remembers other shots.
- **Props with sides** (phone, tablet, book, card, laptop): say which side faces the camera — "screen faces the camera showing <content>" when presenting, "back cover faces the camera while she looks down at it" during private use. Never leave the facing ambiguous.
- For motion prompts that rotate a prop: describe the turn ("she rotates the phone back toward herself; by the end the back cover faces the camera") so the video model animates the rotation instead of freezing the prop.
- Screen/UI inserts (POV): state that the screen fills the frame, its orientation, the UI style and colors from the fact sheet, and that fingers hold the bezel without covering content.

## Reference images
- Character portrait references define appearance: the prompt must agree with them (wardrobe, hair, accessories), not fight them.
- When a previous frame is used as continuity reference, describe what persists from it and what changes.

## Realism requirements in every prompt
- State anatomy expectations explicitly when hands interact with props: "her two hands", "his right hand (five fingers)" — the generator must render exactly two hands per person, never extra limbs.
- **Grip language**: never write "loosely holding", "rests near", or "beside the hand" for a held prop — the prompt states the grip: "grips the phone in her right hand, fingers curled around its edges, base resting on her palm". The first frame shows the prop IN HAND with visible contact; motions move from that contact.
- **Opaque device backs**: when the back of a device faces the camera, the prompt says the back cover is plain/opaque — no screen content. Screen content appears only when the front faces the camera.
- Rotations in motion prompts follow real physics: front → edge → back → edge → front; the side facing away is never visible mid-turn.
- Props have weight: held objects rest in hands, set-down objects stay put, nothing floats or morphs.
- Quoted dialogue inside prompts stays in the user's original language, verbatim.

## Style and frame shape
- Compose for the configured aspect ratio (portrait compositions center the subject vertically with headroom; landscape compositions use horizontal balance).
- Match the configured visual style; forbid overlays, captions, watermarks, and logos in generated frames when the pipeline adds them in post.
