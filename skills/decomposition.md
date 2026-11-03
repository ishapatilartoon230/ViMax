# Frame decomposition rules

## First frame / last frame are frozen snapshots
- Each frame description is a pure static snapshot: composition, posture, light, prop states. No ongoing actions ("about to tap" is wrong; "thumb resting on the screen edge" is right).
- The last frame must reflect everything the motion description changes.

## Devices and screens survive decomposition
- When the shot description mentions a device, both frame descriptions must repeat its orientation, which side faces the camera, and the screen content state — do not let these details vanish between the storyboard and the frame prompts.
- POV screen shots: first/last frames describe the screen content filling the frame, the device orientation, and hand positions that do not occlude content.
- Motion descriptions refer to characters by visible features (never bare names) and separate camera movement from in-frame movement.

## Consistency in, consistency out
- Wardrobe, prop positions, and set details in the first frame must match the last frame except for what the motion explicitly changes.
- Preserve the target aspect ratio and shot size language from the shot description.

## Prop rotations survive decomposition
- When a shot turns a prop (phone shown to camera then rotated back to the user, book opened, card flipped, laptop closed), the first frame shows the **starting side** facing the camera, the motion describes **the turn itself** (who rotates it, which way), and the last frame shows the **ending side**.
- A prop rotation shot is never `variation_type: "small"`. The turn changes the most important object in the frame between first and last frame — the same camera framing does not make it small; use at least `medium` so a last-frame image anchors the end state (a last frame is what forces the video model to actually perform the rotation).
- The next shot's first frame continues from that ending facing — never silently flip a prop's side between shots.

## Chain state across shots
- Shot videos are concatenated in order: shot N ends near its last-frame state and shot N+1 starts at its first-frame state. So the **last frame of shot N and the first frame of shot N+1 must describe the same world state** — same prop positions, same device screen state, same wardrobe — differing only in camera framing. If shot N's motion ends with a changed state (card played, phone lowered), shot N+1's first frame must start from that changed state, not from a fresh one.
- Never reset world state between shots: what the previous shot established (dealt cards, opened app screen, moved objects) persists into the next shot's frames.
