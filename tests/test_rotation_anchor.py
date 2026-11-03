"""Rotation shots must be anchored end-to-end.

Regression guard for the observed render failure: a shot planned as
"screen faces camera, then the phone rotates back so the rear casing faces
the camera" was classified variation_type 'small', so no last-frame image
was generated — anchored only to the lit-screen first frame, the video model
never performed the rotation and the screen stayed facing the camera for the
whole clip.
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from interfaces.shot_description import ShotBriefDescription, ShotDescription
from pipelines.script2video_pipeline import Script2VideoPipeline, _requires_last_frame_anchor


ROTATION_SHOT = {
    "idx": 1,
    "is_last": True,
    "cam_idx": 0,
    "visual_desc": "She displays the phone with its screen facing the camera, then rotates it back toward herself so the rear casing faces the camera.",
    "variation_type": "small",
    "variation_reason": "Consistent medium close-up framing throughout.",
    "ff_desc": "Both hands hold the landscape phone with the bright screen facing the camera showing the card game.",
    "ff_vis_char_idxs": [0],
    "lf_desc": "Her left hand holds the landscape phone with its black rear casing facing outward toward the camera.",
    "lf_vis_char_idxs": [0],
    "motion_desc": 'She rotates the smartphone inward so its screen faces her and the rear casing faces the camera, then points at the lens and says: "TAKE THAT, PARTNER!"',
    "audio_desc": '[Speaker] Grandma: "TAKE THAT, PARTNER!"',
    "duration_seconds": 10,
}

PLAIN_SHOT = dict(
    ROTATION_SHOT,
    visual_desc="She speaks to the camera, expressions changing.",
    ff_desc="She sits holding nothing, smiling at the camera.",
    lf_desc="She sits laughing at the camera.",
    motion_desc='She smiles and says: "THIS is my game!"',
)


class RotationDetectionTests(unittest.TestCase):
    def test_rotation_language_requires_last_frame_anchor(self):
        self.assertTrue(_requires_last_frame_anchor(ShotDescription.model_validate(ROTATION_SHOT)))
        self.assertTrue(_requires_last_frame_anchor(ROTATION_SHOT))  # plain dicts too

    def test_plain_shot_does_not(self):
        self.assertFalse(_requires_last_frame_anchor(ShotDescription.model_validate(PLAIN_SHOT)))


class RotationPromotionTests(unittest.IsolatedAsyncioTestCase):
    async def test_small_rotation_shot_is_promoted_and_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline(
                chat_model=object(), image_generator=object(), video_generator=object(),
                working_dir=tmp, review_config={"enabled": False},
            )
            brief = ShotBriefDescription.model_validate(
                {"idx": 1, "is_last": True, "cam_idx": 0, "visual_desc": "rotates the phone", "audio_desc": "", "duration_seconds": 10}
            )
            pipeline.shot_desc_events[brief.idx] = asyncio.Event()  # design_storyboard registers this in production
            pipeline.storyboard_artist.decompose_visual_description = AsyncMock(
                return_value=ShotDescription.model_validate(ROTATION_SHOT)
            )
            shot = await pipeline.decompose_visual_description_for_single_shot_brief_description(brief, [], quiet=True)
            self.assertEqual(shot.variation_type, "medium")
            self.assertIn("Mechanical override", shot.variation_reason)
            # frame events must include the last frame so it actually generates
            self.assertIn("last_frame", pipeline.frame_events[1])
            persisted = json.loads((Path(tmp) / "shots" / "1" / "shot_description.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["variation_type"], "medium")

    async def test_plain_small_shot_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline(
                chat_model=object(), image_generator=object(), video_generator=object(),
                working_dir=tmp, review_config={"enabled": False},
            )
            brief = ShotBriefDescription.model_validate(
                {"idx": 0, "is_last": True, "cam_idx": 0, "visual_desc": "speaks", "audio_desc": "", "duration_seconds": 10}
            )
            pipeline.shot_desc_events[brief.idx] = asyncio.Event()  # design_storyboard registers this in production
            pipeline.storyboard_artist.decompose_visual_description = AsyncMock(
                return_value=ShotDescription.model_validate(PLAIN_SHOT)
            )
            shot = await pipeline.decompose_visual_description_for_single_shot_brief_description(brief, [], quiet=True)
            self.assertEqual(shot.variation_type, "small")
            self.assertNotIn("last_frame", pipeline.frame_events[0])


class VideoPromptRotationContractTests(unittest.TestCase):
    def _pipeline(self, tmp):
        return Script2VideoPipeline(
            chat_model=object(), image_generator=object(), video_generator=object(),
            working_dir=tmp, review_config={"enabled": False},
        )

    def test_rotation_shot_prompt_pins_the_end_facing(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = self._pipeline(tmp)._build_video_prompt(ShotDescription.model_validate(ROTATION_SHOT), 10.0, chained=False)
        # Rotation is never prompted - even a rotation-language shot gets no turn instruction.
        self.assertNotIn("Rotation contract", prompt)
        self.assertIn("Devices never rotate", prompt)

    def test_plain_shot_prompt_has_no_rotation_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = self._pipeline(tmp)._build_video_prompt(ShotDescription.model_validate(PLAIN_SHOT), 10.0, chained=False)
        self.assertNotIn("Rotation contract", prompt)


if __name__ == "__main__":
    unittest.main()
