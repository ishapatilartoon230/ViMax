import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pipelines.script2video_pipeline import Script2VideoPipeline
from utils.dialogue_fit import dialogue_fit_error_message, dialogue_fit_report, rebalance_over_budget_shots
from utils.render_geometry import clamp_max_clip_seconds, normalize_render_geometry


def _pipeline(working_dir: str, geometry: dict | None = None) -> Script2VideoPipeline:
    return Script2VideoPipeline(
        chat_model=object(), image_generator=object(), video_generator=object(),
        working_dir=working_dir,
        render_geometry=geometry,
    )


class MaxClipSecondsTests(unittest.TestCase):
    def test_default_is_ten(self):
        geometry = normalize_render_geometry({})
        self.assertEqual(geometry["max_clip_seconds"], 10.0)

    def test_clamp_falls_back_to_default(self):
        for value in ("abc", "0", "-3", None, ""):
            with self.subTest(value=value):
                self.assertEqual(clamp_max_clip_seconds(value), 10.0)

    def test_clamp_upper_bound(self):
        self.assertEqual(clamp_max_clip_seconds("120"), 60.0)
        self.assertEqual(clamp_max_clip_seconds("4"), 4.0)

    def test_env_override(self):
        with patch.dict(os.environ, {"VIMAX_MAX_CLIP_SECONDS": "6"}):
            geometry = normalize_render_geometry({})
        self.assertEqual(geometry["max_clip_seconds"], 6.0)

    def test_geometry_dict_overrides_default(self):
        geometry = normalize_render_geometry({"max_clip_seconds": 8})
        self.assertEqual(geometry["max_clip_seconds"], 8.0)


class EffectiveShotSecondsCapTests(unittest.TestCase):
    def test_shot_duration_is_capped_at_max_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp, {"max_clip_seconds": 10})
            shot = SimpleNamespace(idx=0, duration_seconds=14.0)
            self.assertEqual(pipeline._effective_shot_seconds(shot), 10.0)

    def test_short_shot_keeps_its_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp, {"max_clip_seconds": 10})
            shot = SimpleNamespace(idx=0, duration_seconds=6.0)
            self.assertEqual(pipeline._effective_shot_seconds(shot), 6.0)

    def test_default_seconds_also_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp, {"video_seconds": 12, "max_clip_seconds": 10})
            shot = SimpleNamespace(idx=0, duration_seconds=None)
            self.assertEqual(pipeline._effective_shot_seconds(shot), 10.0)


class DialogueFitSplitTests(unittest.TestCase):
    def test_split_recommendation_when_recommended_exceeds_clip_cap(self):
        # 4 lines, 36 words -> ~13.76s speech at 2.8 w/s; beyond the 11.5s
        # tolerance band of a 10s budget and the 10s clip cap -> must split.
        shot = {
            "audio_desc": '[Speaker] A: "I been playing this game for fifty years tonight!"\n'
                          '[Speaker] B (off-camera): "Well, Grandma, you try it on your phone right now!"\n'
                          '[Speaker] A: "Hold up, y\'all got that new Downtown mode on here?"\n'
                          '[Speaker] B (off-camera): "Yes, ma\'am, and you will love it!"',
            "duration_seconds": 10.0,
        }
        report = dialogue_fit_report([shot], 8.0)
        self.assertFalse(report["all_fit"])
        message = dialogue_fit_error_message(report, max_clip_seconds=10.0)
        self.assertIn("split this shot", message)
        self.assertIn("at most 10s", message)

    def test_extend_recommendation_when_under_clip_cap(self):
        shot = {"audio_desc": '[Speaker] A: "One two three four five six seven eight nine ten eleven twelve thirteen"', "duration_seconds": 2.0}
        report = dialogue_fit_report([shot], 8.0)
        message = dialogue_fit_error_message(report, max_clip_seconds=10.0)
        self.assertIn("extend this shot", message)

    def test_rebalance_never_exceeds_per_shot_cap(self):
        over_shot = {
            "audio_desc": '[Speaker] A: "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"',
            "duration_seconds": 4.0,
        }
        donor = {"audio_desc": "", "duration_seconds": 8.0}
        adjustments = rebalance_over_budget_shots([over_shot, donor], 8.0, 40.0, per_shot_cap=10.0)
        for item in adjustments:
            if item.get("source") != "donor":
                self.assertLessEqual(item["to_seconds"], 10.0)


GRANDMA_CHARACTERS = [
    {
        "idx": 0,
        "identifier_in_scene": "Grandma Mae",
        "is_visible": True,
        "static_features": "70-year-old African-American woman, short salt-and-pepper curls, rectangular reading glasses, gold hoop earrings",
        "dynamic_features": "orange patterned blouse",
        "voice_features": "mature feminine voice, medium-low pitch, warm and slightly husky, calm rhythm",
    }
]


def _shot(**overrides):
    base = dict(
        idx=0,
        is_last=False,
        motion_desc="She taps the horizontal phone with her right thumb.",
        audio_desc='[Speaker] Grandma Mae: "I been playing for fifty years!"\n[Speaker] Family member (off-camera): "Yes, ma\'am!"',
        ff_vis_char_idxs=[0],
        lf_vis_char_idxs=[0],
        screen_guidance="green card table with simple card shapes, no readable text",
        variation_type="small",
        duration_seconds=10.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class VideoPromptContractTests(unittest.TestCase):
    def _pipeline_with_characters(self, tmp: str) -> Script2VideoPipeline:
        pipeline = _pipeline(tmp)
        with open(os.path.join(tmp, "characters.json"), "w", encoding="utf-8") as f:
            json.dump(GRANDMA_CHARACTERS, f)
        return pipeline

    def test_flow_style_sections_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline_with_characters(tmp)
            prompt = pipeline._build_video_prompt(_shot(), 10.0, chained=False)
        self.assertIn("[Identity lock]", prompt)
        self.assertIn("Grandma Mae", prompt)
        self.assertIn("salt-and-pepper curls", prompt)
        self.assertIn("orange patterned blouse", prompt)
        self.assertIn("[Voice lock]", prompt)
        self.assertIn("mature feminine voice", prompt)
        self.assertIn("never the underlying voice", prompt)
        self.assertIn("[Off-camera speaker]", prompt)
        self.assertIn("[Screen guidance]", prompt)
        self.assertIn("green card table", prompt)
        self.assertIn("ready for the next clip", prompt)
        self.assertIn("No text", prompt)

    def test_last_shot_has_no_next_clip_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline_with_characters(tmp)
            prompt = pipeline._build_video_prompt(_shot(is_last=True), 10.0, chained=False)
        self.assertNotIn("ready for the next clip", prompt)

    def test_chained_sentence_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline_with_characters(tmp)
            prompt = pipeline._build_video_prompt(_shot(), 10.0, chained=True)
        self.assertIn("continues directly from the previous shot", prompt)

    def test_no_characters_file_means_no_locks(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            prompt = pipeline._build_video_prompt(_shot(), 10.0, chained=False)
        self.assertNotIn("[Identity lock]", prompt)
        self.assertNotIn("[Voice lock]", prompt)

    def test_no_off_camera_line_no_off_camera_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline_with_characters(tmp)
            prompt = pipeline._build_video_prompt(
                _shot(audio_desc='[Speaker] Grandma Mae: "I been playing for fifty years!"'), 10.0, chained=False
            )
        self.assertNotIn("[Off-camera speaker]", prompt)

    def test_no_screen_guidance_no_screen_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline_with_characters(tmp)
            prompt = pipeline._build_video_prompt(_shot(screen_guidance=None), 10.0, chained=False)
        self.assertNotIn("[Screen guidance]", prompt)

    def test_missing_voice_features_gets_a_derived_voice_lock(self):
        # p4: voice_features was null for BOTH characters -> no voice lock at all ->
        # the off-camera family line came out in Grandma's voice. The lock must
        # fall back to age/gender cues from the character's own description.
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            without_voice = [dict(GRANDMA_CHARACTERS[0], voice_features=None)]
            with open(os.path.join(tmp, "characters.json"), "w", encoding="utf-8") as f:
                json.dump(without_voice, f)
            prompt = pipeline._build_video_prompt(_shot(), 10.0, chained=False)
        self.assertIn("[Identity lock]", prompt)
        self.assertIn("[Voice lock]", prompt)
        self.assertIn("feminine voice", prompt)
        self.assertIn("70 years old", prompt)

    def test_dict_shots_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline_with_characters(tmp)
            prompt = pipeline._build_video_prompt(_shot().__dict__, 10.0, chained=False)
        self.assertIn("[Identity lock]", prompt)
        self.assertIn("[Screen guidance]", prompt)


class ClipShortEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_short_clip_emits_clip_short_progress(self):
        events = []

        class FakeVideoOutput:
            def save(self, path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(b"fake-video")

        class FakeVideoGenerator:
            async def generate_single_video(self, **kwargs):
                return FakeVideoOutput()

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp, {"max_clip_seconds": 10})
            pipeline.video_generator = FakeVideoGenerator()
            pipeline.frame_events[0] = {"first_frame": asyncio.Event()}
            pipeline.frame_events[0]["first_frame"].set()
            shot = _shot(duration_seconds=14.0)  # capped to a 10s budget at request time
            with patch.object(pipeline, "_ensure_video_end_frame", return_value=None), \
                 patch.object(pipeline, "_probe_duration", return_value=8.0), \
                 patch("pipelines.script2video_pipeline._trim_video_to_seconds", return_value=False):
                await pipeline.generate_video_for_single_shot(
                    shot_description=shot,
                    progress=lambda *args, **meta: events.append((args[0], args[2] if len(args) > 2 else {})),
                )
        stages = [stage for stage, _ in events]
        self.assertIn("clip_short", stages)
        short = next(meta for stage, meta in events if stage == "clip_short")
        self.assertEqual(short["requested_seconds"], 10.0)
        self.assertEqual(short["delivered_seconds"], 8.0)
        self.assertEqual(short["short_by_seconds"], 2.0)

    async def test_full_length_clip_emits_nothing(self):
        events = []

        class FakeVideoOutput:
            def save(self, path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(b"fake-video")

        class FakeVideoGenerator:
            async def generate_single_video(self, **kwargs):
                return FakeVideoOutput()

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            pipeline.video_generator = FakeVideoGenerator()
            pipeline.frame_events[0] = {"first_frame": asyncio.Event()}
            pipeline.frame_events[0]["first_frame"].set()
            with patch.object(pipeline, "_ensure_video_end_frame", return_value=None), \
                 patch.object(pipeline, "_probe_duration", return_value=9.8), \
                 patch("pipelines.script2video_pipeline._trim_video_to_seconds", return_value=False):
                await pipeline.generate_video_for_single_shot(
                    shot_description=_shot(),
                    progress=lambda *args, **meta: events.append((args[0], args[2] if len(args) > 2 else {})),
                )
        stages = [stage for stage, _ in events]
        self.assertNotIn("clip_short", stages)


if __name__ == "__main__":
    unittest.main()


class ThreeFixTests(unittest.TestCase):
    """The user's 3 project1 problems: no phone rotation, Grandma consistency,
    and the off-camera family member's voice."""

    def test_video_prompt_bans_rotation_for_fixed_devices(self):
        from pipelines.script2video_pipeline import REALISM_VIDEO_SUFFIX
        self.assertIn("Devices never rotate, flip, or change grip", REALISM_VIDEO_SUFFIX)
        self.assertIn("keeps its orientation for the entire clip", REALISM_VIDEO_SUFFIX)

    def test_image_prompts_carry_identity_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            character = GRANDMA_CHARACTERS[0]
            from interfaces import CharacterInScene
            model_character = CharacterInScene.model_validate(character)
            locks = pipeline._frame_prompt_locks([model_character])
        self.assertIn("[Identity lock]", locks)
        self.assertIn("Grandma Mae", locks)
        self.assertIn("orange patterned blouse", locks)
        self.assertIn("reference portraits", locks)
        self.assertEqual(pipeline._frame_prompt_locks([]), "")


    def test_screen_guidance_locks_device_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            prompt = pipeline._build_video_prompt(_shot(), 10.0, chained=False)
        self.assertIn("keeps its stated orientation for the ENTIRE clip", prompt)
        self.assertIn("never rotates, flips, or turns", prompt)
        self.assertIn("FACE stays visible to the camera", prompt)

    def test_off_camera_voice_must_be_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            prompt = pipeline._build_video_prompt(_shot(), 10.0, chained=False)
        self.assertIn("clearly DISTINCT", prompt)
        self.assertIn("Never deliver an off-camera line", prompt)
        self.assertIn("on-screen character's voice", prompt)

    def test_storyboard_skill_forbids_rotation_for_fixed_orientation(self):
        repo_root = Path(__file__).resolve().parent.parent
        text = (repo_root / "skills" / "storyboard.md").read_text(encoding="utf-8")
        self.assertIn("Rotation is never planned", text)
        self.assertIn("flag it to the user as a feasibility note", text)


class CopyChainTests(unittest.IsolatedAsyncioTestCase):
    """The video loop copies the previous clip's end frame over the next
    shot's keyframe — EVERY shot, camera changes included. The chain is
    literal pixels."""

    def _pipeline(self, tmp):
        import asyncio
        pipeline = _pipeline(tmp)
        pipeline.frame_events = {i: {"first_frame": asyncio.Event()} for i in range(3)}
        return pipeline

    async def test_end_frame_copied_to_next_same_camera_keyframe(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            for i in range(2):
                (Path(tmp) / "shots" / str(i)).mkdir(parents=True, exist_ok=True)
            end_frame = Path(tmp) / "shots" / "0" / "video_last_frame.png"
            end_frame.write_bytes(b"end-pixels")
            shots = [SimpleNamespace(idx=0, cam_idx=0), SimpleNamespace(idx=1, cam_idx=0)]

            async def fake_video(shot_description, progress=None, previous_end_frame_path=None):
                pass

            pipeline.generate_video_for_single_shot = fake_video
            pipeline._ensure_video_end_frame = lambda video_path, shot_idx: str(end_frame) if shot_idx == 0 else None
            await pipeline._generate_all_shot_videos(shots)
            copied = Path(tmp) / "shots" / "1" / "first_frame.png"
            self.assertTrue(copied.exists())
            self.assertEqual(copied.read_bytes(), b"end-pixels")
            self.assertTrue(pipeline.frame_events[1]["first_frame"].is_set())

    async def test_end_frame_copied_across_camera_change(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            for i in range(2):
                (Path(tmp) / "shots" / str(i)).mkdir(parents=True, exist_ok=True)
            end_frame = Path(tmp) / "shots" / "0" / "video_last_frame.png"
            end_frame.write_bytes(b"end-pixels")
            shots = [SimpleNamespace(idx=0, cam_idx=0), SimpleNamespace(idx=1, cam_idx=1)]  # camera change

            async def fake_video(shot_description, progress=None, previous_end_frame_path=None):
                pass

            pipeline.generate_video_for_single_shot = fake_video
            pipeline._ensure_video_end_frame = lambda video_path, shot_idx: str(end_frame) if shot_idx == 0 else None
            await pipeline._generate_all_shot_videos(shots)
            copied = Path(tmp) / "shots" / "1" / "first_frame.png"
            self.assertTrue(copied.exists(), "camera changes must chain too — no fresh keyframe mid-video")
            self.assertEqual(copied.read_bytes(), b"end-pixels")
            self.assertTrue(pipeline.frame_events[1]["first_frame"].is_set())


class RotationNeverTests(unittest.TestCase):
    """Rotation is never prompted, and the grip rule is explicit."""

    def test_video_prompt_has_no_rotation_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            prompt = pipeline._build_video_prompt(_shot(), 10.0, chained=False)
        self.assertNotIn("Rotation contract", prompt)

    def test_video_suffix_says_devices_never_rotate(self):
        from pipelines.script2video_pipeline import REALISM_VIDEO_SUFFIX
        self.assertIn("Devices never rotate, flip, or change grip", REALISM_VIDEO_SUFFIX)

    def test_screen_guidance_says_screen_faces_the_character(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            prompt = pipeline._build_video_prompt(_shot(), 10.0, chained=False)
        self.assertIn("FACE stays visible to the camera", prompt)
        self.assertIn("screen tilted toward her", prompt)


class DialogueWindowTests(unittest.TestCase):
    """Tight shots get their real speech time as the dialogue window, so the
    model is never told to finish lines before they can be spoken."""

    def test_tight_shot_window_covers_speech_estimate(self):
        # 9 words ≈ 3.51s speech in a 3.5s shot: window must exceed the 85% default
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            window = pipeline._dialogue_completion_window('[Speaker] A: "one two three four five six seven eight nine"', 3.5)
        self.assertGreater(window, 3.5 * 0.85)
        self.assertGreaterEqual(window, 3.21)  # covers the 3.21s speech estimate

    def test_comfortable_shot_keeps_the_85_percent_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            window = pipeline._dialogue_completion_window('[Speaker] A: "hello there"', 10.0)
        self.assertAlmostEqual(window, 10.0 * 0.85, places=2)

    def test_budget_payload_shows_dialogue_time_per_shot(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            shots = [
                {"idx": 0, "audio_desc": 'A: "hello there friend"'},
                {"idx": 1, "audio_desc": ""},
            ]
            budget = pipeline._duration_budget(shots)
        self.assertEqual(budget["dialogue_needs_seconds"][0], 1.07)
        self.assertEqual(budget["dialogue_needs_seconds"][1], 0.0)


class FrameGripLockTests(unittest.TestCase):
    """Image prompts carry the phone-grip rule for device shots: the screen
    faces the character (she looks down at it), never the camera."""

    def test_device_grip_in_image_locks(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            from interfaces import CharacterInScene
            locks = pipeline._frame_prompt_locks(
                [CharacterInScene.model_validate(GRANDMA_CHARACTERS[0])],
                screen_guidance="green card table, no readable text",
            )
        self.assertIn("[Device grip]", locks)
        self.assertIn("FACE stays visible to the camera", locks)
        self.assertIn("screen tilted toward her", locks)
        self.assertIn("app UI is NOT visible", locks)

    def test_no_device_no_grip_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            from interfaces import CharacterInScene
            locks = pipeline._frame_prompt_locks(
                [CharacterInScene.model_validate(GRANDMA_CHARACTERS[0])], screen_guidance=None)
        self.assertNotIn("[Device grip]", locks)


class NoDialogueTests(unittest.TestCase):
    """Shots without dialogue explicitly forbid hallucinated speech - the
    model invented Grandma's closing words in p3's last two shots."""

    def test_no_dialogue_shot_forbids_speech(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            prompt = pipeline._build_video_prompt(
                _shot(audio_desc="[Ambient] warm living room tone"), 10.0, chained=False)
        self.assertIn("[No dialogue]", prompt)
        self.assertIn("does NOT speak", prompt)
        self.assertIn("no mumbling", prompt)

    def test_dialogue_shot_has_no_no_dialogue_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            prompt = pipeline._build_video_prompt(_shot(), 10.0, chained=False)
        self.assertNotIn("[No dialogue]", prompt)

    def test_off_camera_voice_separation_is_strong(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            prompt = pipeline._build_video_prompt(_shot(), 10.0, chained=False)
        self.assertIn("different pitch, different timbre", prompt)
        self.assertIn("mouth stays closed", prompt)


class ChainFallbackTests(unittest.IsolatedAsyncioTestCase):
    """If the previous clip's end frame is unavailable, the shot's keyframe is
    GENERATED as a fallback - never a missing first_frame.png reference.
    Applies to EVERY shot, camera changes included."""

    async def test_broken_chain_generates_the_keyframe(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmp:
            import asyncio
            pipeline = _pipeline(tmp)
            pipeline.frame_events = {i: {"first_frame": asyncio.Event()} for i in range(2)}
            shots = [
                SimpleNamespace(idx=0, cam_idx=0, ff_desc="start", ff_vis_char_idxs=[], screen_guidance=None),
                SimpleNamespace(idx=1, cam_idx=0, ff_desc="next start", ff_vis_char_idxs=[], screen_guidance=None),
            ]
            generated = []

            async def fake_video(shot_description, progress=None, previous_end_frame_path=None):
                pass

            async def fake_frame(shot_idx, frame_type, first_shot_ff_path_and_text_pair, frame_desc, visible_characters, character_portraits_registry, progress=None, previous_frame_path_and_text_pair=None, screen_guidance=None):
                generated.append((shot_idx, frame_type, frame_desc))
                (Path(tmp) / "shots" / str(shot_idx)).mkdir(parents=True, exist_ok=True)
                (Path(tmp) / "shots" / str(shot_idx) / "first_frame.png").write_bytes(b"gen")

            pipeline.generate_video_for_single_shot = fake_video
            pipeline.generate_frame_for_single_shot = fake_frame
            pipeline._ensure_video_end_frame = lambda video_path, shot_idx: None  # chain broken
            await pipeline._generate_all_shot_videos(shots)
            self.assertIn((1, "first_frame", "next start"), generated)  # fallback fired for shot 1

    async def test_broken_chain_generates_keyframe_across_camera_change(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmp:
            import asyncio
            pipeline = _pipeline(tmp)
            pipeline.frame_events = {i: {"first_frame": asyncio.Event()} for i in range(2)}
            shots = [
                SimpleNamespace(idx=0, cam_idx=0, ff_desc="start", ff_vis_char_idxs=[], screen_guidance=None),
                SimpleNamespace(idx=1, cam_idx=7, ff_desc="new angle start", ff_vis_char_idxs=[], screen_guidance=None),
            ]
            generated = []

            async def fake_video(shot_description, progress=None, previous_end_frame_path=None):
                pass

            async def fake_frame(shot_idx, frame_type, first_shot_ff_path_and_text_pair, frame_desc, visible_characters, character_portraits_registry, progress=None, previous_frame_path_and_text_pair=None, screen_guidance=None):
                generated.append((shot_idx, frame_type, frame_desc, first_shot_ff_path_and_text_pair[0]))
                (Path(tmp) / "shots" / str(shot_idx)).mkdir(parents=True, exist_ok=True)
                (Path(tmp) / "shots" / str(shot_idx) / "first_frame.png").write_bytes(b"gen")

            pipeline.generate_video_for_single_shot = fake_video
            pipeline.generate_frame_for_single_shot = fake_frame
            pipeline._ensure_video_end_frame = lambda video_path, shot_idx: None  # chain broken
            await pipeline._generate_all_shot_videos(shots)
            self.assertIn((1, "first_frame", "new angle start", os.path.join(tmp, "shots", "0", "first_frame.png")), generated)


class ChainColorAndVoiceTests(unittest.TestCase):
    """p4's three model-behavior bugs: color wash-out down the chain, the
    off-camera line spoken in Grandma's voice (no voice lock at all), and the
    silently trimmed over-delivered clip that cut her line mid-sentence."""

    def test_chained_prompt_locks_the_color_grade(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            prompt = pipeline._build_video_prompt(_shot(), 10.0, chained=True)
        self.assertIn("[Color lock]", prompt)
        self.assertIn("exact color grade", prompt)
        self.assertIn("never re-grade, wash out, soften, darken", prompt)
        self.assertIn("identical to the reference frame", prompt)

    def test_unchained_prompt_has_no_color_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            prompt = pipeline._build_video_prompt(_shot(), 10.0, chained=False)
        self.assertNotIn("[Color lock]", prompt)

    def test_multi_speaker_clip_gets_back_to_back_pacing(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            prompt = pipeline._build_video_prompt(_shot(), 4.0, chained=False)
        self.assertIn("[Pacing]", prompt)
        self.assertIn("2 different speakers", prompt)
        self.assertIn("back-to-back", prompt)

    def test_single_speaker_clip_has_no_pacing_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            prompt = pipeline._build_video_prompt(
                _shot(audio_desc='[Speaker] Grandma Mae: "I been playing for fifty years!"'), 4.0, chained=False
            )
        self.assertNotIn("[Pacing]", prompt)

    def test_fallback_voice_needs_no_cues_to_stay_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            from interfaces import CharacterInScene
            fallback = pipeline._fallback_voice_features(
                CharacterInScene.model_validate({"idx": 0, "identifier_in_scene": "Neighbor", "is_visible": True, "static_features": "a person", "voice_features": None})
            )
        self.assertIn("one consistent natural voice", fallback)

    def test_distinct_speakers_parses_audio_desc(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
        audio = (
            '[Speaker] Family (Off-Camera) (Warm, encouraging): "Try it on your phone!"\n'
            '[Speaker] Grandma Mae (Curious, intrigued): "Hold up… y\'all got Downtown on here?"'
        )
        self.assertEqual(pipeline._distinct_speakers(audio), ["Family", "Grandma Mae"])
        self.assertEqual(pipeline._distinct_speakers("[Ambient] room tone"), [])


class ClipOverEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_over_delivered_clip_emits_clip_over(self):
        events = []

        class FakeVideoOutput:
            def save(self, path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(b"fake-video")

        class FakeVideoGenerator:
            async def generate_single_video(self, **kwargs):
                return FakeVideoOutput()

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            pipeline.video_generator = FakeVideoGenerator()
            pipeline.frame_events[0] = {"first_frame": asyncio.Event()}
            pipeline.frame_events[0]["first_frame"].set()
            shot = _shot(duration_seconds=4.0)  # model delivered 6.04s in p4 -> line cut
            with patch.object(pipeline, "_ensure_video_end_frame", return_value=None), \
                 patch.object(pipeline, "_probe_duration", return_value=6.04), \
                 patch("pipelines.script2video_pipeline._trim_video_to_seconds", return_value=True):
                await pipeline.generate_video_for_single_shot(
                    shot_description=shot,
                    progress=lambda *args, **meta: events.append((args[0], args[2] if len(args) > 2 else {})),
                )
        stages = [stage for stage, _ in events]
        self.assertIn("clip_over", stages)
        over = next(meta for stage, meta in events if stage == "clip_over")
        self.assertEqual(over["requested_seconds"], 4.0)
        self.assertEqual(over["delivered_seconds"], 6.04)
        self.assertEqual(over["over_by_seconds"], 2.04)

    async def test_on_budget_clip_emits_nothing(self):
        events = []

        class FakeVideoOutput:
            def save(self, path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(b"fake-video")

        class FakeVideoGenerator:
            async def generate_single_video(self, **kwargs):
                return FakeVideoOutput()

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            pipeline.video_generator = FakeVideoGenerator()
            pipeline.frame_events[0] = {"first_frame": asyncio.Event()}
            pipeline.frame_events[0]["first_frame"].set()
            with patch.object(pipeline, "_ensure_video_end_frame", return_value=None), \
                 patch.object(pipeline, "_probe_duration", return_value=4.1), \
                 patch("pipelines.script2video_pipeline._trim_video_to_seconds", return_value=False):
                await pipeline.generate_video_for_single_shot(
                    shot_description=_shot(duration_seconds=4.0),
                    progress=lambda *args, **meta: events.append((args[0], args[2] if len(args) > 2 else {})),
                )
        stages = [stage for stage, _ in events]
        self.assertNotIn("clip_over", stages)
        self.assertNotIn("clip_short", stages)
