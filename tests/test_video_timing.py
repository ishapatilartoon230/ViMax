import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pipelines.script2video_pipeline import Script2VideoPipeline, _trim_video_to_seconds
from utils.render_geometry import clamp_max_total_video_seconds, clamp_video_seconds


class ClampVideoSecondsTests(unittest.TestCase):
    def test_valid_values(self):
        self.assertEqual(clamp_video_seconds("3"), 3.0)
        self.assertEqual(clamp_video_seconds(4.5), 4.5)
        self.assertEqual(clamp_video_seconds(None), 8.0)
        self.assertEqual(clamp_video_seconds(""), 8.0)

    def test_invalid_values_fall_back_to_default(self):
        for value in ("abc", "0", "-2", 0, None):
            with self.subTest(value=value):
                self.assertEqual(clamp_video_seconds(value), 8.0)

    def test_upper_bound(self):
        self.assertEqual(clamp_video_seconds("120"), 60.0)

    def test_max_total_clamp(self):
        self.assertEqual(clamp_max_total_video_seconds("60"), 60.0)
        self.assertEqual(clamp_max_total_video_seconds("abc"), 40.0)
        self.assertEqual(clamp_max_total_video_seconds("-5"), 40.0)
        self.assertEqual(clamp_max_total_video_seconds("9999"), 600.0)


class DurationBudgetTests(unittest.TestCase):
    def _pipeline(self, seconds: float, maximum: float) -> Script2VideoPipeline:
        return Script2VideoPipeline(
            chat_model=object(), image_generator=object(), video_generator=object(),
            working_dir=tempfile.mkdtemp(),
            render_geometry={"video_seconds": seconds, "max_total_video_seconds": maximum},
        )

    def test_budget_math_with_default_seconds(self):
        pipeline = self._pipeline(3.0, 40.0)
        shots = [{"idx": i} for i in range(13)]  # no duration_seconds -> default 3s
        budget = pipeline._duration_budget(shots)
        self.assertEqual(budget["total_seconds"], 39.0)
        self.assertTrue(budget["within_limit"])

    def test_budget_sums_per_shot_durations_from_script(self):
        pipeline = self._pipeline(3.0, 40.0)
        durations = [3.0, 4.0, 4.0, 3.0, 3.0, 3.0]  # the 6-scene 20s ad
        shots = [{"idx": i, "duration_seconds": d} for i, d in enumerate(durations)]
        budget = pipeline._duration_budget(shots)
        self.assertEqual(budget["shots"], 6)
        self.assertEqual(budget["total_seconds"], 20.0)
        self.assertTrue(budget["within_limit"])

    def test_over_budget_plan_is_rejected(self):
        pipeline = self._pipeline(3.0, 40.0)
        shots = [{"idx": i} for i in range(14)]  # 14 x 3s = 42s > 40s
        with self.assertRaises(ValueError) as ctx:
            pipeline._validate_total_duration(shots)
        self.assertIn("42s", str(ctx.exception))

    def test_over_budget_with_mixed_durations_is_rejected(self):
        pipeline = self._pipeline(5.0, 40.0)
        shots = [{"idx": i, "duration_seconds": 10.0} for i in range(4)]  # 10+10+10+10=40 ok
        self.assertTrue(pipeline._duration_budget(shots)["within_limit"])
        shots.append({"idx": 4, "duration_seconds": 1.0})  # 41 > 40
        with self.assertRaises(ValueError):
            pipeline._validate_total_duration(shots)

    def test_stated_duration_over_clip_cap_budgets_as_cap(self):
        """A stated 30s shot is budgeted at max_clip_seconds (10s): one
        generation call cannot return more, so planning 30s would silently
        cut 20s of content — the budget reflects what is renderable."""
        pipeline = self._pipeline(5.0, 40.0)
        shots = [{"idx": 0, "duration_seconds": 30.0}, {"idx": 1}, {"idx": 2}]  # 10+5+5=20
        budget = pipeline._duration_budget(shots)
        self.assertTrue(budget["within_limit"])
        self.assertEqual(budget["per_shot_seconds"][0], 10.0)
        self.assertEqual(budget["total_seconds"], 20.0)

    def test_effective_shot_seconds_prefers_script_timing(self):
        pipeline = self._pipeline(3.0, 40.0)
        self.assertEqual(pipeline._effective_shot_seconds({"duration_seconds": 4.0}), 4.0)
        self.assertEqual(pipeline._effective_shot_seconds({}), 3.0)
        self.assertEqual(pipeline._effective_shot_seconds({"duration_seconds": 0}), 3.0)


class RenderDurationGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_render_refused_when_plan_exceeds_total_limit(self):
        from unittest.mock import AsyncMock, patch

        from agent_runtime.session_index import SessionIndex
        from agent_runtime.tools import ToolRuntimeContext
        from agent_runtime.vimax_adapters import ViMaxAdapters

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            config = workspace / "configs" / "agent.local.yaml"
            config.parent.mkdir()
            config.write_text('render:\n  video_seconds: "3"\n  max_total_video_seconds: "40"\n', encoding="utf-8")
            index = SessionIndex(tmp)
            session = index.create(idea="too many shots")
            script_dir = workspace / session["working_dir"] / "script2video"
            for name in ("characters.json", "storyboard.json", "camera_tree.json"):
                (script_dir / name).write_text("[]", encoding="utf-8")
            (script_dir / "script.txt").write_text("script", encoding="utf-8")
            for shot in range(14):  # 14 x 3s = 42s > 40s cap
                shot_dir = script_dir / "shots" / str(shot)
                shot_dir.mkdir(parents=True)
                (shot_dir / "shot_description.json").write_text("{}", encoding="utf-8")
            adapter = ViMaxAdapters(workspace, index)
            runtime = ToolRuntimeContext("vimax_render_video", "vimax_render_video", progress_callback=lambda event: None)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", return_value=AsyncMock()) as factory:
                result = await adapter.vimax_render_video({}, runtime)
            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "duration_limit_exceeded")
            self.assertEqual(result.metadata["duration_budget"]["shots"], 14)
            self.assertFalse(result.metadata["duration_budget"]["within_limit"])
            factory.assert_not_called()

    async def test_render_gate_reads_per_shot_durations_from_plan(self):
        import json as jsonlib
        from unittest.mock import AsyncMock, patch

        from agent_runtime.session_index import SessionIndex
        from agent_runtime.tools import ToolRuntimeContext
        from agent_runtime.vimax_adapters import ViMaxAdapters

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            config = workspace / "configs" / "agent.local.yaml"
            config.parent.mkdir()
            config.write_text('render:\n  video_seconds: "3"\n  max_total_video_seconds: "40"\n', encoding="utf-8")
            index = SessionIndex(tmp)
            session = index.create(idea="scripted durations")
            script_dir = workspace / session["working_dir"] / "script2video"
            for name in ("characters.json", "storyboard.json", "camera_tree.json"):
                (script_dir / name).write_text("[]", encoding="utf-8")
            (script_dir / "script.txt").write_text("script", encoding="utf-8")
            for shot in range(6):  # 6 shots x 8s each = 48s > 40s despite small shot count
                shot_dir = script_dir / "shots" / str(shot)
                shot_dir.mkdir(parents=True)
                (shot_dir / "shot_description.json").write_text(jsonlib.dumps({"duration_seconds": 8}), encoding="utf-8")
            adapter = ViMaxAdapters(workspace, index)
            runtime = ToolRuntimeContext("vimax_render_video", "vimax_render_video", progress_callback=lambda event: None)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", return_value=AsyncMock()) as factory:
                result = await adapter.vimax_render_video({}, runtime)
            self.assertFalse(result.ok)
            budget = result.metadata["duration_budget"]
            self.assertEqual(budget["total_seconds"], 48.0)
            self.assertEqual(budget["per_shot_seconds"], [8.0] * 6)
            factory.assert_not_called()


class ChainedFrameGenerationTests(unittest.IsolatedAsyncioTestCase):
    """Flow-style chain: within a camera, only the FIRST shot's keyframe is
    generated; later shots' first frames come from the previous clip's end
    frame (copied in the video phase)."""

    def _pipeline(self, tmp: str) -> Script2VideoPipeline:
        from interfaces.shot_description import ShotDescription

        pipeline = Script2VideoPipeline(chat_model=object(), image_generator=object(), video_generator=object(), working_dir=tmp)
        desc0 = ShotDescription(idx=0, is_last=False, cam_idx=0, visual_desc="hook", variation_type="medium", variation_reason="lean", ff_desc="start", ff_vis_char_idxs=[], lf_desc="end", lf_vis_char_idxs=[], motion_desc="leans", audio_desc="")
        desc1 = ShotDescription(idx=1, is_last=True, cam_idx=0, visual_desc="reaction", variation_type="small", variation_reason="minor", ff_desc="next start", ff_vis_char_idxs=[], lf_desc="next end", lf_vis_char_idxs=[], motion_desc="smiles", audio_desc="")
        self.descriptions = [desc0, desc1]
        import asyncio

        self.events = {idx: {"first_frame": asyncio.Event(), "last_frame": asyncio.Event()} for idx in (0, 1)}
        pipeline.frame_events = self.events
        return pipeline

    async def test_chained_camera_generates_only_the_first_keyframe(self):
        from interfaces.camera import Camera

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            shots0 = Path(tmp) / "shots" / "0"
            shots0.mkdir(parents=True)
            (shots0 / "first_frame.png").write_bytes(b"ff")
            (shots0 / "last_frame.png").write_bytes(b"lf")
            calls = []

            async def fake_generate(shot_idx, frame_type, first_shot_ff_path_and_text_pair, frame_desc, visible_characters, character_portraits_registry, progress=None, previous_frame_path_and_text_pair=None, screen_guidance=None):
                calls.append((shot_idx, frame_type))
                self.events[shot_idx][frame_type].set()
                return None

            pipeline.generate_frame_for_single_shot = fake_generate
            camera = Camera(idx=0, active_shot_idxs=[0, 1])
            await pipeline.generate_frames_for_single_camera(camera=camera, shot_descriptions=self.descriptions, characters=[], character_portraits_registry={}, priority_shot_idxs=[])

            # medium shot 0: last_frame anchor generated; shot 1 first frame NOT generated
            self.assertEqual([call[:2] for call in calls], [(0, "last_frame")])
            self.assertTrue(self.events[1]["first_frame"].is_set())  # armed for the video phase

    async def test_chained_small_first_shot_generates_nothing_extra(self):
        from interfaces.camera import Camera

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            self.descriptions[0].variation_type = "small"
            shots0 = Path(tmp) / "shots" / "0"
            shots0.mkdir(parents=True)
            (shots0 / "first_frame.png").write_bytes(b"ff")
            calls = []

            async def fake_generate(shot_idx, frame_type, first_shot_ff_path_and_text_pair, frame_desc, visible_characters, character_portraits_registry, progress=None, previous_frame_path_and_text_pair=None, screen_guidance=None):
                calls.append((shot_idx, frame_type))
                self.events[shot_idx][frame_type].set()

            pipeline.generate_frame_for_single_shot = fake_generate
            camera = Camera(idx=0, active_shot_idxs=[0, 1])
            await pipeline.generate_frames_for_single_camera(camera=camera, shot_descriptions=self.descriptions, characters=[], character_portraits_registry={}, priority_shot_idxs=[])

            self.assertEqual(calls, [])  # small + not-first: nothing to generate
            self.assertTrue(self.events[1]["first_frame"].is_set())


class VideoChainTests(unittest.IsolatedAsyncioTestCase):
    def _pipeline(self, tmp: str) -> Script2VideoPipeline:
        return Script2VideoPipeline(
            chat_model=object(), image_generator=object(), video_generator=object(),
            working_dir=tmp,
            render_geometry={"video_seconds": 3.0, "max_total_video_seconds": 40.0},
        )

    def test_previous_state_pair_prefers_real_video_end_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            shots0 = Path(tmp) / "shots" / "0"
            shots0.mkdir(parents=True)
            self.assertTrue(pipeline._previous_state_pair(0)[0].endswith("first_frame.png"))  # nothing exists -> fallback
            (shots0 / "first_frame.png").write_bytes(b"ff")
            (shots0 / "last_frame.png").write_bytes(b"lf")
            self.assertTrue(pipeline._previous_state_pair(0)[0].endswith("last_frame.png"))
            (shots0 / "video_last_frame.png").write_bytes(b"vf")
            self.assertTrue(pipeline._previous_state_pair(0)[0].endswith("video_last_frame.png"))

    def test_video_prompt_carries_dialogue_and_no_text_rules(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            shot = SimpleNamespace(motion_desc="She leans forward", audio_desc='[Speaker] Grandma: "Line!"')
            plain = pipeline._build_video_prompt(shot, 10.0, chained=False)
            self.assertIn("fully completed within the first 8 seconds", plain)  # 10s * 0.85
            self.assertIn("No text, letters, numbers", plain)
            self.assertIn("exactly two hands", plain)
            self.assertIn("back cover is opaque", plain)
            self.assertNotIn("continues directly from the previous shot", plain)
            chained = pipeline._build_video_prompt(shot, 10.0, chained=True)
            self.assertIn("continues directly from the previous shot", chained)
            self.assertIn("FIRST reference image is the previous", chained)

    async def test_shot_videos_chain_real_end_frames(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            shots = [SimpleNamespace(idx=0), SimpleNamespace(idx=1), SimpleNamespace(idx=2)]
            calls = []
            end_frames = {0: "end0.png", 1: "end1.png", 2: "end2.png"}

            async def fake_video(shot_description, progress=None, previous_end_frame_path=None):
                calls.append((shot_description.idx, previous_end_frame_path))

            pipeline.generate_video_for_single_shot = fake_video
            pipeline._ensure_video_end_frame = lambda video_path, shot_idx: end_frames[shot_idx]
            await pipeline._generate_all_shot_videos(shots)
            self.assertEqual(calls, [(0, None), (1, "end0.png"), (2, "end1.png")])


    async def test_chained_video_starts_exactly_from_previous_end_frame(self):
        import asyncio

        class CapturingVideoGenerator:
            def __init__(self):
                self.calls = []

            async def generate_single_video(self, prompt, reference_image_paths, aspect_ratio=None, resolution=None, progress=None, **kwargs):
                self.calls.append({"prompt": prompt, "references": list(reference_image_paths)})
                return SimpleNamespace(save=lambda path: None)

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            pipeline.frame_events = {1: {"first_frame": asyncio.Event()}}
            pipeline.frame_events[1]["first_frame"].set()  # frames ready — else the pipeline awaits forever
            shots1 = Path(tmp) / "shots" / "1"
            shots1.mkdir(parents=True)
            (shots1 / "first_frame.png").write_bytes(b"own")  # must NOT be used when chained
            prev_end = Path(tmp) / "shots" / "0" / "video_last_frame.png"
            prev_end.parent.mkdir(parents=True, exist_ok=True)
            prev_end.write_bytes(b"prev-end")
            generator = CapturingVideoGenerator()
            pipeline.video_generator = generator
            shot = SimpleNamespace(idx=1, variation_type="small", motion_desc="continues", audio_desc="", duration_seconds=4.0)
            await pipeline.generate_video_for_single_shot(shot, previous_end_frame_path=str(prev_end))
            self.assertEqual(len(generator.calls), 1)
            call = generator.calls[0]
            self.assertEqual(call["references"], [str(prev_end)])  # literal continuation, own keyframe excluded
            self.assertIn("FIRST reference image is the previous", call["prompt"])


class NoTextPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_generated_image_prompts_forbid_text(self):
        class CapturingGenerator:
            def __init__(self):
                self.prompts = []

            async def generate_single_image(self, prompt, reference_image_paths, size=None, aspect_ratio=None, allow_portrait=None, **kwargs):
                self.prompts.append(prompt)
                return SimpleNamespace(save=lambda path: None)

        from types import SimpleNamespace

        generator = CapturingGenerator()
        pipeline = Script2VideoPipeline(
            chat_model=object(), image_generator=generator, video_generator=object(),
            working_dir=tempfile.mkdtemp(),
            num_image_candidates=1,
            render_geometry={"video_seconds": 3.0},
        )
        await pipeline.generate_and_select_best_image(
            prompt="Grandma holds her phone",
            reference_image_paths=[],
            reference_image_path_and_text_pairs=[],
            target_description="Grandma holds her phone",
        )
        self.assertEqual(len(generator.prompts), 1)
        self.assertIn("No text, letters, numbers", generator.prompts[0])
        self.assertIn("exactly two hands", generator.prompts[0])
        self.assertIn("never displays screen content", generator.prompts[0])
        self.assertIn("Grandma holds her phone", generator.prompts[0])


class TrimVideoTests(unittest.TestCase):
    def _write_clip(self, path: str, duration: float) -> None:
        from moviepy import ColorClip

        ColorClip(size=(64, 64), color=(180, 40, 40), duration=duration).write_videofile(
            path, fps=10, codec="libx264", preset="ultrafast", logger=None
        )

    def test_long_clip_is_trimmed_to_budget(self):
        from moviepy import VideoFileClip

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "clip.mp4")
            try:
                self._write_clip(path, 5.0)
            except Exception as exc:  # ffmpeg unavailable in this environment
                self.skipTest(f"cannot encode test video: {exc}")
            self.assertTrue(_trim_video_to_seconds(path, 2.0))
            with VideoFileClip(path) as trimmed:
                self.assertLessEqual(trimmed.duration, 2.3)
                self.assertGreaterEqual(trimmed.duration, 1.7)

    def test_short_clip_is_left_alone(self):
        from moviepy import VideoFileClip

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "clip.mp4")
            try:
                self._write_clip(path, 2.0)
            except Exception as exc:
                self.skipTest(f"cannot encode test video: {exc}")
            self.assertFalse(_trim_video_to_seconds(path, 5.0))
            with VideoFileClip(path) as clip:
                self.assertGreaterEqual(clip.duration, 1.7)


if __name__ == "__main__":
    unittest.main()
