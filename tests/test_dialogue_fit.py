import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from interfaces.shot_description import ShotBriefDescription
from pipelines.script2video_pipeline import Script2VideoPipeline
from utils.dialogue_fit import (
    DialogueFitError,
    dialogue_fit_error_message,
    dialogue_fit_for_shot,
    dialogue_fit_report,
    estimate_speech_seconds,
    extend_over_budget_shots,
    extract_dialogue_lines,
)


class DialogueFitTests(unittest.TestCase):
    def test_extracts_quoted_lines(self):
        audio = '[Speaker] Grandma: "I been playing Bid Whist for fifty years!" [Speaker] Family: "Try it on your phone!"'
        lines = extract_dialogue_lines(audio)
        self.assertEqual(lines, ["I been playing Bid Whist for fifty years!", "Try it on your phone!"])

    def test_speech_estimate_includes_pauses(self):
        self.assertEqual(estimate_speech_seconds([]), 0.0)
        # 13 words / 2.8 = 4.64 + one 0.3s pause = 4.94
        self.assertAlmostEqual(estimate_speech_seconds(["I been playing Bid Whist for fifty years", "Try it on your phone"]), 4.94, places=1)

    def test_fit_statuses(self):
        fits = dialogue_fit_for_shot({"audio_desc": 'Grandma: "Hi there"', "duration_seconds": 5}, 8)
        self.assertEqual(fits["status"], "fits")
        over = dialogue_fit_for_shot(
            {"audio_desc": 'A: "I been playing Bid Whist for fifty years!" B: "Try it on your phone!"', "duration_seconds": 3}, 8
        )
        self.assertEqual(over["status"], "over")
        self.assertGreater(over["recommended_seconds"], 3)
        tight = dialogue_fit_for_shot({"audio_desc": 'A: "seven spoken words right here in this line"', "duration_seconds": 3}, 8)
        self.assertEqual(tight["status"], "tight")  # within the tolerance band, accepted

    def test_report_and_error_message(self):
        shots = [
            {"audio_desc": 'A: "short line"', "duration_seconds": 4},
            {"audio_desc": 'A: "I been playing Bid Whist for fifty years!" B: "Try it on your phone!"', "duration_seconds": 3},
        ]
        report = dialogue_fit_report(shots, 8)
        self.assertEqual(report["over_shots"], [1])
        message = dialogue_fit_error_message(report)
        self.assertIsNotNone(message)
        self.assertIn("Shot 1", message)
        self.assertIn("Recommended: extend this shot", message)
        self.assertIsNone(dialogue_fit_error_message(dialogue_fit_report(shots[:1], 8)))


class PipelineDialogueFitGateTests(unittest.TestCase):
    def _pipeline(self, tmp):
        return Script2VideoPipeline(
            chat_model=object(), image_generator=object(), video_generator=object(),
            working_dir=tmp, render_geometry={"video_seconds": 3.0, "max_total_video_seconds": 40.0},
            review_config={"enabled": False},
        )

    def test_over_budget_dialogue_fails_validation_with_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            shots = [
                {"idx": 0, "duration_seconds": 3.0, "audio_desc": 'Grandma: "I been playing Bid Whist for fifty years!" Family: "Try it on your phone!"'},
            ]
            with self.assertRaises(ValueError) as ctx:
                pipeline._validate_dialogue_fit(shots)
            self.assertIn("cannot complete", str(ctx.exception))
            self.assertIn("Recommended: extend", str(ctx.exception))

    def test_fitting_dialogue_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            shots = [{"idx": 0, "duration_seconds": 4.0, "audio_desc": 'Grandma: "TAKE THAT, PARTNER!"'}]
            pipeline._validate_dialogue_fit(shots)  # no raise

    def test_video_prompt_keeps_original_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            shot = SimpleNamespace(motion_desc="She speaks", audio_desc='[Speaker] Grandma: "मैं पचास साल से बिड व्हिस्ट खेल रही हूँ!"')
            prompt = pipeline._build_video_prompt(shot, 6.0, chained=False)
            self.assertIn("exactly as written, in its original language", prompt)
            self.assertIn("मैं पचास साल से बिड व्हिस्ट खेल रही हूँ!", prompt)


class AutoExtendDialogueFitTests(unittest.TestCase):
    """The mechanical extension that keeps dialogue intact instead of failing.

    Regression guard for the failure mode where ad dialogue in a short shot
    killed the whole narrative planning. Extension still applies when the
    recommended duration fits under ``max_clip_seconds`` (default 10s).

    With the recalibrated conversational pace (2.8 w/s), the 22-word ad audio
    is "tight" in a 10s shot — accepted, like Google Flow renders it. Speech
    that exceeds one clip even with the tolerance band (31 words ≈ 11.97s) is
    never extended beyond the clip cap: the video model returns at most ~10s
    per clip, so an over-cap plan rendered as a shorter clip with the dialogue
    cut mid-line (the observed test-13 failure). The pipeline refuses with a
    split-the-shot recommendation instead of a plan it cannot render.
    """

    LINES = ["Arey waah! Bada Joker mil gaya!", "Le beta! Ab khel ke dikha!", "Ab aaya maza!", "THINK YOU CAN BEAT DADI? INSTALL NOW!"]
    HEAVY_LINES = [
        "Arey waah! Bada Joker mil gaya!",
        "Le beta! Ab khel ke dikha!",
        "Ab aaya maza! Dekho kitna aasan hai ye naya game!",
        "THINK YOU CAN BEAT DADI? INSTALL NOW TODAY ONLY!",
    ]

    def _audio(self) -> str:
        return " ".join(f'[Speaker] Dadi: "{line}"' for line in self.LINES)

    def _heavy_audio(self) -> str:
        return " ".join(f'[Speaker] Dadi: "{line}"' for line in self.HEAVY_LINES)

    def _pipeline(self, tmp):
        return Script2VideoPipeline(
            chat_model=object(), image_generator=object(), video_generator=object(),
            working_dir=tmp, render_geometry={"video_seconds": 10.0, "max_total_video_seconds": 40.0},
            review_config={"enabled": False},
        )

    def test_full_ad_audio_is_tight_in_a_10s_shot(self):
        """Flow parity: 22 conversational words fit a 10s clip — the plan is
        accepted with a tight status instead of refused."""
        fit = dialogue_fit_for_shot({"audio_desc": self._audio(), "duration_seconds": 10}, 10)
        self.assertEqual(fit["words"], 22)
        self.assertEqual(fit["estimated_speech_seconds"], 8.76)
        self.assertEqual(fit["status"], "tight")

    def test_extends_over_shot_to_recommended_within_cap(self):
        two_lines = " ".join(f'[Speaker] Dadi: "{line}"' for line in self.LINES[:2])
        shots = [{"duration_seconds": 3, "audio_desc": two_lines}]
        adjustments = extend_over_budget_shots(shots, default_seconds=3, max_total_seconds=40)
        self.assertEqual(len(adjustments), 1)
        self.assertEqual(adjustments[0]["shot"], 0)
        self.assertEqual(adjustments[0]["from_seconds"], 3)
        self.assertEqual(adjustments[0]["to_seconds"], 6)
        self.assertEqual(shots[0]["duration_seconds"], 6)
        self.assertEqual(dialogue_fit_report(shots, 3)["over_shots"], [])

    def test_no_extension_when_total_cap_blocks_it(self):
        two_lines = " ".join(f'[Speaker] Dadi: "{line}"' for line in self.LINES[:2])
        shots = [{"duration_seconds": 3, "audio_desc": two_lines}] + [
            {"duration_seconds": 3, "audio_desc": ""} for _ in range(3)
        ]
        self.assertEqual(extend_over_budget_shots(shots, 3, 6), [])  # 6+9 exceeds the 6s total cap
        self.assertEqual(shots[0]["duration_seconds"], 3)  # nothing half-applied

    def test_validate_raises_dialogue_fit_error_after_extension_impossible(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            shots = [{"idx": 0, "duration_seconds": 10.0, "audio_desc": self._heavy_audio()}] + [
                {"idx": i, "duration_seconds": 10.0, "audio_desc": ""} for i in range(1, 4)
            ]
            with self.assertRaises(DialogueFitError):
                pipeline._validate_dialogue_fit(shots)

    def test_pipeline_adjustment_persists_to_disk(self):
        """Under the clip cap the extension still applies and persists."""
        # Two lines (12 words ≈ 4.59s of speech) in a 3s shot: the recommended
        # duration (6s) is below max_clip_seconds (10s), so it is renderable.
        audio = " ".join(f'[Speaker] Dadi: "{line}"' for line in self.LINES[:2])
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            brief = ShotBriefDescription.model_validate(
                {"idx": 0, "is_last": True, "cam_idx": 0, "visual_desc": "Dadi plays cards", "audio_desc": audio, "duration_seconds": 3.0}
            )
            storyboard_path = Path(tmp) / "storyboard.json"
            storyboard_path.write_text(json.dumps([brief.model_dump()]), encoding="utf-8")
            shot_dir = Path(tmp) / "shots" / "0"
            shot_dir.mkdir(parents=True)
            (shot_dir / "shot_description.json").write_text(
                json.dumps({"duration_seconds": 3.0, "audio_desc": audio}), encoding="utf-8"
            )
            adjustments = pipeline._adjust_dialogue_fit([brief], None, quiet=True)
            self.assertEqual(len(adjustments), 1)
            self.assertGreater(brief.duration_seconds, 3.0)
            self.assertLessEqual(brief.duration_seconds, pipeline._max_clip_seconds())
            persisted = json.loads(storyboard_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted[0]["duration_seconds"], brief.duration_seconds)
            shot_payload = json.loads((shot_dir / "shot_description.json").read_text(encoding="utf-8"))
            self.assertEqual(shot_payload["duration_seconds"], brief.duration_seconds)
            pipeline._validate_dialogue_fit([brief])  # the fitted plan now passes the gate

    def test_over_cap_dialogue_is_refused_with_split_recommendation(self):
        """Speech longer than one clip can hold is never extended.

        31 words ≈ 11.97s in a 10s shot: even with the 15% tolerance band the
        lines cannot complete inside one clip (the model renders at most ~10s),
        so the old behavior's 13s plan would render short and cut the dialogue
        mid-line. Nothing is adjusted, and validation refuses with the split
        recommendation.
        """
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            brief = ShotBriefDescription.model_validate(
                {"idx": 0, "is_last": True, "cam_idx": 0, "visual_desc": "Dadi plays cards", "audio_desc": self._heavy_audio(), "duration_seconds": 10.0}
            )
            self.assertEqual(pipeline._adjust_dialogue_fit([brief], None, quiet=True), [])
            self.assertEqual(brief.duration_seconds, 10.0)  # nothing half-applied
            with self.assertRaises(DialogueFitError) as ctx:
                pipeline._validate_dialogue_fit([brief])
            message = str(ctx.exception)
            self.assertIn("split this shot", message)
            self.assertIn("at most 10s", message)

    def test_a_larger_clip_cap_restores_the_extension(self):
        """The refusal comes from the clip cap, not from the total cap: with a
        15s clip cap the 31 heavy words extend to the recommended 14s again."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline(
                chat_model=object(), image_generator=object(), video_generator=object(),
                working_dir=tmp,
                render_geometry={"video_seconds": 10.0, "max_total_video_seconds": 40.0, "max_clip_seconds": 15.0},
                review_config={"enabled": False},
            )
            brief = ShotBriefDescription.model_validate(
                {"idx": 0, "is_last": True, "cam_idx": 0, "visual_desc": "Dadi plays cards", "audio_desc": self._heavy_audio(), "duration_seconds": 10.0}
            )
            (Path(tmp) / "storyboard.json").write_text(json.dumps([brief.model_dump()]), encoding="utf-8")
            adjustments = pipeline._adjust_dialogue_fit([brief], None, quiet=True)
            self.assertEqual(len(adjustments), 1)
            self.assertEqual(brief.duration_seconds, 14.0)
            pipeline._validate_dialogue_fit([brief])


class RebalanceDialogueFitTests(unittest.TestCase):
    """Dialogue lands inside the requested time — the Flow-style behavior."""

    HEAVY_LINES = [
        "Arey waah! Bada Joker mil gaya!",
        "Le beta! Ab khel ke dikha!",
        "Ab aaya maza! Dekho kitna aasan hai ye naya game!",
        "THINK YOU CAN BEAT DADI? INSTALL NOW TODAY ONLY!",
    ]

    def _audio(self) -> str:
        return " ".join(f'[Speaker] Dadi: "{line}"' for line in self.HEAVY_LINES)

    def test_rebalance_keeps_the_requested_total(self):
        from utils.dialogue_fit import rebalance_over_budget_shots

        shots = [{"duration_seconds": 10, "audio_desc": self._audio()}, {"duration_seconds": 10, "audio_desc": ""}]
        adjustments = rebalance_over_budget_shots(shots, default_seconds=10, max_total_seconds=40)
        by_source = {item["source"]: item for item in adjustments}
        self.assertIn("rebalance", by_source)
        self.assertIn("donor", by_source)
        self.assertEqual(shots[0]["duration_seconds"], 14)  # lines now fit
        self.assertEqual(shots[1]["duration_seconds"], 6)  # funded the 4s
        self.assertEqual(sum(shot["duration_seconds"] for shot in shots), 20)  # total unchanged
        self.assertEqual(dialogue_fit_report(shots, 10)["over_shots"], [])

    def test_donor_never_shrinks_below_minimum(self):
        from utils.dialogue_fit import MIN_DONOR_SECONDS, rebalance_over_budget_shots

        # Donor has 3s with no dialogue: only 1s is donatable (floor 2s), so the
        # 3s deficit falls back to extension within the cap.
        shots = [{"duration_seconds": 10, "audio_desc": self._audio()}, {"duration_seconds": 3, "audio_desc": ""}]
        adjustments = rebalance_over_budget_shots(shots, default_seconds=10, max_total_seconds=40)
        self.assertEqual(shots[0]["duration_seconds"], 14)
        self.assertGreaterEqual(shots[1]["duration_seconds"], MIN_DONOR_SECONDS)
        self.assertEqual(shots[1]["duration_seconds"], 3)  # untouched by extension
        self.assertTrue(all(item.get("source") != "donor" for item in adjustments))

    def test_returns_empty_when_nothing_can_fit(self):
        from utils.dialogue_fit import rebalance_over_budget_shots

        shots = [{"duration_seconds": 10, "audio_desc": self._audio()}]
        self.assertEqual(rebalance_over_budget_shots(shots, 10, 10), [])  # cap 10s blocks everything
        self.assertEqual(shots[0]["duration_seconds"], 10)  # unchanged


class ScreenwriterTimingContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_contract_text_carries_real_numbers(self):
        from agents.screenwriter import timing_contract_text

        contract = timing_contract_text(video_seconds=10, max_total_seconds=40)
        self.assertIn("[Timing Contract]", contract)
        self.assertIn("10s", contract)
        self.assertIn("40s", contract)
        self.assertIn("2.8 words per second", contract)

    async def test_contract_reaches_the_screenwriter_system_prompt(self):
        from agents.screenwriter import Screenwriter, timing_contract_text

        captured: dict[str, Any] = {}

        class CapturingModel:
            async def ainvoke(self, messages):
                captured["messages"] = messages
                return SimpleNamespace(content='{"script": ["scene one"]}')

        writer = Screenwriter(chat_model=CapturingModel())
        script = await writer.write_script_based_on_story(
            story="a story",
            user_requirement="",
            timing_contract=timing_contract_text(video_seconds=10, max_total_seconds=40),
        )
        self.assertEqual(script, ["scene one"])
        system_message = captured["messages"][0][1]
        self.assertIn("[Timing Contract]", system_message)
        self.assertIn("about 23 words", system_message)  # 10s * 2.8 * 0.85


class RenderDialogueGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_render_refused_when_dialogue_cannot_fit(self):
        from agent_runtime.session_index import SessionIndex
        from agent_runtime.tools import ToolRuntimeContext
        from agent_runtime.vimax_adapters import ViMaxAdapters

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            config = workspace / "configs" / "agent.local.yaml"
            config.parent.mkdir()
            config.write_text('render:\n  video_seconds: "3"\n', encoding="utf-8")
            index = SessionIndex(tmp)
            session = index.create(idea="tight dialogue")
            script_dir = workspace / session["working_dir"] / "script2video"
            for name in ("characters.json", "storyboard.json", "camera_tree.json"):
                (script_dir / name).write_text("[]", encoding="utf-8")
            (script_dir / "script.txt").write_text("script", encoding="utf-8")
            shot_dir = script_dir / "shots" / "0"
            shot_dir.mkdir(parents=True)
            (shot_dir / "shot_description.json").write_text(
                json.dumps({
                    "duration_seconds": 3,
                    "audio_desc": 'Grandma: "I been playing Bid Whist for fifty years!" Family: "Try it on your phone!"',
                }),
                encoding="utf-8",
            )
            adapter = ViMaxAdapters(workspace, index)
            runtime = ToolRuntimeContext("vimax_render_video", "vimax_render_video", progress_callback=lambda event: None)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", return_value=AsyncMock()) as factory:
                result = await adapter.vimax_render_video({}, runtime)
            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "dialogue_fit_exceeded")
            self.assertIn("Recommended: extend", result.content)
            factory.assert_not_called()


class NarrativePlanningDialogueRefusalTests(unittest.IsolatedAsyncioTestCase):
    """Planning failures caused by the dialogue-fit gate must not look retryable.

    Regression guard for the run where the agent retried/flailed through its
    whole tool budget after a deterministic dialogue-fit refusal.
    """

    async def test_dialogue_fit_failure_is_deterministic_and_actionable(self):
        from agent_runtime import vimax_adapters as va
        from agent_runtime.session_index import SessionIndex
        from agent_runtime.tools import ToolRuntimeContext

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            index = SessionIndex(tmp)
            adapter = va.ViMaxAdapters(workspace, index)
            runtime = ToolRuntimeContext("vimax_narrative_planning", "vimax_narrative_planning", progress_callback=lambda event: None)

            async def failing_plan(**kwargs):
                try:
                    raise DialogueFitError(
                        "Dialogue cannot fit its shot budgets. Shot 0: 22 words of dialogue need about 10.62s "
                        "but the shot is only 10s — the lines cannot complete and will be cut. "
                        "Recommended: extend this shot to 13s (or split its lines into separate shots)."
                    )
                except DialogueFitError as inner:
                    # Mirrors _run_planning_step wrapping stage errors in RuntimeError.
                    raise RuntimeError("Planning scene 0 storyboard and shots failed") from inner

            idea_pipeline = MagicMock()
            idea_pipeline.develop_story = AsyncMock(return_value="story")
            idea_pipeline.extract_characters = AsyncMock(return_value=[])
            idea_pipeline.write_script_based_on_story = AsyncMock(return_value=["scene one"])
            script_pipeline = MagicMock()
            script_pipeline.plan_text_artifacts = failing_plan

            with patch.object(va, "_build_chat_model", return_value=object()), \
                 patch.object(va, "Idea2VideoPipeline", return_value=idea_pipeline), \
                 patch.object(va, "Script2VideoPipeline", return_value=script_pipeline):
                result = await adapter.vimax_narrative_planning({"idea": "dadi card game ad", "style": "Cinematic"}, runtime)

            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "dialogue_fit_exceeded")
            self.assertFalse(result.metadata["retryable"])
            self.assertIn("Do not retry narrative planning", result.metadata["recommended_action"])
            self.assertIn("split the over-budget lines", result.metadata["recommended_action"])

    async def test_other_planning_failures_stay_retryable(self):
        from agent_runtime import vimax_adapters as va
        from agent_runtime.session_index import SessionIndex
        from agent_runtime.tools import ToolRuntimeContext

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            index = SessionIndex(tmp)
            adapter = va.ViMaxAdapters(workspace, index)
            runtime = ToolRuntimeContext("vimax_narrative_planning", "vimax_narrative_planning", progress_callback=lambda event: None)

            async def failing_plan(**kwargs):
                raise RuntimeError("LLM returned invalid JSON")

            idea_pipeline = MagicMock()
            idea_pipeline.develop_story = AsyncMock(return_value="story")
            idea_pipeline.extract_characters = AsyncMock(return_value=[])
            idea_pipeline.write_script_based_on_story = AsyncMock(return_value=["scene one"])
            script_pipeline = MagicMock()
            script_pipeline.plan_text_artifacts = failing_plan

            with patch.object(va, "_build_chat_model", return_value=object()), \
                 patch.object(va, "Idea2VideoPipeline", return_value=idea_pipeline), \
                 patch.object(va, "Script2VideoPipeline", return_value=script_pipeline):
                result = await adapter.vimax_narrative_planning({"idea": "dadi card game ad", "style": "Cinematic"}, runtime)

            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "recoverable_planning_step_failed")
            self.assertTrue(result.metadata["retryable"])
            self.assertNotIn("recommended_action", result.metadata)


if __name__ == "__main__":
    unittest.main()


class PureExtensionTests(unittest.TestCase):
    """User rule: dialogue is never cut or re-timed elsewhere - a shot whose
    speech exceeds its budget simply grows; the total video may get longer."""

    def test_tight_overshoot_shot_is_extended(self):
        # 8 words ≈ 2.86s speech in a 2.5s shot: extended to 4s (within cap)
        shots = [{"duration_seconds": 2.5, "audio_desc": 'Dadi: "I been playing Bid Whist for fifty years!"'}]
        adjustments = extend_over_budget_shots(shots, default_seconds=8, max_total_seconds=40)
        self.assertEqual(len(adjustments), 1)
        self.assertEqual(shots[0]["duration_seconds"], 4)
        self.assertEqual(dialogue_fit_report(shots, 8)["over_shots"], [])

    def test_extension_grows_the_total_no_donor_borrowing(self):
        shots = [
            {"duration_seconds": 2.5, "audio_desc": 'Dadi: "I been playing Bid Whist for fifty years!"'},
            {"duration_seconds": 2.0, "audio_desc": ""},
        ]
        extend_over_budget_shots(shots, default_seconds=8, max_total_seconds=40)
        self.assertEqual(shots[1]["duration_seconds"], 2.0)  # other shots untouched
        self.assertGreater(shots[0]["duration_seconds"], 2.5)  # the talking shot grew

    def test_flow_parity_tight_at_the_cap_is_accepted(self):
        # 22 words ≈ 8.76s in a 10s shot at the 10s cap: stays as-is (model paces it)
        audio = " ".join(f'[Speaker] Dadi: "{line}"' for line in [
            "Arey waah! Bada Joker mil gaya!", "Le beta! Ab khel ke dikha!", "Ab aaya maza!", "THINK YOU CAN BEAT DADI? INSTALL NOW!"])
        shots = [{"duration_seconds": 10, "audio_desc": audio}]
        self.assertEqual(extend_over_budget_shots(shots, 10, 40), [])
        self.assertEqual(shots[0]["duration_seconds"], 10)
        self.assertEqual(dialogue_fit_report(shots, 10)["over_shots"], [])
