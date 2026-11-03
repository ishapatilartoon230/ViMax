import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pipelines.script2video_pipeline import Script2VideoPipeline
from utils.script_fidelity import (
    expected_shot_count,
    missing_dialogue_lines,
    script_fidelity_error_message,
    script_fidelity_report,
)

HINDI_SCRIPT = """**SHOT 1**

दादी: "अरे बेटा! ताश खेलते हुए मुझे पचास साल हो गए!"

परिवार का सदस्य: "दादी, अब अपने फोन पर Bid Whist खेलकर देखो!"

दादी: "अच्छा! इसमें Downtown भी है क्या?"

परिवार का सदस्य: "हाँ दादी!"

**SHOT 2**

दादी: "अरे वाह! बड़ा जोकर मिल गया!"
"""

ENGLISH_SCRIPT = """**SHOT 1**
Grandma: "I been playing Bid Whist for fifty years!"
Family (off-camera): "Try it on your phone!"

**SHOT 2**
Grandma: "Ooooh! I got the Big Joker!"
"""


class ScriptFidelityTests(unittest.TestCase):
    def test_shot_marker_count(self):
        self.assertEqual(expected_shot_count(HINDI_SCRIPT), 2)
        self.assertEqual(expected_shot_count(ENGLISH_SCRIPT), 2)
        self.assertIsNone(expected_shot_count("just a vague idea, no shots"))

    def test_complete_plan_passes_including_hindi(self):
        shots = [
            {"audio_desc": 'दादी: "अरे बेटा! ताश खेलते हुए मुझे पचास साल हो गए!" परिवार: "दादी, अब अपने फोन पर Bid Whist खेलकर देखो!" दादी: "अच्छा! इसमें Downtown भी है क्या?" परिवार: "हाँ दादी!"'},
            {"audio_desc": 'दादी: "अरे वाह! बड़ा जोकर मिल गया!"'},
        ]
        report = script_fidelity_report(HINDI_SCRIPT, shots)
        self.assertTrue(report["all_present"], report)
        self.assertIsNone(script_fidelity_error_message(report))

    def test_missing_line_is_reported_with_exact_text(self):
        shots = [
            {"audio_desc": 'दादी: "अरे बेटा! ताश खेलते हुए मुझे पचास साल हो गए!"'},
            {"audio_desc": 'दादी: "अरे वाह! बड़ा जोकर मिल गया!"'},
        ]
        report = script_fidelity_report(HINDI_SCRIPT, shots)
        self.assertFalse(report["all_present"])
        # three of the four shot-1 lines are missing
        self.assertEqual(len(report["missing_dialogue"]), 3)
        message = script_fidelity_error_message(report)
        self.assertIn("must appear verbatim", message)
        self.assertIn("Downtown", message)

    def test_shot_count_mismatch_fails(self):
        shots = [{"audio_desc": 'Grandma: "I been playing Bid Whist for fifty years!" Family: "Try it on your phone!" Grandma: "Ooooh! I got the Big Joker!"'}]
        report = script_fidelity_report(ENGLISH_SCRIPT, shots)
        self.assertFalse(report["shot_count_ok"])
        self.assertIn("2 shots", script_fidelity_error_message(report))

    def test_punctuation_and_case_dont_matter(self):
        shots = [{"audio_desc": 'Grandma says: "i been playing bid whist for FIFTY years!" and "try it on your phone!" "oooooh! i got the big joker!"'}]
        report = script_fidelity_report(ENGLISH_SCRIPT, shots)
        self.assertEqual(report["missing_dialogue"], [])


class MissingDialogueLinesTests(unittest.TestCase):
    def test_missing_lines_between_texts(self):
        source = 'Grandma: "I been playing Bid Whist for fifty years!" Family: "Try it on your phone!"'
        kept = 'Grandma says "I been playing Bid Whist for fifty years!" and nothing else'
        self.assertEqual(missing_dialogue_lines(source, kept), ["Try it on your phone!"])
        self.assertEqual(missing_dialogue_lines(source, source), [])


class ScriptIntakeTests(unittest.IsolatedAsyncioTestCase):
    async def test_unstructured_script_runs_intake_and_stores_enhanced(self):
        from agent_runtime.session_index import SessionIndex
        from agent_runtime.vimax_adapters import ViMaxAdapters

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            index = SessionIndex(tmp)
            adapter = ViMaxAdapters(workspace, index)
            messy = 'A grandma plays cards on her phone. Grandma: "I been playing Bid Whist for fifty years!" Family: "Try it on your phone!"'
            enhanced = "**SHOT 1**\nGrandma: \"I been playing Bid Whist for fifty years!\"\nFamily: \"Try it on your phone!\"\nWarm living room, afternoon light."
            fake_planner = SimpleNamespace(plan_script=lambda basic_idea: SimpleNamespace(planned_script="planned " + basic_idea))
            fake_enhancer = SimpleNamespace(enhance_script=AsyncMock(return_value=SimpleNamespace(enhanced_script=enhanced)))
            with patch("agent_runtime.vimax_adapters.llm_api_key", return_value="key"), \
                 patch("agent_runtime.vimax_adapters.llm_base_url", return_value="https://example.invalid/v1"), \
                 patch("agent_runtime.vimax_adapters.llm_model", return_value="model"), \
                 patch("agent_runtime.vimax_adapters.ScriptPlanner", return_value=fake_planner) as planner_factory, \
                 patch("agent_runtime.vimax_adapters.ScriptEnhancer", return_value=fake_enhancer) as enhancer_factory, \
                 patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", return_value=AsyncMock()) as pipeline_factory:
                pipeline_factory.return_value.plan_text_artifacts = AsyncMock(return_value=None)
                result = await adapter.vimax_narrative_planning({"script": messy, "user_requirement": "short"})
            self.assertTrue(result.ok, result.content)
            session = index.active()
            script_dir = workspace / session["working_dir"] / "script2video"
            self.assertEqual((script_dir / "script.txt").read_text(encoding="utf-8"), enhanced)
            self.assertIn("I been playing Bid Whist", (script_dir / "script_source.txt").read_text(encoding="utf-8"))
            self.assertEqual(planner_factory.call_count, 1)
            self.assertEqual(enhancer_factory.call_count, 1)
            self.assertIn("enhancement_skills", enhancer_factory.call_args.kwargs)  # skills threaded

    async def test_intake_that_drops_dialogue_keeps_original(self):
        from agent_runtime.session_index import SessionIndex
        from agent_runtime.vimax_adapters import ViMaxAdapters

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            index = SessionIndex(tmp)
            adapter = ViMaxAdapters(workspace, index)
            messy = 'Grandma: "I been playing Bid Whist for fifty years!" Family: "Try it on your phone!"'
            broken = '**SHOT 1**\nGrandma: "I been playing Bid Whist for fifty years!"'  # family line lost
            fake_planner = SimpleNamespace(plan_script=lambda basic_idea: SimpleNamespace(planned_script="planned"))
            fake_enhancer = SimpleNamespace(enhance_script=AsyncMock(return_value=SimpleNamespace(enhanced_script=broken)))
            with patch("agent_runtime.vimax_adapters.llm_api_key", return_value="key"), \
                 patch("agent_runtime.vimax_adapters.llm_base_url", return_value="https://example.invalid/v1"), \
                 patch("agent_runtime.vimax_adapters.llm_model", return_value="model"), \
                 patch("agent_runtime.vimax_adapters.ScriptPlanner", return_value=fake_planner), \
                 patch("agent_runtime.vimax_adapters.ScriptEnhancer", return_value=fake_enhancer), \
                 patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", return_value=AsyncMock()) as pipeline_factory:
                pipeline_factory.return_value.plan_text_artifacts = AsyncMock(return_value=None)
                result = await adapter.vimax_narrative_planning({"script": messy})
            self.assertTrue(result.ok, result.content)
            session = index.active()
            script_dir = workspace / session["working_dir"] / "script2video"
            self.assertIn("Try it on your phone!", (script_dir / "script.txt").read_text(encoding="utf-8"))
            self.assertFalse((script_dir / "script_source.txt").exists())

    async def test_structured_script_bypasses_intake(self):
        from agent_runtime.session_index import SessionIndex
        from agent_runtime.vimax_adapters import ViMaxAdapters

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            index = SessionIndex(tmp)
            adapter = ViMaxAdapters(workspace, index)
            structured = '**SHOT 1**\nGrandma: "I been playing Bid Whist for fifty years!"'
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", return_value=AsyncMock()) as pipeline_factory, \
                 patch("agent_runtime.vimax_adapters.ScriptPlanner") as planner_factory:
                pipeline_factory.return_value.plan_text_artifacts = AsyncMock(return_value=None)
                result = await adapter.vimax_narrative_planning({"script": structured})
            self.assertTrue(result.ok, result.content)
            planner_factory.assert_not_called()
            session = index.active()
            script_dir = workspace / session["working_dir"] / "script2video"
            self.assertEqual((script_dir / "script.txt").read_text(encoding="utf-8"), structured)


class PipelineFidelityBounceTests(unittest.IsolatedAsyncioTestCase):
    async def test_storyboard_bounces_until_script_lines_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline(
                chat_model=object(), image_generator=object(), video_generator=object(),
                working_dir=tmp,
                render_geometry={"video_seconds": 8.0, "max_total_video_seconds": 40.0},
                review_config={"enabled": False},
            )
            attempts = []

            async def fake_design(script, characters, user_requirement, quiet=False):
                attempts.append(user_requirement)
                if len(attempts) == 1:
                    # first attempt drops the family line and merges shots
                    return [{"idx": 0, "is_last": True, "cam_idx": 0, "visual_desc": "one shot", "audio_desc": 'Grandma: "I been playing Bid Whist for fifty years!"'}]
                return [
                    {"idx": 0, "is_last": False, "cam_idx": 0, "visual_desc": "hook", "audio_desc": 'Grandma: "I been playing Bid Whist for fifty years!" Family: "Try it on your phone!"'},
                    {"idx": 1, "is_last": True, "cam_idx": 0, "visual_desc": "joker", "audio_desc": 'Grandma: "Ooooh! I got the Big Joker!"'},
                ]

            pipeline.design_storyboard = fake_design
            events = []
            final = await pipeline._design_storyboard_with_review(ENGLISH_SCRIPT, [], "", True, lambda stage, message, metadata=None: events.append(stage))
            self.assertEqual(len(final), 2)
            self.assertEqual(len(attempts), 2)  # one bounce
            self.assertIn("Script fidelity feedback", attempts[1])
            self.assertIn("Try it on your phone!", attempts[1])  # exact missing text in the fix
            self.assertIn("review_storyboard_fidelity", events)

    async def test_unfixable_fidelity_raises_with_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline(
                chat_model=object(), image_generator=object(), video_generator=object(),
                working_dir=tmp,
                render_geometry={"video_seconds": 8.0, "max_total_video_seconds": 40.0},
                review_config={"enabled": False, "max_retries": 0},
            )

            async def always_broken(script, characters, user_requirement, quiet=False):
                return [{"idx": 0, "is_last": True, "cam_idx": 0, "visual_desc": "lost", "audio_desc": "no dialogue at all"}]

            pipeline.design_storyboard = always_broken
            with self.assertRaises(ValueError) as ctx:
                await pipeline._design_storyboard_with_review(ENGLISH_SCRIPT, [], "", True, None)
            self.assertIn("fidelity", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
