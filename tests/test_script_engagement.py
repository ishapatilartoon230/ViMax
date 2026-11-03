import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pipelines.idea2video_pipeline import Idea2VideoPipeline
from utils.script_engagement import (
    MAX_BOUNCES,
    engagement_error_message,
    engagement_report,
)


BAD_SCENE = 'Ana looks at the dashboard. Ana says: "We will leverage our seamless platform for growth."'
GOOD_SCENE = 'Ana looks at the dashboard. Ana says: "We ship the update tonight. Ten minutes."'


class EngagementReportTests(unittest.TestCase):
    def test_banned_word_in_quoted_dialogue_is_hard_failure(self):
        report = engagement_report(BAD_SCENE)
        # One hard entry per offending line; the banned words are inside it.
        self.assertEqual(len(report["hard_failures"]), 1)
        self.assertIn("leverage", " ".join(report["hard_failures"]))
        self.assertIn("seamless", " ".join(report["hard_failures"]))
        self.assertEqual(report["dialogue_lines"], 1)

    def test_banned_word_in_action_text_only_is_ignored(self):
        report = engagement_report("The leverage of the machine is seamless and robust.")
        self.assertEqual(report["hard_failures"], [])
        self.assertEqual(report["dialogue_lines"], 0)

    def test_clean_dialogue_passes(self):
        report = engagement_report(GOOD_SCENE)
        self.assertEqual(report["hard_failures"], [])
        self.assertEqual(report["score"], 100)
        self.assertIsNone(engagement_error_message(report))

    def test_not_just_x_but_y_is_hard_failure(self):
        report = engagement_report('He says: "It is not just a car but a statement."')
        self.assertTrue(report["hard_failures"])

    def test_its_not_x_its_y_is_hard_failure(self):
        report = engagement_report('She says: "It\'s not about the money, it\'s about respect."')
        self.assertTrue(report["hard_failures"])

    def test_fake_suspense_and_stage_directions_are_hard_failures(self):
        for line in (
            'He says: "Spoiler: the butler did it."',
            'She says: "Here\'s where it gets interesting."',
            'He says: "Here\'s the thing, we have no time."',
            'She says: "In today\'s world, nothing is free."',
        ):
            report = engagement_report(line)
            self.assertTrue(report["hard_failures"], line)

    def test_no_dialogue_passes_trivially(self):
        report = engagement_report("Wide shot of the station. The train leaves.")
        self.assertEqual(report["hard_failures"], [])
        self.assertEqual(report["dialogue_lines"], 0)
        self.assertEqual(report["score"], 100)

    def test_natural_one_word_replies_are_not_flagged(self):
        script = 'He says: "No." She says: "Wait." He says: "What?"'
        report = engagement_report(script)
        self.assertEqual(report["soft_flags"], [])

    def test_dramatic_one_word_lines_are_soft_flag(self):
        script = 'He says: "Silence." She says: "Gone." He says: "Oblivion."'
        report = engagement_report(script)
        self.assertEqual(report["hard_failures"], [])
        self.assertTrue(any("one-word" in flag for flag in report["soft_flags"]))
        self.assertIsNone(engagement_error_message(report))

    def test_excessive_em_dashes_are_soft_flag(self):
        script = 'She says: "I told you—twice—he left—and the door—was open—again."'
        report = engagement_report(script)
        self.assertEqual(report["hard_failures"], [])
        self.assertTrue(any("em dash" in flag for flag in report["soft_flags"]))

    def test_non_english_dialogue_is_unaffected(self):
        report = engagement_report('वह कहता है: "हम यह सब आज ही पूरा करेंगे।"')
        self.assertEqual(report["hard_failures"], [])
        self.assertEqual(report["score"], 100)

    def test_score_math(self):
        script = (
            'He says: "This will leverage our growth." '
            'She says: "Silence." He says: "Gone." She says: "Oblivion."'
        )
        report = engagement_report(script)
        # 1 hard (leverage) + 1 soft (3 dramatic one-word lines) -> 100 - 15 - 5
        self.assertEqual(report["score"], 100 - 15 - 5)

    def test_error_message_names_the_offending_line(self):
        report = engagement_report(BAD_SCENE)
        message = engagement_error_message(report)
        self.assertIsNotNone(message)
        self.assertIn("leverage", message)
        self.assertIn("Rewrite", message)


class FakeScreenwriter:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def write_script_based_on_story(self, story, user_requirement=None, timing_contract=""):
        self.calls.append({"story": story, "user_requirement": user_requirement})
        return self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]


class Idea2VideoEngagementBounceTests(unittest.IsolatedAsyncioTestCase):
    def _pipeline(self, tmp, screenwriter):
        pipeline = Idea2VideoPipeline(
            chat_model=object(), image_generator=object(), video_generator=object(),
            working_dir=str(Path(tmp) / "work"),
        )
        pipeline.screenwriter = screenwriter
        return pipeline

    async def test_hard_failure_triggers_rewrite_bounce_and_passes(self):
        writer = FakeScreenwriter([[BAD_SCENE], [GOOD_SCENE]])
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp, writer)
            script = await pipeline.write_script_based_on_story(story="story", user_requirement="")
        self.assertEqual(script, [GOOD_SCENE])
        self.assertEqual(len(writer.calls), 2)
        self.assertIn("Dialogue engagement feedback", writer.calls[1]["user_requirement"])

    async def test_clean_script_needs_no_bounce(self):
        writer = FakeScreenwriter([[GOOD_SCENE]])
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp, writer)
            script = await pipeline.write_script_based_on_story(story="story", user_requirement="")
        self.assertEqual(script, [GOOD_SCENE])
        self.assertEqual(len(writer.calls), 1)

    async def test_persists_bounced_script_and_proceeds_after_exhaustion(self):
        writer = FakeScreenwriter([[BAD_SCENE]])  # never improves
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp, writer)
            script = await pipeline.write_script_based_on_story(story="story", user_requirement="")
            saved = json.loads((Path(tmp) / "work" / "script.json").read_text(encoding="utf-8"))
        self.assertEqual(script, [BAD_SCENE])
        self.assertEqual(saved, [BAD_SCENE])
        self.assertEqual(len(writer.calls), MAX_BOUNCES + 1)


class ScriptIntakeEngagementBounceTests(unittest.IsolatedAsyncioTestCase):
    async def test_intake_bounces_once_on_ai_tells(self):
        from agent_runtime import vimax_adapters as va

        events = []

        class FakePlanner:
            def __init__(self, **kwargs):
                pass

            def plan_script(self, script):
                return SimpleNamespace(planned_script="planned draft, no dialogue quotes")

        class FakeEnhancer:
            def __init__(self):
                self.calls = []

            async def enhance_script(self, planned_script):
                self.calls.append(planned_script)
                if len(self.calls) == 1:
                    return SimpleNamespace(enhanced_script=BAD_SCENE)
                return SimpleNamespace(enhanced_script=GOOD_SCENE)

        enhancer = FakeEnhancer()
        with tempfile.TemporaryDirectory() as tmp:
            adapters = va.ViMaxAdapters(Path(tmp), None)
            with patch.object(va, "llm_api_key", return_value="key"), \
                 patch.object(va, "llm_model", return_value="model"), \
                 patch.object(va, "llm_base_url", return_value="https://example.invalid/v1"), \
                 patch.object(va, "ScriptPlanner", FakePlanner), \
                 patch.object(va, "ScriptEnhancer", lambda **kwargs: enhancer):
                runtime = SimpleNamespace(emit_progress=lambda message="", stage="", **meta: events.append(stage))
                result = await adapters._run_script_intake("unstructured script with no quotes", runtime, "session-1")

        self.assertEqual(result, GOOD_SCENE)
        self.assertEqual(len(enhancer.calls), 2)
        self.assertIn("Dialogue engagement feedback", enhancer.calls[1])
        self.assertIn("script_intake_engagement_bounce", events)

    async def test_intake_keeps_fidelity_as_final_authority(self):
        from agent_runtime import vimax_adapters as va

        original = 'He says: "Stay here." She says: "I will wait for you outside."'
        # Engagement-clean (no bounce) but drops She's quoted line entirely.
        line_dropping_enhanced = 'He says: "Stay here."'

        class FakePlanner:
            def __init__(self, **kwargs):
                pass

            def plan_script(self, script):
                return SimpleNamespace(planned_script="planned")

        class FakeEnhancer:
            async def enhance_script(self, planned_script):
                return SimpleNamespace(enhanced_script=line_dropping_enhanced)

        with tempfile.TemporaryDirectory() as tmp:
            adapters = va.ViMaxAdapters(Path(tmp), None)
            with patch.object(va, "llm_api_key", return_value="key"), \
                 patch.object(va, "llm_model", return_value="model"), \
                 patch.object(va, "llm_base_url", return_value="https://example.invalid/v1"), \
                 patch.object(va, "ScriptPlanner", FakePlanner), \
                 patch.object(va, "ScriptEnhancer", lambda **kwargs: FakeEnhancer()):
                result = await adapters._run_script_intake(original, None, "session-2")

        # No engagement hard failures, but the verbatim fidelity check still
        # rejects the plan because a quoted line was dropped -> original kept.
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
