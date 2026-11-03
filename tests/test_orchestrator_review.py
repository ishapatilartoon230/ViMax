import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents.orchestrator import OrchestratorReviewer, ReviewVerdict
from pipelines.script2video_pipeline import Script2VideoPipeline
from utils.review_config import review_config_from_workspace
from utils.robust_json_parser import parse_json


class FakeChatModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.replies.pop(0))


class ParseJsonTests(unittest.TestCase):
    def test_plain_code_fence_and_trailing_comma(self):
        self.assertEqual(parse_json('{"verdict": "pass"}')["verdict"], "pass")
        self.assertEqual(parse_json('```json\n{"verdict": "fail",}\n```')["verdict"], "fail")
        self.assertEqual(parse_json('Sure!\n{"verdict": "pass", "stage": "x",}')['stage'], "x")
        self.assertIsNone(parse_json("no json here at all"))


class OrchestratorReviewerTests(unittest.IsolatedAsyncioTestCase):
    async def test_parses_fail_verdict_with_fix(self):
        model = FakeChatModel(['```json\n{"verdict": "fail", "stage": "storyboard", "target_agent": "storyboard_artist", "issues": ["shot 3 dropped"], "fix_instruction": "restore shot 3"}\n```'])
        reviewer = OrchestratorReviewer(chat_model=model, review_skills="SKILL RULE")
        verdict = await reviewer.review("storyboard", "user script", [{"idx": 0}])
        self.assertEqual(verdict.verdict, "fail")
        self.assertEqual(verdict.target_agent, "storyboard_artist")
        self.assertEqual(verdict.fix_instruction, "restore shot 3")
        system_text = model.calls[0][0][1]
        self.assertIn("SKILL RULE", system_text)
        self.assertIn("user script", model.calls[0][1][1])

    async def test_pass_verdict_clears_fix_fields(self):
        model = FakeChatModel(['{"verdict": "pass", "stage": "characters", "issues": ["ignored"], "fix_instruction": "ignored"}'])
        reviewer = OrchestratorReviewer(chat_model=model)
        verdict = await reviewer.review("characters", "script", [])
        self.assertEqual(verdict.verdict, "pass")
        self.assertEqual(verdict.issues, [])

    async def test_unparseable_review_passes_without_blocking(self):
        model = FakeChatModel(["the storyboard looks fine to me!"])
        reviewer = OrchestratorReviewer(chat_model=model)
        verdict = await reviewer.review("storyboard", "script", [])
        self.assertEqual(verdict.verdict, "pass")


class ReviewConfigTests(unittest.TestCase):
    def test_defaults_and_env(self):
        with patch("utils.review_config.os.environ", {}):
            config = review_config_from_workspace(tempfile.mkdtemp())
        self.assertTrue(config["enabled"])
        self.assertEqual(config["max_retries"], 2)
        with patch("utils.review_config.os.environ", {"VIMAX_REVIEW_ENABLED": "false", "VIMAX_REVIEW_MAX_RETRIES": "3"}):
            config = review_config_from_workspace(tempfile.mkdtemp())
        self.assertFalse(config["enabled"])
        self.assertEqual(config["max_retries"], 3)

    def test_yaml_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            (config_dir / "agent.local.yaml").write_text("review:\n  enabled: false\n  max_retries: 1\n  model: gemini-flash-lite\n", encoding="utf-8")
            from utils.review_config import _load_review_section_cached

            _load_review_section_cached.cache_clear()
            with patch("utils.review_config.os.environ", {}):
                config = review_config_from_workspace(tmp)
            self.assertFalse(config["enabled"])
            self.assertEqual(config["max_retries"], 1)
            self.assertEqual(config["model"], "gemini-flash-lite")

    def test_model_from_env(self):
        with patch("utils.review_config.os.environ", {"VIMAX_REVIEW_MODEL": "cheap-model"}):
            config = review_config_from_workspace(tempfile.mkdtemp())
        self.assertEqual(config["model"], "cheap-model")


class ReviewBounceTests(unittest.IsolatedAsyncioTestCase):
    def _pipeline(self, tmp, **review):
        config = {"enabled": True, "max_retries": 2}
        config.update(review)
        pipeline = Script2VideoPipeline(
            chat_model=object(), image_generator=object(), video_generator=object(),
            working_dir=tmp, review_config=config,
        )
        return pipeline

    def test_reviewer_uses_dedicated_review_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            review_model = object()
            pipeline = Script2VideoPipeline(
                chat_model=object(), image_generator=object(), video_generator=object(),
                working_dir=tmp, review_config={"enabled": True}, review_chat_model=review_model,
            )
            self.assertIs(pipeline.reviewer.chat_model, review_model)
            fallback = Script2VideoPipeline(
                chat_model=object(), image_generator=object(), video_generator=object(),
                working_dir=tmp, review_config={"enabled": True},
            )
            self.assertIsNotNone(fallback.reviewer)

    async def test_bounce_sends_fix_and_stops_on_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            verdicts = [
                ReviewVerdict(verdict="fail", stage="storyboard", target_agent="storyboard_artist", issues=["bad"], fix_instruction="restore shot 2"),
                ReviewVerdict(verdict="pass", stage="storyboard"),
            ]
            pipeline.reviewer = SimpleNamespace(review=lambda **kwargs: _araise(verdicts.pop(0)))

            async def bounce(fix):
                self.assertEqual(fix, "restore shot 2")
                return {"fixed": True}

            events, final = await pipeline._review_with_bounce(stage="storyboard", user_material="script", artifact={"v": 1}, serialize=lambda a: a, extra_context=None, bounce=bounce)
            self.assertEqual(final, {"fixed": True})
            self.assertEqual([event["verdict"] for event in events], ["fail", "pass"])

    async def test_gives_up_after_max_retries_and_keeps_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp, max_retries=1)

            async def always_fail(**kwargs):
                return ReviewVerdict(verdict="fail", stage="storyboard", target_agent="storyboard_artist", issues=["bad"], fix_instruction="fix")

            pipeline.reviewer = SimpleNamespace(review=always_fail)
            bounces = []

            async def bounce(fix):
                bounces.append(fix)
                return {"attempt": len(bounces)}

            events, final = await pipeline._review_with_bounce(stage="storyboard", user_material="script", artifact={"v": 0}, serialize=lambda a: a, extra_context=None, bounce=bounce)
            self.assertEqual(len(bounces), 1)  # max_retries=1 -> one bounce, then give up
            self.assertTrue(events[-1].get("gave_up"))
            self.assertEqual(final, {"attempt": 1})

    async def test_disabled_review_never_calls_reviewer(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp, enabled=False)
            self.assertIsNone(pipeline.reviewer)
            events, final = await pipeline._review_with_bounce(stage="storyboard", user_material="s", artifact={"v": 1}, serialize=lambda a: a, extra_context=None, bounce=None)
            self.assertEqual(events, [])
            self.assertEqual(final, {"v": 1})

    async def test_review_returns_domain_objects_not_serialized_dicts(self):
        from interfaces.shot_description import ShotBriefDescription

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            storyboard = [ShotBriefDescription(idx=0, is_last=True, cam_idx=0, visual_desc="shot", audio_desc="", duration_seconds=3.0)]
            serialized = []
            pipeline.reviewer = SimpleNamespace(review=lambda **kwargs: _araise(ReviewVerdict(verdict="pass", stage="storyboard")))

            def capture_serialize(shots):
                payload = [shot.model_dump() if hasattr(shot, "model_dump") else shot for shot in shots]
                serialized.append(payload)
                return payload

            events, final = await pipeline._review_with_bounce(stage="storyboard", user_material="script", artifact=storyboard, serialize=capture_serialize, extra_context=None, bounce=None)
            self.assertIs(final[0], storyboard[0])  # domain object flows through untouched
            self.assertIsInstance(serialized[0][0], dict)  # reviewer saw the serialized form

    async def test_characters_review_bounces_with_feedback(self):
        from interfaces import CharacterInScene

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp)
            calls = []
            base_character = CharacterInScene(idx=0, identifier_in_scene="Grandma", is_visible=True, static_features="short curly hair, orange blouse", dynamic_features="smiles")

            async def fake_extract(script, quiet=False):
                calls.append(script)
                return [base_character]

            pipeline.extract_characters = fake_extract
            characters_path = Path(tmp) / "characters.json"
            characters_path.write_text("[]", encoding="utf-8")

            async def fail_then_pass(**kwargs):
                if len(calls) == 1:
                    return ReviewVerdict(verdict="fail", stage="characters", target_agent="character_extractor", issues=["vague features"], fix_instruction="add wardrobe details")
                return ReviewVerdict(verdict="pass", stage="characters")

            pipeline.reviewer = SimpleNamespace(review=fail_then_pass)
            characters = await pipeline._extract_characters_with_review("my script", quiet=True, progress=None)
            self.assertEqual(len(characters), 1)
            self.assertEqual(len(calls), 2)
            self.assertIn("Orchestrator review feedback", calls[1])
            self.assertIn("add wardrobe details", calls[1])
            self.assertFalse(characters_path.exists())  # bounce removed the cached file


async def _araise(value):
    return value


if __name__ == "__main__":
    unittest.main()
