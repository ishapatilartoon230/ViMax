import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_runtime.prompts import PromptBuilder
from agent_runtime.session_index import SessionIndex
from agent_runtime.tools import build_builtin_registry
from agents.camera_image_generator import CameraImageGenerator
from agents.screenwriter import Screenwriter
from agents.storyboard_artist import StoryboardArtist
from pipelines.script2video_pipeline import Script2VideoPipeline
from utils.skills import StageSkills, load_skill


def _write_skills(root: Path, files: dict[str, str]) -> None:
    skills_dir = root / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (skills_dir / name).write_text(body, encoding="utf-8")


class SkillLoaderTests(unittest.TestCase):
    def test_missing_skills_directory_yields_empty_everywhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(load_skill(root, "shared"), "")
            self.assertEqual(StageSkills(root).stage("storyboard"), "")

    def test_stage_combines_shared_and_stage_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_skills(root, {
                "shared.md": "SHARED RULE: extract a fact sheet",
                "storyboard.md": "STORYBOARD RULE: state device orientation",
            })
            skills = StageSkills(root)
            block = skills.stage("storyboard")
            self.assertIn("SHARED RULE", block)
            self.assertIn("STORYBOARD RULE", block)
            self.assertLess(block.index("SHARED RULE"), block.index("STORYBOARD RULE"))
            self.assertNotIn("STORYBOARD RULE", skills.stage("decomposition"))
            self.assertIn("SHARED RULE", skills.stage("decomposition"))

    def test_stage_skills_without_root_returns_empty(self):
        self.assertEqual(StageSkills(None).stage("storyboard"), "")
        self.assertEqual(StageSkills(None).stages("screenwriter", "script_engagement"), "")

    def test_stages_combines_shared_once_and_multiple_stages_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_skills(root, {
                "shared.md": "SHARED RULE",
                "screenwriter.md": "SCREENWRITER RULE",
                "script_engagement.md": "ENGAGEMENT RULE",
            })
            block = StageSkills(root).stages("screenwriter", "script_engagement")
            self.assertIn("SHARED RULE", block)
            self.assertIn("SCREENWRITER RULE", block)
            self.assertIn("ENGAGEMENT RULE", block)
            self.assertEqual(block.count("SHARED RULE"), 1)
            self.assertLess(block.index("SHARED RULE"), block.index("SCREENWRITER RULE"))
            self.assertLess(block.index("SCREENWRITER RULE"), block.index("ENGAGEMENT RULE"))
            # A duplicate stage name must not duplicate the block either.
            duplicated = StageSkills(root).stages("screenwriter", "screenwriter")
            self.assertEqual(duplicated.count("SCREENWRITER RULE"), 1)

    def test_stages_truncates_at_stage_cap(self):
        from utils.skills import MAX_STAGE_CHARS
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_skills(root, {
                "shared.md": "S" * 5000,
                "screenwriter.md": "W" * 4000,
                "script_engagement.md": "E" * 4000,
                "clip_contract.md": "C" * 4000,
            })
            block = StageSkills(root).stages("screenwriter", "script_engagement", "clip_contract")
            self.assertLessEqual(len(block), MAX_STAGE_CHARS + 200)
            self.assertIn("[skills truncated]", block)

    def test_real_repo_engagement_combos_stay_under_stage_cap(self):
        """The shipped skill files must not truncate when combined."""
        from utils.skills import MAX_STAGE_CHARS
        repo_root = Path(__file__).resolve().parent.parent
        skills = StageSkills(repo_root)
        for combo in (
            ("screenwriter", "script_engagement", "clip_contract"),
            ("script_planning", "script_engagement", "clip_contract"),
            ("script_enhancement", "script_engagement", "clip_contract"),
            ("novel_adaptation", "script_engagement"),
            ("storyboard",),
        ):
            block = skills.stages(*combo)
            self.assertLessEqual(len(block), MAX_STAGE_CHARS, combo)
            self.assertNotIn("[skills truncated]", block, combo)

    def test_clip_contract_stage_is_registered(self):
        from utils.skills import STAGES
        self.assertIn("clip_contract", STAGES)

    def test_every_shipped_stage_file_fits_the_file_cap(self):
        """No skill file may exceed MAX_SKILL_CHARS — the loader would
        silently drop its tail (the shared.md/storyboard.md bug)."""
        from utils.skills import MAX_SKILL_CHARS, STAGES
        skills_dir = Path(__file__).resolve().parent.parent / "skills"
        for stage in STAGES:
            path = skills_dir / f"{stage}.md"
            if not path.exists():
                continue
            with self.subTest(stage=stage):
                self.assertLessEqual(
                    path.read_text(encoding="utf-8").strip().__len__(), MAX_SKILL_CHARS,
                    f"{stage}.md exceeds the loader cap and is being truncated",
                )

    def test_stages_excludes_shared_when_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_skills(root, {"shared.md": "SHARED RULE", "screenwriter.md": "SCREENWRITER RULE"})
            block = StageSkills(root).stages("screenwriter", include_shared=False)
            self.assertNotIn("SHARED RULE", block)
            self.assertIn("SCREENWRITER RULE", block)

    def test_script_engagement_stage_is_registered(self):
        from utils.skills import STAGES
        self.assertIn("script_engagement", STAGES)

    def test_per_file_size_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_skills(root, {"shared.md": "x" * 10000})
            self.assertLessEqual(len(load_skill(root, "shared")), 4200)
            self.assertIn("[skill truncated]", load_skill(root, "shared"))


class AgentThreadingTests(unittest.TestCase):
    def test_script2video_pipeline_threads_skills_into_sub_agents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_skills(root, {
                "shared.md": "SHARED RULE",
                "storyboard.md": "STORYBOARD RULE",
                "decomposition.md": "DECOMPOSITION RULE",
                "frame_prompts.md": "FRAME RULE",
                "image_selection.md": "SELECTION RULE",
                "character_extraction.md": "EXTRACTION RULE",
                "character_portraits.md": "PORTRAIT RULE",
                "reference_selection.md": "REFERENCE RULE",
                "novel_adaptation.md": "NOVEL RULE",
            })
            pipeline = Script2VideoPipeline(
                chat_model=object(), image_generator=object(), video_generator=object(),
                working_dir=str(root / "work"), skills=StageSkills(root),
            )
            self.assertIn("STORYBOARD RULE", pipeline.storyboard_artist.storyboard_skills)
            self.assertIn("DECOMPOSITION RULE", pipeline.storyboard_artist.decomposition_skills)
            self.assertIn("FRAME RULE", pipeline.camera_image_generator.frame_prompt_skills)
            self.assertIn("SELECTION RULE", pipeline.best_image_selector.selection_skills)
            self.assertIn("EXTRACTION RULE", pipeline.character_extractor.extraction_skills)
            self.assertIn("PORTRAIT RULE", pipeline.character_portraits_generator.portrait_skills)
            self.assertIn("REFERENCE RULE", pipeline.reference_image_selector.reference_skills)

    def test_novel_builders_pass_novel_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_skills(root, {"shared.md": "SHARED RULE", "novel_adaptation.md": "NOVEL RULE"})
            from agent_runtime import vimax_adapters as va

            with patch.object(va, "llm_api_key", return_value="key"), \
                 patch.object(va, "llm_base_url", return_value="https://example.invalid/v1"), \
                 patch.object(va, "llm_model", return_value="model"), \
                 patch.object(va, "_build_embedding_model", return_value=object()), \
                 patch.object(va, "_build_reranker", return_value=object()):
                novel_pipeline = va._build_novel_pipeline(root / "novel", skills=StageSkills(root))
            self.assertIn("NOVEL RULE", novel_pipeline.novel_compressor.novel_skills)
            self.assertIn("NOVEL RULE", novel_pipeline.event_extractor.novel_skills)
            self.assertIn("NOVEL RULE", novel_pipeline.scene_extractor.novel_skills)
            self.assertIn("NOVEL RULE", novel_pipeline.global_information_planner.novel_skills)

    def test_pipeline_without_skills_keeps_empty_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline(
                chat_model=object(), image_generator=object(), video_generator=object(),
                working_dir=str(Path(tmp) / "work"),
            )
            self.assertEqual(pipeline.storyboard_artist.storyboard_skills, "")
            self.assertEqual(pipeline.camera_image_generator.frame_prompt_skills, "")

    def test_skill_suffixes_stay_empty_when_no_skills(self):
        artist = StoryboardArtist(chat_model=object())
        self.assertEqual(artist._storyboard_skill_suffix(), "")
        artist = StoryboardArtist(chat_model=object(), storyboard_skills="RULE")
        self.assertTrue(artist._storyboard_skill_suffix().endswith("RULE"))

    def test_frame_prompt_prefix_includes_skills(self):
        generator = CameraImageGenerator(
            chat_model=object(), image_generator=object(), video_generator=object(),
            frame_prompt_skills="SCREEN FACES CAMERA",
        )
        self.assertTrue(generator._skill_prefix().startswith("SCREEN FACES CAMERA"))
        self.assertEqual(CameraImageGenerator(chat_model=object(), image_generator=object(), video_generator=object())._skill_prefix(), "")

    def test_screenwriter_carries_skills(self):
        writer = Screenwriter(chat_model=object(), screenwriter_skills="HOOK EARLY")
        self.assertEqual(writer._skill_suffix().strip(), "HOOK EARLY")


class AgentPromptInjectionTests(unittest.TestCase):
    def test_prompt_builder_injects_agent_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "prompts").mkdir()
            (root / "prompts" / "agent.md").write_text("agent rules", encoding="utf-8")
            (root / "prompts" / "workflow.md").write_text("workflow rules", encoding="utf-8")
            _write_skills(root, {"shared.md": "SHARED RULE", "agent.md": "AGENT SKILL RULE"})
            index = SessionIndex(root)
            index.create(idea="cat")
            registry = build_builtin_registry(root, index)
            builder = PromptBuilder(root / "prompts", index, registry)
            system = builder.build_messages("start")[0]["content"]
            self.assertIn("Project Skills", system)
            self.assertIn("AGENT SKILL RULE", system)
            self.assertIn("SHARED RULE", system)

    def test_prompt_builder_without_skills_omits_part_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "prompts").mkdir()
            (root / "prompts" / "agent.md").write_text("agent rules", encoding="utf-8")
            (root / "prompts" / "workflow.md").write_text("workflow rules", encoding="utf-8")
            index = SessionIndex(root)
            index.create(idea="cat")
            registry = build_builtin_registry(root, index)
            builder = PromptBuilder(root / "prompts", index, registry)
            parts = {part.id: part for part in builder.build_parts("start")}
            self.assertEqual(parts["skills.agent"].body, "")


if __name__ == "__main__":
    unittest.main()
