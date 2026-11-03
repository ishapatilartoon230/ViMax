import os
import re
import shutil
import json
import logging
import asyncio
import time
import hashlib
from pathlib import Path
from uuid import uuid4
from typing import Any, Callable, Optional, Dict, List, Tuple, Literal, Type, TypeVar
from moviepy import VideoFileClip, concatenate_videoclips
from PIL import Image
from agents import *
from agents.best_image_selector import BestImageSelector
import yaml
from interfaces import *
from langchain.chat_models import init_chat_model
from tools.render_backend import RenderBackend
from utils.provider_presets import resolve_chat_model_config
from utils.skills import StageSkills
from utils.review_config import review_config_from_workspace
from utils.dialogue_fit import DialogueFitError, extend_over_budget_shots, dialogue_fit_error_message, dialogue_fit_report, estimate_speech_seconds, extract_dialogue_lines
from utils.script_fidelity import script_fidelity_error_message, script_fidelity_report
from utils.image_selection import DEFAULT_IMAGE_CANDIDATES, image_candidate_count_from_config, validate_image_candidate_count
from utils.render_geometry import normalize_render_geometry




TModel = TypeVar("TModel")


def _normalize_model_list(items: Any, model_cls: Type[TModel], field_name: str) -> List[TModel]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise TypeError(f"{field_name} must be a list, got {type(items).__name__}")
    normalized: List[TModel] = []
    for idx, item in enumerate(items):
        if isinstance(item, model_cls):
            normalized.append(item)
        elif isinstance(item, dict):
            normalized.append(model_cls.model_validate(item))
        else:
            raise TypeError(f"{field_name}[{idx}] must be {model_cls.__name__} or dict, got {type(item).__name__}")
    return normalized


def _group_shots_into_cameras(shot_descriptions: List[ShotDescription]) -> List[Camera]:
    cameras_by_idx: Dict[int, Camera] = {}
    for shot_description in shot_descriptions:
        camera = cameras_by_idx.get(shot_description.cam_idx)
        if camera is None:
            camera = Camera(idx=shot_description.cam_idx, active_shot_idxs=[])
            cameras_by_idx[shot_description.cam_idx] = camera
        camera.active_shot_idxs.append(shot_description.idx)
    return list(cameras_by_idx.values())

def _collect_priority_shot_idxs(camera_tree: List[Camera]) -> List[int]:
    """Shot indices that other cameras depend on."""
    return [camera.parent_shot_idx for camera in camera_tree if camera.parent_shot_idx is not None]


def _pipeline_print(quiet: bool, message: str) -> None:
    if not quiet:
        print(message)


def _emit_text_plan_progress(progress, stage: str, message: str, metadata: Dict[str, Any] | None = None) -> None:
    if progress is not None:
        progress(stage, message, metadata or {})


def _emit_render_progress(progress, stage: str, message: str, metadata: Dict[str, Any] | None = None) -> None:
    if progress is not None:
        progress(stage, message, metadata or {})


NO_TEXT_IMAGE_SUFFIX = "\nNo text, letters, numbers, captions, subtitles, logos, or watermarks anywhere in the image. Overlays and captions are added in post-production."
NO_TEXT_VIDEO_SUFFIX = "\nNo text, letters, numbers, captions, subtitles, logos, or watermarks anywhere in the video. Overlays and captions are added in post-production."
REALISM_IMAGE_SUFFIX = (
    "\nRender like real recorded footage of real people: natural human anatomy — exactly two hands and two arms per person, five fingers per hand, "
    "correct proportions, physically possible poses and grips. Objects are real: solid materials, believable weight, no floating, no morphing, "
    "no impossible geometry. A device's back cover is opaque — it never displays screen content; the screen exists only on the front face, "
    "and during any turn only the side actually facing the camera is visible. "
    "Natural smartphone-camera color grade: balanced contrast, natural saturation and true skin tones, soft realistic dynamic range — "
    "no heavy contrast, no oversaturation, no HDR pop, no cinematic teal-orange grading."
)
REALISM_VIDEO_SUFFIX = (
    "\nMove like real people on real footage: exactly two hands per person with five fingers each, natural body mechanics, no extra or duplicated limbs, "
    "no morphing faces or objects, nothing appears or vanishes. Objects keep their mass: props carried, turned, and set down follow real physics. "
    "Devices never rotate, flip, or change grip - a held device keeps its orientation for the entire clip; the back cover is opaque and never shows screen content. "
    "Natural smartphone-camera color grade throughout: balanced contrast, natural saturation, true skin tones, consistent exposure between shots — "
    "no heavy contrast, no oversaturation, no flickering brightness."
)


def _trim_video_to_seconds(path: str, seconds: float) -> bool:
    """Hard-trim a generated clip to the per-shot budget.

    Video models return clips at model-chosen lengths (typically ~8s) and treat
    requested durations as suggestions, so shot timing is enforced after the
    fact: ask the model for the budget in the prompt, then trim deterministically.
    Returns True when the file was re-encoded shorter.
    """
    try:
        clip = VideoFileClip(path)
    except Exception as exc:
        logging.warning("Could not open %s for trimming (%s); keeping original clip.", path, exc)
        return False
    try:
        if not clip.duration or clip.duration <= seconds + 0.05:
            return False
        trimmed = clip.subclipped(0, seconds)
        trimmed.write_videofile(path, codec="libx264", preset="veryfast", audio_codec="aac", logger=None)
        return True
    finally:
        clip.close()


def _scoped_progress(progress, **scope):
    if progress is None:
        return None

    def emit(stage: str, message: str, metadata: Dict[str, Any] | None = None) -> None:
        payload = dict(scope)
        payload.update(metadata or {})
        _emit_render_progress(progress, stage, message, payload)

    return emit


# Language that means a prop/device turns to show a different face mid-shot.
_ROTATION_HINT_RE = re.compile(
    r"rotat\w*|flip\w*\s+(?:the|it)\b|rear\s+(?:casing|cover)|back\s+(?:cover|casing)|screen\s+faces\s+(?:her|him|away)",
    re.IGNORECASE,
)


def _shot_texts(shot: Any) -> str:
    parts = []
    for attr in ("motion_desc", "visual_desc", "ff_desc", "lf_desc"):
        value = shot.get(attr) if isinstance(shot, dict) else getattr(shot, attr, None)
        if value:
            parts.append(str(value))
    return " ".join(parts)


def _requires_last_frame_anchor(shot: Any) -> bool:
    """A shot whose prop turns to a different face needs a last-frame anchor.

    Observed failure: a shot planned as "screen faces camera, then the phone
    rotates back so the rear casing faces the camera" was classified 'small'
    (similar camera framing), so no last-frame image was generated — anchored
    only to the lit-screen first frame, the video model never performed the
    rotation. Any rotation/facing-flip language in the shot text therefore
    demands a generated last frame (variation_type at least 'medium').
    """
    return bool(_ROTATION_HINT_RE.search(_shot_texts(shot)))


class Script2VideoPipeline:

    def __init__(
        self,
        chat_model: str,
        image_generator,
        video_generator,
        working_dir: str,
        num_image_candidates: int = DEFAULT_IMAGE_CANDIDATES,
        render_geometry: Optional[Dict[str, Any]] = None,
        skills: Optional["StageSkills"] = None,
        review_config: Optional[Dict[str, Any]] = None,
        review_chat_model: Optional[Any] = None,
    ):

        self.chat_model = chat_model
        self.image_generator = image_generator
        self.video_generator = video_generator
        self.num_image_candidates = validate_image_candidate_count(num_image_candidates)
        self.render_geometry = normalize_render_geometry(render_geometry)
        self.skills = skills
        resolved_review = dict(review_config or {"enabled": True, "max_retries": 2})
        self.review_enabled = bool(resolved_review.get("enabled", True))
        self.review_max_retries = max(0, min(5, int(resolved_review.get("max_retries", 2))))
        self.reviewer = None
        if self.review_enabled:
            from agents.orchestrator import OrchestratorReviewer

            self.reviewer = OrchestratorReviewer(chat_model=review_chat_model or chat_model, review_skills=self._stage_skills("orchestrator"))
        self.best_image_selector = BestImageSelector(chat_model=self.chat_model, selection_skills=self._stage_skills("image_selection"))

        self.character_extractor = CharacterExtractor(chat_model=self.chat_model, extraction_skills=self._stage_skills("character_extraction"))
        self.character_portraits_generator = CharacterPortraitsGenerator(image_generator=self.image_generator, portrait_skills=self._stage_skills("character_portraits"))
        self.storyboard_artist = StoryboardArtist(chat_model=self.chat_model, storyboard_skills=self._stage_skills("storyboard"), decomposition_skills=self._stage_skills("decomposition"))
        self.camera_image_generator = CameraImageGenerator(chat_model=self.chat_model, image_generator=self.image_generator, video_generator=self.video_generator, render_geometry=self.render_geometry, frame_prompt_skills=self._stage_skills("frame_prompts"))
        self.reference_image_selector = ReferenceImageSelector(chat_model=self.chat_model, reference_skills=self._stage_skills("reference_selection"))

        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)
        self.character_portrait_events = {}
        self.shot_desc_events = {}
        self.frame_events = {}

    async def _review_with_bounce(self, *, stage: str, user_material: str, artifact: Any, serialize, extra_context: Optional[str], bounce):
        """Run orchestrator review, bouncing failed work back to its agent.

        ``artifact`` is the domain object the pipeline keeps working with;
        ``serialize(artifact)`` produces what the reviewer reads (plain JSON),
        and ``bounce(fix_instruction)`` regenerates the artifact with the fix
        applied. Returns ``(review_events, final_artifact)`` — the final
        artifact is always the domain object, never the serialized form.
        After ``max_retries`` failed bounces the last artifact is kept and the
        final event is marked ``gave_up`` so callers can warn instead of blocking.
        """
        events: List[Dict[str, Any]] = []
        if self.reviewer is None:
            return events, artifact
        current = artifact
        for attempt in range(self.review_max_retries + 1):
            try:
                verdict = await self.reviewer.review(stage=stage, user_material=user_material, artifact=serialize(current), extra_context=extra_context)
            except Exception as exc:  # a broken reviewer must never block the pipeline
                logging.warning("Orchestrator review failed at %s (%s); proceeding", stage, exc)
                return events, current
            events.append({"attempt": attempt + 1, "verdict": verdict.verdict, "issues": verdict.issues, "target_agent": verdict.target_agent, "fix_instruction": verdict.fix_instruction})
            if verdict.verdict == "pass":
                return events, current
            if attempt < self.review_max_retries:
                current = await bounce(verdict.fix_instruction)
            else:
                events[-1]["gave_up"] = True
        return events, current

    def _emit_review_events(self, progress, stage: str, events: List[Dict[str, Any]]) -> None:
        for event in events:
            message = f"Review {stage}: {event['verdict'].upper()}"
            if event["verdict"] == "fail":
                message += f" -> {event['target_agent']}: {event['fix_instruction']}"
                if event.get("gave_up"):
                    message += " (retries exhausted; proceeding with warning)"
            _emit_render_progress(progress, f"review_{stage}", message, {"stage": stage, **event})

    async def _extract_characters_with_review(self, script: str, quiet: bool, progress) -> List[CharacterInScene]:
        characters = await self.extract_characters(script=script, quiet=quiet)
        if self.reviewer is None:
            return characters

        characters_path = os.path.join(self.working_dir, "characters.json")

        async def bounce(fix_instruction: str):
            if os.path.exists(characters_path):
                os.remove(characters_path)
            return await self.extract_characters(script=script + f"\n\n[Orchestrator review feedback — fix these issues: {fix_instruction}]", quiet=quiet)

        events, final = await self._review_with_bounce(
            stage="characters",
            user_material=script,
            artifact=characters,
            serialize=lambda items: [item.model_dump() if hasattr(item, "model_dump") else item for item in items],
            extra_context="Review only whether each character is drawable, complete, consistent, and faithful to the script's cast.",
            bounce=bounce,
        )
        self._emit_review_events(progress, "characters", events)
        return final

    async def _design_storyboard_with_review(self, script: str, characters: List[CharacterInScene], user_requirement: str, quiet: bool, progress) -> List[ShotBriefDescription]:
        storyboard = await self.design_storyboard(script=script, characters=characters, user_requirement=user_requirement, quiet=quiet)

        storyboard_path = os.path.join(self.working_dir, "storyboard.json")

        async def bounce(fix_instruction: str):
            if os.path.exists(storyboard_path):
                os.remove(storyboard_path)
            return await self.design_storyboard(script=script, characters=characters, user_requirement=user_requirement + f"\n\n[Script fidelity feedback — fix these issues: {fix_instruction}]", quiet=quiet)

        # Mechanical fidelity first: every dialogue line verbatim + shot count.
        # An LLM never judges this — the user's script is diffed against the plan.
        fidelity_events: List[Dict[str, Any]] = []
        for attempt in range(self.review_max_retries + 1):
            report = script_fidelity_report(script, storyboard)
            if report["all_present"]:
                break
            message = script_fidelity_error_message(report)
            fidelity_events.append({"attempt": attempt + 1, "verdict": "fail", "issues": report["missing_dialogue"][:8], "target_agent": "storyboard_artist", "fix_instruction": message or "", "fidelity": report})
            if attempt >= self.review_max_retries:
                fidelity_events[-1]["gave_up"] = True
                break
            storyboard = await bounce(message or "restore missing script content")
        if fidelity_events:
            _emit_render_progress(progress, "review_storyboard_fidelity", "Mechanical script fidelity check", {"stage": "storyboard_fidelity", "events": fidelity_events})
            final_report = script_fidelity_report(script, storyboard)
            if not final_report["all_present"]:
                raise ValueError(script_fidelity_error_message(final_report))

        if self.reviewer is None:
            return storyboard

        events, final = await self._review_with_bounce(
            stage="storyboard",
            user_material=script + (f"\n\nUser requirements: {user_requirement}" if user_requirement else ""),
            artifact=storyboard,
            serialize=lambda shots: [shot.model_dump() if hasattr(shot, "model_dump") else shot for shot in shots],
            extra_context=(
                f"Pipeline timing: default per-shot seconds {self._video_seconds():g}, "
                f"total cap {self._max_total_seconds():g}s; per-shot duration_seconds come only from the script. "
                f"Characters: {[character.identifier_in_scene for character in characters]}. "
                f"Mechanical dialogue fit (words->seconds vs budget): {self._dialogue_fit(storyboard)}. "
                f"Mechanical fidelity already verified (dialogue verbatim + shot count)."
            ),
            bounce=bounce,
        )
        self._emit_review_events(progress, "storyboard", events)
        return final

    def _stage_skills(self, stage: str) -> str:
        return self.skills.stage(stage) if self.skills is not None else ""

    def _image_size(self) -> str:
        return str(self.render_geometry["image_size"])

    def _image_aspect_ratio(self) -> str:
        return str(self.render_geometry["image_aspect_ratio"])

    def _allow_portrait(self) -> bool:
        return bool(self.render_geometry["allow_portrait"])

    def _video_aspect_ratio(self) -> str:
        return str(self.render_geometry["video_aspect_ratio"])

    def _video_resolution(self) -> str:
        return str(self.render_geometry["video_resolution"])

    def _video_seconds(self) -> float:
        return float(self.render_geometry.get("video_seconds", 8.0))

    def _effective_shot_seconds(self, shot: Any) -> float:
        """Per-shot clip budget: the script's explicit timing when present,
        otherwise the pipeline-wide default (render.video_seconds). Never
        longer than max_clip_seconds — one generation call cannot return a
        longer clip, so a bigger budget would silently lose the tail."""
        if isinstance(shot, dict):
            value = shot.get("duration_seconds")
        else:
            value = getattr(shot, "duration_seconds", None)
        if isinstance(value, (int, float)) and value > 0:
            return min(float(value), self._max_clip_seconds())
        return min(self._video_seconds(), self._max_clip_seconds())

    def _max_clip_seconds(self) -> float:
        return float(self.render_geometry.get("max_clip_seconds", 10.0))

    def _max_total_seconds(self) -> float:
        return float(self.render_geometry.get("max_total_video_seconds", 40.0))

    def _dialogue_completion_window(self, audio_desc: Any, seconds: float) -> float:
        """When the dialogue must finish inside the clip.

        Adaptive: normally 85% of the clip (a reaction beat stays), but a
        tight shot - whose speech estimate nearly fills the budget - gets its
        full estimated speech time plus a small margin. Demanding 85% there
        guaranteed cut lines (the observed failure); the estimate is the
        honest minimum the model needs.
        """
        try:
            needed = estimate_speech_seconds(extract_dialogue_lines(str(audio_desc or "")))
        except Exception:
            needed = 0.0
        relaxed = min(seconds * 0.95, seconds - 0.2) if seconds > 1 else seconds
        return max(seconds * 0.85, min(relaxed, needed + 0.3)) if needed else seconds * 0.85

    def _duration_budget(self, shots: List[Any]) -> Dict[str, Any]:
        per_shot = [self._effective_shot_seconds(shot) for shot in shots]
        total = round(sum(per_shot), 2)
        maximum = self._max_total_seconds()
        dialogue = []
        for shot in shots:
            audio = shot.get("audio_desc") if isinstance(shot, dict) else getattr(shot, "audio_desc", "") or ""
            dialogue.append(estimate_speech_seconds(extract_dialogue_lines(str(audio or ""))))
        return {
            "shots": len(shots),
            "seconds_per_shot": self._video_seconds(),
            "per_shot_seconds": per_shot if len(per_shot) else [],
            "dialogue_needs_seconds": dialogue,
            "total_seconds": total,
            "max_total_seconds": maximum,
            "within_limit": total <= maximum + 1e-6,
        }

    def _validate_total_duration(self, shots: List[Any]) -> None:
        budget = self._duration_budget(shots)
        if not budget["within_limit"]:
            raise ValueError(
                f"Storyboard plans {budget['shots']} shots totalling {budget['total_seconds']:g}s "
                f"(per-shot: {budget['per_shot_seconds']}), which exceeds the "
                f"{budget['max_total_seconds']:g}s total video limit. Reduce the number of shots, "
                "their duration_seconds, or the default render.video_seconds."
            )

    def _dialogue_fit(self, shots: List[Any]) -> Dict[str, Any]:
        return dialogue_fit_report(
            shots,
            self._video_seconds(),
            overage_tolerance=self._dialogue_overage_tolerance(),
        )

    def _dialogue_overage_tolerance(self) -> float:
        return float(self.render_geometry.get("dialogue_overage_tolerance", 1.15))

    def _validate_dialogue_fit(self, shots: List[Any]) -> None:
        """Refuse plans whose dialogue cannot fit its shot budgets.

        Speech time is estimated mechanically (words -> seconds); an unfit
        shot would be trimmed mid-line, so the plan must come back to the
        user with a recommended duration instead of rendering cut dialogue.
        """
        message = dialogue_fit_error_message(self._dialogue_fit(shots), max_clip_seconds=self._max_clip_seconds())
        if message:
            raise DialogueFitError(message)

    def _adjust_dialogue_fit(self, storyboard: List[Any], progress, quiet: bool) -> List[Dict[str, Any]]:
        """Fit over-budget dialogue into the plan the user asked for.

        Mechanical and transparent: the dialogue is never cut or compressed.
        First the missing seconds are borrowed from dialogue-light shots in the
        same scene (the requested total is preserved); only when that cannot
        cover the overage does the total grow, and only up to the cap. The
        adjustment is announced as a progress event and persisted, so resume
        runs and the render gate read the same budgets. Returns the applied
        adjustments; empty when neither strategy was possible, in which case
        the subsequent validation raises with the split-lines recommendation.
        """
        # Extension-only, by user rule: dialogue is never cut or re-timed
        # elsewhere - a shot whose speech exceeds its budget simply grows to
        # fit (within the single-clip cap), even if the total video gets
        # longer than the script stated (20s -> 23s is fine).
        adjustments = extend_over_budget_shots(storyboard, self._video_seconds(), self._max_total_seconds(), per_shot_cap=self._max_clip_seconds())
        if not adjustments:
            return []
        storyboard_path = os.path.join(self.working_dir, "storyboard.json")
        try:
            with open(storyboard_path, "w", encoding="utf-8") as f:
                json.dump([shot.model_dump() if hasattr(shot, "model_dump") else shot for shot in storyboard], f, ensure_ascii=False, indent=4)
        except OSError as exc:
            logging.warning("Could not persist dialogue-fit adjustments to %s (%s)", storyboard_path, exc)
        for item in adjustments:
            shot_description_path = os.path.join(self.working_dir, "shots", str(item["shot"]), "shot_description.json")
            if not os.path.exists(shot_description_path):
                continue
            try:
                with open(shot_description_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                payload["duration_seconds"] = item["to_seconds"]
                with open(shot_description_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=4)
            except (OSError, json.JSONDecodeError) as exc:
                logging.warning("Could not update %s after dialogue-fit adjustment (%s)", shot_description_path, exc)
        parts = []
        for item in adjustments:
            parts.append(f"shot {item['shot']} {item['from_seconds']:g}s -> {item['to_seconds']:g}s (dialogue needs {item['estimated_speech_seconds']:g}s; extended to fit, never cut)")
        message = "Extended shots to fit their dialogue: " + ", ".join(parts)
        _pipeline_print(quiet, f"⏱️ {message}.")
        _emit_text_plan_progress(progress, "dialogue_fit_extended", message, {"adjustments": adjustments})
        return adjustments


    async def plan_text_artifacts(
        self,
        script: str,
        user_requirement: str,
        style: str,
        characters: List[CharacterInScene] = None,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        quiet: bool = False,
    ):
        """Generate only structured text artifacts required before rendering.

        This helper intentionally stops before character portraits, frame generation,
        video generation, and final concatenation so an agent loop can pause for
        user review after narrative planning.
        """
        self.character_portrait_events = {}
        self.shot_desc_events = {}
        self.frame_events = {}

        if characters is None:
            _emit_text_plan_progress(progress, "extract_characters", "Extracting characters from script")
            characters = await self._extract_characters_with_review(script=script, quiet=quiet, progress=progress)
        else:
            characters = _normalize_model_list(characters, CharacterInScene, "characters")
            _emit_text_plan_progress(progress, "extract_characters", "Using provided characters", {"provided": True, "count": len(characters)})
            characters_path = os.path.join(self.working_dir, "characters.json")
            if not os.path.exists(characters_path):
                with open(characters_path, "w", encoding="utf-8") as f:
                    json.dump([character.model_dump() for character in characters], f, ensure_ascii=False, indent=4)
            for character in characters:
                self.character_portrait_events[character.idx] = asyncio.Event()

        _emit_text_plan_progress(progress, "design_storyboard", "Designing storyboard")
        storyboard = await self._design_storyboard_with_review(
            script=script,
            characters=characters,
            user_requirement=user_requirement,
            quiet=quiet,
            progress=progress,
        )
        self._validate_total_duration(storyboard)
        # Mechanical dialogue-fit extension before the hard gate: a shot whose
        # lines cannot fit its budget is extended to the recommended duration
        # (within the total cap) instead of failing the whole plan. Only when
        # extension is impossible does the gate below refuse with the
        # recommendation, so the user decides how to restructure the scene.
        self._adjust_dialogue_fit(storyboard, progress, quiet=quiet)
        self._validate_total_duration(storyboard)
        _emit_text_plan_progress(progress, "duration_budget", "Duration budget", {"duration": self._duration_budget(storyboard), "dialogue_fit": self._dialogue_fit(storyboard)})
        self._validate_dialogue_fit(storyboard)
        _emit_text_plan_progress(progress, "decompose_shots", "Decomposing shot visual descriptions", {"shot_count": len(storyboard)})
        shot_descriptions = await self.decompose_visual_descriptions(
            shot_brief_descriptions=storyboard,
            characters=characters,
            quiet=quiet,
        )
        camera_tree = None
        for attempt in range(2):
            try:
                stage = "construct_camera_tree" if attempt == 0 else "construct_camera_tree_retry"
                message = "Constructing camera tree" if attempt == 0 else "Retrying camera tree construction after schema/type failure"
                _emit_text_plan_progress(progress, stage, message, {"shot_count": len(shot_descriptions), "attempt": attempt + 1})
                camera_tree = await self.construct_camera_tree(
                    shot_descriptions=shot_descriptions,
                    quiet=quiet,
                )
                break
            except Exception:
                camera_tree_path = os.path.join(self.working_dir, "camera_tree.json")
                if os.path.exists(camera_tree_path):
                    os.remove(camera_tree_path)
                if attempt == 1:
                    raise
        assert camera_tree is not None
        return {
            "characters": characters,
            "storyboard": storyboard,
            "shot_descriptions": shot_descriptions,
            "camera_tree": camera_tree,
        }


    @classmethod
    def init_from_config(cls, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        chat_model_args = resolve_chat_model_config(config["chat_model"]["init_args"])
        chat_model = init_chat_model(**chat_model_args)
        backend = RenderBackend.from_config(config)

        review_cfg = review_config_from_workspace(Path(config_path).resolve().parent.parent)
        review_chat = None
        if review_cfg.get("model"):
            review_args = resolve_chat_model_config(dict(config["chat_model"]["init_args"]))
            review_args["model"] = review_cfg["model"]
            review_chat = init_chat_model(**review_args)
        return cls(
            chat_model=chat_model,
            image_generator=backend.image_generator,
            video_generator=backend.video_generator,
            working_dir=config["working_dir"],
            num_image_candidates=image_candidate_count_from_config(config),
            render_geometry=config.get("render") if isinstance(config.get("render"), dict) else None,
            skills=StageSkills(Path(config_path).resolve().parent.parent),
            review_config=review_cfg,
            review_chat_model=review_chat,
        )

    async def __call__(
        self,
        script: str,
        user_requirement: str,
        style: str,
        characters: List[CharacterInScene] = None,
        character_portraits_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
        quiet: bool = False,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
    ):
        _emit_render_progress(progress, "render_start", "Starting script2video render")
        if characters is None:
            _emit_render_progress(progress, "extract_characters", "Extracting characters before render")
            characters = await self._extract_characters_with_review(script=script, quiet=quiet, progress=progress)

            # characters_path = os.path.join(self.working_dir, "characters.json")
            # if os.path.exists(characters_path):
            #     with open(characters_path, "r", encoding="utf-8") as f:
            #         characters = [CharacterInScene.model_validate(c) for c in json.load(f)]
            #     print(f"🚀 Loaded {len(characters)} characters from existing file.")
            # else:
            #     print(f"🔍 Extracting characters from script...")
            #     characters = await self.extract_characters(script=script)
            #     with open(characters_path, "w", encoding="utf-8") as f:
            #         json.dump([c.model_dump() for c in characters], f, ensure_ascii=False, indent=4)
            #     print(f"☑️ Extracted {len(characters)} characters from script and saved to {characters_path}.")
        else:
            characters = _normalize_model_list(characters, CharacterInScene, "characters")
            _emit_render_progress(progress, "extract_characters", "Using provided characters for render", {"provided": True, "count": len(characters)})
            for character in characters:
                self.character_portrait_events[character.idx] = asyncio.Event()

        if character_portraits_registry is None:
            character_portraits_registry_path = os.path.join(self.working_dir, "character_portraits_registry.json")
            if os.path.exists(character_portraits_registry_path):
                with open(character_portraits_registry_path, "r", encoding="utf-8") as f:
                    character_portraits_registry = json.load(f)
                print(f"🚀 Loaded {len(character_portraits_registry)} character portraits from existing file.")
                _emit_render_progress(progress, "character_portraits_loaded", "Loaded existing character portraits", {"count": len(character_portraits_registry)})
            else:
                print(f"🔍 Generating character portraits...")
                _emit_render_progress(progress, "character_portraits_start", "Generating character portraits", {"character_count": len(characters)})
                character_portraits_registry = await self.generate_character_portraits(
                    characters=characters,
                    character_portraits_registry=None,
                    style=style,
                    progress=progress,
                )

                with open(character_portraits_registry_path, "w", encoding="utf-8") as f:
                    json.dump(character_portraits_registry, f, ensure_ascii=False, indent=4)
                print(f"☑️ Generated {len(character_portraits_registry)} character portraits and saved to {character_portraits_registry_path}.")
                _emit_render_progress(progress, "character_portraits_done", "Character portraits ready", {"count": len(character_portraits_registry)})



        # design shots
        _emit_render_progress(progress, "load_storyboard", "Loading or designing storyboard")
        storyboard = await self._design_storyboard_with_review(
            script=script,
            characters=characters,
            user_requirement=user_requirement,
            quiet=quiet,
            progress=progress,
        )
        self._validate_total_duration(storyboard)
        self._validate_dialogue_fit(storyboard)
        _emit_render_progress(progress, "storyboard_ready", "Storyboard ready", {"shot_count": len(storyboard), "duration_budget": self._duration_budget(storyboard), "dialogue_fit": self._dialogue_fit(storyboard)})

        # decompose visual descriptions of shots
        _emit_render_progress(progress, "load_shot_descriptions", "Loading or decomposing shot descriptions", {"shot_count": len(storyboard)})
        shot_descriptions = await self.decompose_visual_descriptions(
            shot_brief_descriptions=storyboard,
            characters=characters,
            quiet=quiet,
        )
        _emit_render_progress(progress, "shot_descriptions_ready", "Shot descriptions ready", {"shot_count": len(shot_descriptions)})

        # construct camera tree
        _emit_render_progress(progress, "load_camera_tree", "Loading or constructing camera tree", {"shot_count": len(shot_descriptions)})
        camera_tree = await self.construct_camera_tree(
            shot_descriptions=shot_descriptions,
            quiet=quiet,
        )
        _emit_render_progress(progress, "camera_tree_ready", "Camera tree ready", {"camera_count": len(camera_tree)})

        priority_shot_idxs = [camera.parent_cam_idx for camera in camera_tree if camera.parent_cam_idx is not None]
        _emit_render_progress(progress, "frames_start", "Generating frames for cameras", {"camera_count": len(camera_tree), "shot_count": len(shot_descriptions)})
        tasks = [
            self.generate_frames_for_single_camera(
                camera=camera,
                shot_descriptions=shot_descriptions,
                characters=characters,
                character_portraits_registry=character_portraits_registry,
                priority_shot_idxs=priority_shot_idxs,
                progress=progress,
            )
            for camera in camera_tree
        ]

        _emit_render_progress(progress, "video_clips_start", "Generating video clips for shots", {"shot_count": len(shot_descriptions)})
        if bool(self.render_geometry.get("chain_frames", True)):
            # Chained mode: finish all frame generation first (per-camera,
            # already chained), then generate the clips strictly in order so
            # every clip starts from the previous clip's real final frame.
            await asyncio.gather(*tasks)
            await self._generate_all_shot_videos(shot_descriptions, progress=progress, characters=characters, character_portraits_registry=character_portraits_registry)
        else:
            video_tasks = [
                self.generate_video_for_single_shot(
                    shot_description=shot_description,
                    progress=progress,
                )
                for shot_description in shot_descriptions
            ]
            tasks.extend(video_tasks)
            await asyncio.gather(*tasks)

        final_video_path = os.path.join(self.working_dir, "final_video.mp4")
        if os.path.exists(final_video_path):
            print(f"🚀 Skipped concatenating videos, already exists.")
            _emit_render_progress(progress, "final_video_exists", "Final video already exists", {"path": final_video_path})
        else:
            print(f"🎬 Starting concatenating videos...")
            _emit_render_progress(progress, "concat_start", "Concatenating video clips", {"shot_count": len(shot_descriptions)})
            video_clips = [
                VideoFileClip(os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "video.mp4"))
                for shot_description in shot_descriptions
            ]
            final_video = concatenate_videoclips(video_clips)
            final_video.write_videofile(final_video_path, codec="libx264", preset="medium")
            # Last line of defense: cached or off-budget clips must never
            # produce a final cut longer than the configured total limit.
            _trim_video_to_seconds(final_video_path, self._max_total_seconds())
            print(f"☑️ Concatenated videos, saved to {final_video_path}.")
            _emit_render_progress(progress, "concat_done", "Final video concatenated", {"path": final_video_path, "max_total_seconds": self._max_total_seconds()})

        _emit_render_progress(progress, "render_done", "Script2video render complete", {"final_video_path": final_video_path})
        return final_video_path


    async def generate_frames_for_single_camera(
        self,
        camera: Camera,
        shot_descriptions: List[ShotDescription],
        characters: List[CharacterInScene],
        character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
        priority_shot_idxs: List[int],
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
    ):
        # 1. generate the first_frame of the first shot of the camera
        first_shot_idx = camera.active_shot_idxs[0]
        first_shot_ff_path = os.path.join(self.working_dir, "shots", f"{first_shot_idx}", "first_frame.png")
        _emit_render_progress(progress, "camera_frames_start", f"Generating frames for camera {camera.idx}", {"camera_idx": camera.idx, "active_shot_idxs": camera.active_shot_idxs})

        if os.path.exists(first_shot_ff_path):
            print(f"🚀 Skipped generating first_frame for shot {first_shot_idx}, already exists.")
            self.frame_events[first_shot_idx]["first_frame"].set()
            _emit_render_progress(progress, "frame_exists", f"First frame for shot {first_shot_idx} already exists", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame", "path": first_shot_ff_path})

        else:
            print(f"🖼️ Starting first_frame generation for shot {first_shot_idx}...")
            _emit_render_progress(progress, "frame_start", f"Generating first frame for shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame"})
            available_image_path_and_text_pairs = []

            for character_idx in shot_descriptions[first_shot_idx].ff_vis_char_idxs:
                identifier_in_scene = characters[character_idx].identifier_in_scene
                registry_item = character_portraits_registry[identifier_in_scene]
                for view, item in registry_item.items():
                    available_image_path_and_text_pairs.append((item["path"], item["description"]))
            
            # generate the first_frame based on the shot_description.ff_desc
            if camera.parent_shot_idx is not None:
                # generate the first_frame based on the transition video
                parent_shot_idx = camera.parent_shot_idx
                await self.frame_events[parent_shot_idx]["first_frame"].wait()
                parent_shot_ff_path = os.path.join(self.working_dir, "shots", f"{parent_shot_idx}", "first_frame.png")
                transition_video_path = os.path.join(self.working_dir, "shots", f"{first_shot_idx}", f"transition_video_from_shot_{parent_shot_idx}.mp4")

                if os.path.exists(transition_video_path):
                    print(f"🚀 Skipped generating transition video for shot {first_shot_idx} from shot {parent_shot_idx}, already exists.")
                    _emit_render_progress(progress, "transition_video_exists", f"Transition video for shot {first_shot_idx} already exists", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "parent_shot_idx": parent_shot_idx, "path": transition_video_path})
                else:
                    print(f"🖼️ Starting transition video generation for shot {first_shot_idx} from shot {parent_shot_idx}...")
                    _emit_render_progress(progress, "transition_video_start", f"Generating transition video for shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "parent_shot_idx": parent_shot_idx})
                    transition_video_output = await self.camera_image_generator.generate_transition_video(
                        first_shot_visual_desc=shot_descriptions[parent_shot_idx].visual_desc,
                        second_shot_visual_desc=shot_descriptions[first_shot_idx].visual_desc,
                        first_shot_ff_path=parent_shot_ff_path,
                        progress=_scoped_progress(progress, camera_idx=camera.idx, shot_idx=first_shot_idx, parent_shot_idx=parent_shot_idx, artifact="transition_video"),
                    )
                    transition_video_output.save(transition_video_path)
                    print(f"☑️ Generated transition video for shot {first_shot_idx} from shot {parent_shot_idx}, saved to {transition_video_path}.")
                    _emit_render_progress(progress, "transition_video_done", f"Transition video for shot {first_shot_idx} generated", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "parent_shot_idx": parent_shot_idx, "path": transition_video_path})

                new_camera_image_path = os.path.join(self.working_dir, "shots", f"{first_shot_idx}", f"new_camera_{camera.idx}.png")
                if os.path.exists(new_camera_image_path):
                    print(f"🚀 Skipped generating new camera image for shot {first_shot_idx}, already exists.")
                    _emit_render_progress(progress, "new_camera_image_exists", f"New camera image for shot {first_shot_idx} already exists", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "path": new_camera_image_path})
                else:
                    print(f"🖼️ Starting new camera image generation for shot {first_shot_idx}...")
                    _emit_render_progress(progress, "new_camera_image_start", f"Extracting new camera image for shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx})
                    new_camera_image = self.camera_image_generator.get_new_camera_image(transition_video_path)
                    new_camera_image.save(new_camera_image_path)
                    print(f"☑️ Generated new camera image for shot {first_shot_idx} (not completed), saved to {new_camera_image_path}.")
                    _emit_render_progress(progress, "new_camera_image_done", f"New camera image for shot {first_shot_idx} extracted", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "path": new_camera_image_path})

                available_image_path_and_text_pairs.append(
                    (
                        new_camera_image_path,
                        f"The composition and background are correct but some elements may be wrong. The wrong elements should be replaced.\nWrong elements: {camera.missing_info}.\nYou must select this image as the main reference and replace the characters in the image with the provided character portraits. Don't change the background."
                    )
                )


            # 如果子镜头缺少信息，则需要选择参考图像生成
            if camera.parent_shot_idx is None or camera.missing_info is not None:
                ff_selector_output_path = os.path.join(self.working_dir, "shots", f"{first_shot_idx}", "first_frame_selector_output.json")
                if os.path.exists(ff_selector_output_path):
                    with open(ff_selector_output_path, 'r', encoding='utf-8') as f:
                        ff_selector_output = json.load(f)
                    print(f"🚀 Loaded existing reference image selection and prompt for first_frame of shot {first_shot_idx} from {ff_selector_output_path}.")
                    _emit_render_progress(progress, "frame_prompt_exists", f"First frame prompt for shot {first_shot_idx} already exists", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame", "path": ff_selector_output_path})
                else:
                    print(f"🔍 Selecting reference images and generating prompt for first_frame of shot {first_shot_idx}...")
                    _emit_render_progress(progress, "frame_prompt_start", f"Selecting references for first frame of shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame"})
                    ff_selector_output = await self.reference_image_selector.select_reference_images_and_generate_prompt(
                        available_image_path_and_text_pairs=available_image_path_and_text_pairs,
                        frame_description=shot_descriptions[first_shot_idx].ff_desc
                    )
                    with open(ff_selector_output_path, 'w', encoding='utf-8') as f:
                        json.dump(ff_selector_output, f, ensure_ascii=False, indent=4)

                    print(f"☑️ Selected reference images and generated prompt for first_frame of shot {first_shot_idx}, saved to {ff_selector_output_path}.")
                    _emit_render_progress(progress, "frame_prompt_done", f"Selected references for first frame of shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame", "path": ff_selector_output_path})

                reference_image_path_and_text_pairs, prompt = ff_selector_output["reference_image_path_and_text_pairs"], ff_selector_output["text_prompt"]
                prefix_prompt = ""
                for i, (image_path, text) in enumerate(reference_image_path_and_text_pairs):
                    prefix_prompt += f"Image {i}: {text}\n"
                prompt = f"{prefix_prompt}\n{prompt}"
                first_shot_visible = [
                    characters[idx] for idx in shot_descriptions[first_shot_idx].ff_vis_char_idxs
                    if isinstance(idx, int) and 0 <= idx < len(characters)
                ]
                prompt += self._frame_prompt_locks(first_shot_visible, getattr(shot_descriptions[first_shot_idx], "screen_guidance", None))
                reference_image_paths = [item[0] for item in reference_image_path_and_text_pairs]
                ff_image: ImageOutput = await self.generate_and_select_best_image(
                    prompt=prompt,
                    reference_image_paths=reference_image_paths,
                    reference_image_path_and_text_pairs=reference_image_path_and_text_pairs,
                    target_description=shot_descriptions[first_shot_idx].ff_desc,
                    size=self._image_size(),
                    candidates_save_dir=os.path.join(self.working_dir, "shots", str(first_shot_idx), "first_frame_candidates"),
                    progress=_scoped_progress(progress, camera_idx=camera.idx, shot_idx=first_shot_idx, frame_type="first_frame"),
                )
                ff_image.save(first_shot_ff_path)
                self.frame_events[first_shot_idx]["first_frame"].set()
                print(f"☑️ Generated first_frame for shot {first_shot_idx}, saved to {first_shot_ff_path}.")
                _emit_render_progress(progress, "frame_done", f"Generated first frame for shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame", "path": first_shot_ff_path})
            else:
                shutil.copy(new_camera_image_path, first_shot_ff_path)
                self.frame_events[first_shot_idx]["first_frame"].set()
                print(f"☑️ Generated first_frame for shot {first_shot_idx}, saved to {first_shot_ff_path}.")
                _emit_render_progress(progress, "frame_done", f"Generated first frame for shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame", "path": first_shot_ff_path})


        # 2. following frames of the camera under the Flow-style chain
        if bool(self.render_geometry.get("chain_frames", True)):
            # Shot 0 of the camera gets a generated keyframe (above). Every
            # later shot's first frame IS the previous clip's real end frame -
            # copied in the video phase, never generated. Here we only arm the
            # events the video phase waits on, and build last-frame anchors
            # for medium/large variations (end conditioning).
            for position, shot_idx in enumerate(camera.active_shot_idxs):
                if position > 0:
                    events = self.frame_events.setdefault(shot_idx, {})
                    events.setdefault("first_frame", asyncio.Event()).set()
                    _emit_render_progress(progress, "frame_from_chain", f"Shot {shot_idx} first frame chains from the previous clip's end frame", {"camera_idx": camera.idx, "shot_idx": shot_idx})
                if shot_descriptions[shot_idx].variation_type in ["medium", "large"]:
                    self.frame_events.setdefault(shot_idx, {}).setdefault("last_frame", asyncio.Event())
                    await self.generate_frame_for_single_shot(
                        shot_idx=shot_idx,
                        frame_type="last_frame",
                        first_shot_ff_path_and_text_pair=(first_shot_ff_path, shot_descriptions[first_shot_idx].ff_desc),
                        frame_desc=shot_descriptions[shot_idx].lf_desc,
                        visible_characters=[characters[idx] for idx in shot_descriptions[shot_idx].lf_vis_char_idxs],
                        character_portraits_registry=character_portraits_registry,
                        progress=progress,
                        screen_guidance=getattr(shot_descriptions[shot_idx], "screen_guidance", None),
                    )
            _emit_render_progress(progress, "camera_frames_done", f"Frames for camera {camera.idx} ready (chained)", {"camera_idx": camera.idx, "active_shot_idxs": camera.active_shot_idxs})
            return

        priority_tasks = []
        normal_tasks = []

        if shot_descriptions[first_shot_idx].variation_type in ["medium", "large"]:
            task = self.generate_frame_for_single_shot(
                shot_idx=first_shot_idx,
                frame_type="last_frame",
                first_shot_ff_path_and_text_pair=(first_shot_ff_path, shot_descriptions[first_shot_idx].ff_desc),
                frame_desc=shot_descriptions[first_shot_idx].lf_desc,
                visible_characters=[characters[idx] for idx in shot_descriptions[first_shot_idx].lf_vis_char_idxs],
                character_portraits_registry=character_portraits_registry,
                progress=progress,
            )
            normal_tasks.append(task)

        for shot_idx in camera.active_shot_idxs[1:]:
            first_frame_task = self.generate_frame_for_single_shot(
                    shot_idx=shot_idx,
                    frame_type="first_frame",
                    first_shot_ff_path_and_text_pair=(first_shot_ff_path, shot_descriptions[first_shot_idx].ff_desc),
                    frame_desc=shot_descriptions[shot_idx].ff_desc,
                    visible_characters=[characters[idx] for idx in shot_descriptions[shot_idx].ff_vis_char_idxs],
                    character_portraits_registry=character_portraits_registry,
                    progress=progress,
                    screen_guidance=getattr(shot_descriptions[shot_idx], "screen_guidance", None),
                )
            if shot_idx in priority_shot_idxs:
                priority_tasks.append(first_frame_task)
            else:
                normal_tasks.append(first_frame_task)


            if shot_descriptions[shot_idx].variation_type in ["medium", "large"]:
                last_frame_task = self.generate_frame_for_single_shot(
                    shot_idx=shot_idx,
                    frame_type="last_frame",
                    first_shot_ff_path_and_text_pair=(first_shot_ff_path, shot_descriptions[first_shot_idx].ff_desc),
                    frame_desc=shot_descriptions[shot_idx].lf_desc,
                    visible_characters=[characters[idx] for idx in shot_descriptions[shot_idx].lf_vis_char_idxs],
                    character_portraits_registry=character_portraits_registry,
                    progress=progress,
                    screen_guidance=getattr(shot_descriptions[shot_idx], "screen_guidance", None),
                )
                normal_tasks.append(last_frame_task)


        await asyncio.gather(*priority_tasks)
        await asyncio.gather(*normal_tasks)
        _emit_render_progress(progress, "camera_frames_done", f"Frames for camera {camera.idx} ready", {"camera_idx": camera.idx, "active_shot_idxs": camera.active_shot_idxs})



    async def _generate_all_shot_videos(
        self,
        shot_descriptions: List[ShotDescription],
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        characters: Optional[List[CharacterInScene]] = None,
        character_portraits_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
    ) -> None:
        """Serial shot chain: clip N+1 starts from clip N's real end frame.

        Flow-style: for same-camera consecutive shots, the previous clip's
        extracted end frame is copied over the next shot's first_frame.png -
        the chain is literal pixels, no separate keyframe generation.
        """
        previous_end_frame_path: Optional[str] = None
        previous_cam_idx: Optional[int] = None
        for position, shot_description in enumerate(shot_descriptions):
            same_camera = (
                position > 0
                and previous_cam_idx is not None
                and getattr(shot_description, "cam_idx", None) == previous_cam_idx
            )
            if same_camera:
                if previous_end_frame_path is not None:
                    self._chain_first_frame_from_previous_clip(shot_description.idx, previous_end_frame_path)
                first_frame_path = os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "first_frame.png")
                if previous_end_frame_path is None and not os.path.exists(first_frame_path):
                    # Chain broken (the previous clip failed or its end frame
                    # was unreadable): fall back to generating this shot's
                    # keyframe instead of referencing a missing file.
                    await self.generate_frame_for_single_shot(
                        shot_idx=shot_description.idx,
                        frame_type="first_frame",
                        first_shot_ff_path_and_text_pair=(
                            os.path.join(self.working_dir, "shots", f"{shot_descriptions[position - 1].idx}", "first_frame.png"),
                            getattr(shot_description, "ff_desc", ""),
                        ),
                        frame_desc=getattr(shot_description, "ff_desc", ""),
                        visible_characters=[
                            characters[i] for i in (getattr(shot_description, "ff_vis_char_idxs", None) or [])
                            if isinstance(i, int) and 0 <= i < len(characters)
                        ] if characters else [],
                        character_portraits_registry=character_portraits_registry or {},
                        progress=progress,
                        screen_guidance=getattr(shot_description, "screen_guidance", None),
                    )
            await self.generate_video_for_single_shot(
                shot_description=shot_description,
                progress=progress,
                previous_end_frame_path=previous_end_frame_path,
            )
            video_path = os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "video.mp4")
            previous_end_frame_path = self._ensure_video_end_frame(video_path, shot_description.idx)
            previous_cam_idx = getattr(shot_description, "cam_idx", None)

    def _chain_first_frame_from_previous_clip(self, shot_idx: int, previous_end_frame_path: str) -> None:
        """Copy the previous clip's real end frame over this shot's keyframe.

        Resume-safe: an existing keyframe is kept. Arms the first_frame event
        the video phase waits on (skipped shots never generate one).
        """
        events = self.frame_events.setdefault(shot_idx, {})
        event = events.setdefault("first_frame", asyncio.Event())
        target = os.path.join(self.working_dir, "shots", f"{shot_idx}", "first_frame.png")
        if not os.path.exists(target) and os.path.exists(previous_end_frame_path):
            shutil.copy(previous_end_frame_path, target)
            print(f"🔗 Shot {shot_idx} first frame chained from the previous clip's end frame.")
        event.set()

    async def generate_video_for_single_shot(
        self,
        shot_description: ShotDescription,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        previous_end_frame_path: Optional[str] = None,
    ):
        video_path = os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "video.mp4")
        if os.path.exists(video_path):
            print(f"🚀 Skipped generating video for shot {shot_description.idx}, already exists.")
            _emit_render_progress(progress, "video_clip_exists", f"Video clip for shot {shot_description.idx} already exists", {"shot_idx": shot_description.idx, "path": video_path})
        else:
            _emit_render_progress(progress, "video_clip_waiting_for_frames", f"Waiting for frames before video clip {shot_description.idx}", {"shot_idx": shot_description.idx})
            await self.frame_events[shot_description.idx]["first_frame"].wait()
            if shot_description.variation_type in ["medium", "large"]:
                await self.frame_events[shot_description.idx]["last_frame"].wait()

            frame_paths = []
            own_last_frame = os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "last_frame.png")
            if previous_end_frame_path and os.path.exists(previous_end_frame_path):
                # Exact continuation chain: the previous clip's real end frame IS
                # this clip's first frame — the video starts from those pixels,
                # never a blend with this shot's own keyframe.
                frame_paths.append(previous_end_frame_path)
                if shot_description.variation_type in ["medium", "large"] and os.path.exists(own_last_frame):
                    frame_paths.append(own_last_frame)
            else:
                frame_paths.append(os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "first_frame.png"))
                if shot_description.variation_type in ["medium", "large"] and os.path.exists(own_last_frame):
                    frame_paths.append(own_last_frame)

            print(f"🎬 Starting video generation for shot {shot_description.idx}...")
            _emit_render_progress(progress, "video_clip_start", f"Generating video clip for shot {shot_description.idx}", {"shot_idx": shot_description.idx, "frame_count": len(frame_paths), "chained": bool(previous_end_frame_path)})
            shot_seconds = self._effective_shot_seconds(shot_description)
            video_output = await self.video_generator.generate_single_video(
                prompt=self._build_video_prompt(shot_description, shot_seconds, chained=bool(previous_end_frame_path)),
                reference_image_paths=frame_paths,
                aspect_ratio=self._video_aspect_ratio(),
                resolution=self._video_resolution(),
                progress=_scoped_progress(progress, shot_idx=shot_description.idx, artifact="video_clip"),
            )
            video_output.save(video_path)
            self._ensure_video_end_frame(video_path, shot_description.idx)
            if _trim_video_to_seconds(video_path, shot_seconds):
                print(f"✂️ Trimmed video for shot {shot_description.idx} to the {shot_seconds:g}s budget.")
                _emit_render_progress(progress, "video_clip_trimmed", f"Trimmed video clip for shot {shot_description.idx} to budget", {"shot_idx": shot_description.idx, "seconds": shot_seconds, "path": video_path})
            delivered = self._probe_duration(video_path)
            if delivered is not None and shot_seconds - delivered > 0.5:
                shortfall = round(shot_seconds - delivered, 2)
                print(f"⚠️ Shot {shot_description.idx} clip delivered {delivered:.2f}s of the {shot_seconds:g}s budget ({shortfall:g}s short) — the tail (possibly dialogue) is missing; consider splitting this shot.")
                _emit_render_progress(
                    progress,
                    "clip_short",
                    f"Video clip for shot {shot_description.idx} is shorter than its budget",
                    {"shot_idx": shot_description.idx, "requested_seconds": shot_seconds, "delivered_seconds": round(delivered, 2), "short_by_seconds": shortfall},
                )
            print(f"☑️ Generated video for shot {shot_description.idx}, saved to {video_path}.")
            _emit_render_progress(progress, "video_clip_done", f"Generated video clip for shot {shot_description.idx}", {"shot_idx": shot_description.idx, "path": video_path})

    async def generate_and_select_best_image(
        self,
        prompt: str,
        reference_image_paths: List[str],
        reference_image_path_and_text_pairs: List[Tuple[str, str]],
        target_description: str,
        size: str | None = None,
        num_candidates: int | None = None,
        candidates_save_dir: str | None = None,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
    ) -> ImageOutput:
        count = validate_image_candidate_count(self.num_image_candidates if num_candidates is None else num_candidates)
        if size is None:
            size = self._image_size()
        prompt = prompt.rstrip() + NO_TEXT_IMAGE_SUFFIX + REALISM_IMAGE_SUFFIX
        if count == 1:
            return await self.image_generator.generate_single_image(
                prompt=prompt, reference_image_paths=reference_image_paths, size=size,
                aspect_ratio=self._image_aspect_ratio(), allow_portrait=self._allow_portrait(),
            )

        save_dir = Path(candidates_save_dir) if candidates_save_dir else Path(self.working_dir) / "image_candidates" / uuid4().hex
        save_dir.mkdir(parents=True, exist_ok=True)
        selection_path = save_dir / "selection.json"
        references = []
        for path, description in reference_image_path_and_text_pairs:
            stat = os.stat(path)
            references.append((path, description, stat.st_size, stat.st_mtime_ns))
        fingerprint = hashlib.sha256(json.dumps({
            "prompt": prompt, "size": size, "references": references,
            "reference_image_paths": reference_image_paths, "target_description": target_description,
            "generator": type(self.image_generator).__name__,
            "model": str(getattr(self.image_generator, "model", "")),
        }, sort_keys=True).encode()).hexdigest()
        previous = {}
        if selection_path.exists():
            try:
                previous = json.loads(selection_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        reuse = isinstance(previous, dict) and previous.get("request_fingerprint") == fingerprint
        reusable_indices = {
            item["index"] for item in previous.get("candidates", [])
            if isinstance(item, dict) and item.get("status") == "ready" and "index" in item
        } if reuse else set()
        selection = {"request_fingerprint": fingerprint, "requested_candidates": count, "status": "generating", "candidates": []}

        def save_selection():
            temporary = selection_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(selection_path)

        save_selection()
        _emit_render_progress(progress, "image_candidates_start", f"Generating {count} image candidates", {"count": count})

        async def generate_candidate(index):
            path = save_dir / f"candidate_{index}.png"
            record = {"index": index, "path": path.name, "status": "generating"}
            selection["candidates"].append(record)
            try:
                if index in reusable_indices and path.is_file():
                    record["reused"] = True
                else:
                    output = await self.image_generator.generate_single_image(
                        prompt=prompt, reference_image_paths=reference_image_paths, size=size,
                        aspect_ratio=self._image_aspect_ratio(), allow_portrait=self._allow_portrait(),
                    )
                    if output is None:
                        raise RuntimeError("Image provider returned no image")
                    await asyncio.to_thread(output.save, str(path))
                with Image.open(path) as image:
                    width, height = image.size
                    output = ImageOutput(fmt="pil", ext="png", data=image.copy())
                record.update(width=width, height=height)
                if width <= height and not self._allow_portrait():
                    raise ValueError("Candidate is not landscape")
                record["status"] = "ready"
                save_selection()
                _emit_render_progress(progress, "image_candidate_done", f"Image candidate {index + 1}/{count} ready", {"index": index, "count": count})
                return index, output, str(path)
            except Exception as exc:
                record.update(status="error", error_type=type(exc).__name__)
                save_selection()
                _emit_render_progress(progress, "image_candidate_failed", f"Image candidate {index + 1}/{count} failed", {"index": index, "error_type": type(exc).__name__})
                raise

        tasks = [asyncio.create_task(generate_candidate(index)) for index in range(count)]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            cancelled = next((result for result in results if isinstance(result, asyncio.CancelledError)), None)
            if cancelled is not None:
                raise cancelled
            valid = [result for result in results if not isinstance(result, Exception)]
            if not valid:
                selection["status"] = "generation_failed"
                save_selection()
                raise RuntimeError(f"All {count} image candidates failed: {results[0]}") from results[0]

            if len(valid) == 1:
                selected = valid[0]
                reason = "Only one valid landscape candidate; VLM comparison skipped."
                selection["selection_method"] = "single_valid_candidate"
            else:
                selection["status"] = "selecting"
                save_selection()
                _emit_render_progress(progress, "image_selection_start", f"VLM selecting from {len(valid)} candidates", {"count": len(valid)})
                try:
                    response = await self.best_image_selector.select(
                        reference_image_path_and_text_pairs=reference_image_path_and_text_pairs,
                        target_description=target_description,
                        candidate_image_paths=[item[2] for item in valid],
                    )
                except Exception as exc:
                    selection.update(status="selection_failed", error_type=type(exc).__name__)
                    save_selection()
                    raise
                selected = valid[response.best_image_index]
                reason = response.reason
                selection["selection_method"] = "vlm"
            selection.update(status="selected", selected_candidate_index=selected[0], reason=reason)
            save_selection()
            _emit_render_progress(progress, "image_selection_done", f"Selected image candidate {selected[0] + 1}", {"index": selected[0], "count": len(valid), "selection_method": selection["selection_method"]})
            return selected[1]
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            selection["status"] = "cancelled"
            save_selection()
            raise


    def _previous_state_pair(self, shot_idx: int) -> Tuple[str, str]:
        """Reference pair carrying the previous shot's final visual state.

        Preference order: the real end frame of the previous shot's rendered
        video, then its planned last_frame image, then its first frame.
        Used for the continuity chain between shots.
        """
        candidates = [
            os.path.join(self.working_dir, "shots", f"{shot_idx}", "video_last_frame.png"),
            os.path.join(self.working_dir, "shots", f"{shot_idx}", "last_frame.png"),
            os.path.join(self.working_dir, "shots", f"{shot_idx}", "first_frame.png"),
        ]
        path = next((candidate for candidate in candidates if os.path.exists(candidate)), candidates[-1])
        return (
            path,
            "The final visual state of the immediately previous shot. The new frame continues directly from this state: keep character appearance, wardrobe, prop and screen states identical unless the frame description explicitly changes them.",
        )

    def _video_last_frame_path(self, shot_idx: int) -> str:
        return os.path.join(self.working_dir, "shots", f"{shot_idx}", "video_last_frame.png")

    @staticmethod
    def _probe_duration(path: str) -> Optional[float]:
        """Delivered duration of a rendered clip, or None when unreadable."""
        try:
            clip = VideoFileClip(path)
        except Exception:
            return None
        try:
            return float(clip.duration) if clip.duration else None
        finally:
            clip.close()

    def _ensure_video_end_frame(self, video_path: str, shot_idx: int) -> Optional[str]:
        """Extract and cache the real final frame of a rendered clip.

        This is the authoritative hand-off state for the shot chain: the next
        clip starts from these actual pixels, not from a planned image.
        """
        if not os.path.exists(video_path):
            return None
        target = self._video_last_frame_path(shot_idx)
        if os.path.exists(target):
            return target
        try:
            clip = VideoFileClip(video_path)
            try:
                frame_time = max(0.0, clip.duration - 1 / max(1.0, clip.fps or 24.0))
                frame = clip.get_frame(frame_time)
            finally:
                clip.close()
            Image.fromarray(frame.astype("uint8"), "RGB").save(target)
            return target
        except Exception as exc:
            logging.warning("Could not extract end frame from %s (%s)", video_path, exc)
            return None

    def _video_prompt_characters(self) -> List[Any]:
        """Characters for the video prompt's identity lock, from characters.json.

        File-based state (written by both the planning and render paths), so
        the lock reaches every clip without threading the list through the
        whole render call graph.
        """
        path = os.path.join(self.working_dir, "characters.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return []
        characters: List[Any] = []
        for item in payload if isinstance(payload, list) else []:
            try:
                characters.append(CharacterInScene.model_validate(item) if isinstance(item, dict) else item)
            except Exception:
                continue
        return characters

    def _identity_lock_text(self, shot_description: Any) -> str:
        """Per-clip character/wardrobe lock (the Flow 'REFERENCE LOCK' pattern)."""
        def _idxs(shot: Any, attr: str) -> List[int]:
            value = shot.get(attr) if isinstance(shot, dict) else getattr(shot, attr, None)
            return list(value or [])
        wanted = sorted(set(_idxs(shot_description, "ff_vis_char_idxs") + _idxs(shot_description, "lf_vis_char_idxs")))
        characters = self._video_prompt_characters()
        lines: List[str] = []
        for idx in wanted:
            if not isinstance(idx, int) or not 0 <= idx < len(characters):
                continue
            character = characters[idx]
            name = getattr(character, "identifier_in_scene", f"character {idx}")
            part = f"{name}: {getattr(character, 'static_features', '')}"
            dynamic = getattr(character, "dynamic_features", None)
            if dynamic:
                part += f"; wearing {dynamic}"
            lines.append(part)
        if not lines:
            return ""
        return (
            "\n[Identity lock] These characters must remain identical throughout the clip — same face, age, hair, "
            "glasses and jewelry, the exact same garments (not similar replacements); no hairstyle, wardrobe or "
            "identity changes, no new visible person enters the frame: " + " | ".join(lines)
        )

    def _frame_prompt_locks(self, visible_characters: List[Any], screen_guidance: Any = None) -> str:
        """Identity lock (+ device grip) for IMAGE prompts: keyframes must
        match the reference portraits exactly, so every shot shows the same
        person (project1 showed a different-looking Grandma per shot), and a
        phone in use is held screen-to-the-character (project1 showed frontal
        portrait poses with the screen toward the camera)."""
        lines: List[str] = []
        for character in visible_characters or []:
            if character is None:
                continue
            name = getattr(character, "identifier_in_scene", "character")
            part = f"{name}: {getattr(character, 'static_features', '')}"
            dynamic = getattr(character, "dynamic_features", None)
            if dynamic:
                part += f"; wearing {dynamic}"
            lines.append(part)
        if not lines:
            return ""
        locks = (
            "\n[Identity lock] These characters must look exactly like their reference portraits - same face, "
            "age, hair, glasses and jewelry, the exact same garments (not similar replacements); no hairstyle, "
            "wardrobe or identity changes, no new visible person enters the frame: " + " | ".join(lines)
        )
        if screen_guidance:
            locks += (
                "\n[Device grip] The character's FACE stays visible to the camera (front-facing) - the phone is "
                "held low in front of her chest with its screen tilted toward her; she looks down at it. The app UI "
                "is NOT visible in this frame (only she sees the screen); fingers wrapped around the edges, thumbs "
                "on the bezels. The screen never faces the camera directly."
            )
        return locks

    def _voice_lock_text(self, shot_description: Any) -> str:
        """Voice-identity continuity for speaking characters (Flow 'VOICE LOCK')."""
        def _idxs(shot: Any, attr: str) -> List[int]:
            value = shot.get(attr) if isinstance(shot, dict) else getattr(shot, attr, None)
            return list(value or [])
        wanted = sorted(set(_idxs(shot_description, "ff_vis_char_idxs") + _idxs(shot_description, "lf_vis_char_idxs")))
        characters = self._video_prompt_characters()
        lines: List[str] = []
        for idx in wanted:
            if not isinstance(idx, int) or not 0 <= idx < len(characters):
                continue
            voice = getattr(characters[idx], "voice_features", None)
            if voice:
                name = getattr(characters[idx], "identifier_in_scene", f"character {idx}")
                lines.append(f"{name}: {voice}")
        if not lines:
            return ""
        return (
            "\n[Voice lock] Keep the same vocal identity, pitch, accent, timbre and speaking rhythm throughout the "
            "clip — excitement changes only expression and energy, never the underlying voice: " + " | ".join(lines)
        )

    def _build_video_prompt(self, shot_description: Any, seconds: float, *, chained: bool) -> str:
        def _field(attr: str, default: str = "") -> str:
            value = shot_description.get(attr) if isinstance(shot_description, dict) else getattr(shot_description, attr, None)
            return value if value is not None else default

        prompt = _field("motion_desc") + "\n" + _field("audio_desc")
        prompt += f"\n(Duration: approximately {int(round(seconds))} seconds.)"
        dialogue_window = self._dialogue_completion_window(_field("audio_desc"), seconds)
        prompt += (
            f"\nDeliver all spoken dialogue naturally across the clip: every line must be fully completed within the first "
            f"{max(1, round(dialogue_window))} seconds of the clip, leaving a short beat for reaction."
        )
        prompt += (
            "\nSpeak every quoted dialogue line exactly as written, in its original language — never translate, "
            "transliterate, or replace it; the speakers' voices must match the language of the lines."
        )
        prompt += self._identity_lock_text(shot_description)
        prompt += self._voice_lock_text(shot_description)
        if not extract_dialogue_lines(_field("audio_desc")):
            prompt += (
                "\n[No dialogue] This clip has NO spoken lines - the character does NOT speak: no words, no "
                "mumbling, no lip-sync speech of any kind. Keep a closed-mouth smile or natural breathing; "
                "only the ambient sound described above is audible."
            )
        if "off-camera" in _field("audio_desc").lower():
            prompt += (
                "\n[Off-camera speaker] A line marked off-camera belongs to one unseen person behind the camera: "
                "they are never visible, never enter the frame, and no second on-screen character appears for them. "
                "Their voice is clearly DISTINCT from the on-screen characters' voices — different pitch, different "
                "timbre, casually spoken from behind the lens — it must NOT sound like the on-screen character. "
                "Never deliver an off-camera line with an on-screen character's voice; the on-screen character's "
                "mouth stays closed while the off-camera line plays."
            )
        screen_guidance = shot_description.get("screen_guidance") if isinstance(shot_description, dict) else getattr(shot_description, "screen_guidance", None)
        if screen_guidance:
            guidance = str(screen_guidance).strip().rstrip(".")
            prompt += (
                f"\n[Screen guidance] The device screen stays as described — {guidance}. "
                "Do not invent detailed interface elements or readable text on the screen. "
                "The device keeps its stated orientation for the ENTIRE clip - it never rotates, flips, or turns "
                "toward the camera. The character's FACE stays visible to the camera (front-facing) while the phone "
                "is held low in front of her chest, screen tilted toward her - she looks down at it; the screen "
                "never faces the camera directly, and the grip stays the same from the first frame to the last."
            )
        if chained:
            prompt += (
                "\nThis shot continues directly from the previous shot: the FIRST reference image is the previous "
                "shot's final frame — the video must begin exactly at that visual state (same pose, wardrobe, prop "
                "and screen state, same framing) and never jump back in time."
            )
        is_last = shot_description.get("is_last") if isinstance(shot_description, dict) else getattr(shot_description, "is_last", False)
        if not is_last:
            prompt += "\nEnd the clip on the final action or the last spoken word, ready for the next clip — no trailing dead air."
        prompt += NO_TEXT_VIDEO_SUFFIX
        prompt += REALISM_VIDEO_SUFFIX
        return prompt

    async def generate_frame_for_single_shot(
        self,
        shot_idx: int,
        frame_type: Literal["first_frame", "last_frame"],
        first_shot_ff_path_and_text_pair: Tuple[str, str],
        frame_desc: str,
        visible_characters: List[CharacterInScene],
        character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        previous_frame_path_and_text_pair: Optional[Tuple[str, str]] = None,
        screen_guidance: Optional[str] = None,
    ) -> ImageOutput:

        frame_image_path = os.path.join(self.working_dir, "shots", f"{shot_idx}", f"{frame_type}.png")

        if os.path.exists(frame_image_path):
            print(f"🚀 Skipped generating {frame_type} for shot {shot_idx}, already exists.")
            _emit_render_progress(progress, "frame_exists", f"{frame_type} for shot {shot_idx} already exists", {"shot_idx": shot_idx, "frame_type": frame_type, "path": frame_image_path})

        else:
            print(f"🖼️ Starting {frame_type} generation for shot {shot_idx}...")
            _emit_render_progress(progress, "frame_start", f"Generating {frame_type} for shot {shot_idx}", {"shot_idx": shot_idx, "frame_type": frame_type})
            available_image_path_and_text_pairs = []
            for visible_character in visible_characters:
                identifier_in_scene = visible_character.identifier_in_scene
                registry_item = character_portraits_registry[identifier_in_scene]
                for view, item in registry_item.items():
                    available_image_path_and_text_pairs.append((item["path"], item["description"]))

            available_image_path_and_text_pairs.append(first_shot_ff_path_and_text_pair)
            if previous_frame_path_and_text_pair is not None:
                available_image_path_and_text_pairs.append(previous_frame_path_and_text_pair)

            selector_output_path = os.path.join(self.working_dir, "shots", f"{shot_idx}", f"{frame_type}_selector_output.json")
            if os.path.exists(selector_output_path):
                with open(selector_output_path, 'r', encoding='utf-8') as f:
                    selector_output = json.load(f)
                print(f"🚀 Loaded existing reference image selection and prompt for {frame_type} frame of shot {shot_idx} from {selector_output_path}.")
                _emit_render_progress(progress, "frame_prompt_exists", f"Prompt for {frame_type} of shot {shot_idx} already exists", {"shot_idx": shot_idx, "frame_type": frame_type, "path": selector_output_path})
            else:
                print(f"🔍 Selecting reference images and generating prompt for {frame_type} frame of shot {shot_idx}...")
                _emit_render_progress(progress, "frame_prompt_start", f"Selecting references for {frame_type} of shot {shot_idx}", {"shot_idx": shot_idx, "frame_type": frame_type})
                selector_output = await self.reference_image_selector.select_reference_images_and_generate_prompt(
                    available_image_path_and_text_pairs=available_image_path_and_text_pairs,
                    frame_description=frame_desc
                )
                with open(selector_output_path, 'w', encoding='utf-8') as f:
                    json.dump(selector_output, f, ensure_ascii=False, indent=4)
                print(f"☑️ Selected reference images and generated prompt for {frame_type} frame of shot {shot_idx}, saved to {selector_output_path}.")
                _emit_render_progress(progress, "frame_prompt_done", f"Selected references for {frame_type} of shot {shot_idx}", {"shot_idx": shot_idx, "frame_type": frame_type, "path": selector_output_path})

            reference_image_path_and_text_pairs, prompt = selector_output["reference_image_path_and_text_pairs"], selector_output["text_prompt"]
            prefix_prompt = ""
            for i, (image_path, text) in enumerate(reference_image_path_and_text_pairs):
                prefix_prompt += f"Image {i}: {text}\n"
            prompt = f"{prefix_prompt}\n{prompt}"
            prompt += self._frame_prompt_locks(visible_characters, screen_guidance)
            reference_image_paths = [item[0] for item in reference_image_path_and_text_pairs]

            frame_image: ImageOutput = await self.generate_and_select_best_image(
                prompt=prompt,
                reference_image_paths=reference_image_paths,
                reference_image_path_and_text_pairs=reference_image_path_and_text_pairs,
                target_description=frame_desc,
                size=self._image_size(),
                candidates_save_dir=os.path.join(self.working_dir, "shots", str(shot_idx), f"{frame_type}_candidates"),
                progress=_scoped_progress(progress, shot_idx=shot_idx, frame_type=frame_type),
            )
            frame_image.save(frame_image_path)
            print(f"☑️ Generated {frame_type} frame for shot {shot_idx}, saved to {frame_image_path}.")
            _emit_render_progress(progress, "frame_done", f"Generated {frame_type} for shot {shot_idx}", {"shot_idx": shot_idx, "frame_type": frame_type, "path": frame_image_path})


        self.frame_events[shot_idx][frame_type].set()
        return frame_image_path


    async def construct_camera_tree(
        self,
        shot_descriptions: List[ShotDescription],
        quiet: bool = False,
    ):
        camera_tree_path = os.path.join(self.working_dir, "camera_tree.json")

        if os.path.exists(camera_tree_path):
            with open(camera_tree_path, "r", encoding="utf-8") as f:
                camera_tree = json.load(f)
            camera_tree = [Camera.model_validate(camera) for camera in camera_tree]
            _pipeline_print(quiet, f"🚀 Loaded {len(camera_tree)} cameras from existing file.")
            return camera_tree

        shot_descriptions = _normalize_model_list(shot_descriptions, ShotDescription, "shot_descriptions")
        cameras = _group_shots_into_cameras(shot_descriptions)

        camera_tree = await self.camera_image_generator.construct_camera_tree(cameras=cameras, shot_descs=shot_descriptions)
        camera_tree = _normalize_model_list(camera_tree, Camera, "camera_tree")
        with open(camera_tree_path, "w", encoding="utf-8") as f:
            json.dump([camera.model_dump() for camera in camera_tree], f, ensure_ascii=False, indent=4)
        _pipeline_print(quiet, f"✅ Constructed camera tree and saved to {camera_tree_path}.")
        return camera_tree




    async def extract_characters(
        self,
        script: str,
        quiet: bool = False,
    ):
        save_path = os.path.join(self.working_dir, "characters.json")

        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                characters = json.load(f)
            characters = [CharacterInScene.model_validate(character) for character in characters]
            _pipeline_print(quiet, f"🚀 Loaded {len(characters)} characters from existing file.")
        else:
            characters = await self.character_extractor.extract_characters(script)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump([character.model_dump() for character in characters], f, ensure_ascii=False, indent=4)
            _pipeline_print(quiet, f"✅ Extracted {len(characters)} characters from script and saved to {save_path}.")

        for character in characters:
            self.character_portrait_events[character.idx] = asyncio.Event()

        return characters


    async def generate_character_portraits(
        self,
        characters: List[CharacterInScene],
        character_portraits_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]],
        style: str,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
    ):
        character_portraits_registry_path = os.path.join(self.working_dir, "character_portraits_registry.json")
        if character_portraits_registry is None:
            if os.path.exists(character_portraits_registry_path):
                with open(character_portraits_registry_path, 'r', encoding='utf-8') as f:
                    character_portraits_registry = json.load(f)
            else:
                character_portraits_registry = {}


        tasks = [
            self.generate_portraits_for_single_character(character, style, progress=progress)
            for character in characters
            if character.identifier_in_scene not in character_portraits_registry
        ]
        if tasks:
            for future in asyncio.as_completed(tasks):
                character_portraits_registry.update(await future)
                with open(character_portraits_registry_path, 'w', encoding='utf-8') as f:
                    json.dump(character_portraits_registry, f, ensure_ascii=False, indent=4)

            print(f"✅ Completed character portrait generation for {len(characters)} characters.")
            _emit_render_progress(progress, "character_portraits_done", "Completed character portrait generation", {"character_count": len(characters)})
        else:
            print("🚀 All characters already have portraits, skipping portrait generation.")
            _emit_render_progress(progress, "character_portraits_exist", "All character portraits already exist", {"character_count": len(characters)})
        return character_portraits_registry


    async def generate_portraits_for_single_character(
        self,
        character: CharacterInScene,
        style: str,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
    ):
        character_dir = os.path.join(self.working_dir, "character_portraits", f"{character.idx}_{character.identifier_in_scene}")
        os.makedirs(character_dir, exist_ok=True)
        _emit_render_progress(progress, "character_portrait_start", f"Generating portraits for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene})

        front_portrait_path = os.path.join(character_dir, "front.png")
        if os.path.exists(front_portrait_path):
            pass
        else:
            _emit_render_progress(progress, "character_portrait_front_start", f"Generating front portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene})
            front_portrait_output = await self.character_portraits_generator.generate_front_portrait(character, style)
            front_portrait_output.save(front_portrait_path)
            _emit_render_progress(progress, "character_portrait_front_done", f"Generated front portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene, "path": front_portrait_path})


        side_portrait_path = os.path.join(character_dir, "side.png")
        if os.path.exists(side_portrait_path):
            pass
        else:
            _emit_render_progress(progress, "character_portrait_side_start", f"Generating side portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene})
            side_portrait_output = await self.character_portraits_generator.generate_side_portrait(character, front_portrait_path)
            side_portrait_output.save(side_portrait_path)
            _emit_render_progress(progress, "character_portrait_side_done", f"Generated side portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene, "path": side_portrait_path})

        back_portrait_path = os.path.join(character_dir, "back.png")
        if os.path.exists(back_portrait_path):
            pass
        else:
            _emit_render_progress(progress, "character_portrait_back_start", f"Generating back portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene})
            back_portrait_output = await self.character_portraits_generator.generate_back_portrait(character, front_portrait_path)
            back_portrait_output.save(back_portrait_path)
            _emit_render_progress(progress, "character_portrait_back_done", f"Generated back portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene, "path": back_portrait_path})

        self.character_portrait_events[character.idx].set()

        print(f"☑️ Completed character portrait generation for {character.identifier_in_scene}.")
        _emit_render_progress(progress, "character_portrait_done", f"Portraits for {character.identifier_in_scene} ready", {"character_idx": character.idx, "identifier": character.identifier_in_scene})

        return {
            character.identifier_in_scene: {
                "front": {
                    "path": front_portrait_path,
                    "description": f"A front view portrait of {character.identifier_in_scene}.",
                },
                "side": {
                    "path": side_portrait_path,
                    "description": f"A side view portrait of {character.identifier_in_scene}.",
                },
                "back": {
                    "path": back_portrait_path,
                    "description": f"A back view portrait of {character.identifier_in_scene}.",
                },
            }
        }



    async def design_storyboard(
        self,
        script: str,
        characters: List[CharacterInScene],
        user_requirement: str,
        quiet: bool = False,
    ):
        storyboard_path = os.path.join(self.working_dir, "storyboard.json")
        if os.path.exists(storyboard_path):
            with open(storyboard_path, 'r', encoding='utf-8') as f:
                storyboard = json.load(f)
            storyboard = [ShotBriefDescription.model_validate(shot) for shot in storyboard]
            _pipeline_print(quiet, f"🚀 Loaded {len(storyboard)} shot brief descriptions from existing file.")
        else:
            _pipeline_print(quiet, f"🔍 Designing storyboard...")
            storyboard = await self.storyboard_artist.design_storyboard(
                script=script,
                characters=characters,
                user_requirement=user_requirement,
                retry_timeout=150,
            )
            storyboard = _normalize_model_list(storyboard, ShotBriefDescription, "storyboard")
            with open(storyboard_path, 'w', encoding='utf-8') as f:
                json.dump([shot.model_dump() for shot in storyboard], f, ensure_ascii=False, indent=4)
            _pipeline_print(quiet, f"✅ Designed storyboard and saved to {storyboard_path}.")

        for shot_brief_description in storyboard:
            self.shot_desc_events[shot_brief_description.idx] = asyncio.Event()

        return storyboard



    async def decompose_visual_descriptions(
        self,
        shot_brief_descriptions: List[ShotBriefDescription],
        characters: List[CharacterInScene],
        quiet: bool = False,
    ):
        tasks = [
            self.decompose_visual_description_for_single_shot_brief_description(shot_brief_description, characters, quiet=quiet)
            for shot_brief_description in shot_brief_descriptions
        ]

        shot_descriptions = await asyncio.gather(*tasks)
        return shot_descriptions


    async def decompose_visual_description_for_single_shot_brief_description(
        self,
        shot_brief_description: ShotBriefDescription,
        characters: List[CharacterInScene],
        quiet: bool = False,
    ):
        shot_description_path = os.path.join(self.working_dir, "shots", f"{shot_brief_description.idx}", "shot_description.json")
        os.makedirs(os.path.dirname(shot_description_path), exist_ok=True)

        if os.path.exists(shot_description_path):
            with open(shot_description_path, 'r', encoding='utf-8') as f:
                shot_description = ShotDescription.model_validate(json.load(f))
            _pipeline_print(quiet, f"🚀 Loaded shot {shot_brief_description.idx} description from existing file.")
        else:
            shot_description = await self.storyboard_artist.decompose_visual_description(
                shot_brief_desc=shot_brief_description,
                characters=characters,
                retry_timeout=120,
            )
            shot_description = _normalize_model_list([shot_description], ShotDescription, "shot_description")[0]
            with open(shot_description_path, 'w', encoding='utf-8') as f:
                json.dump(shot_description.model_dump(), f, ensure_ascii=False, indent=4)
            _pipeline_print(quiet, f"✅ Decomposed visual description for shot {shot_brief_description.idx} and saved to {shot_description_path}.")

        self.shot_desc_events[shot_brief_description.idx].set()

        if shot_description.variation_type == "small" and _requires_last_frame_anchor(shot_description):
            # Mechanical guarantee on top of the prompt taxonomy: a prop
            # rotation changes the shot's end state, so a last-frame image
            # must anchor it — otherwise the video model keeps the lit-screen
            # state facing the camera and silently drops the rotation.
            shot_description.variation_type = "medium"
            shot_description.variation_reason = (
                (shot_description.variation_reason or "").strip()
                + " [Mechanical override: this shot turns a device to a different face, so a last-frame anchor is generated to pin the end state.]"
            ).strip()
            with open(shot_description_path, "w", encoding="utf-8") as f:
                json.dump(shot_description.model_dump(), f, ensure_ascii=False, indent=4)
            _pipeline_print(quiet, f"🔒 Shot {shot_brief_description.idx}: promoted to 'medium' variation — device rotation needs a last-frame anchor.")

        if shot_description.variation_type in ["medium", "large"]:
            self.frame_events[shot_brief_description.idx] = {
                "first_frame": asyncio.Event(),
                "last_frame": asyncio.Event(),
            }
        else:
            self.frame_events[shot_brief_description.idx] = {
                "first_frame": asyncio.Event(),
            }

        return shot_description
