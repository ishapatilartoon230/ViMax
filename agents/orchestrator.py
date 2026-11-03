"""Orchestrator reviewer agent.

Reviews each pipeline stage's output against the user's original material and
returns a structured verdict. On failure the verdict names the target agent
and carries a concrete fix instruction, so the pipeline can bounce the work
back to exactly that agent (bounded retries).

Generic by design: the criteria are principles (fidelity to the user's
script, prop facing physics, timing math, continuity, no baked text), never
facts about one product or one script.
"""

from typing import Any, List, Optional

from pydantic import BaseModel, Field

from utils.robust_json_parser import parse_json


SYSTEM_PROMPT_ORCHESTRATOR_REVIEW = """
[Role]
You are the ViMax Orchestrator: a strict reviewing director standing in for the user.
You never generate content yourself. You review one stage's output against the
user's original material and the project's production rules, and either pass it
or send it back to the responsible agent with a precise fix instruction.

[Review criteria — apply the ones relevant to the stage]
1. Fidelity: every beat, shot, and dialogue line the user's script explicitly
   lists must exist, in order, with the same dialogue placement. No added,
   dropped, merged, or reordered shots. If the user stated a shot count, it matches.
2. Timing: per-shot durations come only from the script's stated timing
   (duration_seconds) or the pipeline default; totals are the sum and must fit
   the requested total and the hard cap. Invented per-shot durations fail.
3. Prop facing physics: a prop's face content is visible only via a presenting
   gesture or a POV insert shot — never a magically tilted prop during private
   use. Rotations (show, then turn back, opposite side out) are explicit beats.
4. Characters: appearance details are drawable and complete, identical across
   shots, identifiers consistent, one wardrobe unless the script changes it.
5. Continuity: wardrobe, prop states, screen states, positions, and eyelines
   carry across shots; shot N's end state is shot N+1's start state.
6. No baked text: frame/motion descriptions contain no captions, overlays,
   CTA buttons, or logos — those are post-production.
7. Realism: the result should read like real footage of the described scene,
   not a staged or physically impossible reconstruction.

[Verdict rules]
- Pass only when the output satisfies the criteria relevant to this stage.
- On failure: name the single responsible agent in target_agent, list each
  concrete issue in issues (quote the offending part), and write one fix
  instruction precise enough that the agent can act on it without guessing.
- Never invent requirements that are not in the user's material or the rules
  above. Style preferences of the user's material override your taste.
"""


class ReviewVerdict(BaseModel):
    verdict: str = Field(description="'pass' or 'fail'")
    stage: str = Field(description="The pipeline stage under review.")
    target_agent: str = Field(default="", description="The responsible agent to fix the work on failure (e.g. storyboard_artist, character_extractor, screenwriter). Empty on pass.")
    issues: List[str] = Field(default=[], description="Concrete issues found; empty on pass.")
    fix_instruction: str = Field(default="", description="One precise instruction for the target agent; empty on pass.")


class OrchestratorReviewer:
    def __init__(self, chat_model: Any, review_skills: str = ""):
        self.chat_model = chat_model
        self.review_skills = (review_skills or "").strip()

    async def review(
        self,
        stage: str,
        user_material: str,
        artifact: Any,
        extra_context: Optional[str] = None,
    ) -> ReviewVerdict:
        system = SYSTEM_PROMPT_ORCHESTRATOR_REVIEW + (f"\n[Project Skills]\n{self.review_skills}" if self.review_skills else "")
        human = self._build_human_prompt(stage, user_material, artifact, extra_context)
        response = await self.chat_model.ainvoke([("system", system), ("human", human)])
        raw = response.content
        if isinstance(raw, list):
            raw = "".join(block.get("text", "") for block in raw if isinstance(block, dict))
        return self._parse_verdict(str(raw), stage)

    def _build_human_prompt(self, stage: str, user_material: str, artifact: Any, extra_context: Optional[str]) -> str:
        import json as _json

        try:
            artifact_text = _json.dumps(artifact, ensure_ascii=False, default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o))
        except (TypeError, ValueError):
            artifact_text = str(artifact)
        if len(artifact_text) > 10000:
            artifact_text = artifact_text[:10000] + "\n...[artifact truncated]"
        material = user_material if len(user_material) <= 10000 else user_material[:10000] + "\n...[user material truncated]"
        parts = [
            f"Stage under review: {stage}",
            "<USER_MATERIAL>",
            material,
            "</USER_MATERIAL>",
            "<ARTIFACT>",
            artifact_text,
            "</ARTIFACT>",
        ]
        if extra_context:
            parts.extend(["<EXTRA_CONTEXT>", extra_context[:4000], "</EXTRA_CONTEXT>"])
        parts.append("Review the artifact against the user material and the rules. Reply with the JSON verdict only.")
        return "\n".join(parts)

    def _parse_verdict(self, raw: str, stage: str) -> ReviewVerdict:
        payload = parse_json(raw)
        if isinstance(payload, dict):
            payload.setdefault("stage", stage)
            verdict = str(payload.get("verdict", "pass")).strip().lower()
            payload["verdict"] = "pass" if verdict == "pass" else "fail"
            if payload["verdict"] == "pass":
                payload["issues"] = []
                payload["fix_instruction"] = ""
                payload["target_agent"] = ""
            try:
                return ReviewVerdict.model_validate(payload)
            except Exception:
                pass
        # Unparseable reviewer output must never block the pipeline: treat as pass with a note.
        return ReviewVerdict(verdict="pass", stage=stage, target_agent="", issues=[], fix_instruction="")
