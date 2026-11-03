"""Dialogue duration fit checks.

A dialogue line cannot be spoken faster than people speak. This module turns
each shot's dialogue (quoted lines in ``audio_desc``) into an estimated speech
duration and compares it with the shot's clip budget, producing a per-shot fit
report with concrete recommended durations.

Used as a mechanical gate: a plan whose dialogue cannot fit its shots must be
shown to the user with a recommendation — never silently rendered and cut.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# Natural conversational delivery, calibrated against real generated ads:
# a 28-word / 5-line scene rendered comfortably inside 10s by video models
# (≈2.8 words/s including turn-taking). The older 2.3 w/s estimate rejected
# such scenes at planning time although the render fits.
WORDS_PER_SECOND = 2.8
# Natural beat between consecutive lines (turn-taking).
SECONDS_BETWEEN_LINES = 0.3
# A plan is still accepted when its speech estimate runs up to this multiple
# of the shot budget — models pace lines naturally inside the clip (the
# clip_short detector catches a real overrun after render).
OVERAGE_TOLERANCE = 1.15
# Share of the clip reserved for pacing/reaction after the last line.
REQUIRED_HEADROOM = 0.15
# A shot is "over" (hard fail) when speech alone exceeds the budget.
TIGHT_THRESHOLD = 0.85


class DialogueFitError(ValueError):
    """A plan's dialogue cannot fit its shot budgets even after extension.

    Raised by pipeline gates only when mechanically extending the shot to its
    recommended duration was impossible (the total-duration cap blocks it), so
    callers can tell a deterministic refusal apart from a transient failure.
    """

_QUOTED_LINE_RE = re.compile(r'"([^"]{2,})"')


def extract_dialogue_lines(audio_desc: Any) -> List[str]:
    text = str(audio_desc or "")
    return [line.strip() for line in _QUOTED_LINE_RE.findall(text) if line.strip()]


def estimate_speech_seconds(lines: List[str], words_per_second: float = WORDS_PER_SECOND) -> float:
    """Estimated time to deliver the lines naturally, including pauses."""
    if not lines:
        return 0.0
    words = sum(len(line.split()) for line in lines)
    speech = words / words_per_second if words_per_second > 0 else 0.0
    pauses = SECONDS_BETWEEN_LINES * (len(lines) - 1)  # natural beat between speakers/lines
    return round(speech + pauses, 2)


def dialogue_fit_for_shot(shot: Any, default_seconds: float, overage_tolerance: float = OVERAGE_TOLERANCE) -> Dict[str, Any]:
    """Fit report for one shot (dict or model with audio_desc/duration_seconds).

    Status bands: ``fits`` when speech lands inside the planning headroom,
    ``tight`` when it runs up to ``overage_tolerance`` × the budget (accepted
    — models pace lines naturally inside the clip), ``over`` beyond that
    (hard fail: the lines cannot complete).
    """
    if isinstance(shot, dict):
        audio_desc = shot.get("audio_desc", "")
        duration = shot.get("duration_seconds")
    else:
        audio_desc = getattr(shot, "audio_desc", "") or ""
        duration = getattr(shot, "duration_seconds", None)
    budget = float(duration) if isinstance(duration, (int, float)) and duration > 0 else float(default_seconds)
    lines = extract_dialogue_lines(audio_desc)
    needed = estimate_speech_seconds(lines)
    tolerance = max(1.0, float(overage_tolerance))
    if not lines or needed <= budget * (1.0 - REQUIRED_HEADROOM):
        status = "fits"
    elif needed <= budget * tolerance:
        status = "tight"
    else:
        status = "over"
    recommended = None
    if status != "fits":
        # Round up to a whole second with headroom, minimum budget + 1.
        recommended = max(int(budget) + 1, int(needed * (1.0 + REQUIRED_HEADROOM)) + 1)
    return {
        "lines": lines,
        "line_count": len(lines),
        "words": sum(len(line.split()) for line in lines),
        "estimated_speech_seconds": needed,
        "budget_seconds": budget,
        "status": status,
        "recommended_seconds": recommended,
    }


def _needs_extension(fit: Dict[str, Any]) -> bool:
    """Speech exceeding the budget needs more time - tight or over.

    A "tight" shot whose speech estimate still fits INSIDE the budget is
    fine (the model paces it); a tight shot whose speech overshoots the
    budget gets extended like an over shot, wherever the clip cap allows.
    """
    return fit["status"] == "over" or (
        fit["status"] == "tight" and fit["estimated_speech_seconds"] > fit["budget_seconds"]
    )


def dialogue_fit_report(shots: List[Any], default_seconds: float, overage_tolerance: float = OVERAGE_TOLERANCE) -> Dict[str, Any]:
    per_shot = [dialogue_fit_for_shot(shot, default_seconds, overage_tolerance) for shot in shots]
    over = [index for index, fit in enumerate(per_shot) if fit["status"] == "over"]
    tight = [index for index, fit in enumerate(per_shot) if fit["status"] == "tight"]
    extension = [index for index, fit in enumerate(per_shot) if _needs_extension(fit)]
    return {
        "per_shot": per_shot,
        "over_shots": over,
        "tight_shots": tight,
        "extension_shots": extension,
        "all_fit": not over,
    }


def dialogue_fit_error_message(report: Dict[str, Any], max_clip_seconds: float | None = None) -> str | None:
    """Actionable error listing each unfittable shot with its recommendation.

    When ``max_clip_seconds`` is given and a shot's recommended duration
    exceeds it, the recommendation becomes splitting the shot: the video
    model cannot generate a longer clip in one take, so extending is not an
    option and a silent cut (the observed failure) is the only alternative.
    """
    if report["all_fit"]:
        return None
    parts = []
    for index in report["over_shots"]:
        fit = report["per_shot"][index]
        lines = " / ".join(f'"{line}"' for line in fit["lines"])
        recommended = fit.get("recommended_seconds")
        if max_clip_seconds is not None and isinstance(recommended, (int, float)) and recommended > max_clip_seconds:
            advice = (
                f"Recommended: split this shot into two shots — one clip can be at most "
                f"{max_clip_seconds:g}s, so extending to {recommended:g}s would silently cut the dialogue."
            )
        else:
            advice = f"Recommended: extend this shot to {recommended}s (or split its lines into separate shots)."
        parts.append(
            f"Shot {index}: {fit['words']} words of dialogue ({lines}) need about {fit['estimated_speech_seconds']:g}s "
            f"but the shot is only {fit['budget_seconds']:g}s — the lines cannot complete and will be cut. "
            + advice
        )
    return "Dialogue cannot fit its shot budgets. " + " ".join(parts)


# A donated-from shot is never shrunk below this — clips under ~2s read as
# glitches, whatever their dialogue content.
MIN_DONOR_SECONDS = 2.0


def _set_duration_seconds(shot: Any, seconds: float) -> None:
    if isinstance(shot, dict):
        shot["duration_seconds"] = seconds
    else:
        setattr(shot, "duration_seconds", seconds)


def _effective_budget(shot: Any, default_seconds: float, per_shot_cap: float) -> float:
    duration = shot.get("duration_seconds") if isinstance(shot, dict) else getattr(shot, "duration_seconds", None)
    if isinstance(duration, (int, float)) and duration > 0:
        return min(float(duration), per_shot_cap)
    return float(default_seconds)


def rebalance_over_budget_shots(
    shots: List[Any],
    default_seconds: float,
    max_total_seconds: float,
    per_shot_cap: float = 60.0,
) -> List[Dict[str, Any]]:
    """Fit over-budget dialogue into the plan, keeping the requested total.

    Like hosted video tools (Flow/Veo), the dialogue must land inside the
    requested time, not stretch it. Two strategies, tried in order:

    1. **Rebalance**: borrow the missing seconds from dialogue-light shots in
       the same plan (never below their own speech time or ``MIN_DONOR_SECONDS``),
       so the plan's total — the duration the user asked for — is unchanged.
    2. **Extend**: grow the total up to ``max_total_seconds`` (see
       :func:`extend_over_budget_shots`).

    All-or-nothing: when neither strategy can fit every line, nothing is
    changed and ``[]`` is returned so the caller's validation raises with the
    split-lines recommendation instead. Returns the applied adjustments;
    donor records carry ``"source": "donor"``.
    """
    report = dialogue_fit_report(shots, default_seconds)
    if not report["extension_shots"]:
        return []

    targets: Dict[int, float] = {}
    deficit = 0.0
    for index in report["extension_shots"]:
        fit = report["per_shot"][index]
        recommended = float(fit.get("recommended_seconds") or (fit["estimated_speech_seconds"] + 0.5))
        target = min(recommended, per_shot_cap)
        if target <= fit["budget_seconds"]:
            # Cap-bound: an "over" shot beyond the cap still goes through the
            # extend fallback (refusal path); a tight overshoot at the cap is
            # accepted - the model paces it (proven by real Flow renders).
            if fit["status"] == "over":
                return extend_over_budget_shots(shots, default_seconds, max_total_seconds, per_shot_cap)
            continue
        if fit["estimated_speech_seconds"] > target and fit["status"] == "over":
            return extend_over_budget_shots(shots, default_seconds, max_total_seconds, per_shot_cap)
        targets[index] = target
        deficit += target - fit["budget_seconds"]

    donors: List[Tuple[int, float, float]] = []  # (index, budget, spare)
    for index, fit in enumerate(report["per_shot"]):
        if index in targets or _needs_extension(fit):
            continue
        spare = fit["budget_seconds"] - max(fit["estimated_speech_seconds"], MIN_DONOR_SECONDS)
        if spare > 0:
            donors.append((index, fit["budget_seconds"], round(spare, 2)))
    donors.sort(key=lambda item: -item[2])  # richest donor first

    donations: List[Tuple[int, float, float]] = []  # (index, from, to)
    remaining = round(deficit, 2)
    original_values: Dict[int, Any] = {
        index: (shot.get("duration_seconds") if isinstance(shot, dict) else getattr(shot, "duration_seconds", None))
        for index, shot in enumerate(shots)
    }
    for index, budget, spare in donors:
        if remaining <= 1e-6:
            break
        take = min(spare, remaining)
        donations.append((index, budget, round(budget - take, 2)))
        remaining = round(remaining - take, 2)
    if remaining > 1e-6:
        # Donors cannot cover the overage: grow the total within the cap.
        return extend_over_budget_shots(shots, default_seconds, max_total_seconds, per_shot_cap)

    for index, target in targets.items():
        _set_duration_seconds(shots[index], target)
    for index, _from, to in donations:
        _set_duration_seconds(shots[index], to)
    if not dialogue_fit_report(shots, default_seconds)["all_fit"]:  # safety: undo
        for index, value in original_values.items():
            if value is None:
                shot = shots[index]
                if isinstance(shot, dict):
                    shot.pop("duration_seconds", None)
                else:
                    setattr(shot, "duration_seconds", None)
            else:
                _set_duration_seconds(shots[index], float(value))
        return []

    adjustments: List[Dict[str, Any]] = []
    for index, target in targets.items():
        fit = report["per_shot"][index]
        adjustments.append(
            {
                "shot": index,
                "from_seconds": fit["budget_seconds"],
                "to_seconds": target,
                "estimated_speech_seconds": fit["estimated_speech_seconds"],
                "words": fit["words"],
                "source": "rebalance",
            }
        )
    for index, from_budget, to_budget in donations:
        adjustments.append({"shot": index, "from_seconds": from_budget, "to_seconds": to_budget, "source": "donor"})
    return adjustments


def extend_over_budget_shots(
    shots: List[Any],
    default_seconds: float,
    max_total_seconds: float,
    per_shot_cap: float = 60.0,
) -> List[Dict[str, Any]]:
    """Mechanically extend over-budget shots so their dialogue fits.

    Each unfit shot's explicit ``duration_seconds`` becomes the fit report's
    own recommendation (speech time + headroom, rounded up) — the dialogue
    itself is never cut or compressed. All-or-nothing: when any extension
    cannot clear its shot (recommendation above the per-shot cap) or the
    extensions would push the plan past ``max_total_seconds``, nothing is
    changed and ``[]`` is returned so the caller's validation raises with the
    split-lines recommendation instead. Returns the applied adjustments.
    """
    report = dialogue_fit_report(shots, default_seconds)
    if not report["extension_shots"]:
        return []
    planned: List[Dict[str, Any]] = []
    for index in report["extension_shots"]:
        fit = report["per_shot"][index]
        recommended = float(fit.get("recommended_seconds") or (fit["estimated_speech_seconds"] + 0.5))
        target = min(recommended, per_shot_cap)
        if target <= fit["budget_seconds"] or fit["estimated_speech_seconds"] > target:
            if fit["status"] == "tight":
                continue  # tight overshoot at the cap: accepted, the model paces it
            return []
        planned.append(
            {
                "shot": index,
                "from_seconds": fit["budget_seconds"],
                "to_seconds": target,
                "estimated_speech_seconds": fit["estimated_speech_seconds"],
                "words": fit["words"],
            }
        )
    projected_total = round(
        sum(_effective_budget(shot, default_seconds, per_shot_cap) for shot in shots)
        + sum(item["to_seconds"] - item["from_seconds"] for item in planned),
        2,
    )
    if projected_total > max_total_seconds + 1e-6:
        return []
    for item in planned:
        _set_duration_seconds(shots[item["shot"]], item["to_seconds"])
    return planned
