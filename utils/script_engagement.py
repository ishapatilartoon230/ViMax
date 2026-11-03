"""Mechanical dialogue engagement checks.

Ported from short-form writing craft and reduced to what a machine can
verify objectively: **quoted dialogue lines** must not carry "AI tells" —
banned buzzwords, "not just X but Y" constructions, fake-suspense
announcements, and spoken stage directions. Scene/action prose is never
scanned; those rules live in the skill files and the other gates.

Scoring follows the source craft's scheme: a hard failure costs 15 points,
a soft flag 5. A script passes when it has **zero hard failures**; soft
flags are advisory only (logged, never blocking). The English-only
patterns intentionally cannot fire on non-English dialogue (Hindi and
other languages are unaffected — no false failures).

The rhetorical-question rule from the source was dropped on purpose: in
acted dialogue, questions between characters are natural speech, and a
regex cannot tell a real question ("Where is she?") from a rhetorical AI
tell. Same for one-word lines: natural replies ("No.", "Wait.") are
whitelisted; only dramatic lone words ("Gone.", "Silence.") count, softly.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from utils.script_fidelity import extract_quoted_lines

# How many rewrite bounces a pipeline should attempt before proceeding
# with a warning. Engagement is quality, never a render blocker.
MAX_BOUNCES = 2

HARD_COST = 15
SOFT_COST = 5

ONE_WORD_MAX = 1
LONG_SENTENCE_WORDS = 32
EM_DASH_MAX = 4

# One-word lines that are natural spoken replies, not dramatic beats.
_NATURAL_ONE_WORD = {
    "no", "yes", "yeah", "nope", "yep", "wait", "what", "who", "where",
    "when", "why", "how", "stop", "go", "okay", "ok", "hey", "now",
    "please", "thanks", "thank you", "sorry", "fine", "sure", "right",
    "good", "ready", "here", "there", "this", "that", "huh", "hello",
}

_HARD_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?i)\b(delve|unlock|unleash|harness|leverage|seamless|robust"
            r"|cutting[- ]edge|game[- ]?changer|revolutionary|transformative"
            r"|landscape|realm|tapestry|elevate|empower|supercharge"
            r"|deep dive|next level)\b"
        ),
        "banned AI-flavor word",
    ),
    (re.compile(r"(?i)\bnot just\b[^.!?]*\bbut\b"), '"not just X but Y" construction'),
    (re.compile(r"(?i)\bit'?s not\b[^.!?]*\bit'?s\b"), '"it\'s not X, it\'s Y" construction'),
    (re.compile(r"(?i)\b(spoiler|plot twist)\s*:"), "fake-suspense announcement"),
    (re.compile(r"(?i)here'?s where it gets (?:interesting|crazy|wild)"), "fake-suspense announcement"),
    (re.compile(r"(?i)\bthe best part\?"), "fake-suspense announcement"),
    (
        re.compile(
            r"(?i)\b(here'?s the thing|let that sink in|buckle up|in today'?s world"
            r"|whether you'?re a beginner|it'?s worth noting|at the end of the day)\b"
        ),
        "spoken stage direction",
    ),
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+")


def _sentences(line: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(line) if s.strip()]


def engagement_report(script_text: str) -> Dict[str, Any]:
    """Score the quoted dialogue of a script for AI tells.

    Returns ``{"score", "hard_failures", "soft_flags", "dialogue_lines"}``.
    Pass criterion is zero hard failures (see :func:`engagement_error_message`).
    Scripts without quoted dialogue (motion/montage) pass trivially.
    """
    lines = extract_quoted_lines(str(script_text or ""))
    hard: List[str] = []
    soft: List[str] = []

    for line in lines:
        for pattern, reason in _HARD_PATTERNS:
            if pattern.search(line):
                hard.append(f'{reason}: "{line}"')

    all_sentences = [s for line in lines for s in _sentences(line)]

    dramatic = [
        s for s in all_sentences
        if len(s.split()) == 1 and s.strip().strip(".!?").lower() not in _NATURAL_ONE_WORD
    ]
    if len(dramatic) > ONE_WORD_MAX:
        soft.append(
            f"{len(dramatic)} dramatic one-word lines (max {ONE_WORD_MAX}): {dramatic[:3]}"
        )

    long_sentences = [s for s in all_sentences if len(s.split()) > LONG_SENTENCE_WORDS]
    if long_sentences:
        soft.append(
            f'{len(long_sentences)} dialogue sentence(s) over {LONG_SENTENCE_WORDS} words, e.g. "{long_sentences[0][:60]}..."'
        )

    em_dashes = sum(line.count("—") for line in lines)
    if em_dashes > EM_DASH_MAX:
        soft.append(
            f"{em_dashes} em dashes in dialogue (interruptions are fine; keep ~{EM_DASH_MAX} max)"
        )

    score = max(0, 100 - HARD_COST * len(hard) - SOFT_COST * len(soft))
    return {
        "score": score,
        "hard_failures": hard,
        "soft_flags": soft,
        "dialogue_lines": len(lines),
    }


def engagement_error_message(report: Dict[str, Any]) -> str | None:
    """Actionable fix instruction for hard failures; ``None`` when passing.

    The message names each offending line so the rewrite instruction can be
    executed without interpretation, mirroring ``script_fidelity``.
    """
    hard = list(report.get("hard_failures") or [])
    if not hard:
        return None
    listing = "\n".join(f"- {item}" for item in hard[:8])
    return (
        "Dialogue engagement check failed (AI tells found in quoted lines). "
        "Rewrite the offending lines in plain spoken language - concrete nouns, no "
        "buzzwords, no fake-suspense or stage-direction phrases - keeping the same "
        "meaning, speakers, language, and line order:\n"
        f"{listing}"
    )
