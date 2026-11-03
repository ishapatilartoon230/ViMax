# Some chat models (observed with gemini-flash-lite via the OpenAI-compatible
# endpoint) frequently emit a trailing comma before a closing `}`/`]` in
# structured JSON responses (e.g. `"variation_reason": "...",\n}`). That is
# invalid JSON, so PydanticOutputParser raises OutputParserException even
# though the payload is otherwise well-formed and semantically complete --
# and resampling burns LLM calls while often failing the same way again.
# Wrap PydanticOutputParser so a parse failure retries locally with trailing
# commas stripped before giving up.
import json
import re
from typing import Any, List, Optional

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import Generation

_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def strip_trailing_commas(text: str) -> str:
    return _TRAILING_COMMA_RE.sub(r"\1", text)


def parse_json(text: str) -> Optional[dict]:
    """Best-effort parse of an LLM JSON reply.

    Tolerates markdown code fences and trailing commas; returns None when the
    text contains no parseable JSON object (callers decide the fallback).
    """
    candidate = text.strip()
    fence = _CODE_FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
    if not candidate.startswith("{"):
        start = candidate.find("{")
        if start == -1:
            return None
        candidate = candidate[start:]
    for attempt in (candidate, strip_trailing_commas(candidate)):
        try:
            payload = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


class TrailingCommaTolerantPydanticOutputParser(PydanticOutputParser):
    """PydanticOutputParser that retries once with trailing commas stripped."""

    def parse_result(self, result: List[Generation], *, partial: bool = False):
        try:
            return super().parse_result(result, partial=partial)
        except OutputParserException:
            if not result:
                raise
            cleaned_text = strip_trailing_commas(result[0].text)
            if cleaned_text == result[0].text:
                raise
            cleaned_result = [Generation(text=cleaned_text)]
            return super().parse_result(cleaned_result, partial=partial)
