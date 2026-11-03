import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent_runtime.llm import _llm_429_max_wait_seconds, _retry_after_seconds

GOOGLE_429_MESSAGE = (
    "Error code: 429 - [{'error': {'code': 429, 'message': 'You exceeded your current quota ... "
    "* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_paid_tier_2_input_token_count, "
    "limit: 3000000, model: gemini-3.8-flash ...\\nPlease retry in 49.986426327s.', 'status': 'RESOURCE_EXHAUSTED', "
    "'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{}]}, "
    "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '49s'}]}}]"
)


class RetryAfterParserTests(unittest.TestCase):
    def test_parses_google_body_retry_in(self):
        exc = SimpleNamespace(status_code=429, args=(GOOGLE_429_MESSAGE,))
        self.assertAlmostEqual(_retry_after_seconds(exc), 49.986426327, places=3)

    def test_parses_retry_delay_field(self):
        exc = SimpleNamespace(status_code=429, args=("quota ... 'retryDelay': '37s'}",))
        self.assertEqual(_retry_after_seconds(exc), 37.0)

    def test_parses_http_retry_after_header(self):
        exc = SimpleNamespace(status_code=429, response=SimpleNamespace(headers={"retry-after": "23"}), args=("rate limited",))
        self.assertEqual(_retry_after_seconds(exc), 23.0)

    def test_no_advice_returns_none(self):
        self.assertIsNone(_retry_after_seconds(SimpleNamespace(status_code=500, args=("server exploded",))))
        self.assertIsNone(_retry_after_seconds(ValueError("nope")))

    def test_max_wait_env_override(self):
        with patch("agent_runtime.llm.os.environ", {"VIMAX_LLM_429_MAX_WAIT_SECONDS": "120"}):
            self.assertEqual(_llm_429_max_wait_seconds(), 120.0)
        with patch("agent_runtime.llm.os.environ", {"VIMAX_LLM_429_MAX_WAIT_SECONDS": "9999"}):
            self.assertEqual(_llm_429_max_wait_seconds(), 300.0)
        with patch("agent_runtime.llm.os.environ", {}):
            self.assertEqual(_llm_429_max_wait_seconds(), 65.0)


if __name__ == "__main__":
    unittest.main()
