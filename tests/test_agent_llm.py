import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agent_runtime.llm import OpenAICompatibleLLM
from agent_runtime.loop import _openai_tool_call


class AgentLLMTests(unittest.IsolatedAsyncioTestCase):
    async def test_string_response_retries_before_clear_error(self):
        llm = OpenAICompatibleLLM(model="test", base_url="https://example.invalid/v1", api_key="test-key")
        create = AsyncMock(side_effect=["data: [DONE]", "bad response"])
        llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with self.assertRaisesRegex(RuntimeError, "returned a string"):
            await llm.complete([], [])
        self.assertEqual(create.await_count, 2)

    async def test_string_response_retry_can_recover(self):
        llm = OpenAICompatibleLLM(model="test", base_url="https://example.invalid/v1", api_key="test-key")
        create = AsyncMock(side_effect=["data: [DONE]", {"choices": [{"message": {"content": "recovered", "tool_calls": []}}]}])
        llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        message = await llm.complete([], [])
        self.assertEqual(message.text, "recovered")
        self.assertEqual(create.await_count, 2)

    async def test_tool_request_falls_back_to_plain_chat_after_bad_tool_responses(self):
        llm = OpenAICompatibleLLM(model="test", base_url="https://example.invalid/v1", api_key="test-key")
        create = AsyncMock(side_effect=[
            "data: [DONE]",
            "data: [DONE]",
            {"choices": [{"message": {"content": "plain fallback", "tool_calls": []}}]},
        ])
        llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        message = await llm.complete([], [{"type": "function", "function": {"name": "x", "parameters": {}}}])
        self.assertEqual(message.text, "plain fallback")
        self.assertEqual(create.await_count, 3)
        self.assertIsNone(create.await_args_list[-1].kwargs.get("tools"))

    async def test_dict_response_is_accepted(self):
        llm = OpenAICompatibleLLM(model="test", base_url="https://example.invalid/v1", api_key="test-key")
        llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value={
            "choices": [{"message": {"content": "hello", "tool_calls": []}}]
        }))))
        message = await llm.complete([], [])
        self.assertEqual(message.text, "hello")

    async def test_thought_signature_is_captured_and_roundtripped(self):
        llm = OpenAICompatibleLLM(model="test", base_url="https://example.invalid/v1", api_key="test-key")
        llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value={
            "choices": [{"message": {"content": "", "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "list_files", "arguments": "{}"},
                "extra_content": {"google": {"thought_signature": "SIG123"}},
            }]}}]}
        ))))
        message = await llm.complete([], [])
        call = message.tool_calls[0]
        self.assertEqual(call.thought_signature, "SIG123")
        # The signature never leaks into logged dicts...
        self.assertNotIn("thought_signature", call.as_dict())
        # ...but the replay payload carries it for Gemini thinking models.
        self.assertEqual(_openai_tool_call(call)["extra_content"], {"google": {"thought_signature": "SIG123"}})

    async def test_replay_payload_omits_signature_when_absent(self):
        from agent_runtime.models import ToolCall

        payload = _openai_tool_call(ToolCall(name="list_files", arguments={}, id="call_2"))
        self.assertNotIn("extra_content", payload)


if __name__ == "__main__":
    unittest.main()
