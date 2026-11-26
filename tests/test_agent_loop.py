import asyncio
import json
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

from agent_runtime.context_compactor import ContextCompactor
from agent_runtime.llm import AssistantMessage
from agent_runtime.loop import AgentLoop, _is_context_limit_error
from agent_runtime.models import ToolCall, ToolResult
from agent_runtime.prompts import PromptBuilder
from agent_runtime.session_index import SessionIndex
from agent_runtime.tool_executor import ToolExecutor
from agent_runtime.tools import ToolArgumentSchema, ToolRegistry, ToolSpec


class ContextLimitError(Exception):
    pass


class FlakyThenRecoveringLLM:
    """Raises the provider's context-limit 400 once, then serves replies."""

    def __init__(self, replies):
        self.replies = list(replies)

    async def complete(self, messages, tools):
        if self.replies and isinstance(self.replies[0], Exception):
            raise self.replies.pop(0)
        return self.replies.pop(0)


class ContextLimitRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def test_detects_provider_context_limit_errors(self):
        google_message = "Error code: 400 - [{'error': {'code': 400, 'message': 'The input token count exceeds the maximum number of tokens allowed (1048576).', 'status': 'INVALID_ARGUMENT'}}]"
        self.assertTrue(_is_context_limit_error(ContextLimitError(google_message)))
        self.assertTrue(_is_context_limit_error(RuntimeError("context length exceeded")))
        self.assertFalse(_is_context_limit_error(RuntimeError("connection refused")))
        self.assertFalse(_is_context_limit_error(RuntimeError("401 unauthorized")))

    async def test_context_limit_error_compacts_and_retries_instead_of_failing(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            registry = ToolRegistry([])
            llm = FlakyThenRecoveringLLM([
                ContextLimitError("Error code: 400 - input token count exceeds the maximum number of tokens allowed (1048576)"),
                AssistantMessage(text="compacted summary of the turn"),
                AssistantMessage(text="recovered answer"),
            ])
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), llm)
            events = [event async for event in loop.stream_events("long turn")]
            done = [event for event in events if event["type"] == "done"][0]
            self.assertEqual(done["assistant"], "recovered answer")
            statuses = [event.get("phase") for event in events if event["type"] == "status"]
            self.assertIn("compact", statuses)

    async def test_mid_turn_pressure_compacts_between_tool_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)

            def ping(args):
                return ToolResult("ping", True, "pong " * 4000)  # large enough to cross the tiny test threshold

            registry = ToolRegistry([ToolSpec("ping", "Ping", ping, schema={})])
            llm = FlakyThenRecoveringLLM([
                AssistantMessage(tool_calls=[ToolCall(name="ping", arguments={})]),
                AssistantMessage(text="turn summary"),
                AssistantMessage(text="done after tools"),
            ])
            with patch.dict("os.environ", {"VIMAX_CONTEXT_WINDOW_TOKENS": "1500", "VIMAX_AUTO_COMPACT_BUFFER_TOKENS": "50"}):
                loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), llm)
                events = [event async for event in loop.stream_events("run the tool")]
            statuses = [event.get("phase") for event in events if event["type"] == "status"]
            self.assertIn("compact", statuses)
            done = [event for event in events if event["type"] == "done"][0]
            self.assertEqual(done["assistant"], "done after tools")


class ContextSafetyTests(unittest.IsolatedAsyncioTestCase):
    """p6 (2026-11-26): a tool result dragging in a giant log file pushed the
    request to 1.64M tokens and killed the turn with an unrecoverable 400.
    Tool results must be capped for the LLM, and recovery must drop images."""

    async def test_giant_tool_result_is_clipped_for_the_llm_but_full_in_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            giant = "g" * 100_000

            def fat(args):
                return ToolResult("fat", True, giant)

            registry = ToolRegistry([ToolSpec("fat", "Fat result", fat, schema={})])
            llm = CapturingLLM(
                [
                    AssistantMessage(tool_calls=[ToolCall(name="fat", arguments={})]),
                    AssistantMessage(text="done"),
                ]
            )
            with patch.dict("os.environ", {"VIMAX_TOOL_RESULT_MAX_CHARS": "2000"}):
                loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), llm)
                events = [event async for event in loop.stream_events("run it")]
            tool_messages = [message for message in llm.calls[1] if message.get("role") == "tool"]
            self.assertEqual(len(tool_messages), 1)
            self.assertLessEqual(len(tool_messages[0]["content"]), 2_300)
            self.assertIn("tool result truncated", tool_messages[0]["content"])
            event_results = [event for event in events if event["type"] == "tool_result"]
            self.assertGreaterEqual(len(event_results[0]["tool_result"]["content"]), 100_000)

    async def test_context_limit_recovery_drops_image_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            data_url = "data:image/jpeg;base64," + "a" * 50_000

            def view(args):
                return ToolResult(
                    "view_image",
                    True,
                    "image loaded",
                    {"path": "frame.png"},
                    model_content=[{"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}],
                )

            registry = ToolRegistry([ToolSpec("view_image", "View image", view, schema={"path": ToolArgumentSchema(str, True)})])
            recovery_reply = AssistantMessage(text="recovered without the pixels")
            with patch.dict("os.environ", {"VIMAX_TOOL_RESULT_MAX_CHARS": "2000"}):

                class FailOnceThenAnswer:
                    def __init__(self, replies):
                        self.replies = list(replies)
                        self.calls = []

                    async def complete(self, messages, tools):
                        self.calls.append(deepcopy(messages))
                        if len(self.calls) == 2:  # the call right after the image arrived
                            raise ContextLimitError("Error code: 400 - input token count exceeds the maximum number of tokens allowed (1048576)")
                        if self.replies:
                            return self.replies.pop(0)
                        return AssistantMessage(text="recovered without the pixels")

                fail_once = FailOnceThenAnswer(
                    [AssistantMessage(tool_calls=[ToolCall(name="view_image", arguments={"path": "frame.png"})])]
                )
                loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), fail_once, ContextCompactor(None))
                events = [event async for event in loop.stream_events("inspect")]
            done = [event for event in events if event["type"] == "done"][0]
            self.assertEqual(done["assistant"], "recovered without the pixels")
            retry_call = fail_once.calls[2]
            self.assertNotIn("image_url", json.dumps(retry_call))
            self.assertIn("removed from context", json.dumps(retry_call))

    async def test_only_the_newest_image_round_keeps_its_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            first_url = "data:image/jpeg;base64,Zmlyc3Q="
            second_url = "data:image/jpeg;base64,c2Vjb25k"

            def view(args):
                url = second_url if args.get("path") == "two.png" else first_url
                return ToolResult(
                    "view_image",
                    True,
                    "image loaded",
                    {"path": args.get("path")},
                    model_content=[{"type": "image_url", "image_url": {"url": url, "detail": "high"}}],
                )

            registry = ToolRegistry([ToolSpec("view_image", "View image", view, schema={"path": ToolArgumentSchema(str, True)})])
            llm = CapturingLLM(
                [
                    AssistantMessage(tool_calls=[ToolCall(name="view_image", arguments={"path": "one.png"})]),
                    AssistantMessage(tool_calls=[ToolCall(name="view_image", arguments={"path": "two.png"})]),
                    AssistantMessage(text="both inspected"),
                ]
            )
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), llm)
            [event async for event in loop.stream_events("inspect both")]
            third_call = llm.calls[2]
            image_messages = [message for message in third_call if message.get("role") == "user" and isinstance(message.get("content"), list)]
            self.assertEqual(len(image_messages), 1)
            self.assertEqual(image_messages[0]["content"][1]["image_url"]["url"], second_url)
            self.assertIn("removed from context", json.dumps(third_call))


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    async def complete(self, messages, tools):
        return self.replies.pop(0)


class FailingLLM:
    async def complete(self, messages, tools):
        raise RuntimeError("provider returned invalid response shape")


class CapturingLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def complete(self, messages, tools):
        self.calls.append(deepcopy(messages))
        return self.replies.pop(0)


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_tool_call_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            registry = ToolRegistry([])
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), FakeLLM([AssistantMessage(text="done")]))
            events = [event async for event in loop.stream_events("hi")]
            self.assertEqual(events[-2]["type"], "done")
            turn_id = events[0]["turn_id"]
            self.assertTrue(all(event.get("turn_id") == turn_id for event in events))
            log_text = (index.logs_dir / "loop_history.jsonl").read_text(encoding="utf-8")
            self.assertIn("assistant_finished_without_tools", log_text)


    async def test_turn_record_follows_session_created_by_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            old = index.create(idea="old")

            def create_actual(args):
                record = index.create(idea="actual")
                return ToolResult("create_actual", True, record["session_id"])

            registry = ToolRegistry([ToolSpec("create_actual", "Create actual session", create_actual, schema={})])
            llm = FakeLLM([AssistantMessage(tool_calls=[ToolCall(name="create_actual", arguments={})]), AssistantMessage(text="finished")])
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), llm)
            events = [event async for event in loop.stream_events("start new project")]
            active = index.active()
            self.assertNotEqual(active["session_id"], old["session_id"])
            self.assertEqual(len(index.get(active["session_id"])["recent_turn_records"]), 1)
            self.assertEqual(index.get(old["session_id"])["recent_turn_records"], [])
            self.assertEqual(events[-1]["session"]["active_session_id"], active["session_id"])


    async def test_tool_progress_streams_before_tool_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            release = asyncio.Event()

            async def slow_tool(args, runtime):
                runtime.emit_progress("started", stage="running")
                await release.wait()
                return ToolResult("slow_tool", True, "done")

            registry = ToolRegistry([ToolSpec("slow_tool", "Slow tool", slow_tool, schema={})])
            llm = FakeLLM([AssistantMessage(tool_calls=[ToolCall(name="slow_tool", arguments={})]), AssistantMessage(text="finished")])
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), llm)
            agen = loop.stream_events("start")
            seen = []
            while True:
                event = await asyncio.wait_for(anext(agen), timeout=1)
                seen.append(event["type"])
                if event["type"] == "tool_progress":
                    self.assertFalse(release.is_set())
                    break
            release.set()
            async for event in agen:
                seen.append(event["type"])
            self.assertLess(seen.index("tool_progress"), seen.index("tool_result"))


    async def test_preflight_compact_summarizes_old_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            registry = ToolRegistry([])
            compactor = ContextCompactor(None, token_threshold=200, buffer_tokens=0, preserve_last_n=2, summary_max_chars=2000)
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), FakeLLM([AssistantMessage(text="after compact")]), compactor)
            loop.history = [
                {"role": "user", "content": "old request " + "x" * 1200},
                {"role": "assistant", "content": "old answer " + "y" * 1200},
                {"role": "user", "content": "recent request"},
                {"role": "assistant", "content": "recent answer"},
            ]
            events = [event async for event in loop.stream_events("continue")]
            self.assertIn("compact", [event.get("phase") for event in events if event["type"] == "status"])
            session = index.active()
            self.assertIn("Reference Context Only", session["compacted_summary"])
            self.assertGreaterEqual(session["compacted_turns"], 1)
            self.assertTrue(session["compaction_snapshots"])
            self.assertEqual(loop.history[0]["role"], "system")
            self.assertIn("after compact", loop.history[-1]["content"])
            self.assertNotIn("old request", index.memory_text())


    async def test_llm_sampling_error_yields_error_without_crashing_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            registry = ToolRegistry([])
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), FailingLLM())
            events = [event async for event in loop.stream_events("start")]
            self.assertTrue(any(event["type"] == "error" and event.get("metadata", {}).get("error_type") == "llm_sampling_failed" for event in events))
            self.assertEqual(events[-2]["type"], "done")
            self.assertEqual(events[-1]["type"], "session")
            self.assertEqual(index.active()["recent_turn_records"][-1]["status"], "failed")

    async def test_tool_call_continues_then_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)

            def hello(args):
                return ToolResult("hello", True, "hello result")

            registry = ToolRegistry([ToolSpec("hello", "Say hello", hello, schema={"name": ToolArgumentSchema(str, False, "x")})])
            llm = FakeLLM([AssistantMessage(tool_calls=[ToolCall(name="hello", arguments={})]), AssistantMessage(text="finished")])
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), llm)
            events = [event async for event in loop.stream_events("start")]
            self.assertTrue(any(event["type"] == "tool_result" for event in events))
            self.assertEqual(events[-2]["assistant"], "finished")

    async def test_transient_tool_images_reach_next_llm_turn_but_not_events_or_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            data_url = "data:image/jpeg;base64,ZmFrZS1pbWFnZQ=="

            def view(args):
                return ToolResult(
                    "view_image",
                    True,
                    "image loaded",
                    {"path": "idea2video/frame.png"},
                    model_content=[{"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}],
                )

            registry = ToolRegistry([ToolSpec("view_image", "View image", view, schema={"path": ToolArgumentSchema(str, True)})])
            llm = CapturingLLM(
                [
                    AssistantMessage(tool_calls=[ToolCall(name="view_image", arguments={"path": "idea2video/frame.png"})]),
                    AssistantMessage(text="The frame is visible."),
                ]
            )
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), llm)
            events = [event async for event in loop.stream_events("inspect the frame")]

            image_messages = [message for message in llm.calls[1] if message.get("role") == "user" and isinstance(message.get("content"), list)]
            self.assertEqual(len(image_messages), 1)
            self.assertEqual(image_messages[0]["content"][1]["image_url"]["url"], data_url)
            self.assertNotIn(data_url, json.dumps(events))
            self.assertNotIn(data_url, json.dumps(loop.history))
            self.assertNotIn(data_url, (index.logs_dir / "tool_calls.jsonl").read_text(encoding="utf-8"))
