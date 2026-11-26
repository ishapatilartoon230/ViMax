from __future__ import annotations

import json
import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from .context_compactor import ContextCompactor, CompactionResult
from .llm import OpenAICompatibleLLM
from .models import ToolCall, ToolResult, TurnControl
from .prompts import PromptBuilder
from .session_index import SessionIndex
from .tool_executor import ToolExecutor
from .tools import ToolRegistry, build_builtin_registry

MAX_TOOL_PASSES = 50


def _tool_result_max_chars() -> int:
    """Char cap for tool results injected into the LLM context.

    One giant tool result (a 365MB log read) can leapfrog every preflight
    threshold in a single round and hard-fail the provider with a 400 that
    compaction cannot fix (the fat message sits in the preserved tail).
    """
    try:
        return max(1_000, int(os.environ.get("VIMAX_TOOL_RESULT_MAX_CHARS", "24000")))
    except ValueError:
        return 24_000


def _clip_tool_content(text: str) -> str:
    limit = _tool_result_max_chars()
    if len(text) <= limit:
        return text
    notice = (
        f"\n...[tool result truncated to stay inside the model context limit: first {limit} of {len(text)} chars kept; "
        "read specific ranges or grep the file instead of loading it whole]"
    )
    return text[:limit] + notice


def _is_image_message(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return isinstance(content, list) and any(part.get("type") == "image_url" for part in content if isinstance(part, dict))


def _strip_stale_images(messages: list[dict[str, Any]], *, keep_last: int = 1) -> None:
    """Drop image payloads from all but the newest image message(s), in place.

    Inline base64 images are counted as text tokens by some gateways; a couple
    of full-resolution views can blow the input limit on their own.
    """
    indices = [i for i, message in enumerate(messages) if _is_image_message(message)]
    for i in indices[: max(0, len(indices) - max(0, keep_last))]:
        messages[i] = {
            "role": "user",
            "content": "[Tool-provided image(s) from an earlier round were removed from context to stay within the "
            "model input limit. Re-run view_image on the path if those pixels are still needed.]",
        }


def _shrink_oversized_tool_messages(messages: list[dict[str, Any]]) -> None:
    """Apply the tool-result cap to messages already in the context, in place."""
    for i, message in enumerate(messages):
        if message.get("role") == "tool" and isinstance(message.get("content"), str):
            messages[i] = {**message, "content": _clip_tool_content(message["content"])}


def _is_context_limit_error(exc: BaseException) -> bool:
    """Provider 'input too large' errors: retryable only after compaction."""
    text = str(exc).lower()
    markers = (
        "exceeds the maximum number of tokens",
        "input token count exceeds",
        "context length exceeded",
        "contextwindowexceeded",
        "maximum context length",
        "too many input tokens",
    )
    return any(marker in text for marker in markers)


class AgentLoop:
    def __init__(self, session_index: SessionIndex, prompt_builder: PromptBuilder, tool_registry: ToolRegistry, tool_executor: ToolExecutor, llm: Any, context_compactor: ContextCompactor | None = None) -> None:
        self.session_index = session_index
        self.prompt_builder = prompt_builder
        self.tool_registry = tool_registry
        self.tool_executor = tool_executor
        self.llm = llm
        self.context_compactor = context_compactor or ContextCompactor(llm)
        self.history: list[dict[str, Any]] = []

    async def compact_history(self, *, reason: str = "manual") -> str:
        if not self.history:
            return "No conversation history to compact."
        session = self.session_index.active() or self.session_index.create()
        result = await self.context_compactor.compact(
            self.history,
            previous_summary=str(session.get("compacted_summary", "") or ""),
            reason=reason,
        )
        self.history = [self.context_compactor.synthetic_summary_message(result.summary), *result.preserved_messages]
        self.session_index.update_compaction(session["session_id"], _compaction_record(result))
        return f"Compacted context {result.estimated_tokens_before} -> {result.estimated_tokens_after} ({result.mode})."

    async def _compact_runtime_messages(self, runtime_messages: list[dict[str, Any]], system: str, *, reason: str) -> list[dict[str, Any]]:
        """Compact the live turn's message list in place of failing it.

        Summarizes the conversation so far (keeping the system prompt and the
        most recent messages) so sampling can continue after token pressure
        instead of the provider rejecting the input outright.
        """
        session = self.session_index.active() or self.session_index.create()
        result = await self.context_compactor.compact(
            runtime_messages,
            previous_summary=str(session.get("compacted_summary", "") or ""),
            reason=reason,
        )
        self.session_index.update_compaction(session["session_id"], _compaction_record(result))
        preserved = [message for message in result.preserved_messages if message.get("role") != "system"]
        return [{"role": "system", "content": system}, self.context_compactor.synthetic_summary_message(result.summary), *preserved]

    async def stream_events(self, user_input: str) -> AsyncIterator[dict[str, Any]]:
        control = TurnControl()
        yield {"type": "turn", "turn_id": control.turn_id, "turn": {"id": control.turn_id}}
        tool_schemas = self.tool_registry.list_function_tools()
        parts = self.prompt_builder.build_parts(user_input)
        system = "\n\n".join(f"## {part.title}\n{part.body}" for part in parts if part.id != "request.user")
        if self.context_compactor.should_preflight_compact(
            [*self.history, {"role": "user", "content": user_input}],
            system_tokens=_prompt_tokens(parts),
            tools_tokens=_tool_schema_tokens(tool_schemas),
        ):
            yield {"type": "status", "turn_id": control.turn_id, "phase": "compact", "message": "Compacting context before sampling"}
            await self.compact_history(reason="token-pressure")
            parts = self.prompt_builder.build_parts(user_input)
            system = "\n\n".join(f"## {part.title}\n{part.body}" for part in parts if part.id != "request.user")
        yield {"type": "prompt_trace", "turn_id": control.turn_id, "prompt_trace": self.prompt_builder.trace(parts)}
        runtime_messages: list[dict[str, Any]] = [{"role": "system", "content": system}, *self.history, {"role": "user", "content": user_input}]
        assistant_turns: list[dict[str, Any]] = []
        tool_rounds: list[dict[str, Any]] = []
        transitions: list[dict[str, str]] = []
        all_tool_results: list[ToolResult] = []
        final_text = ""
        status = "completed"
        tool_round = 0
        context_recovered = False

        while True:
            # Mid-turn pressure check: compaction only at turn start lets a
            # long tool loop grow the context past the model's input limit and
            # hard-fail the turn with a 400. Compact between tool rounds too.
            if tool_round > 0 and self.context_compactor.should_preflight_compact(runtime_messages, system_tokens=_prompt_tokens(parts), tools_tokens=_tool_schema_tokens(tool_schemas)):
                yield {"type": "status", "turn_id": control.turn_id, "phase": "compact", "message": "Compacting context mid-turn"}
                runtime_messages = await self._compact_runtime_messages(runtime_messages, system, reason="mid-turn-token-pressure")

            yield {"type": "status", "turn_id": control.turn_id, "phase": "sampling_assistant", "message": "Sampling assistant"}
            try:
                assistant = await self.llm.complete(runtime_messages, tools=tool_schemas)
            except Exception as exc:
                if _is_context_limit_error(exc) and not context_recovered:
                    # The provider refused the input as too large. Shrink the
                    # fat payloads FIRST (giant tool results, stale images) —
                    # they usually sit in the compactor's preserved tail, so
                    # compaction alone cannot remove them and every retry
                    # would fail with the same 400.
                    context_recovered = True
                    yield {"type": "status", "turn_id": control.turn_id, "phase": "compact", "message": "Context limit hit; shrinking oversized content and retrying"}
                    _strip_stale_images(runtime_messages, keep_last=0)
                    _shrink_oversized_tool_messages(runtime_messages)
                    try:
                        runtime_messages = await self._compact_runtime_messages(runtime_messages, system, reason="context-limit-recovery")
                    except Exception:
                        pass
                    continue
                status = "failed"
                final_text = f"Agent LLM request failed: {exc}"
                transitions.append(_transition("sampling_assistant", "finalizing_answer", "llm_sampling_failed"))
                yield {"type": "error", "turn_id": control.turn_id, "message": final_text, "metadata": {"error_type": "llm_sampling_failed"}}
                break
            assistant_turns.append({"phase": "initial" if tool_round == 0 else f"followup_{tool_round}", "text": assistant.text, "tool_calls": [call.as_dict() for call in assistant.tool_calls]})
            if not assistant.tool_calls:
                transitions.append(_transition("sampling_assistant", "finalizing_answer", "assistant_finished_without_tools"))
                final_text = assistant.text
                if final_text:
                    yield {"type": "token", "turn_id": control.turn_id, "delta": final_text}
                break
            transitions.append(_transition("sampling_assistant", "executing_tools", "assistant_requested_tools"))
            if tool_round >= MAX_TOOL_PASSES:
                status = "halted"
                final_text = "Tool loop halted after max tool passes."
                transitions.append(_transition("executing_tools", "finalizing_answer", "max_tool_passes_reached"))
                yield {"type": "error", "turn_id": control.turn_id, "message": final_text, "metadata": {"max_tool_passes": MAX_TOOL_PASSES}}
                break
            tool_round += 1
            yield {"type": "status", "turn_id": control.turn_id, "phase": "executing_tools", "message": f"Running tools (round {tool_round})"}
            runtime_messages.append({"role": "assistant", "content": assistant.text or "", "tool_calls": [_openai_tool_call(call) for call in assistant.tool_calls]})
            round_results: list[ToolResult] = []
            round_model_content: list[dict[str, Any]] = []

            for call in assistant.tool_calls:
                yield {"type": "tool_start", "turn_id": control.turn_id, "tool": call.as_dict()}
                progress_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

                def on_progress(event: dict[str, Any]) -> None:
                    progress_queue.put_nowait(event)

                task = asyncio.create_task(self.tool_executor.execute(call, control, progress_callback=on_progress))
                while not task.done():
                    try:
                        yield await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                while not progress_queue.empty():
                    yield progress_queue.get_nowait()
                record = await task
                result = record.result
                round_results.append(result)
                all_tool_results.append(result)
                yield {"type": "tool_result", "turn_id": control.turn_id, "tool_result": result.as_dict()}
                runtime_messages.append({"role": "tool", "tool_call_id": call.id, "name": result.name, "content": _clip_tool_content(json.dumps(result.as_dict(), ensure_ascii=False))})
                if result.model_content:
                    round_model_content.extend(result.model_content)
            if round_model_content:
                # Only the newest round keeps its pixels: older image payloads
                # count as text tokens on several gateways and would pile up.
                _strip_stale_images(runtime_messages, keep_last=0)
                runtime_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Tool-provided image observation(s). Inspect these pixels as evidence for the active task; this is not a new user request.",
                            },
                            *round_model_content,
                        ],
                    }
                )
            tool_rounds.append({"tool_round": tool_round, "requested_tools": [call.as_dict() for call in assistant.tool_calls], "tool_results": [result.as_dict() for result in round_results]})
            transitions.append(_transition("executing_tools", "post_tool_decision", "tool_round_completed"))
            transitions.append(_transition("post_tool_decision", "sampling_assistant", "runtime_continuation_after_tools"))

        self.history.extend([{"role": "user", "content": user_input}, {"role": "assistant", "content": final_text}])
        turn_record = {"turn_id": control.turn_id, "status": status, "raw_user_input": user_input, "assistant_turns": assistant_turns, "tool_rounds": tool_rounds, "transitions": transitions, "final_assistant_text": final_text, "created_at": datetime.now().isoformat(timespec="seconds")}
        final_session = self.session_index.active() or self.session_index.create()
        self.session_index.append_turn_record(final_session["session_id"], turn_record)
        yield {"type": "done", "turn_id": control.turn_id, "assistant": final_text, "tool_results": [result.as_dict() for result in all_tool_results]}
        yield {"type": "session", "turn_id": control.turn_id, "session": self.session_index.snapshot()}


def _compaction_record(result: CompactionResult) -> dict[str, Any]:
    return {
        "summary": result.summary,
        "preserved_message_count": len(result.preserved_messages),
        "compacted_message_count": result.compacted_message_count,
        "estimated_tokens_before": result.estimated_tokens_before,
        "estimated_tokens_after": result.estimated_tokens_after,
        "reason": result.reason,
        "mode": result.mode,
        "created_at": result.created_at,
    }


def _prompt_tokens(parts: list[Any]) -> int:
    return sum(max(1, len(str(getattr(part, "body", ""))) // 4) for part in parts)


def _tool_schema_tokens(tool_schemas: list[dict[str, Any]]) -> int:
    try:
        return max(0, len(json.dumps(tool_schemas, ensure_ascii=False, default=str)) // 4)
    except TypeError:
        return max(0, len(str(tool_schemas)) // 4)


def _transition(src: str, dst: str, reason: str) -> dict[str, str]:
    return {"from": src, "to": dst, "reason": reason}


def _openai_tool_call(call: ToolCall) -> dict[str, Any]:
    payload: dict[str, Any] = {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}}
    if call.thought_signature:
        # Gemini thinking models attach a thought signature to function calls;
        # replaying the call without it is rejected with HTTP 400.
        payload["extra_content"] = {"google": {"thought_signature": call.thought_signature}}
    return payload


def build_runtime(workspace_root: str | Path = ".", llm: Any | None = None, adapter_specs: list[Any] | None = None) -> AgentLoop:
    from .vimax_adapters import build_vimax_adapter_specs
    root = Path(workspace_root).resolve()
    session_index = SessionIndex(root)
    specs = adapter_specs if adapter_specs is not None else build_vimax_adapter_specs(root, session_index)
    registry = build_builtin_registry(root, session_index, specs)
    executor = ToolExecutor(registry, session_index)
    prompt_builder = PromptBuilder(root / "prompts", session_index, registry)
    resolved_llm = llm or OpenAICompatibleLLM()
    return AgentLoop(session_index, prompt_builder, registry, executor, resolved_llm, ContextCompactor(resolved_llm))
