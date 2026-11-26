import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_runtime.session_index import SessionIndex


class SessionIndexTests(unittest.TestCase):
    def test_generated_session_id_round_trips_after_slug_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="A red ball rolls across a white table.")
            self.assertIsNotNone(index.get(record["session_id"]))
            self.assertEqual(index.working_dir(record["session_id"]).name, record["session_id"])

    def test_create_session_and_checklist(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="Moon cat", user_requirement="short", style="anime")
            self.assertEqual(index.active()["session_id"], record["session_id"])
            working_dir = Path(tmp) / record["working_dir"]
            self.assertTrue((working_dir / "idea2video").exists())
            self.assertTrue((working_dir / "script2video").exists())
            checklist = index.artifact_checklist(record["session_id"])
            self.assertFalse(checklist["script2video/storyboard.json"])
            self.assertFalse(checklist["idea2video/scene_*/storyboard.json"])
            self.assertEqual(record["compacted_summary"], "")
            self.assertEqual(record["compaction_snapshots"], [])

    def test_create_session_preserves_project_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(project_name="Ocean campaign")
            self.assertEqual(record["project_name"], "Ocean campaign")
            self.assertIn("ocean-campaign", record["session_id"])
            self.assertEqual(index.get(record["session_id"])["project_name"], "Ocean campaign")


    def test_session_id_is_sanitized_and_stays_under_working_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(session_id="../../escaped-review")
            self.assertEqual(record["session_id"], "escaped-review")
            working_dir = (Path(tmp) / record["working_dir"]).resolve()
            self.assertTrue(str(working_dir).startswith(str((Path(tmp) / ".working_dir").resolve())))
            self.assertFalse((Path(tmp).parent / "escaped-review").exists())


    def test_update_compaction_writes_session_state_not_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="compact")
            index.update_compaction(record["session_id"], {
                "summary": "## Reference Context Only\n- old context",
                "compacted_message_count": 4,
                "preserved_message_count": 2,
                "estimated_tokens_before": 1000,
                "estimated_tokens_after": 300,
                "reason": "manual",
                "mode": "fallback-local",
            })
            session = index.get(record["session_id"])
            self.assertIn("old context", session["compacted_summary"])
            self.assertEqual(session["compacted_turns"], 2)
            self.assertEqual(session["last_compaction_reason"], "manual")
            self.assertTrue(session["compaction_snapshots"])
            self.assertNotIn("old context", index.memory_text())

    def test_memory_and_turn_record_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create()
            index.write_memory("# User Preferences\n- 16:9\n")
            self.assertIn("16:9", index.memory_text())
            index.append_turn_record(record["session_id"], {"turn_id": "t1", "status": "completed", "tool_rounds": [], "final_assistant_text": "done"})
            self.assertTrue((Path(tmp) / ".vimax" / "logs" / "loop_history.jsonl").exists())
            self.assertEqual(len(index.get(record["session_id"])["recent_turn_records"]), 1)


class LoopLogSafetyTests(unittest.TestCase):
    """p6 (2026-11-26): loop_history.jsonl reached 365MB because full tool
    results were logged verbatim forever; reading it back then exploded the
    agent's context. Slim the writes and rotate the file when huge."""

    def test_turn_record_logged_with_clipped_tool_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create()
            fat = {
                "turn_id": "t1",
                "status": "completed",
                "raw_user_input": "go",
                "tool_rounds": [
                    {"tool_round": 1, "requested_tools": [], "tool_results": [{"name": "read_file", "ok": True, "content": "F" * 40_000}]},
                ],
                "assistant_turns": [{"phase": "initial", "text": "A" * 30_000, "tool_calls": []}],
                "final_assistant_text": "B" * 30_000,
            }
            index.append_turn_record(record["session_id"], fat)
            log_text = (Path(tmp) / ".vimax" / "logs" / "loop_history.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("F" * 1_000, log_text)
            self.assertIn("...[clipped in log]", log_text)
            logged = json.loads(log_text.strip().splitlines()[-1])
            self.assertEqual(logged["tool_rounds"][0]["tool_results"][0]["name"], "read_file")
            self.assertLessEqual(len(logged["tool_rounds"][0]["tool_results"][0]["content"]), 700)
            # The in-memory/session record keeps its bounded preview fields only.
            preview = index.get(record["session_id"])["recent_turn_records"][0]
            self.assertEqual(preview["turn_id"], "t1")

    def test_append_log_archives_oversized_file_under_a_dated_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            logs = Path(tmp) / ".vimax" / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            live = logs / "loop_history.jsonl"
            live.write_text("old" * 600_000, encoding="utf-8")  # 1.8MB > the 1MB test cap
            with patch.dict("os.environ", {"VIMAX_LOG_ROTATE_BYTES": "1000000"}):
                index.append_log("loop_history", {"session_id": "s", "event": "fresh"})
            archives = sorted(logs.glob("loop_history-*.jsonl"))
            self.assertEqual(len(archives), 1)
            self.assertIn("old", archives[0].read_text(encoding="utf-8"))
            fresh = live.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(fresh), 1)
            self.assertIn("fresh", fresh[0])

    def test_append_log_archives_previous_day_date_wise(self):
        import os
        from datetime import datetime, timedelta

        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            logs = Path(tmp) / ".vimax" / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            live = logs / "loop_history.jsonl"
            live.write_text('{"event": "yesterday"}\n', encoding="utf-8")
            yesterday = datetime.now() - timedelta(days=1)
            stamp = yesterday.strftime("%Y-%m-%d")
            os.utime(live, (yesterday.timestamp(), yesterday.timestamp()))
            index.append_log("loop_history", {"session_id": "s", "event": "today"})
            archive = logs / f"loop_history-{stamp}.jsonl"
            self.assertTrue(archive.exists(), f"expected {archive}")
            self.assertIn("yesterday", archive.read_text(encoding="utf-8"))
            self.assertIn("today", live.read_text(encoding="utf-8"))
