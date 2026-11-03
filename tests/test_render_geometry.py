import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from utils.render_geometry import normalize_render_geometry, resolve_render_geometry


def _write_config(root: Path, render_section: dict) -> None:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "agent.local.yaml").write_text(yaml.safe_dump({"render": render_section}), encoding="utf-8")


class RenderGeometryTests(unittest.TestCase):
    def setUp(self):
        from utils import render_geometry

        render_geometry._load_render_section.cache_clear()

    def tearDown(self):
        from utils import render_geometry

        render_geometry._load_render_section.cache_clear()

    def test_defaults_are_landscape(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            geometry = resolve_render_geometry(tmp)
            self.assertEqual(geometry["image_size"], "1600x900")
            self.assertEqual(geometry["image_aspect_ratio"], "16:9")
            self.assertEqual(geometry["video_aspect_ratio"], "16:9")
            self.assertEqual(geometry["video_resolution"], "720p")
            self.assertEqual(geometry["video_seconds"], 8.0)
            self.assertEqual(geometry["max_total_video_seconds"], 40.0)
            self.assertTrue(geometry["chain_frames"])
            self.assertFalse(geometry["allow_portrait"])

    def test_chain_frames_disabled_via_env_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"VIMAX_CHAIN_FRAMES": "false"}, clear=True):
                self.assertFalse(resolve_render_geometry(tmp)["chain_frames"])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            _write_config(Path(tmp), {"chain_frames": False})
            self.assertFalse(resolve_render_geometry(tmp)["chain_frames"])

    def test_max_total_video_seconds_from_env_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"VIMAX_MAX_TOTAL_VIDEO_SECONDS": "60"}, clear=True):
                self.assertEqual(resolve_render_geometry(tmp)["max_total_video_seconds"], 60.0)
            with patch.dict(os.environ, {"VIMAX_MAX_TOTAL_VIDEO_SECONDS": "abc"}, clear=True):
                self.assertEqual(resolve_render_geometry(tmp)["max_total_video_seconds"], 40.0)
            with patch.dict(os.environ, {"VIMAX_MAX_TOTAL_VIDEO_SECONDS": "0"}, clear=True):
                self.assertEqual(resolve_render_geometry(tmp)["max_total_video_seconds"], 40.0)
            with patch.dict(os.environ, {"VIMAX_MAX_TOTAL_VIDEO_SECONDS": "9999"}, clear=True):
                self.assertEqual(resolve_render_geometry(tmp)["max_total_video_seconds"], 600.0)

    def test_video_seconds_from_config_env_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_config(Path(tmp), {"video_seconds": "3"})
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(resolve_render_geometry(tmp)["video_seconds"], 3.0)
            with patch.dict(os.environ, {"VIMAX_VIDEO_SECONDS": "4"}, clear=True):
                self.assertEqual(resolve_render_geometry(tmp)["video_seconds"], 4.0)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VIMAX_VIDEO_SECONDS": "nope"}, clear=True):
            self.assertEqual(resolve_render_geometry(tmp)["video_seconds"], 8.0)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VIMAX_VIDEO_SECONDS": "0"}, clear=True):
            self.assertEqual(resolve_render_geometry(tmp)["video_seconds"], 8.0)

    def test_workspace_config_selects_vertical(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            _write_config(Path(tmp), {
                "image_aspect_ratio": "9:16",
                "image_size": "1080x1920",
                "video_aspect_ratio": "9:16",
                "video_resolution": "1080p",
                "allow_portrait": True,
            })
            geometry = resolve_render_geometry(tmp)
            self.assertEqual(geometry["image_aspect_ratio"], "9:16")
            self.assertEqual(geometry["image_size"], "1080x1920")
            self.assertEqual(geometry["video_resolution"], "1080p")
            self.assertTrue(geometry["allow_portrait"])

    def test_environment_overrides_workspace_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_config(Path(tmp), {"video_resolution": "1080p"})
            with patch.dict(os.environ, {"VIMAX_VIDEO_RESOLUTION": "720p", "VIMAX_ALLOW_PORTRAIT": "true"}, clear=True):
                geometry = resolve_render_geometry(tmp)
                self.assertEqual(geometry["video_resolution"], "720p")
                self.assertTrue(geometry["allow_portrait"])

    def test_string_allow_portrait_from_web_config(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            _write_config(Path(tmp), {"allow_portrait": "true"})
            self.assertTrue(resolve_render_geometry(tmp)["allow_portrait"])

    def test_normalize_merges_partial_geometry_over_defaults(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            geometry = normalize_render_geometry({"video_aspect_ratio": "9:16"}, workspace_root=tmp)
            self.assertEqual(geometry["video_aspect_ratio"], "9:16")
            self.assertEqual(geometry["image_aspect_ratio"], "16:9")
            self.assertFalse(geometry["allow_portrait"])


if __name__ == "__main__":
    unittest.main()
