"""Google Gemini Omni Flash video generation via the Interactions API.

Veo preview models were shut down on the Gemini API (October 2026); video
generation now runs through the Interactions API with Gemini Omni Flash
models. This adapter speaks the REST surface directly so it works with the
google-genai SDK version already pinned in this repository.

Docs: https://ai.google.dev/gemini-api/docs/omni
"""

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any, List, Optional

import aiohttp

from interfaces.video_output import VideoOutput
from utils.rate_limiter import RateLimiter

_SUPPORTED_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


class VideoGeneratorOmniGoogleAPI:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-omni-1.1-flash",
        base_url: str = "https://generativelanguage.googleapis.com",
        resolution: str = "720p",
        request_timeout_seconds: float = 600.0,
        max_attempts: int = 3,
        uri_poll_interval_seconds: float = 5.0,
        uri_poll_attempts: int = 120,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.resolution = resolution
        self.request_timeout_seconds = request_timeout_seconds
        self.max_attempts = max_attempts
        self.uri_poll_interval_seconds = uri_poll_interval_seconds
        self.uri_poll_attempts = uri_poll_attempts
        self.rate_limiter = rate_limiter

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

    @staticmethod
    def _image_part(path: str) -> dict:
        data = Path(path).read_bytes()
        mime = _SUPPORTED_MIME_BY_SUFFIX.get(Path(path).suffix.lower(), "image/png")
        return {"type": "image", "data": base64.b64encode(data).decode("ascii"), "mime_type": mime}

    def _build_payload(
        self,
        prompt: str,
        reference_image_paths: List[str],
        resolution: Optional[str],
        aspect_ratio: str,
        duration: Optional[int],
    ) -> dict:
        if len(reference_image_paths) > 3:
            raise ValueError("The number of reference images must be no more than 3")
        input_parts = [self._image_part(path) for path in reference_image_paths]
        text = prompt if duration is None else f"{prompt}\n(Duration: approximately {duration} seconds.)"
        input_parts.append({"type": "text", "text": text})
        return {
            "model": self.model,
            "input": input_parts,
            "response_format": {
                "type": "video",
                "aspect_ratio": aspect_ratio,
                "resolution": resolution or self.resolution,
            },
        }

    @staticmethod
    def _find_video_part(node: Any) -> Optional[dict]:
        """Locate the generated video part anywhere in the interaction response.

        REST responses expose output through the ``steps`` array; the nesting
        has shifted between API revisions, so search recursively for any
        ``{"type": "video", ...}`` part carrying inline data or a URI.
        """
        if isinstance(node, dict):
            if node.get("type") == "video" and (node.get("data") or node.get("uri")):
                return node
            for value in node.values():
                found = VideoGeneratorOmniGoogleAPI._find_video_part(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = VideoGeneratorOmniGoogleAPI._find_video_part(item)
                if found is not None:
                    return found
        return None

    async def generate_single_video(
        self,
        prompt: str,
        reference_image_paths: List[str],
        resolution: Optional[str] = None,
        aspect_ratio: str = "16:9",
        duration: Optional[int] = None,
        **kwargs,
    ) -> VideoOutput:
        payload = self._build_payload(
            prompt=prompt,
            reference_image_paths=reference_image_paths,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            duration=duration,
        )
        logging.info("Calling %s to generate video...", self.model)

        if self.rate_limiter:
            await self.rate_limiter.acquire()

        url = f"{self.base_url}/v1beta/interactions"
        timeout = aiohttp.ClientTimeout(total=self.request_timeout_seconds)
        last_error: Exception | None = None
        interaction: dict | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, headers=self._headers(), json=payload) as response:
                        body_text = await response.text()
                        if response.status >= 400:
                            message = f"Omni video generation failed with HTTP {response.status}: {body_text[:500]}"
                            if response.status >= 500 and attempt < self.max_attempts:
                                logging.error("%s (attempt %s/%s)", message, attempt, self.max_attempts)
                                await asyncio.sleep(attempt)
                                continue
                            raise RuntimeError(message)
                        interaction = json.loads(body_text)
                break
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = exc
                logging.error(
                    "Error occurred while calling %s (attempt %s/%s): %s",
                    self.model, attempt, self.max_attempts, exc,
                )
                if attempt < self.max_attempts:
                    await asyncio.sleep(attempt)
                continue

        if interaction is None:
            raise RuntimeError(f"Failed to generate video with {self.model} after {self.max_attempts} attempts.") from last_error

        video_part = self._find_video_part(interaction)
        if video_part is None:
            status = interaction.get("status", "unknown")
            raise RuntimeError(f"Video generation returned no video output (status: {status}): {str(interaction)[:500]}")

        if video_part.get("data"):
            return VideoOutput(fmt="bytes", ext="mp4", data=base64.b64decode(video_part["data"]))

        # URI delivery (videos larger than 4MB): poll the Files API until the
        # file is ACTIVE, then download the bytes.
        uri = str(video_part["uri"])
        file_name = uri.rstrip("/").split("/")[-1].split("?")[0]
        files_url = f"{self.base_url}/v1beta/files/{file_name}"
        download_url = f"{files_url}:download?alt=media"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.request_timeout_seconds)) as session:
            for poll in range(self.uri_poll_attempts):
                async with session.get(files_url, headers=self._headers()) as response:
                    file_info = await response.json(content_type=None)
                state = file_info.get("state")
                if isinstance(state, dict):
                    state = state.get("name") or state.get("state")
                if state == "ACTIVE" or state == "STATE_ACTIVE":
                    break
                if state in {"FAILED", "ERROR"}:
                    raise RuntimeError(f"Omni video file processing failed: {file_info}")
                logging.info("Omni video file status: %s, waiting %s seconds...", state, self.uri_poll_interval_seconds)
                await asyncio.sleep(self.uri_poll_interval_seconds)
            else:
                raise TimeoutError(f"Omni video file did not become ACTIVE after {self.uri_poll_attempts} polls.")
            async with session.get(download_url, headers=self._headers()) as response:
                if response.status >= 400:
                    raise RuntimeError(f"Omni video download failed with HTTP {response.status}: {await response.text()}")
                data = await response.read()
        return VideoOutput(fmt="bytes", ext="mp4", data=data)
