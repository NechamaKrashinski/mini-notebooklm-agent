from __future__ import annotations

import os
import uuid
from pathlib import Path

import httpx

MAX_CHUNK_CHARS = 800


def _chunk_lines(lines: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """Group consecutive lines into chunks of up to ~MAX_CHUNK_CHARS characters."""
    chunks: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    size = 0
    for text, voice in lines:
        if current and size + len(text) > MAX_CHUNK_CHARS:
            chunks.append(current)
            current, size = [], 0
        current.append((text, voice))
        size += len(text)
    if current:
        chunks.append(current)
    return chunks


class CloudRunTTS:
    """Synthesizes through the personal Cloud Run service (see tts-services/).

    Only text goes out and only metadata plus a Drive link comes back, so no
    audio bytes cross a content-filtered connection. The result is a URL, not
    a local file: `path` is only used to derive the Drive file name.
    """

    def __init__(self) -> None:
        self._url = os.environ["TTS_SERVICE_URL"].rstrip("/")
        self._headers = {"x-service-secret": os.environ["TTS_SERVICE_SECRET"]}

    def synthesize_dialogue(self, lines: list[tuple[str, str]], path: Path) -> str:
        job_id = uuid.uuid4().hex
        with httpx.Client(timeout=300) as client:
            for index, chunk in enumerate(_chunk_lines(lines)):
                client.post(
                    f"{self._url}/synthesize",
                    headers=self._headers,
                    json={
                        "job_id": job_id,
                        "chunk_index": index,
                        "lines": [{"text": t, "voice_id": v} for t, v in chunk],
                    },
                ).raise_for_status()
            resp = client.post(
                f"{self._url}/finalize",
                headers=self._headers,
                json={"job_id": job_id, "filename": Path(path).name},
            )
            resp.raise_for_status()
        return resp.json()["url"]
