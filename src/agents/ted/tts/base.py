from __future__ import annotations

from pathlib import Path
from typing import Protocol


class TTSProvider(Protocol):
    def synthesize_dialogue(self, lines: list[tuple[str, str]], path: Path) -> Path | str:
        """Return a local file path, or a URL when the provider hosts the audio."""
        ...
    