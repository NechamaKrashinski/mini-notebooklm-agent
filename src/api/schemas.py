"""The API contract shared between backend and client.

These Pydantic models are the *stable* boundary: the web client is written against them,
and each stage fills in the backend behind them. Keep them additive.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Impl = Literal["A", "B"]


# -- sources -------------------------------------------------------------------


class SourceInfo(BaseModel):
    """A document in the notebook (list view — no content)."""

    id: str
    name: str
    chars: int
    active: bool


class SourceDetail(SourceInfo):
    """A source with its full text (for the viewer)."""

    content: str


class AddSourceRequest(BaseModel):
    """Add a source by pasting text."""

    name: str | None = None
    content: str


class SetActiveRequest(BaseModel):
    active: bool


# -- chat ----------------------------------------------------------------------


class Citation(BaseModel):
    """A source the answer was grounded on."""

    source: str
    snippet: str | None = None


class ChatRequest(BaseModel):
    """A grounded-chat turn."""

    message: str
    thread_id: str | None = None  # used from stage 4 (short-term memory)


class ChatResponse(BaseModel):
    """The model's grounded answer plus the sources it was based on."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    engine: str


# -- studio (artifacts) --------------------------------------------------------


class ArtifactKind(BaseModel):
    """A generatable artifact shown in the Studio panel."""

    key: str
    title: str
    icon: str
    status: Literal["ready", "planned"]


class GenerateArtifactRequest(BaseModel):
    kind: str
    impl: Impl = "A"


# -- ted talks -----------------------------------------------------------------


class TedJob(BaseModel):
    """A TED talk job. Status comes from ted_jobs; the content from the checkpoint."""

    job_id: str
    status: Literal["pending_approval", "audio_ready", "failed"]
    created_at: str
    topic: str | None = None
    script_he: str | None = None
    word_count: int | None = None
    revision_count: int | None = None
    critique_notes: list[str] = Field(default_factory=list)
    audio_url: str | None = None
    error: str | None = None


class TedResumeRequest(BaseModel):
    action: Literal["approve", "revise"]
    feedback: str | None = None


# -- notes ---------------------------------------------------------------------


class Note(BaseModel):
    id: str
    title: str
    content: str


class AddNoteRequest(BaseModel):
    title: str
    content: str

