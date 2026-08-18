"""Resampling primitives (implemented in PR2 and reused by later stages)."""

from __future__ import annotations


class ScientificStageNotImplemented(NotImplementedError):
    """Raised when a post-PR1 scientific stage is requested."""


def not_implemented(stage: str) -> None:
    raise ScientificStageNotImplemented(
        f"stage '{stage}' is scaffolded but intentionally deferred until the PR1 audit is reviewed"
    )

