"""Reproducible, staged analysis for the Thought Atlas paper.

PR1 intentionally exposes only the audited data layer, outcome registry, and
grouped split registry. Scientific stages are scaffolded and remain gated on a
reviewed, passing data audit.
"""

from .constants import BEHAVIORS, DIALECTICAL, EXECUTIVE, SOT_CORE

__all__ = ["BEHAVIORS", "DIALECTICAL", "EXECUTIVE", "SOT_CORE"]

