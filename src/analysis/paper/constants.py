"""Shared constants for paper analyses."""

from __future__ import annotations

DIALECTICAL = (
    "Question_and_Answering",
    "Perspective_Shift",
    "Conflict_of_Perspectives",
    "Reconciliation",
)
SOT_CORE = (
    "Perspective_Shift",
    "Conflict_of_Perspectives",
    "Reconciliation",
)
EXECUTIVE = (
    "verification",
    "backtracking",
    "subgoal",
    "backward_chaining",
)
BEHAVIORS = DIALECTICAL + EXECUTIVE

CORE_DOMAINS = ("math", "code", "gpqa", "planning", "moral", "idea")
EXTENSION_DOMAINS = ("safety", "security")

STAGES = (
    "audit",
    "family_structure",
    "amount_shape",
    "trace_features",
    "outcome_models",
    "motifs",
    "context_sensitivity",
    "adaptive_differentiation",
    "kline",
    "signatures",
    "report",
    "all",
)

AUDIT_FILENAMES = (
    "trace_coverage.csv",
    "segment_coverage.csv",
    "context_pair_coverage.csv",
    "completion_by_cell.csv",
    "outcome_coverage.csv",
    "analysis_source_by_model.csv",
    "data_quality_flags.csv",
    "analysis_manifest.json",
)
