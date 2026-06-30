"""src/perf/features.py — behavior features for the mechanism analysis.

Three escalating representations per trace:
  1. counts      : Track A whole-trace counts (how MUCH of each behavior)
  2. positional  : Track B reduced summaries per behavior (WHERE it fires)
                   - early_share  (fraction of behavior events in first third)
                   - late_share   (fraction in last third)
                   - center_mass  (mean normalized position of events, 0..1)
                   - dispersion   (std of event positions; how spread vs concentrated)
  3. trajectory  : shape features over the full positional profile
                   - the arc summary: does it follow open->deliberate->close
                   - monotonicity / peakedness of each behavior's positional curve

All features are per (trace_id). counts come from Track A; positional+trajectory from Track B.
"""
from __future__ import annotations
import numpy as np
import polars as pl

KIM = ["Question_and_Answering", "Perspective_Shift", "Conflict_of_Perspectives", "Reconciliation"]
GAN = ["verification", "backtracking", "subgoal", "backward_chaining"]
BEH = GAN + KIM


def count_features(trackA: pl.DataFrame) -> pl.DataFrame:
    """Track A counts + total behavior volume + cognitive/conversational split."""
    df = trackA.select(["trace_id"] + [b for b in BEH if b in trackA.columns])
    df = df.with_columns([
        pl.sum_horizontal([pl.col(b) for b in GAN if b in df.columns]).alias("cog_total"),
        pl.sum_horizontal([pl.col(b) for b in KIM if b in df.columns]).alias("cnv_total"),
    ])
    df = df.with_columns(
        (pl.col("cog_total") + pl.col("cnv_total")).alias("beh_total"))
    return df.rename({b: f"count_{b}" for b in BEH if b in df.columns})


def positional_features(trackB: pl.DataFrame) -> pl.DataFrame:
    """Per-trace positional summary of each behavior, from per-sentence labels.

    Zero-event behaviors get NEUTRAL defaults (com=0.5, disp=0, early=late=0) + n_events,
    NOT NaN — so sparse-behavior domains (moral/idea) aren't silently subsampled by NaN-dropping
    (which inflated apparent R^2 in the first run)."""
    rows = []
    for tid, sub in trackB.group_by("trace_id"):
        sub = sub.sort("seg_idx")
        pos = sub["norm_pos"].to_numpy()
        row = {"trace_id": tid[0] if isinstance(tid, tuple) else tid}
        for b in BEH:
            if b not in sub.columns:
                continue
            v = sub[b].to_numpy().astype(float)
            ev = pos[v > 0]                       # positions where behavior fires
            n = len(ev)
            row[f"{b}_nevents"] = int(n)
            if n == 0:
                row[f"{b}_early"] = 0.0; row[f"{b}_late"] = 0.0
                row[f"{b}_com"] = 0.5; row[f"{b}_disp"] = 0.0   # neutral, NOT NaN
            else:
                row[f"{b}_early"] = float(np.mean(ev < 1/3))
                row[f"{b}_late"] = float(np.mean(ev > 2/3))
                row[f"{b}_com"] = float(np.mean(ev))          # center of mass
                row[f"{b}_disp"] = float(np.std(ev))
        rows.append(row)
    return pl.DataFrame(rows)


def trajectory_features(trackB: pl.DataFrame, nbins=10) -> pl.DataFrame:
    """Shape features: per-trace, does deliberation follow the open->deliberate->close arc?
    - arc_score: corr between the trace's cognitive-late vs conversational-early structure
    - subgoal_frontload, verify_backload, reconcile_terminal: canonical arc components
    These summarize the *sequence*, not just amount."""
    rows = []
    for tid, sub in trackB.group_by("trace_id"):
        sub = sub.sort("seg_idx")
        pos = sub["norm_pos"].to_numpy()
        row = {"trace_id": tid[0] if isinstance(tid, tuple) else tid}
        # canonical arc components (early openers, late closers)
        def share(beh, lo, hi):
            if beh not in sub.columns:
                return 0.0
            v = sub[beh].to_numpy().astype(float)
            m = (pos >= lo) & (pos < hi)
            return float(v[m].sum() / max(1.0, v.sum())) if v.sum() > 0 else 0.0
        row["subgoal_frontload"] = share("subgoal", 0, 1/3)
        row["qa_frontload"] = share("Question_and_Answering", 0, 1/3)
        row["verify_backload"] = share("verification", 2/3, 1.01)
        row["reconcile_terminal"] = share("Reconciliation", 2/3, 1.01)
        # arc adherence: front-load openers + back-load closers, normalized 0..1
        row["arc_score"] = float(np.mean([row["subgoal_frontload"], row["qa_frontload"],
                                          row["verify_backload"], row["reconcile_terminal"]]))
        # overall deliberation density and length
        row["n_segments"] = int(sub.height)
        rows.append(row)
    return pl.DataFrame(rows)


def build_features(trackA, trackB) -> pl.DataFrame:
    c = count_features(trackA)
    p = positional_features(trackB)
    t = trajectory_features(trackB)
    return c.join(p, on="trace_id", how="left").join(t, on="trace_id", how="left")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackA", required=True)
    ap.add_argument("--trackB", required=True)
    ap.add_argument("--out", default="data/perf/features.parquet")
    a = ap.parse_args()
    A = pl.read_parquet(a.trackA); B = pl.read_parquet(a.trackB)
    f = build_features(A, B)
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    f.write_parquet(a.out)
    print(f"features: {f.shape} -> {a.out}")
    print("columns:", f.columns)