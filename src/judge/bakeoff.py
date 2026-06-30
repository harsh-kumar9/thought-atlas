"""src/judge/bakeoff.py — Judge selection by inter-judge IRR (no gold set).

Inputs: per-candidate per-sentence label parquets (Track B full-context) produced by
run_judge.py on the SAME stratified calibration sample. Computes, per behavior:
  * pairwise Krippendorff's alpha (nominal/binary) and Cohen's kappa between candidates
  * each candidate's CENTRALITY = mean pairwise agreement with all other candidates
Then ranks candidates by mean centrality across behaviors and reports a decision table.

IMPORTANT (documented limitation): IRR measures RELIABILITY, not VALIDITY. With no gold
labels we pick the most-central judge that clears the reliability bar and is cheapest.
A high-centrality judge is one that agrees with the consensus; it is not proven correct.
Production option: an LLM-jury (aggregate 2-3 cross-family candidates) instead of one.

Usage:
    python -m src.judge.bakeoff --labels "data/judge/calib/trackB_full__*.parquet" \
        --out data/analysis/bakeoff.json
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.judge.schemas import KIM_BEHAVIORS, GANDHI_BEHAVIORS

ALL_BEHAVIORS = KIM_BEHAVIORS + GANDHI_BEHAVIORS
KEY = ["trace_id", "seg_idx"]


def _krippendorff_binary(a: np.ndarray, b: np.ndarray) -> float:
    """Krippendorff's alpha for two raters, binary (nominal) data. 1=perfect, 0=chance."""
    # observed disagreement
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n == 0:
        return float("nan")
    Do = np.mean(a != b)
    # expected disagreement from pooled marginal
    vals = np.concatenate([a, b])
    p1 = np.mean(vals)
    De = 2 * p1 * (1 - p1)
    if De == 0:
        return 1.0 if Do == 0 else 0.0
    return 1.0 - Do / De


def _cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return float("nan")
    po = np.mean(a == b)
    pa1, pb1 = np.mean(a), np.mean(b)
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def load_candidates(label_glob: str) -> dict[str, pl.DataFrame]:
    out = {}
    for p in sorted(glob.glob(label_glob)):
        name = Path(p).stem.split("__", 1)[-1]
        out[name] = pl.read_parquet(p)
    if len(out) < 2:
        raise ValueError(f"need >=2 candidate label files; matched {list(out)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="glob of trackB_full__<cand>.parquet")
    ap.add_argument("--out", default="data/analysis/bakeoff.json")
    ap.add_argument("--bar", type=float, default=0.6)
    args = ap.parse_args()

    cands = load_candidates(args.labels)
    names = list(cands)
    # align all candidates on common (trace_id, seg_idx)
    common = None
    for df in cands.values():
        k = df.select(KEY).unique()
        common = k if common is None else common.join(k, on=KEY, how="inner")
    aligned = {n: df.join(common, on=KEY, how="inner").sort(KEY) for n, df in cands.items()}
    n_units = common.height
    print(f"[bakeoff] {len(names)} candidates, {n_units} aligned segments")

    result = {"candidates": names, "n_aligned_segments": n_units, "bar": args.bar,
              "per_behavior": {}, "centrality": {}}
    cent_acc = {n: [] for n in names}
    for beh in ALL_BEHAVIORS:
        pair_alpha, pair_kappa = {}, {}
        vecs = {n: aligned[n][beh].cast(pl.Float64).to_numpy() for n in names}
        for x, y in itertools.combinations(names, 2):
            al = _krippendorff_binary(vecs[x], vecs[y])
            ka = _cohen_kappa(vecs[x], vecs[y])
            pair_alpha[f"{x}|{y}"] = round(float(al), 4)
            pair_kappa[f"{x}|{y}"] = round(float(ka), 4)
        # centrality: mean alpha vs all others
        cent = {}
        for n in names:
            others = [pair_alpha[k] for k in pair_alpha if n in k.split("|")]
            cent[n] = round(float(np.nanmean(others)), 4) if others else float("nan")
            cent_acc[n].append(cent[n])
        result["per_behavior"][beh] = {"alpha": pair_alpha, "kappa": pair_kappa, "centrality": cent,
                                       "prevalence": {n: round(float(np.nanmean(vecs[n])), 4) for n in names}}

    result["centrality"] = {n: round(float(np.nanmean(cent_acc[n])), 4) for n in names}
    ranked = sorted(names, key=lambda n: result["centrality"][n], reverse=True)
    result["ranking_by_centrality"] = ranked
    result["recommendation"] = {
        "most_central": ranked[0],
        "clears_bar": [n for n in ranked if result["centrality"][n] >= args.bar],
        "note": "IRR=reliability not validity; pick most-central clearing bar that is cheapest. "
                "Consider an LLM-jury of top cross-family candidates for production.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[bakeoff] ranking: {ranked}")
    print(f"[bakeoff] centrality: {result['centrality']}")
    print(f"[bakeoff] -> {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
