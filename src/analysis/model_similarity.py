"""src/analysis/model_similarity.py — cross-model deliberation-structure similarity.

Given per-sentence behavior labels (Track B) for >=2 generation models, asks:
  do different reasoning models deploy the SAME temporal grammar of deliberation?

Each model -> heartbeat tensor H[behavior, domain, position] (per-behavior-per-domain
normalized positional profile, the same object the heartbeat figures plot). Model-to-model
distance = aggregate over (behavior, domain) of a per-curve distance:
  - shape distance  : 1 - Pearson r between the two positional curves (timing only; scale-free)
  - magnitude dist. : L1 between shape-normalized curves (timing + relative amount)

Produces a model x model distance matrix (per behavior-family and overall), plus a per-(behavior,
domain) breakdown of WHERE models agree/diverge most. Self-consistency check: split one model's
traces in half -> distance should be ~0 (sets the noise floor for "meaningfully different").

Usage:
  python -m src.analysis.model_similarity \
      --trackB data/judge/prod/trackB_full__<tag>.parquet \
      --traces-glob "data/traces/traces_*.parquet" --out-dir data/analysis/cross_model
"""
from __future__ import annotations
import argparse, glob, itertools, json
from pathlib import Path
import numpy as np
import polars as pl

KIM = ["Question_and_Answering", "Perspective_Shift", "Conflict_of_Perspectives", "Reconciliation"]
GAN = ["verification", "backtracking", "subgoal"]          # drop backward_chaining (too rare)
BEH = GAN + KIM
DOMAINS = ["math", "code", "gpqa", "planning", "moral", "idea"]


def heartbeat_tensor(trackB: pl.DataFrame, model: str, nbins=12, min_traces=20):
    """H[behavior, domain, position] for one model. NaN where a (behavior,domain) cell is too sparse."""
    b = trackB.filter(pl.col("gen_model") == model)
    b = b.with_columns((pl.col("norm_pos") * (nbins - 1)).round().cast(pl.Int32).alias("bin"))
    H = np.full((len(BEH), len(DOMAINS), nbins), np.nan)
    support = np.zeros((len(BEH), len(DOMAINS)))
    for di, dom in enumerate(DOMAINS):
        sub = b.filter(pl.col("task_type") == dom)
        ntr = sub["trace_id"].n_unique()
        if ntr < min_traces:
            continue
        prof = sub.group_by("bin").agg([pl.col(bh).mean() for bh in BEH if bh in sub.columns]).sort("bin")
        for bi, bh in enumerate(BEH):
            if bh not in prof.columns:
                continue
            y = prof[bh].to_numpy().astype(float)
            if y.sum() > 0:
                H[bi, di] = y / y.sum()          # shape-normalized positional profile
                support[bi, di] = float(b.filter(pl.col("task_type") == dom)[bh].mean())
    return H, support


def curve_distance(a, b, kind="shape"):
    """Distance between two positional curves. shape: 1-corr; mag: L1 of normalized curves."""
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        return np.nan
    if kind == "shape":
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return np.nan
        return float(1.0 - np.corrcoef(a, b)[0, 1]) / 2.0      # 0 (identical) .. 1 (anti) , scaled to [0,1]
    return float(np.abs(a - b).sum())                           # L1 on shape-normalized curves


def model_distance(Ha, Hb, kind="shape"):
    """Aggregate (behavior, domain) curve distances into one model-pair distance + breakdown."""
    cells = {}
    vals = []
    for bi, bh in enumerate(BEH):
        for di, dom in enumerate(DOMAINS):
            d = curve_distance(Ha[bi, di], Hb[bi, di], kind=kind)
            if not np.isnan(d):
                cells[f"{bh}|{dom}"] = round(d, 4); vals.append(d)
    return (float(np.mean(vals)) if vals else np.nan), cells


def family_distance(Ha, Hb, family, kind="shape"):
    idx = [BEH.index(b) for b in family]
    vals = []
    for bi in idx:
        for di in range(len(DOMAINS)):
            d = curve_distance(Ha[bi, di], Hb[bi, di], kind=kind)
            if not np.isnan(d):
                vals.append(d)
    return float(np.mean(vals)) if vals else np.nan


def run(trackB, traces_meta, out_dir, nbins=12, kind="shape", completed_only=True):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    meta_cols = [c for c in ["trace_id", "task_type", "gen_model", "completed"]
                 if c in traces_meta.columns]
    b = trackB.join(traces_meta.select(meta_cols), on="trace_id", how="left")
    # Budgets may differ across models (e.g. Qwen regenerated at 64k while R1/anchor are 32k).
    # Truncated traces have compressed heartbeats; comparing across models with different
    # truncation rates confounds budget with temporal structure. Restrict to COMPLETED traces
    # so the comparison is budget-agnostic. Report what fraction we keep per model.
    if completed_only and "completed" in b.columns:
        before = b.group_by("gen_model").agg(pl.col("trace_id").n_unique().alias("n_all"))
        b = b.filter(pl.col("completed").fill_null(True))
        after = b.group_by("gen_model").agg(pl.col("trace_id").n_unique().alias("n_kept"))
        cov = before.join(after, on="gen_model", how="left").with_columns(
            (pl.col("n_kept") / pl.col("n_all")).round(3).alias("frac_completed"))
        print("completed-only filter (heartbeats computed on finished traces):")
        print(cov.sort("gen_model"))
    models = [m for m in b["gen_model"].unique().to_list() if m is not None]
    print(f"models present: {models}")

    # build tensors
    H = {m: heartbeat_tensor(b, m, nbins=nbins)[0] for m in models}

    # SELF-CONSISTENCY noise floor: split each model's traces in half, distance should be ~0
    floor = {}
    for m in models:
        bm = b.filter(pl.col("gen_model") == m)
        tids = bm["trace_id"].unique().to_list()
        rng = np.random.default_rng(0); rng.shuffle(tids)
        h1 = tids[:len(tids)//2]; h2 = tids[len(tids)//2:]
        H1 = heartbeat_tensor(bm.filter(pl.col("trace_id").is_in(h1)).with_columns(pl.lit(m).alias("gen_model")), m, nbins)[0]
        H2 = heartbeat_tensor(bm.filter(pl.col("trace_id").is_in(h2)).with_columns(pl.lit(m).alias("gen_model")), m, nbins)[0]
        floor[m], _ = model_distance(H1, H2, kind=kind)
    noise = float(np.nanmean(list(floor.values()))) if floor else np.nan
    print(f"self-consistency noise floor (split-half): {noise:.4f} "
          f"-> model pairs below ~{noise:.3f} are statistically indistinguishable in timing")

    # pairwise model distances + family breakdown
    rows = []
    for ma, mb in itertools.combinations(models, 2):
        overall, cells = model_distance(H[ma], H[mb], kind=kind)
        cog = family_distance(H[ma], H[mb], GAN, kind=kind)
        cnv = family_distance(H[ma], H[mb], KIM, kind=kind)
        # which (behavior,domain) cells diverge most
        top = sorted(cells.items(), key=lambda kv: -kv[1])[:5]
        rows.append({"model_a": ma, "model_b": mb, "overall_dist": round(overall, 4),
                     "cognitive_dist": round(cog, 4), "conversational_dist": round(cnv, 4),
                     "vs_noise_floor": round(overall / noise, 2) if noise and not np.isnan(noise) else None,
                     "top_divergences": "; ".join(f"{k}={v}" for k, v in top)})
    out = pl.DataFrame(rows) if rows else pl.DataFrame()
    if out.height:
        out.write_csv(f"{out_dir}/model_distance_{kind}.csv")
        print(out.select(["model_a", "model_b", "overall_dist", "cognitive_dist",
                          "conversational_dist", "vs_noise_floor"]))
    json.dump({"noise_floor": noise, "per_model_floor": floor},
              open(f"{out_dir}/noise_floor_{kind}.json", "w"), indent=2)
    print(f"-> {out_dir}/model_distance_{kind}.csv")
    return out, noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackB", required=True)
    ap.add_argument("--traces-glob", default="data/traces/traces_*.parquet")
    ap.add_argument("--out-dir", default="data/analysis/cross_model")
    ap.add_argument("--kind", choices=["shape", "mag"], default="shape")
    ap.add_argument("--nbins", type=int, default=12)
    ap.add_argument("--include-truncated", action="store_true",
                    help="use ALL traces (default: completed-only, to avoid budget confound)")
    a = ap.parse_args()
    trackB = pl.read_parquet(a.trackB)
    tr = pl.concat([pl.read_parquet(p) for p in glob.glob(a.traces_glob)], how="diagonal_relaxed")
    run(trackB, tr, a.out_dir, nbins=a.nbins, kind=a.kind,
        completed_only=not a.include_truncated)


if __name__ == "__main__":
    main()