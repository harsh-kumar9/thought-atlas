"""D-64 stage 3: which archetype is each convergent domain's "own efficient" target for Gemma-4-12B.

The Qwen rule (success_dir) used within-task slopes from 4 samples per task. The Gemma readout corpus is the atlas: one
trace per task, so slopes are between tasks here (rates z-scored across tasks, outcome = correct from the re-graded atlas
grades, difficulty as a covariate where the task set has one), with a task bootstrap. The own archetype per domain is the
one whose (phi - centroid) has the largest predicted gain in accuracy at a 12k budget (finished within 12k and correct, the
arms' primary outcome; raw correctness is near ceiling at the atlas budget); the A2-pattern score (subgoal up, verification and Q&A down) is
reported next to it as a check. Run with MC_MODEL=gemma4_12b after archetypes.json exists.
Output: data/gemma4_12b/sigctrl/success_dir_gemma.{json,csv}.
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from .archetypes import whole_signature
from .common import BEH, DATA, JUDGE_MODEL, REPO

SIG = DATA / "sigctrl"
DOMAINS = ["math", "gpqa", "planning", "idea"]
TAG = JUDGE_MODEL.replace("/", "_")
GRADES = REPO / "data/v2/perf/success_grades_freex.parquet"


def load_domain(dom: str) -> pl.DataFrame:
    rd = DATA / "runs" / f"atlas_{dom}"
    tr = pl.read_parquet(rd / "traces_baseline.parquet")
    tb = pl.concat([pl.read_parquet(p) for p in sorted((rd / "judge").rglob(f"trackB_full__{TAG}*.parquet"))], how="diagonal_relaxed")
    sig = whole_signature(tb).filter(pl.col("n_valid") >= 4)
    if dom in ("math", "gpqa", "planning") and GRADES.exists():
        g = pl.read_parquet(GRADES)
        scol = next(c for c in ("success", "correct", "is_correct") if c in g.columns)
        g = g.select(["trace_id", pl.col(scol).cast(pl.Float64).alias("correct")]).unique("trace_id", keep="last")
        tr = tr.join(g, on="trace_id", how="left")
    else:
        tr = tr.with_columns(pl.lit(None, dtype=pl.Float64).alias("correct"))
    tr = tr.with_columns(pl.Series("answered", [int(bool((a or "").strip())) for a in tr["answer_text"].to_list()]).cast(pl.Float64))
    # budget-contingent outcomes from the atlas traces by truncation: finished within B generated tokens and correct (the arms' primary outcome)
    for B in (8192, 12288):
        tr = tr.with_columns(((pl.col("n_new_tokens") <= B) & pl.col("completed")).cast(pl.Float64).alias(f"finished_{B // 1024}k"),
                             (pl.when((pl.col("n_new_tokens") <= B) & pl.col("completed")).then(pl.col("correct")).otherwise(0.0)).alias(f"correct_{B // 1024}k"))
    keep = ["trace_id", "instance_id", "task_type", "answered", "correct", "correct_8k", "correct_12k", "finished_8k", "finished_12k", "n_new_tokens"] + (["difficulty_raw"] if "difficulty_raw" in tr.columns else [])
    return tr.select(keep).join(sig, on="trace_id", how="inner")


def fit(df: pl.DataFrame, y: str, X: list[str]):
    g = df.filter(pl.col(y).is_not_null())
    if g.height < 30 or g[y].std() == 0:
        return g, np.full(len(X), np.nan), np.full(len(X), np.nan)
    Xm = g.select(X).to_numpy().astype(float)
    mu, sd = Xm.mean(0), Xm.std(0) + 1e-9
    Z = (Xm - mu) / sd
    ym = g[y].to_numpy().astype(float)
    cov = [Z]
    if "difficulty_raw" in g.columns and g["difficulty_raw"].dtype.is_numeric() and g["difficulty_raw"].null_count() == 0 and g["difficulty_raw"].n_unique() > 1:
        d = g["difficulty_raw"].to_numpy().astype(float); cov.append(((d - d.mean()) / (d.std() + 1e-9))[:, None])
    A = np.hstack(cov + [np.ones((len(ym), 1))])
    beta = np.linalg.lstsq(A, ym, rcond=None)[0]
    uni = np.array([np.corrcoef(Z[:, k], ym)[0, 1] if Z[:, k].std() > 0 else np.nan for k in range(Z.shape[1])])
    return g, uni, beta[: len(X)] / sd            # beta per unit rate (undo the z-scoring)


def main():
    arch = json.loads((SIG / "archetypes.json").read_text())
    X = [f"all_{b}" for b in BEH]
    rng = np.random.default_rng(0)
    rows, preds, summary = [], [], {}
    for dom in DOMAINS:
        df = load_domain(dom)
        ys = ["correct", "correct_12k", "correct_8k", "finished_12k"] if dom != "idea" else ["finished_12k", "finished_8k"]
        cent = np.array(arch[f"{dom}_centroid"]["phi"][:8])
        cand = {k: np.array(arch[k]["phi"][:8]) - cent for k in arch if k.startswith(f"{dom}_A")}
        summary[dom] = {"n_traces": df.height, "accuracy": float(df["correct"].mean()) if df["correct"].null_count() < df.height else None,
                        "accuracy_12k": float(df["correct_12k"].mean()) if df["correct_12k"].null_count() < df.height else None,
                        "finished_12k": float(df["finished_12k"].mean()), "answered": float(df["answered"].mean()), "centroid": dict(zip(BEH, cent.round(4).tolist()))}
        for y in ys:
            g, uni, beta = fit(df, y, X)
            bs = {k: [] for k in cand}
            for _ in range(500):
                idx = rng.integers(0, g.height, g.height)
                _, _, b = fit(g[idx.tolist()], y, X)
                for k, d in cand.items():
                    bs[k].append(float(np.nansum(b * d)))
            for k, b in enumerate(BEH):
                rows.append({"domain": dom, "outcome": y, "behavior": b, "between_task_corr": float(uni[k]), "beta": float(beta[k])})
            for k, d in cand.items():
                v = np.array(bs[k])
                preds.append({"domain": dom, "outcome": y, "archetype": k, "pred": float(np.nansum(beta * d)),
                              "pred_lo": float(np.quantile(v, 0.025)), "pred_hi": float(np.quantile(v, 0.975)),
                              "a2_pattern": float(d[BEH.index("subgoal")] - d[BEH.index("verification")] - d[BEH.index("Question_and_Answering")]),
                              "gap": dict(zip(BEH, d.round(4).tolist()))})
        if dom != "idea":
            own = max(cand, key=lambda k: next(p["pred"] for p in preds if p["domain"] == dom and p["outcome"] == "correct_12k" and p["archetype"] == k))
            summary[dom]["own_archetype"] = own
            summary[dom]["own_by_correct_32k"] = max(cand, key=lambda k: next(p["pred"] for p in preds if p["domain"] == dom and p["outcome"] == "correct" and p["archetype"] == k))
            summary[dom]["a2_pattern_best"] = max(cand, key=lambda k: next(p["a2_pattern"] for p in preds if p["domain"] == dom and p["outcome"] == "correct" and p["archetype"] == k))
    pl.DataFrame(rows).write_csv(SIG / "success_dir_gemma.csv")
    P = pl.DataFrame([{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in p.items()} for p in preds])
    P.write_csv(SIG / "success_pred_gemma.csv")
    (SIG / "success_dir_gemma.json").write_text(json.dumps(summary, indent=1))
    pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(200)
    print(json.dumps({d: {k: v for k, v in s.items() if k != "centroid"} for d, s in summary.items()}, indent=1))
    print(P.select(["domain", "outcome", "archetype", "pred", "pred_lo", "pred_hi", "a2_pattern"]))


if __name__ == "__main__":
    main()
