"""F1 analysis (D-50): demand vs topic vs format.

Inputs: mode_control/data/f1/judge_in/traces_<model>.parquet (reasoning text cut to 8,192 tokens), Track B under
data/runs/f1/judge/trackB_<model>/, the atlas traces and Track B (data/v2), and for Qwen/R1 the prompt-time activations
(data/f1/acts_<model>.npz) with the existing Math/Idea Stage A activations and prompt-contrast directions.

  profiles   per-trace rates of the 8 behaviours over the window, F1 and atlas -> data/f1/analysis/rates_<model>.parquet
  distances  within-item format / demand changes vs between-item topic changes, atlas log-ratio metric -> distances.csv,
             per-behaviour paired differences -> paired_behaviour.csv
  classify   each model's atlas domain classifier (8 domains, the same windowed rates) applied to F1 prompts -> classify.csv
  project    last-prompt-token states of F1 prompts on the Math->Idea axis and the explore-minus-check axis -> project.csv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from .common import BEH, MC
from .f1_pairs import F1, GEN_MODELS

OUT = F1 / "analysis"
V2 = MC.parent / "data/v2"
TAG = "google_gemma-4-31B-it"
EPS = 0.005
ATLAS_DOMS = ["math", "code", "gpqa", "planning", "security", "safety", "moral", "idea"]
R = [f"{b}__rate" for b in BEH]


def _tok(model: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(GEN_MODELS[model])


def _rates(tr: pl.DataFrame, tb: pl.DataFrame, tok, window: int = 8192) -> pl.DataFrame:
    from .phenotype import windowed_amounts
    return windowed_amounts(tr, tb, window=window, tok=tok)


def cmd_profiles(models: list[str]):
    OUT.mkdir(parents=True, exist_ok=True)
    for m in models:
        tok = _tok(m)
        f1 = pl.read_parquet(F1 / "judge_in" / f"traces_{m}.parquet")
        tb = pl.concat([pl.read_parquet(p) for p in sorted((MC / "data/runs/f1/judge").rglob(f"trackB_full__{TAG}*.parquet"))], how="diagonal_relaxed")
        tb = tb.filter(pl.col("trace_id").is_in(f1["trace_id"].to_list())).unique(["trace_id", "seg_idx"], keep="last")
        fr = _rates(f1, tb, tok).join(f1.select(["trace_id", "instance_id", "task_type", "metadata" if "metadata" in f1.columns else "instance_metadata"]), on="trace_id")
        meta_col = "metadata" if "metadata" in fr.columns else "instance_metadata"
        md = [json.loads(x) for x in fr[meta_col].to_list()]
        fr = fr.with_columns(pl.Series("item", [d["f1_item"] for d in md]), pl.Series("topic", [d["topic"] for d in md]),
                             pl.Series("ptype", [d["ptype"] for d in md]), pl.lit("f1").alias("source")).drop(meta_col)
        at = pl.read_parquet(V2 / "traces" / f"traces_{m}.parquet", columns=["trace_id", "instance_id", "task_type", "reasoning_text_for_analysis"])
        atb = (pl.scan_parquet(V2 / "judge" / f"trackB_full__{TAG}.parquet").filter(pl.col("trace_id").is_in(at["trace_id"].to_list()))
               .filter(pl.col("context_mode") == "full").collect())
        ar = _rates(at, atb, tok).join(at.select(["trace_id", "instance_id", "task_type"]), on="trace_id").with_columns(
            pl.col("instance_id").alias("item"), pl.col("task_type").alias("topic"), pl.lit("atlas").alias("ptype"), pl.lit("atlas").alias("source"))
        df = pl.concat([fr, ar], how="diagonal_relaxed")
        df.write_parquet(OUT / f"rates_{m}.parquet")
        print(f"[f1 profiles] {m}: {fr.height} F1 traces with labels ({fr.group_by('ptype').len().sort('ptype').rows()}), {ar.height} atlas traces")


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.log2((a + EPS) / (b + EPS)))))


def cmd_distances(models: list[str], reps: int = 2000):
    rows, prows = [], []
    rng = np.random.default_rng(0)
    for m in models:
        df = pl.read_parquet(OUT / f"rates_{m}.parquet").filter(pl.col("source") == "f1")
        P = {(r["item"], r["ptype"]): np.array([r[c] for c in R]) for r in df.iter_rows(named=True)}
        topic = dict(zip(df["item"], df["topic"]))
        items = sorted(set(df["item"]))
        comp = {"format (solve_free vs solve_mcq)": ("solve_free", "solve_mcq"), "demand (solve_free vs generate)": ("solve_free", "generate"),
                "demand (solve_free vs evaluate)": ("solve_free", "evaluate"), "demand (generate vs evaluate)": ("generate", "evaluate")}
        for name, (x, y) in comp.items():
            for tp in ("math", "gpqa", "both"):
                v = np.array([dist(P[(i, x)], P[(i, y)]) for i in items if (i, x) in P and (i, y) in P and (tp == "both" or topic[i] == tp)])
                bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(reps)] if len(v) else [np.nan]
                rows.append({"model": m, "contrast": name, "topic": tp, "kind": "within item", "n": len(v), "dist": float(v.mean()) if len(v) else np.nan,
                             "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975))})
        # between items at the same prompt type: different topic vs same topic (the latter is the item noise floor)
        for pt in ("solve_free", "solve_mcq", "generate", "evaluate"):
            mi = [i for i in items if topic[i] == "math" and (i, pt) in P]; gi = [i for i in items if topic[i] == "gpqa" and (i, pt) in P]
            pairs_diff = [(a, b) for a in mi for b in gi]
            pairs_same = [(a, b) for grp in (mi, gi) for k, a in enumerate(grp) for b in grp[k + 1:]]
            for kind, pairs in (("between items, different topic", pairs_diff), ("between items, same topic", pairs_same)):
                v = np.array([dist(P[(a, pt)], P[(b, pt)]) for a, b in pairs])
                rows.append({"model": m, "contrast": f"topic at {pt}", "topic": "both", "kind": kind, "n": len(v), "dist": float(v.mean()) if len(v) else np.nan,
                             "lo": np.nan, "hi": np.nan})
        # per-behaviour paired differences (rate points x 100), within item
        for name, (x, y) in comp.items():
            for k, b in enumerate(BEH):
                v = np.array([100 * (P[(i, y)][k] - P[(i, x)][k]) for i in items if (i, x) in P and (i, y) in P])
                bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(reps)] if len(v) else [np.nan]
                prows.append({"model": m, "contrast": name, "behaviour": b, "n": len(v), "diff": float(v.mean()) if len(v) else np.nan,
                              "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975))})
    d = pl.DataFrame(rows); d.write_csv(OUT / "distances.csv"); pl.DataFrame(prows).write_csv(OUT / "paired_behaviour.csv")
    pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(200); print(d)


def cmd_classify(models: list[str]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows = []
    for m in models:
        df = pl.read_parquet(OUT / f"rates_{m}.parquet").drop_nulls(R)
        at = df.filter(pl.col("source") == "atlas"); f1 = df.filter(pl.col("source") == "f1")
        f1_items = set(f1["item"].unique())
        train = at.filter(~pl.col("item").is_in(list(f1_items)))                 # no F1 item leaks into training
        X, y = train.select(R).to_numpy(), train["topic"].to_numpy()
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        cv = cross_val_predict(clf, X, y, cv=GroupKFold(5), groups=train["item"].to_numpy())
        clf.fit(X, y)
        pr = clf.predict(f1.select(R).to_numpy()); pp = clf.predict_proba(f1.select(R).to_numpy()); cls = list(clf.classes_)
        f1 = f1.with_columns(pl.Series("pred", pr), *[pl.Series(f"p_{c}", pp[:, k]) for k, c in enumerate(cls)])
        for (tp, pt), g in f1.group_by(["topic", "ptype"]):
            rec = {"model": m, "topic": tp, "ptype": pt, "n": g.height, "atlas_cv_acc": float((cv == y).mean())}
            for c in ATLAS_DOMS:
                rec[f"share_{c}"] = float((g["pred"] == c).mean())
            for c in ("math", "gpqa", "idea", "moral", "planning"):
                rec[f"mean_p_{c}"] = float(g[f"p_{c}"].mean()) if f"p_{c}" in g.columns else np.nan
            rows.append(rec)
    d = pl.DataFrame(rows).sort(["model", "topic", "ptype"]); d.write_csv(OUT / "classify.csv")
    pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(260)
    print(d.select(["model", "topic", "ptype", "n", "atlas_cv_acc", "share_math", "share_gpqa", "share_planning", "share_idea", "share_moral", "share_security"]))


def cmd_project(models: list[str]):
    """Last-prompt-token states of F1 prompts on two axes per layer: Math->Idea (unprompted atlas Math and Idea prompts, Stage A)
    in units of that gap (0 = Math mean, 1 = Idea mean), and the explore-minus-check prompt-contrast direction (D-04) in units of
    its own Math->Idea gap. The prompt-contrast axis comes from teacher-forcing identical tokens under two system messages, so
    it carries no task-template wording."""
    rows = []
    for m in models:
        dd = MC / "data" if m == "qwen35_9b" else MC / "data" / m
        A = np.load(next(p for p in (dd / "acts" / "stageA_k64.npz", dd / "acts" / "stageA_k192.npz") if p.exists()), allow_pickle=True)
        ph = A["phases"].tolist().index("last_prompt"); tt = A["task_type"]
        Z = np.load(F1 / f"acts_{m}.npz", allow_pickle=True); H = Z["H"].astype(np.float32)
        dirs = next(p for p in sorted((dd / "directions").glob("directions_w4096*.npz"), reverse=True))
        D = np.load(dirs)
        for l in range(H.shape[1]):
            a = A["acts"][:, l, ph, :].astype(np.float32); mu_m, mu_i = a[tt == "math"].mean(0), a[tt == "idea"].mean(0)
            u = mu_i - mu_m; gap = float(u @ u)
            s_dom = (H[:, l, :] - mu_m) @ u / gap
            key = f"prompt_r1|{l}|last_prompt"
            if key in D.files:
                v = D[key][:, 0]; gv = float((mu_i - mu_m) @ v)
                s_pr = (H[:, l, :] - mu_m) @ v / gv if abs(gv) > 1e-6 else np.full(H.shape[0], np.nan)
            else:
                s_pr = np.full(H.shape[0], np.nan)
            for tp in ("math", "gpqa"):
                for pt in ("solve_free", "solve_mcq", "generate", "evaluate"):
                    k = (Z["topic"] == tp) & (Z["ptype"] == pt)
                    rows.append({"model": m, "layer": l, "topic": tp, "ptype": pt, "n": int(k.sum()),
                                 "math_to_idea": float(s_dom[k].mean()), "math_to_idea_sd": float(s_dom[k].std()),
                                 "explore_axis": float(np.nanmean(s_pr[k])), "explore_axis_sd": float(np.nanstd(s_pr[k]))})
    d = pl.DataFrame(rows); d.write_csv(OUT / "project.csv")
    pl.Config.set_tbl_rows(100); print(d.filter(pl.col("layer").is_in([3, 10, 16, 19, 25])))


def cmd_noise(models: list[str], reps: int = 2000):
    """D-50 amendment: second samples (seed 43) of the solve_free and generate prompts give a sampling-noise floor. For each item:
    noise = d(sample 1, sample 2) of the same prompt; format = d(solve_free, solve_mcq); demand = d(solve_free, generate), all in
    the atlas log-ratio metric. Excess = within-item distance minus the noise floor, item bootstrap."""
    rows = []
    rng = np.random.default_rng(0)
    for m in models:
        tok = _tok(m)
        s2 = pl.read_parquet(F1 / "judge_in_s2" / f"traces_{m}.parquet")
        tb = pl.concat([pl.read_parquet(p) for p in sorted((MC / "data/runs/f1/judge").rglob(f"trackB_full__{TAG}*.parquet"))], how="diagonal_relaxed")
        tb = tb.filter(pl.col("trace_id").is_in(s2["trace_id"].to_list())).unique(["trace_id", "seg_idx"], keep="last")
        if not tb.height:
            print(f"[f1 noise] {m}: no labels for the second samples yet"); continue
        r2 = _rates(s2, tb, tok).join(s2.select(["trace_id", "instance_id"]), on="trace_id")
        r1 = pl.read_parquet(OUT / f"rates_{m}.parquet").filter(pl.col("source") == "f1")
        P1 = {r["instance_id"]: np.array([r[c] for c in R]) for r in r1.iter_rows(named=True)}
        P2 = {r["instance_id"]: np.array([r[c] for c in R]) for r in r2.iter_rows(named=True)}
        items = sorted({i.split("::")[0] for i in P2})
        rec = {"model": m}
        def get(i, pt, d):
            return d.get(f"{i}::{pt}")
        cols = {}
        for i in items:
            sf1, sf2, g1, g2, mc1 = get(i, "solve_free", P1), get(i, "solve_free", P2), get(i, "generate", P1), get(i, "generate", P2), get(i, "solve_mcq", P1)
            if any(x is None for x in (sf1, sf2, g1, g2, mc1)):
                continue
            cols.setdefault("noise_solve_free", []).append(dist(sf1, sf2))
            cols.setdefault("noise_generate", []).append(dist(g1, g2))
            cols.setdefault("format", []).append(dist(sf1, mc1))
            cols.setdefault("demand_generate", []).append(dist(sf1, g1))
            # cross-sample versions: the two prompt types from different samples, so both terms carry one sample's noise each
            cols.setdefault("demand_generate_cross", []).append(dist(sf2, g1))
            cols.setdefault("format_cross", []).append(dist(sf2, mc1))
        A = {k: np.array(v) for k, v in cols.items()}
        n = len(next(iter(A.values()))) if A else 0
        rec["n_items"] = n
        if not n:
            continue
        idx = [rng.integers(0, n, n) for _ in range(reps)]
        def stat(f):
            v = f(np.arange(n)); bs = np.array([f(ix) for ix in idx]); return float(v), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
        for k in A:
            rec[k], rec[f"{k}_lo"], rec[f"{k}_hi"] = stat(lambda ix, k=k: A[k][ix].mean())
        rec["format_excess"], rec["format_excess_lo"], rec["format_excess_hi"] = stat(lambda ix: (A["format_cross"][ix] - A["noise_solve_free"][ix]).mean())
        rec["demand_excess"], rec["demand_excess_lo"], rec["demand_excess_hi"] = stat(lambda ix: (A["demand_generate_cross"][ix] - 0.5 * (A["noise_solve_free"][ix] + A["noise_generate"][ix])).mean())
        rec["demand_minus_format"], rec["dmf_lo"], rec["dmf_hi"] = stat(lambda ix: (A["demand_generate_cross"][ix] - A["format_cross"][ix]).mean())
        rows.append(rec)
        print(f"[f1 noise] {m}: {n} items; noise sf {rec['noise_solve_free']:.2f} gen {rec['noise_generate']:.2f}; format {rec['format_cross']:.2f} "
              f"(excess {rec['format_excess']:+.2f} [{rec['format_excess_lo']:+.2f},{rec['format_excess_hi']:+.2f}]); demand {rec['demand_generate_cross']:.2f} "
              f"(excess {rec['demand_excess']:+.2f} [{rec['demand_excess_lo']:+.2f},{rec['demand_excess_hi']:+.2f}]); demand-format {rec['demand_minus_format']:+.2f} [{rec['dmf_lo']:+.2f},{rec['dmf_hi']:+.2f}]")
    if rows:
        pl.DataFrame(rows).write_csv(OUT / "noise_floor.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["profiles", "distances", "classify", "project", "noise"])
    ap.add_argument("--models", default="qwen35_9b,gemma4_12b,reasoner")
    a = ap.parse_args()
    ms = a.models.split(",")
    {"profiles": cmd_profiles, "distances": cmd_distances, "classify": cmd_classify, "project": cmd_project, "noise": cmd_noise}[a.cmd](ms)


if __name__ == "__main__":
    main()
