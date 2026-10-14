"""E5 steering outcomes (D-59): line vs data-path vectors, both directions, against the shared baselines.

Per run (nm_math with p1_budget baseline/ol_random borrowed, nm_idea with tx_idea baseline/ol_random) and arm: traces,
hard-loop and soft-loop rates, answered (non-empty final answer), finish-by-length rate, mean generated tokens, and for
Math the accuracy (free-text extraction grades when present, the repo grader with bold stripped otherwise). Paired
contrasts against the baseline over tasks (seeds averaged within task), bootstrap 95% CI. Vector metadata (norm, axis
projection, s) joins from data/neuro/vectors_L16.json. Track B shifts (Idea-likeness) are added by mc.analyze once the
seed-1 judge has run; this script covers the outcome side.

Outputs: data/neuro/analysis/steer_outcomes.csv, steer_paired.csv, figures/fig_neuro_steer.{pdf,png}
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from .common import DATA

NEURO = DATA / "neuro"; OUT = NEURO / "analysis"; FIG = NEURO / "figures"
# shared arms borrowed from runs with the same fingerprint: the unsteered baseline, the norm-6.75 random vector, the
# low-rank signature vector of the other domain (sig_idea on Math, sig_math on Idea; D-52) and A2 on Math
RUNS = {"nm_math": ([("p1_budget", ["baseline", "ol_random", "ol_math_A2"]), ("tx_math", ["sig_idea", "idea_commit"])], "toward_idea"),
        "nm_idea": ([("tx_idea", ["baseline", "ol_random", "sig_math"])], "toward_math")}
ARMS = ["line_pm", "line_s0p5", "curve_n0p5", "curve_s0p5", "line_s1p0", "rand_n0p5", "rand_n1p0"]


def _grades() -> dict:
    files = sorted(DATA.glob("extract_free*/grades.parquet"))
    if not files:
        return {}
    g = pl.concat([pl.read_parquet(f, columns=["trace_id", "success"]) for f in files]).unique("trace_id", keep="last")
    return dict(zip(g["trace_id"].to_list(), g["success"].to_list()))


def _math_grade(answer, ref):
    from src.perf.answer_grading import grade_math_answer
    if not answer:
        return 0
    return int(bool(grade_math_answer(answer.replace("**", ""), ref).get("success")))


def load_run(run: str) -> pl.DataFrame:
    rd = DATA / "runs" / run
    parts = [pl.read_parquet(p) for p in sorted(rd.glob("traces_*.parquet"))]
    borrow, _ = RUNS[run]
    for brun, bconds in borrow:
        for c in bconds:
            p = DATA / "runs" / brun / f"traces_{c}.parquet"
            if p.exists():
                parts.append(pl.read_parquet(p))
    tr = pl.concat(parts, how="diagonal_relaxed")
    fps = tr["run_fingerprint"].unique().to_list()
    assert len(fps) == 1, f"{run}: mixed fingerprints {fps} (arms would not be paired)"
    return tr


def outcomes(tr: pl.DataFrame, grades: dict) -> pl.DataFrame:
    """Grades: the free-text extraction when EVERY math trace of the frame has one, else the direct grader for every
    trace (mixing the two biases the contrast: the direct grader under-grades, D-46/D-47)."""
    math_ids = tr.filter(pl.col("task_type") == "math")["trace_id"].to_list()
    if math_ids and not all(t in grades for t in math_ids):
        missing = sum(1 for t in math_ids if t not in grades)
        print(f"[steer] {missing}/{len(math_ids)} math traces lack extraction grades -> direct grader for all (run freexg on the new arms for the final numbers)")
        grades = {}
    rows = []
    for r in tr.iter_rows(named=True):
        ans = (r.get("answer_text") or "").strip()
        rec = {"condition": r["condition"], "instance_id": r["instance_id"], "task_type": r["task_type"], "seed": int(r["seed"]),
               "tokens": int(r["n_new_tokens"]), "hard_loop": int(bool(r.get("hard_loop"))), "soft_loop": int(bool(r.get("soft_loop"))),
               "answered": int(bool(ans)), "by_length": int(r.get("finish_reason") == "length"), "completed": int(bool(r.get("completed")))}
        if r["task_type"] == "math":
            g = grades.get(r["trace_id"])
            rec["correct"] = int(g) if g is not None else _math_grade(ans, r.get("reference_answer"))
            rec["grade_source"] = "extraction" if g is not None else "direct"
        rows.append(rec)
    return pl.DataFrame(rows)


def paired(df: pl.DataFrame, ref: str = "baseline", reps: int = 2000) -> pl.DataFrame:
    metrics = [m for m in ("hard_loop", "soft_loop", "answered", "by_length", "tokens", "correct") if m in df.columns]
    task = df.group_by(["condition", "instance_id"]).agg([pl.col(m).mean() for m in metrics])
    base = task.filter(pl.col("condition") == ref).drop("condition")
    rng = np.random.default_rng(0); out = []
    for (c,), g in task.filter(pl.col("condition") != ref).group_by(["condition"]):
        j = g.join(base, on="instance_id", suffix="_ref")
        rec = {"condition": c, "n_tasks": j.height}
        for m in metrics:
            d = (j[m] - j[f"{m}_ref"]).to_numpy().astype(float)
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)])
            rec.update({f"{m}_ref": float(j[f"{m}_ref"].mean()), m: float(j[m].mean()), f"d_{m}": float(d.mean()),
                        f"d_{m}_lo": float(np.quantile(bs, 0.025)), f"d_{m}_hi": float(np.quantile(bs, 0.975))})
        out.append(rec)
    return pl.DataFrame(out).sort("condition")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="nm_math,nm_idea"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    vec = json.loads((NEURO / "vectors_L16.json").read_text())
    grades = _grades()
    summ_all, pair_all = [], []
    for run in a.runs.split(","):
        tr = load_run(run); df = outcomes(tr, grades)
        _, direction = RUNS[run]
        axis = np.load(NEURO / "geometry_L16.npz")["axis"].astype(np.float32)
        summ = df.group_by("condition").agg([pl.len().alias("n")] + [pl.col(m).mean() for m in ("hard_loop", "soft_loop", "answered", "by_length", "tokens") if m in df.columns]
                                             + ([pl.col("correct").mean()] if "correct" in df.columns else []))
        meta = []
        for c in summ["condition"].to_list():
            v = vec.get(f"nm_{direction}_{c}", {})
            if not v and c != "baseline":                                    # borrowed arm: read its vsteer vector file
                vp = DATA / "vsteer" / f"vec_{c}.npz"
                if vp.exists():
                    z = np.load(vp, allow_pickle=True); vv = z["v"].astype(np.float32)
                    v = {"norm": float(np.linalg.norm(vv)), "axis_proj": float(vv @ axis) if int(z["layer"]) == 16 else None, "s": None}
            meta.append({"condition": c, "norm": v.get("norm"), "axis_proj": v.get("axis_proj"), "s": v.get("s"),
                         "path": ("line" if c.startswith("line") else "curve" if c.startswith("curve") else "random" if c.startswith("rand") or c == "ol_random"
                                  else "signature (low-rank)" if c.startswith("sig_") or c in ("ol_math_A2", "idea_commit") else c)})
        n_full = int(summ["n"].max())
        summ = summ.join(pl.DataFrame(meta), on="condition").with_columns(pl.lit(run).alias("run"), (pl.col("n") < n_full).alias("partial")).sort("path", "norm")
        pr = paired(df).with_columns(pl.lit(run).alias("run"))
        summ_all.append(summ); pair_all.append(pr)
        pl.Config.set_tbl_width_chars(230); pl.Config.set_tbl_rows(40); pl.Config.set_fmt_float("mixed")
        print(f"\n[steer] {run} (grades: {df['grade_source'].value_counts().to_dicts() if 'grade_source' in df.columns else 'n/a'})")
        print(summ.with_columns(pl.col(pl.Float64).round(3)))
        print(pr.select([c for c in pr.columns if c == "condition" or c.startswith("d_") or c == "n_tasks"]).with_columns(pl.col(pl.Float64).round(3)))
    S = pl.concat(summ_all, how="diagonal_relaxed"); P = pl.concat(pair_all, how="diagonal_relaxed")
    S.write_csv(OUT / "steer_outcomes.csv"); P.write_csv(OUT / "steer_paired.csv")
    # figure: loops and answered vs vector norm, by path type, one panel per run
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "pdf.fonttype": 42})
    runs = S["run"].unique().to_list()
    fig, axs = plt.subplots(2, len(runs), figsize=(7, 4.4), squeeze=False)
    col = {"line": "k", "curve": "#c7521d", "random": "grey", "baseline": "#1b6ca8", "signature (low-rank)": "#5e8c3a"}
    mk = {"line": "o", "curve": "s", "random": "^", "baseline": "D", "signature (low-rank)": "P"}
    for k, run in enumerate(runs):
        s = S.filter(pl.col("run") == run)
        for row, m in enumerate(("hard_loop", "answered")):
            ax = axs[row, k]
            for p_ in s["path"].unique().to_list():
                g = s.filter(pl.col("path") == p_).sort("norm")
                x = g["norm"].fill_null(0.0).to_numpy(); y = g[m].to_numpy()
                ax.plot(x, y, marker=mk.get(p_, "o"), color=col.get(p_, "k"), ls="-" if p_ in ("line", "curve") else "none", label=p_, ms=4)
                for xi, yi, c in zip(x, y, g["condition"].to_list()):
                    ax.annotate(c.replace("_", " "), (xi, yi), fontsize=4.5, xytext=(2, 2), textcoords="offset points")
            ax.set_xlabel("vector norm (layer 16)"); ax.set_ylabel({"hard_loop": "hard-loop rate", "answered": "answered rate"}[m])
            if row == 0: ax.set_title({"nm_math": "Math tasks, toward Idea", "nm_idea": "Idea tasks, toward Math"}.get(run, run)); ax.text(-0.12, 1.06, "ab"[k], transform=ax.transAxes, fontsize=9, fontweight="bold")
            if row == 0 and k == 0: ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "fig_neuro_steer.pdf", bbox_inches="tight"); fig.savefig(FIG / "fig_neuro_steer.png", bbox_inches="tight")
    print("[steer] ->", OUT / "steer_outcomes.csv", FIG / "fig_neuro_steer.pdf")


if __name__ == "__main__":
    main()
