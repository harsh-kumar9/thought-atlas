"""Slide-ready figures for the deck (white background, deck palette: blue = conversational, green = cognitive)."""
from __future__ import annotations
import json
import numpy as np, polars as pl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import DATA, MC

OUT = MC / "figures_slides"; OUT.mkdir(exist_ok=True)
BLUE, GREEN, GREY, RED, DARK = "#3b6fd6", "#3d9a5a", "#8a8a8a", "#d1495b", "#222222"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def load_shifts(run):
    return pl.read_csv(DATA / f"runs/{run}/analysis/paired_shifts.csv")


def fig_steering():
    """Dot plot: paired shift of the combined score (positive = more Idea-like) per condition and domain."""
    p1, p2, p3, pw = load_shifts("pilot1"), load_shifts("pilot2"), load_shifts("pilot3"), load_shifts("power1")
    rows = [  # (label, run, condition, group)
        ("Checking prompt (n=150)", pw, "prompt_checking", "Prompts, powered run"),
        ("Exploratory prompt (n=150)", pw, "prompt_exploratory", "Prompts, powered run"),
        ("Amplify (γ=2)", p1, "amplify_g2", "64-token pulse, layer 19 (n=40)"),
        ("Neutralize (γ=0)", p1, "neutralize_g0", "64-token pulse, layer 19 (n=40)"),
        ("Reflect (γ=−1)", p1, "reflect_gm1", "64-token pulse, layer 19 (n=40)"),
        ("Random direction", p1, "random_r1_a", "64-token pulse, layer 19 (n=40)"),
        ("Reflect γ=−1, layer 19 (n=150)", pw, "reflect1_sus_L19", "Sustained, toward Math"),
        ("Reflect γ=−3, layer 19 (n=150)", pw, "reflect3_sus_L19", "Sustained, toward Math"),
        ("Shift −1 gap, layers 12–24 (n=150)", pw, "band_add_minus_sus", "Sustained, toward Math"),
        ("Random direction, matched (n=150)", pw, "band_random_sus", "Sustained, toward Math"),
        ("Shift +2 gap, layer 19 (n=40)", p3, "add_plus2_sus_L19", "Sustained, toward Idea"),
        ("Shift +0.5 gap, layers 12–24 (n=40)", p3, "band_add_plus_half_sus", "Sustained, toward Idea"),
    ]
    # y layout with a header row per group
    ylabels, ypos, entries = [], [], []
    y = 0.0; last_group = None
    for label, df, cond, group in rows:
        if group != last_group:
            y -= 0.35; ylabels.append(group); ypos.append(y); entries.append(None); y -= 1.0; last_group = group
        ylabels.append(label); ypos.append(y); entries.append((df, cond)); y -= 1.0
    fig, ax = plt.subplots(figsize=(11, 6.6))
    first = True
    for yy, e in zip(ypos, entries):
        if e is None:
            continue
        df, cond = e
        for dom, col, off in (("math", BLUE, 0.16), ("idea", RED, -0.16)):
            src = df
            if cond.startswith("add_plus") or cond.startswith("band_add_plus"):
                src = pl.read_csv(DATA / "runs/pilot3/analysis/paired_shifts_noloop.csv")
            r = src.filter((pl.col("metric") == "s_all") & (pl.col("condition") == cond) & (pl.col("task_type") == dom)).row(0, named=True)
            ax.errorbar(r["paired_shift"], yy + off, xerr=[[r["paired_shift"] - r["shift_lo"]], [r["shift_hi"] - r["paired_shift"]]],
                        fmt="o", color=col, capsize=3, ms=7, lw=1.5, label=("Math tasks" if dom == "math" else "Idea tasks") if first else None)
        first = False
    ax.axvline(0, color=DARK, lw=1)
    ax.axvspan(-1, 1, color=GREY, alpha=0.12, lw=0)
    ax.set_yticks(ypos); ax.set_yticklabels(ylabels)
    for lab, e in zip(ax.get_yticklabels(), entries):
        if e is None:
            lab.set_fontweight("bold"); lab.set_color(DARK)
    ax.set_xlabel("Change in Idea-likeness score vs. the same tasks unsteered (logits, 95% CI)\n← more Math-like (verification)          more Idea-like (revision, perspective shifts) →")
    ax.set_xlim(-11.5, 10.5); ax.set_ylim(min(ypos) - 0.7, 0.4)
    ax.legend(loc="lower left", frameon=False)
    fig.suptitle("Sustained steering moves reasoning in both directions; an early pulse does nothing", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    ax.set_title("Qwen3.5-9B, one seed. Grey band: control-direction noise floor. Idea-ward arms use loop-free traces.", fontsize=10, loc="left", color=GREY)
    fig.tight_layout(); fig.savefig(OUT / "fig_steering_shifts.png", dpi=200); plt.close(fig)


def fig_profile():
    """Bars: behavior rates for Math baseline, Idea baseline, Idea under sustained γ=−3 (pilot2)."""
    su = pl.read_csv(DATA / "runs/pilot2/analysis/summary.csv")
    def get(cond, dom):
        return su.filter((pl.col("condition") == cond) & (pl.col("task_type") == dom)).row(0, named=True)
    conds = [("Math, unsteered", get("baseline", "math"), BLUE), ("Idea, unsteered", get("baseline", "idea"), RED),
             ("Idea, steered toward Math\n(sustained γ=−3, layer 19)", get("reflect3_sus_L19", "idea"), "#b0475a"),
             ("Idea, random direction\n(matched control)", get("band_random_sus", "idea"), GREY)]
    beh = [("verification__rate", "Verification", GREEN), ("backtracking__rate", "Backtracking", GREEN),
           ("Perspective_Shift__rate", "Perspective shift", BLUE), ("Conflict_of_Perspectives__rate", "Conflict of\nperspectives", BLUE)]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    w = 0.19; x = np.arange(len(beh))
    for j, (name, r, col) in enumerate(conds):
        vals = [r[b] for b, _, _ in beh]; err = [[r[b] - r[b + "_lo"] for b, _, _ in beh], [r[b + "_hi"] - r[b] for b, _, _ in beh]]
        hatch = "//" if j == 2 else None
        ax.bar(x + (j - 1.5) * w, vals, w, yerr=err, color=col, alpha=0.9 if j != 3 else 0.6, capsize=3, label=name, hatch=hatch, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([n for _, n, _ in beh])
    for i, (_, _, fam) in enumerate(beh):
        ax.get_xticklabels()[i].set_color(fam)
    ax.set_ylabel("Fraction of reasoning sentences with the behavior")
    fig.suptitle("Under steering, Idea traces take on the Math profile: more verification, less revision and dialogue", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    ax.set_title("Green labels: cognitive behaviors. Blue: conversational. First 4096 reasoning tokens, 95% CI.", fontsize=10.5, loc="left", color=GREY)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "fig_behavior_profile.png", dpi=200); plt.close(fig)


def fig_readout():
    """Within-domain early readout by phase for Qwen and R1 (64-PC ridge, out of fold), best layer band."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (name, path) in zip(axes, [("Qwen3.5-9B", DATA / "directions/readout_by_phase_k192.csv"), ("R1-Distill-Llama-8B", DATA / "reasoner/directions/readout_by_phase_k192.csv")]):
        df = pl.read_csv(path).filter(pl.col("layer").is_in([16, 19, 22, 24]))
        phases = ["last_prompt", "mean32", "mean64", "late"]; labels = ["last prompt\ntoken", "tokens\n1–32", "tokens\n1–64", "tokens\n65–192"]
        x = np.arange(len(phases)); w = 0.2
        series = [("math", "rho_cog", BLUE, "Math · cognitive", "-"), ("math", "rho_conv", BLUE, "Math · conversational", "--"),
                  ("idea", "rho_cog", RED, "Idea · cognitive", "-"), ("idea", "rho_conv", RED, "Idea · conversational", "--")]
        for dom, col, c, lab, ls in series:
            vals = [df.filter((pl.col("phase") == p) & (pl.col("domain") == dom))[col].mean() for p in phases]
            ax.plot(x, vals, marker="o", color=c, ls=ls, label=lab, lw=2)
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.axhline(0, color=DARK, lw=0.8); ax.set_ylim(-0.1, 0.6)
        ax.set_title(name, fontsize=12); ax.set_xlabel("Where activations are read (mean over layers 16–24)")
    axes[0].set_ylabel("Predicts the trace's later signature\n(out-of-fold Spearman, within domain)")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=10, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Early activations predict how a trace will reason later, within each domain", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(OUT / "fig_early_readout.png", dpi=200); plt.close(fig)




def fig_outcomes():
    """RQ3: Idea originality vs feasibility (powered Qwen run) and Math accuracy per condition."""
    import glob
    from src.perf.answer_grading import grade_math_answer
    rd = DATA / "runs/power1"
    q = pl.read_parquet(glob.glob(str(rd / "judge/quality__*.parquet"))[0])
    tr = pl.concat([pl.read_parquet(p, columns=["trace_id", "condition", "instance_id", "task_type", "answer_text", "reference_answer"]) for p in glob.glob(str(rd / "traces_*.parquet"))], how="diagonal_relaxed")
    conds = [("baseline", "Unsteered", GREY), ("prompt_checking", "Checking prompt", "#7a9cdc"), ("prompt_exploratory", "Exploratory prompt", "#a3bff0"),
             ("reflect1_sus_L19", "Steer toward Math, γ=−1", "#c97a86"), ("reflect3_sus_L19", "Steer toward Math, γ=−3", RED),
             ("band_random_sus", "Random direction (control)", "#b5b5b5")]
    j = q.join(tr, on="trace_id").filter(pl.col("task_type") == "idea")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for c, lab, col in conds:
        g = j.filter(pl.col("condition") == c)
        if g.height < 30:
            continue
        o, f = g["axis_originality"].to_numpy().astype(float), g["axis_feasibility"].to_numpy().astype(float)
        o, f = o[np.isfinite(o)], f[np.isfinite(f)]
        ax.errorbar(o.mean(), f.mean(), xerr=1.96 * o.std() / np.sqrt(len(o)), yerr=1.96 * f.std() / np.sqrt(len(f)), fmt="o", color=col, ms=9, capsize=3, label=lab)
    ax.set_xlabel("Originality (judge, 1–10)"); ax.set_ylabel("Feasibility (judge, 1–10)")
    ax.set_title("Idea tasks: steering toward Math trades originality for feasibility", fontsize=11.5, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    # math accuracy, all tasks (unanswered = wrong)
    ax = axes[1]
    m = tr.filter(pl.col("task_type") == "math")
    vals = []
    for c, lab, col in conds:
        g = m.filter(pl.col("condition") == c)
        if g.height < 30:
            continue
        acc = np.array([(grade_math_answer(a.replace("**", ""), r).get("success") or 0) if a else 0 for a, r in zip(g["answer_text"].to_list(), g["reference_answer"].to_list())], float)
        vals.append((lab, acc.mean(), 1.96 * acc.std() / np.sqrt(len(acc)), col))
    y = np.arange(len(vals))[::-1]
    ax.barh(y, [v[1] for v in vals], xerr=[v[2] for v in vals], color=[v[3] for v in vals], capsize=3)
    ax.set_yticks(y); ax.set_yticklabels([v[0] for v in vals]); ax.set_xlim(0, 0.8)
    ax.set_xlabel("Math accuracy (unanswered counts as wrong)")
    ax.set_title("Math tasks: accuracy unchanged", fontsize=11.5, loc="left")
    fig.suptitle("Does the steered reasoning change task outcomes? (Qwen3.5-9B, 150 tasks per domain, 12k-token budget)", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig_outcomes.png", dpi=200); plt.close(fig)


def fig_methods():
    """Deck-style diagram: how the direction is found, and the ways to steer."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(12, 5.6)); ax.set_xlim(0, 120); ax.set_ylim(0, 56); ax.axis("off")
    def box(x, y, w, h, text, fc="#f1f3f4", ec="#9aa0a6", fs=11.5, bold=False, tc="#202124"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal", linespacing=1.3)
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, color="#5f6368", lw=1.4))
    # ---- top row: finding the direction
    ax.text(1, 53.5, "1. Finding the knob (no topic words involved)", fontsize=13, fontweight="bold", color="#202124")
    box(1, 40, 24, 9, "Same task, two instructions:\n“check each step”  vs.\n“explore alternatives”", fs=10.5)
    arrow(25.6, 44.5, 29.4, 44.5)
    box(30, 40, 24, 9, "Force the same first 32\nreasoning tokens under both", fs=10.5)
    arrow(54.6, 44.5, 58.4, 44.5)
    box(59, 40, 26, 9, "Subtract the hidden activations\n(exploratory − checking)", fs=10.5)
    arrow(85.6, 44.5, 89.4, 44.5)
    box(90, 40, 29, 9, "One direction per layer\n(layer 19 of 32 used)", fc="#fbe0e3", ec="#d1495b", fs=11, bold=True)
    ax.text(60, 37.2, "Check: the same direction separates unprompted Math from Idea traces (AUROC ≈ 1) and predicts each trace’s later signature within a domain.",
            fontsize=10, color="#5f6368", ha="center")
    # ---- bottom row: ways to steer
    ax.text(1, 31, "2. Ways to move it", fontsize=13, fontweight="bold", color="#202124")
    box(1, 15, 26, 12, "Prompt\n\nAdd the instruction as a\nsystem message", fc="#d7e3fb", ec="#6b8fd6", fs=11)
    box(31, 15, 26, 12, "Early pulse\n\nEdit the activation on the\nfirst 64 generated tokens only", fs=10.5)
    box(61, 15, 26, 12, "Sustained\n\nEdit the activation on\nevery generated token", fc="#dcecd6", ec="#7bb072", fs=11)
    box(91, 15, 28, 12, "Edit = scale the position\non the direction (γ: 2 amplify,\n0 neutralize, −1 reflect) or\nadd a fixed shift", fs=10.5)
    ax.text(60, 11.5, "Controls: a random direction of the same size, the raw Math−Idea difference (a topic direction), wrong layers; loops and lengths tracked on every trace.",
            fontsize=10, color="#5f6368", ha="center")
    ax.text(60, 7.5, "Readout: the same Thought Atlas judge on the first 4096 reasoning tokens → one frozen “Idea-likeness” score (fit once on unsteered traces).",
            fontsize=10, color="#5f6368", ha="center")
    ax.text(60, 3.5, "Verdict from the data: the prompt and the sustained edit work; the early pulse does nothing.", fontsize=11.5, color="#d1495b", ha="center", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig_methods.png", dpi=200); plt.close(fig)


def fig_schedule(run: str = "sched1", dose_label: str = "2 gaps"):
    """Equal-dose order swap: Idea frontier, Math accuracy, and the early/late phase profile per arm."""
    import glob
    from src.perf.answer_grading import grade_math_answer
    rd = DATA / f"runs/{run}"
    L = 19
    arms = [("baseline", "Unsteered", GREY, "o"), (f"sus_idea_L{L}", "Sustained explore", "#e08a94", "o"), (f"sus_math_L{L}", "Sustained check", "#7a9cdc", "o"),
            (f"idea_then_math_L{L}", "Explore → check", "#b0203a", "D"), (f"math_then_idea_L{L}", "Check → explore", "#1f4fa3", "D"),
            (f"random_then_opp_L{L}", "Random + then −", "#b5b5b5", "s")]
    tr = pl.concat([pl.read_parquet(p, columns=["trace_id", "condition", "instance_id", "task_type", "answer_text", "reference_answer", "hard_loop"]) for p in glob.glob(str(rd / "traces_*.parquet"))], how="diagonal_relaxed")
    # only arms whose generation is complete (every domain at the run's task count)
    n_full = tr.filter(pl.col("condition") == "baseline").group_by("task_type").len()["len"].min()
    cnt = tr.group_by(["condition", "task_type"]).len().group_by("condition").agg(pl.col("len").min().alias("n"))
    have = set(cnt.filter(pl.col("n") >= n_full)["condition"].to_list()); arms = [a for a in arms if a[0] in have]
    qp = glob.glob(str(rd / "judge/quality__*.parquet"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.1, 0.9, 1.2]})
    # A: frontier, paired against the same tasks unsteered
    ax = axes[0]
    if qp:
        q = pl.read_parquet(qp[0]); j = q.join(tr, on="trace_id").filter(pl.col("task_type") == "idea")
        base = j.filter(pl.col("condition") == "baseline").select(["instance_id", pl.col("axis_originality").alias("o0"), pl.col("axis_feasibility").alias("f0")])
        for c, lab, col, mk in arms:
            g = j.filter(pl.col("condition") == c)
            if g.height < 30: continue
            o, f = g["axis_originality"].to_numpy().astype(float), g["axis_feasibility"].to_numpy().astype(float)
            o, f = o[np.isfinite(o)], f[np.isfinite(f)]
            ax.errorbar(o.mean(), f.mean(), xerr=1.96 * o.std() / np.sqrt(len(o)), yerr=1.96 * f.std() / np.sqrt(len(f)), fmt=mk, color=col, ms=9, capsize=3, label=f"{lab} (n={len(o)})")
        ax.set_xlabel("Originality (judge, 1–10)"); ax.set_ylabel("Feasibility (judge, 1–10)")
        ax.set_title("Idea tasks: originality vs feasibility of the answer", fontsize=11, loc="left"); ax.legend(frameon=False, fontsize=8.5)
    else:
        ax.text(0.5, 0.5, "quality judge pending", ha="center", va="center", transform=ax.transAxes, color=GREY); ax.set_axis_off()
    # B: math accuracy
    ax = axes[1]; m = tr.filter(pl.col("task_type") == "math"); vals = []
    for c, lab, col, mk in arms:
        g = m.filter(pl.col("condition") == c)
        if g.height < 30: continue
        acc = np.array([(grade_math_answer(a.replace("**", ""), r).get("success") or 0) if a else 0 for a, r in zip(g["answer_text"].to_list(), g["reference_answer"].to_list())], float)
        vals.append((lab, acc.mean(), 1.96 * acc.std() / np.sqrt(len(acc)), col))
    y = np.arange(len(vals))[::-1]
    ax.barh(y, [v[1] for v in vals], xerr=[v[2] for v in vals], color=[v[3] for v in vals], capsize=3)
    ax.set_yticks(y); ax.set_yticklabels([v[0] for v in vals]); ax.set_xlim(0, 0.8); ax.set_xlabel("Math accuracy (unanswered = wrong)")
    ax.set_title("Math tasks: accuracy by schedule", fontsize=11, loc="left")
    # C: phase profile
    ax = axes[2]; ps = pl.read_csv(rd / "analysis/phase_paired_shifts.csv").filter(pl.col("metric") == "s_all")
    x = np.arange(len(arms) - 1); w = 0.19
    for k, (dom, ph, hatch, off) in enumerate([("idea", "early", None, -1.5), ("idea", "late", "//", -0.5), ("math", "early", None, 0.5), ("math", "late", "//", 1.5)]):
        vals, err, cols = [], [], []
        for c, lab, col, mk in arms[1:]:
            r = ps.filter((pl.col("condition") == c) & (pl.col("task_type") == dom) & (pl.col("phase") == ph))
            if r.height == 0: vals.append(0); err.append(0); cols.append(col); continue
            vals.append(float(r["paired_shift"][0])); err.append(float(r["shift_hi"][0] - r["paired_shift"][0])); cols.append(col)
        ax.bar(x + off * w, vals, w, yerr=err, color=[RED if dom == "idea" else BLUE] * len(vals), alpha=0.9 if ph == "early" else 0.45, hatch=hatch, capsize=2,
               label=f"{'Idea' if dom == 'idea' else 'Math'} tasks, {ph} half")
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels([a[1] for a in arms[1:]], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Idea-likeness change vs unsteered (logits)"); ax.set_title("Did the schedule take hold? early vs late half", fontsize=11, loc="left"); ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Equal-dose order swap (Qwen3.5-9B, layer 19, |dose| = {dose_label}, 200 tasks per domain, switch at the task's own midpoint)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / f"fig_schedule_{run}.png", dpi=200); plt.close(fig)


def fig_trajectory(run: str = "sched1", edges=(0, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192), min_segs: int = 3):
    """Idea-likeness by token position for each arm (mean over traces with >= min_segs valid segments in the bin)."""
    import glob, json
    from .phenotype import binned_amounts, score_with_frozen
    from .common import JUDGE_MODEL, load_tokenizer
    rd = DATA / f"runs/{run}"; L = 19
    spec = json.loads((rd / "run.json").read_text()); window = spec["window"]
    frozen = json.loads((DATA / "phenotype" / f"frozen_w{window}.json").read_text())
    tr = pl.concat([pl.read_parquet(p) for p in sorted(rd.glob("traces_*.parquet"))], how="diagonal_relaxed")
    tag = JUDGE_MODEL.replace("/", "_")
    tb = pl.concat([pl.read_parquet(p) for p in sorted((rd / "judge").rglob(f"trackB_full__{tag}*.parquet"))], how="diagonal_relaxed").unique(["trace_id", "seg_idx"], keep="last")
    n_full = tr.filter(pl.col("condition") == "baseline").group_by("task_type").len()["len"].min()
    cnt = tr.group_by(["condition", "task_type"]).len().group_by("condition").agg(pl.col("len").min().alias("n"))
    full = set(cnt.filter(pl.col("n") >= n_full)["condition"].to_list())
    judged = set(tb["trace_id"].to_list())
    arms = [("baseline", "Unsteered", GREY, "-"), (f"sus_idea_L{L}", "Sustained explore", "#e08a94", "-"), (f"sus_math_L{L}", "Sustained check", "#7a9cdc", "-"),
            (f"idea_then_math_L{L}", "Explore → check", "#b0203a", "-"), (f"math_then_idea_L{L}", "Check → explore", "#1f4fa3", "-"), (f"random_then_opp_L{L}", "Random + then −", "#8a8a8a", "--"),
            (f"idea_then_math_fix_L{L}", "Explore → check (switch 2k)", "#b0203a", ":"), (f"math_then_idea_fix_L{L}", "Check → explore (switch 2k)", "#1f4fa3", ":")]
    arms = [a for a in arms if a[0] in full and tr.filter(pl.col("condition") == a[0])["trace_id"].is_in(list(judged)).mean() > 0.9]
    ba = binned_amounts(tr, tb, list(edges), tok=load_tokenizer())
    sc = score_with_frozen(ba, frozen).join(tr.select(["trace_id", "condition", "task_type", "instance_id"]), on="trace_id")
    sc = sc.filter(pl.col("n_valid_segments") >= min_segs)
    sw = tr.filter(pl.col("switch_at").is_not_null() & ~pl.col("condition").str.contains("_fix_"))   # band = midpoint-rule arms only
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False)
    mids = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    for ax, dom in zip(axes, ["idea", "math"]):
        for c, lab, col, ls in arms:
            g = sc.filter((pl.col("condition") == c) & (pl.col("task_type") == dom)).group_by("bin").agg(pl.col("s_all").mean().alias("m"), (pl.col("s_all").std() / pl.len().sqrt()).alias("se"), pl.len().alias("n")).sort("bin")
            x = [mids[int(b)] for b in g["bin"].to_list()]
            ax.plot(x, g["m"].to_numpy(), ls, color=col, lw=2.2 if "then" in c else 1.6, marker="o", ms=4, label=lab)
            ax.fill_between(x, (g["m"] - 1.96 * g["se"]).to_numpy(), (g["m"] + 1.96 * g["se"]).to_numpy(), color=col, alpha=0.12)
        if sw.height:
            q = sw.filter(pl.col("task_type") == dom)["switch_at"].to_numpy()
            ax.axvspan(np.percentile(q, 25), np.percentile(q, 75), color="#000", alpha=0.06); ax.axvline(np.median(q), color="#000", lw=0.8, ls=":")
            ax.text(np.median(q), ax.get_ylim()[1], " switch (median, IQR)", fontsize=8, color=DARK, va="top")
        ax.set_xlabel("Position in reasoning (tokens)"); ax.set_title(f"{'Idea' if dom == 'idea' else 'Math'} tasks", fontsize=11.5, loc="left")
    axes[0].set_ylabel("Idea-likeness score (logits, + = Idea-like)"); axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle(f"How the reasoning signature moves along the trace under each schedule ({run}, layer 19, 200 tasks per domain)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / f"fig_trajectory_{run}.png", dpi=200); plt.close(fig)


def fig_diversity(run: str = "sched1_div"):
    """Per-input diversity (40 tasks x 5 seeds), paired against unsteered, with bootstrap CIs."""
    rd = DATA / f"runs/{run}/analysis"; L = 19
    res = pl.read_csv(rd / "diversity_per_input_paired.csv")
    arms = [(f"sus_idea_L{L}", "Sustained\nexplore", "#e08a94"), (f"sus_math_L{L}", "Sustained\ncheck", "#7a9cdc"),
            (f"idea_then_math_L{L}", "Explore →\ncheck", "#b0203a"), (f"math_then_idea_L{L}", "Check →\nexplore", "#1f4fa3")]
    def panel(ax, dom, metric, title, ylabel, flip=False, scale=1.0):
        xs, ys, los, his, cols = [], [], [], [], []
        for k, (c, lab, col) in enumerate(arms):
            r = res.filter((pl.col("task_type") == dom) & (pl.col("metric") == metric) & (pl.col("condition") == c))
            if r.height == 0: continue
            sgn = -1 if flip else 1
            ys.append(sgn * float(r["paired_diff"][0]) * scale); lo, hi = sgn * float(r["lo"][0]) * scale, sgn * float(r["hi"][0]) * scale
            los.append(min(lo, hi)); his.append(max(lo, hi)); xs.append(k); cols.append(col)
        ys, los, his = np.array(ys), np.array(los), np.array(his)
        ax.bar(xs, ys, 0.62, color=cols, yerr=[ys - los, his - ys], capsize=3)
        ax.axhline(0, color="k", lw=0.7); ax.set_xticks(range(len(arms))); ax.set_xticklabels([a[1] for a in arms], fontsize=9)
        ax.set_title(title, fontsize=10.5, loc="left"); ax.set_ylabel(ylabel, fontsize=9)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panel(axes[0, 0], "math", "self_bleu_think", "Math: reasoning-path diversity across seeds (lexical)", "−Δ self-BLEU (higher = more varied)", flip=True)
    panel(axes[0, 1], "math", "vendi_think", "Math: reasoning-path diversity across seeds (semantic)", "Δ Vendi score of bge-m3 embeddings")
    panel(axes[0, 2], "math", "n_methods", "Math: distinct solution methods per task (judge)", "Δ methods per task (5 samples)")
    panel(axes[1, 0], "math", "pass@1", "Math: accuracy", "Δ pass@1")
    panel(axes[1, 1], "math", "pass@5", "Math: any of 5 samples correct", "Δ pass@5")
    panel(axes[1, 2], "idea", "cos_dist_answer", "Idea: distinctness of the 5 ideas per task", "Δ mean pairwise cosine distance of answers")
    fig.suptitle("Does the schedule change how varied the reasoning is? (40 tasks per domain × 5 seeds, paired vs unsteered, 95% CI)", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / f"fig_diversity_{run}.png", dpi=200); plt.close(fig)


def fig_controller(run: str = "sigpilot3_math"):
    """Signature-controller replication on held-out Math: accuracy per arm, behavior shifts, gap covered."""
    rd = DATA / f"runs/{run}/analysis"
    su = pl.read_csv(rd / "summary.csv"); sh = pl.read_csv(rd / "paired_shifts.csv").filter(pl.col("task_type") == "math")
    arms = [("baseline", "Unsteered", GREY), ("cl_math_A2_g2p0", "Controller → A2\n(closed-loop)", "#b0203a"), ("ol_math_A2_g3p0", "Controller → A2\n(open-loop)", "#d1495b"),
            ("cl_random_g2p0", "Random subspace\n(closed-loop)", "#b5b5b5"), ("ol_random_g3p0", "Random subspace\n(open-loop)", "#d0d0d0"), ("sus_math_L19", "Rank-one check\naxis (reference dose)", "#7a9cdc")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.1, 1.2, 0.9]})
    ax = axes[0]; y = np.arange(len(arms))[::-1]
    for k, (c, lab, col) in enumerate(arms):
        r = su.filter(pl.col("condition") == c).row(0, named=True)
        ax.barh(y[k], r["math_success_all"], color=col, xerr=[[r["math_success_all"] - r["math_success_all_lo"]], [r["math_success_all_hi"] - r["math_success_all"]]], capsize=3)
        ax.text(r["math_success_all_hi"] + 0.015, y[k], f"{r['math_success_all']:.2f}   answered {r['has_answer']:.2f}, loops {r['hard_loop']:.2f}", va="center", fontsize=8.5, color=DARK)
    ax.set_yticks(y); ax.set_yticklabels([a[1] for a in arms], fontsize=9); ax.set_xlim(0, 1.0); ax.set_xlabel("Math accuracy at a 12k-token budget (unanswered = wrong)")
    ax.set_title("199 held-out MATH tasks", fontsize=11, loc="left")
    ax = axes[1]
    behs = [("subgoal__rate", "subgoal"), ("verification__rate", "verification"), ("Question_and_Answering__rate", "Q&A"), ("Conflict_of_Perspectives__rate", "conflict"), ("Perspective_Shift__rate", "shift"), ("backtracking__rate", "backtracking")]
    x = np.arange(len(behs)); w = 0.16
    for k, (c, lab, col) in enumerate(arms[1:]):
        vals, err = [], []
        for m, _ in behs:
            r = sh.filter((pl.col("condition") == c) & (pl.col("metric") == m))
            if r.height == 0:
                # subgoal/Q&A/conflict are not in paired_shifts for every run: fall back to summary difference
                b = su.filter(pl.col("condition") == "baseline").row(0, named=True); a = su.filter(pl.col("condition") == c).row(0, named=True)
                vals.append(a[m] - b[m]); err.append(0.0)
            else:
                vals.append(float(r["paired_shift"][0])); err.append(float(r["shift_hi"][0] - r["paired_shift"][0]))
        ax.bar(x + (k - 2) * w, vals, w, yerr=err, color=col, capsize=2, label=lab.replace("\n", " "))
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels([b for _, b in behs], fontsize=8.5, rotation=20, ha="right"); ax.set_ylabel("Δ rate vs unsteered (paired)")
    ax.set_ylim(-0.045, 0.1); ax.set_title("What changed in the reasoning", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax = axes[2]
    ax.axis("off")
    txt = ("How the direction was found\n\n"
           "1. 800 natural traces × 4 domains,\n   judged with the atlas labels\n"
           "2. Residual state at every segment\n   boundary → future behaviour rates\n   (rank-8 ridge readout, layer 16)\n"
           "3. Targets = archetype medoids of\n   high-quality traces per domain\n"
           "4. Steer so the readout matches the\n   target; random subspace = control\n\n"
           "A2 = 'set subgoals, verify less':\n   a direction the explore/check\n   axis does not contain (cos 0.2)")
    ax.text(0.0, 0.98, txt, va="top", ha="left", fontsize=9.5, color=DARK, family="monospace")
    fig.suptitle("A low-rank readout of the reasoning signature gives a new controllable direction, and it raises MATH accuracy at fixed budget", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / f"fig_controller_{run}.png", dpi=200); plt.close(fig)


def fig_crossdomain():
    """The 2x2 of direction x task domain: outcome per arm (paired vs unsteered) with matched random controls."""
    import glob
    def paired(sc, a, b, col, rng):
        x = sc.filter(pl.col("condition") == a).select(["instance_id", pl.col(col).cast(pl.Float64).alias("x")]).join(sc.filter(pl.col("condition") == b).select(["instance_id", pl.col(col).cast(pl.Float64).alias("y")]), on="instance_id")
        d = (x["x"] - x["y"]).drop_nulls().to_numpy(); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    rng = np.random.default_rng(0)
    cells = [
        ("Math direction → Math tasks", "sigpilot3_math", "math", [("cl_math_A2_g2p0", "A2 closed-loop", RED), ("ol_math_A2_g3p0", "A2 open-loop", "#d1495b"), ("cl_random_g2p0", "random (closed)", "#b5b5b5"), ("ol_random_g3p0", "random (open)", "#d0d0d0")]),
        ("Idea directions → Math tasks", "sigx_math", "math", [("ol_idea_A4_samevec_g3p0", "Idea A4 (low shift) vector", "#7a9cdc"), ("ol_idea_A3_samevec_g3p0", "Idea A3 (high shift) vector", "#e08a94"), ("cl_idea_A3_abs_g2p0", "absolute A3 profile", "#b0203a"), ("ol_random_g3p0", "random (open)", "#d0d0d0")]),
        ("Math direction → Idea tasks", "sigx_idea", "idea", [("ol_math_A2_samevec_g3p0", "A2 vector (same as on Math)", RED), ("cl_math_A2_abs_g2p0", "absolute A2 profile", "#b0203a"), ("cl_random_g2p0", "random (closed)", "#b5b5b5")]),
        ("Idea directions → Idea tasks", "sigpilot2_idea", "idea", [("ol_idea_A3_g3p0", "Idea A3 (high shift) vector", "#e08a94"), ("ol_idea_A4_g3p0", "Idea A4 (low shift) vector", "#7a9cdc"), ("cl_idea_A3_g3p0", "A3 closed-loop", "#b0203a"), ("ol_random_g3p0", "random (open)", "#d0d0d0")]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (title, run, dom, arms) in zip(axes.ravel(), cells):
        rd = DATA / f"runs/{run}"; sc = pl.read_parquet(rd / "analysis/scored.parquet").filter(pl.col("task_type") == dom)
        su = pl.read_csv(rd / "analysis/summary.csv").filter(pl.col("task_type") == dom)
        if dom == "math":
            y = np.arange(len(arms))[::-1]
            for k, (c, lab, col) in enumerate(arms):
                m, lo, hi = paired(sc, c, "baseline", "math_success_all", rng); r = su.filter(pl.col("condition") == c).row(0, named=True)
                ax.barh(y[k], m, color=col, xerr=[[m - lo], [hi - m]], capsize=3)
                ax.text(max(hi, 0) + 0.01, y[k], f"loops {r['hard_loop']:.2f}, answered {r['has_answer']:.2f}", va="center", fontsize=8, color=GREY)
            ax.axvline(0, color="k", lw=0.7); ax.set_yticks(y); ax.set_yticklabels([a[1] for a in arms], fontsize=9); ax.set_xlim(-0.45, 0.45)
            ax.set_xlabel("Δ accuracy vs unsteered (paired, 95% CI)")
        else:
            q = pl.read_parquet(glob.glob(str(rd / "judge/quality__*.parquet"))[0]).select(["trace_id", "axis_originality", "axis_feasibility"])
            tr = pl.concat([pl.read_parquet(p, columns=["trace_id", "condition", "instance_id"]) for p in glob.glob(str(rd / "traces_*.parquet"))], how="diagonal_relaxed")
            j = q.join(tr, on="trace_id")
            for k, (c, lab, col) in enumerate(arms):
                if j.filter((pl.col("condition") == c) & pl.col("axis_originality").is_not_null()).height < 8: continue
                o = paired(j, c, "baseline", "axis_originality", rng); f = paired(j, c, "baseline", "axis_feasibility", rng); r = su.filter(pl.col("condition") == c).row(0, named=True)
                ax.errorbar(o[0], f[0], xerr=[[o[0] - o[1]], [o[2] - o[0]]], yerr=[[f[0] - f[1]], [f[2] - f[0]]], fmt="o", color=col, ms=8, capsize=3, label=f"{lab} (loops {r['hard_loop']:.2f})")
            ax.axhline(0, color="k", lw=0.7); ax.axvline(0, color="k", lw=0.7); ax.set_xlim(-1.1, 0.5); ax.set_ylim(-0.8, 0.5)
            ax.set_xlabel("Δ originality vs unsteered (paired)"); ax.set_ylabel("Δ feasibility"); ax.legend(frameon=False, fontsize=7.5, loc="lower left")
        ax.set_title(title, fontsize=11, loc="left")
    fig.suptitle("Direction × task domain: the structure-inducing direction helps Math from either origin and never helps Idea", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig_crossdomain.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    fig_steering(); fig_profile(); fig_readout(); fig_outcomes(); fig_methods(); print("->", OUT)
