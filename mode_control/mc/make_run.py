"""Write data/runs/<run>/run.json for a steering run.

Magnitudes are derived from the fitted directions on baseline activations so that controls are
matched: random subspaces get |Δh| equal to the mean |Δh| of the reflect (gamma=-1) condition;
additive injections use the prompt-induced projection shift E[u^T d] from Stage B.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from .common import DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--layer", default="auto", help="int or auto (Gate 2 argmax in the middle band)")
    ap.add_argument("--phase", default="mean32")
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--n-per-domain", type=int, default=40)
    ap.add_argument("--task-seed", type=int, default=20260903)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--pulse-k", type=int, default=64)
    ap.add_argument("--wave", default="core", choices=["core", "full", "strong", "strong_sd", "power_steer", "dose_idea", "schedule", "corpus", "sigctrl", "sigctrl2", "sigctrl3", "sigctrl4"])
    ap.add_argument("--band", default="12,14,16,18,20,22,24", help="layer band for strong-wave conditions")
    ap.add_argument("--direction", default="prompt_r1", help="direction key for the strong wave")
    ap.add_argument("--wrong-layers", default="2,29")
    ap.add_argument("--tag", default="", help="directions/map tag, e.g. _v2")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="generation budget (default = window)")
    ap.add_argument("--only", default=None, help="comma list of condition names to keep")
    ap.add_argument("--domains", default="math,idea", help="comma list of task domains (data/v2/tasks/<dom>.parquet)")
    ap.add_argument("--map", default=None, help="sigctrl wave: mapping npz name under data/sigctrl")
    ap.add_argument("--ref-domain", default="math", help="sigctrl4: domain whose mean defines the open-loop direction")
    ap.add_argument("--exclude-runs", default=None, help="comma list of runs whose tasks must not be selected (held-out evaluation)")
    ap.add_argument("--targets", default=None, help="sigctrl wave: comma list of archetype names from archetypes.json (default: all for the run's domains)")
    ap.add_argument("--gains", default="1,2", help="sigctrl wave: controller gains")
    ap.add_argument("--rel-lam", type=float, default=1e-3, help="sigctrl2: controller ridge relative to the readout scale")
    ap.add_argument("--cap", type=float, default=0.25, help="sigctrl2: per-token cap on |delta h| / |h|")
    ap.add_argument("--open-norm", type=float, default=7.0, help="sigctrl2: norm of the open-loop constant direction (rank-1 dose = 2 gaps = 7.0 at layer 19)")
    ap.add_argument("--dose-gap", type=float, default=2.0, help="schedule wave: |alpha| in gap units at the single layer")
    ap.add_argument("--switch-frac", type=float, default=0.5, help="schedule wave: switch at this fraction of the task's own unsteered length")
    ap.add_argument("--switch-fixed", type=int, default=2048, help="schedule wave: fixed-token switch for the robustness arms")
    a = ap.parse_args()
    selection = None
    if a.layer == "auto":
        m = pl.read_csv(DATA / f"directions/layer_phase_map{a.tag}.csv").filter(
            (pl.col("direction") == "prompt_r1") & (pl.col("phase") == a.phase) &
            (pl.col("layer") >= 6) & (pl.col("layer") <= 26) &
            (pl.col("delta_sd_units") > 0) & (pl.col("paired_sign_frac") > 0.5))
        m = m.with_columns(((pl.col("g2_cog_rho_math") + pl.col("g2_cog_rho_idea") +
                             pl.col("g2_conv_rho_math") + pl.col("g2_conv_rho_idea")) / 4).alias("g2_mean_rho"))
        best = m.sort("g2_mean_rho", descending=True).row(0, named=True)
        a.layer = int(best["layer"])
        selection = {"rule": "argmax mean within-domain rho (cog+conv, math+idea) over layers 6-26 with delta>0 and sign_frac>0.5",
                     **{k: best[k] for k in ["layer", "g2_mean_rho", "delta_sd_units", "proj_domain_auroc", "paired_sign_frac", "cos_prompt_rawdom"]}}
        print("auto layer selection:", selection)
    a.layer = int(a.layer)
    dirs = np.load(DATA / "directions" / f"directions_w{a.window}{a.tag}.npz")
    def _first(*names):
        for n in names:
            if (DATA / "acts" / n).exists():
                return DATA / "acts" / n
        raise FileNotFoundError(names)
    A = np.load(_first("stageA_k64.npz", "stageA_k192.npz"), allow_pickle=True)
    B = np.load(_first("stageB_k32.npz", "stageB_k192.npz"), allow_pickle=True)
    phasesA = A["phases"].tolist(); phasesB = B["phases"].tolist()
    l, ph = a.layer, a.phase
    acts = A["acts"][:, l, phasesA.index(ph), :].astype(np.float32)
    center = dirs[f"center|{l}|{ph}"]
    u = dirs[f"prompt_r1|{l}|{ph}"] if f"prompt_r1|{l}|{ph}" in dirs.files else dirs[f"rawdom_r1|{l}|{ph}"]
    z = (acts - center) @ u[:, 0]
    rms_reflect = float(2.0 * np.abs(z).mean())            # |Δh| for gamma=-1 (rank 1)
    D = B["diffs"][:, :, l, phasesB.index(ph), :].reshape(-1, u.shape[0]).astype(np.float32)
    prompt_shift = float((D @ u[:, 0]).mean())              # exploratory - checking, projection units
    common = {"layer": l, "phase": ph, "pulse_k": a.pulse_k}
    conds = [
        {"name": "baseline", "mode": "baseline"},
        {"name": "amplify_g2", "mode": "gamma", "direction": "prompt_r1", "gamma": 2.0, **common},
        {"name": "neutralize_g0", "mode": "gamma", "direction": "prompt_r1", "gamma": 0.0, **common},
        {"name": "reflect_gm1", "mode": "gamma", "direction": "prompt_r1", "gamma": -1.0, **common},
        {"name": "random_r1_a", "mode": "random", "rank": 1, "seed": 1, "alpha_raw": rms_reflect, **common},
        {"name": "random_r1_b", "mode": "random", "rank": 1, "seed": 2, "alpha_raw": -rms_reflect, **common},
        {"name": "rawdom_r1_gm1", "mode": "gamma", "direction": "rawdom_r1", "gamma": -1.0, **common},
        {"name": "prompt_pw_gm1", "mode": "gamma", "direction": "prompt_pw_r1", "gamma": -1.0, **common},
        {"name": "prompt_checking", "mode": "baseline", "system_prompt": "checking:0"},
        {"name": "prompt_exploratory", "mode": "baseline", "system_prompt": "exploratory:0"},
    ]
    if a.wave == "strong":
        band = [int(x) for x in a.band.split(",")]
        L1 = {"layer": l, "phase": ph, "sustained": True}
        LB = {"layers": band, "phase": ph, "sustained": True}
        dk = a.direction
        conds = [
            {"name": "baseline", "mode": "baseline"},
            {"name": f"reflect3_sus_L{l}", "mode": "gamma", "direction": dk, "gamma": -3.0, **L1},
            {"name": "band_reflect_sus", "mode": "gamma", "direction": dk, "gamma": -1.0, **LB},
            {"name": "band_add_minus_sus", "mode": "add", "direction": dk, "alpha_gap": -1.0, **LB},
            {"name": "band_add_plus_sus", "mode": "add", "direction": dk, "alpha_gap": 1.0, **LB},
            {"name": "band_random_sus", "mode": "random", "rank": 1, "seed": 7, "direction": dk, "alpha_gap": 1.0, **LB},
            {"name": "band_swap_sus", "mode": "swap", "direction": dk, **LB},
            {"name": "band_rawdom_add_minus_sus", "mode": "add", "direction": "rawdom_r1", "alpha_gap": -1.0, **LB},
        ]
    if a.wave == "strong_sd":
        band = [int(x) for x in a.band.split(",")]
        LB = {"layers": band, "phase": ph, "sustained": True}
        dk = a.direction
        conds = [
            {"name": "baseline", "mode": "baseline"},
            {"name": "band_add_minus_sd3", "mode": "add", "direction": dk, "alpha_sd": -3.0, **LB},
            {"name": "band_add_plus_sd3", "mode": "add", "direction": dk, "alpha_sd": 3.0, **LB},
            {"name": "band_random_sd3", "mode": "random", "rank": 1, "seed": 7, "direction": dk, "alpha_sd": 3.0, **LB},
            {"name": "band_rawdom_add_minus_gap1", "mode": "add", "direction": "rawdom_r1", "alpha_gap": -1.0, **LB},
            {"name": "band_rawdom_add_plus_gap1", "mode": "add", "direction": "rawdom_r1", "alpha_gap": 1.0, **LB},
        ]
    if a.wave == "power_steer":
        band = [int(x) for x in a.band.split(",")]
        L1 = {"layer": l, "phase": ph, "sustained": True}; LB = {"layers": band, "phase": ph, "sustained": True}
        conds = [
            {"name": "baseline", "mode": "baseline"},
            {"name": f"reflect1_sus_L{l}", "mode": "gamma", "direction": "prompt_r1", "gamma": -1.0, **L1},
            {"name": f"reflect3_sus_L{l}", "mode": "gamma", "direction": "prompt_r1", "gamma": -3.0, **L1},
            {"name": "band_add_minus_sus", "mode": "add", "direction": "prompt_r1", "alpha_gap": -1.0, **LB},
            {"name": "band_add_plus_half_sus", "mode": "add", "direction": "prompt_r1", "alpha_gap": 0.5, **LB},
            {"name": "band_random_sus", "mode": "random", "rank": 1, "seed": 7, "direction": "prompt_r1", "alpha_gap": 1.0, **LB},
            {"name": "band_swap_sus", "mode": "swap", "direction": "prompt_r1", **LB},
        ]
    if a.wave == "dose_idea":
        band = [int(x) for x in a.band.split(",")]
        L1 = {"layer": l, "phase": ph, "sustained": True}; LB = {"layers": band, "phase": ph, "sustained": True}
        conds = [
            {"name": "baseline", "mode": "baseline"},
            {"name": "band_add_plus_q_sus", "mode": "add", "direction": "prompt_r1", "alpha_gap": 0.25, **LB},
            {"name": "band_add_plus_half_sus", "mode": "add", "direction": "prompt_r1", "alpha_gap": 0.5, **LB},
            {"name": f"add_plus1_sus_L{l}", "mode": "add", "direction": "prompt_r1", "alpha_gap": 1.0, **L1},
            {"name": f"add_plus2_sus_L{l}", "mode": "add", "direction": "prompt_r1", "alpha_gap": 2.0, **L1},
            {"name": f"add_minus2_sus_L{l}", "mode": "add", "direction": "prompt_r1", "alpha_gap": -2.0, **L1},
            {"name": "band_rawdom_add_minus_q_sus", "mode": "add", "direction": "rawdom_r1", "alpha_gap": -0.25, **LB},
        ]
    if a.wave == "sigctrl":
        # signature controller pilot (D-33 stage 4): every archetype of the run's domains as a target, the domain centroid,
        # and a norm-matched random-subspace control per gain; all sustained at the mapping's layer
        arch = json.load(open(DATA / "sigctrl" / "archetypes.json"))
        doms = a.domains.split(",")
        names = a.targets.split(",") if a.targets else [k for k, v in arch.items() if v["domain"] in doms]
        gains = [float(g) for g in a.gains.split(",")]
        conds = [{"name": "baseline", "mode": "baseline"}]
        for g in gains:
            gtag = str(g).replace(".", "p")
            for nm in names:
                conds.append({"name": f"tgt_{nm}_g{gtag}", "mode": "target", "map": a.map, "phi_star": nm, "gain": g, "sustained": True})
            conds.append({"name": f"tgt_random_g{gtag}", "mode": "target", "map": a.map, "phi_star": names[0], "gain": g, "sustained": True, "random_subspace": True, "seed": 7})
    if a.wave == "sigctrl2":
        # second pilot: (i) closed-loop controller with ridge relative to the readout scale and a per-token norm cap,
        # (ii) open-loop constant direction toward the archetype at a fixed norm (comparable to the rank-one dose),
        # (iii) random-subspace controls for both, at each gain
        arch = json.load(open(DATA / "sigctrl" / "archetypes.json"))
        doms = a.domains.split(",")
        names = a.targets.split(",") if a.targets else [k for k, v in arch.items() if v["domain"] in doms]
        gains = [float(g) for g in a.gains.split(",")]
        conds = [{"name": "baseline", "mode": "baseline"}]
        for g in gains:
            gtag = str(g).replace(".", "p")
            for nm in names:
                conds.append({"name": f"cl_{nm}_g{gtag}", "mode": "target", "map": a.map, "phi_star": nm, "gain": g, "rel_lam": a.rel_lam, "cap": a.cap, "sustained": True})
                conds.append({"name": f"ol_{nm}_g{gtag}", "mode": "target", "map": a.map, "phi_star": nm, "gain": g, "rel_lam": a.rel_lam, "open_loop": True, "open_norm": a.open_norm, "sustained": True})
            conds.append({"name": f"cl_random_g{gtag}", "mode": "target", "map": a.map, "phi_star": names[0], "gain": g, "rel_lam": a.rel_lam, "cap": a.cap, "sustained": True, "random_subspace": True, "seed": 7})
            conds.append({"name": f"ol_random_g{gtag}", "mode": "target", "map": a.map, "phi_star": names[0], "gain": g, "rel_lam": a.rel_lam, "open_loop": True, "open_norm": a.open_norm, "sustained": True, "random_subspace": True, "seed": 7})
    if a.wave == "sigctrl3":
        # powered replication of the off-axis archetype result (D-36): the two A2 controllers that worked in the pilot,
        # their random-subspace controls, and the rank-one sustained check arm as the axis comparator
        nm = a.targets
        conds = [
            {"name": "baseline", "mode": "baseline"},
            {"name": f"cl_{nm}_g2p0", "mode": "target", "map": a.map, "phi_star": nm, "gain": 2.0, "rel_lam": a.rel_lam, "cap": a.cap, "sustained": True},
            {"name": f"ol_{nm}_g3p0", "mode": "target", "map": a.map, "phi_star": nm, "gain": 3.0, "rel_lam": a.rel_lam, "open_loop": True, "open_norm": a.open_norm, "sustained": True},
            {"name": "cl_random_g2p0", "mode": "target", "map": a.map, "phi_star": nm, "gain": 2.0, "rel_lam": a.rel_lam, "cap": a.cap, "sustained": True, "random_subspace": True, "seed": 7},
            {"name": "ol_random_g3p0", "mode": "target", "map": a.map, "phi_star": nm, "gain": 3.0, "rel_lam": a.rel_lam, "open_loop": True, "open_norm": a.open_norm, "sustained": True, "random_subspace": True, "seed": 7},
            {"name": "sus_math_L19", "mode": "add", "direction": "prompt_r1", "alpha_gap": -2.0, "layer": 19, "phase": ph, "sustained": True},
        ]
    if a.wave == "sigctrl4":
        # D-38/D-39: apply foreign archetypes to this run's domain. "ol_*_samevec": the open-loop vector exactly as used in
        # its home domain (reference mean = --ref-domain); "cl_*_abs": closed-loop toward the absolute archetype profile relative
        # to the row's own domain mean (first target only); matched random subspaces for both.
        names = a.targets.split(",")
        conds = [{"name": "baseline", "mode": "baseline"}]
        for nm in names:
            conds.append({"name": f"ol_{nm}_samevec_g3p0", "mode": "target", "map": a.map, "phi_star": nm, "gain": 3.0, "rel_lam": a.rel_lam, "open_loop": True, "open_norm": a.open_norm, "sustained": True, "ref_domain": a.ref_domain})
        conds.append({"name": f"cl_{names[0]}_abs_g2p0", "mode": "target", "map": a.map, "phi_star": names[0], "gain": 2.0, "rel_lam": a.rel_lam, "cap": a.cap, "sustained": True})
        conds.append({"name": "ol_random_g3p0", "mode": "target", "map": a.map, "phi_star": names[0], "gain": 3.0, "rel_lam": a.rel_lam, "open_loop": True, "open_norm": a.open_norm, "sustained": True, "random_subspace": True, "seed": 7, "ref_domain": a.ref_domain})
        conds.append({"name": "cl_random_g2p0", "mode": "target", "map": a.map, "phi_star": names[0], "gain": 2.0, "rel_lam": a.rel_lam, "cap": a.cap, "sustained": True, "random_subspace": True, "seed": 7})
    if a.wave == "corpus":
        conds = [{"name": "baseline", "mode": "baseline"}]   # natural multi-sample corpus for the signature controller (D-33)
    if a.wave == "schedule":
        # equal-dose order swap (reviewer design): additive +-dose along the prompt direction at one layer,
        # every generated position steered, so sum_t |delta h_t|^2 = dose^2 x positions in every steered arm.
        # Ordered arms differ only in which sign comes first; the switch is half of the task's own unsteered length.
        L1 = {"layer": l, "phase": ph, "sustained": True, "direction": "prompt_r1"}
        g = a.dose_gap
        swb = {"type": "baseline_frac", "frac": a.switch_frac, "min": 512, "max": 6144}
        swf = {"type": "fixed", "tokens": a.switch_fixed}
        conds = [
            {"name": "baseline", "mode": "baseline"},
            {"name": f"sus_idea_L{l}", "mode": "add", "alpha_gap": g, **L1},
            {"name": f"sus_math_L{l}", "mode": "add", "alpha_gap": -g, **L1},
            {"name": f"idea_then_math_L{l}", "mode": "add", "schedule": {"early": g, "late": -g, "switch": swb}, **L1},
            {"name": f"math_then_idea_L{l}", "mode": "add", "schedule": {"early": -g, "late": g, "switch": swb}, **L1},
            {"name": f"random_then_opp_L{l}", "mode": "random", "rank": 1, "seed": 7, "schedule": {"early": g, "late": -g, "switch": swb}, **L1},
            {"name": f"idea_then_math_fix_L{l}", "mode": "add", "schedule": {"early": g, "late": -g, "switch": swf}, **L1},
            {"name": f"math_then_idea_fix_L{l}", "mode": "add", "schedule": {"early": -g, "late": g, "switch": swf}, **L1},
        ]
    if a.wave == "full":
        conds += [
            {"name": "add_plus", "mode": "add", "direction": "prompt_r1", "alpha_raw": prompt_shift, **common},
            {"name": "add_minus", "mode": "add", "direction": "prompt_r1", "alpha_raw": -prompt_shift, **common},
            {"name": "prompt_checking_neutralized", "mode": "gamma", "direction": "prompt_r1", "gamma": 0.0,
             "system_prompt": "checking:0", **common},
            {"name": "prompt_exploratory_neutralized", "mode": "gamma", "direction": "prompt_r1", "gamma": 0.0,
             "system_prompt": "exploratory:0", **common},
            {"name": "swap_centroid", "mode": "swap", "direction": "prompt_r1", **common},
            {"name": "reflect_r4", "mode": "gamma", "direction": "prompt_r4", "gamma": -1.0, **common},
            {"name": "reflect_sustained", "mode": "gamma", "direction": "prompt_r1", "gamma": -1.0, **common, "sustained": True},
        ]
        for wl in [int(x) for x in a.wrong_layers.split(",")]:
            conds.append({"name": f"reflect_layer{wl}", "mode": "gamma", "direction": "prompt_r1", "gamma": -1.0,
                          "layer": wl, "phase": ph, "pulse_k": a.pulse_k})
    if a.only:
        keep = set(a.only.split(",")); conds = [c for c in conds if c["name"] in keep]
    spec = {"run": a.run, "window": a.window, "max_new_tokens": int(a.max_new_tokens or a.window),
            "n_per_domain": a.n_per_domain, "task_seed": a.task_seed, "domains": a.domains.split(","),
            "exclude_instances": (sorted({i for r in a.exclude_runs.split(",") for i in pl.read_parquet(DATA / "runs" / r / "tasks.parquet")["instance_id"].to_list()}) if a.exclude_runs else None),
            "seeds": [int(s) for s in a.seeds.split(",")], "directions": f"directions_w{a.window}{a.tag}.npz",
            "layer_selection": selection,
            "derived": {"rms_reflect": rms_reflect, "prompt_shift_proj": prompt_shift,
                        "proj_sd": float(z.std()), "proj_mean_math": float(z[A["task_type"] == "math"].mean()),
                        "proj_mean_idea": float(z[A["task_type"] == "idea"].mean())},
            "conditions": conds}
    rd = DATA / "runs" / a.run; rd.mkdir(parents=True, exist_ok=True)
    (rd / "run.json").write_text(json.dumps(spec, indent=2))
    print(json.dumps(spec["derived"], indent=2)); print(f"{len(conds)} conditions -> {rd / 'run.json'}")


if __name__ == "__main__":
    main()
