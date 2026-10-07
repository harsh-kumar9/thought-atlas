"""Transplant matrix (D-52), Countdown crossover vectors (D-51), F3 arms (D-53) and the H2a pre-registration.

Every vector lives in the readout A2 came from (map_corpus1_dm8_L16_r8, layer 16, rel_lam 1e-3) and has A2's norm, 6.75.
A domain's signature vector moves the 16 readout targets (future and next-4 behaviour rates over the first 8k reasoning
tokens) from the mean of the five domain profiles toward that domain's profile.

Subcommands (CPU on ada unless noted)
  corpus_specs  run.json for corpus1_moral_{a,b} (200 tasks x 4 seeds, 12k budget, as corpus1 for the other domains)
  eval_tasks    tasks for the new targets: tx_moral (200 held-out moral tasks), tx_idea (200 unused LiveIdeaBench keywords)
  profiles      16-d domain profiles from the corpus Track B labels, computed like signature_ctrl.extract (needs the
                tokenizer only); checks the four corpus1 domains against the map's stored means
  vectors       vec_sig_<dom>[_x0p5|_x1p5], vec_conv_up/down, vec_idea_commit (data/vsteer) + cosine and readout tables
  specs         run.json for tx_{math,gpqa,planning,idea,moral}, cd_countdown, cd_math
  prereg        within-task behaviour -> outcome slopes per target domain and the H2a predictions, written to
                data/transplant/prereg.json (with its sha256) before any transplant outcome is graded
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import polars as pl

from .common import BEH, DATA, JUDGE_MODEL

TX = DATA / "transplant"
VS = DATA / "vsteer"
SIG = DATA / "sigctrl"
DOMS = ["math", "gpqa", "planning", "idea", "moral"]
MAP = "map_corpus1_dm8_L16_r8.npz"
NORM = 6.75
REL_LAM = 1e-3
WINDOW = 8192
PROMPT_PORTS = {"prompt_math": "Think about this the way you would think about a math problem.",
                "prompt_idea": "Think about this the way you would brainstorm a research idea."}


def corpus_runs(dom: str) -> list[str]:
    return [f"corpus1_{dom}_a", f"corpus1_{dom}_b"]


# ---------------------------------------------------------------- corpus and task specs
def cmd_corpus_specs():
    for s, seeds in (("a", [1, 2]), ("b", [3, 4])):
        rd = DATA / "runs" / f"corpus1_moral_{s}"; rd.mkdir(parents=True, exist_ok=True)
        spec = {"run": rd.name, "engine": "vllm", "domains": ["moral"], "n_per_domain": 200, "task_seed": 20260909,
                "seeds": seeds, "max_new_tokens": 12288, "window": WINDOW, "conditions": [{"name": "baseline", "mode": "baseline"}],
                "note": "D-52: moral readout corpus, same size, budget and window as corpus1 for the other domains (vLLM engine)"}
        (rd / "run.json").write_text(json.dumps(spec, indent=2)); print("->", rd / "run.json")


def cmd_eval_tasks(seed: int = 20261006):
    from .steer import select_tasks
    corpus = select_tasks(200, 20260909, domains=["moral"])                       # the corpus1_moral task set
    mor = select_tasks(200, seed, domains=["moral"], exclude=corpus["instance_id"].to_list())
    assert not set(mor["instance_id"]) & set(corpus["instance_id"])
    for name, t in (("tx_moral", mor),):
        rd = DATA / "runs" / name; rd.mkdir(parents=True, exist_ok=True); t.write_parquet(rd / "tasks.parquet")
        print(f"[eval_tasks] {name}: {t.height} tasks")
    # idea: LiveIdeaBench keywords the atlas never used (src/utils/data_loaders.load_liveideabench wording and ids)
    from datasets import load_dataset
    from src.utils.data_loaders import _hash_norm
    d = load_dataset("6cf/LiveIdeaBench"); split = d["train"] if "train" in d else d[list(d.keys())[0]]
    from .common import TASKS_DIR
    used = set(pl.read_parquet(TASKS_DIR / "idea.parquet")["instance_id"].to_list())
    kws = sorted(set(k for k in split["keywords"] if k))
    instr = ("You are a scientist. Propose one novel, specific research idea related to the "
             "keyword below. State the idea and briefly justify its originality and feasibility.")
    rows = [{"instance_id": f"idea:{_hash_norm(kw)}", "task_type": "idea", "prompt": f"Keyword: {kw}\n\n{instr}", "reference_answer": None,
             "difficulty_raw": None, "metadata": json.dumps({"keyword": kw, "judge_scored": True, "rubric": ["originality", "feasibility"],
                                                              "source_dataset": "6cf/LiveIdeaBench", "note": "D-52 fresh keyword"}, ensure_ascii=False)}
            for kw in kws]
    fresh = [r for r in rows if r["instance_id"] not in used]
    pick = np.random.default_rng(seed).choice(len(fresh), 200, replace=False)
    t = pl.DataFrame([fresh[i] for i in sorted(pick)])
    rd = DATA / "runs" / "tx_idea"; rd.mkdir(parents=True, exist_ok=True); t.write_parquet(rd / "tasks.parquet")
    print(f"[eval_tasks] tx_idea: {t.height} fresh keywords ({len(kws)} total, {len(used)} used before, {len(fresh)} fresh)")


# ---------------------------------------------------------------- profiles
def _segment_states(run: str, tok, max_segments: int = 80) -> tuple[np.ndarray, list[str]]:
    """The 16 readout targets for every kept segment state of a run's baseline traces, exactly as extract() keeps them."""
    from .phenotype import _segments_with_offsets
    from .signature_ctrl import _load_run, segment_targets
    tr, tb = _load_run(run, WINDOW)
    n_seg = dict(zip(tb["trace_id"].to_list(), tb["n_segments"].to_list()))
    lab_by = {(k[0] if isinstance(k, tuple) else k): v.sort("seg_idx") for k, v in tb.group_by("trace_id")}
    out, tids = [], []
    for r in tr.iter_rows(named=True):
        tid = r["trace_id"]
        if tid not in n_seg or not r["reasoning_text_for_analysis"]:
            continue
        starts = _segments_with_offsets(tok, r["reasoning_text_for_analysis"], n_seg[tid])
        if starts is None:
            continue
        g = lab_by[tid]
        valid = (g["kim_parsed"].fill_null(False) & g["gandhi_parsed"].fill_null(False)).to_numpy()
        labels = g.select(BEH).fill_null(0).to_numpy().astype(np.float32)
        n_tok = len(tok(r["reasoning_text_for_analysis"], add_special_tokens=False)["input_ids"])
        ends = starts[1:] + [n_tok]
        if len(valid) != len(starts) or valid.sum() < 4:
            continue
        vidx = np.where(valid)[0]
        tg_v = segment_targets(labels[vidx]); pos = {int(i): k for k, i in enumerate(vidx)}
        keep = [int(i) for i in vidx if ends[i] - 1 < WINDOW]
        if len(keep) > max_segments:
            keep = [keep[j] for j in np.linspace(0, len(keep) - 1, max_segments).round().astype(int)]
        if keep:
            out.append(tg_v[[pos[i] for i in keep]][:, :16]); tids.append(tid)
    return np.concatenate(out), tids


def cmd_profiles():
    from .common import load_tokenizer
    tok = load_tokenizer(); TX.mkdir(parents=True, exist_ok=True)
    z = np.load(SIG / MAP, allow_pickle=True)
    stored = {str(d): m for d, m in zip(z["domain_names"].tolist(), z["domain_means"])}
    prof = {}
    for dom in DOMS:
        X, n_tr = [], 0
        for run in corpus_runs(dom):
            s, tids = _segment_states(run, tok); X.append(s); n_tr += len(tids)
        X = np.concatenate(X); m = X.mean(0)
        prof[dom] = {"mean16": m.tolist(), "n_traces": n_tr, "n_states": int(X.shape[0])}
        chk = f"; max |diff| vs map {np.abs(m - stored[dom]).max():.4f}" if dom in stored else " (not in the map)"
        print(f"[profiles] {dom}: {n_tr} traces, {X.shape[0]} states{chk}; future rates " + ", ".join(f"{b[:6]} {v:.3f}" for b, v in zip(BEH, m[:8])))
    (TX / "profiles.json").write_text(json.dumps(prof, indent=1)); print("->", TX / "profiles.json")


# ---------------------------------------------------------------- vectors
def readout():
    z = np.load(SIG / MAP, allow_pickle=True)
    W, sd = z["W"].astype(np.float32), z["sd"].astype(np.float32); k = W.shape[1]
    lam = REL_LAM * float(np.trace(W.T @ W) / k)
    G = (sd[:, None] * W) @ np.linalg.inv(W.T @ W + lam * np.eye(k, dtype=np.float32))       # as Steerer (common.py)
    return {"W": W, "sd": sd, "G": G, "layer": int(z["layer"]), "target_sd": z["target_sd"].astype(np.float32),
            "domain_means": {str(d): m for d, m in zip(z["domain_names"].tolist(), z["domain_means"])}}


def unit(x):
    return x / (np.linalg.norm(x) + 1e-9)


def save_vec(name: str, v: np.ndarray, layer: int, meta: dict):
    VS.mkdir(parents=True, exist_ok=True)
    np.savez(VS / f"vec_{name}.npz", v=v.astype(np.float32), layer=layer, meta=json.dumps(meta | {"norm": float(np.linalg.norm(v))}))


def load_vec(name: str) -> np.ndarray:
    return np.load(VS / f"vec_{name}.npz", allow_pickle=True)["v"].astype(np.float32)


def cmd_vectors(only_conv: bool = False):
    R = readout()
    # sanity: A2 rebuilt with this G equals the vector used since D-40
    arch = json.loads((SIG / "archetypes.json").read_text())
    a2 = NORM * unit((np.asarray(arch["math_A2"]["phi"], np.float32)[:16] - R["domain_means"]["math"]) @ R["G"].T)
    print(f"[vectors] A2 rebuilt vs vec_ol_math_A2: cosine {float(unit(a2) @ unit(load_vec('ol_math_A2'))):.6f}, norms {np.linalg.norm(a2):.3f} / {np.linalg.norm(load_vec('ol_math_A2')):.3f}")
    meta0 = {"source": "transplant", "map": MAP, "rel_lam": REL_LAM}
    ts = R["target_sd"]; conv = np.zeros(16, np.float32); conv[0:4] = ts[0:4]; conv[8:12] = ts[8:12]   # +1 within-domain SD, 4 Kim rates
    uc = unit(conv @ R["G"].T)
    save_vec("conv_up", NORM * uc, R["layer"], meta0 | {"kind": "conversational", "sign": 1, "dphi16": conv.tolist()})
    save_vec("conv_down", -NORM * uc, R["layer"], meta0 | {"kind": "conversational", "sign": -1, "dphi16": (-conv).tolist()})
    names = ["conv_up", "ol_math_A2", "ol_random"]
    if not only_conv:
        prof = json.loads((TX / "profiles.json").read_text())
        P = {d: np.asarray(prof[d]["mean16"], np.float32) for d in DOMS}
        grand = np.mean([P[d] for d in DOMS], 0)
        for d in DOMS:
            dphi = P[d] - grand
            u = unit(dphi @ R["G"].T)
            for tag, sc in (("", 1.0), ("_x0p5", 0.5), ("_x1p5", 1.5)):
                save_vec(f"sig_{d}{tag}", sc * NORM * u, R["layer"], meta0 | {"kind": "domain_signature", "domain": d, "scale": sc, "dphi16": dphi.tolist()})
        ic = NORM * unit(unit(load_vec("sig_idea")) + 0.5 * unit(load_vec("ol_math_A2")))
        save_vec("idea_commit", ic, R["layer"], meta0 | {"kind": "idea_plus_half_A2"})
        names = [f"sig_{d}" for d in DOMS] + ["ol_math_A2", "ol_gpqa_A4", "ol_planning_A2", "conv_up", "idea_commit", "ol_random",
                                               "sealdir_L17_n6p75", "rl_zero8k_T_s60"]
    report(R, names)


def report(R, names):
    V = {n: load_vec(n) for n in names}
    C = pl.DataFrame({"vector": names, **{n[:12]: [round(float(unit(V[a]) @ unit(V[n])), 2) for a in names] for n in names}})
    # readout-implied change of the 8 future rates per vector (rate points x 100): (v / sd) @ W
    imp = pl.DataFrame([{"vector": n, "norm": round(float(np.linalg.norm(V[n])), 2),
                         **{b[:8]: round(float(100 * ((V[n] / R["sd"]) @ R["W"])[k]), 2) for k, b in enumerate(BEH)}} for n in names])
    TX.mkdir(parents=True, exist_ok=True); C.write_csv(TX / "vector_cosines.csv"); imp.write_csv(TX / "vector_readout_shift.csv")
    pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_rows(30)
    print(C); print(imp)


# ---------------------------------------------------------------- run specs
def _vec(name, vector, **kw):
    return {"name": name, "mode": "vector", "vector": vector, "positions": "all", **kw}


def cmd_specs():
    base = {"engine": "vllm", "seeds": [1, 2], "max_new_tokens": 32768, "window": WINDOW}
    sig = [_vec(f"sig_{d}", f"sig_{d}") for d in DOMS]
    ports = [{"name": k, "mode": "baseline", "system": v} for k, v in PROMPT_PORTS.items()]
    dose = [_vec(f"sig_{d}{t}", f"sig_{d}{t}", n_tasks=100, seeds=[1]) for d in DOMS for t in ("_x0p5", "_x1p5")]
    f3 = [_vec("sealdir_matched", "sealdir_L17_n6p75"), _vec("rl_zeroT_s60", "rl_zero8k_T_s60")]
    commit = [_vec("idea_commit", "idea_commit")]
    fresh = [{"name": "baseline", "mode": "baseline"}, _vec("ol_random", "ol_random")]
    specs = {
        "tx_math": {"tasks_from": "sigpilot3_math", "borrow": "p1_budget:baseline,ol_math_A2,ol_random", "conditions": sig + ports + commit + dose},
        "tx_gpqa": {"tasks_from": "p2_gpqa", "borrow": "p2_gpqa:baseline,ol_math_A2,ol_gpqa_A4,ol_random", "conditions": sig + ports + commit + f3 + dose},
        "tx_planning": {"tasks_from": "p2_planning", "borrow": "p2_planning:baseline,ol_math_A2,ol_planning_A2,ol_random", "conditions": sig + ports + f3 + dose},
        "tx_idea": {"tasks_from": "tx_idea", "conditions": fresh + sig + ports + dose},
        "tx_moral": {"tasks_from": "tx_moral", "conditions": fresh + sig + ports + dose},
        "cd_countdown": {"tasks_from": "cd_countdown", "conditions": fresh + [_vec("ol_math_A2", "ol_math_A2"), _vec("conv_up", "conv_up"), _vec("conv_down", "conv_down")]},
        "cd_math": {"tasks_from": "sigpilot3_math", "borrow": "p1_budget:baseline,ol_math_A2,ol_random", "conditions": [_vec("conv_up", "conv_up"), _vec("conv_down", "conv_down")]},
    }
    for run, s in specs.items():
        rd = DATA / "runs" / run; rd.mkdir(parents=True, exist_ok=True)
        spec = {"run": run, **base, **s, "decision": "D-51" if run.startswith("cd_") else "D-52/D-53"}
        if (rd / "run.json").exists():
            old = json.loads((rd / "run.json").read_text())
            assert {k: old.get(k) for k in ("seeds", "max_new_tokens", "window", "tasks_from")} == {k: spec.get(k) for k in ("seeds", "max_new_tokens", "window", "tasks_from")}, f"{run}: fingerprint fields changed"
        (rd / "run.json").write_text(json.dumps(spec, indent=2))
        print(f"[specs] {run}: {len(spec['conditions'])} conditions -> {rd / 'run.json'}")


# ---------------------------------------------------------------- pre-registration (H2a)
OUTCOMES = {"math": ["correct"], "gpqa": ["correct"], "planning": ["correct"], "idea": ["originality", "feasibility", "quality"],
            "moral": ["quality"]}


def corpus_outcomes(dom: str) -> pl.DataFrame:
    """Per corpus trace: the 8 whole-trace rates (archetypes.whole_signature over Track B) and the domain's outcomes.
    Math uses the free-text extraction grades (D-47) when present; gpqa/planning the repo MCQ grader; idea and moral
    the quality judge (originality and feasibility are 1-10; quality = their mean / 10; moral = rubric score 0-1)."""
    from .archetypes import whole_signature
    from .budget import grade, load_grades
    tag = JUDGE_MODEL.replace("/", "_")
    trs, tbs, qs = [], [], []
    for run in corpus_runs(dom):
        rd = DATA / "runs" / run
        trs.append(pl.concat([pl.read_parquet(p) for p in sorted(rd.glob("traces_*.parquet"))], how="diagonal_relaxed"))
        tbs.append(pl.concat([pl.read_parquet(p) for p in sorted((rd / "judge").rglob(f"trackB_full__{tag}*.parquet"))], how="diagonal_relaxed"))
        qs += [pl.read_parquet(p) for p in sorted((rd / "judge").rglob(f"quality__{tag}*.parquet"))]
    tr = pl.concat(trs, how="diagonal_relaxed"); tb = pl.concat(tbs, how="diagonal_relaxed").unique(["trace_id", "seg_idx"], keep="last")
    sig = whole_signature(tb).filter(pl.col("n_valid") >= 4)
    df = tr.select(["trace_id", "instance_id", "seed", "answer_text", "reference_answer", "task_type", "prompt"])
    if dom in ("math", "gpqa", "planning"):
        g = load_grades()
        df = df.with_columns(pl.Series("correct", [int(g[t]) if t in g else grade(a or "", r, k, p) for t, a, r, k, p in
                                                   zip(df["trace_id"], df["answer_text"], df["reference_answer"], df["task_type"], df["prompt"])], dtype=pl.Float64))
        df = df.with_columns(pl.Series("graded_by", ["extraction" if t in g else "direct" for t in df["trace_id"]]))
    else:
        q = pl.concat(qs, how="diagonal_relaxed").unique("trace_id", keep="last")
        cols = {"quality_score": "quality"} | ({"axis_originality": "originality", "axis_feasibility": "feasibility"} if dom == "idea" else {})
        df = df.join(q.select(["trace_id"] + list(cols)).rename(cols), on="trace_id", how="left")
    return df.join(sig, on="trace_id", how="inner")


def within_task_beta(df: pl.DataFrame, y: str, X: list[str]):
    g = df.filter(pl.col(y).is_not_null())
    g = g.filter(pl.col(y).std().over("instance_id") > 0)
    d = g.with_columns([(pl.col(c) - pl.col(c).mean().over("instance_id")).alias(c) for c in X + [y]])
    Xm, ym = d.select(X).to_numpy(), d[y].to_numpy().astype(float)
    return g, np.linalg.lstsq(Xm, ym, rcond=None)[0], float(g[y].std())


def cmd_prereg(reps: int = 500):
    """H2a predictions. For each target domain and outcome: beta from the unsteered corpus (task fixed effects), then
    prediction (i) = beta . (readout-implied shift of the 8 future rates) for every vector arm. Prediction (ii) (observed
    Track B shifts) is added by `prereg_observed` before outcomes are graded."""
    R = readout(); X = [f"all_{b}" for b in BEH]
    arms = [f"sig_{d}" for d in DOMS] + ["ol_math_A2", "ol_gpqa_A4", "ol_planning_A2", "ol_random", "idea_commit", "sealdir_L17_n6p75",
                                         "rl_zero8k_T_s60", "conv_up", "conv_down"]
    implied = {a: ((load_vec(a) / R["sd"]) @ R["W"])[:8] for a in arms}
    rng = np.random.default_rng(0)
    out = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "decision": "D-52 H2a", "map": MAP,
           "implied_shift_x100": {a: dict(zip(BEH, np.round(100 * v, 3).tolist())) for a, v in implied.items()}, "domains": {}}
    for dom in DOMS:
        df = corpus_outcomes(dom)
        rec = {"n_traces": df.height, "n_tasks": df["instance_id"].n_unique()}
        if "graded_by" in df.columns:
            rec["graded_by_extraction"] = float((df["graded_by"] == "extraction").mean())
        for y in OUTCOMES[dom]:
            g, beta, ysd = within_task_beta(df, y, X)
            tasks = g["instance_id"].unique().to_list(); bs = []
            for _ in range(reps):
                pick = pl.DataFrame({"instance_id": [tasks[i] for i in rng.integers(0, len(tasks), len(tasks))]}).with_row_index("rep")
                gb = pick.join(g, on="instance_id").with_columns((pl.col("instance_id") + "#" + pl.col("rep").cast(pl.Utf8)).alias("instance_id"))
                bs.append(within_task_beta(gb, y, X)[1])
            bs = np.array(bs)
            pred = {}
            for a, v in implied.items():
                p = bs @ v
                pred[a] = {"pred": float(beta @ v), "lo": float(np.quantile(p, 0.025)), "hi": float(np.quantile(p, 0.975)),
                           "pred_sd_units": float(beta @ v / ysd) if ysd > 0 else None}
            rec[y] = {"n_tasks_varying": len(tasks), "outcome_sd": ysd, "beta": dict(zip(BEH, beta.round(4).tolist())),
                      "beta_lo": dict(zip(BEH, np.quantile(bs, 0.025, 0).round(4).tolist())), "beta_hi": dict(zip(BEH, np.quantile(bs, 0.975, 0).round(4).tolist())),
                      "pred_implied": pred}
        out["domains"][dom] = rec
        print(f"[prereg] {dom}: {rec['n_traces']} traces / {rec['n_tasks']} tasks; " +
              "; ".join(f"{y}: {rec[y]['n_tasks_varying']} varying tasks" for y in OUTCOMES[dom]))
    blob = json.dumps(out, indent=1, sort_keys=True)
    TX.mkdir(parents=True, exist_ok=True); (TX / "prereg.json").write_text(blob)
    h = hashlib.sha256(blob.encode()).hexdigest()
    (TX / "prereg.sha256").write_text(h + "\n"); print(f"[prereg] -> {TX / 'prereg.json'} sha256 {h}")
    rows = [{"target": d, "outcome": y, "arm": a, **out["domains"][d][y]["pred_implied"][a]} for d in DOMS for y in OUTCOMES[d] for a in arms]
    pl.Config.set_tbl_rows(200); print(pl.DataFrame(rows).filter(pl.col("arm").str.starts_with("sig_")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["corpus_specs", "eval_tasks", "profiles", "vectors", "conv_vectors", "specs", "prereg"])
    a = ap.parse_args()
    {"corpus_specs": cmd_corpus_specs, "eval_tasks": cmd_eval_tasks, "profiles": cmd_profiles, "vectors": cmd_vectors,
     "conv_vectors": lambda: cmd_vectors(only_conv=True), "specs": cmd_specs, "prereg": cmd_prereg}[a.cmd]()


if __name__ == "__main__":
    main()
