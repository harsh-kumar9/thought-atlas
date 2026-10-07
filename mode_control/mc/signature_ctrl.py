"""Low-rank signature control subspace (D-33).

extract : teacher-force each labelled trace once; at the last token of every judge segment (within the
          judge window) store the residual stream at the chosen layers, together with the segment's
          signature targets: phi_future (behavior rates over the remaining segments), phi_next4 (over the
          next 4 segments) and phi_sofar (over segments so far). Output per run:
          data/sigctrl/<run>__acts.npz  (H [n_seg, L, d] float16; targets [n_seg, 24] float32; meta table)
fit     : reduced-rank ridge from H[:, l] to phi (future + next4, 16 dims) per layer and rank, evaluated
          leave-one-domain-out (train on the other domains, R^2 per component on the held-out one).
          Output: data/sigctrl/fit_<tag>.csv and the chosen mapping data/sigctrl/map_<tag>_L<l>_r<r>.npz
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from .common import BEH, DATA, JUDGE_MODEL, decoder_layers, load_model, load_tokenizer, render_prompt, _layer_output
from .phenotype import _segments_with_offsets

SIG = DATA / "sigctrl"
TARGET_NAMES = [f"future_{b}" for b in BEH] + [f"next4_{b}" for b in BEH] + [f"sofar_{b}" for b in BEH]


def _load_run(run: str, window: int):
    rd = DATA / "runs" / run
    tr = pl.concat([pl.read_parquet(p) for p in sorted(rd.glob("traces_*.parquet"))], how="diagonal_relaxed")
    tr = tr.filter(pl.col("condition") == "baseline")
    tag = JUDGE_MODEL.replace("/", "_")
    tb = pl.concat([pl.read_parquet(p) for p in sorted((rd / "judge").rglob(f"trackB_full__{tag}*.parquet"))], how="diagonal_relaxed").unique(["trace_id", "seg_idx"], keep="last")
    return tr, tb


def segment_targets(labels: np.ndarray) -> np.ndarray:
    """labels [n_seg, 8] in {0,1} -> targets [n_seg, 24]: future mean (excl. current), next-4 mean, so-far mean (incl. current)."""
    n = labels.shape[0]
    cs = np.vstack([np.zeros((1, labels.shape[1])), np.cumsum(labels, axis=0)])           # cs[i] = sum of first i rows
    out = np.zeros((n, 3 * labels.shape[1]), np.float32)
    for i in range(n):
        rem = n - (i + 1)
        out[i, :8] = (cs[n] - cs[i + 1]) / rem if rem > 0 else labels[i]
        k = min(4, rem)
        out[i, 8:16] = (cs[i + 1 + k] - cs[i + 1]) / k if k > 0 else labels[i]
        out[i, 16:] = cs[i + 1] / (i + 1)
    return out


def extract(runs: list[str], layers: list[int], window: int, batch: int, max_segments: int):
    SIG.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(); model = load_model(); dl = decoder_layers(model)
    for run in runs:
        out = SIG / f"{run}__acts.npz"
        if out.exists():
            print(f"[sigctrl] {run}: exists, skipping"); continue
        tr, tb = _load_run(run, window)
        n_seg = dict(zip(tb["trace_id"].to_list(), tb["n_segments"].to_list()))
        lab_by = {t: g.sort("seg_idx") for t, g in tb.group_by("trace_id")}
        lab_by = {(k[0] if isinstance(k, tuple) else k): v for k, v in lab_by.items()}
        items = []
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
            rids = tok(r["reasoning_text_for_analysis"], add_special_tokens=False)["input_ids"]
            ends = starts[1:] + [len(rids)]
            if len(valid) != len(starts) or valid.sum() < 4:
                continue
            # targets over the valid subsequence only (unparsed segments are treated as missing, not as zeros)
            vidx = np.where(valid)[0]
            tg_v = segment_targets(labels[vidx])
            pos_in_valid = {int(i): k for k, i in enumerate(vidx)}
            keep = [int(i) for i in vidx if ends[i] - 1 < window]
            if len(keep) > max_segments:
                keep = [keep[j] for j in np.linspace(0, len(keep) - 1, max_segments).round().astype(int)]
            if not keep:
                continue
            tg = tg_v[[pos_in_valid[i] for i in keep]]
            assert np.isfinite(tg).all()
            items.append({"trace_id": tid, "instance_id": r["instance_id"], "task_type": r["task_type"], "seed": r["seed"],
                          "prompt": r["prompt"], "rids": rids[:window], "seg_last": [ends[i] - 1 for i in keep], "seg_idx": keep,
                          "n_segments": len(starts), "targets": tg})
        print(f"[sigctrl] {run}: {len(items)} traces, {sum(len(x['seg_idx']) for x in items)} segment states", flush=True)
        import torch
        H_all, T_all, meta = [], [], []
        items.sort(key=lambda x: len(x["rids"]))
        store = {}
        def hook(l):
            def f(module, args, output):
                store[l] = _layer_output(output).detach()
            return f
        handles = [dl[l].register_forward_hook(hook(l)) for l in layers]
        t0 = time.time()
        try:
            for i in range(0, len(items), batch):
                chunk = items[i:i + batch]
                ids = [tok(render_prompt(tok, x["prompt"]), add_special_tokens=False)["input_ids"] + x["rids"] for x in chunk]
                plen = [len(tok(render_prompt(tok, x["prompt"]), add_special_tokens=False)["input_ids"]) for x in chunk]
                T = max(len(s) for s in ids); pad = tok.pad_token_id
                inp = torch.full((len(chunk), T), pad, dtype=torch.long); att = torch.zeros((len(chunk), T), dtype=torch.long)
                for j, s in enumerate(ids):
                    inp[j, T - len(s):] = torch.tensor(s); att[j, T - len(s):] = 1
                with torch.no_grad():
                    model(input_ids=inp.to(model.device), attention_mask=att.to(model.device))
                for j, x in enumerate(chunk):
                    off = T - len(ids[j])
                    pos = [off + plen[j] + p for p in x["seg_last"]]
                    H = torch.stack([store[l][j, pos, :] for l in layers], dim=1).float().cpu().numpy().astype(np.float16)   # [n_seg, L, d]
                    H_all.append(H); T_all.append(x["targets"])
                    for k, si in enumerate(x["seg_idx"]):
                        meta.append({"trace_id": x["trace_id"], "instance_id": x["instance_id"], "task_type": x["task_type"], "seed": x["seed"],
                                     "seg_idx": si, "n_segments": x["n_segments"], "seg_last_token": x["seg_last"][k]})
                store.clear()
                if (i // batch) % 20 == 0:
                    print(f"[sigctrl] {run}: {min(i + batch, len(items))}/{len(items)} traces, {time.time() - t0:.0f}s", flush=True)
        finally:
            for h in handles: h.remove()
        H_all = np.concatenate(H_all); T_all = np.concatenate(T_all)
        np.savez(out, H=H_all, targets=T_all, layers=np.array(layers), target_names=np.array(TARGET_NAMES))
        pl.DataFrame(meta).write_parquet(SIG / f"{run}__meta.parquet")
        print(f"[sigctrl] {run}: saved {H_all.shape} -> {out}", flush=True)


def _rrr(X, Y, rank, lam):
    """Reduced-rank ridge on standardized X (float64): ridge fit, then rank truncation of the fitted values via the
    k x k Gram of the fitted responses (Izenman). Returns W [d, k], b [k]; X must already be standardized."""
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    ym = Y.mean(0); Yc = Y - ym
    G = X.T @ X; G[np.diag_indices_from(G)] += lam
    W = np.linalg.solve(G, X.T @ Yc)
    F = X @ W
    evals, evecs = np.linalg.eigh(F.T @ F)                 # k x k, ascending
    V = evecs[:, ::-1][:, :rank]
    Wr = W @ (V @ V.T)
    return Wr.astype(np.float32), (ym - 0.0).astype(np.float32)


def _solve_ranks(G, XtY, Xtr, Ytr, Xte, Yte, lams, ranks):
    """Given the train Gram and cross-products, fit ridge for each lam, truncate to each rank, return held-out R^2 per component."""
    out = {}
    ym = Ytr.mean(0)
    for lam in lams:
        Gl = G.copy(); Gl[np.diag_indices_from(Gl)] += lam
        W = np.linalg.solve(Gl, XtY)
        F = Xtr @ W
        evals, evecs = np.linalg.eigh(F.T @ F); evecs = evecs[:, ::-1]
        for r in ranks:
            V = evecs[:, :r]; Wr = W @ (V @ V.T)
            P = Xte @ Wr + ym
            ss_res = ((Yte - P) ** 2).sum(0); ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-9
            Pc, Yc = P - P.mean(0), Yte - Yte.mean(0)                                   # level offset removed
            out[(lam, r)] = (1 - ss_res / ss_tot, 1 - ((Yc - Pc) ** 2).sum(0) / ss_tot, np.array([np.corrcoef(Pc[:, k], Yc[:, k])[0, 1] for k in range(Yc.shape[1])]))
    return out


def fit(runs: list[str], tag: str, ranks: list[int], lams: list[float], target_slice: slice = slice(0, 16), n_cv: int = 5, demean_domain: bool = False):
    Hs, Ts, metas = [], [], []
    for run in runs:
        z = np.load(SIG / f"{run}__acts.npz"); m = pl.read_parquet(SIG / f"{run}__meta.parquet")
        Hs.append(z["H"]); Ts.append(z["targets"]); metas.append(m); layers = z["layers"].tolist()
    H = np.concatenate(Hs); T = np.concatenate(Ts)[:, target_slice].astype(np.float64); meta = pl.concat(metas)
    dom = meta["task_type"].to_numpy(); doms = sorted(set(dom)); trace = meta["trace_id"].to_numpy()
    dom_means = {d: T[dom == d].mean(0) for d in doms}
    if demean_domain:
        # targets as deviations from each domain's own mean signature: the controller sets targets relative to the domain anyway
        T = T - np.stack([dom_means[d] for d in dom])
    rng = np.random.default_rng(0); utr = np.unique(trace); fold_of = dict(zip(utr, rng.integers(0, n_cv, len(utr)))); folds = np.array([fold_of[t] for t in trace])
    rows = []
    names = TARGET_NAMES[target_slice]
    for li, l in enumerate(layers):
        X = H[:, li, :].astype(np.float64)
        mu, sd = X.mean(0), X.std(0) + 1e-3; X = (X - mu) / sd
        splits = [("lodo", held, dom != held, dom == held) for held in doms] + [("cv", f"fold{k}", folds != k, folds == k) for k in range(n_cv)]
        for kind, name, trm, tem in splits:
            Xtr, Ytr = X[trm], T[trm]; Ytr_c = Ytr - Ytr.mean(0)
            G = Xtr.T @ Xtr; XtY = Xtr.T @ Ytr_c
            res = _solve_ranks(G, XtY, Xtr, Ytr, X[tem], T[tem], lams, ranks)
            for (lam, r), (r2, r2c, rho) in res.items():
                rows.append({"layer": l, "split": kind, "held_out": name, "lam": lam, "rank": r, "n_train": int(trm.sum()), "n_test": int(tem.sum()),
                             "r2_mean": float(np.mean(r2)), "r2c_mean": float(np.mean(r2c)), "rho_mean": float(np.nanmean(rho)),
                             **{f"r2_{names[k]}": float(r2[k]) for k in range(len(r2))}, **{f"r2c_{names[k]}": float(r2c[k]) for k in range(len(r2c))},
                             **{f"rho_{names[k]}": float(rho[k]) for k in range(len(rho))}})
        sub = pl.DataFrame([x for x in rows if x["layer"] == l])
        best_l = sub.group_by(["split", "lam", "rank"]).agg(pl.col("r2_mean").mean(), pl.col("r2c_mean").mean(), pl.col("rho_mean").mean()).sort("r2c_mean", descending=True)
        print(f"[sigctrl fit] layer {l}: best LODO (lam, rank, r2, r2_centered, rho) {best_l.filter(pl.col('split') == 'lodo').row(0)[1:]}, best CV {best_l.filter(pl.col('split') == 'cv').row(0)[1:]}", flush=True)
    res = pl.DataFrame(rows); res.write_csv(SIG / f"fit_{tag}.csv")
    agg = res.group_by(["split", "layer", "lam", "rank"]).agg(pl.col("r2_mean").mean(), pl.col("r2c_mean").mean(), pl.col("rho_mean").mean()).sort("r2c_mean", descending=True)
    best = agg.filter(pl.col("split") == "lodo").row(0, named=True)
    print("[sigctrl fit] best (LODO mean):", best, "| best (CV mean):", agg.filter(pl.col("split") == "cv").row(0, named=True))
    li = layers.index(best["layer"]); X = H[:, li, :].astype(np.float64); mu, sd = X.mean(0), X.std(0) + 1e-3
    W, b = _rrr((X - mu) / sd, T, int(best["rank"]), float(best["lam"]))
    np.savez(SIG / f"map_{tag}_L{best['layer']}_r{best['rank']}.npz", W=W, b=b, mu=mu.astype(np.float32), sd=sd.astype(np.float32),
             layer=best["layer"], rank=best["rank"], target_names=np.array(names), lam=float(best["lam"]), target_mean=T.mean(0), target_sd=T.std(0),
             demean_domain=demean_domain, domain_names=np.array(doms), domain_means=np.stack([dom_means[d] for d in doms]))
    print("[sigctrl fit] saved mapping for layer", best["layer"], "rank", best["rank"], "lam", best["lam"])


def mapfit(runs: list[str], tag: str, layer: int, rank: int, lam: float, demean_domain: bool, target_slice: slice = slice(0, 16)):
    """Fit the final mapping at a chosen layer/rank/lambda on all runs and save it for the controller."""
    Hs, Ts, metas = [], [], []
    for run in runs:
        z = np.load(SIG / f"{run}__acts.npz"); m = pl.read_parquet(SIG / f"{run}__meta.parquet")
        li = z["layers"].tolist().index(layer); Hs.append(z["H"][:, li, :]); Ts.append(z["targets"]); metas.append(m)
    X = np.concatenate(Hs).astype(np.float64); T = np.concatenate(Ts)[:, target_slice].astype(np.float64); meta = pl.concat(metas)
    dom = meta["task_type"].to_numpy(); doms = sorted(set(dom)); dom_means = {d: T[dom == d].mean(0) for d in doms}
    if demean_domain: T = T - np.stack([dom_means[d] for d in dom])
    mu, sd = X.mean(0), X.std(0) + 1e-3
    W, b = _rrr((X - mu) / sd, T, rank, lam)
    P = ((X - mu) / sd) @ W + b; r2 = 1 - ((T - P) ** 2).sum(0) / (((T - T.mean(0)) ** 2).sum(0) + 1e-9)
    out = SIG / f"map_{tag}_L{layer}_r{rank}.npz"
    np.savez(out, W=W, b=b, mu=mu.astype(np.float32), sd=sd.astype(np.float32), layer=layer, rank=rank, lam=lam,
             target_names=np.array(TARGET_NAMES[target_slice]), target_mean=T.mean(0), target_sd=T.std(0),
             demean_domain=demean_domain, domain_names=np.array(doms), domain_means=np.stack([dom_means[d] for d in doms]))
    print(f"[sigctrl mapfit] in-sample R^2 mean {r2.mean():.3f}; per target {np.round(r2, 3).tolist()} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--runs", required=True); e.add_argument("--layers", default="10,13,16,19,22")
    e.add_argument("--window", type=int, default=8192); e.add_argument("--batch", type=int, default=2); e.add_argument("--max-segments", type=int, default=80)
    f = sub.add_parser("fit"); f.add_argument("--runs", required=True); f.add_argument("--tag", default="v1")
    f.add_argument("--ranks", default="1,2,4,8"); f.add_argument("--lams", default="10,100,1000,10000"); f.add_argument("--demean-domain", action="store_true")
    mf = sub.add_parser("mapfit"); mf.add_argument("--runs", required=True); mf.add_argument("--tag", required=True); mf.add_argument("--layer", type=int, required=True)
    mf.add_argument("--rank", type=int, required=True); mf.add_argument("--lam", type=float, default=1e5); mf.add_argument("--demean-domain", action="store_true")
    a = ap.parse_args()
    if a.cmd == "mapfit":
        mapfit(a.runs.split(","), a.tag, a.layer, a.rank, a.lam, a.demean_domain); return
    if a.cmd == "extract":
        extract(a.runs.split(","), [int(x) for x in a.layers.split(",")], a.window, a.batch, a.max_segments)
    else:
        fit(a.runs.split(","), a.tag, [int(x) for x in a.ranks.split(",")], [float(x) for x in a.lams.split(",")], demean_domain=a.demean_domain)


if __name__ == "__main__":
    main()
