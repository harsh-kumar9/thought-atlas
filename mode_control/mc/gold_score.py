"""Score the human gold set against the judge (D-65a).

Inputs: data/gold/answer_key.csv (judge labels, cell, sampling probability) and one or more filled coding sheets
data/gold/coded_<coder>.csv (the coding_sheet.csv with the eight label columns filled 0/1; blank = not coded).

Per behaviour, over items that every coder filled:
  - Cohen's kappa coder vs judge (each coder), coder vs coder (if two or more), with a 2000-draw item bootstrap CI
  - judge sensitivity and specificity against the human majority, inverse-probability weighted by each item's sampling
    probability so the estimate refers to the population of labelled segments in the four sources, and unweighted
  - the same split by source (unsteered atlas vs steered)

Usage: python -m mode_control.mc.gold_score --coders harsh,coder2
Output: data/gold/score.csv and data/gold/score_by_source.csv, printed.
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from .common import BEH, DATA

GOLD = DATA / "gold"


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pe = float((a.mean() * b.mean()) + ((1 - a.mean()) * (1 - b.mean())))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def boot(fn, *arrs, reps: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(arrs[0])
    vals = [fn(*[x[idx] for x in arrs]) for idx in (rng.integers(0, n, n) for _ in range(reps))]
    vals = np.array([v for v in vals if np.isfinite(v)])
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) if len(vals) else (float("nan"), float("nan"))


def sens_spec(judge: np.ndarray, human: np.ndarray, w: np.ndarray):
    pos, neg = human == 1, human == 0
    sens = float((w[pos] * judge[pos]).sum() / w[pos].sum()) if w[pos].sum() > 0 else float("nan")
    spec = float((w[neg] * (1 - judge[neg])).sum() / w[neg].sum()) if w[neg].sum() > 0 else float("nan")
    return sens, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coders", required=True, help="comma list; reads data/gold/coded_<coder>.csv")
    a = ap.parse_args()
    coders = a.coders.split(",")
    key = pl.read_csv(GOLD / "answer_key.csv")
    sheets = {c: pl.read_csv(GOLD / f"coded_{c}.csv", infer_schema_length=10000) for c in coders}
    rows, rows_src = [], []
    for beh in BEH:
        j = key.select(["item_id", "source", "sampling_prob", pl.col(beh).alias("judge")])
        cols = [j]
        for c in coders:
            s = sheets[c].select(["item_id", pl.col(beh).cast(pl.Utf8).alias(f"h_{c}")])
            s = s.with_columns(pl.when(pl.col(f"h_{c}").is_in(["1", "1.0", "true", "True", "yes"])).then(1)
                               .when(pl.col(f"h_{c}").is_in(["0", "0.0", "false", "False", "no"])).then(0)
                               .otherwise(None).alias(f"h_{c}"))
            cols.append(s)
        d = cols[0]
        for s in cols[1:]:
            d = d.join(s, on="item_id", how="inner")
        d = d.drop_nulls([f"h_{c}" for c in coders])
        if d.height == 0:
            print(f"[gold] {beh}: no coded items yet")
            continue
        judge = d["judge"].to_numpy().astype(int)
        hs = {c: d[f"h_{c}"].to_numpy().astype(int) for c in coders}
        human = (np.mean(np.stack(list(hs.values())), axis=0) >= 0.5).astype(int)
        w = 1.0 / np.clip(d["sampling_prob"].to_numpy(), 1e-9, None)
        rec = {"behaviour": beh, "n_items": int(d.height), "judge_pos_rate": float(judge.mean()), "human_pos_rate": float(human.mean())}
        for c in coders:
            k = kappa(hs[c], judge); lo, hi = boot(kappa, hs[c], judge)
            rec[f"kappa_{c}_vs_judge"] = k; rec[f"kappa_{c}_vs_judge_lo"] = lo; rec[f"kappa_{c}_vs_judge_hi"] = hi
        if len(coders) >= 2:
            c1, c2 = coders[:2]
            rec["kappa_coder_vs_coder"] = kappa(hs[c1], hs[c2])
        se, sp = sens_spec(judge, human, np.ones_like(w)); rec["sens_unweighted"], rec["spec_unweighted"] = se, sp
        se, sp = sens_spec(judge, human, w); rec["sens_ipw"], rec["spec_ipw"] = se, sp
        rows.append(rec)
        for src, g in d.group_by("source"):
            jg = g["judge"].to_numpy().astype(int)
            hg = (np.mean(np.stack([g[f"h_{c}"].to_numpy().astype(int) for c in coders]), axis=0) >= 0.5).astype(int)
            se, sp = sens_spec(jg, hg, np.ones(len(jg)))
            rows_src.append({"behaviour": beh, "source": src[0] if isinstance(src, tuple) else src, "n": int(len(jg)),
                             "kappa_human_vs_judge": kappa(hg, jg), "sens": se, "spec": sp})
    out = pl.DataFrame(rows); out.write_csv(GOLD / "score.csv"); print(out)
    if rows_src:
        o2 = pl.DataFrame(rows_src).sort(["behaviour", "source"]); o2.write_csv(GOLD / "score_by_source.csv"); print(o2)


if __name__ == "__main__":
    main()
