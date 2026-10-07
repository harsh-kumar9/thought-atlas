"""Is the within-domain early readout just difficulty or prompt length? Partial correlations + opening boilerplate."""
import numpy as np, polars as pl
from scipy.stats import spearmanr, rankdata
from .common import DATA, REPO, load_tokenizer

def resid(y, X):
    X1 = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(X1, y, rcond=None)[0]; return y - X1 @ b

def main():
    A = np.load(DATA / "acts/stageA_k64.npz", allow_pickle=True); d = np.load(DATA / "directions/directions_w4096_v2.npz")
    ph = pl.read_parquet(DATA / "phenotype/baseline_w4096.parquet")
    tr = pl.read_parquet(REPO / "data/v2/traces/traces_qwen35_9b.parquet", columns=["trace_id", "prompt", "difficulty_raw", "full_text"])
    ph = ph.join(tr, on="trace_id")
    idx = {t: i for i, t in enumerate(A["trace_id"].tolist())}
    ph = ph.filter(pl.col("trace_id").is_in(list(idx)))
    order = np.array([idx[t] for t in ph["trace_id"].to_list()])
    tok = load_tokenizer()
    plen = np.array([len(tok(p, add_special_tokens=False)["input_ids"]) for p in ph["prompt"].to_list()])
    for l, phase, pi in [(19, "mean32", 1), (19, "last_prompt", 0), (16, "mean32", 1)]:
        u = d[f"prompt_r1|{l}|{phase}"][:, 0]; c = d[f"center|{l}|{phase}"]
        z = (A["acts"][order, l, pi, :].astype(np.float32) - c) @ u
        for dom in ["math", "idea"]:
            m = (ph["task_type"] == dom).to_numpy()
            s = ph["s_cog"].to_numpy()[m]; zz = z[m]; L = np.log(plen[m]); nt = np.log(ph["n_new_tokens"].to_numpy()[m])
            lev = ph["difficulty_raw"].to_numpy()[m].astype(float) if dom == "math" else None
            covs = np.column_stack([L] + ([lev] if lev is not None else []))
            pr = np.corrcoef(resid(rankdata(zz), covs), resid(rankdata(s), covs))[0, 1]
            extra = f" rho(z,level)={spearmanr(zz, lev).correlation:+.2f} rho(s_cog,level)={spearmanr(s, lev).correlation:+.2f}" if lev is not None else ""
            print(f"L{l} {phase:11s} {dom}: rho(z,s_cog)={spearmanr(zz, s).correlation:+.2f} rho(z,logPromptLen)={spearmanr(zz, L).correlation:+.2f} "
                  f"rho(z,logTokens)={spearmanr(zz, nt).correlation:+.2f} partial(z,s_cog|len,level)={pr:+.2f} z_sd={zz.std():.2f}{extra}")
    first = np.array([tok.decode(tok(t, add_special_tokens=False)["input_ids"][:12]) for t in ph["full_text"].to_list()])
    for dom in ["math", "idea"]:
        m = (ph["task_type"] == dom).to_numpy(); vals, cnt = np.unique(first[m], return_counts=True)
        top = np.argsort(-cnt)[:3]; print(dom, "distinct 12-token openings:", len(vals), "of", int(m.sum()), "top:", [(vals[i][:50].replace("\n", " "), int(cnt[i])) for i in top])

if __name__ == "__main__":
    main()
