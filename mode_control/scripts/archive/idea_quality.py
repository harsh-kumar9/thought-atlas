import polars as pl, glob, numpy as np, sys
rd = f"mode_control/data/runs/{sys.argv[1]}/"
q = pl.read_parquet(glob.glob(rd + "judge/quality__*.parquet")[0]).select(["trace_id", "parsed", "axis_originality", "axis_feasibility"])
tr = pl.concat([pl.read_parquet(p, columns=["trace_id", "condition", "instance_id"]) for p in glob.glob(rd + "traces_*.parquet")], how="diagonal_relaxed")
j = q.join(tr, on="trace_id")
base = j.filter(pl.col("condition") == "baseline").select(["instance_id", pl.col("axis_originality").alias("o0"), pl.col("axis_feasibility").alias("f0")])
rng = np.random.default_rng(0)
print("condition                answered  originality (paired)             feasibility (paired)")
for c in sorted(j["condition"].unique().to_list()):
    g = j.filter(pl.col("condition") == c); ans = float(g["parsed"].mean())
    if c == "baseline":
        print("%-22s %.2f   orig %.2f  feas %.2f" % (c, ans, g["axis_originality"].mean(), g["axis_feasibility"].mean())); continue
    gg = g.join(base, on="instance_id"); out = []
    for m, b in [("axis_originality", "o0"), ("axis_feasibility", "f0")]:
        x = (gg[m].cast(pl.Float64) - gg[b].cast(pl.Float64)).drop_nulls().to_numpy()
        if len(x) < 5: out.append("n<5"); continue
        bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        out.append("%+.2f [%+.2f,%+.2f] n=%d" % (x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(x)))
    print("%-22s %.2f   %-32s %s" % (c, ans, out[0], out[1]))
