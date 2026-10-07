import polars as pl
from mode_control.mc.common import DATA
g = pl.read_parquet(DATA / "extract_free/grades.parquet").select(["trace_id", "success"])
R = DATA / "runs/sigpilot3_math"
t = pl.concat([pl.read_parquet(p, columns=["trace_id", "condition", "instance_id", "finish_reason"]) for p in sorted(R.glob("traces_*.parquet"))], how="diagonal_relaxed")
j = t.join(g, on="trace_id", how="left")
print(j.group_by("condition").agg(pl.len(), pl.col("success").is_null().sum().alias("ungraded"), pl.col("success").mean().round(4)).sort("condition"))
