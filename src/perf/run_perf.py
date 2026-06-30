"""src/perf/run_perf.py — apply per-task performance metrics to generated traces.
math/gpqa/planning: pure-CPU. code: sandboxed subprocess (run inside a compute job, NOT login).
moral/idea: needs a judge callable or the separate run_quality.py path.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import polars as pl
from src.perf.metrics import assess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-glob", default="data/traces/traces_*.parquet")
    ap.add_argument("--out", default="data/perf.parquet")
    ap.add_argument("--code-timeout", type=float, default=6.0)
    a = ap.parse_args()
    traces = pl.concat([pl.read_parquet(p) for p in sorted(Path().glob(a.traces_glob))],
                       how="diagonal_relaxed")
    rows = []
    for r in traces.iter_rows(named=True):
        meta = json.loads(r.get("instance_metadata") or "{}")
        res = assess(task_type=r["task_type"], answer_text=r.get("answer_text") or "",
                     reference_answer=r.get("reference_answer"), metadata=meta,
                     judge_fn=None, code_timeout_s=a.code_timeout)  # moral judge wired separately
        rows.append({"trace_id": r["trace_id"], **{f"perf_{k}": v for k, v in res.items()}})
    pl.DataFrame(rows).write_parquet(a.out)
    print(f"[perf] {len(rows)} traces -> {a.out}")


if __name__ == "__main__":
    main()
