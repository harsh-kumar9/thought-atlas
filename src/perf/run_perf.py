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
from src.utils.answer_extractions import load_extraction_index, select_answer
from src.utils.io import resolve_trace_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-glob", default="data/traces/traces_*.parquet")
    ap.add_argument("--extractions", default=None)
    ap.add_argument("--out", default="data/perf.parquet")
    ap.add_argument("--code-timeout", type=float, default=6.0)
    a = ap.parse_args()
    paths = resolve_trace_paths(a.traces_glob)
    if not paths:
        raise SystemExit(f"no traces matched {a.traces_glob}")
    traces = pl.concat([pl.read_parquet(p) for p in paths],
                       how="diagonal_relaxed")
    if traces.filter(pl.col("task_type") == "code").height:
        raise SystemExit("code traces require src.perf.grade_code_exec in an isolated compute job")
    extraction_index = load_extraction_index(a.extractions)
    rows = []
    for r in traces.iter_rows(named=True):
        meta = json.loads(r.get("instance_metadata") or "{}")
        selected = select_answer(r, extraction_index)
        completed = r.get("completed")
        if completed is None:
            completed = r.get("finish_reason") == "stop"
        res = assess(task_type=r["task_type"],
                     answer_text=selected["answer"] if completed else "",
                     reference_answer=r.get("reference_answer"), metadata=meta,
                     judge_fn=None, code_timeout_s=a.code_timeout)  # moral judge wired separately
        rows.append({"trace_id": r["trace_id"],
                     "extraction_used": selected["used"],
                     "extraction_selection_status": selected["selection_status"],
                     **{f"perf_{k}": v for k, v in res.items()}})
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    pl.DataFrame(rows).write_parquet(tmp); tmp.replace(out)
    print(f"[perf] {len(rows)} traces -> {a.out}")


if __name__ == "__main__":
    main()
