"""scripts/02_prepare_tasks.py — Load + stratify all configured domains -> data/tasks/*.parquet.
Runs on Ada (no GPU). Writes a setup_notes.md with strata breakdowns + caveats.
Usage: python scripts/02_prepare_tasks.py --config configs/exp.yaml [--tasks math code ...]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import polars as pl
from omegaconf import OmegaConf
from src.utils.data_loaders import LOADERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp.yaml")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--out-dir", default="data/tasks")
    a = ap.parse_args()
    cfg = OmegaConf.load(a.config)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    notes = [f"# Task setup notes\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"]
    tasks = a.tasks or list(cfg.tasks.keys())
    for t in tasks:
        tc = cfg.tasks[t]; loader = LOADERS[t]
        kw = {"n": int(tc.n_instances), "seed": int(cfg.seed)}
        if t == "code":
            kw.update(hf_id=tc.hf_id, version_files=list(tc.source_files), post_date=tc.post_date)
        elif t == "moral":
            kw.update(hf_id=tc.hf_id, config_name=tc.config_name)
        elif t == "planning":
            kw.update(hf_id=tc.hf_id, configs=tuple(tc.acp_configs))
        elif t == "gpqa":
            kw.update(hf_id=tc.hf_id, primary_config=tc.primary_config, fill_config=tc.fill_config)
        else:
            kw.update(hf_id=tc.hf_id)
        try:
            t0 = time.time(); df = loader(**kw); dt = time.time() - t0
        except Exception as e:
            print(f"[prep] FAILED {t}: {type(e).__name__}: {str(e)[:200]}")
            notes.append(f"## {t}\n- HF: {tc.hf_id}\n- FAILED: {type(e).__name__}: {str(e)[:300]}\n")
            continue
        df.write_parquet(out / f"{t}.parquet")
        sb = df.group_by("difficulty_raw").len().sort("difficulty_raw").to_dicts() if "difficulty_raw" in df.columns else []
        notes.append(f"## {t}\n- HF: {tc.hf_id}\n- N: {df.height}\n- strata: {sb}\n- load: {dt:.1f}s\n")
        print(f"[prep] {t}: {df.height} rows -> {out/f'{t}.parquet'} ({dt:.1f}s)")
    Path("data/analysis").mkdir(parents=True, exist_ok=True)
    Path("data/analysis/setup_notes.md").write_text("\n".join(notes))


if __name__ == "__main__":
    main()
