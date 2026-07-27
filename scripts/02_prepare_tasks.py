"""scripts/02_prepare_tasks.py — Load + stratify all configured domains -> data/tasks/*.parquet.
Runs on Ada (no GPU). Writes a setup_notes.md with strata breakdowns + caveats.
Usage: python scripts/02_prepare_tasks.py --config configs/exp.yaml [--tasks math code ...]
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import polars as pl
from omegaconf import OmegaConf
from src.utils.data_loaders import LOADERS


TASK_BUILD_VERSION = "tasks-v2"


def _validate_task(name: str, df: pl.DataFrame) -> None:
    required = {"instance_id", "task_type", "prompt", "reference_answer",
                "difficulty_raw", "metadata"}
    missing = required - set(df.columns)
    if missing or df.height == 0:
        raise ValueError(f"{name}: invalid task frame; missing={sorted(missing)}, rows={df.height}")
    if df["instance_id"].n_unique() != df.height:
        raise ValueError(f"{name}: duplicate instance_id")
    if set(df["task_type"].unique().to_list()) != {name}:
        raise ValueError(f"{name}: task_type mismatch")
    for row in df.iter_rows(named=True):
        try:
            md = json.loads(row["metadata"])
        except Exception as exc:
            raise ValueError(f"{name}/{row['instance_id']}: invalid metadata JSON") from exc
        if name in {"planning", "gpqa", "security"}:
            options = re.findall(r"(?m)^\s*([A-E])\.\s+.+$", row["prompt"])
            n_options = int(md.get("n_options", 4))
            final_labels = options[-n_options:]
            expected_labels = [chr(65 + i) for i in range(n_options)]
            duplicate_inline = (name == "planning" and len(re.findall(
                r"(?:^|\s)([A-E])\.\s+", row["prompt"])) != n_options)
            if final_labels != expected_labels or duplicate_inline:
                raise ValueError(
                    f"{name}/{row['instance_id']}: invalid final {n_options}-choice block; got {options}"
                )
            if row["reference_answer"] not in final_labels:
                raise ValueError(f"{name}/{row['instance_id']}: invalid MCQ reference")
    if name == "gpqa":
        diamond = df.filter(pl.col("metadata").str.contains('"gpqa_subset": "diamond"')).height
        if diamond != 198:
            raise ValueError(f"gpqa: expected all 198 Diamond records, found {diamond}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp.yaml")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--out-dir", default="data/tasks")
    a = ap.parse_args()
    cfg = OmegaConf.load(a.config)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    notes = [f"# Task setup notes\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
             f"Build contract: {TASK_BUILD_VERSION}\n"]
    tasks = a.tasks or list(cfg.tasks.keys())
    built = {}
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
        elif t == "security":
            kw.update(hf_id=tc.hf_id, config_name=tc.config_name,
                      revision=getattr(tc, "revision", None))
        elif t == "safety":
            kw.update(
                hf_id=tc.hf_id,
                config_name=tc.config_name,
                split=getattr(tc, "split", "train"),
                revision=getattr(tc, "revision", None),
                harm_threshold=float(getattr(tc, "harm_threshold", 0.5)),
                prompt_variant=str(getattr(tc, "prompt_variant", "direct_request")),
            )
        else:
            kw.update(hf_id=tc.hf_id)
        t0 = time.time(); df = loader(**kw); dt = time.time() - t0
        _validate_task(t, df)
        if df.height != int(tc.n_instances):
            raise ValueError(f"{t}: expected {int(tc.n_instances)} rows, loader returned {df.height}")
        built[t] = df
        sb = df.group_by("difficulty_raw").len().sort("difficulty_raw").to_dicts() if "difficulty_raw" in df.columns else []
        revision = getattr(tc, "revision", None)
        revision_note = f"- revision: {revision}\n" if revision else ""
        notes.append(
            f"## {t}\n- HF: {tc.hf_id}\n{revision_note}- N: {df.height}\n"
            f"- strata: {sb}\n- load: {dt:.1f}s\n")
        print(f"[prep] validated {t}: {df.height} rows ({dt:.1f}s)")
    manifest = {"task_build_version": TASK_BUILD_VERSION,
                "config_sha256": hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
                "seed": int(cfg.seed), "tasks": {}}
    for t, df in built.items():
        dest = out / f"{t}.parquet"; tmp = dest.with_suffix(dest.suffix + ".tmp")
        df.write_parquet(tmp); tmp.replace(dest)
        manifest["tasks"][t] = {"rows": df.height, "sha256": _sha256(dest)}
        print(f"[prep] wrote {dest}")
    mp = out / "manifest.json"; mt = mp.with_suffix(mp.suffix + ".tmp")
    mt.write_text(json.dumps(manifest, indent=2) + "\n"); mt.replace(mp)
    notes_path = out / "setup_notes.md"; nt = notes_path.with_suffix(".md.tmp")
    nt.write_text("\n".join(notes)); nt.replace(notes_path)


if __name__ == "__main__":
    main()
