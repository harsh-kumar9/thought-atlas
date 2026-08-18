"""Config-driven stage orchestration for paper analyses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from src.analysis.paper import (
    adaptive_differentiation,
    amount_shape,
    context_sensitivity,
    family_structure,
    kline_reinstatement,
    motifs,
    outcome_models,
    report,
    signature_analysis,
    trace_features,
)
from src.analysis.paper.audit import run_audit
from src.analysis.paper.io import load_trace_index
from src.analysis.paper.outcomes import build_outcome_table
from src.analysis.paper.splits import make_split_registry, write_splits


SCIENTIFIC_RUNNERS = {
    "family_structure": family_structure.run,
    "amount_shape": amount_shape.run,
    "trace_features": trace_features.run,
    "outcome_models": outcome_models.run,
    "motifs": motifs.run,
    "context_sensitivity": context_sensitivity.run,
    "adaptive_differentiation": adaptive_differentiation.run,
    "kline": kline_reinstatement.run,
    "signatures": signature_analysis.run,
    "report": report.run,
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError("paper analysis config must be a mapping")
    repo_root = path.parent.parent
    paths = config.get("paths", {})
    path_keys = [
        "experiment_config",
        "tasks_dir",
        "traces_glob",
        "answer_extractions",
        "track_b_full",
        "track_b_isolated",
        "quality",
        "output_dir",
        "dev_output_dir",
        "report_dir",
    ]
    for key in path_keys:
        value = paths.get(key)
        if value is not None and not Path(str(value)).is_absolute():
            paths[key] = str(repo_root / str(value))
    paths["grades"] = [
        str(Path(value) if Path(value).is_absolute() else repo_root / value)
        for value in paths.get("grades", [])
    ]
    config["paths"] = paths
    config["_repo_root"] = str(repo_root)
    config["_config_path"] = str(path)
    return config


def output_dir_for(config: Mapping[str, Any], *, dev: bool) -> Path:
    key = "dev_output_dir" if dev else "output_dir"
    return Path(config["paths"][key])


def _model_family_map(config: Mapping[str, Any]) -> dict[str, str]:
    return {
        model: str(values.get("family", model))
        for model, values in config.get("model_metadata", {}).items()
    }


def _write_registry_artifacts(
    *,
    output_dir: Path,
    trace_index,
    outcomes,
    splits,
    force: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        output_dir / "trace_index.parquet": trace_index,
        output_dir / "outcomes.parquet": outcomes,
        output_dir / "splits.parquet": splits,
    }
    existing = [str(path) for path in destinations if path.exists()]
    if existing and not force:
        raise FileExistsError(
            f"registry outputs already exist: {existing}; pass --force to replace them"
        )
    for path, frame in destinations.items():
        frame.write_parquet(path)


def _audit_ready(output_dir: Path) -> bool:
    manifest = output_dir / "00_audit" / "analysis_manifest.json"
    if not manifest.exists():
        return False
    payload = json.loads(manifest.read_text())
    return bool(payload.get("metadata", {}).get("paper_analysis_ready"))


def run_pipeline(
    config_path: str | Path,
    *,
    stage: str,
    dev: bool = False,
    force: bool = False,
    jobs: int = 1,
) -> dict[str, Any]:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    config = load_config(config_path)
    output_dir = output_dir_for(config, dev=dev)
    repo_root = Path(config["_repo_root"])

    if stage in {"audit", "all"}:
        dev_cfg = config.get("dev", {})
        trace_index = load_trace_index(
            config["paths"]["traces_glob"],
            model_metadata=config.get("model_metadata", {}),
            include_reasoning_text=False,
            dev=dev,
            dev_max_per_cell=int(dev_cfg.get("max_traces_per_model_domain", 200)),
        )
        outcomes = build_outcome_table(
            trace_index,
            config["paths"].get("grades", []),
            config["paths"].get("quality"),
        )
        splits = make_split_registry(
            trace_index,
            n_folds=int(config.get("resampling", {}).get("folds", 5)),
            seed=int(config.get("seed", 42)),
            model_families=_model_family_map(config),
        )
        _write_registry_artifacts(
            output_dir=output_dir,
            trace_index=trace_index,
            outcomes=outcomes,
            splits=splits,
            force=force,
        )
        result = run_audit(
            config_path=Path(config["_config_path"]),
            config=config,
            trace_index=trace_index,
            outcomes=outcomes,
            output_dir=output_dir,
            repo_root=repo_root,
            force=force,
        )
        result.update(
            {
                "stage": stage,
                "mode": "dev" if dev else "final",
                "output_dir": output_dir,
                "trace_index_rows": trace_index.height,
                "split_rows": splits.height,
                "deferred_stages": list(SCIENTIFIC_RUNNERS) if stage == "all" else [],
            }
        )
        if stage == "all" and result["ready"]:
            from src.analysis.paper.final_analysis import run_all

            return run_all(
                config=config,
                output_dir=output_dir,
                dev=dev,
                force=force,
                jobs=jobs,
            )
        return result

    if stage not in SCIENTIFIC_RUNNERS:
        raise ValueError(f"unknown stage: {stage}")
    if not _audit_ready(output_dir):
        raise RuntimeError(
            f"stage '{stage}' is blocked: {output_dir / '00_audit/analysis_manifest.json'} "
            "does not record paper_analysis_ready=true"
        )
    return SCIENTIFIC_RUNNERS[stage](
        config=config, output_dir=output_dir, dev=dev, force=force, jobs=jobs
    )
