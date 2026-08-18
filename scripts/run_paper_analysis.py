#!/usr/bin/env python3
"""Run one reproducible Thought Atlas paper-analysis stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.paper.constants import STAGES  # noqa: E402
from src.analysis.paper.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the staged, audit-gated Thought Atlas paper analysis."
    )
    parser.add_argument("--config", default="configs/paper_analysis.yaml")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_pipeline(
        args.config,
        stage=args.stage,
        dev=args.dev,
        force=args.force,
        jobs=args.jobs,
    )
    summary = {
        "stage": result.get("stage", args.stage),
        "mode": result.get("mode", "dev" if args.dev else "final"),
        "ready": result.get("ready"),
        "output_dir": str(result.get("output_dir", "")),
        "trace_index_rows": result.get("trace_index_rows"),
        "split_rows": result.get("split_rows"),
        "strict_v2_issues": result.get("upstream", {}).get("issues", []),
        "deferred_stages": result.get("deferred_stages", []),
        "report": str(result.get("report", "")),
    }
    print(json.dumps(summary, indent=2))
    return 0 if result.get("ready", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
