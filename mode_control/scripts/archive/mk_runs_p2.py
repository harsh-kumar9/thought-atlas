"""Write run.json for p2_gpqa and p2_planning (D-42 step 2)."""
import json
from pathlib import Path
R = Path("/ada1/u/harsh/thought-atlas/mode_control/data/runs")
for dom, own in (("gpqa", "ol_gpqa_A4"), ("planning", "ol_planning_A2")):
    spec = {"run": f"p2_{dom}", "engine": "vllm", "domains": [dom], "n_per_domain": 200, "task_seed": 20260923,
            "exclude_runs": [f"corpus1_{dom}_a"], "seeds": [1, 2], "max_new_tokens": 32768, "window": 8192,
            "conditions": [{"name": "baseline", "mode": "baseline"},
                           {"name": "ol_math_A2", "mode": "vector", "vector": "ol_math_A2", "positions": "all"},
                           {"name": "ol_random", "mode": "vector", "vector": "ol_random", "positions": "all"},
                           {"name": own, "mode": "vector", "vector": own, "positions": "all"}]}
    d = R / f"p2_{dom}"; d.mkdir(parents=True, exist_ok=True); p = d / "run.json"
    if p.exists() and json.loads(p.read_text()) != spec:
        raise SystemExit(f"{p} exists with a different spec")
    p.write_text(json.dumps(spec, indent=1)); print("wrote", p)
