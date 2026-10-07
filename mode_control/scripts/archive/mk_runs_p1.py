import json
from pathlib import Path
R = Path("/ada1/u/harsh/thought-atlas/mode_control/data/runs")
val = {"run": "p1_sealval", "engine": "vllm", "domains": ["math"], "n_per_domain": 200, "task_seed": 20260923, "valid_only": False,
       "exclude_runs": ["corpus1_math_a", "sigpilot3_math"], "seeds": [1], "max_new_tokens": 12288, "window": 8192,
       "conditions": [{"name": "baseline", "mode": "baseline"}] +
                     [{"name": f"seal_L{L}", "mode": "vector", "vector": f"seal_L{L}", "positions": "seal"} for L in (17, 20, 23, 26)]}
d = R / "p1_sealval"; d.mkdir(parents=True, exist_ok=True)
if not (d / "run.json").exists():
    (d / "run.json").write_text(json.dumps(val, indent=1)); print("wrote", d / "run.json")
