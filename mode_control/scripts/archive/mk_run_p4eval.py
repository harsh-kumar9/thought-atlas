"""run.json for p4_eval (D-45): RL-trained bias at steps 20/40/60 on the P1 tasks and seeds (same run fingerprint as
p1_budget, so p1_budget's unsteered and A2 arms share sampling seeds and are the comparison)."""
import json
from pathlib import Path
R = Path("/ada1/u/harsh/thought-atlas/mode_control/data/runs")
spec = {"run": "p4_eval", "engine": "vllm", "tasks_from": "sigpilot3_math", "seeds": [1, 2], "max_new_tokens": 32768, "window": 8192,
        "conditions": [{"name": f"rl_zero8k_s{k}", "mode": "vector", "vector": f"rl_zero8k_s{k}", "positions": "all"} for k in (20, 40, 60)]}
d = R / "p4_eval"; d.mkdir(parents=True, exist_ok=True); p = d / "run.json"
if p.exists() and json.loads(p.read_text()) != spec:
    raise SystemExit(f"{p} exists with a different spec")
p.write_text(json.dumps(spec, indent=1)); print("wrote", p)
