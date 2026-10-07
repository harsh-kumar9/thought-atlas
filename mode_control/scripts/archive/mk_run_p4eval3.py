"""run.json for p4_eval3 (D-45, fixed gradient): zero-start RL bias (zero8k_T) at steps 30 and 60 on the P1 tasks and
seeds (same fingerprint as p1_budget)."""
import json
from pathlib import Path
R = Path("/ada1/u/harsh/thought-atlas/mode_control/data/runs")
spec = {"run": "p4_eval3", "engine": "vllm", "tasks_from": "sigpilot3_math", "seeds": [1, 2], "max_new_tokens": 32768, "window": 8192,
        "conditions": [{"name": f"rl_zeroT_s{k}", "mode": "vector", "vector": f"rl_zero8k_T_s{k}", "positions": "all"} for k in (30, 60)]}
d = R / "p4_eval3"; d.mkdir(parents=True, exist_ok=True); p = d / "run.json"
if p.exists() and json.loads(p.read_text()) != spec:
    raise SystemExit(f"{p} exists with a different spec")
p.write_text(json.dumps(spec, indent=1)); print("wrote", p)
