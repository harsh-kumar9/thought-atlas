"""run.json for p4_eval2 (D-45, fixed gradient): RL-refined A2 (a2init8k_T step 60) on the P1 tasks and seeds (same run
fingerprint as p1_budget, so its unsteered and A2 arms are the comparison). The zero-start arm is added when it finishes."""
import json
from pathlib import Path
R = Path("/ada1/u/harsh/thought-atlas/mode_control/data/runs")
spec = {"run": "p4_eval2", "engine": "vllm", "tasks_from": "sigpilot3_math", "seeds": [1, 2], "max_new_tokens": 32768, "window": 8192,
        "conditions": [{"name": "rl_a2initT_s60", "mode": "vector", "vector": "rl_a2init8k_T_s60", "positions": "all"}]}
d = R / "p4_eval2"; d.mkdir(parents=True, exist_ok=True); p = d / "run.json"
if p.exists() and json.loads(p.read_text()) != spec:
    raise SystemExit(f"{p} exists with a different spec")
p.write_text(json.dumps(spec, indent=1)); print("wrote", p)
