"""Write data/runs/p1_budget/run.json (D-41). Usage: python3 mk_run_p1_budget.py <seal_layer e.g. 23>"""
import json, sys
from pathlib import Path
L = int(sys.argv[1])
R = Path("/ada1/u/harsh/thought-atlas/mode_control/data/runs")
spec = {"run": "p1_budget", "engine": "vllm", "tasks_from": "sigpilot3_math", "seeds": [1, 2], "max_new_tokens": 32768, "window": 8192,
        "conditions": [
            {"name": "baseline", "mode": "baseline"},
            {"name": "ol_math_A2", "mode": "vector", "vector": "ol_math_A2", "positions": "all"},
            {"name": "ol_random", "mode": "vector", "vector": "ol_random", "positions": "all"},
            {"name": "seal", "mode": "vector", "vector": f"seal_L{L}", "positions": "seal"},
            {"name": "sealdir_matched", "mode": "vector", "vector": "sealdir_L17_n6p75", "positions": "all"},
            {"name": "concise", "mode": "baseline", "system": "Be concise."}]}
d = R / "p1_budget"; d.mkdir(parents=True, exist_ok=True)
p = d / "run.json"
if p.exists() and json.loads(p.read_text()) != spec:
    sys.exit(f"{p} exists with a different spec; not overwriting")
p.write_text(json.dumps(spec, indent=1)); print("wrote", p)
