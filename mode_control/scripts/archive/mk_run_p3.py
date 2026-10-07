"""run.json for p3_strat (D-44 step 1): generic strategy prompts (Polya) vs unsteered on the 199 held-out Math tasks."""
import json
from pathlib import Path
R = Path("/ada1/u/harsh/thought-atlas/mode_control/data/runs")
P = {"cases": "Before solving it in general, try small cases or specific values, look for a pattern, and then confirm it.",
     "backward": "Work backwards: start from what the question asks for and reason back to what is given.",
     "figure": "Represent the problem as a picture (a diagram, a number line, or coordinates) and reason from that picture."}
spec = {"run": "p3_strat", "engine": "vllm", "tasks_from": "sigpilot3_math", "seeds": [1, 2, 3, 4], "max_new_tokens": 16384, "window": 8192,
        "conditions": [{"name": "baseline", "mode": "baseline"}] + [{"name": f"prompt_{k}", "mode": "baseline", "system": v} for k, v in P.items()]}
d = R / "p3_strat"; d.mkdir(parents=True, exist_ok=True); p = d / "run.json"
if p.exists() and json.loads(p.read_text()) != spec:
    raise SystemExit(f"{p} exists with a different spec")
p.write_text(json.dumps(spec, indent=1)); print("wrote", p)
