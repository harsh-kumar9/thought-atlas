"""vec_pdir_figure_x4 + matched random, and run.json for p3_steer (D-44 step 2)."""
import json
import numpy as np
from pathlib import Path
V = Path("/ada1/u/harsh/thought-atlas/mode_control/data/vsteer")
z = np.load(V / "vec_pdir_figure_x1.npz", allow_pickle=True); S = z["v"]; meta = json.loads(str(z["meta"]))
if not (V / "vec_pdir_figure_x4.npz").exists():
    np.savez(V / "vec_pdir_figure_x4.npz", v=(4 * S).astype(np.float32), layer=16, meta=json.dumps(meta | {"scale": 4}))
    u = np.random.default_rng(12).standard_normal(S.shape[0]); u /= np.linalg.norm(u)
    np.savez(V / "vec_rand_pdir_figure_x4.npz", v=(u * 4 * np.linalg.norm(S)).astype(np.float32), layer=16,
             meta=json.dumps({"source": "random", "norm": float(4 * np.linalg.norm(S))}))
R = Path("/ada1/u/harsh/thought-atlas/mode_control/data/runs")
spec = {"run": "p3_steer", "engine": "vllm", "tasks_from": "sigpilot3_math", "seeds": [1, 2, 3, 4], "max_new_tokens": 16384, "window": 8192,
        "conditions": [{"name": "baseline", "mode": "baseline"},
                       {"name": "pdir_figure_x2", "mode": "vector", "vector": "pdir_figure_x2", "positions": "all"},
                       {"name": "pdir_figure_x4", "mode": "vector", "vector": "pdir_figure_x4", "positions": "all"},
                       {"name": "rand_x4", "mode": "vector", "vector": "rand_pdir_figure_x4", "positions": "all"}]}
d = R / "p3_steer"; d.mkdir(parents=True, exist_ok=True); p = d / "run.json"
if p.exists() and json.loads(p.read_text()) != spec:
    raise SystemExit(f"{p} exists with a different spec")
p.write_text(json.dumps(spec, indent=1)); print("wrote", p)
