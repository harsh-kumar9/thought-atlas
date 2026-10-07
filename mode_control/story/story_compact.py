"""story_data.json (from story_data.py on ada) -> story_compact3.json, the DATA blob of the story page.
Values not in story_data.json are pasted from mode_control/RESULTS.md (D-47 re-graded section) and cited inline."""
import json

d = json.load(open("story_data.json"))
B = [4096, 8192, 12288, 16384, 24576, 32768]
BEH_ORDER = ["verification", "subgoal", "Question_and_Answering", "Perspective_Shift", "Conflict_of_Perspectives", "backtracking", "Reconciliation", "backward_chaining"]
LAB = {"verification": "Verification", "subgoal": "Subgoal setting", "Question_and_Answering": "Self-questioning (Q&A)", "Perspective_Shift": "Perspective shift",
       "Conflict_of_Perspectives": "Conflict", "backtracking": "Backtracking", "Reconciliation": "Reconciliation", "backward_chaining": "Backward chaining"}
out = {"behaviours": [LAB[b] for b in BEH_ORDER]}
out["sig"] = {dom: [round(d["signatures"][k][b] * 100, 1) for b in BEH_ORDER] for dom, k in [("Math", "math"), ("GPQA", "gpqa"), ("Planning", "planning"), ("Idea", "idea")]}
out["sig_tokens"] = {k: round(v["tokens"]) for k, v in d["signatures"].items()}
p = {(r["task_type"], r["condition"]): r["s_all"] for r in d["power1"]}
out["knob"] = {"math_base": p[("math", "baseline")], "idea_base": p[("idea", "baseline")], "idea_check_prompt": p[("idea", "prompt_checking")],
               "idea_explore_prompt": p[("idea", "prompt_exploratory")], "idea_g1": p[("idea", "reflect1_sus_L19")], "idea_g3": p[("idea", "reflect3_sus_L19")],
               "idea_random": p[("idea", "band_random_sus")], "math_ideaward": p[("math", "band_add_plus_half_sus")], "math_random": p[("math", "band_random_sus")],
               "math_g3": p[("math", "reflect3_sus_L19")]}


def curves(rows, conds):
    res = {}
    for c in conds:
        rr = {r["budget"]: r for r in rows if r["condition"] == c}
        res[c] = {"correct": [round(rr[b]["correct"], 4) for b in B], "answered": [round(rr[b]["answered"], 4) for b in B], "tokens": [round(rr[b]["tokens"]) for b in B], "n": rr[B[-1]]["n"]}
    return res


out["budgets"] = B
out["math"] = curves(d["budget_math"], ["baseline", "ol_math_A2", "seal", "sealdir_matched", "concise", "ol_random", "rl_zeroT_s60", "rl_a2initT_s60"])
out["gpqa"] = curves(d["budget_gpqa"], ["baseline", "ol_math_A2", "ol_gpqa_A4", "ol_random"])
out["planning"] = curves(d["budget_planning"], ["baseline", "ol_math_A2", "ol_planning_A2", "ol_random"])
out["loops"] = {k: round(v, 4) for k, v in d["loops"].items()}


def pr(rows, ref, cond, b):
    for r in rows:
        if r["reference"] == ref and r["condition"] == cond and r["budget"] == b:
            return [round(r["d_correct"], 3), round(r["d_correct_lo"], 3), round(r["d_correct_hi"], 3)]


out["paired"] = {"math_A2_12k": pr(d["paired_math"], "baseline", "ol_math_A2", 12288), "math_A2_32k": pr(d["paired_math"], "baseline", "ol_math_A2", 32768),
                 "math_seal_12k": pr(d["paired_math"], "baseline", "seal", 12288), "math_sealdir_12k": pr(d["paired_math"], "baseline", "sealdir_matched", 12288),
                 "math_concise_12k": pr(d["paired_math"], "baseline", "concise", 12288),
                 "gpqa_A2_12k": pr(d["paired_gpqa"], "baseline", "ol_math_A2", 12288), "gpqa_own_12k": pr(d["paired_gpqa"], "baseline", "ol_gpqa_A4", 12288),
                 "gpqa_A2_32k": pr(d["paired_gpqa"], "baseline", "ol_math_A2", 32768), "gpqa_own_32k": pr(d["paired_gpqa"], "baseline", "ol_gpqa_A4", 32768),
                 "plan_A2_12k": pr(d["paired_planning"], "baseline", "ol_math_A2", 12288), "plan_own_12k": pr(d["paired_planning"], "baseline", "ol_planning_A2", 12288)}
SH = ["subgoal", "Question_and_Answering", "verification", "Conflict_of_Perspectives", "Reconciliation", "Perspective_Shift", "backtracking"]
out["shift_behaviours"] = [LAB[b] for b in SH]
out["shift"] = {k: {"n": v["n"], "v": [[round(x * 100, 2) for x in v["shift"][b]] for b in SH]} for k, v in d["shifts"].items()}
out["base_rates"] = {k: [round(v[b] * 100, 1) for b in SH] for k, v in d["base_rates"].items()}
out["cos"] = {"names": d["cos"]["names"], "residual": [[round(x, 3) for x in r] for r in d["cos"]["residual"]],
              "behaviour": [[round(x, 3) for x in r] for r in d["cos"]["behaviour"]], "norms": {k: round(v, 2) for k, v in d["cos"]["norms"].items()}}
out["rl"] = {k: [[x["step"], round(x["reward"], 4), round(x["tokens"]), round(x["answered"], 4)] for x in v] for k, v in d["rl"].items()}
mix = {(r["condition"], r["method_category"]): r["share"] for r in d["p3_mix"]}
out["p3"] = {"off": {r["condition"]: round(r["off"], 4) for r in d["p3_offmode"]}, "off_steer": {r["condition"]: round(r["off"], 4) for r in d["p3s_offmode"]},
             "geo": {c: round(mix.get((c, "geometric_or_coordinate_argument"), 0), 4) for c in ["baseline", "prompt_figure", "prompt_cases", "prompt_backward"]},
             "pass1": {r["condition"]: round(r["correct"], 4) for r in d["p3_budget"] if r["budget"] == 16384},
             "pass1_steer": {r["condition"]: round(r["correct"], 4) for r in d["p3s_budget"] if r["budget"] == 16384},
             "loops": {k: round(v, 4) for k, v in d["loops3"].items()}}
# p3_strat/analysis/complementarity_math.csv, K = 4 (re-graded 2026-09-28)
out["p3"]["pools"] = {"unprompted": 0.8442, "figure": 0.8342, "mixed_figure": 0.8608, "mixed_figure_ci": [-0.0057, 0.0409], "backward": 0.8794, "mixed_backward": 0.8763}
# sigpilot3_math (D-36 replication, HF, 12k, 199 tasks, one sample), re-graded with extract_free/grades.parquet
out["ctrl"] = [["Unsteered", 0.5729, "base"], ["Random direction, same size", 0.4573, "random"], ["Knob pushed to check", 0.6281, "knob"], ["A2 target", 0.8141, "a2"]]
# idea judge scores (not affected by the math re-grade); D-33/D-34 in RESULTS.md
out["trade"] = [
    {"name": "Knob toward exploring", "x": [0.30, 0.17, 0.43], "y": [-0.28, -0.57, 0.00], "c": "knob"},
    {"name": "Knob toward checking", "x": [-0.30, -0.44, -0.18], "y": [0.12, -0.08, 0.32], "c": "knob"},
    {"name": "Explore first, then check", "x": [0.09, -0.03, 0.21], "y": [-0.22, -0.41, -0.03], "c": "knob"},
    {"name": "Random schedule", "x": [-0.02, -0.14, 0.09], "y": [-0.21, -0.41, -0.01], "c": "random"},
    {"name": "Math A2 vector", "x": [-0.25, -0.42, -0.09], "y": [-0.35, -0.62, -0.07], "c": "a2"}]
json.dump(out, open("story_compact3.json", "w"), separators=(",", ":"))

# numbers quoted in the page text
M, G, P = out["math"], out["gpqa"], out["planning"]
for k, v in M.items():
    print(f"math {k:16s} correct {[round(x * 100) for x in v['correct']]} answered32k {v['answered'][5]:.3f} right|answered {v['correct'][5] / v['answered'][5]:.3f} tok32k {v['tokens'][5]}")
for nm, T in (("gpqa", G), ("planning", P)):
    for k, v in T.items():
        print(f"{nm} {k:16s} correct {[round(x * 100) for x in v['correct']]} tok12k {v['tokens'][2]} tok32k {v['tokens'][5]}")
print("paired", out["paired"])
print("loops", {k: v for k, v in out["loops"].items() if k.startswith(("p1", "p4"))})
print("p3", {k: out["p3"][k] for k in ("off", "pass1", "pass1_steer", "off_steer")})
print("shift rl_zero", dict(zip(out["shift_behaviours"], [x[0] for x in out["shift"]["math/rl_zero"]["v"]])))
print("shift A2", dict(zip(out["shift_behaviours"], [x[0] for x in out["shift"]["math/A2"]["v"]])))
print("cos beh A2 vs RL0", out["cos"]["behaviour"][0][2], "res", out["cos"]["residual"][0][2], "norms", out["cos"]["norms"])
