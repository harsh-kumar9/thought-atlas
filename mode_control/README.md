# mode_control: is there a controllable reasoning-mode state?

Extension of Thought Atlas. We ask whether the Math vs Idea difference in visible deliberation
(more verification in Math, more backtracking and perspective shift in Idea) is organized by a
low-dimensional internal state that can be read early and moved causally.

Model: Qwen3.5-9B (32 decoder layers, hidden 4096, hybrid linear-attention/full-attention).
Behaviors: the eight Thought Atlas labels, split into cognitive (verification, backtracking,
subgoal, backward chaining) and conversational (question-answer, perspective shift, conflict,
reconciliation). We score all three families: cognitive alone, conversational alone, combined.

## Research questions

- RQ1. Do early activations carry a low-dimensional state that predicts the eventual signature,
  within each domain and not just Math vs Idea?
- RQ2. Does moving that state (amplify, neutralize, reflect) change the later signature while the
  model keeps answering the original task?
- RQ3. Does the mismatch cost or help task performance?
- RQ4. Do strategy prompts act through the same state (sufficiency and necessity)?

## Pipeline

1. `mc.phenotype fit` (CPU, ada). Windowed behavior amounts from the existing Track B labels for
   the 783 valid Qwen3.5-9B Math/Idea traces; frozen ridge-logistic mode scores (cog, conv, all)
   with out-of-fold values on the paper's prompt-disjoint folds. Writes `data/phenotype/`.
2. `mc.activations` (1 GPU). Stage A: residual stream for every baseline trace under teacher
   forcing of its own first 64 tokens (last prompt token, mean of tokens 1-32, mean of 1-64) at all
   32 layers. Stage B: the same first 32 tokens teacher-forced under checking vs exploratory
   system prompts (4 paraphrase pairs) for 150 tasks per domain; stores the paired differences.
3. `mc.directions` (CPU). Per layer and phase: whitened paired-prompt direction, raw domain
   direction (control), rank-2/4 variants. Gate 1 (spontaneous domain gap on the prompt-derived
   direction) and Gate 2 (within-domain out-of-fold prediction of the frozen scores). Writes
   `data/directions/layer_phase_map.csv`.
4. `mc.steer` (1 GPU). Steered generation on a task sample with a pulse over the first 64
   generated positions at the selected layer: baseline, gamma in {2, 0, -1}, random subspace,
   raw-domain direction, strategy-prompt conditions, additive injections.
5. Judge with the repo's Track A + Track B pipeline (`src.judge.run_judge`, gemma-4-31B-it,
   guided JSON) on the new traces. `mc.analyze` then computes the frozen scores, the
   Idea-minus-Math gap per condition, attenuation, reversal, the eight raw rates, and task
   preservation (completion, answer present, math correctness, loop flags, length).

## Running

All from the repo root on ada.

```bash
python -m mode_control.mc.phenotype fit --window 4096
python -m mode_control.mc.power --window 4096
sbatch -w mira mode_control/scripts/mode.sbatch smoke
sbatch -w mira mode_control/scripts/mode.sbatch acts
python -m mode_control.mc.directions --window 4096
sbatch -w mira mode_control/scripts/mode.sbatch steer pilot1
sbatch -w mira mode_control/scripts/mode.sbatch judge pilot1
python -m mode_control.mc.analyze --run pilot1
```

`data/runs/<run>/run.json` defines the task sample, window, seeds, and the condition list.

## Status

See DECISIONS.md for the judgment calls that need ratification and RESULTS.md for findings.
