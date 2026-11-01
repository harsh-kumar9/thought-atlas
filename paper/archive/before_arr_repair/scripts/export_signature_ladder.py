"""Reproduce frozen domain fits and export cumulative gains with joint resampling.

No generations, new splits, hyperparameter changes, or new significance tests.
Run from the repository root with PYTHONPATH=. and the repository Python.
"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

from src.analysis.paper.signature_analysis import (
    MODELS, FAMILIES, add_timing_contrasts, _cv_probabilities,
    _stable_seed, _log_loss, _balanced_accuracy,
)

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'paper'
S = ROOT / 'docs/paper_synthesis_2026-10-31'
inputs = {
    'features': S / 'reframing/sources/trace_features.parquet',
    'splits': S / 'sources/ada_corrected_atlas/splits.parquet',
    'reference': P / 'data/figure_inputs/gains.csv',
}
data = pd.read_parquet(inputs['features'])
data = data[data.gen_model.isin(MODELS) & (data.n_valid > 0)].copy()
splits = pd.read_parquet(inputs['splits'], columns=['trace_id', 'instance_id', 'prompt_fold'])
assert splits.groupby('instance_id').prompt_fold.nunique().eq(1).all()
data = data.merge(splits[['trace_id', 'prompt_fold']], on='trace_id', validate='one_to_one')
assert len(data) == 21803 and data.prompt_fold.notna().all()
data = add_timing_contrasts(data)
ref = pd.read_csv(inputs['reference']).set_index(['gen_model', 'signature_family'])
rows, checks = [], []
reps, seed = 1000, 42
for model in MODELS:
    part = data[data.gen_model.eq(model)].reset_index(drop=True)
    for family, behaviors in FAMILIES.items():
        amount = [f'{b}__rate' for b in behaviors]
        timing = [f'{b}__timing_{c}' for b in behaviors for c in ['middle_minus_early', 'late_minus_early']]
        y, pred, labels = _cv_probabilities(part, 'task_type', {
            'length': ['log_tokens', 'log_segments'],
            'amount': ['log_tokens', 'log_segments', *amount],
            'full': ['log_tokens', 'log_segments', *amount, *timing],
        }, folds=5, seed=_stable_seed(f'domain|{model}|{family}', seed),
            fold_ids=part.prompt_fold.to_numpy(int))
        source = ref.loc[(model, family)]
        assert len(y) == source['n'] and len(labels) == source['domains'] == 8
        for stage in ['length', 'amount', 'full']:
            loss = _log_loss(y, pred[stage])
            error = abs(loss - source[f'{stage}_log_loss'])
            # Cross-platform optimizer/BLAS differences are allowed only below
            # 0.00001 nat per trace; classification decisions must still match.
            assert error < 1e-5, (model, family, stage, error)
            assert abs(_balanced_accuracy(y, pred[stage]) - source[f'{stage}_balanced_accuracy']) < 1e-12
            checks.append({'model': model, 'family': family, 'stage': stage, 'log_loss_abs_error': error})
        # The uniform eight-domain predictive distribution has exactly 3-bit loss.
        # Resample all stages jointly; never add marginal interval endpoints.
        per_trace = np.column_stack([
            math.log2(len(labels)) + np.log2(np.clip(pred[k][np.arange(len(y)), y], 1e-12, 1))
            for k in ['length', 'amount', 'full']
        ])
        rng = np.random.default_rng(_stable_seed(f'ladder|{model}|{family}', seed))
        strata = [np.flatnonzero(y == cls) for cls in np.unique(y)]
        boot = np.empty((reps, 3))
        for b in range(reps):
            sampled = np.concatenate([idx[rng.integers(0, len(idx), len(idx))] for idx in strata])
            boot[b] = per_trace[sampled].mean(axis=0)
        for j, stage in enumerate(['length', 'amount', 'full']):
            rows.append(dict(gen_model=model, signature_family=family, stage=stage,
                n=len(y), domains=len(labels), gain_bits=per_trace[:, j].mean(),
                ci_low=np.quantile(boot[:, j], .025), ci_high=np.quantile(boot[:, j], .975)))
        print(f'Reproduced {model} / {family}', flush=True)
out = P / 'data/signature_ladder.csv'
pd.DataFrame(rows).to_csv(out, index=False)
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = dict(
    inputs=[dict(file=str(f.relative_to(ROOT)), sha256=digest(f)) for f in inputs.values()],
    output=dict(file=str(out.relative_to(P)), sha256=digest(out)),
    script_sha256=digest(Path(__file__)), seed=seed, bootstrap_reps=reps,
    baseline='Uniform probability 1/8 for each domain; log loss = 3 bits per trace.',
    interval='Pointwise 95% domain-stratified prompt bootstrap, fixed out-of-fold predictions, stages resampled jointly.',
    interpretation='Cumulative held-out log-loss reduction; adjacent differences reproduce stored amount and timing gains. Length-only includes fitted intercepts/class-frequency information.',
    original_fit_checks=checks, log_loss_tolerance_nats=1e-5,
)
(P / 'data/signature_ladder_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Exported {len(rows)} cumulative estimates; all 42 original fit checks passed.')
