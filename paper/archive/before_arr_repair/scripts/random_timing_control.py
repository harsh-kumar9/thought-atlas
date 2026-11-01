"""Dimension-matched negative control for the domain timing increment.

Shuffle the complete eight-column timing block across rows, independently in
each training and held-out partition. Keep length/amount columns, labels,
frozen prompt folds, scaling, classifier and ridge penalty fixed. This is a
negative-control sensitivity analysis, not an exact conditional permutation test.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import scipy

from src.analysis.paper.signature_analysis import (
    MODELS, FAMILIES, add_timing_contrasts, _cv_probabilities, _design,
    _stable_seed, _log_loss, ridge_multinomial_predict,
)

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / 'docs/paper_synthesis_2026-10-31/reframing/sources/trace_features.parquet'
SPLITS = ROOT / 'docs/paper_synthesis_2026-10-31/sources/ada_corrected_atlas/splits.parquet'
REFERENCE = ROOT / 'paper/data/figure_inputs/gains.csv'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shuffled_design(train, test, n_base, seed):
    """Keep each partition's timing rows intact, but unlink them from prompts."""
    rng = np.random.default_rng(seed)
    tr, te = train.copy(), test.copy()
    tr[:, n_base:] = train[rng.permutation(len(train)), n_base:]
    te[:, n_base:] = test[rng.permutation(len(test)), n_base:]
    return tr, te


def run_cell(model, family, out, repeats, seed, identity):
    out = Path(out)
    path = out / f'{model}__{family}.npz'
    if path.exists():
        with np.load(path) as z:
            assert str(z['identity']) == identity, 'Stale checkpoint: ' + str(path)
            assert z['random_gain'].shape[0] == repeats
        return dict(model=model, family=family, resumed=True)
    data = pd.read_parquet(FEATURES)
    data = data[data.gen_model.eq(model) & (data.n_valid > 0)].copy()
    assert not data.instance_id.duplicated().any()
    n = len(data)
    splits = pd.read_parquet(SPLITS, columns=['trace_id', 'instance_id', 'prompt_fold'])
    assert splits.groupby('instance_id').prompt_fold.nunique().eq(1).all()
    data = data.merge(splits[['trace_id', 'prompt_fold']], on='trace_id', validate='one_to_one')
    assert len(data) == n and data.prompt_fold.notna().all()
    data = add_timing_contrasts(data).reset_index(drop=True)
    behaviors = FAMILIES[family]
    base = ['log_tokens', 'log_segments', *[f'{b}__rate' for b in behaviors]]
    timing = [f'{b}__timing_{c}' for b in behaviors for c in ['middle_minus_early', 'late_minus_early']]
    assert len(timing) == 8
    fold = data.prompt_fold.to_numpy(int)
    y, pred, labels = _cv_probabilities(data, 'task_type',
        {'amount': base, 'full': base + timing}, folds=5,
        seed=_stable_seed(f'domain|{model}|{family}', seed), fold_ids=fold)
    ref = pd.read_csv(REFERENCE).set_index(['gen_model', 'signature_family']).loc[(model, family)]
    assert n == ref['n'] and len(labels) == 8
    errors = {}
    for stage in ['amount', 'full']:
        errors[stage] = abs(_log_loss(y, pred[stage]) - ref[f'{stage}_log_loss'])
        assert errors[stage] < 1e-5, (model, family, stage, errors[stage])
    rows = np.arange(n)
    base_logp = np.log2(np.clip(pred['amount'][rows, y], 1e-12, 1))
    real_gain = np.log2(np.clip(pred['full'][rows, y], 1e-12, 1)) - base_logp
    random_gain = np.empty((repeats, n))
    # Scaling is fitted to original training rows only. Permuting the complete
    # timing block preserves its training means/scales and within-block covariance.
    designs = []
    for f in sorted(np.unique(fold)):
        train, test = fold != f, fold == f
        tr, te = _design(data[train], data[test], base + timing)
        designs.append((int(f), train, test, tr, te))
    for rep in range(repeats):
        for f, train, test, tr, te in designs:
            trr, ter = shuffled_design(tr, te, 1 + len(base),
                _stable_seed(f'random-timing|{model}|{family}|{rep}|{f}', seed))
            probability = ridge_multinomial_predict(trr, y[train], ter, 8, l2=1.0)
            random_gain[rep, test] = np.log2(np.clip(probability[np.arange(test.sum()), y[test]], 1e-12, 1)) - base_logp[test]
        if (rep + 1) % 20 == 0:
            print(f'{model}/{family}: {rep + 1}/{repeats}', flush=True)
    temp = out / (path.name + '.tmp')
    with temp.open('wb') as handle:
        np.savez_compressed(handle, identity=identity, y=y, real_gain=real_gain,
            random_gain=random_gain, base_gain=3 + base_logp,
            trace_id=data.trace_id.astype(str).to_numpy(dtype=str),
            fit_error_amount=errors['amount'], fit_error_full=errors['full'])
    temp.replace(path)
    return dict(model=model, family=family, resumed=False,
        real_gain=float(real_gain.mean()), random_gain=float(random_gain.mean()),
        max_fit_error=max(errors.values()))


def summarize(out, repeats, boot_reps, seed):
    out = Path(out)
    summaries, seeds = [], []
    for model in MODELS:
        for family in FAMILIES:
            with np.load(out / f'{model}__{family}.npz') as z:
                y, real, random, base = z['y'], z['real_gain'], z['random_gain'], z['base_gain']
            assert random.shape == (repeats, len(y))
            per_seed = random.mean(axis=1)
            for i, value in enumerate(per_seed):
                seeds.append(dict(gen_model=model, signature_family=family, replicate=i, gain_bits=value))
            # Prompt uncertainty conditional on the fitted predictors, plus finite
            # Monte Carlo uncertainty in the average randomized-control score.
            rng = np.random.default_rng(_stable_seed(f'random-timing-bootstrap|{model}|{family}', seed))
            strata = [np.flatnonzero(y == cls) for cls in np.unique(y)]
            boot = np.empty((boot_reps, 4))
            for i in range(boot_reps):
                sampled = np.concatenate([idx[rng.integers(0, len(idx), len(idx))] for idx in strata])
                rep_weights = rng.multinomial(repeats, np.full(repeats, 1 / repeats)) / repeats
                mean_random = rep_weights @ random
                rg = float(real[sampled].mean()); ng = float(mean_random[sampled].mean())
                boot[i] = [rg, ng, rg - ng, float(base[sampled].mean()) + ng]
            result = dict(gen_model=model, signature_family=family, n=len(y), randomizations=repeats,
                timing_gain_bits=real.mean(), random_gain_bits=random.mean(),
                random_seed_p025=np.quantile(per_seed, .025), random_seed_p975=np.quantile(per_seed, .975),
                random_seed_min=per_seed.min(), random_seed_max=per_seed.max(),
                randomizations_at_least_timing=int((per_seed >= real.mean()).sum()),
                timing_minus_random_bits=real.mean() - random.mean(),
                random_cumulative_bits=base.mean() + random.mean())
            for j, label in enumerate(['timing', 'random_mean', 'timing_minus_random', 'random_cumulative']):
                result[label + '_ci_low'] = np.quantile(boot[:, j], .025)
                result[label + '_ci_high'] = np.quantile(boot[:, j], .975)
            # Bonferroni intervals for the 14 paired timing-minus-control contrasts
            # use the same bootstrap distribution. No exact permutation p-values.
            result['contrast_fwer_ci_low'] = np.quantile(boot[:, 2], .025 / 14)
            result['contrast_fwer_ci_high'] = np.quantile(boot[:, 2], 1 - .025 / 14)
            summaries.append(result)
    pd.DataFrame(summaries).to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(seeds).to_csv(out / 'randomization_gains.csv', index=False)
    return summaries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--repeats', type=int, default=100)
    ap.add_argument('--bootstrap', type=int, default=10000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--models', nargs='+', default=MODELS, choices=MODELS)
    ap.add_argument('--families', nargs='+', default=list(FAMILIES), choices=list(FAMILIES))
    ap.add_argument('--summarize-only', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(repeats=args.repeats, bootstrap_reps=args.bootstrap, seed=42,
        features_added=8, shuffle='Joint timing-block row permutation, separately in train and test in each frozen fold; across domains, not within domain.',
        bootstrap='Domain-stratified prompts and independent replicate weights, fixed fitted predictions; pointwise percentile intervals plus Bonferroni intervals for 14 contrasts.',
        caveat='Negative-control sensitivity analysis, not an exact conditional permutation test; seed percentiles are not confidence intervals.',
        inputs=[dict(file=str(f.relative_to(ROOT)), sha256=digest(f)) for f in [FEATURES, SPLITS, REFERENCE]],
        code=[dict(file=str(f.relative_to(ROOT)), sha256=digest(f)) for f in [Path(__file__), ROOT / 'src/analysis/paper/signature_analysis.py', ROOT / 'src/analysis/paper/constants.py']],
        source_commit=(ROOT / 'SOURCE_COMMIT').read_text().strip() if (ROOT / 'SOURCE_COMMIT').exists() else subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    mp = args.out / 'run_manifest.json'
    if mp.exists():
        assert json.loads(mp.read_text())['identity'] == identity, 'Output directory belongs to a different run.'
    manifest = dict(identity=identity, config=config, started_utc=datetime.now(timezone.utc).isoformat(),
        host=platform.node(), python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    mp.write_text(json.dumps(manifest, indent=2) + '\n')
    if not args.summarize_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_cell, m, f, str(args.out), args.repeats, 42, identity)
                for m in args.models for f in args.families]
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)
    if len(list(args.out.glob('*.npz'))) == 14:
        summaries = summarize(args.out, args.repeats, args.bootstrap, 42)
        manifest['summary'] = dict(cells=len(summaries),
            total_randomizations=14 * args.repeats,
            timing_exceeds_random_fwer=int(sum(x['contrast_fwer_ci_low'] > 0 for x in summaries)),
            randomization_draws_at_least_timing=int(sum(x['randomizations_at_least_timing'] for x in summaries)))
        manifest['completed_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['outputs'] = [dict(file=f.name, sha256=digest(f)) for f in [args.out / 'summary.csv', args.out / 'randomization_gains.csv']]
        mp.write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(manifest['summary']), flush=True)


if __name__ == '__main__':
    main()
