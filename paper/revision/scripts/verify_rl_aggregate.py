"""Independently reproduce paired estimates and retain parser uncertainty bounds."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd


def verify(root):
    runs = ['baseline', 'a2'] + [f'{i}_seed{s}' for i in ['zero', 'ol_math_A2'] for s in range(3)]
    frames, hashes, fallback = {}, {}, {}
    reference_config = json.loads((root/'baseline/config.json').read_text())
    for run in runs:
        folder = root/run
        assert json.loads((folder/'status.json').read_text())['stage'] == 'complete'
        config = json.loads((folder/'config.json').read_text())
        for key in ['decode', 'budget', 'sampling_seed', 'samples_per_problem', 'task_source_sha256', 'code_sha256']:
            assert config[key] == reference_config[key], (run, key)
        data = pd.read_csv(folder/'outcomes.csv')
        assert len(data) == 398 and data.instance_id.nunique() == 199
        assert not data.duplicated(['instance_id', 'sample']).any()
        assert set(data['sample']) == {0, 1}
        assert data.groupby('instance_id')['sample'].count().eq(2).all()
        assert data.success.isin([0, 1]).all() and data.completed.isin([False, True]).all()
        assert data.loc[~data.completed, 'success'].eq(0).all()
        summary = json.loads((folder/'summary.json').read_text())
        assert np.isclose(data.success.mean(), summary['accuracy'])
        assert np.isclose(data.completed.mean(), summary['completion'])
        digest = hashlib.sha256((folder/'outcomes.csv').read_bytes()).hexdigest()
        if run not in ['baseline', 'a2']:
            audit = json.loads((folder/'completion_audit.json').read_text())
            assert audit['hashes']['outcomes.csv'] == digest
            assert audit['grade_mismatches'] == 0 and audit['corrected_grade_reproduction_rows'] == 398
        hashes[run] = digest
        frames[run] = data.groupby('instance_id')[['success', 'completed']].mean().sort_index()
        assert frames[run].index.equals(frames['baseline'].index)
        is_fallback = data.grade_source.str.startswith('fallback')
        fallback[run] = {'n': int(is_fallback.sum()),
                         'successes': int(data.loc[is_fallback, 'success'].sum())}
    observed = pd.read_csv(root/'paired_effects.csv').set_index(['metric', 'run', 'reference'])
    assert len(observed) == 26 and observed.index.is_unique
    rng = np.random.default_rng(20261102)
    sensitivity = []
    for metric in ['success', 'completed']:
        draws = rng.integers(0, 199, size=(2000, 199))
        for run in runs[1:]:
            for reference in ['baseline', 'a2']:
                if run == reference:
                    continue
                delta = frames[run][metric].to_numpy() - frames[reference][metric].to_numpy()
                low, high = np.quantile(np.mean(delta[draws], axis=1), [.025, .975])
                row = observed.loc[(metric, run, reference)]
                assert np.allclose([row.delta, row.lo, row.hi], [delta.mean(), low, high], atol=1e-14, rtol=0)
                assert row.n_tasks == 199
                assert np.isclose(row.absolute, frames[run][metric].mean())
                assert np.isclose(row.reference_absolute, frames[reference][metric].mean())
                if metric == 'success':
                    a, b = fallback[run], fallback[reference]
                    # Deterministic bounds on the point estimate, not confidence intervals.
                    lower = delta.mean() - (a['successes'] + b['n'] - b['successes'])/398
                    upper = delta.mean() + (a['n'] - a['successes'] + b['successes'])/398
                    sensitivity.append(dict(run=run, reference=reference, delta=delta.mean(),
                        fallback_min=lower, fallback_max=upper,
                        run_fallbacks=a['n'], reference_fallbacks=b['n']))
    pd.DataFrame(sensitivity).to_csv(root/'fallback_sensitivity.csv', index=False)
    result = dict(runs=runs, attempts=3184, tasks=199, exact_paired_grid=True,
        grade_reproduction='All six learned-vector arms: 398/398 grades each.',
        paired_estimates_reproduced=26, bootstrap_replicates=2000,
        outcomes_sha256=hashes, fallback_rows=fallback,
        interval_scope='Pointwise task uncertainty conditional on each trained vector; not training-population uncertainty.',
        evaluation_scope='Historically reused evaluation items; no fresh-holdout claim.')
    (root/'aggregate_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    verify(parser.parse_args().root)
