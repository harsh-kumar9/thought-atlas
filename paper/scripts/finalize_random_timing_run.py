"""Validate completed random-control arrays and repair only manifest serialization."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd

out = Path(sys.argv[1])
mp = out / 'run_manifest.json'
manifest = json.loads(mp.read_text())
summary = pd.read_csv(out / 'summary.csv')
draws = pd.read_csv(out / 'randomization_gains.csv')
repeats = manifest['config']['repeats']
assert len(summary) == 14 and len(draws) == 14 * repeats
assert not summary.duplicated(['gen_model', 'signature_family']).any()
assert summary.notna().all().all()
for row in summary.itertuples():
    with np.load(out / f'{row.gen_model}__{row.signature_family}.npz') as z:
        assert str(z['identity']) == manifest['identity']
        assert z['random_gain'].shape == (repeats, row.n)
        real, random = z['real_gain'], z['random_gain']
        assert np.isfinite(real).all() and np.isfinite(random).all()
        assert abs(real.mean() - row.timing_gain_bits) < 1e-12
        assert abs(random.mean() - row.random_gain_bits) < 1e-12
        assert abs(real.mean() - random.mean() - row.timing_minus_random_bits) < 1e-12
        stored = draws[(draws.gen_model == row.gen_model) & (draws.signature_family == row.signature_family)].sort_values('replicate')
        assert np.allclose(random.mean(axis=1), stored.gain_bits, atol=1e-12, rtol=0)
        assert np.allclose(np.quantile(random.mean(axis=1), [.025, .975]), [row.random_seed_p025, row.random_seed_p975], atol=1e-12, rtol=0)
        assert int((random.mean(axis=1) >= real.mean()).sum()) == row.randomizations_at_least_timing
        assert float(z['fit_error_amount']) < 1e-5 and float(z['fit_error_full']) < 1e-5
    assert row.contrast_fwer_ci_low <= row.timing_minus_random_ci_low
    assert row.contrast_fwer_ci_high >= row.timing_minus_random_ci_high
backup = out / 'run_manifest.before_finalization.json'
if not backup.exists():
    shutil.copy2(mp, backup)
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest['summary'] = dict(cells=14, total_randomizations=len(draws),
    timing_exceeds_random_fwer=int((summary.contrast_fwer_ci_low > 0).sum()),
    randomization_draws_at_least_timing=int(summary.randomizations_at_least_timing.sum()))
manifest['completed_utc'] = datetime.now(timezone.utc).isoformat()
manifest['outputs'] = [dict(file=f.name, sha256=digest(f)) for f in [out / 'summary.csv', out / 'randomization_gains.csv']]
manifest['finalization'] = dict(reason='All fits and bootstrap CSVs completed; original job exited during JSON serialization of a NumPy integer. No numerical results changed.',
    script_sha256=digest(__file__), validated_arrays=14, validated_randomizations=len(draws))
mp.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest['summary']))
