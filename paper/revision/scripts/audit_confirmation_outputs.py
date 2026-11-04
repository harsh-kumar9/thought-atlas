"""Verify frozen confirmation outputs without changing samples, grades or tests."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def audit(root, out):
    sys.path.insert(0, str(root))
    from mode_control.mc.common import load_tokenizer
    from mode_control.mc.readiness import FORCE, grade_forced
    from elicitation import prefix_plan
    protocol = json.loads((out/'protocol.json').read_text())
    manifest = json.loads((out/'analysis_manifest.json').read_text())
    assert sha(out/'protocol.json') == manifest['protocol_sha256']
    assert sha(out/'tasks.parquet') == protocol['tasks_sha256']
    for p, h in manifest['outcome_files'].items():
        assert sha(out/p) == h, p
    tasks = pd.read_parquet(out/'tasks.parquet').set_index('instance_id')
    expected = {(i, s) for i in tasks.index for s in [0, 1]}
    tokenizer = load_tokenizer()
    cue = tokenizer.encode(FORCE['math'], add_special_tokens=False)
    end = tokenizer.convert_tokens_to_ids('</think>')
    records, coverage = [], []
    for arm in protocol['arms']:
        work = out/arm
        tr = pd.read_parquet(work/'rollouts_s001.parquet').set_index('trace_id')
        gr = pd.read_parquet(work/'rewards_s001.parquet').set_index('trace_id')
        el = pd.read_parquet(work/'elicitation.parquet').set_index('trace_id')
        assert len(tr) == len(gr) == len(el) == 512
        assert all(d.index.is_unique for d in [tr, gr, el])
        assert set(tr.index) == set(gr.index) == set(el.index)
        assert set(zip(tr.instance_id, tr['sample'])) == expected
        assert sha(out/f'vector_{arm}.npy') == protocol['vectors'][arm]['vector_sha256']
        assert np.array_equal(np.load(work/'b_s000.npy'), np.load(out/f'vector_{arm}.npy'))
        for tid, r in tr.iterrows():
            g, e = gr.loc[tid], el.loc[tid]
            assert r.prompt == tasks.loc[r.instance_id, 'prompt']
            assert r.reference_answer == tasks.loc[r.instance_id, 'reference_answer']
            assert e.instance_id == r.instance_id and e['sample'] == r['sample']
            ids = list(r.gen_token_ids)
            plan = prefix_plan(ids, end, 8192, len(cue), completed=r.completed)
            assert e.own == plan['own']
            assert e.total_tokens == e.prefix_tokens+e.cue_tokens+e.answer_tokens <= 8192
            if e.own:
                assert e.correct == g.success and e.prefix_tokens == len(ids)
                assert e.cue_tokens == e.answer_tokens == 0
            else:
                assert e.prefix_tokens == plan['prefix_len'] and e.cue_tokens == len(cue)
                assert 0 < e.answer_tokens <= 48
                answer, correct = grade_forced('math', e.raw_answer, r.reference_answer, r.prompt)
                assert answer == e.answer and correct == e.correct
            records.append(dict(condition=arm, trace_id=tid, instance_id=r.instance_id,
                                sample=r['sample'], natural_correct=g.success, elicited_correct=e.correct))
        forced = el[~el.own]
        coverage.append(dict(condition=arm, attempts=512, tasks=256, own_answers=int(el.own.sum()),
                             forced_answers=len(forced), forced_parseable=int(forced.answer.fillna('').str.strip().ne('').sum()),
                             forced_correct=int(forced.correct.sum()), natural_correct=int(gr.success.sum()),
                             elicited_correct=int(el.correct.sum()),
                             extraction_fallbacks=int(gr.grade_source.str.startswith('fallback_').sum())))
    observed = pd.DataFrame(records)
    stored = pd.read_csv(out/'outcomes.csv')
    checked = observed.merge(stored, on=['condition','trace_id','instance_id','sample'], validate='one_to_one', suffixes=('_raw','_saved'))
    assert len(checked) == 1536
    for metric in ['natural_correct','elicited_correct']:
        assert checked[metric+'_raw'].equals(checked[metric+'_saved'])
    effects = pd.read_csv(out/'paired_effects.csv')
    for r in effects.itertuples():
        t = stored.groupby(['condition','instance_id'])[r.metric].mean().unstack('condition')
        assert np.isclose((t[r.condition]-t.baseline).mean(), r.delta)
    report = dict(passed=True, attempts=1536, tasks=256, samples=2, budget=8192,
                  pairing='Exact frozen task/sample grid in every arm; raw-to-analysis one-to-one joins checked.',
                  grading='All forced answers re-parsed and regraded with frozen grader; retained own grades match original rewards.',
                  scope='Integrity and reproducibility checks; not independent semantic validation of extraction or grading.',
                  code_sha256=sha(Path(__file__)), coverage=coverage)
    (out/'output_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    pd.DataFrame(coverage).to_csv(out/'endpoint_coverage.csv', index=False)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.root,a.out)
