"""Audit complete matched-total-budget outputs and pair tasks after averaging seeds."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from elicitation import ARMS

def paired(d,reps=2000):
    metrics=['correct','answered','finished','total_generated_and_cue_tokens']
    task=d.groupby(['task_type','budget','condition','instance_id'])[metrics].mean().reset_index()
    rng=np.random.default_rng(20261101);rows=[]
    for (target,budget),g in task.groupby(['task_type','budget']):
        base=g[g.condition.eq('baseline')].set_index('instance_id')
        for condition,a in g.groupby('condition'):
            if condition=='baseline':continue
            a=a.set_index('instance_id');assert set(a.index)==set(base.index)
            a=a.loc[base.index]
            ix=rng.integers(0,len(a),(reps,len(a)))
            for metric in metrics:
                delta=(a[metric]-base[metric]).to_numpy();boot=delta[ix].mean(1)
                rows.append(dict(task_type=target,budget=budget,condition=condition,metric=metric,n_tasks=len(a),
                  baseline=base[metric].mean(),intervention=a[metric].mean(),delta=delta.mean(),
                  lo=np.quantile(boot,.025),hi=np.quantile(boot,.975)))
    return pd.DataFrame(rows)

def main(out):
    parts=[];inputs={}
    for target,arms in ARMS.items():
        p=out/f'{target}.parquet';d=pd.read_parquet(p)
        status=json.loads((out/f'status_{target}.json').read_text())
        config=json.loads((out/f'config_{target}.json').read_text())
        assert len(d)==status['completed']==status['expected']
        assert not d.duplicated(['trace_id','budget']).any()
        assert set(d.condition)==set(arms) and set(d.budget)==set(config['budgets'])
        assert d.total_generated_and_cue_tokens.le(d.budget).all()
        assert d.correct.isin([0,1]).all()
        assert d.loc[d.finished,'original_completed'].all()
        assert d.loc[d.finished,'original_output_tokens'].le(d.loc[d.finished,'budget']).all()
        assert (d.loc[~d.finished,'prefix_tokens']+d.loc[~d.finished,'cue_tokens']+
                d.loc[~d.finished,'answer_tokens']).eq(d.loc[~d.finished,'total_generated_and_cue_tokens']).all()
        base=set(map(tuple,d[d.condition.eq('baseline')][['instance_id','seed','budget']].to_numpy()))
        for condition,g in d.groupby('condition'):
            assert set(map(tuple,g[['instance_id','seed','budget']].to_numpy()))==base,condition
        d['answered']=d.answer.fillna('').str.strip().ne('').astype(int)
        d['forced']=~d.finished
        parts.append(d);inputs[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    d=pd.concat(parts,ignore_index=True)
    summary=d.groupby(['task_type','condition','budget']).agg(
      attempts=('trace_id','size'),tasks=('instance_id','nunique'),correct=('correct','mean'),
      answered=('answered','mean'),own_answer=('finished','mean'),tokens=('total_generated_and_cue_tokens','mean'))
    summary.reset_index().to_csv(out/'summary.csv',index=False)
    paired(d).to_csv(out/'paired.csv',index=False)
    d.groupby(['task_type','condition','budget','grade_source']).size().rename('attempts').reset_index().to_csv(out/'grade_sources.csv',index=False)
    (out/'analysis_manifest.json').write_text(json.dumps(dict(inputs=inputs,rows=len(d),
       code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       design='All allocated rows; same prompt/seed/budget grid per arm; task bootstrap after seed averaging; 2000 replicates.',
       estimand='Correct and parseable answers after common-budget elicitation. Retrospective on reused historical evaluation items.',
       grading='Corrected saved extraction for own answers when available; direct math/MCQ parsing for forced answers; sources reported.',
       budget='Own responses must be completed and fit in full; otherwise prefix, cue, and continuation sum to at most budget.'),indent=2))
    print('Elicitation analysis complete',len(d),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
