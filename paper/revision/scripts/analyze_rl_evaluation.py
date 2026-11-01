"""Report fixed-checkpoint effects by training seed; task and training variability stay distinct."""
from pathlib import Path
import argparse,json
import pandas as pd
import numpy as np

def main(out):
    runs=['baseline','a2']+[f'{init}_seed{s}' for init in ['zero','ol_math_A2'] for s in range(3)]
    parts=[]
    for run in runs:
        assert json.loads((out/run/'status.json').read_text())['stage']=='complete'
        d=pd.read_csv(out/run/'outcomes.csv');assert len(d)==398 and d.instance_id.nunique()==199
        assert not d.duplicated(['instance_id','sample']).any()
        d['run']=run;parts.append(d)
    d=pd.concat(parts,ignore_index=True)
    task=d.groupby(['run','instance_id'])[['success','completed']].mean().reset_index()
    estimates=[];rng=np.random.default_rng(20261102)
    for metric in ['success','completed']:
        p=task.pivot(index='instance_id',columns='run',values=metric)
        assert not p.isna().any().any()
        indices=rng.integers(0,len(p),(2000,len(p)))
        for run in runs[1:]:
            for reference in ['baseline','a2']:
                if run==reference:continue
                delta=(p[run]-p[reference]).to_numpy();b=delta[indices].mean(1)
                estimates.append(dict(run=run,reference=reference,metric=metric,n_tasks=len(p),
                   absolute=p[run].mean(),reference_absolute=p[reference].mean(),delta=delta.mean(),
                   lo=np.quantile(b,.025),hi=np.quantile(b,.975)))
    pd.DataFrame(estimates).to_csv(out/'paired_effects.csv',index=False)
    task.groupby('run')[['success','completed']].mean().to_csv(out/'absolute_by_training_seed.csv')
    (out/'analysis_manifest.json').write_text(json.dumps(dict(runs=runs,budget=8192,checkpoint=60,
      interval='2000 paired task bootstraps after averaging two samples. Conditional on each trained vector; not a training-population interval.',
      training_variability='All three seeds per initialization reported separately; no best-seed or checkpoint selection.',
      scope='Corrected-reward replication on historically reused evaluation problems, with fresh baseline/A2 controls. Behavioral annotation remains a separate analysis.'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
