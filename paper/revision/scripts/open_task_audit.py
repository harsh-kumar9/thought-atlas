"""Actual allocated and paired-answer denominators for the displayed idea arms."""
from pathlib import Path
import argparse,hashlib,json
import pandas as pd
import numpy as np

def main(root,out):
    data=root/'mode_control/data';rows=[];inputs=[];summaries=[];exclusions=[]
    for run,arm in [('sched1','sus_math_L19'),('sched1','sus_idea_L19'),('tx_idea','sig_math'),('tx_idea','prompt_math')]:
        frames=[]
        for name in ['baseline',arm]:
            p=data/f'runs/{run}/traces_{name}.parquet'
            d=pd.read_parquet(p,columns=['trace_id','instance_id','task_type','seed','answer_text','completed'])
            d=d[d.task_type.eq('idea')];assert not d.duplicated(['instance_id','seed']).any()
            d['answered']=d.answer_text.fillna('').str.strip().ne('')
            frames.append(d);inputs.append(dict(file=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        base,armdata=frames
        both=armdata.merge(base,on=['instance_id','seed'],suffixes=('_arm','_base'),validate='1:1',how='left',indicator=True)
        paired=both._merge.eq('both')&both.answered_arm&both.answered_base.fillna(False)
        for name,d in [('baseline',base),(arm,armdata)]:
            rows.append(dict(run=run,comparison=arm,condition=name,attempted=len(d),unique_keywords=d.instance_id.nunique(),
              seeds=','.join(map(str,sorted(d.seed.unique()))),completed=int(d.completed.sum()),answered=int(d.answered.sum()),
              paired_answer_rows=int(paired.sum()),paired_keywords=both.loc[paired,'instance_id'].nunique(),
              missing_baseline=int(both._merge.ne('both').sum())))
        for judge in ['google_gemma-4-31B-it','Qwen_Qwen3.5-27B']:
            p=data/f'pairwise/{run}__{judge}.parquet';all_calls=pd.read_parquet(p)
            calls=all_calls[all_calls.condition.eq(arm)].copy()
            keep=calls.trace_id.isin(armdata.trace_id)
            exclusions.append(dict(run=run,condition=arm,judge=judge,original_calls=len(calls),
               excluded_other_task_calls=int((~keep).sum()),kept_idea_calls=int(keep.sum())))
            calls=calls[keep]
            assert not calls.duplicated(['instance_id','seed','axis','order']).any()
            assert calls.trace_id.isin(both.loc[paired,'trace_id_arm']).all()
            for axis,g in calls.groupby('axis'):
                x=(g.dropna(subset=['score']).groupby(['instance_id','seed']).score.mean()
                     .groupby('instance_id').mean().sort_index().to_numpy())
                assert len(x)==both.loc[paired,'instance_id'].nunique()
                rng=np.random.default_rng(20261101);boot=x[rng.integers(0,len(x),(2000,len(x)))].mean(1)
                summaries.append(dict(run=run,condition=arm,judge=judge,axis=axis,n_keywords=len(x),n_calls=len(g),
                   net_win=x.mean(),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975),unparsed=int(g.score.isna().sum())))
            inputs.append(dict(file=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    pd.DataFrame(rows).to_csv(out/'open_task_coverage.csv',index=False)
    for judge,g in pd.DataFrame(summaries).groupby('judge'):
        g.to_csv(out/f'idea_pairwise_{judge}.csv',index=False)
    pd.DataFrame(exclusions).to_csv(out/'idea_pairwise_exclusions.csv',index=False)
    (out/'open_task_manifest.json').write_text(json.dumps(dict(inputs=inputs,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      interpretation='Actual saved attempts, restricted to idea tasks. Pair eligibility is nonempty final text in both arms; keyword and generation-row counts differ with two seeds.'),indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root,a.out)
