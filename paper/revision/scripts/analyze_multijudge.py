"""Cross-family annotation robustness; no judge or consensus is ground truth.

All estimands use the frozen common-prefix sample. Agreement is reported per
label, including positive agreement (so abundant negatives cannot hide failures
on rare operations). Uncertainty resamples whole traces, not sentences.
"""
from pathlib import Path
import argparse, itertools, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from multijudge import BEH, JUDGES

def agreement(counts):
    """Counts ordered 00, 01, 10, 11; symmetric positive/negative agreement."""
    a,b,c,d=np.asarray(counts,float);n=a+b+c+d
    if not n:return {k:np.nan for k in ['agreement','positive_agreement','negative_agreement','kappa']}
    po=(a+d)/n;pe=((a+b)*(a+c)+(c+d)*(b+d))/n**2
    return dict(agreement=po,positive_agreement=2*d/(2*d+b+c) if 2*d+b+c else np.nan,
                negative_agreement=2*a/(2*a+b+c) if 2*a+b+c else np.nan,
                kappa=(po-pe)/(1-pe) if pe<1-1e-12 else np.nan)

def bootstrap_mean(x,rng,reps=2000):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if not len(x):return np.nan,np.nan,np.nan
    bs=x[rng.integers(0,len(x),(reps,len(x)))].mean(1)
    return float(x.mean()),float(np.quantile(bs,.025)),float(np.quantile(bs,.975))

def main(out,allow_partial=False):
    key=pd.read_parquet(out/'sample_key.parquet');expected=set(key.loc[key.analysis_tokens.gt(0),'trace_id'])
    labels={};coverage=[]
    for judge in JUDGES:
        path=out/f'labels_{judge}.parquet'
        if not path.exists():
            if allow_partial:continue
            raise FileNotFoundError(path)
        d=pd.read_parquet(path)
        assert not d.duplicated(['trace_id','seg_idx']).any()
        if not allow_partial:assert expected<=set(d.trace_id),f'{judge}: incomplete annotations'
        for b in BEH:
            parsed='kim_parsed' if b in BEH[:4] else 'gandhi_parsed'
            d.loc[~d[parsed].fillna(False),b]=np.nan
        d=d.merge(key,on='trace_id',validate='many_to_one')
        labels[judge]=d
        for keys,g in d.groupby(['source','gen_model','task_type'],dropna=False):
            coverage.append(dict(judge=judge,source=keys[0],gen_model=keys[1],task_type=keys[2],
                traces=g.trace_id.nunique(),segments=len(g),kim_valid=g.kim_parsed.mean(),
                cognitive_valid=g.gandhi_parsed.mean(),truncated=g.judge_context_truncated.mean()))
    if not labels:raise ValueError('No annotation files available')
    pd.DataFrame(coverage).to_csv(out/'parse_coverage.csv',index=False)
    rng=np.random.default_rng(20261101);agreements=[]
    for j1,j2 in itertools.combinations(labels,2):
        a=labels[j1];b=labels[j2]
        cols=['trace_id','seg_idx','n_segments']+BEH
        both=a[cols+['source','task_type']].merge(b[cols],on=['trace_id','seg_idx'],suffixes=('_a','_b'),validate='one_to_one')
        assert both.n_segments_a.eq(both.n_segments_b).all(),'Judges saw different segmentation'
        for source in sorted(both.source.unique()):
          for domain in ['all']+sorted(both.loc[both.source.eq(source),'task_type'].unique()):
            g=both[both.source.eq(source)&(True if domain=='all' else both.task_type.eq(domain))]
            for label in BEH:
                z=g[['trace_id',label+'_a',label+'_b']].dropna()
                if not len(z):continue
                z['cell']=(2*z[label+'_a']+z[label+'_b']).astype(int)
                ct=pd.crosstab(z.trace_id,z.cell).reindex(columns=range(4),fill_value=0).to_numpy()
                # Each trace contributes unit total weight, regardless of segment count.
                weighted=ct/ct.sum(1,keepdims=True);estimate=agreement(weighted.sum(0))
                bs=[agreement(weighted[rng.integers(0,len(ct),len(ct))].sum(0)) for _ in range(1000)]
                rec=dict(judge_a=j1,judge_b=j2,source=source,domain=domain,label=label,
                         traces=len(ct),valid_paired_segments=len(z),**estimate)
                rec.update({f'n_{i:02b}':int(ct[:,i].sum()) for i in range(4)})
                for metric in estimate:
                    v=np.array([x[metric] for x in bs]);v=v[np.isfinite(v)]
                    rec[metric+'_lo']=np.quantile(v,.025) if len(v) else np.nan
                    rec[metric+'_hi']=np.quantile(v,.975) if len(v) else np.nan
                means=z.groupby('trace_id')[[label+'_a',label+'_b']].mean()
                rec['trace_rate_spearman']=spearmanr(means.iloc[:,0],means.iloc[:,1]).statistic if means.nunique().min()>1 else np.nan
                agreements.append(rec)
    pd.DataFrame(agreements).to_csv(out/'pairwise_agreement.csv',index=False)
    profiles=[]
    for label in BEH:
        common=None
        for d in labels.values():
            ids=set(map(tuple,d.loc[d[label].notna(),['trace_id','seg_idx']].to_numpy()))
            common=ids if common is None else common&ids
        for judge,d in labels.items():
            for scope in ['judge_valid','common_valid']:
                g=d if scope=='judge_valid' else d[[tuple(x) in common for x in d[['trace_id','seg_idx']].to_numpy()]]
                agg=g.groupby('trace_id')[label].agg(['mean','count']).reset_index().rename(columns={'mean':'rate','count':'valid_segments'})
                agg=agg.merge(key,on='trace_id',validate='one_to_one');agg['judge']=judge;agg['label']=label;agg['scope']=scope
                profiles.append(agg.drop(columns=['prompt','reasoning_text_for_analysis','think_text'],errors='ignore'))
    profiles=pd.concat(profiles,ignore_index=True);profiles.to_parquet(out/'trace_profiles.parquet',index=False)
    atlas=[]
    for keys,g in profiles[profiles.source.eq('atlas')].groupby(['judge','scope','gen_model','task_type','label']):
        g=g[g.rate.notna()];w=1/g.inclusion_probability.to_numpy();y=g.rate.to_numpy()
        if not len(g):continue
        # Ratio (Hajek) estimate. Sampling uncertainty is not an instrument-validity interval.
        atlas.append(dict(zip(['judge','scope','gen_model','task_type','label'],keys),
                          rate=np.average(y,weights=w),n=len(g),effective_n=w.sum()**2/(w@w),
                          incomplete=int((~g.completed).sum())))
    atlas=pd.DataFrame(atlas);atlas.to_csv(out/'atlas_weighted_profiles.csv',index=False)
    contrasts=[]
    selected=[('idea','math','Perspective_Shift'),('idea','math','backtracking'),
              ('planning','moral','verification'),('math','planning','subgoal')]
    for judge in labels:
      for scope in ['judge_valid','common_valid']:
        p=atlas[atlas.judge.eq(judge)&atlas.scope.eq(scope)]
        for high,low,label in selected:
            pivot=p[p.label.eq(label)].pivot(index='gen_model',columns='task_type',values='rate')
            if high not in pivot or low not in pivot:continue
            delta=(pivot[high]-pivot[low]).dropna()
            contrasts.append(dict(judge=judge,scope=scope,high=high,low=low,label=label,
                                  mean_difference=delta.mean(),models_positive=int(delta.gt(0).sum()),models=len(delta)))
    pd.DataFrame(contrasts).to_csv(out/'atlas_contrasts.csv',index=False)
    paired=[]
    for keys,g in profiles[profiles.source.isin(['intervention','demand'])].groupby(['source','judge','scope','gen_model','task_type','label']):
        source,judge,scope,model,domain,label=keys
        item='base_item' if source=='demand' else 'instance_id'
        p=g.pivot(index=item,columns='condition',values='rate')
        baseline='solve_free' if source=='demand' else 'baseline'
        # The source data encode demand conditions explicitly; never guess the reference.
        if baseline not in p:continue
        for condition in p.columns:
            if condition==baseline:continue
            delta=(p[condition]-p[baseline]).dropna();mean,lo,hi=bootstrap_mean(delta,rng)
            paired.append(dict(source=source,judge=judge,scope=scope,gen_model=model,task_type=domain,
                               label=label,condition=condition,reference=baseline,n=len(delta),delta=mean,lo=lo,hi=hi))
    pd.DataFrame(paired).to_csv(out/'paired_behavior_shifts.csv',index=False)
    (out/'analysis_manifest.json').write_text(json.dumps(dict(partial=allow_partial,judges=list(labels),
      requested_nonempty=len(expected),empty_prefixes=int(key.analysis_tokens.eq(0).sum()),
      agreement='Trace-balanced binary agreement; trace-cluster percentile intervals; no gold judge.',
      atlas='Inverse inclusion-probability weighted rates within model/domain; equal-model selected contrasts.',
      interpretation='Fixed-prefix robustness, not independent human validation. No majority consensus treated as truth.'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();main(a.out,a.allow_partial)
