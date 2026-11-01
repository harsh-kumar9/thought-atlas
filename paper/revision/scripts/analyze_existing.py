"""Paired fidelity, dose, duration and matched-demand checks from frozen data."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from multijudge import BEH
from readout_validation import TARGETS

def cosine(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den>1e-12 else np.nan
def interval(x,rng):
    x=np.asarray(x,float)
    bs=x[rng.integers(0,len(x),(2000,len(x)))].mean(1)
    return x.mean(),*np.quantile(bs,[.025,.975])

def analyze(revision):
    out=revision/'results/behavior';src=revision/'sources';rng=np.random.default_rng(20261101)
    mapfile=np.load(src/'vectors/map_corpus1_dm8_L16_r8.npz',allow_pickle=True)
    W,sd=mapfile['W'],mapfile['sd'];vectors={}
    for p in sorted((src/'vectors').glob('vec_sig_*.npz')):
        z=np.load(p,allow_pickle=True);meta=json.loads(str(z['meta']))
        vectors[p.stem.removeprefix('vec_')]=(np.array(meta['dphi16']),((z['v']/sd)@W).astype(float))
    shiftrows=[];fidelity=[];dose=[];sims=[]
    outcomes=pd.read_parquet(revision/'results/audit/budget_outcomes.parquet')
    for path in sorted(out.glob('profiles_*.parquet')):
        d=pd.read_parquet(path);target=d.target.iloc[0];base=d[d.condition.eq('baseline')].set_index('instance_id')
        for arm,group in d.groupby('condition'):
            if arm=='baseline':continue
            g=group.set_index('instance_id');ids=g.index.intersection(base.index);g=g.loc[ids];b=base.loc[ids]
            for window in [1024,2048,4096,8192]:
                cols=[f'{x}_{window}' for x in BEH];diff=g[cols].to_numpy()-b[cols].to_numpy()
                for j,label in enumerate(BEH):
                    x=diff[:,j];x=x[np.isfinite(x)]
                    if not len(x):continue
                    v,lo,hi=interval(x,rng)
                    shiftrows.append(dict(target=target,condition=arm,window=window,label=label,n=len(x),delta=v,lo=lo,hi=hi))
            if arm in vectors:
                wanted,implied=vectors[arm];delta=(g[TARGETS]-b[TARGETS]).dropna().to_numpy()
                observed=delta.mean(0);ix=rng.integers(0,len(delta),(2000,len(delta)));boots=delta[ix].mean(1)
                c=np.array([cosine(x,wanted) for x in boots]);proj=boots@wanted/(wanted@wanted)
                fidelity.append(dict(target=target,condition=arm,n=len(delta),
                  intended_cosine=cosine(observed,wanted),cosine_lo=np.nanquantile(c,.025),cosine_hi=np.nanquantile(c,.975),
                  implied_cosine=cosine(observed,implied),intended_projection=float(observed@wanted/(wanted@wanted)),
                  projection_lo=np.quantile(proj,.025),projection_hi=np.quantile(proj,.975),
                  observed_norm=np.linalg.norm(observed),implied_norm=np.linalg.norm(implied),
                  **{f'intended_{x}':v for x,v in zip(TARGETS,wanted)},
                  **{f'implied_{x}':v for x,v in zip(TARGETS,implied)},
                  **{f'observed_{x}':v for x,v in zip(TARGETS,observed)}))
        # Equal task set and seed for every dose, including the 1x arm and baseline.
        for source in ['math','gpqa','planning','idea','moral']:
            arms=['sig_'+source+s for s in ['_x0p5','','_x1p5']]
            dd={arm:d[d.condition.eq(arm)].set_index('instance_id') for arm in ['baseline']+arms}
            ids=sorted(set.intersection(*(set(x.index) for x in dd.values())))
            if not ids:continue
            for arm,scale in zip(arms,[.5,1.,1.5]):
                g=dd[arm].loc[ids];b=dd['baseline'].loc[ids]
                rec=dict(target=target,source=source,scale=scale,condition=arm,n=len(ids),seed=1,
                    hard_loop_rate=g.hard_loop.fillna(False).mean(),baseline_hard_loop_rate=b.hard_loop.fillna(False).mean(),
                    generated_tokens=g.generated_tokens.mean(),baseline_tokens=b.generated_tokens.mean(),
                    behavior_shift_norm=np.linalg.norm((g[TARGETS]-b[TARGETS]).mean().to_numpy()))
                for budget in [12288,32768]:
                    o=outcomes[outcomes.task_type.eq(target)&outcomes.budget.eq(budget)&outcomes.seed.eq(1)&outcomes.instance_id.isin(ids)]
                    p=o.pivot(index='instance_id',columns='condition',values='correct')
                    av=o.pivot(index='instance_id',columns='condition',values='answered')
                    if arm in p and 'baseline' in p:
                        delta=(p[arm]-p.baseline).dropna();v,lo,hi=interval(delta,rng)
                        rec.update({f'd_accuracy_{budget}':v,f'accuracy_lo_{budget}':lo,f'accuracy_hi_{budget}':hi,
                            f'accuracy_{budget}':p[arm].mean(),f'baseline_accuracy_{budget}':p.baseline.mean(),
                            f'd_answered_{budget}':(av[arm]-av.baseline).mean(),f'outcome_n_{budget}':len(delta)})
                dose.append(rec)
        if target=='math':
            reference=d[d.condition.eq('ol_math_A2')].set_index('instance_id')
            for arm in ['rl_zeroT_s60','rl_a2initT_s60','sealdir_matched','seal','concise','ol_random']:
                other=d[d.condition.eq(arm)].set_index('instance_id')
                ids=sorted(set(base.index)&set(reference.index)&set(other.index))
                for window in [1024,2048,4096,8192]:
                    cols=[f'{x}_{window}' for x in BEH]
                    for population in ['all_paired','all_three_reach_window']:
                        keep=ids if population=='all_paired' else [i for i in ids if min(base.loc[i,'reasoning_tokens'],reference.loc[i,'reasoning_tokens'],other.loc[i,'reasoning_tokens'])>=window]
                        if len(keep)<10:continue
                        a=(reference.loc[keep,cols]-base.loc[keep,cols]).to_numpy();b=(other.loc[keep,cols]-base.loc[keep,cols]).to_numpy()
                        valid=np.isfinite(a).all(1)&np.isfinite(b).all(1);a=a[valid];b=b[valid]
                        boot=[];ratio=[];split=[]
                        for _ in range(2000):
                            ix=rng.integers(0,len(a),len(a));am=a[ix].mean(0);bm=b[ix].mean(0)
                            boot.append(cosine(am,bm));ratio.append(np.linalg.norm(bm)/np.linalg.norm(am))
                            # Disjoint item halves avoid literally sharing baseline observations.
                            perm=rng.permutation(len(a));half=len(a)//2
                            split.append(cosine(a[perm[:half]].mean(0),b[perm[half:]].mean(0)))
                        sims.append(dict(condition=arm,window=window,population=population,n=len(a),
                          cosine=cosine(a.mean(0),b.mean(0)),lo=np.nanquantile(boot,.025),hi=np.nanquantile(boot,.975),
                          norm_A2=np.linalg.norm(a.mean(0)),norm_other=np.linalg.norm(b.mean(0)),
                          magnitude_ratio=np.linalg.norm(b.mean(0))/np.linalg.norm(a.mean(0)),
                          ratio_lo=np.quantile(ratio,.025),ratio_hi=np.quantile(ratio,.975),
                          disjoint_half_median=np.nanmedian(split),disjoint_half_p025=np.nanquantile(split,.025),disjoint_half_p975=np.nanquantile(split,.975)))
    for name,rows in [('paired_shifts',shiftrows),('fidelity',fidelity),('dose_matched',dose),('similarity_sensitivity',sims)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    # Report all demand contrasts and alternate smoothing values, with same items.
    demand=[];operations=[]
    for p in sorted((src/'f1').glob('rates_*.parquet')):
        d=pd.read_parquet(p);d=d[d.source.eq('f1')];model=p.stem.removeprefix('rates_')
        for domain in ['all','math','gpqa']:
            g=d if domain=='all' else d[d.topic.eq(domain)];base=g[g.ptype.eq('solve_free')].set_index('item')
            for condition in ['solve_mcq','generate','evaluate']:
                a=g[g.ptype.eq(condition)].set_index('item');ids=base.index.intersection(a.index)
                cols=[b+'__rate' for b in BEH];x=a.loc[ids,cols].to_numpy();y=base.loc[ids,cols].to_numpy()
                for eps in [.001,.005,.01]:
                    distances=np.abs(np.log2((x+eps)/(y+eps))).mean(1);v,lo,hi=interval(distances,rng)
                    demand.append(dict(model=model,domain=domain,condition=condition,n=len(ids),metric='mean_abs_log2_ratio',epsilon=eps,estimate=v,lo=lo,hi=hi))
                v,lo,hi=interval(np.abs(x-y).mean(1),rng)
                demand.append(dict(model=model,domain=domain,condition=condition,n=len(ids),metric='mean_abs_rate_difference',epsilon=0,estimate=v,lo=lo,hi=hi))
                for j,label in enumerate(BEH):
                    v,lo,hi=interval(x[:,j]-y[:,j],rng)
                    operations.append(dict(model=model,domain=domain,condition=condition,label=label,n=len(ids),delta=v,lo=lo,hi=hi))
    pd.DataFrame(demand).to_csv(out/'matched_demand_sensitivity.csv',index=False)
    pd.DataFrame(operations).to_csv(out/'matched_demand_operations.csv',index=False)
    (out/'analysis_manifest.json').write_text(json.dumps(dict(seed=20261101,bootstrap_reps=2000,
      fidelity='Intended 16-target contrast, fixed-map predicted change, and achieved paired change. Projection is descriptive, not mediated causal effect.',
      dose='Identical 100 tasks and seed 1 at all doses; intervals exploratory.',
      duration='Maximum prefix windows; all-three-reach-window sensitivity conditions on post-treatment duration.',
      shared_baseline='Joint item bootstrap retains pairing. Disjoint-half sensitivity uses separate baseline observations; its percentiles describe random splits, not a CI.',
      demand='Includes evaluation condition; alternative smoothing and absolute-rate distances; no repeat-noise correction for evaluation condition.'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',type=Path,required=True);a=p.parse_args();analyze(a.revision)
