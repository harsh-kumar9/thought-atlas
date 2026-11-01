"""Timing control that preserves each trace's joint label counts and length.

Uniformly reallocates the 256 possible eight-label patterns to the trace's
existing early/middle/late valid positions. Multivariate hypergeometric draws
are exactly the phase-count distribution of a within-trace row permutation.
"""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
BEH=['Question_and_Answering','Perspective_Shift','Conflict_of_Perspectives','Reconciliation','verification','backtracking','subgoal','backward_chaining']
PATTERNS=((np.arange(256)[:,None]>>np.arange(8))&1).astype(np.int64)

def phase_draw(counts,phase_sizes,rng):
    remaining=np.asarray(counts,dtype=np.int64).copy();parts=[]
    assert remaining.sum()==sum(phase_sizes)
    for n in phase_sizes[:2]:
        draw=rng.multivariate_hypergeometric(remaining,int(n)) if n else np.zeros_like(remaining)
        parts.append(draw);remaining-=draw
    return np.stack(parts+[remaining])

def export(root,out):
    import polars as pl
    folder=root/'data/v2/judge';stem='__google_gemma-4-31B-it.parquet'
    keys=['trace_id','seg_idx']
    full=pl.read_parquet(folder/('trackB_full'+stem),columns=keys+['norm_pos','kim_parsed','gandhi_parsed']+BEH)
    iso=pl.read_parquet(folder/('trackB_isolated'+stem),columns=keys+['kim_parsed','gandhi_parsed'])
    full=full.join(iso,on=keys,suffix='_iso',validate='1:1').filter(pl.col('kim_parsed')&pl.col('gandhi_parsed')&pl.col('kim_parsed_iso')&pl.col('gandhi_parsed_iso'))
    frame=full.to_pandas();ids,codes=np.unique(frame.trace_id.to_numpy(str),return_inverse=True)
    labels=frame[BEH].to_numpy(np.int64);mask=labels@(1<<np.arange(8))
    counts=np.bincount(codes*256+mask,minlength=len(ids)*256).reshape(-1,256)
    phase=np.minimum((frame.norm_pos.to_numpy(np.float32)*3).astype(int),2)
    sizes=np.bincount(codes*3+phase,minlength=len(ids)*3).reshape(-1,3)
    observed=np.stack([np.bincount(codes*3+phase,weights=labels[:,j],minlength=len(ids)*3).reshape(-1,3) for j in range(8)],axis=2)
    np.savez_compressed(out/'joint_counts.npz',trace_id=ids,counts=counts,phase_sizes=sizes,observed=observed)
    (out/'export_manifest.json').write_text(json.dumps(dict(valid_segments=len(frame),traces=len(ids),
      source_files=[str(folder/('trackB_'+mode+stem)) for mode in ['full','isolated']],
      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))

def run_cell(model,out,repeats=100):
    sys.path.insert(0,str(ROOT))
    from src.analysis.paper.signature_analysis import FAMILIES,add_timing_contrasts,_cv_probabilities,_design,ridge_multinomial_predict,_log_loss
    data=pd.read_parquet(ROOT/'docs/paper_synthesis_2026-10-31/reframing/sources/trace_features.parquet')
    data=data[data.gen_model.eq(model)&data.n_valid.gt(0)].copy()
    splits=pd.read_parquet(ROOT/'docs/paper_synthesis_2026-10-31/sources/ada_corrected_atlas/splits.parquet')
    data=data.merge(splits[['trace_id','prompt_fold']],on='trace_id',validate='1:1').reset_index(drop=True)
    z=np.load(out/'joint_counts.npz');lookup={t:i for i,t in enumerate(z['trace_id'])};idx=np.array([lookup[t] for t in data.trace_id])
    counts=z['counts'][idx];sizes=z['phase_sizes'][idx];observed=z['observed'][idx]
    assert np.array_equal(counts.sum(1),data.n_valid.to_numpy())
    rate=(counts@PATTERNS)/counts.sum(1)[:,None]
    assert np.max(np.abs(rate-data[[b+'__rate' for b in BEH]].to_numpy()))<1e-7
    phase=np.divide(observed,sizes[:,:,None],out=np.full(observed.shape,np.nan),where=sizes[:,:,None]>0)
    for j,b in enumerate(BEH):
        assert np.allclose(phase[:,:,j],data[[b+'__'+p for p in ['early','middle','late']]].to_numpy(),equal_nan=True,atol=1e-7)
    data=add_timing_contrasts(data);fold=data.prompt_fold.to_numpy();familydata={}
    ref=pd.read_csv(ROOT/'paper/data/figure_inputs/gains.csv').set_index(['gen_model','signature_family'])
    for family,behaviors in FAMILIES.items():
        base=['log_tokens','log_segments']+[b+'__rate' for b in behaviors]
        timing=[b+'__timing_'+c for b in behaviors for c in ['middle_minus_early','late_minus_early']]
        y,pred,labels=_cv_probabilities(data,'task_type',{'amount':base,'full':base+timing},folds=5,seed=42,fold_ids=fold)
        for name in ['amount','full']:assert abs(_log_loss(y,pred[name])-ref.loc[(model,family),name+'_log_loss'])<1e-5
        base_log=np.log2(np.clip(pred['amount'][np.arange(len(y)),y],1e-12,1))
        familydata[family]=dict(base=base,timing=timing,y=y,real_gain=np.log2(np.clip(pred['full'][np.arange(len(y)),y],1e-12,1))-base_log,
          base_log=base_log,random_gain=np.zeros((repeats,len(y))))
    seed=int.from_bytes(hashlib.sha256(('within-trace|20261101|'+model).encode()).digest()[:4],'big');rng=np.random.default_rng(seed)
    for rep in range(repeats):
        perms=np.stack([phase_draw(c,n,rng)@PATTERNS for c,n in zip(counts,sizes)])
        assert np.array_equal(perms.sum(1),counts@PATTERNS)
        phase=np.divide(perms,sizes[:,:,None],out=np.full(perms.shape,np.nan,dtype=float),where=sizes[:,:,None]>0)
        randomdata=data.copy()
        for j,b in enumerate(BEH):
            for k,p in enumerate(['early','middle','late']):randomdata[b+'__'+p]=phase[:,k,j]
        randomdata=add_timing_contrasts(randomdata)
        for family,a in familydata.items():
            for f in np.unique(fold):
                train=fold!=f;test=~train;xtr,xte=_design(randomdata[train],randomdata[test],a['base']+a['timing'])
                p=ridge_multinomial_predict(xtr,a['y'][train],xte,8,l2=1.)
                a['random_gain'][rep,test]=np.log2(np.clip(p[np.arange(test.sum()),a['y'][test]],1e-12,1))-a['base_log'][test]
        if (rep+1)%20==0:print(model,rep+1,flush=True)
    for family,a in familydata.items():
        np.savez_compressed(out/f'{model}__{family}.npz',y=a['y'],real_gain=a['real_gain'],random_gain=a['random_gain'],base_gain=3+a['base_log'])
    return model

def fit(out,repeats,workers,summarize_only=False):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    sys.path.insert(0,str(ROOT))
    from src.analysis.paper.signature_analysis import MODELS
    if not summarize_only:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs=[pool.submit(run_cell,m,out,repeats) for m in MODELS]
            for job in as_completed(jobs):print('complete',job.result(),flush=True)
    sys.path.insert(0,str(ROOT/'paper/scripts'))
    from random_timing_control import summarize
    rows=summarize(out,repeats,10000,20261101)
    (out/'analysis_manifest.json').write_text(json.dumps(dict(repeats=repeats,seed=20261101,
      control='Joint within-trace label-row permutation conditional on valid positions and all eight label counts; generated by exact multivariate hypergeometric draws.',
      all_feature_and_real_score_reproduction_checks_passed=True,
      positive_bonferroni_intervals=int(sum(r['contrast_fwer_ci_low']>0 for r in rows)),
      code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        out/'joint_counts.npz',ROOT/'docs/paper_synthesis_2026-10-31/reframing/sources/trace_features.parquet',
        ROOT/'docs/paper_synthesis_2026-10-31/sources/ada_corrected_atlas/splits.parquet']},
      outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.npz'))},
      recovery='Numerical fits and bootstrap summaries completed before a NumPy-integer metadata serialization error. Summary regenerated from preserved arrays after converting the metadata count to int; numerical fits unchanged.' if summarize_only else None),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['export','fit','summarize']);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--repeats',type=int,default=100);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(exist_ok=True,parents=True)
    export(a.root,a.out) if a.stage=='export' else fit(a.out,a.repeats,a.workers,a.stage=='summarize')
