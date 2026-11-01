"""Prompt-disjoint validation of the historical layer-16, rank-8 readout.

Never overwrites a historical map. The fixed setting was selected previously;
these are retrospective validation estimates, not nested model-selection scores.
All normalization and target reference means are estimated on training rows.
An unseen domain uses the training grand mean, never its held-out labels.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve

BEH = ['Question_and_Answering','Perspective_Shift','Conflict_of_Perspectives',
       'Reconciliation','verification','backtracking','subgoal','backward_chaining']
TARGETS = [f'{part}_{b}' for part in ['future','next4'] for b in BEH]
RUNS = [f'corpus1_{d}_{s}' for d in ['math','idea','gpqa','planning'] for s in ['a','b']]

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()

def fold_ids(keys, n=5):
    return np.array([int.from_bytes(hashlib.sha256(('20261101|'+str(k)).encode()).digest()[:8],'big')%n for k in keys])

def references(y, domains):
    means={d:y[domains==d].mean(0) for d in np.unique(domains)}
    return means,y.mean(0)

def reference_rows(domains,means,fallback):
    return np.stack([means.get(d,fallback) for d in domains])

def fit_rrr(x,y,rank=8,lam=1e5):
    mu=x.mean(0); sd=x.std(0)+1e-3
    z=(x-mu)/sd; ym=y.mean(0); yc=y-ym
    gram=z.T@z; cross=z.T@yc
    reg=gram.copy(); reg.flat[::len(reg)+1]+=lam
    w=solve(reg,cross,assume_a='pos',check_finite=False)
    _,v=np.linalg.eigh(w.T@gram@w)
    vr=v[:,-min(rank,y.shape[1]):]; w=w@vr@vr.T
    return dict(mu=mu,sd=sd,w=w,b=ym)

def predict(model,x):
    return ((x-model['mu'])/model['sd'])@model['w']+model['b']

def prepare(root,out):
    sig=root/'mode_control/data/sigctrl'; xs=[]; ys=[]; metas=[]; inputs=[]
    for run in RUNS:
        ap=sig/f'{run}__acts.npz'; mp=sig/f'{run}__meta.parquet'
        with np.load(ap) as z:
            li=z['layers'].tolist().index(16)
            xs.append(z['H'][:,li,:].astype(np.float32)); ys.append(z['targets'].astype(np.float64))
        m=pd.read_parquet(mp); m['source_run']=run; metas.append(m)
        inputs.extend([dict(path=str(p),sha256=digest(p)) for p in [ap,mp]])
        print('loaded',run,len(m),flush=True)
    m=pd.concat(metas,ignore_index=True); x=np.concatenate(xs); y=np.concatenate(ys)
    assert len(x)==len(y)==len(m)
    assert not m.duplicated(['trace_id','seg_idx']).any() if 'seg_idx' in m else True
    m['prompt_key']=m.task_type.astype(str)+'|'+m.instance_id.astype(str)
    m['fold']=fold_ids(m.prompt_key)
    historical=sig/'map_corpus1_dm8_L16_r8.npz'
    with np.load(historical) as h:
        names=h['domain_names'].tolist(); expected=h['domain_means']
        actual=np.stack([y[m.task_type.eq(d)].mean(0) for d in names])
        delta=float(np.max(np.abs(expected-actual)))
    # Matching recorded reference means identifies the fitted corpus independently of prose.
    if delta>1e-6: raise ValueError(f'Corpus does not reproduce historical map means: {delta}')
    np.save(out/'X.npy',x); np.save(out/'Y.npy',y); m.to_parquet(out/'meta.parquet',index=False)
    manifest=dict(stage='prepared',runs=RUNS,rows=len(m),prompts=m.prompt_key.nunique(),
      traces=m.trace_id.nunique(),layer=16,rank=8,ridge=1e5,fold_seed=20261101,
      historical_map_sha256=digest(historical),historical_domain_mean_max_error=delta,
      inputs=inputs,script_sha256=digest(Path(__file__)),
      interpretation='Retrospective validation at a previously selected setting; no historical map modified.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))

def fit_fold(out,index):
    dest=out/f'fold_{index}.parquet'
    if dest.exists(): print('complete',dest); return
    m=pd.read_parquet(out/'meta.parquet'); dom=m.task_type.to_numpy()
    if index<5:
        test=m.fold.to_numpy()==index; split='prompt_cv'; held=f'fold{index}'
    else:
        held=sorted(m.task_type.unique())[index-5]; test=dom==held; split='heldout_domain'
    train=~test
    assert not set(m.loc[train,'prompt_key'])&set(m.loc[test,'prompt_key'])
    x=np.load(out/'X.npy',mmap_mode='r'); y=np.load(out/'Y.npy',mmap_mode='r')
    means,fallback=references(y[train],dom[train])
    ref_train=reference_rows(dom[train],means,fallback); ref_test=reference_rows(dom[test],means,fallback)
    model=fit_rrr(np.asarray(x[train],dtype=np.float64),y[train]-ref_train)
    predictions=predict(model,np.asarray(x[test],dtype=np.float64))+ref_test
    data=m.loc[test,['trace_id','instance_id','task_type','prompt_key']].copy()
    data['split']=split; data['held_out']=held
    for j,t in enumerate(TARGETS):
        data['y_'+t]=y[test,j]; data['pred_'+t]=predictions[:,j]; data['base_'+t]=ref_test[:,j]
    tmp=dest.with_suffix('.tmp.parquet'); data.to_parquet(tmp,index=False); tmp.replace(dest)
    (out/f'fold_{index}.json').write_text(json.dumps(dict(split=split,held_out=held,n_train=int(train.sum()),
        n_test=int(test.sum()),train_prompts=m.loc[train,'prompt_key'].nunique(),
        test_prompts=m.loc[test,'prompt_key'].nunique(),overlap_prompts=0,
        train_means={k:v.tolist() for k,v in means.items()},unseen_domain_reference='training grand mean',
        script_sha256=digest(Path(__file__))),indent=2))
    print('complete',index,split,held,flush=True)

def summarize(out):
    assert all((out/f'fold_{i}.parquet').exists() for i in range(9))
    d=pd.concat([pd.read_parquet(out/f'fold_{i}.parquet') for i in range(9)],ignore_index=True)
    rows=[]; rng=np.random.default_rng(20261101)
    for split in d.split.unique():
      for domain in ['all']+sorted(d.task_type.unique()):
        g=d[d.split.eq(split) & (True if domain=='all' else d.task_type.eq(domain))]
        for t in TARGETS:
          y=g['y_'+t].to_numpy(); p=g['pred_'+t].to_numpy(); b=g['base_'+t].to_numpy()
          err=pd.DataFrame({'prompt':g.prompt_key.to_numpy(),'trace':g.trace_id.to_numpy(),
               'model':(y-p)**2,'baseline':(y-b)**2})
          e=err.groupby(['prompt','trace'])[['model','baseline']].mean().groupby('prompt').mean()
          a=e.to_numpy(); bs=[]
          for _ in range(1000):
              v=a[rng.integers(0,len(a),len(a))].mean(0)
              bs.append(1-v[0]/v[1] if v[1]>1e-12 else np.nan)
          v=a.mean(0); skill=1-v[0]/v[1] if v[1]>1e-12 else np.nan
          rows.append(dict(split=split,domain=domain,target=t,n_prompts=len(a),n_positions=len(g),
            prompt_weighted_mse=float(v[0]),baseline_mse=float(v[1]),skill_over_reference=skill,
            lo=float(np.nanquantile(bs,.025)),hi=float(np.nanquantile(bs,.975)),
            position_r2=float(1-((y-p)**2).sum()/(((y-y.mean())**2).sum()+1e-12)),
            position_correlation=float(np.corrcoef(y,p)[0,1])))
    pd.DataFrame(rows).to_csv(out/'validation_summary.csv',index=False)
    print(pd.DataFrame(rows).query('domain=="all"').groupby('split')[['skill_over_reference','position_r2']].mean())

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['prepare','fold','summarize'])
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--fold',type=int,default=0); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    if a.stage=='prepare': prepare(a.root,a.out)
    elif a.stage=='fold': fit_fold(a.out,a.fold)
    else:summarize(a.out)
