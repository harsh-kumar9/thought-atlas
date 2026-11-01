"""All-allocated paired confirmation and separately measured behavioral fidelity."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import polars as pl
from confirmation import ARMS,N,BUDGET,SEED
from rl_repair import sha,atomic_json

def holm(ps):
    ps=np.asarray(ps,float);order=np.argsort(ps);adjusted=np.empty(len(ps));ceiling=0.
    for rank,i in enumerate(order):
        ceiling=max(ceiling,(len(ps)-rank)*ps[i]);adjusted[i]=min(1.,ceiling)
    return adjusted

def paired_effects(d,reps=10000):
    assert not d.duplicated(['condition','instance_id','sample']).any()
    for _,g in d.groupby('condition'):
        assert set(g['sample'])=={0,1} and g.groupby('instance_id').size().eq(2).all()
    metrics=['natural_correct','elicited_correct','answered','completed','hard_loop','tokens']
    task=d.groupby(['condition','instance_id'])[metrics].mean().reset_index();rows=[]
    for metric in metrics:
        p=task.pivot(index='instance_id',columns='condition',values=metric).sort_index()
        assert set(p.columns)==set(ARMS) and not p.isna().any().any()
        for arm in ARMS[1:]:
            key=int.from_bytes(__import__('hashlib').sha256((metric+'|'+arm).encode()).digest()[:8],'big')
            rng=np.random.default_rng(key);delta=(p[arm]-p.baseline).to_numpy()
            boots=delta[rng.integers(0,len(delta),(reps,len(delta)))].mean(1)
            signs=rng.choice([-1.,1.],(reps,len(delta)))
            pvalue=(1+np.sum(np.abs((signs*delta).mean(1))>=abs(delta.mean())))/(reps+1)
            rows.append(dict(metric=metric,condition=arm,n_tasks=len(p),baseline=p.baseline.mean(),intervention=p[arm].mean(),
                delta=delta.mean(),lo=np.quantile(boots,.025),hi=np.quantile(boots,.975),p_signflip=pvalue))
    results=pd.DataFrame(rows);results['p_holm_primary']=np.nan
    primary=results.metric.eq('natural_correct');results.loc[primary,'p_holm_primary']=holm(results.loc[primary,'p_signflip'])
    return results

def outcomes(root,out):
    sys.path.insert(0,str(root));from src.utils.characterize import loop_collapse_detect
    parts=[]
    for arm in ARMS:
        assert json.loads((out/arm/'status.json').read_text())['stage']=='complete'
        tr=pd.read_parquet(out/arm/'rollouts_s001.parquet');gr=pd.read_parquet(out/arm/'rewards_s001.parquet')
        el=pd.read_parquet(out/arm/'elicitation.parquet')
        assert len(tr)==len(gr)==len(el)==N*2 and el.total_tokens.le(BUDGET).all()
        d=tr.merge(gr[['trace_id','success','grade_source']],on='trace_id',validate='one_to_one').merge(el[['trace_id','correct','own']],on='trace_id',validate='one_to_one')
        d['natural_correct']=d.success;d['elicited_correct']=d.correct;d['answered']=d.answer_text.fillna('').str.strip().ne('').astype(int)
        d['hard_loop']=d.full_text.map(loop_collapse_detect).astype(int);d['tokens']=d.gen_token_ids.map(len);d['condition']=arm
        parts.append(d)
    d=pd.concat(parts);paired_effects(d).to_csv(out/'paired_effects.csv',index=False)
    d.groupby(['condition','grade_source']).size().rename('attempts').reset_index().to_csv(out/'grade_sources.csv',index=False)
    d[['trace_id','instance_id','condition','sample','natural_correct','elicited_correct','answered','completed','hard_loop','tokens']].to_csv(out/'outcomes.csv',index=False)
    atomic_json(dict(tasks=N,samples=2,budget=BUDGET,protocol_sha256=sha(out/'protocol.json'),code_sha256=sha(Path(__file__)),
        primary='Two natural-completion accuracy contrasts with Holm-adjusted two-sided paired sign-flip tests,10000 draws. Intervals10000 paired task bootstraps after two-sample averaging.',
        secondary='Elicited accuracy and other endpoints descriptive; no overall family-wise claim. Natural extraction and direct forced-answer grading sources remain distinct.',
        outcome_files={str(p.relative_to(out)):sha(p) for arm in ARMS for p in [out/arm/'rollouts_s001.parquet',out/arm/'rewards_s001.parquet',out/arm/'elicitation.parquet']}),out/'analysis_manifest.json')

def behavior(root,out):
    sys.path.insert(0,str(root))
    from mode_control.mc.common import BEH,load_tokenizer
    from mode_control.mc.phenotype import _segments_with_offsets
    from mode_control.mc.signature_ctrl import segment_targets,TARGET_NAMES
    dest=out/'behavior';status=json.loads((dest/'status_gemma.json').read_text());assert status['completed']==status['requested']
    labels=pl.read_parquet(dest/'labels_gemma.parquet');key=pd.read_parquet(dest/'sample_key.parquet').set_index('trace_id')
    tr=pl.read_parquet(dest/'blinded_traces.parquet');tok=load_tokenizer();rows=[];missing=[]
    groups={k[0] if isinstance(k,tuple) else k:g.sort('seg_idx') for k,g in labels.group_by('trace_id')}
    for r in tr.iter_rows(named=True):
        g=groups.get(r['trace_id'])
        if g is None:missing.append(dict(trace_id=r['trace_id'],reason='no_annotation'));continue
        text=r['reasoning_text_for_analysis'];starts=_segments_with_offsets(tok,text,int(g['n_segments'][0]))
        if starts is None or len(starts)!=g.height:missing.append(dict(trace_id=r['trace_id'],reason='alignment'));continue
        valid=(g['kim_parsed'].fill_null(False)&g['gandhi_parsed'].fill_null(False)).to_numpy()
        values=g.select(BEH).fill_null(0).to_numpy().astype(float);ix=np.flatnonzero(valid)
        if not len(ix):missing.append(dict(trace_id=r['trace_id'],reason='no_valid_segments'));continue
        ends=np.r_[np.asarray(starts)[1:],len(tok.encode(text,add_special_tokens=False))]
        kept=ix[ends[ix]-1<BUDGET]
        if len(kept)>80:kept=kept[np.linspace(0,len(kept)-1,80).round().astype(int)]
        if not len(kept):missing.append(dict(trace_id=r['trace_id'],reason='no_valid_endpoints'));continue
        tg=segment_targets(values[ix]);mapping={int(v):i for i,v in enumerate(ix)}
        average=tg[[mapping[int(v)] for v in kept],:16].mean(0)
        rows.append(dict(trace_id=r['trace_id'],**key.loc[r['trace_id']].to_dict(),valid_segments=int(valid.sum()),**dict(zip(TARGET_NAMES[:16],average))))
    profiles=pd.DataFrame(rows);profiles.to_csv(dest/'profiles.csv',index=False);pd.DataFrame(missing).to_csv(dest/'missing.csv',index=False)
    target=list(TARGET_NAMES[:16]);base=profiles[profiles.condition.eq('baseline')].set_index('instance_id');results=[]
    for arm in ARMS[1:]:
        a=profiles[profiles.condition.eq(arm)].set_index('instance_id');ids=sorted(set(a.index)&set(base.index))
        change=(a.loc[ids,target]-base.loc[ids,target]).to_numpy();rng=np.random.default_rng(SEED)
        boots=change[rng.integers(0,len(change),(10000,len(change)))].mean(1)
        for j,t in enumerate(target):results.append(dict(condition=arm,target=t,n=len(ids),delta=change[:,j].mean(),lo=np.quantile(boots[:,j],.025),hi=np.quantile(boots[:,j],.975)))
        # The domain contrast has a saved intended16-target change. Do not invent
        # a matching target for an archetype with a different construction.
        if arm=='sig_moral':
            z=np.load(root/'mode_control/data/vsteer/vec_sig_moral.npz',allow_pickle=True);wanted=np.asarray(json.loads(str(z['meta']))['dphi16'])
            cosine=lambda x:np.dot(x,wanted)/(np.linalg.norm(x)*np.linalg.norm(wanted)) if np.linalg.norm(x)>0 else np.nan
            cs=np.asarray([cosine(x) for x in boots]);projection=boots@wanted/(wanted@wanted)
            atomic_json(dict(condition=arm,n_tasks=len(ids),intended_cosine=cosine(change.mean(0)),cosine_lo=float(np.nanquantile(cs,.025)),cosine_hi=float(np.nanquantile(cs,.975)),
                intended_projection=float(change.mean(0)@wanted/(wanted@wanted)),projection_lo=float(np.quantile(projection,.025)),projection_hi=float(np.quantile(projection,.975)),
                scope='Descriptive alignment and magnitude on fixed first64 tasks/sample0; validity and missingness retained; no mediated-causal-effect claim.'),dest/'fidelity.json')
    pd.DataFrame(results).to_csv(dest/'paired_shifts.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['outcomes','behavior']);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a.root.resolve(),a.out.resolve())
