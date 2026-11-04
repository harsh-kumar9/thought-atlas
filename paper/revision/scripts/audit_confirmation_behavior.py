"""Audit completed frozen behavior outputs; never refit or alter annotations."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(root,out):
    key=pd.read_parquet(out/'sample_key.parquet');profiles=pd.read_csv(out/'profiles.csv')
    labels=pd.read_parquet(out/'labels_gemma.parquet');tr=pd.read_parquet(out/'blinded_traces.parquet')
    manifest=json.loads((out/'manifest.json').read_text())
    assert sha(out/'blinded_traces.parquet')==manifest['input_sha256']
    assert len(key)==len(profiles)==len(tr)==192 and profiles.trace_id.is_unique
    assert set(key.trace_id)==set(profiles.trace_id)==set(tr.trace_id)==set(labels.trace_id)
    assert not labels.duplicated(['trace_id','seg_idx']).any()
    assert not labels.judge_context_truncated.any()
    assert labels.kim_parsed.all() and labels.gandhi_parsed.all()
    assert (out/'missing.csv').read_text().strip()==''
    chosen=set(pd.read_parquet(out.parent/'tasks.parquet').head(64).instance_id)
    for arm,g in key.groupby('condition'):assert len(g)==64 and set(g.instance_id)==chosen
    targets=[c for c in profiles if c.startswith(('future_','next4_'))];assert len(targets)==16
    saved=pd.read_csv(out/'paired_shifts.csv');base=profiles[profiles.condition.eq('baseline')].set_index('instance_id')
    for arm in ['ol_math_A2','sig_moral']:
        a=profiles[profiles.condition.eq(arm)].set_index('instance_id');ids=sorted(chosen)
        changes=(a.loc[ids,targets]-base.loc[ids,targets]).to_numpy()
        rng=np.random.default_rng(20261103);boots=changes[rng.integers(0,64,(10000,64))].mean(1)
        for j,t in enumerate(targets):
            s=saved[saved.condition.eq(arm)&saved.target.eq(t)].iloc[0]
            assert s.n==64
            assert np.allclose([s.delta,s.lo,s.hi],[changes[:,j].mean(),*np.quantile(boots[:,j],[.025,.975])],atol=1e-6)
        if arm=='sig_moral':
            z=np.load(root/'mode_control/data/vsteer/vec_sig_moral.npz',allow_pickle=True)
            wanted=np.array(json.loads(str(z['meta']))['dphi16']);mean=changes.mean(0)
            cs=boots@wanted/(np.linalg.norm(boots,axis=1)*np.linalg.norm(wanted));ps=boots@wanted/(wanted@wanted)
            result=dict(intended_cosine=float(mean@wanted/(np.linalg.norm(mean)*np.linalg.norm(wanted))),cosine_lo=np.quantile(cs,.025),cosine_hi=np.quantile(cs,.975),intended_projection=float(mean@wanted/(wanted@wanted)),projection_lo=np.quantile(ps,.025),projection_hi=np.quantile(ps,.975))
            f=json.loads((out/'fidelity.json').read_text())
            for k,v in result.items():assert np.isclose(v,f[k],atol=1e-6),k
    report=dict(passed=True,traces=192,tasks_per_arm=64,segments=len(labels),kim_parse_rate=1.,gandhi_parse_rate=1.,context_truncated=0,missing_profiles=0,minimum_valid_segments=int(profiles.valid_segments.min()),paired_shift_cells=len(saved),bootstrap_reproduced=10000,fidelity_reproduced=True,scope='All paired point estimates and intervals reproduced from exported profiles; descriptive fidelity, not semantic ground truth or causal mediation.',files={n:sha(out/n) for n in ['manifest.json','labels_gemma.parquet','sample_key.parquet','profiles.csv','paired_shifts.csv','fidelity.json']},audit_code_sha256=sha(Path(__file__)))
    (out/'output_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit(a.root,a.out)
