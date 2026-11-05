"""Verify frozen three-judge coverage and independently reproduce reported point estimates."""
from pathlib import Path
import hashlib,json,datetime,subprocess
import numpy as np
import pandas as pd
R=Path('/ada1/u/harsh/thought-atlas-reruns/arr_repair_20261101');O=R/'results/multijudge_analysis_indexed'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((O/'analysis_manifest.json').read_text());assert manifest['partial'] is False
key=pd.read_parquet(O/'sample_key.parquet');expected=set(key.loc[key.analysis_tokens.gt(0),'trace_id']);assert len(key)==1381 and len(expected)==1370
accounting=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-n','-P','-j','437190,435691,435756,435757','-o','JobID,State,ExitCode'],text=True).strip().splitlines()
assert len(accounting)==4 and all(x.endswith('|COMPLETED|0:0') for x in accounting)
beh=['Question_and_Answering','Perspective_Shift','Conflict_of_Perspectives','Reconciliation','verification','backtracking','subgoal','backward_chaining'];labels={};checks={}
for j in ['gemma','qwen','llama']:
 p=O/f'labels_{j}.parquet';d=pd.read_parquet(p);assert set(d.trace_id)==expected and not d.duplicated(['trace_id','seg_idx']).any();assert len(d)==105302
 g=d.groupby('trace_id');assert (g.size()==g.n_segments.first()).all();assert (g.seg_idx.nunique()==g.n_segments.first()).all();assert not d.judge_context_truncated.any()
 origin=R/('results/multijudge_llama_indexed' if j=='llama' else 'results/multijudge')/p.name;assert sha(origin)==sha(p)
 checks[j]={'segments':len(d),'traces':d.trace_id.nunique(),'labels_sha256':sha(p),'kim_parse_rate':float(d.kim_parsed.mean()),'cognitive_parse_rate':float(d.gandhi_parsed.mean()),'context_truncation':0}
 for b in beh:d.loc[~d['kim_parsed' if b in beh[:4] else 'gandhi_parsed'].fillna(False),b]=np.nan
 labels[j]=d.merge(key[['trace_id','source','task_type']],on='trace_id',validate='many_to_one')
base=labels['gemma'][['trace_id','seg_idx','n_segments']].sort_values(['trace_id','seg_idx']).reset_index(drop=True)
for d in labels.values():assert base.equals(d[['trace_id','seg_idx','n_segments']].sort_values(['trace_id','seg_idx']).reset_index(drop=True))
ag=pd.read_csv(O/'pairwise_agreement.csv')
for _,r in ag.iterrows():
 a=labels[r.judge_a];b=labels[r.judge_b];z=a[['trace_id','seg_idx','source','task_type',r.label]].merge(b[['trace_id','seg_idx',r.label]],on=['trace_id','seg_idx'],validate='one_to_one',suffixes=('_a','_b'));z=z[z.source.eq(r.source)]
 if r.domain!='all':z=z[z.task_type.eq(r.domain)]
 z=z.dropna(subset=[r.label+'_a',r.label+'_b']);z['cell']=(2*z[r.label+'_a']+z[r.label+'_b']).astype(int)
 ct=pd.crosstab(z.trace_id,z.cell).reindex(columns=range(4),fill_value=0);assert len(ct)==r.traces and len(z)==r.valid_paired_segments
 assert np.array_equal(ct.sum().to_numpy(),[r['n_00'],r['n_01'],r['n_10'],r['n_11']])
 a,b,c,d=(ct.div(ct.sum(axis=1),axis=0).sum()).to_numpy();n=a+b+c+d;po=(a+d)/n;pe=((a+b)*(a+c)+(c+d)*(b+d))/n**2
 metrics={'agreement':po,'positive_agreement':2*d/(2*d+b+c) if 2*d+b+c else np.nan,'negative_agreement':2*a/(2*a+b+c) if 2*a+b+c else np.nan,'kappa':(po-pe)/(1-pe) if pe<1-1e-12 else np.nan}
 for k,v in metrics.items():assert np.isclose(r[k],v,atol=1e-12,equal_nan=True)
 for k in metrics:
  lo,hi=r[k+'_lo'],r[k+'_hi']
  if np.isfinite(lo):assert lo<=hi and lo>=(-1 if k=='kappa' else 0)-1e-12 and hi<=1+1e-12
profiles=pd.read_parquet(O/'trace_profiles.parquet');paired=pd.read_csv(O/'paired_behavior_shifts.csv')
for _,r in paired.iterrows():
 g=profiles
 for col in ['source','judge','scope','gen_model','task_type','label']:g=g[g[col].eq(r[col])]
 p=g.pivot(index='base_item' if r.source=='demand' else 'instance_id',columns='condition',values='rate');delta=(p[r.condition]-p[r.reference]).dropna();assert len(delta)==r.n and np.isclose(delta.mean(),r.delta,atol=1e-12);assert -1<=r.lo<=r.hi<=1
atlas=pd.read_csv(O/'atlas_weighted_profiles.csv')
for _,r in atlas.iterrows():
 g=profiles[profiles.source.eq('atlas')]
 for col in ['judge','scope','gen_model','task_type','label']:g=g[g[col].eq(r[col])]
 g=g.dropna(subset=['rate']);w=1/g.inclusion_probability;assert len(g)==r.n and np.isclose(np.average(g.rate,weights=w),r.rate,atol=1e-12)
contrasts=pd.read_csv(O/'atlas_contrasts.csv')
for _,r in contrasts.iterrows():
 a=atlas[atlas.judge.eq(r.judge)&atlas.scope.eq(r.scope)&atlas.label.eq(r.label)].pivot(index='gen_model',columns='task_type',values='rate');delta=(a[r.high]-a[r.low]).dropna();assert len(delta)==r.models and int(delta.gt(0).sum())==r.models_positive and np.isclose(delta.mean(),r.mean_difference,atol=1e-12)
result={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accounting':accounting,'allocated_traces':1381,'nonempty_traces':1370,'empty_prefixes':11,'judges':checks,'exact_frozen_coverage':True,'same_segment_grid':True,'copied_labels_hash_match':True,'agreement_rows_reproduced':len(ag),'paired_shift_means_reproduced':len(paired),'weighted_rates_reproduced':len(atlas),'contrast_means_reproduced':len(contrasts),'interval_checks':'Finite bounds ordered and in valid range; original deterministic bootstrap used, not independently regenerated here.','files':{p.name:sha(p) for p in O.iterdir() if p.is_file()},'interpretation':'Cross-family robustness; no semantic ground truth or majority-gold interpretation.'}
(O/'completion_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
