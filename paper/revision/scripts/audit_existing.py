"""Export auditable denominators, prompt overlaps, controls, and grade changes on Ada.
Read-only against the historical repository; outputs are written to --out only.
"""
from pathlib import Path
import argparse,hashlib,json,sys,re,itertools
import numpy as np
import pandas as pd
import polars as pl

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def paired_budget(data,reps=2000):
 """Stable group/item ordering and keyed random streams make bootstrap output reproducible."""
 d=data.to_pandas() if hasattr(data,'to_pandas') else data
 task=d.groupby(['task_type','condition','budget','instance_id'])[['correct','answered','tokens']].mean().reset_index()
 rows=[]
 for (target,condition,budget),g in task.groupby(['task_type','condition','budget'],sort=True):
  if condition=='baseline':continue
  base=task[task.task_type.eq(target)&task.condition.eq('baseline')&task.budget.eq(budget)].set_index('instance_id')
  g=g.set_index('instance_id').sort_index();assert set(g.index)<=set(base.index)
  base=base.loc[g.index];rec=dict(task_type=target,condition=condition,budget=budget,n_tasks=len(g),reference='baseline')
  for metric in ['correct','answered','tokens']:
   delta=(g[metric]-base[metric]).to_numpy()
   seed=int.from_bytes(hashlib.sha256(f'20261101|{target}|{condition}|{budget}|{metric}'.encode()).digest()[:8],'big')
   rng=np.random.default_rng(seed);boot=delta[rng.integers(0,len(delta),(reps,len(delta)))].mean(1)
   rec.update({metric:g[metric].mean(),metric+'_ref':base[metric].mean(),'d_'+metric:delta.mean(),
               'd_'+metric+'_lo':np.quantile(boot,.025),'d_'+metric+'_hi':np.quantile(boot,.975)})
  rows.append(rec)
 return pd.DataFrame(rows)

def main(root,out):
 out.mkdir(exist_ok=True,parents=True);sys.path.insert(0,str(root))
 from mode_control.mc.budget import load_grades
 from mode_control.mc.common import DATA
 sources=[]
 def record(p,hash_file=True):
  sources.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p) if hash_file else None,
                      hash_status='sha256' if hash_file else 'metadata only; not a content identity'))
 def write(d,name):d.to_csv(out/name,index=False)
 # Actual task membership, not a prose inference about splits.
 groups={}; ledger=[];config_rows=[]
 runs=[f'corpus1_{d}_{s}' for d in ['math','idea','gpqa','planning','moral'] for s in ['a','b']]
 runs+=['sigpilot_math','sigpilot_idea','sigpilot2_math','sigpilot2_idea','sigpilot3_math',
        'p1_sealval','p1_budget','p2_gpqa','p2_planning','p4_eval2','p4_eval3',
        'tx_math','tx_gpqa','tx_planning','tx_idea','tx_moral','cd_countdown','cd_math','power1','sched1']
 for run in runs:
  rd=DATA/'runs'/run;tp=rd/'tasks.parquet'
  if not tp.exists():continue
  d=pd.read_parquet(tp);record(tp)
  groups[run]=set(d.instance_id.astype(str))
  for _,r in d.iterrows():
   raw=str(r.get('prompt',''));norm=re.sub(r'\s+',' ',raw).strip()
   ledger.append(dict(component=run,instance_id=str(r.instance_id),task_type=r.get('task_type',''),
      prompt_sha256=hashlib.sha256(norm.encode()).hexdigest()))
  cfg=rd/'run.json'
  if cfg.exists():
   record(cfg);c=json.loads(cfg.read_text());(out/'run_configs').mkdir(exist_ok=True)
   (out/'run_configs'/f'{run}.json').write_text(json.dumps(c,indent=2))
   config_rows.append(dict(run=run,n_tasks=len(d),task_seed=c.get('task_seed'),
      max_new_tokens=c.get('max_new_tokens'),config_sha256=sha(cfg),conditions=json.dumps(c.get('conditions',[]))))
 train=pd.read_parquet(root/'data/v2/tasks/math.parquet')
 groups['rl_training']=set(train.instance_id.astype(str))-groups['sigpilot3_math']
 for k in ['zero8k_T','a2init8k_T']:
  rd=DATA/'rlbias'/k;files=sorted(x.name for x in rd.iterdir())
  (out/f'rl_{k}_files.json').write_text(json.dumps(files,indent=2))
  for p in [rd/'config.json',rd/'metrics.jsonl']:
   if p.exists():record(p);(out/p.with_name(f'{k}_{p.name}').name).write_bytes(p.read_bytes())
 write(pd.DataFrame(ledger),'prompt_ledger.csv');write(pd.DataFrame(config_rows),'run_inventory.csv')
 overlaps=[dict(component_a=a,component_b=b,n_a=len(groups[a]),n_b=len(groups[b]),
      shared_instance_ids=len(groups[a]&groups[b])) for a,b in itertools.combinations(groups,2)]
 write(pd.DataFrame(overlaps),'prompt_overlaps.csv')
 content=pd.DataFrame(ledger).groupby('prompt_sha256').agg(n_components=('component','nunique'),
      components=('component',lambda x:';'.join(sorted(set(x)))),n_ids=('instance_id','nunique')).reset_index()
 write(content.query('n_components>1'),'content_overlaps.csv')
 # Reconcile all natural-completion arms, including previously omitted controls.
 grades=load_grades();grade_audit=[];all_parts=[];all_changes=[]
 for p in sorted(DATA.glob('extract_free*/grades.parquet')):record(p)
 targets={'p1_budget':'math','p4_eval2':'math','p4_eval3':'math','tx_math':'math',
          'p2_gpqa':'gpqa','tx_gpqa':'gpqa','p2_planning':'planning','tx_planning':'planning'}
 for run,target in targets.items():
  rd=DATA/'runs'/run;bp=rd/'analysis/budget_outcomes.parquet'
  if not bp.exists():continue
  old=pl.read_parquet(bp);record(bp)
  tr=[]
  for f in sorted(rd.glob('traces_*.parquet')):
   cols=pl.read_parquet_schema(f)
   keep=[c for c in ['trace_id','condition','instance_id','seed','n_new_tokens','run_fingerprint'] if c in cols]
   tr.append(pl.read_parquet(f,columns=keep));record(f,False)
  tr=pl.concat(tr,how='diagonal_relaxed')
  missing=set(old['condition'])-set(tr['condition'])
  shared=DATA/'runs'/({'math':'p1_budget','gpqa':'p2_gpqa','planning':'p2_planning'}[target])
  for arm in sorted(missing):
   candidates=[shared/f'traces_{arm}.parquet']+[DATA/'runs'/r/f'traces_{arm}.parquet' for r,t in targets.items() if t==target]
   f=next((p for p in candidates if p.exists()),None)
   assert f is not None,f'Missing borrowed arm {run}:{arm}'
   add=pl.read_parquet(f,columns=tr.columns)
   if 'run_fingerprint' in tr.columns:
    assert set(add['run_fingerprint'])==set(tr['run_fingerprint']),f'Unpaired borrowed arm {run}:{arm}'
   tr=pl.concat([tr,add],how='diagonal_relaxed');record(f,False)
  d=old.join(tr,on=['condition','instance_id','seed'],validate='m:1')
  assert d.height==old.height
  recs=d.to_dicts();changes=[];values=[]
  for r in recs:
   g=grades.get(r['trace_id']);use=r['n_new_tokens']<=r['budget'] and g is not None
   value=int(g) if use else r['correct'];values.append(value)
   if value!=r['correct']:changes.append(dict(run=run,trace_id=r['trace_id'],instance_id=r['instance_id'],
       condition=r['condition'],seed=r['seed'],budget=r['budget'],old=r['correct'],new=value))
  if changes:
   write(pd.DataFrame(changes),f'grade_changes_{run}.csv')
   all_changes.extend(changes)
  grade_audit.append(dict(run=run,cutoff_rows=len(recs),changed_cutoff_rows=len(changes),
    changed_unique_traces=len({r['trace_id'] for r in changes}),
    changed_unique_problems=len({r['instance_id'] for r in changes})))
  d=d.with_columns(pl.Series('correct',values),pl.lit(run).alias('source_run')).select(old.columns+['source_run'])
  all_parts.append(d)
 write(pd.DataFrame(grade_audit),'grade_change_counts.csv')
 if all_changes:
  write(pd.DataFrame(all_changes).groupby(['run','condition']).agg(changed_cutoff_rows=('trace_id','size'),
      changed_unique_traces=('trace_id','nunique'),changed_unique_problems=('instance_id','nunique')).reset_index(),
      'grade_changes_by_condition.csv')
 d=pl.concat(all_parts,how='diagonal_relaxed')
 # Reused arms are required to agree. Keep their first occurrence by explicit run priority.
 keys=['task_type','condition','instance_id','seed','budget']
 dup=d.group_by(keys).agg(pl.col('correct').n_unique().alias('n_correct'),pl.col('answered').n_unique().alias('n_answered'))
 assert dup.filter((pl.col('n_correct')>1)|(pl.col('n_answered')>1)).height==0
 d=d.unique(keys,keep='first',maintain_order=True);d.write_parquet(out/'budget_outcomes.parquet')
 summary=d.group_by('task_type','condition','budget').agg(pl.len().alias('n'),pl.col('correct').mean(),
       pl.col('answered').mean(),pl.col('tokens').mean()).sort('task_type','condition','budget')
 summary.write_csv(out/'budget_summary.csv')
 paired_budget(d).to_csv(out/'budget_paired.csv',index=False)
 # Atlas coverage joins attempted, completed, valid segment counts, and included traces.
 full=pl.read_parquet(root/'data/v2/judge/trackB_full__google_gemma-4-31B-it.parquet')
 iso=pl.read_parquet(root/'data/v2/judge/trackB_isolated__google_gemma-4-31B-it.parquet')
 keys=['trace_id','seg_idx']
 valid=full.select(keys+['kim_parsed','gandhi_parsed']).join(iso.select(keys+['kim_parsed','gandhi_parsed']),on=keys,suffix='_iso',validate='1:1')
 valid=valid.with_columns((pl.col('kim_parsed')&pl.col('gandhi_parsed')&pl.col('kim_parsed_iso')&pl.col('gandhi_parsed_iso')).alias('valid'))
 seg=valid.group_by('trace_id').agg(pl.len().alias('annotated_segments'),pl.col('valid').sum().alias('valid_segments'))
 cover=[]
 for model in ['reasoner','gemma4_e4b','gemma4_12b','gemma4_31b','qwen35_4b','qwen35_9b','qwen35_27b']:
  p=root/f'data/v2/traces/traces_{model}.parquet'
  t=pl.read_parquet(p,columns=['trace_id','task_type','completed']).join(seg,on='trace_id',how='left').fill_null(0)
  for (dom,),g in t.group_by('task_type'):
   cover.append(dict(model=model,domain=dom,attempted=g.height,completed=int(g['completed'].sum()),
     annotated=int((g['annotated_segments']>0).sum()),usable=int((g['valid_segments']>0).sum()),
     usable_incomplete=g.filter(~pl.col('completed')&(pl.col('valid_segments')>0)).height,
     no_annotation=g.filter(pl.col('annotated_segments')==0).height,
     all_annotation_invalid=g.filter((pl.col('annotated_segments')>0)&(pl.col('valid_segments')==0)).height,
     annotated_segments=int(g['annotated_segments'].sum()),valid_segments=int(g['valid_segments'].sum()),
     retained_segment_fraction=float(g['valid_segments'].sum()/max(g['annotated_segments'].sum(),1))))
 write(pd.DataFrame(cover),'coverage.csv')
 for f in ['mode_control/mc/rlbias.py','mode_control/mc/archetypes.py','mode_control/mc/signature_ctrl.py',
           'src/judge/prompts/kim_per_sentence.txt','src/judge/prompts/gandhi_per_sentence.txt']:
  p=root/f;record(p)
 (out/'manifest.json').write_text(json.dumps(dict(sources=sources,script_sha256=sha(Path(__file__)),
   training_reward_audit='Exact historical training reward errors not recoverable from aggregate metrics/checkpoints; per-rollout records absent in inspected RL directories.',
   selection_status='Prompt overlaps computed; chronology and outcome access require decision-log reconciliation.',
   grade_policy='Current extraction grades for completed traces; historical direct grades for cutoff-truncated prefixes.'),indent=2))
 print('Audit completed',out,flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
 args=a.parse_args();main(args.root,args.out)
