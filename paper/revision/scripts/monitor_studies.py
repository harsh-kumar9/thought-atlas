"""Read-only study verification; writes a timestamped evidence snapshot on Ada.

Current training/evaluation IDs come from STATUS.json. Scientific settings and
completed-output checks are unchanged by runtime node recovery.
"""
from pathlib import Path
import json,datetime,subprocess,re,hashlib,math
import pandas as pd
R=Path('/ada1/u/harsh/thought-atlas-reruns/arr_repair_20261101');S=json.loads((R/'STATUS.json').read_text())
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
x=dict(checked_utc=now,jobs={},outputs={},rl_checkpoints={},code_mismatches=[],log_alerts={},completed_stage_checks={})
training=S['jobs']['rl_zero_seeds_0_1_2']+S['jobs']['rl_a2_seeds_0_1_2']
evaluations=[S['jobs']['rl_evaluation'][k] for k in ['zero_seed0','zero_seed1','zero_seed2','ol_math_A2_seed0','ol_math_A2_seed1','ol_math_A2_seed2']]
ids=list(dict.fromkeys([435690,435691,435756,435757,*training,*evaluations,435646,435647,*range(435654,435657),430532,430778,430779,435692,435693,430782]))
for jid in ids:
 p=subprocess.run(['/opt/slurm/bin/scontrol','show','job',str(jid),'-o'],capture_output=True,text=True,timeout=15)
 fields=dict(re.findall(r'(\w+)=([^\s]+)',p.stdout));entry={k:fields.get(k) for k in ['JobState','Reason','ExitCode','Restarts','QOS','TimeLimit','Dependency','NodeList','ReqNodeList','TresPerNode','Requeue']};x['jobs'][str(jid)]=entry
 for k in ['StdOut','StdErr']:
  f=Path(fields.get(k,'/nonexistent'))
  if f.is_file():
   entry['log_mtime_utc']=datetime.datetime.fromtimestamp(f.stat().st_mtime,datetime.timezone.utc).isoformat()
   if fields.get('JobState') in ['RUNNING','PENDING','COMPLETING']:
    lines=f.read_text(errors='replace').splitlines()[-100:]
    alerts=[l[:500] for l in lines if re.search(r'Traceback|RuntimeError|AssertionError|OutOfMemoryError|FileNotFoundError|NVMLError',l)]
    if alerts:x['log_alerts'][str(jid)+':'+k]=alerts
p=subprocess.run(['/opt/slurm/bin/sacct','-X','-j',','.join(map(str,ids)),'-n','-P','-o','JobID,State,ExitCode,Elapsed,Restarts'],capture_output=True,text=True,timeout=20)
assert p.returncode==0,p.stderr
x['accounting']={}
for line in p.stdout.splitlines():
 fields=line.split('|')
 if len(fields)>=5:x['accounting'][fields[0]]={k:v for k,v in zip(['State','ExitCode','Elapsed','Restarts'],fields[1:5])}
for jid in [435756,435693,430782]:assert x['accounting'][str(jid)]['State']=='COMPLETED' and x['accounting'][str(jid)]['ExitCode']=='0:0'
for jid in [435690,435691,*training,*evaluations]:
 j=x['jobs'][str(jid)]
 if j['JobState'] is None:
  assert x['accounting'][str(jid)]['State']=='COMPLETED',x['accounting'][str(jid)]
  continue
 assert j['QOS']=='ashton' and j['TimeLimit']=='00:15:00' and j['TresPerNode']=='gres/gpu:1' and j['ReqNodeList'] in ['mira','vega'] and j['Requeue']=='1'
for parent,child in zip(training,evaluations):
 if x['accounting'][str(parent)]['State']!='COMPLETED':
  assert x['jobs'][str(child)]['Dependency']==f'afterok:{parent}(unfulfilled)'
for child,parents in [(435757,[435690,435691]),(430532,evaluations)]:
 expected={str(p) for p in parents if x['accounting'][str(p)]['State']!='COMPLETED'}
 actual=set(re.findall(r'afterok:(\d+)',x['jobs'][str(child)]['Dependency'] or ''))
 assert expected==actual,(child,expected,actual)
x['scheduler_recovery_checks']={'munge':'active','completed_exits_verified':[435756,435693,430782],'gpu_policy_verified':True,'rl_evaluation_dependencies_verified':True,'analysis_dependencies_verified':True,'mutations':[]}
for folder in ['multijudge','multijudge_llama_indexed','confirmation','rl_evaluation','rl']:
 for f in (R/'results'/folder).rglob('status*.json'):
  x['outputs'][str(f.relative_to(R))]=dict(mtime=datetime.datetime.fromtimestamp(f.stat().st_mtime,datetime.timezone.utc).isoformat(),status=json.loads(f.read_text()))
 if folder=='rl':continue
 for f in (R/'results'/folder).rglob('*.parquet'):
  if not (f.name.startswith(('labels_','rollouts_','rewards_')) or f.name=='elicitation.parquet'):continue
  d=pd.read_parquet(f);v=dict(rows=len(d),mtime=datetime.datetime.fromtimestamp(f.stat().st_mtime,datetime.timezone.utc).isoformat())
  if f.name.startswith('labels_'):
   v.update(unique_traces=int(d.trace_id.nunique()),duplicate_segments=int(d.duplicated(['trace_id','seg_idx']).sum()));assert v['duplicate_segments']==0
  x['outputs'][str(f.relative_to(R))]=v
for d in (R/'results/rl').iterdir():
 if not d.is_dir() or d.name=='smoke':continue
 metrics=sorted(d.glob('metrics_s*.json'))
 if metrics:
  m=json.loads(metrics[-1].read_text());assert all(math.isfinite(m[k]) for k in ['reward','gradient_norm','vector_norm'])
  if m['gradient_norm']==0:
   step=m['step'];tr=pd.read_parquet(d/f'rollouts_s{step:03d}.parquet');rw=pd.read_parquet(d/f'rewards_s{step:03d}.parquet')
   both=tr[['trace_id','prompt_index','sample']].merge(rw[['trace_id','success']],on='trace_id',validate='one_to_one')
   groups=both.groupby('prompt_index').success.agg(['count','nunique','mean'])
   assert len(both)==128 and len(groups)==16 and groups['count'].eq(8).all() and groups['nunique'].eq(1).all()
   assert (d/f'b_s{step:03d}.npy').read_bytes()==(d/f'b_s{step-1:03d}.npy').read_bytes()
   for kind in ['rollouts','rewards']:assert hashlib.sha256((d/f'{kind}_s{step:03d}.parquet').read_bytes()).hexdigest()==m[kind+'_sha256']
   x.setdefault('zero_gradient_checks',{})[d.name]={'step':step,'rows':128,'groups':16,'samples_per_group':8,'all_within_group_advantages_zero':True,'vector_unchanged':True,'source_hashes_match':True,'all_correct_groups':int(groups['mean'].eq(1).sum()),'all_incorrect_groups':int(groups['mean'].eq(0).sum()),'interpretation':'Expected no-op for group-centered rewards; preserve frozen training schedule and optimizer behavior.'}
  x['rl_checkpoints'][d.name]=dict(latest_metrics=m,saved_gradients=len(list(d.glob('gradient_s*.pt'))),saved_vectors=len(list(d.glob('b_s*.npy'))),completed_step_metric_count=len(metrics))
for relative,h in S['code_files'].items():
 if hashlib.sha256((R/relative).read_bytes()).hexdigest()!=h:x['code_mismatches'].append(relative)
for family,arms,n in [('rl_evaluation',['baseline','a2'],199),('confirmation',['baseline','ol_math_A2','sig_moral'],256)]:
 for arm in arms:
  d=R/'results'/family/arm
  if json.loads((d/'status.json').read_text())['stage']!='complete':continue
  tr=pd.read_parquet(d/'rollouts_s001.parquet');gr=pd.read_parquet(d/'rewards_s001.parquet');tasks=pd.read_parquet((d if family=='rl_evaluation' else d.parent)/'tasks.parquet')
  assert len(tr)==len(gr)==2*n and tr.trace_id.is_unique and gr.trace_id.is_unique
  assert set(tr.trace_id)==set(gr.trace_id)
  assert set(zip(tr.instance_id,tr['sample']))=={(i,s) for i in tasks.instance_id for s in [0,1]}
  assert tr.gen_token_ids.map(len).max()<=8192 and gr.success.isin([0,1]).all()
  result=dict(rows=len(tr),unique_tasks=n,exact_frozen_task_sample_grid=True,max_generated_tokens=int(tr.gen_token_ids.map(len).max()),reward_rows=len(gr),grade_sources=gr.grade_source.value_counts().to_dict())
  if family=='rl_evaluation':
   out=pd.read_csv(d/'outcomes.csv');q=out.merge(gr[['trace_id','success']],on='trace_id',validate='one_to_one',suffixes=('_out','_gr'));assert len(q)==n*2 and (q.success_out==q.success_gr).all()
   summary=json.loads((d/'summary.json').read_text());assert abs(summary['accuracy']-gr.success.mean())<1e-12
   result.update(outcomes_rows=len(out),summary_matches_rewards=True)
  else:
   el=pd.read_parquet(d/'elicitation.parquet');assert len(el)==n*2 and el.trace_id.is_unique and set(el.trace_id)==set(tr.trace_id) and el.total_tokens.le(8192).all()
   result.update(elicitation_rows=len(el),elicitation_token_bounds=True)
  result['status']='Individual arm verified; whole-family analyses remain gated.';x['completed_stage_checks'][family+'/'+arm]=result
assert not x['code_mismatches']
x['assessment']='Scheduler access restored; inspect states, accounting and dependencies alongside saved-output checks.'
x['log_tails']={}
for j in ids:
 for lf in sorted((R/'logs').glob('*'+str(j)+'*')):
  x['log_tails'][lf.name]={'mtime':datetime.datetime.fromtimestamp(lf.stat().st_mtime,datetime.timezone.utc).isoformat(),'tail':lf.read_bytes()[-5000:].decode(errors='replace').splitlines()[-6:]}
f=R/'results'/('heartbeat_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M')+'.json');f.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x,indent=2))
