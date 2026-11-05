"""CPU-only integrity audit of a frozen 60-step RL run and its evaluator handoff."""
from pathlib import Path
import argparse,datetime,json,subprocess
import numpy as np
import pandas as pd
import torch
from rl_repair import sha,task_indices

def audit(root,name,job,evaluation_job):
 d=root/'results/rl'/name;e=root/'results/rl_evaluation'/name
 cfg=json.loads((d/'config.json').read_text());status=json.loads((d/'status.json').read_text())
 assert cfg['steps']==60 and cfg['prompts']==16 and cfg['group']==8 and cfg['budget']==8192
 assert status=={'step':60,'stage':'complete'}
 assert json.loads((d/'progress.json').read_text())['step']==60
 accounting=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-n','-P','-j',str(job),'--format=JobID,State,ExitCode'],text=True).strip()
 assert f'{job}|COMPLETED|0:0' in accounting
 for p,h in cfg['source_hashes'].items():assert sha(p)==h,p
 assert sha(d/'tasks.parquet')==cfg['tasks_sha256']
 tasks=pd.read_parquet(d/'tasks.parquet');assert len(tasks)==301 and tasks.instance_id.is_unique
 assert len(list(d.glob('metrics_s*.json')))==60 and len(list(d.glob('b_s*.npy')))==61
 optim=torch.load(d/'optimizer.pt',map_location='cpu',weights_only=False)
 assert optim['step']==60 and np.array_equal(optim['bias'].numpy(),np.load(d/'b_s060.npy'))
 steps=[];zero=[]
 for step in range(1,61):
  m=json.loads((d/f'metrics_s{step:03d}.json').read_text());assert m['step']==step
  assert np.isfinite([m[k] for k in ['reward','gradient_norm','vector_norm']]).all()
  vec=np.load(d/f'b_s{step:03d}.npy');assert np.isfinite(vec).all()
  assert np.isclose(np.linalg.norm(vec),m['vector_norm'],rtol=1e-5)
  for kind in ['rollouts','rewards']:
   assert (d/f'{kind}_s{step:03d}.done').exists()
   assert sha(d/f'{kind}_s{step:03d}.parquet')==m[kind+'_sha256']
  tr=pd.read_parquet(d/f'rollouts_s{step:03d}.parquet');rw=pd.read_parquet(d/f'rewards_s{step:03d}.parquet')
  assert len(tr)==len(rw)==128 and tr.trace_id.is_unique and rw.trace_id.is_unique
  assert set(tr.trace_id)==set(rw.trace_id) and tr.step.eq(step).all()
  assert set(zip(tr.prompt_index,tr['sample']))=={(i,s) for i in range(16) for s in range(8)}
  expected=tasks.iloc[task_indices(step-1,301,16,cfg['seed'])].instance_id.to_list()
  assert all(tr.loc[tr.prompt_index.eq(i),'instance_id'].eq(v).all() for i,v in enumerate(expected))
  assert tr.gen_token_ids.map(len).le(8192).all() and rw.success.isin([0,1]).all()
  both=tr[['trace_id','prompt_index','completed']].merge(rw[['trace_id','success']],on='trace_id',validate='one_to_one')
  assert both.loc[~both.completed,'success'].eq(0).all()
  assert np.isclose(rw.success.mean(),m['reward'])
  if m['gradient_norm']==0:
   assert both.groupby('prompt_index').success.nunique().eq(1).all()
   assert (d/f'b_s{step:03d}.npy').read_bytes()==(d/f'b_s{step-1:03d}.npy').read_bytes()
   zero.append(step)
  steps.append({'step':step,'rollouts_sha256':m['rollouts_sha256'],'rewards_sha256':m['rewards_sha256'],'vector_sha256':sha(d/f'b_s{step:03d}.npy')})
 ec=json.loads((e/'config.json').read_text());et=pd.read_parquet(e/'tasks.parquet')
 assert ec['checkpoint']==60 and ec['budget']==8192 and ec['n_tasks']==199 and ec['samples_per_problem']==2
 assert ec['decode']==cfg['decode'] and ec['sampling_seed']==20261102 and ec['layer']==16
 assert ec['vector_source']==str(d/'b_s060.npy')
 assert ec['vector_sha256']==sha(e/'b_s000.npy')==sha(d/'b_s060.npy')
 assert len(et)==199 and et.instance_id.is_unique and not set(tasks.instance_id)&set(et.instance_id)
 for control in ['baseline','a2']:
  c=root/'results/rl_evaluation'/control;cc=json.loads((c/'config.json').read_text())
  assert et.equals(pd.read_parquet(c/'tasks.parquet'))
  for k in ['decode','budget','sampling_seed','samples_per_problem','task_source_sha256']:assert cc[k]==ec[k]
 result={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':name,'training_job':job,'evaluation_job':evaluation_job,'accounting':accounting,'completed_steps':60,'training_tasks':301,'rollouts_and_rewards':7680,'exact_seeded_prompt_sample_schedule':True,'token_bounds':True,'unfinished_rewards_zero':True,'all_metric_source_hashes_match':True,'all_vectors_finite':True,'optimizer_progress_final_vector_match':True,'zero_gradient_noop_steps':zero,'evaluation_handoff':{'checkpoint':60,'vector_sha256':ec['vector_sha256'],'tasks':199,'samples':2,'budget':8192,'common_control_tasks_and_sampling':True,'training_eval_task_ids_disjoint':True,'caveat':'Historical evaluation problems reused; not a fresh holdout. Evaluation not declared complete.'},'config_sha256':sha(d/'config.json'),'steps':steps}
 out=d/'completion_audit.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='steps'},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',required=True);p.add_argument('--job',type=int,required=True);p.add_argument('--evaluation-job',type=int,required=True);a=p.parse_args();audit(a.root,a.run,a.job,a.evaluation_job)
