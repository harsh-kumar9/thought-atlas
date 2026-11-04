"""One-time, journaled recovery of four CUDA/NVML startup failures; no protocol edits."""
from pathlib import Path
import datetime, hashlib, json, re, subprocess
R=Path('/ada1/u/harsh/thought-atlas-reruns/arr_repair_20261101')
ledger=R/'results/rl_gpu_recovery_20261104_2212.json'
def run(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=30)
 if p.returncode:raise RuntimeError((args,p.stdout,p.stderr))
 return p.stdout.strip()
def job(j):return dict(re.findall(r'(\w+)=([^\s]+)',run(['/opt/slurm/bin/scontrol','show','job',str(j),'-o'])))
def save():
 x['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=ledger.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(ledger)
x=json.loads(ledger.read_text()) if ledger.exists() else {'reason':'Four Mira jobs failed during GPU initialization: NVML Unknown Error or No CUDA GPUs are available. No driver/admin changes attempted. Vega uses the same GPU model; change node placement only.','checkpoint_audit':'results/rl_checkpoint_audit_20261104_2212.json','replacements':{},'pending_training_moved':[435642,435643],'evaluation_node_updates':[],'protocol':'Frozen source/config/sample/seed/dose/reward/fixed-step60 schedule and existing output directories unchanged. One GPU, normal ashton, 15-minute resumable jobs; sampler disabled; absolute scontrol requeue.'}
audit=json.loads((R/x['checkpoint_audit']).read_text())
plan=[(435640,435648,'zero_seed0',0,'zero'),(435641,435649,'zero_seed1',1,'zero'),(435644,435652,'ol_math_A2_seed1',1,'ol_math_A2'),(435645,435653,'ol_math_A2_seed2',2,'ol_math_A2')]
save()
for old,child,name,seed,init in plan:
 key=str(old)
 if key not in x['replacements']:
  state=run(['/opt/slurm/bin/sacct','-X','-n','-P','-j',str(old),'--format=JobID,State,ExitCode'])
  assert f'{old}|FAILED|1:0' in state,state
  assert str(old) not in run(['/opt/slurm/bin/squeue','-h','-u','harsh','-o','%i']).splitlines(),old
  a=audit['runs'][name];assert a['optimizer_vector_progress_match'] and a['frozen_sources_tasks_match']
  d=R/'results/rl'/name
  assert hashlib.sha256((d/'optimizer.pt').read_bytes()).hexdigest()==a['optimizer_sha256']
  assert hashlib.sha256((d/'config.json').read_bytes()).hexdigest()==a['config_sha256']
  cmd=['/opt/slurm/bin/sbatch','--parsable','--hold','--qos=ashton','--nodelist=vega','--gres=gpu:1','--time=00:15:00','--requeue','--export=ALL,VLLM_USE_FLASHINFER_SAMPLER=0',f'--chdir={R}',f'--output={R}/logs/vega_recovery_{old}_%j.log',str(R/'scripts/rl_repair.sbatch'),str(R),name,'--seed',str(seed),'--init',init]
  new=int(run(cmd).split(';')[0]);x['replacements'][key]={'job_id':new,'run':name,'evaluation':child,'submit_command':cmd,'checkpoint_step':a['completed_step'],'released':False};save()
  assert job(new)['JobState']=='PENDING'
 e=x['replacements'][key];new=e['job_id']
 if not e.get('dependency_reconnected'):
  assert job(child)['JobState']=='PENDING'
  run(['/opt/slurm/bin/scontrol','update',f'JobId={child}',f'Dependency=afterok:{new}'])
  assert job(child)['Dependency']==f'afterok:{new}(unfulfilled)'
  e['dependency_reconnected']=True;save()
for jid in range(435648,435654):
 if jid not in x['evaluation_node_updates']:
  assert job(jid)['JobState']=='PENDING'
  run(['/opt/slurm/bin/scontrol','update',f'JobId={jid}','ReqNodeList=vega'])
  assert job(jid)['ReqNodeList']=='vega';x['evaluation_node_updates'].append(jid);save()
for e in x['replacements'].values():
 if not e['released']:
  j=job(e['job_id']);assert j['QOS']=='ashton' and j['TimeLimit']=='00:15:00' and j['ReqNodeList']=='vega' and j['TresPerNode']=='gres/gpu:1'
  run(['/opt/slurm/bin/scontrol','release',str(e['job_id'])]);e['released']=True;save()
S=json.loads((R/'STATUS.json').read_text())
S['jobs']['rl_zero_seeds_0_1_2']=[x['replacements']['435640']['job_id'],x['replacements']['435641']['job_id'],435642]
S['jobs']['rl_a2_seeds_0_1_2']=[435643,x['replacements']['435644']['job_id'],x['replacements']['435645']['job_id']]
S['rl_gpu_recovery']='results/rl_gpu_recovery_20261104_2212.json'
S['recoveries'].append('Four Mira RL GPU-initialization failures resumed from verified optimizer/vector checkpoints on Vega; two pending training jobs and six dependent evaluations moved to Vega. Existing evaluation IDs retained and failed-parent dependencies reconnected. Frozen scientific settings unchanged.')
S['updated_utc']=x['updated_utc'];S['gpu_status']='Four Mira CUDA/NVML startup failures diagnosed; saved checkpoints verified and replacements released on Vega. Verify actual stage progress before calling recovery successful.'
(R/'STATUS.json').write_text(json.dumps(S,indent=2)+'\n')
print(json.dumps(x,indent=2))
