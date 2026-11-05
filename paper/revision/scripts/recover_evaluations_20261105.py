"""Restore saved evaluation work to Vega after repeated Mira NVML startup failures."""
from pathlib import Path
import datetime,hashlib,json,subprocess,re
import pandas as pd
R=Path('/ada1/u/harsh/thought-atlas-reruns/arr_repair_20261101');f=R/'results/evaluation_recovery_20261105_0542.json'
def run(args):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=30).strip()
def save():
 x['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();t=f.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(f)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
x=json.loads(f.read_text()) if f.exists() else {'reason':'Repeated Mira NVML Unknown Error on new allocations, before model startup. One successful GPU probe does not establish allocation reliability. Resume failed evaluations on Vega; keep all saved work and frozen configurations.','replacements':{},'pending_zero_seed1_placement':'435649 moved to Vega between chunks','scientific_changes':False};save()
S=json.loads((R/'STATUS.json').read_text())
for old,name,stem in [(435650,'zero_seed2',435593),(435653,'ol_math_A2_seed2',435596)]:
 key=str(old)
 if key not in x['replacements']:
  assert run(['/opt/slurm/bin/sacct','-X','-n','-P','-j',key,'--format=JobID,State,ExitCode'])==f'{old}|FAILED|1:0'
  assert key not in run(['/opt/slurm/bin/squeue','-h','-u','harsh','-o','%i']).splitlines()
  d=R/'results/rl_evaluation'/name;c=json.loads((d/'config.json').read_text());assert c['checkpoint']==60 and c['budget']==8192 and c['samples_per_problem']==2
  assert sha(d/'b_s000.npy')==c['vector_sha256']==sha(Path(c['vector_source']))
  files={}
  for p in d.glob('*.parquet'):
   q=pd.read_parquet(p)
   if 'trace_id' in q:assert q.trace_id.is_unique
   if p.name.startswith('rollouts'):assert not q.duplicated(['instance_id','sample']).any() and q.gen_token_ids.map(len).le(8192).all()
   files[p.name]={'rows':len(q),'sha256':sha(p)}
  log=R/f'logs/normal_{stem}_{old}.log';assert 'NVMLError_Unknown' in log.read_text(errors='replace')[-5000:]
  cmd=['/opt/slurm/bin/sbatch','--parsable','--hold','--nodelist=vega','--qos=ashton','--gres=gpu:1','--time=00:15:00','--requeue','--export=ALL,VLLM_USE_FLASHINFER_SAMPLER=0',f'--chdir={R}',f'--output={R}/logs/eval_vega_recovery_{old}_%j.log',str(R/'scripts/rl_evaluate.sbatch'),str(R),name]
  new=int(run(cmd).split(';')[0]);x['replacements'][key]={'new_job':new,'run':name,'config_sha256':sha(d/'config.json'),'saved_files':files,'failed_log_sha256':sha(log),'failed_log_tail':log.read_text(errors='replace').splitlines()[-12:],'submit_command':cmd,'released':False};save()
 S['jobs']['rl_evaluation'][name]=x['replacements'][key]['new_job']
parents=[S['jobs']['rl_evaluation'][n] for n in ['zero_seed0','zero_seed1','zero_seed2','ol_math_A2_seed2']]
run(['/opt/slurm/bin/scontrol','update','JobId=430532','Dependency=afterok:'+':'.join(map(str,parents))]);x['analysis_dependency']=parents;save()
for e in x['replacements'].values():
 if not e['released']:run(['/opt/slurm/bin/scontrol','release',str(e['new_job'])]);e['released']=True;save()
S['evaluation_recovery']='results/evaluation_recovery_20261105_0542.json';(R/'STATUS.json').write_text(json.dumps(S,indent=2)+'\n');print(json.dumps(x,indent=2))
