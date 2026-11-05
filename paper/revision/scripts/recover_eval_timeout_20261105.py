import json,hashlib,subprocess,datetime
from pathlib import Path
import pandas as pd
R=Path('/ada1/u/harsh/thought-atlas-reruns/arr_repair_20261101'); f=R/'results/evaluation_timeout_recovery_20261105_1424.json'
def run(a):return subprocess.check_output(a,text=True,stderr=subprocess.STDOUT).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((R/'STATUS.json').read_text());d=R/'results/rl_evaluation/ol_math_A2_seed2';c=json.loads((d/'config.json').read_text())
assert not f.exists(),'Recovery already recorded; inspect ledger rather than duplicate'
assert run(['/opt/slurm/bin/sacct','-X','-n','-P','-j','437230','-o','JobID,State,ExitCode'])=='437230|TIMEOUT|0:0'
assert s['jobs']['rl_evaluation']['ol_math_A2_seed2']==437230
assert sha(d/'b_s000.npy')==c['vector_sha256']==sha(Path(c['vector_source']))
files={}
for p in d.glob('*.parquet'):
 q=pd.read_parquet(p)
 if 'trace_id' in q:assert q.trace_id.is_unique
 files[p.name]={'rows':len(q),'sha256':sha(p)}
assert files['rollouts_s001.parquet']['rows']==398
log=R/'logs/eval_vega_recovery_435653_437230.log'
cmd=['/opt/slurm/bin/sbatch','--parsable','--hold','--nodelist=vega','--qos=ashton','--gres=gpu:1','--time=00:15:00','--requeue','--export=ALL,VLLM_USE_FLASHINFER_SAMPLER=0',f'--chdir={R}',f'--output={R}/logs/eval_timeout_recovery_%j.log',str(R/'scripts/rl_evaluate_guarded.sbatch'),str(R),'ol_math_A2_seed2']
new=int(run(cmd).split(';')[0]);x={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'old_job':437230,'new_job':new,'reason':'Checkpoint loading stalled on NFS during extraction initialization; walltime expired before normal soft-deadline return. Add pre-walltime batch-shell signal and responsive background-child wait.','scientific_changes':False,'saved_files':files,'config_sha256':sha(d/'config.json'),'old_log_sha256':sha(log),'old_log_tail':log.read_text(errors='replace').splitlines()[-10:],'submit_command':cmd,'released':False};f.write_text(json.dumps(x,indent=2)+'\n')
s['jobs']['rl_evaluation']['ol_math_A2_seed2']=new
parents=[s['jobs']['rl_evaluation'][k] for k in ['zero_seed0','zero_seed1','zero_seed2','ol_math_A2_seed2']]
run(['/opt/slurm/bin/scontrol','update','JobId=430532','Dependency=afterok:'+':'.join(map(str,parents))]);x['analysis_dependencies']=parents
s['code_files']['scripts/rl_evaluate_guarded.sbatch']=sha(R/'scripts/rl_evaluate_guarded.sbatch');s['evaluation_timeout_recovery']=str(f.relative_to(R));(R/'STATUS.json').write_text(json.dumps(s,indent=2)+'\n')
run(['/opt/slurm/bin/scontrol','release',str(new)]);x['released']=True;f.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x,indent=2))
