"""Journaled node-only recovery after allocation-specific CUDA/NVML probe passes."""
from pathlib import Path
import datetime,hashlib,json,re,subprocess
R=Path('/ada1/u/harsh/thought-atlas-reruns/arr_repair_20261101');out=R/'results/mira_return_20261105_0512.json'
def run(args):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=30).strip()
def state(j):return dict(re.findall(r'(\w+)=([^\s]+)',run(['/opt/slurm/bin/scontrol','show','job',str(j),'-o'])))
def save():
 x['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=out.with_suffix('.tmp');p.write_text(json.dumps(x,indent=2)+'\n');p.replace(out)
probe=run(['/opt/slurm/bin/sacct','-X','-n','-P','-j','437187','--format=JobID,State,ExitCode']);assert probe=='437187|COMPLETED|0:0'
p=R/'logs/mira_probe_20261105_437187.log';log=p.read_text();assert '"cuda_compute_verified": true' in log and 'DeviceCapability(major=12, minor=0)' in log
x=json.loads(out.read_text()) if out.exists() else {'reason':'Vega8/8 GPUs allocated and scheduling estimate shifted to17:16UTC (~12h). Mira4/8 allocated. Assigned Mira GPU0 passed CUDA computation and vLLM/NVML probe; this does not establish every GPU is healthy. Resume existing jobs and verify actual work.','probe_job':437187,'probe_accounting':probe,'probe_log':log,'probe_log_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'actions':{},'scientific_changes':False,'policy':'Existing IDs/checkpoints/dependencies; normal ashton, one GPU,15-minute resumable jobs. Node placement only.'};save()
for jid in [435690,436712,435648,435649,435650,435653]:
 key=str(jid)
 if key in x['actions']:continue
 before=state(jid)
 assert before['QOS']=='ashton' and before['TimeLimit']=='00:15:00' and before['TresPerNode']=='gres/gpu:1'
 if before['JobState']!='PENDING':
  x['actions'][key]={'action':'preserved-active-job','state':before['JobState'],'node':before.get('NodeList')};save();continue
 assert before['ReqNodeList']=='vega'
 run(['/opt/slurm/bin/scontrol','update',f'JobId={jid}','ReqNodeList=mira'])
 after=state(jid);assert after['ReqNodeList']=='mira' and after['Dependency']==before['Dependency']
 x['actions'][key]={'action':'ReqNodeList=mira','previous':'vega','dependency_preserved':after['Dependency'],'state_after':after['JobState']};save()
print(json.dumps(x,indent=2))
