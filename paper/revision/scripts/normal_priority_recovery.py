from pathlib import Path
import subprocess as sp,json,re,shlex,datetime
root=Path('/ada1/u/harsh/thought-atlas-reruns/arr_repair_20261101')
failed=[430419,430420,430421,430436,430437,430438,430457,430775,430776,430777]
cpu=[430486,430487,430532,430778,430779,430782]
record_path=root/'results/normal_priority_20261103.json'
record=json.loads(record_path.read_text()); pending=sorted(record['changed_job_ids'])
assert 'restarted_utc' not in record,'Restart already complete'
run=lambda args:sp.check_output(args,text=True).strip()
infos={j:run(['scontrol','show','job',str(j),'-o']) for j in pending+cpu}
for j in pending+cpu:
 assert 'JobState=PENDING' in infos[j] and str(root) in infos[j]
for j in pending:sp.run(['scontrol','hold',str(j)],check=True)
raw=run(['sacct','-X','-n','-P','-j',','.join(map(str,failed)),'--format=JobID,SubmitLine%4000'])
lines={int(x.split('|',1)[0]):x.split('|',1)[1] for x in raw.splitlines()}
for j in pending:lines[j]=infos[j].split(' SubmitLine=',1)[1].split(' WorkDir=',1)[0]
record['previous_scheduler']=infos;record['submission_lines']=lines;record.setdefault('replacement_jobs',{})
def save():record_path.write_text(json.dumps(record,indent=2)+'\n')
save();mapping={int(k):v for k,v in record['replacement_jobs'].items()}
for old in sorted(failed+pending):
 if old in mapping:continue
 parts=shlex.split(lines[old]);idx=next(i for i,x in enumerate(parts) if x.endswith('.sbatch'));script=Path(parts[idx]);script=script if script.is_absolute() else root/script
 assert script.parent==root/'scripts'
 deps=[]
 for part in parts[:idx]:
  if part.startswith('--dependency='):
   for dep in re.findall(r'\d+',part.split('=',1)[1]):
    jid=int(dep)
    if jid in mapping:deps.append(mapping[jid])
    elif jid in cpu:deps.append(jid)
    else:
     state=run(['sacct','-X','-n','-P','-j',str(jid),'--format=State']).splitlines()[0]
     if jid==430418:
      import polars as pl
      blind=pl.read_parquet(root/'results/multijudge/blinded_traces.parquet');key=pl.read_parquet(root/'results/multijudge/sample_key.parquet')
      assert blind.height==key.height==1381 and blind['trace_id'].n_unique()==1381
      assert set(blind['trace_id'])==set(key['trace_id'])
      record['obsolete_preparation_dependency']='430418 removed: repaired frozen1381-trace inputs verified; judge pilots already loaded these before engine failure.'
     else:assert state=='COMPLETED',(old,jid,state)
 args=['sbatch','--parsable','--hold','--qos=ashton','--time=00:15:00','--export=ALL,VLLM_USE_FLASHINFER_SAMPLER=0','--chdir='+str(root),'--output='+str(root/'logs'/f'normal_{old}_%j.log')]
 if deps:args.append('--dependency=afterok:'+':'.join(map(str,deps)))
 args += [str(script),*parts[idx+1:]]
 new=int(run(args).split(';')[0]);mapping[old]=new;record['replacement_jobs'][old]=new;save()
for old in cpu:
 dep=infos[old].split(' Dependency=',1)[1].split(' ',1)[0]
 ids=[int(x) for x in re.findall(r'afterok:(\d+)',dep)]
 assert ids,(old,dep)
 updated=[mapping.get(x,x) for x in ids]
 sp.run(['scontrol','update',f'JobId={old}','Dependency=afterok:'+':'.join(map(str,updated))],check=True)
record['cpu_dependencies_reconnected']=cpu;save()
for old in pending:sp.run(['scancel',str(old)],check=True)
for new in mapping.values():sp.run(['scontrol','release',str(new)],check=True)
record['restarted_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
record['runtime_repair']='VLLM_USE_FLASHINFER_SAMPLER=0 avoids observed sampling-JIT nvcc failure. Earlier Mira CUDA initialization failures preceded node reboot; retry uses current allocation. No scientific Python, sample, vectors, or endpoints changed.'
record['pending_replaced']=pending;save()
print(json.dumps(record['replacement_jobs'],indent=2))
print(run(['squeue','-h','-u','harsh','-o','%i|%j|%q|%T|%R']))
