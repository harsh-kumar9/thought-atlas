"""Verify a completed clean-RL evaluation and reproduce grades from saved extractions."""
from pathlib import Path
import argparse,datetime,json,subprocess,sys
import numpy as np
import pandas as pd
from rl_repair import sha

def audit(root,historical,name,job):
 sys.path.insert(0,str(historical))
 from mode_control.mc.extract_free import grade_row
 d=root/'results/rl_evaluation'/name;c=json.loads((d/'config.json').read_text())
 assert json.loads((d/'status.json').read_text())['stage']=='complete'
 accounting=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-n','-P','-j',str(job),'--format=JobID,State,ExitCode'],text=True).strip()
 assert f'{job}|COMPLETED|0:0' in accounting
 assert c['checkpoint']==60 and c['budget']==8192 and c['samples_per_problem']==2 and c['n_tasks']==199
 tr=pd.read_parquet(d/'rollouts_s001.parquet');rw=pd.read_parquet(d/'rewards_s001.parquet');tasks=pd.read_parquet(d/'tasks.parquet')
 assert (d/'rollouts_s001.done').exists() and (d/'rewards_s001.done').exists()
 assert len(tr)==len(rw)==398 and len(tasks)==199 and tasks.instance_id.is_unique
 assert tr.trace_id.is_unique and rw.trace_id.is_unique and set(tr.trace_id)==set(rw.trace_id)
 assert set(zip(tr.instance_id,tr['sample']))=={(i,s) for i in tasks.instance_id for s in [0,1]}
 assert tr.gen_token_ids.map(len).le(8192).all() and rw.success.isin([0,1]).all()
 assert sha(Path(c['vector_source']))==c['vector_sha256']==sha(d/'b_s000.npy')
 source_paths={'rl_evaluate.py':root/'scripts/rl_evaluate.py','rl_repair.py':root/'scripts/rl_repair.py','extract_free.py':historical/'mode_control/mc/extract_free.py'}
 for k,h in c['code_sha256'].items():assert sha(source_paths[k])==h
 training=root/'results/rl'/name;tc=json.loads((training/'config.json').read_text())
 for p,h in tc['source_hashes'].items():assert sha(p)==h
 assert not set(pd.read_parquet(training/'tasks.parquet').instance_id)&set(tasks.instance_id)
 for control in ['baseline','a2']:
  cd=root/'results/rl_evaluation'/control;cc=json.loads((cd/'config.json').read_text())
  assert tasks.equals(pd.read_parquet(cd/'tasks.parquet'))
  for k in ['decode','budget','sampling_seed','samples_per_problem','task_source_sha256']:assert c[k]==cc[k]
 saved={z['trace_id']:z for z in rw.to_dict('records')};mismatches=[]
 for row in tr.to_dict('records'):
  e=saved[row['trace_id']];fresh=grade_row(row,e)
  if fresh['success']!=e['success'] or fresh['grade_source']!=e['grade_source']:mismatches.append(row['trace_id'])
 assert not mismatches,mismatches
 outcomes=pd.read_csv(d/'outcomes.csv');both=outcomes.merge(rw[['trace_id','success','grade_source']],on='trace_id',suffixes=('_out','_saved'),validate='one_to_one')
 assert len(both)==398 and both.success_out.eq(both.success_saved).all() and both.grade_source_out.eq(both.grade_source_saved).all()
 summary=json.loads((d/'summary.json').read_text());assert summary['attempts']==398 and summary['n_tasks']==199 and summary['fixed_step']==60 and summary['budget']==8192
 assert np.isclose(summary['accuracy'],rw.success.mean()) and np.isclose(summary['completion'],tr.completed.mean())
 joined=tr[['trace_id','completed']].merge(rw[['trace_id','success']],on='trace_id',validate='one_to_one');assert joined.loc[~joined.completed,'success'].eq(0).all()
 fallback=rw.grade_source.str.startswith('fallback');sources=rw.grade_source.value_counts().to_dict()
 result={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':name,'job':job,'accounting':accounting,'tasks':199,'attempts':398,'exact_task_sample_grid':True,'token_bounds':True,'frozen_final_vector_and_sources_match':True,'control_tasks_and_sampling_match':True,'training_eval_task_ids_disjoint':True,'corrected_grade_reproduction_rows':398,'grade_mismatches':0,'summary_and_outcomes_match':True,'grade_sources':sources,'fallback_rows':int(fallback.sum()),'fallback_worst_case_accuracy_range_pp':float(100*fallback.sum()/398),'accuracy':float(rw.success.mean()),'completion':float(tr.completed.mean()),'hashes':{f:sha(d/f) for f in ['config.json','tasks.parquet','b_s000.npy','rollouts_s001.parquet','rewards_s001.parquet','outcomes.csv','summary.json']},'scope':'Individual seed integrity; no aggregate replication conclusion until all six seeds and paired analysis complete. Historical 199 evaluation items reused.'}
 (d/'completion_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--historical',type=Path,required=True);p.add_argument('--run',required=True);p.add_argument('--job',type=int,required=True);a=p.parse_args();audit(a.root,a.historical,a.run,a.job)
