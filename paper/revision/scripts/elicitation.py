"""Matched total-budget answer elicitation on frozen historical interventions.

Unlike the historical readiness endpoint, 'finished' requires the WHOLE response
to fit within the budget. The forcing-cue tokens and continuation allowance are
reserved within that budget. All prefixes are read without steering. This is a
retrospective protocol control, not a fresh held-out intervention evaluation.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time,os
import numpy as np
import polars as pl

ARMS={
 'math':{'baseline':'p1_budget','sig_math':'tx_math','sig_gpqa':'tx_math','sig_idea':'tx_math',
 'ol_math_A2':'p1_budget','sealdir_matched':'p1_budget','ol_random':'p1_budget','concise':'p1_budget',
 'rl_zeroT_s60':'p4_eval3','rl_a2initT_s60':'p4_eval2'},
 'gpqa':{'baseline':'p2_gpqa','sig_math':'tx_gpqa','sig_gpqa':'tx_gpqa','sig_idea':'tx_gpqa','ol_random':'p2_gpqa'},
 'planning':{'baseline':'p2_planning','sig_math':'tx_planning','sig_gpqa':'tx_planning','sig_idea':'tx_planning','ol_random':'p2_planning'}}

def prefix_plan(ids,end_id,budget,cue_tokens,allowance=48,completed=True):
 if completed and len(ids)<=budget:return dict(own=True,prefix_len=len(ids),allowance=0)
 end=ids.index(end_id) if end_id in ids else len(ids)
 n=min(end,budget-cue_tokens-allowance)
 if n<0:raise ValueError('Budget smaller than cue and answer allowance')
 return dict(own=False,prefix_len=n,allowance=allowance)

def run(root,out,target,minutes=10):
 started=time.monotonic()
 sys.path.insert(0,str(root));os.chdir(root)
 from mode_control.mc.common import DATA,load_tokenizer,render_prompt
 from mode_control.mc.readiness import FORCE,grade_forced,build_engine
 from mode_control.mc.budget import load_grades,grade
 from vllm import SamplingParams
 out.mkdir(parents=True,exist_ok=True);dest=out/f'{target}.parquet'
 existing=pl.read_parquet(dest) if dest.exists() else None
 done=set(zip(existing['trace_id'],existing['budget'])) if existing is not None else set()
 tok=load_tokenizer();cue=tok.encode(FORCE[target],add_special_tokens=False)
 end=tok.convert_tokens_to_ids('</think>');grades=load_grades()
 budgets=[2048,4096,12288,32768] if target=='math' else [4096,12288,32768]
 config=dict(target=target,arms=ARMS[target],budgets=budgets,answer_allowance=48,
   cue=FORCE[target],cue_tokens=len(cue),steering_during_answer=False,
   population='historical evaluation prompts; no fresh holdout claim',
   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 cp=out/f'config_{target}.json'
 if cp.exists():assert json.loads(cp.read_text())==config
 else:cp.write_text(json.dumps(config,indent=2))
 jobs=[];recs=[];expected=0
 for arm,source in ARMS[target].items():
  p=DATA/f'runs/{source}/traces_{arm}.parquet';d=pl.read_parquet(p).sort('instance_id','seed')
  for r in d.iter_rows(named=True):
   ids=list(r['gen_token_ids']);pid=tok.encode(render_prompt(tok,r['prompt']),add_special_tokens=False)
   for b in budgets:
    expected+=1
    if (r['trace_id'],b) in done:continue
    plan=prefix_plan(ids,end,b,len(cue),completed=bool(r['completed']))
    meta=dict(trace_id=r['trace_id'],instance_id=r['instance_id'],seed=r['seed'],condition=arm,source_run=source,
      task_type=target,budget=b,original_output_tokens=len(ids),prefix_tokens=plan['prefix_len'],
      cue_tokens=0 if plan['own'] else len(cue),finished=plan['own'],original_completed=bool(r['completed']),answer_allowance=plan['allowance'])
    if plan['own']:
     ok=grades.get(r['trace_id']);grade_source='corrected_extraction' if ok is not None else 'direct_fallback'
     ok=grade(r['answer_text'] or '',r['reference_answer'],target,r['prompt']) if ok is None else int(ok)
     recs.append({**meta,'correct':ok,'grade_source':grade_source,'answer':r['answer_text'] or '', 'completion':'','answer_tokens':0,'total_generated_and_cue_tokens':len(ids)})
    else:
     jobs.append(dict(ids=pid+ids[:plan['prefix_len']]+cue,meta=meta,reference=r['reference_answer'],prompt=r['prompt']))
 def save(new):
  nonlocal existing
  if not new:return
  df=pl.DataFrame(new,infer_schema_length=None)
  existing=pl.concat([existing,df],how='diagonal_relaxed') if existing is not None else df
  tmp=dest.with_suffix('.tmp.parquet');existing.write_parquet(tmp);tmp.replace(dest)
 save(recs)
 if jobs:
  llm=build_engine(max_model_len=40000)
  for offset in range(0,len(jobs),32):
   batch=jobs[offset:offset+32]
   outputs=llm.generate([{'prompt_token_ids':r['ids']} for r in batch],SamplingParams(temperature=0,max_tokens=48,seed=20261101),use_tqdm=False)
   new=[]
   for r,o in zip(batch,outputs):
    raw=o.outputs[0];answer,ok=grade_forced(target,raw.text,r['reference'],r['prompt'])
    total=r['meta']['prefix_tokens']+len(cue)+len(raw.token_ids)
    assert total<=r['meta']['budget']
    new.append({**r['meta'],'correct':ok,'grade_source':'forced_direct','answer':answer,'completion':raw.text,'answer_tokens':len(raw.token_ids),'total_generated_and_cue_tokens':total})
   save(new);print('checkpoint',target,offset+len(batch),'/',len(jobs),flush=True)
   # Check parsing, not accuracy or agreement, before scaling beyond the first batch.
   if offset==0 and sum(bool(x['answer']) for x in new)/len(new)<.6:
    raise RuntimeError('Answer-format pilot failed: under 60% parseable; raw continuations retained')
   if time.monotonic()-started>minutes*60:break
 status=dict(expected=expected,completed=existing.height if existing is not None else 0)
 (out/f'status_{target}.json').write_text(json.dumps(status,indent=2));print(status,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--target',choices=list(ARMS),required=True);p.add_argument('--minutes',type=float,default=10)
 a=p.parse_args();run(a.root,a.out.resolve(),a.target,a.minutes)
