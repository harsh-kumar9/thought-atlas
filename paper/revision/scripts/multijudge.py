"""Fixed-prefix, blinded multi-family annotation robustness.

Uses unchanged indexed-sentence Kim/Gandhi templates. Agreement is not gold
accuracy. Sampling is outcome-blind and includes unfinished atlas attempts.
The common prefix is 2,048 Qwen3.5-9B tokenizer units across all generators.
No generation outcomes or original labels are supplied to a judge.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
import numpy as np
import pandas as pd
import polars as pl

MODELS=['reasoner','gemma4_e4b','gemma4_12b','gemma4_31b','qwen35_4b','qwen35_9b','qwen35_27b']
JUDGES={'gemma':'google/gemma-4-31B-it','qwen':'Qwen/Qwen3.5-27B','llama':'meta-llama/Llama-3.1-8B-Instruct'}
BEH=['Question_and_Answering','Perspective_Shift','Conflict_of_Perspectives','Reconciliation','verification','backtracking','subgoal','backward_chaining']
def h(x):return hashlib.sha256(str(x).encode()).hexdigest()

def prepare(root,out):
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B',local_files_only=True)
 rng=np.random.default_rng(20261101); rows=[];counts=[]
 def append(r,source,prob=1.,**meta):
  text=r.get('reasoning_text_for_analysis')
  if not isinstance(text,str):text=r.get('think_text')
  if not isinstance(text,str):text=''
  ids=tok.encode(text,add_special_tokens=False);cut=tok.decode(ids[:2048],skip_special_tokens=False)
  rows.append(dict(trace_id=h(source+'|'+r['trace_id'])[:24],original_trace_id=r['trace_id'],
      source=source,instance_id=str(r['instance_id']),task_type=r['task_type'],gen_model=r.get('gen_model','qwen35_9b'),
      prompt=r.get('prompt',''),reasoning_text_for_analysis=cut,think_text=cut,analysis_source='think_text',
      completed=bool(r.get('completed',False)),inclusion_probability=prob,
      analysis_tokens=min(2048,len(ids)),full_reasoning_tokens_common_tokenizer=len(ids),**meta))
 for model in MODELS:
  d=pd.read_parquet(root/f'data/v2/traces/traces_{model}.parquet')
  for domain,g in d.groupby('task_type',sort=True):
   g=g.sort_values('trace_id');n=len(g);first=min(12,n)
   chosen=set(rng.choice(n,first,replace=False).tolist())
   unfinished=np.flatnonzero(~g.completed.fillna(False).to_numpy(bool));extra=min(4,len(unfinished))
   if extra:chosen.update(rng.choice(unfinished,extra,replace=False).tolist())
   for i in sorted(chosen):
    r=g.iloc[i].to_dict();p=first/n
    if not bool(r.get('completed')):p=1-(1-p)*(1-extra/len(unfinished))
    append(r,'atlas',p,condition='unsteered',ptype='atlas')
   counts.append(dict(source='atlas',model=model,domain=domain,population=n,unfinished=len(unfinished),sample=len(chosen)))
 # Paired intervention traces: the same 16 problems, one generation seed.
 spec={'math':{'baseline':'p1_budget','ol_math_A2':'p1_budget','rl_zeroT_s60':'p4_eval3',
      'sealdir_matched':'p1_budget','ol_random':'p1_budget','sig_math':'tx_math','sig_gpqa':'tx_math','sig_idea':'tx_math'},
      'gpqa':{'baseline':'p2_gpqa','sig_math':'tx_gpqa','sig_gpqa':'tx_gpqa'},
      'planning':{'baseline':'p2_planning','sig_math':'tx_planning','sig_gpqa':'tx_planning'},
      'idea':{'baseline':'tx_idea','sig_math':'tx_idea'}}
 for domain,arms in spec.items():
  loaded={c:pd.read_parquet(root/f'mode_control/data/runs/{run}/traces_{c}.parquet').query('seed==1') for c,run in arms.items()}
  common=sorted(set.intersection(*(set(d.instance_id.astype(str)) for d in loaded.values())))
  chosen=set(rng.choice(common,min(16,len(common)),replace=False).tolist())
  for c,d in loaded.items():
   for r in d[d.instance_id.astype(str).isin(chosen)].to_dict('records'):
    append(r,'intervention',len(chosen)/len(common),condition=c,ptype='intervention')
 # Matched demands: all four activities for identical sampled items.
 for model in ['qwen35_9b','gemma4_12b','reasoner','r1qwen7b','openthinker3_7b']:
  d=pd.read_parquet(root/f'mode_control/data/f1/traces/traces_{model}.parquet')
  d['base_item']=d.instance_id.str.rsplit('::',n=1).str[0]
  d['ptype']=d.instance_id.str.rsplit('::',n=1).str[-1]
  for domain,g in d.groupby('task_type'):
   complete=g.groupby('base_item').ptype.nunique();pool=complete[complete==4].index.to_list()
   chosen=set(rng.choice(pool,min(8,len(pool)),replace=False).tolist())
   for r in g[g.base_item.isin(chosen)].to_dict('records'):
    append(r,'demand',len(chosen)/len(pool),condition=r['ptype'],ptype=r['ptype'],base_item=r['base_item'])
 df=pd.DataFrame(rows);assert not df.trace_id.duplicated().any()
 df['blind_order']=[h('blind|'+t) for t in df.trace_id];df=df.sort_values('blind_order')
 df.to_parquet(out/'sample_key.parquet',index=False)
 blind=df[['trace_id','prompt','reasoning_text_for_analysis','think_text','analysis_source']].copy()
 blind.to_parquet(out/'blinded_traces.parquet',index=False)
 pd.DataFrame(counts).to_csv(out/'sampling_cells.csv',index=False)
 templates={x:(root/f'src/judge/prompts/{x}_per_sentence.txt').read_text() for x in ['kim','gandhi']}
 for k,v in templates.items():(out/f'{k}_per_sentence.txt').write_text(v)
 manifest=dict(seed=20261101,rows=len(df),sources=df.source.value_counts().to_dict(),judges=JUDGES,
  sampling='12 simple-random atlas attempts per model/domain plus independent 4 unfinished draws; union inclusion probabilities retained. Interventions and demands sampled by shared problem.',
  prefix='first 2048 Qwen3.5-9B tokenizer units, identical text for all judges; shorter traces retained',
  prompts={k:h(v) for k,v in templates.items()},blinded_input_sha256=hashlib.sha256((out/'blinded_traces.parquet').read_bytes()).hexdigest(),
  estimand='Multi-judge robustness on a fixed-prefix sample, not human validity or exact replication of whole-trace signatures.',
  completion='Unfinished attempts included; empty prefixes reported, never silently converted to all-zero labels.')
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)

def judge(root,out,name,limit=0,minutes=11):
 sys.path.insert(0,str(root));os.chdir(root)
 from src.judge.run_judge import run_track_B
 from src.judge.vllm_engine import EngineConfig,build_llm
 from transformers import AutoTokenizer
 from omegaconf import OmegaConf
 cfg=OmegaConf.create({'judge':{'batch_size_sentences':20,'max_model_len':16384,
  'prompts':{'kim_per_sentence':str(out/'kim_per_sentence.txt'),'gandhi_per_sentence':str(out/'gandhi_per_sentence.txt')}}})
 dest=out/f'labels_{name}.parquet';all_tr=pl.read_parquet(out/'blinded_traces.parquet')
 # The pilot is selected before reading any labels and covers each source/model/domain.
 if limit:
  key=pd.read_parquet(out/'sample_key.parquet')
  keep=key.groupby(['source','gen_model','task_type'],dropna=False).head(limit).trace_id.tolist()
  all_tr=all_tr.filter(pl.col('trace_id').is_in(keep))
 empty=all_tr.filter(pl.col('reasoning_text_for_analysis').str.len_chars()==0).height
 all_tr=all_tr.filter(pl.col('reasoning_text_for_analysis').str.len_chars()>0)
 old=pl.read_parquet(dest) if dest.exists() else None
 done=set(old['trace_id']) if old is not None else set()
 todo=all_tr.filter(~pl.col('trace_id').is_in(list(done)))
 def finish_status():
  requested=set(all_tr['trace_id']);present=set(old['trace_id'])&requested if old is not None else set()
  status=dict(judge=JUDGES[name],requested=all_tr.height,completed=len(present),empty_prefixes=empty,
    pilot=bool(limit),input_hash=hashlib.sha256((out/'blinded_traces.parquet').read_bytes()).hexdigest())
  (out/f'status_{name}.json').write_text(json.dumps(status,indent=2))
  if limit and len(present)==len(requested):
   evaluated=old.filter(pl.col('trace_id').is_in(list(requested)))
   gate=dict(judge=JUDGES[name],input_hash=status['input_hash'],traces=len(present),
     kim_parse_rate=float(evaluated['kim_parsed'].mean()),gandhi_parse_rate=float(evaluated['gandhi_parsed'].mean()),
     input_truncation_rate=float(evaluated['judge_context_truncated'].mean()),
     conversational_positive_labels=int(evaluated.select(BEH[:4]).fill_null(0).to_numpy().sum()),
     cognitive_positive_labels=int(evaluated.select(BEH[4:]).fill_null(0).to_numpy().sum()))
   gate['passed']=bool(gate['kim_parse_rate']>=.95 and gate['gandhi_parse_rate']>=.95 and
      gate['input_truncation_rate']<.01 and gate['conversational_positive_labels']>0 and gate['cognitive_positive_labels']>0)
   (out/f'pilot_gate_{name}.json').write_text(json.dumps(gate,indent=2))
   if not gate['passed']:raise RuntimeError('Schema/truncation pilot failed; inspect retained raw judge outputs before full run')
 if not todo.height:finish_status();print('all requested traces complete');return
 if not limit:
  gate=json.loads((out/f'pilot_gate_{name}.json').read_text())
  assert gate['passed'] and gate['judge']==JUDGES[name] and gate['input_hash']==hashlib.sha256((out/'blinded_traces.parquet').read_bytes()).hexdigest()
 model=JUDGES[name];tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
 start=time.monotonic()
 llm=build_llm(EngineConfig(model=model,dtype='bfloat16',max_model_len=16384,max_num_seqs=64,
     max_num_batched_tokens=8192,gpu_memory_utilization=.90))
 for offset in range(0,todo.height,8):
  chunk=todo.slice(offset,8);new=run_track_B(llm,tok,chunk,cfg,isolated=False)
  if new.height:
   new=new.with_columns(pl.lit(model).alias('judge_model'))
   old=pl.concat([old,new],how='diagonal_relaxed') if old is not None else new
   tmp=dest.with_suffix('.tmp.parquet');old.write_parquet(tmp);tmp.replace(dest)
  else:raise RuntimeError('No annotations produced for nonempty chunk')
  print('checkpoint',name,offset+len(chunk),'/',todo.height,flush=True)
  if time.monotonic()-start>minutes*60:break
 finish_status()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','judge']);p.add_argument('--root',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True);p.add_argument('--judge',choices=list(JUDGES),default='gemma')
 p.add_argument('--pilot',type=int,default=0);p.add_argument('--minutes',type=float,default=11)
 a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
 if a.stage=='prepare':prepare(a.root,a.out)
 else:judge(a.root,a.out,a.judge,a.pilot,a.minutes)
