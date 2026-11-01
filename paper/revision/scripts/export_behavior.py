"""Reconstruct comparable per-trace behavioral endpoints without model inference.
Stores original 8192-token rates, shorter absolute windows, and the exact 16
future/next-four target definition used by the historical readout.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import polars as pl

def main(root,out,target):
 sys.path.insert(0,str(root))
 from mode_control.mc.common import BEH,DATA,load_tokenizer
 from mode_control.mc.tx_analyze import target_traces,trackb,SHIFT_TARGETS
 from mode_control.mc.phenotype import _segments_with_offsets
 from mode_control.mc.signature_ctrl import segment_targets,TARGET_NAMES
 tok=load_tokenizer();out.mkdir(exist_ok=True,parents=True)
 tr=target_traces(target).filter(pl.col('seed')==1)
 wanted=['baseline','ol_math_A2','ol_random','sealdir_matched','rl_zeroT_s60','rl_a2initT_s60','concise','seal']
 wanted += [f'sig_{d}{suffix}' for d in ['math','gpqa','planning','idea','moral'] for suffix in ['', '_x0p5','_x1p5']]
 tr=tr.filter(pl.col('condition').is_in(wanted))
 # A2-initialized RL is not in tx_analyze's default run list; add it explicitly.
 runs=list(SHIFT_TARGETS[target])
 if target=='math':
  f=DATA/'runs/p4_eval2/traces_rl_a2initT_s60.parquet'
  extra=pl.read_parquet(f).filter(pl.col('seed')==1)
  tr=pl.concat([tr,extra],how='diagonal_relaxed');runs.append('p4_eval2')
 tb=trackb(runs).filter(pl.col('trace_id').is_in(tr['trace_id']))
 groups={(k[0] if isinstance(k,tuple) else k):g.sort('seg_idx') for k,g in tb.group_by('trace_id')}
 rows=[];exclusions=[]
 for r in tr.iter_rows(named=True):
  tid=r['trace_id'];g=groups.get(tid)
  if g is None:exclusions.append(dict(trace_id=tid,reason='no_annotation',condition=r['condition']));continue
  text=r['reasoning_text_for_analysis'] or ''
  starts=_segments_with_offsets(tok,text,int(g['n_segments'][0]))
  if starts is None or len(starts)!=g.height:
   exclusions.append(dict(trace_id=tid,reason='segmentation_mismatch',condition=r['condition']));continue
  valid=(g['kim_parsed'].fill_null(False)&g['gandhi_parsed'].fill_null(False)).to_numpy()
  labels=g.select(BEH).fill_null(0).to_numpy().astype(float)
  starts=np.array(starts);rids=tok.encode(text,add_special_tokens=False)
  # Historical implementation uses per-sentence token sums for offsets.
  ends=np.r_[starts[1:],len(rids)]
  rec=dict(trace_id=tid,instance_id=r['instance_id'],condition=r['condition'],target=target,seed=1,
     reasoning_tokens=len(rids),generated_tokens=r['n_new_tokens'],completed=r['completed'],
     hard_loop=r.get('hard_loop'),soft_loop=r.get('soft_loop'))
  for window in [1024,2048,4096,8192]:
   take=valid&(starts<window);rec[f'n_segments_{window}']=int(take.sum())
   for j,b in enumerate(BEH):rec[f'{b}_{window}']=float(labels[take,j].mean()) if take.any() else None
  vi=np.flatnonzero(valid);kept=vi[ends[vi]-1<8192]
  if len(kept)>80:kept=kept[np.linspace(0,len(kept)-1,80).round().astype(int)]
  if len(kept):
   tg=segment_targets(labels[vi]);mapping={int(i):k for k,i in enumerate(vi)}
   avg=tg[[mapping[int(i)] for i in kept],:16].mean(0)
   rec.update({t:float(v) for t,v in zip(TARGET_NAMES[:16],avg)})
  rows.append(rec)
  if len(rows)%100==0:print(target,len(rows),'/',tr.height,flush=True)
 pl.DataFrame(rows).write_parquet(out/f'profiles_{target}.parquet')
 (out/f'exclusions_{target}.json').write_text(json.dumps(exclusions,indent=2))
 print('complete',target,len(rows),len(exclusions),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--target',choices=['math','gpqa','planning','idea','moral'],required=True)
 a=p.parse_args();main(a.root,a.out,a.target)
