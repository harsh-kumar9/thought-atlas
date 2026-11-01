"""Versioned RL replication using the corrected extraction reward.

One GPU stage per invocation (generate, reference-blind extract, update), with
atomic per-batch/per-rollout checkpoints. Historical vectors are never changed.
Every rollout, extracted answer, reward and optimizer state is retained. This
separates engine lifetimes and fits the cluster's preemptible 15-minute jobs.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
import numpy as np
import polars as pl

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_parquet(d,p):
    tmp=p.with_suffix('.tmp.parquet');d.write_parquet(tmp);tmp.replace(p)
def atomic_json(x,p):
    tmp=p.with_suffix('.tmp.json');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def sampling_logprob(logits,targets,temperature):
    """Temperature is an explicit argument, never a sequence-length closure."""
    import torch
    return torch.log_softmax(logits.float()/float(temperature),-1).gather(-1,targets[...,None])[...,0].sum()
def task_indices(step,n,prompts,seed):
    result=[]
    for absolute in range(step*prompts,(step+1)*prompts):
        epoch,pos=divmod(absolute,n)
        result.append(int(np.random.default_rng(seed+epoch).permutation(n)[pos]))
    return result

def setup(a):
    from mode_control.mc.common import DECODE
    from mode_control.mc.rlbias import train_tasks
    from mode_control.mc.vsteer import load_vector,VS
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    tasks=train_tasks();tasks_path=out/'tasks.parquet'
    if not tasks_path.exists():atomic_parquet(tasks,tasks_path)
    assert tasks.equals(pl.read_parquet(tasks_path))
    config={k:getattr(a,k) for k in ['seed','init','steps','prompts','group','budget','lr']}
    code=[Path(__file__),a.root/'mode_control/mc/extract_free.py',a.root/'src/judge/run_answer_extraction.py',
          a.root/'src/perf/answer_grading.py',a.root/'mode_control/mc/common.py',a.root/'mode_control/mc/vsteer.py']
    config.update(layer=16,decode=dict(DECODE),temperature=float(DECODE['temperature']),
      reward='Gemma4-31B-it reference-blind free-text extraction, grounding and math_correct; unfinished=0; same grade_row as evaluation',
      source_hashes={str(p):sha(p) for p in code},tasks_sha256=sha(tasks_path),
      initialization_sha256=sha(VS/'vec_ol_math_A2.npz'),
      protocol='Three training seeds per initialization; fixed step 60; no evaluation-based checkpoint selection. Historical evaluation set reused, not fresh holdout.',
      logprob_policy='temperature-corrected untruncated softmax; top-k/top-p sampling truncation ignored as in historical implementation')
    cp=out/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==config,'Frozen replication config changed'
    else:atomic_json(config,cp)
    if not (out/'b_s000.npy').exists():
        b=load_vector('ol_math_A2')['v'].astype(np.float32)
        if a.init=='zero':b=np.zeros_like(b)
        np.save(out/'b_s000.npy',b)
    return tasks,config

def next_stage(out,steps):
    done=json.loads((out/'progress.json').read_text())['step'] if (out/'progress.json').exists() else 0
    if (out/'optimizer.pt').exists():
        import torch
        done=int(torch.load(out/'optimizer.pt',map_location='cpu',weights_only=False)['step'])
    if done>=steps:return done,'complete'
    step=done+1
    if not (out/f'rollouts_s{step:03d}.done').exists():return step,'generate'
    if not (out/f'rewards_s{step:03d}.done').exists():return step,'extract'
    return step,'update'

def generate(a,tasks,cfg,step,started):
    from mode_control.mc.common import load_tokenizer,render_prompt,json_hash,canon_text
    from mode_control.mc.vsteer import build_vllm,_set_steer
    from src.utils.parse import parse_generation_detailed,is_completed
    from vllm import SamplingParams
    path=a.out/f'rollouts_s{step:03d}.parquet'
    data=pl.read_parquet(path) if path.exists() else None
    done=set(data['prompt_index']) if data is not None else set()
    batch=tasks[task_indices(step-1,tasks.height,a.prompts,a.seed)]
    pending=[(i,r) for i,r in enumerate(batch.iter_rows(named=True)) if i not in done]
    if pending:
        tok=load_tokenizer();llm=build_vllm(a.budget+2048,gpu_mem=.88,max_num_seqs=64)
        _set_steer(llm,tok,{'v':np.load(a.out/f'b_s{step-1:03d}.npy'),'layer':16},'all')
        for off in range(0,len(pending),4):
            chunk=pending[off:off+4];pids=[tok.encode(render_prompt(tok,r['prompt']),add_special_tokens=False) for i,r in chunk]
            seeds=[int(json_hash(['arr_repair',a.seed,step,i])[:8],16) for i,r in chunk]
            sp=[SamplingParams(n=a.group,max_tokens=a.budget,seed=s,skip_special_tokens=False,**cfg['decode']) for s in seeds]
            outputs=llm.generate([{'prompt_token_ids':p} for p in pids],sp,use_tqdm=False);rows=[]
            for (i,r),pid,seed,o in zip(chunk,pids,seeds,outputs):
                for sample,g in enumerate(o.outputs):
                    finish='stop' if g.finish_reason=='stop' else 'length';text=canon_text(g.text)
                    for marker in (tok.eos_token,tok.pad_token,'<|im_end|>'):
                        if marker and text.endswith(marker):text=text[:-len(marker)]
                    parsed=parse_generation_detailed(text,'reasoning',finish_reason=finish)
                    completed=is_completed(kind='reasoning',finish_reason=finish,
                        has_close_tag=parsed['close_tag_count']>0 or parsed['parse_status']=='direct_answer_no_close',
                        has_answer=bool(parsed.get('answer_text')))
                    rows.append(dict(trace_id=json_hash([str(a.out.name),step,i,sample]),instance_id=r['instance_id'],
                        prompt=r['prompt'],reference_answer=str(r['reference_answer']),task_type='math',
                        step=step,prompt_index=i,sample=sample,sampling_seed=seed,prompt_token_ids=pid,
                        gen_token_ids=list(g.token_ids),full_text=text,answer_text=parsed.get('answer_text') or '',
                        think_text=parsed.get('think_text') or '',parse_status=parsed.get('parse_status'),
                        finish_reason=finish,completed=completed,generation_kind='reasoning'))
            new=pl.DataFrame(rows,infer_schema_length=None);data=pl.concat([data,new],how='diagonal_relaxed') if data is not None else new
            atomic_parquet(data,path);print('generated',data.height,flush=True)
            if time.monotonic()-started>a.minutes*60:break
    if data is not None and data.height==a.prompts*a.group:(a.out/f'rollouts_s{step:03d}.done').touch()

def extract(a,cfg,step,started):
    from mode_control.mc.extract_free import normalise,grade_row
    from src.judge.run_answer_extraction import build_prompt,_score_rows,_fingerprint
    from src.judge.vllm_engine import EngineConfig,build_llm,chat_batch,make_sampling
    from transformers import AutoTokenizer
    path=a.out/f'rewards_s{step:03d}.parquet';tr=pl.read_parquet(a.out/f'rollouts_s{step:03d}.parquet')
    data=pl.read_parquet(path) if path.exists() else None;done=set(data['trace_id']) if data is not None else set()
    pending=tr.filter(~pl.col('trace_id').is_in(list(done))).to_dicts()
    model='google/gemma-4-31B-it';maximum=16384;mt=768
    if pending:
        tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
        llm=build_llm(EngineConfig(model=model,dtype='bfloat16',max_model_len=maximum,max_num_seqs=64,gpu_memory_utilization=.9))
        for off in range(0,len(pending),32):
            rows=pending[off:off+32];pairs=[build_prompt(r,tok,maximum-mt-32) for r in rows]
            raw,trunc=chat_batch(llm,tok,[p[0] for p in pairs],make_sampling(temperature=0,max_tokens=mt),max_input_tokens=maximum-mt-32,return_truncation=True)
            metas=[p[1] for p in pairs]
            for m,t in zip(metas,trunc):m['hard_prompt_truncated']=t
            ex=_score_rows([normalise(x) for x in raw],rows,metas,model,_fingerprint(model,mt,maximum)+'|free-text-v1').with_columns(pl.Series('raw_output',raw))
            grades=[grade_row(t,e) for t,e in zip(rows,ex.iter_rows(named=True))]
            new=ex.join(pl.DataFrame(grades),on='trace_id',validate='1:1')
            data=pl.concat([data,new],how='diagonal_relaxed') if data is not None else new
            atomic_parquet(data,path);print('graded',data.height,flush=True)
            if time.monotonic()-started>a.minutes*60:break
    if data is not None and data.height==tr.height:(a.out/f'rewards_s{step:03d}.done').touch()

def update(a,cfg,step,started):
    import torch
    import torch.utils.checkpoint
    from mode_control.mc.common import load_model,decoder_layers,_layer_output,_replace_output
    torch.manual_seed(a.seed+step);torch.cuda.manual_seed_all(a.seed+step)
    d=pl.read_parquet(a.out/f'rollouts_s{step:03d}.parquet').join(pl.read_parquet(a.out/f'rewards_s{step:03d}.parquet').select('trace_id','success'),on='trace_id',validate='1:1').sort('prompt_index','sample')
    d=d.with_columns((pl.col('success')-pl.col('success').mean().over('prompt_index')).alias('advantage'))
    b=torch.nn.Parameter(torch.as_tensor(np.load(a.out/f'b_s{step-1:03d}.npy'),device='cuda:0',dtype=torch.float32))
    opt=torch.optim.Adam([b],lr=a.lr);state=a.out/'optimizer.pt'
    if state.exists():
        old=torch.load(state,map_location='cuda:0',weights_only=False)
        assert old['step']==step-1;opt.load_state_dict(old['optimizer'])
    model=load_model()
    for p in model.parameters():p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
    decoder_layers(model)[16].register_forward_hook(lambda m,args,o:_replace_output(o,(_layer_output(o).float()+b).to(_layer_output(o).dtype)))
    norms=[m for name,m in model.named_modules() if name.endswith('language_model.norm')];assert len(norms)==1
    grab={};norms[0].register_forward_hook(lambda m,i,o:grab.__setitem__('h',o))
    temperature=float(cfg['temperature']);partial=a.out/f'gradient_s{step:03d}.pt';offset=0
    if not (a.out/'gradient_canary.json').exists():
        # A structural gradient check independent of whether pilot rewards vary.
        r=d.row(0,named=True);sample=list(r['prompt_token_ids'])[:32]+list(r['gen_token_ids'])[:1]
        probe=torch.tensor([sample],device=b.device)
        model(input_ids=probe,logits_to_keep=1,use_cache=False);h=grab.pop('h')[0]
        lp=sampling_logprob(model.lm_head(h[-2:-1]),probe[0,-1:],temperature)
        derivative,=torch.autograd.grad(lp,b)
        assert torch.isfinite(derivative).all() and float(derivative.norm())>0
        atomic_json(dict(passed=True,gradient_norm=float(derivative.norm()),temperature=temperature,
             sequence_length=len(sample),optimizer_changed=False),a.out/'gradient_canary.json')
        grab.clear()
    opt.zero_grad(set_to_none=True)
    if partial.exists():
        q=torch.load(partial,map_location='cpu',weights_only=False);offset=q['next'];b.grad=q['gradient'].to(b.device) if q['gradient'] is not None else None
        torch.set_rng_state(q['rng']);torch.cuda.set_rng_state_all(q['cuda_rng'])
    for i,r in enumerate(d.iter_rows(named=True)):
        if i<offset:continue
        if r['advantage'] and len(r['gen_token_ids']):
            ids=torch.tensor([list(r['prompt_token_ids'])+list(r['gen_token_ids'])],device=b.device)
            model(input_ids=ids,logits_to_keep=1,use_cache=False);h=grab.pop('h')[0]
            prompt_length=len(r['prompt_token_ids']);sequence_length=ids.shape[1]
            targets=ids[0,prompt_length:sequence_length];hidden=h[prompt_length-1:sequence_length-1]
            def chunk_lp(states,tgt):return sampling_logprob(model.lm_head(states),tgt,temperature)
            lp=sum(torch.utils.checkpoint.checkpoint(chunk_lp,hidden[s:s+1024],targets[s:s+1024],use_reentrant=False) for s in range(0,len(hidden),1024))
            (-r['advantage']*lp/(a.prompts*a.group*a.budget)).backward()
        tmp=partial.with_suffix('.tmp.pt');torch.save(dict(next=i+1,gradient=b.grad.detach().cpu() if b.grad is not None else None,
          rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),tmp);tmp.replace(partial)
        if time.monotonic()-started>a.minutes*60 and i+1<d.height:return
    gradient_norm=float(b.grad.norm()) if b.grad is not None else 0.
    if b.grad is not None:
        assert torch.isfinite(b.grad).all(),'Nonfinite gradient';opt.step()
    np.save(a.out/f'b_s{step:03d}.npy',b.detach().cpu().numpy())
    tmp=state.with_suffix('.tmp.pt');torch.save(dict(step=step,bias=b.detach().cpu(),optimizer=opt.state_dict()),tmp);tmp.replace(state)
    atomic_json(dict(step=step,reward=d['success'].mean(),gradient_norm=gradient_norm,
         vector_norm=float(b.norm()),rollouts_sha256=sha(a.out/f'rollouts_s{step:03d}.parquet'),
         rewards_sha256=sha(a.out/f'rewards_s{step:03d}.parquet')),a.out/f'metrics_s{step:03d}.json')
    atomic_json(dict(step=step),a.out/'progress.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--init',choices=['zero','ol_math_A2'],default='zero')
    p.add_argument('--steps',type=int,default=60);p.add_argument('--prompts',type=int,default=16);p.add_argument('--group',type=int,default=8)
    p.add_argument('--budget',type=int,default=8192);p.add_argument('--lr',type=float,default=.002);p.add_argument('--minutes',type=float,default=10)
    p.add_argument('--status-only',action='store_true');a=p.parse_args();a.root=a.root.resolve();a.out=a.out.resolve()
    sys.path.insert(0,str(a.root));os.chdir(a.root);started=time.monotonic()
    tasks,cfg=setup(a);step,stage=next_stage(a.out,a.steps)
    print(json.dumps(dict(step=step,stage=stage)),flush=True)
    if not a.status_only and stage!='complete':globals()[stage](a,tasks,cfg,step,started) if stage=='generate' else globals()[stage](a,cfg,step,started)
    step,stage=next_stage(a.out,a.steps);atomic_json(dict(step=step,stage=stage),a.out/'status.json')
