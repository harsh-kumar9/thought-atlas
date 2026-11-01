"""Frozen fresh-item confirmation of A2 and the high-fidelity moral contrast.

No training or vector selection. Every stage checkpoints in a separate directory.
The sample excludes saved project tasks and all MATH-500 items before any outputs
are generated. This is project-level freshness, not pretraining decontamination.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,os,re,sys,time,unicodedata
import numpy as np
import pandas as pd
import polars as pl
from rl_repair import sha,atomic_json,atomic_parquet,generate,extract
from elicitation import prefix_plan

DATASET='DigitalLearningGmbH/MATH-lighteval'
REVISION='0530c78699ea5e8eb5530600900e1f328b48acad'
ARMS=['baseline','ol_math_A2','sig_moral']
N=256
BUDGET=8192
SEED=20261103

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def canonical(text):
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',str(text)).casefold())

def boxed(solution):
    matches=list(re.finditer(r'\\(?:boxed|fbox)\s*\{',solution))
    if not matches:return None
    start=matches[-1].end();depth=1
    for i in range(start,len(solution)):
        depth+=(solution[i]=='{')-(solution[i]=='}')
        if depth==0:return solution[start:i].strip()
    return None

def prepare(root,out):
    from huggingface_hub import hf_hub_download
    from difflib import SequenceMatcher
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise RuntimeError('Confirmation already frozen; never resample after outputs')
    source=Path(hf_hub_download(DATASET,'data/test-00000-of-00001.parquet',repo_type='dataset',revision=REVISION))
    source_df=pl.read_parquet(source)
    assert source_df.height==5000
    # Full inventory of saved task tables, plus observed math prompts in traces.
    files=set((root/'data').rglob('*tasks*.parquet'))|set((root/'data').glob('*/tasks/math.parquet'))|set((root/'data/tasks').glob('math.parquet'))|set((root/'mode_control/data').rglob('*tasks*.parquet'))
    files|=set((root/'data/v2/traces').glob('traces_*.parquet'))
    files|=set((root/'mode_control/data/f1/traces').glob('traces_*.parquet'))
    seen=set();inventory=[]
    for f in sorted(files):
        resolved=f
        with f.open('rb') as handle:magic=handle.read(4)
        if magic!=b'PAR1':
            pointer=f.read_text();oid=re.search(r'oid sha256:([0-9a-f]{64})',pointer)
            assert oid, f'Unrecognized task file: {f}'
            resolved=out/'exclusions_cache'/f'{oid[1]}.parquet'
            assert resolved.exists() and sha(resolved)==oid[1],f'Unresolved exclusion input: {f}'
        schema=pl.read_parquet_schema(resolved)
        if 'prompt' not in schema:continue
        columns=[c for c in ['prompt','task_type'] if c in schema]
        d=pl.read_parquet(resolved,columns=columns)
        if 'task_type' in d.columns:d=d.filter(pl.col('task_type')=='math')
        values={canonical(x) for x in d['prompt'].drop_nulls().to_list() if x}
        seen.update(values)
        inventory.append(dict(file=str(f.relative_to(root)),sha256=sha(resolved),lfs_resolved=resolved!=f,math_prompt_count=len(values)))
    oldmath=pl.read_parquet(root/'data/v2/tasks/math.parquet')
    assert oldmath.height==500
    assert all(canonical(x) in seen for x in oldmath['prompt'])
    old500={canonical(x) for x in oldmath['prompt']}
    assert old500.issubset({canonical(x) for x in source_df['problem']})
    eligible=[];excluded={'previous_or_contained':0,'near_duplicate':0,'no_boxed_reference':0,'duplicate_source':0}
    unique=set()
    for index,r in enumerate(source_df.iter_rows(named=True)):
        key=canonical(r['problem'])
        if key in unique:excluded['duplicate_source']+=1;continue
        unique.add(key)
        if any(key==s or (len(s)>40 and (key in s or s in key)) for s in seen):
            excluded['previous_or_contained']+=1;continue
        reference=boxed(r['solution'])
        if reference is None:excluded['no_boxed_reference']+=1;continue
        eligible.append(dict(instance_id='math-confirm:'+hashlib.sha256(key.encode()).hexdigest()[:24],
            task_type='math',prompt=r['problem'],reference_answer=reference,
            difficulty_raw=int(str(r['level']).split()[-1]),subject=r['type'],source_row=index,
            normalized_prompt_sha256=hashlib.sha256(key.encode()).hexdigest()))
    # Deterministic order without outcome/difficulty selection. Conservative near-duplicate
    # check uses problem text only; numbers remain intact, and the decision is logged.
    ordered=sorted(eligible,key=lambda r:hashlib.sha256((str(SEED)+'|'+r['instance_id']).encode()).hexdigest())
    selected=[]
    for r in ordered:
        key=canonical(r['prompt']);similar=False
        for s in seen:
            if min(len(s),len(key))/max(len(s),len(key))<.90:continue
            matcher=SequenceMatcher(None,key,s,autojunk=False)
            if matcher.quick_ratio()>=.95 and matcher.ratio()>=.95:similar=True;break
        if similar:excluded['near_duplicate']+=1;continue
        selected.append(r)
        if len(selected)==N:break
    assert len(selected)==N
    tasks=pl.DataFrame(selected);atomic_parquet(tasks,out/'tasks.parquet')
    vectors={}
    for arm in ARMS:
        path=root/'mode_control/data/vsteer'/f'vec_{"ol_math_A2" if arm=="baseline" else arm}.npz'
        z=np.load(path,allow_pickle=True);assert int(z['layer'])==16
        v=z['v'].astype(np.float32)
        if arm=='baseline':v=np.zeros_like(v)
        np.save(out/f'vector_{arm}.npy',v)
        vectors[arm]=dict(source=str(path),source_sha256=sha(path),vector_sha256=sha(out/f'vector_{arm}.npy'),norm=float(np.linalg.norm(v)))
    protocol=dict(frozen_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        dataset=DATASET,dataset_revision=REVISION,dataset_file='data/test-00000-of-00001.parquet',source_sha256=sha(source),
        source_rows=source_df.height,excluded=excluded,previous_unique_math_texts=len(seen),inventory=inventory,
        sample_seed=SEED,n_tasks=N,tasks_sha256=sha(out/'tasks.parquet'),arms=ARMS,vectors=vectors,
        selection='First256 in SHA256(seed|instance_id) order after exclusions. No generated outcomes inspected.',
        comparison_rationale='A2 tests the historically selected archetype; moral contrast selected for high retrospective math fidelity, not as a prospective optimum.',
        budget=BUDGET,samples=2,layer=16,decode=dict(temperature=.6,top_p=.95,top_k=20),
        primary='Natural-completion accuracy at8192; all allocated attempts; paired problem mean over2 samples; two-sided sign-flip with Holm over the two intervention-baseline contrasts.',
        secondary='Common-total-budget answer elicitation at8192, answer availability, tokens, literal repetition; paired95% intervals descriptive.',
        behavior='First64 selected tasks, sample0; same8192 reasoning window; Gemma4-31B-it contextual labels, paired achieved16-target shifts; descriptive fidelity with uncertainty.',
        interpretation='Fresh relative to inventoried project data and MATH-500; not a claim of model-pretraining exclusion. Fixed vectors and dose, not a validated intervention-selection algorithm.',
        code_sha256=sha(Path(__file__)))
    atomic_json(protocol,out/'protocol.json');tasks.drop('prompt','reference_answer').write_csv(out/'selected_items.csv')
    print(json.dumps({k:protocol[k] for k in ['n_tasks','excluded','previous_unique_math_texts','tasks_sha256','vectors']},indent=2))

def run(root,out,arm,minutes):
    plan=json.loads((out/'analysis_plan.json').read_text())
    for relative,expected in plan['canonical_code_sha256'].items():
        assert sha(root/relative)==expected,f'Frozen confirmation dependency changed: {relative}'
    cfg0=json.loads((out/'protocol.json').read_text());assert sha(out/'tasks.parquet')==cfg0['tasks_sha256']
    tasks=pl.read_parquet(out/'tasks.parquet');work=out/arm;work.mkdir(parents=True,exist_ok=True)
    vector=out/f'vector_{arm}.npy';assert sha(vector)==cfg0['vectors'][arm]['vector_sha256']
    if not (work/'b_s000.npy').exists():np.save(work/'b_s000.npy',np.load(vector))
    else:assert np.array_equal(np.load(work/'b_s000.npy'),np.load(vector))
    cfg=dict(decode=cfg0['decode'],temperature=cfg0['decode']['temperature'],layer=16,
        frozen_protocol_sha256=sha(out/'protocol.json'),analysis_plan_sha256=sha(out/'analysis_plan.json'),code_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('rl_repair.py'),root/'mode_control/mc/extract_free.py']})
    if (work/'config.json').exists():assert json.loads((work/'config.json').read_text())==cfg
    else:atomic_json(cfg,work/'config.json')
    a=SimpleNamespace(out=work,budget=BUDGET,seed=SEED,prompts=N,group=2,minutes=minutes)
    started=time.monotonic()
    if not (work/'rollouts_s001.done').exists():generate(a,tasks,cfg,1,started)
    elif not (work/'rewards_s001.done').exists():extract(a,cfg,1,started)
    elif not (work/'elicitation.done').exists():elicit(work,minutes)
    complete=(work/'elicitation.done').exists()
    atomic_json(dict(stage='complete' if complete else 'pending',arm=arm),work/'status.json')

def elicit(work,minutes):
    from mode_control.mc.common import load_tokenizer,render_prompt
    from mode_control.mc.readiness import FORCE,grade_forced,build_engine
    from vllm import SamplingParams
    tok=load_tokenizer();cue=tok.encode(FORCE['math'],add_special_tokens=False);end=tok.convert_tokens_to_ids('</think>')
    d=pl.read_parquet(work/'rollouts_s001.parquet').join(pl.read_parquet(work/'rewards_s001.parquet').select('trace_id','success'),on='trace_id',validate='1:1')
    dest=work/'elicitation.parquet';old=pl.read_parquet(dest) if dest.exists() else None
    done=set(old['trace_id']) if old is not None else set();jobs=[];own=[];started=time.monotonic()
    def save(rows):
        nonlocal old
        if rows:
            df=pl.DataFrame(rows,infer_schema_length=None);old=pl.concat([old,df],how='diagonal_relaxed') if old is not None else df
            atomic_parquet(old,dest)
    for r in d.iter_rows(named=True):
        if r['trace_id'] in done:continue
        ids=list(r['gen_token_ids']);plan=prefix_plan(ids,end,BUDGET,len(cue),completed=r['completed'])
        meta={k:r[k] for k in ['trace_id','instance_id','sample']}
        if plan['own']:
            own.append(dict(**meta,correct=r['success'],answer=r['answer_text'],raw_answer='',own=True,
                prefix_tokens=len(ids),cue_tokens=0,answer_tokens=0,total_tokens=len(ids),grade_source='natural_extraction'))
        else:
            jobs.append(dict(row=r,meta=meta,prefix=plan['prefix_len'],ids=tok.encode(render_prompt(tok,r['prompt']),add_special_tokens=False)+ids[:plan['prefix_len']]+cue))
    save(own)
    if jobs:
        llm=build_engine(max_model_len=BUDGET+2048)
        for off in range(0,len(jobs),16):
            batch=jobs[off:off+16];outputs=llm.generate([{'prompt_token_ids':j['ids']} for j in batch],SamplingParams(temperature=0,max_tokens=48,seed=SEED),use_tqdm=False)
            rows=[]
            for j,o in zip(batch,outputs):
                g=o.outputs[0];answer,correct=grade_forced('math',g.text,j['row']['reference_answer'],j['row']['prompt'])
                total=j['prefix']+len(cue)+len(g.token_ids);assert total<=BUDGET
                rows.append(dict(**j['meta'],correct=correct,answer=answer,raw_answer=g.text,own=False,
                    prefix_tokens=j['prefix'],cue_tokens=len(cue),answer_tokens=len(g.token_ids),total_tokens=total,grade_source='forced_direct'))
            save(rows)
            if off==0 and np.mean([bool(r['answer']) for r in rows])<.6:raise RuntimeError('Confirmation answer-format gate failed; raw answers retained')
            if time.monotonic()-started>minutes*60:break
    if old is not None and old.height==N*2:(work/'elicitation.done').touch()

def prepare_labels(root,out):
    from mode_control.mc.common import load_tokenizer
    tok=load_tokenizer();chosen=set(pl.read_parquet(out/'tasks.parquet').head(64)['instance_id'])
    rows=[];keys=[]
    for arm in ARMS:
        assert json.loads((out/arm/'status.json').read_text())['stage']=='complete'
        d=pl.read_parquet(out/arm/'rollouts_s001.parquet').filter(pl.col('instance_id').is_in(list(chosen))&(pl.col('sample')==0))
        assert d.height==64
        for r in d.iter_rows(named=True):
            text=tok.decode(tok.encode(r['think_text'],add_special_tokens=False)[:BUDGET],skip_special_tokens=False)
            rows.append(dict(trace_id=r['trace_id'],prompt=r['prompt'],reasoning_text_for_analysis=text,think_text=text,analysis_source='think_text'))
            keys.append(dict(trace_id=r['trace_id'],instance_id=r['instance_id'],condition=arm,source=arm,gen_model='qwen35_9b',task_type='math'))
    dest=out/'behavior';dest.mkdir(exist_ok=True)
    atomic_parquet(pl.DataFrame(rows).sort('trace_id'),dest/'blinded_traces.parquet')
    atomic_parquet(pl.DataFrame(keys).sort('trace_id'),dest/'sample_key.parquet')
    for name in ['kim','gandhi']:(dest/f'{name}_per_sentence.txt').write_bytes((root/f'src/judge/prompts/{name}_per_sentence.txt').read_bytes())
    atomic_json(dict(traces=len(rows),tasks=64,sample=0,window=BUDGET,input_sha256=sha(dest/'blinded_traces.parquet'),selection='First64 frozen task IDs; no outcome-dependent selection'),dest/'manifest.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','prepare_labels']);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,default='baseline');p.add_argument('--minutes',type=float,default=10)
    a=p.parse_args();a.root=a.root.resolve();a.out=a.out.resolve();sys.path.insert(0,str(a.root));os.chdir(a.root)
    if a.stage=='prepare':prepare(a.root,a.out)
    elif a.stage=='prepare_labels':prepare_labels(a.root,a.out)
    else:run(a.root,a.out,a.arm,a.minutes)
