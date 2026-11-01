"""Fixed-step corrected-RL evaluation, with fresh common-random-number controls.

Two sampled completions per held-out problem at the training objective's 8192
token budget. The historical 199 items are reused; this is not a fresh holdout.
Generation and extraction are separate resumable GPU invocations.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,os,sys,time
import numpy as np
import polars as pl
from rl_repair import sha,atomic_json,atomic_parquet,generate,extract

def run(a):
    a.root=a.root.resolve();a.revision=a.revision.resolve()
    sys.path.insert(0,str(a.root));os.chdir(a.root)
    from mode_control.mc.common import DECODE,DATA
    from mode_control.mc.vsteer import load_vector
    out=a.revision/'results/rl_evaluation'/a.run;out.mkdir(parents=True,exist_ok=True)
    task_path=DATA/'runs/p1_budget/tasks.parquet';tasks=pl.read_parquet(task_path)
    assert tasks.height==199 and tasks['instance_id'].n_unique()==199
    if a.run in ['baseline','a2']:
        b=load_vector('ol_math_A2')['v'].astype(np.float32)
        if a.run=='baseline':b=np.zeros_like(b)
        vector_source='zero' if a.run=='baseline' else str(DATA/'vsteer/vec_ol_math_A2.npz')
    else:
        training=a.revision/'results/rl'/a.run
        assert json.loads((training/'status.json').read_text())['stage']=='complete'
        assert json.loads((training/'config.json').read_text())['steps']==60
        vector_source=str(training/'b_s060.npy');b=np.load(vector_source)
    if (out/'b_s000.npy').exists():assert np.array_equal(np.load(out/'b_s000.npy'),b)
    else:np.save(out/'b_s000.npy',b)
    if (out/'tasks.parquet').exists():assert tasks.equals(pl.read_parquet(out/'tasks.parquet'))
    else:atomic_parquet(tasks,out/'tasks.parquet')
    cfg=dict(run=a.run,decode=dict(DECODE),temperature=float(DECODE['temperature']),layer=16,
      budget=8192,sampling_seed=20261102,samples_per_problem=2,n_tasks=199,checkpoint=60,
      task_source_sha256=sha(task_path),vector_sha256=sha(out/'b_s000.npy'),vector_source=vector_source,
      code_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('rl_repair.py'),a.root/'mode_control/mc/extract_free.py']},
      interpretation='Fixed checkpoint and budget, reused historical evaluation problems; compare each training seed with fresh baseline and A2 under common sampling seeds.')
    cp=out/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==cfg
    else:atomic_json(cfg,cp)
    args=SimpleNamespace(out=out,budget=8192,seed=20261102,prompts=199,group=2,minutes=a.minutes)
    started=time.monotonic()
    if not (out/'rollouts_s001.done').exists():generate(args,tasks,cfg,1,started)
    elif not (out/'rewards_s001.done').exists():extract(args,cfg,1,started)
    complete=(out/'rewards_s001.done').exists()
    atomic_json(dict(stage='complete' if complete else 'pending',run=a.run),out/'status.json')
    if complete:
        tr=pl.read_parquet(out/'rollouts_s001.parquet');gr=pl.read_parquet(out/'rewards_s001.parquet')
        d=tr.join(gr.select('trace_id','success','grade_source'),on='trace_id',validate='1:1')
        assert d.height==398
        d.select('trace_id','instance_id','sample','completed','success','grade_source').write_csv(out/'outcomes.csv')
        atomic_json(dict(n_tasks=199,attempts=398,accuracy=float(d['success'].mean()),
          completion=float(d['completed'].mean()),fixed_step=60,budget=8192),out/'summary.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--revision',type=Path,required=True)
    p.add_argument('--run',required=True);p.add_argument('--minutes',type=float,default=10)
    run(p.parse_args())
